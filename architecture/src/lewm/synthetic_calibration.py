"""Synthetic-shift utilities for Gaussian Correction Depth (GCD) ablations.

For Phase C of the GCD plan in ``docs/gaussian_correction_depth.md``, we
calibrate ``G_ψ`` against a *known* invertible shift in latent space —
rotation, translation, or affine — applied to source-domain latents from a
frozen LeWM chassis. The controlled ground truth lets us measure exactly
how fast each ``G_ψ`` configuration learns the shift, how the four-term
loss tradeoffs play out (proximal vs calibration accuracy, retention vs
calibration accuracy), and how MLP vs causal-transformer correction blocks
compare on convergence speed.

This module provides:

- :class:`LatentShift` — an invertible affine transform in latent space,
  parameterized by an optional orthogonal rotation ``R`` and an optional
  translation ``t``.
- :func:`random_rotation_matrix` — uniform random proper rotation (det = +1)
  in ``D`` dimensions.
- :func:`random_translation` — Gaussian-sampled translation vector.
- :func:`make_shift` — named factory: ``"identity"``, ``"rotation"``,
  ``"translation"``, ``"affine"``.
- :func:`latent_diagnostics` — quick distributional summary used in the
  per-epoch logging of the synthetic experiments.
- :func:`calibration_error` — direct MSE between ``gpsi(emb)`` and the
  shift target on a held-out batch (the cleanest "did G_ψ learn the
  shift?" probe).

All transforms work for any leading-dim shape: ``(D,)``, ``(B, D)``,
``(B, T, D)``, ``(M, T, D)``, etc., as long as the last dim equals the
shift's ``dim``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LatentShift:
    """An invertible affine transformation in a fixed-dimension latent space.

    Forward: ``apply(x) = (x @ R^T) + t``
    Inverse: ``inverse(x) = (x - t) @ R``

    where ``R`` is a ``(dim, dim)`` orthogonal matrix and ``t`` is a
    ``(dim,)`` translation. Either field may be ``None`` to mean identity in
    that component (e.g., a pure rotation has ``translation=None``).

    The transform is shape-agnostic over leading dims: ``(..., dim)`` input,
    same shape output.
    """

    name: str
    rotation: torch.Tensor | None = None
    translation: torch.Tensor | None = None

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if self.rotation is not None:
            if x.shape[-1] != self.rotation.shape[0]:
                raise ValueError(
                    f"x last dim {x.shape[-1]} != rotation dim "
                    f"{self.rotation.shape[0]}"
                )
            out = out @ self.rotation.t()
        if self.translation is not None:
            if x.shape[-1] != self.translation.shape[0]:
                raise ValueError(
                    f"x last dim {x.shape[-1]} != translation dim "
                    f"{self.translation.shape[0]}"
                )
            out = out + self.translation
        return out

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if self.translation is not None:
            out = out - self.translation
        if self.rotation is not None:
            out = out @ self.rotation
        return out

    def to(self, device: str | torch.device) -> "LatentShift":
        return LatentShift(
            name=self.name,
            rotation=None if self.rotation is None else self.rotation.to(device),
            translation=None if self.translation is None else self.translation.to(device),
        )

    @property
    def is_identity(self) -> bool:
        return self.rotation is None and self.translation is None

    def magnitude(self, sample_norm: float = 1.0) -> float:
        """Approximate per-vector magnitude of ``apply(x) - x`` for a unit-norm
        sample. Useful as a "shift size" annotation in ablation tables.
        """

        if self.is_identity:
            return 0.0
        if self.rotation is None and self.translation is not None:
            return float(self.translation.norm())
        if self.translation is None and self.rotation is not None:
            d = self.rotation.shape[0]
            i = torch.eye(d, device=self.rotation.device, dtype=self.rotation.dtype)
            # ||(R - I) x|| over a unit-norm Gaussian sample, expected
            # ≈ sqrt(||R - I||_F² / d) * sample_norm.
            fro = (self.rotation - i).pow(2).sum().sqrt()
            return float(fro / (d ** 0.5) * sample_norm)
        # affine
        d = self.rotation.shape[0]
        i = torch.eye(d, device=self.rotation.device, dtype=self.rotation.dtype)
        fro = (self.rotation - i).pow(2).sum().sqrt()
        return float(
            ((fro / (d ** 0.5) * sample_norm) ** 2 + float(self.translation.norm()) ** 2) ** 0.5
        )


def random_rotation_matrix(
    dim: int, *, seed: int | None = None,
) -> torch.Tensor:
    """Uniform random proper rotation (``det = +1``) in ``dim`` dimensions.

    Construction: QR of a Gaussian matrix, sign-fixed by the diagonal of
    ``R`` so ``Q`` has unit-norm columns with positive diagonal, then a
    final column-flip if needed to enforce ``det(Q) = +1``.
    """

    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(int(seed))
    a = torch.randn(dim, dim, generator=g)
    q, r = torch.linalg.qr(a)
    # Sign-fix so Q has positive diagonal pattern (canonical QR).
    sign = torch.sign(torch.diagonal(r))
    sign[sign == 0] = 1
    q = q * sign.unsqueeze(0)
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def random_translation(
    dim: int, *, scale: float = 1.0, seed: int | None = None,
) -> torch.Tensor:
    """Gaussian-sampled translation with overall norm scaled by ``scale``.

    The expected ``||t||`` is approximately ``scale * sqrt(dim)``.
    """

    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if scale < 0:
        raise ValueError(f"scale must be >= 0, got {scale}")
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(int(seed))
    return torch.randn(dim, generator=g) * scale


SUPPORTED_SHIFT_NAMES = (
    "identity",
    "rotation",
    "translation",
    "affine",
    "progressive_rotation",
    "random_rotations_per_t",
)


def make_shift(
    name: str,
    dim: int,
    *,
    scale: float = 1.0,
    seed: int = 0,
    sequence_len: int | None = None,
):
    """Named factory for a :class:`LatentShift` or :class:`TimeVaryingShift`.

    Static (returns :class:`LatentShift`):

    - ``"identity"`` — no rotation, no translation.
    - ``"rotation"`` — random proper rotation, ``det = +1``, norm-preserving.
    - ``"translation"`` — pure translation with expected norm ``scale * sqrt(dim)``.
    - ``"affine"`` — rotation + translation (separately seeded).

    Time-varying (returns :class:`TimeVaryingShift`; requires ``sequence_len``):

    - ``"progressive_rotation"`` — per-position transform ``shift_t(x) = R^t · x``
      where ``R`` is one random rotation. At ``t=0`` the shift is identity; at
      ``t=T-1`` it has rotated ``(T-1)·angle_of(R)`` cumulatively. Phase C₂
      ablation: tests whether the correction stack can use temporal context
      (transformer attention) to recover the per-position transform — an MLP
      block that's pointwise per-position cannot.
    - ``"random_rotations_per_t"`` — independent random rotation per ``t``
      (no recoverable structure, each position needs its own transform).
    """

    if name == "identity":
        return LatentShift(name="identity")
    if name == "rotation":
        return LatentShift(
            name="rotation",
            rotation=random_rotation_matrix(dim, seed=seed),
        )
    if name == "translation":
        return LatentShift(
            name="translation",
            translation=random_translation(dim, scale=scale, seed=seed),
        )
    if name == "affine":
        return LatentShift(
            name="affine",
            rotation=random_rotation_matrix(dim, seed=seed),
            translation=random_translation(dim, scale=scale, seed=seed + 1),
        )
    if name == "progressive_rotation":
        if sequence_len is None:
            raise ValueError(
                "progressive_rotation requires sequence_len"
            )
        if sequence_len < 1:
            raise ValueError(f"sequence_len must be >= 1, got {sequence_len}")
        base_R = random_rotation_matrix(dim, seed=seed)
        rotations = torch.empty(sequence_len, dim, dim)
        rotations[0] = torch.eye(dim)
        for t in range(1, sequence_len):
            rotations[t] = rotations[t - 1] @ base_R
        return TimeVaryingShift(name="progressive_rotation", rotations=rotations)
    if name == "random_rotations_per_t":
        if sequence_len is None:
            raise ValueError(
                "random_rotations_per_t requires sequence_len"
            )
        if sequence_len < 1:
            raise ValueError(f"sequence_len must be >= 1, got {sequence_len}")
        rotations = torch.stack([
            random_rotation_matrix(dim, seed=seed + t)
            for t in range(sequence_len)
        ])
        return TimeVaryingShift(
            name="random_rotations_per_t", rotations=rotations,
        )
    raise ValueError(
        f"unknown shift name {name!r}; supported: {SUPPORTED_SHIFT_NAMES}"
    )


@dataclass
class TimeVaryingShift:
    """Per-time-position invertible linear transform in latent space.

    Operates on tensors with shape ``(..., T, D)``. At position ``t``, applies
    ``rotations[t]`` (and optionally ``translations[t]``). The ``T`` dimension
    of the input must match ``rotations.shape[0]``.

    Phase C₂ usage: ``progressive_rotation`` and ``random_rotations_per_t``
    test whether the GPsi correction stack can leverage temporal context
    (transformer attention) when the calibration target depends on position.
    A pointwise MLP block has no per-position information; a causal
    transformer has it via self-attention.

    Same interface (``apply``, ``inverse``, ``to``, ``magnitude``, ``name``,
    ``is_identity``) as :class:`LatentShift` so the ablation driver can
    treat them polymorphically.
    """

    name: str
    rotations: torch.Tensor | None = None       # (T, D, D)
    translations: torch.Tensor | None = None    # (T, D)

    def __post_init__(self) -> None:
        if self.rotations is None and self.translations is None:
            raise ValueError(
                "TimeVaryingShift needs at least one of rotations/translations"
            )
        if self.rotations is not None:
            if self.rotations.dim() != 3:
                raise ValueError(
                    f"rotations must be [T, D, D], got "
                    f"{tuple(self.rotations.shape)}"
                )
            t, d, d2 = self.rotations.shape
            if d != d2:
                raise ValueError(
                    f"rotations dims [T, D, D] must square: got [{t}, {d}, {d2}]"
                )
        if self.translations is not None:
            if self.translations.dim() != 2:
                raise ValueError(
                    f"translations must be [T, D], got "
                    f"{tuple(self.translations.shape)}"
                )
            if self.rotations is not None:
                if self.translations.shape != (
                    self.rotations.shape[0], self.rotations.shape[1],
                ):
                    raise ValueError(
                        f"rotations and translations T/D mismatch: "
                        f"rotations={tuple(self.rotations.shape)} "
                        f"translations={tuple(self.translations.shape)}"
                    )

    @property
    def sequence_len(self) -> int:
        if self.rotations is not None:
            return int(self.rotations.shape[0])
        assert self.translations is not None
        return int(self.translations.shape[0])

    @property
    def dim(self) -> int:
        if self.rotations is not None:
            return int(self.rotations.shape[-1])
        assert self.translations is not None
        return int(self.translations.shape[-1])

    @property
    def is_identity(self) -> bool:
        return False  # by construction, time-varying shifts are non-trivial

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., T, D)`` input → same shape output, with per-``t`` transform."""

        if x.dim() < 2:
            raise ValueError(
                f"x must have at least (T, D) trailing shape; got "
                f"{tuple(x.shape)}"
            )
        if x.shape[-2] != self.sequence_len:
            raise ValueError(
                f"x sequence length {x.shape[-2]} does not match shift "
                f"sequence_len {self.sequence_len}"
            )
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"x last dim {x.shape[-1]} does not match shift dim {self.dim}"
            )
        out = x
        if self.rotations is not None:
            # Apply per-step rotation: out[..., t, :] = x[..., t, :] @ R[t].T
            # einsum: 'tij,...tj->...ti' with rotations transposed
            out = torch.einsum("tij,...tj->...ti", self.rotations, out)
        if self.translations is not None:
            out = out + self.translations
        return out

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        if self.translations is not None:
            out = out - self.translations
        if self.rotations is not None:
            # Inverse rotation per t: R[t]^T = R[t]^-1 (orthogonal).
            inv = self.rotations.transpose(-1, -2)
            out = torch.einsum("tij,...tj->...ti", inv, out)
        return out

    def to(self, device) -> "TimeVaryingShift":
        return TimeVaryingShift(
            name=self.name,
            rotations=None if self.rotations is None else self.rotations.to(device),
            translations=None if self.translations is None else self.translations.to(device),
        )

    def magnitude(self, sample_norm: float = 1.0) -> float:
        """Approximate per-vector ``||apply(x) - x||`` averaged over t for a
        unit-norm-Gaussian sample."""

        d = self.dim
        t_count = self.sequence_len
        accum = 0.0
        if self.rotations is not None:
            i = torch.eye(d, device=self.rotations.device, dtype=self.rotations.dtype)
            for t in range(t_count):
                fro = (self.rotations[t] - i).pow(2).sum().sqrt()
                accum += float(fro / (d ** 0.5) * sample_norm) ** 2
        if self.translations is not None:
            for t in range(t_count):
                accum += float(self.translations[t].norm()) ** 2
        return (accum / t_count) ** 0.5


def latent_diagnostics(latents: torch.Tensor) -> dict[str, float | int]:
    """Distributional summary of a ``(..., D)`` latent tensor.

    Used in per-epoch ablation logs alongside training-loss values.
    """

    flat = latents.reshape(-1, latents.shape[-1])
    if flat.shape[0] == 0:
        raise ValueError("empty latent tensor")
    centered = flat - flat.mean(dim=0)
    return {
        "n_samples": int(flat.shape[0]),
        "dim": int(flat.shape[1]),
        "mean_norm": float(flat.norm(dim=-1).mean()),
        "std_norm": float(flat.norm(dim=-1).std()),
        "mean_abs": float(flat.abs().mean()),
        "covariance_trace": float(centered.pow(2).sum(dim=-1).mean()),
    }


def sinusoidal_positional_encoding(
    sequence_len: int,
    dim: int,
    *,
    base: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Standard sinusoidal positional encoding ``[T, D]`` from "Attention Is
    All You Need" (Vaswani et al., 2017).

    For position ``t`` and embedding dimension ``2i`` (even) /``2i+1`` (odd):

    .. math::

        PE(t, 2i)   = \\sin(t / base^{2i / D})
        PE(t, 2i+1) = \\cos(t / base^{2i / D})

    Used in Phase C₂.5 to feed explicit per-position info to the GPsi cond
    channel. Each position gets a unique encoding, so AdaLN modulation in
    the Transformer block (or concat-input in the MLP block) can use the
    positional signal to apply per-``t`` transforms.

    The encoding is constant across the batch dimension — caller broadcasts
    it manually if cond is needed at shape ``[B, T, D]``.
    """

    if sequence_len < 1:
        raise ValueError(f"sequence_len must be >= 1, got {sequence_len}")
    if dim < 2:
        raise ValueError(
            f"dim must be >= 2 (need at least one sin/cos pair), got {dim}"
        )
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")

    pe = torch.zeros(sequence_len, dim, dtype=dtype, device=device)
    positions = torch.arange(sequence_len, dtype=dtype, device=device).unsqueeze(1)
    # div_term[j] = base ** (2j / dim) for j = 0, 1, ..., (dim // 2) - 1
    half = dim // 2
    div_term = torch.exp(
        torch.arange(half, dtype=dtype, device=device)
        * (-torch.log(torch.tensor(base, dtype=dtype, device=device)) * 2.0 / dim)
    )
    angles = positions * div_term  # [T, half]
    pe[:, 0::2] = torch.sin(angles)
    if dim % 2 == 0:
        pe[:, 1::2] = torch.cos(angles)
    else:
        # Odd dim: last column is unmatched; fill with cos as a half-pair.
        pe[:, 1::2] = torch.cos(angles[:, :half])
    return pe


def calibration_error(g: torch.Tensor, target: torch.Tensor) -> float:
    """Direct MSE between ``G_ψ(emb)`` and the calibration target.

    On a held-out batch, this is the cleanest "did ``G_ψ`` learn the shift?"
    probe — independent of all other loss terms. Lower is better. At init,
    ``g == emb`` so this equals ``MSE(emb, target) == ||shift||²``.
    """

    if g.shape != target.shape:
        raise ValueError(
            f"shape mismatch: g={tuple(g.shape)} target={tuple(target.shape)}"
        )
    return float(F.mse_loss(g, target).detach())
