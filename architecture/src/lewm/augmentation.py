"""Symmetry augmentation helpers for palette-indexed LeWM frames.

The helpers operate on tensors whose last two dimensions are spatial
``(..., H, W)`` and whose values are integer palette ids.  They intentionally do
not infer whether a game mechanic is symmetry-safe; callers must choose the
augmentation strength from a recorded condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch


@dataclass(frozen=True)
class SymmetryAug:
    """D4 + palette-index augmentation for ARC-style frame tensors."""

    color_perm_prob: float = 0.8
    d4_prob: float = 0.5
    num_colors: int = 16
    background_idx: int | None = 0

    D4_TRANSFORMS: ClassVar[tuple[str, ...]] = (
        "identity",
        "r90",
        "r180",
        "r270",
        "flip_h",
        "flip_v",
        "flip_diag",
        "flip_antidiag",
    )

    ARC_CLUSTER_PROBS: ClassVar[dict[int, tuple[float, float]]] = {
        1: (0.6, 0.6),  # click-only puzzles
        2: (0.8, 0.5),  # D-pad navigation
        3: (0.6, 0.3),  # click + sprite move
        4: (0.5, 0.8),  # click + place + move, m0r0-style mirroring
        5: (0.6, 0.5),  # click + agent move
    }
    ARC_GAME_CLUSTER: ClassVar[dict[str, int]] = {
        "re86": 1,
        "m0r0": 4,
        "sc25": 5,
    }

    def __post_init__(self) -> None:
        if not 0.0 <= self.color_perm_prob <= 1.0:
            raise ValueError("color_perm_prob must be in [0, 1]")
        if not 0.0 <= self.d4_prob <= 1.0:
            raise ValueError("d4_prob must be in [0, 1]")
        if self.num_colors < 1:
            raise ValueError("num_colors must be >= 1")
        if self.background_idx is not None and not 0 <= self.background_idx < self.num_colors:
            raise ValueError("background_idx must be in [0, num_colors)")

    @classmethod
    def for_arc_game(cls, game_id: str, *, background_idx: int | None = 0) -> SymmetryAug:
        """Return the recorded default strength for an ARC game or cluster id.

        ``re86`` gets the explicit low color-permutation override because its
        same-color matching mechanic can depend on color identity.
        """
        short_id = game_id.split("-", 1)[0]
        if short_id == "re86":
            return cls(color_perm_prob=0.1, d4_prob=0.6, background_idx=background_idx)
        cluster = cls.ARC_GAME_CLUSTER.get(short_id)
        if cluster is None:
            return cls(background_idx=background_idx)
        color_prob, d4_prob = cls.ARC_CLUSTER_PROBS[cluster]
        return cls(color_perm_prob=color_prob, d4_prob=d4_prob, background_idx=background_idx)

    @staticmethod
    def color_permutation(frame: torch.Tensor, permutation_map: torch.Tensor) -> torch.Tensor:
        """Apply ``permutation_map[pixel_id]`` to a palette-indexed frame."""
        if permutation_map.dim() != 1:
            raise ValueError("permutation_map must be 1-D")
        if frame.numel() == 0:
            return frame.clone()
        idx = frame.long()
        if int(idx.min().item()) < 0 or int(idx.max().item()) >= permutation_map.numel():
            raise ValueError("frame contains palette ids outside permutation_map")
        mapped = permutation_map.to(device=frame.device)[idx]
        return mapped.to(dtype=frame.dtype)

    @staticmethod
    def rotation_90k(frame: torch.Tensor, k: int) -> torch.Tensor:
        """Rotate spatial dimensions by ``90 * k`` degrees."""
        return torch.rot90(frame, k=int(k) % 4, dims=(-2, -1))

    @staticmethod
    def reflect(frame: torch.Tensor, axis: str) -> torch.Tensor:
        """Reflect across a horizontal, vertical, diagonal, or antidiagonal axis."""
        if axis in {"horizontal", "h", "flip_h"}:
            return torch.flip(frame, dims=(-1,))
        if axis in {"vertical", "v", "flip_v"}:
            return torch.flip(frame, dims=(-2,))
        if axis in {"diag", "diagonal", "flip_diag"}:
            return frame.transpose(-2, -1)
        if axis in {"antidiag", "anti_diagonal", "flip_antidiag"}:
            return torch.flip(frame.transpose(-2, -1), dims=(-2, -1))
        raise ValueError(f"unknown reflection axis: {axis}")

    def random_d4(
        self,
        frame: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample uniformly from the eight D4 transforms."""
        idx = int(torch.randint(0, len(self.D4_TRANSFORMS), (), generator=generator).item())
        return self.apply_d4(frame, self.D4_TRANSFORMS[idx])

    def apply_d4(self, frame: torch.Tensor, transform: str) -> torch.Tensor:
        if transform == "identity":
            return frame.clone()
        if transform == "r90":
            return self.rotation_90k(frame, 1)
        if transform == "r180":
            return self.rotation_90k(frame, 2)
        if transform == "r270":
            return self.rotation_90k(frame, 3)
        if transform == "flip_h":
            return self.reflect(frame, "horizontal")
        if transform == "flip_v":
            return self.reflect(frame, "vertical")
        if transform == "flip_diag":
            return self.reflect(frame, "diagonal")
        if transform == "flip_antidiag":
            return self.reflect(frame, "antidiag")
        raise ValueError(f"unknown D4 transform: {transform}")

    def random_color_permutation(
        self,
        *,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        perm = torch.randperm(self.num_colors, device=device, generator=generator)
        if self.background_idx is not None:
            bg = int(self.background_idx)
            non_bg = torch.tensor(
                [idx for idx in range(self.num_colors) if idx != bg],
                device=device,
                dtype=torch.long,
            )
            shuffled = non_bg[torch.randperm(non_bg.numel(), device=device, generator=generator)]
            perm = torch.arange(self.num_colors, device=device, dtype=torch.long)
            perm[non_bg] = shuffled
            perm[bg] = bg
        return perm

    def __call__(
        self,
        frame: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        out = frame.clone()
        if torch.rand((), generator=generator).item() < self.d4_prob:
            out = self.random_d4(out, generator=generator)
        if torch.rand((), generator=generator).item() < self.color_perm_prob:
            perm = self.random_color_permutation(device=out.device, generator=generator)
            out = self.color_permutation(out, perm)
        return out
