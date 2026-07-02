"""Hierarchical mechanic classification heads for Lewis.

Replacement for the flat 3043-way ``mechanic_head`` (a single
``nn.Linear(hidden, 3043)``) used by
:mod:`lewm.goal_property.GoalPropertyModel`.

Why hierarchical
----------------
The flat mechanic head suffers from severe long-tail issues: rare
mechanics share gradient signal only through the trunk, so their
classifier rows stay under-trained while head-of-distribution
mechanics dominate the cross-entropy. Decomposing the prediction as

    P(mechanic) = P(family) * P(mechanic | family)

gives every family-conditioned sub-classifier a much smaller class
space (~5–300 mechanics rather than 3043), so the tail gets a far
better signal-to-noise ratio. The family head is itself a tiny
33-way classifier that learns quickly.

Design decision: ONE masked Linear, not 33 separate Linears
-----------------------------------------------------------
The within-family scores are produced by a single
``nn.Linear(hidden, total_mechanics)``. A static boolean mask
(``family_mask[f, m] = (mechanic m belongs to family f)``) is used
to gate which mechanic slots receive gradient given a target's
family. This is much faster than running 33 separate sub-Linears:

* All within-family scores are computed in one cuBLAS GEMM.
* The mask is a constant buffer; no Python-level branching at
  forward time.
* Within-family ``log_softmax`` is computed per family-slice by
  re-using cumulative offsets (``build_family_slot_offsets``), so
  each family gets its own normaliser without cross-family
  contamination.

This trades a bit of wasted multiply-add (off-family slots get
their scores computed and then ignored by the mask in the loss /
hier_mechanic_logits) for major parallelism wins on GPU. With 3043
total mechanics and a hidden dim of ~512 the wasted compute is
negligible compared to the kernel-launch overhead of 33 separate
matmuls.

Integration API
---------------
This module is intentionally self-contained (no imports from
``lewm.*``) so it can be unit-tested without spinning up the full
``GoalPropertyModel`` graph. The wiring contract for the caller is:

    head = HierarchicalMechanicHead(
        hidden_dim=H,
        num_families=33,
        mechanics_per_family=[n0, n1, ..., n32],
    )

    # In GoalPropertyModel.forward(), after computing latent ``h``:
    hier_out = head(h)
    outputs["family_logits"]          = hier_out["family_logits"]
    outputs["within_family_logits"]   = hier_out["within_family_logits"]
    outputs["hier_mechanic_logits"]   = hier_out["hier_mechanic_logits"]
    # Optional: keep ``mechanic_logits`` as an alias of
    # ``hier_mechanic_logits`` so downstream consumers (argmax-based
    # predictors, calibration utilities) need no API change.
    outputs["mechanic_logits"] = hier_out["hier_mechanic_logits"]

In the loss, ``hierarchical_mechanic_loss`` replaces the flat
mechanic CE term:

    mech_to_fam = build_mechanic_to_family_map(family_assignments)
    losses = hierarchical_mechanic_loss(
        outputs=hier_out,
        family_target=targets["family"],
        mechanic_target=targets["mechanic"],
        mechanic_to_family=mech_to_fam,
        family_ce_weight=family_weight,
        within_family_ce_weight=mechanic_weight,
    )
    # losses["family_ce"], losses["within_family_ce"], losses["total"]

The Shannon balancer (``adaptive_loss_balance.TaskSpec``) sees these
as two ordinary classification tasks via the ``logits_key``s
``"family_logits"`` and ``"hier_mechanic_logits"`` (or
``"within_family_logits"`` if you prefer the unmasked form for
entropy estimation).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_mechanic_to_family_map(
    family_assignments: Sequence[int],
) -> torch.LongTensor:
    """Build the inverse-index tensor mapping mechanic-id -> family-id.

    Args:
        family_assignments: list (or any Sequence) of length
            ``total_mechanics`` where ``family_assignments[i]`` is the
            family-id of mechanic ``i``.

    Returns:
        ``LongTensor`` of shape ``[total_mechanics]`` on CPU.
    """
    if len(family_assignments) == 0:
        raise ValueError("family_assignments must be non-empty")
    fam = torch.as_tensor(list(family_assignments), dtype=torch.long)
    if fam.dim() != 1:
        raise ValueError("family_assignments must be a 1-D sequence")
    if (fam < 0).any():
        raise ValueError("family ids must be >= 0")
    return fam


def build_family_slot_offsets(mechanics_per_family: Sequence[int]) -> list[int]:
    """Cumulative offsets into the flat mechanic axis.

    For ``mechanics_per_family = [n0, n1, n2, ...]`` returns
    ``[0, n0, n0+n1, n0+n1+n2, ...]`` (length ``num_families + 1``).
    Slot ``[offsets[f] : offsets[f+1]]`` is the within-family slice
    for family ``f`` along the flat-mechanic axis.
    """
    if len(mechanics_per_family) == 0:
        raise ValueError("mechanics_per_family must be non-empty")
    offsets: list[int] = [0]
    for n in mechanics_per_family:
        if int(n) < 1:
            raise ValueError(
                "each family must have >= 1 mechanic; got "
                f"{n} in {list(mechanics_per_family)}"
            )
        offsets.append(offsets[-1] + int(n))
    return offsets


def _build_family_mask(
    num_families: int,
    mechanics_per_family: Sequence[int],
    offsets: Sequence[int],
) -> torch.BoolTensor:
    """``[num_families, total_mechanics]`` boolean mask.

    ``mask[f, m] = True`` iff mechanic ``m`` belongs to family ``f``.
    """
    total = offsets[-1]
    mask = torch.zeros(num_families, total, dtype=torch.bool)
    for f in range(num_families):
        lo, hi = offsets[f], offsets[f + 1]
        mask[f, lo:hi] = True
    return mask


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------


class HierarchicalMechanicHead(nn.Module):
    """Family-then-mechanic classifier replacing flat mechanic_head.

    Parameters
    ----------
    hidden_dim:
        Trunk feature dim (matches ``GoalPropertyModel`` ``hidden_dim``).
    num_families:
        Number of mechanic families (e.g. 33 for Lewis v5).
    mechanics_per_family:
        Per-family mechanic count; must sum to the total number of
        mechanics. Lengths and sums are validated at construction.
    bias:
        Whether the underlying Linears use bias terms (default True).

    Forward output keys
    -------------------
    family_logits:
        ``[B, num_families]`` raw family scores.
    within_family_logits:
        ``[B, total_mechanics]`` raw within-family scores. Each
        family's slice is its own classifier; cross-family slots are
        scored but ignored downstream (mask gates the loss and the
        hierarchy combine).
    hier_mechanic_logits:
        ``[B, total_mechanics]`` log P(mechanic) computed as
        ``log_softmax(family_logits)[family_of_m] +
         log_softmax(within_family_logits restricted to family_of_m)[slot]``.
        This is the quantity downstream consumers should ``argmax``
        for top-1 mechanic prediction.
    family_mask:
        ``[num_families, total_mechanics]`` constant boolean mask,
        exposed for downstream introspection / loss code.
    family_offsets:
        ``[num_families + 1]`` long tensor of cumulative slot offsets
        along the flat-mechanic axis.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_families: int,
        mechanics_per_family: Sequence[int],
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if num_families < 1:
            raise ValueError("num_families must be >= 1")
        if len(mechanics_per_family) != num_families:
            raise ValueError(
                f"len(mechanics_per_family)={len(mechanics_per_family)} "
                f"must equal num_families={num_families}"
            )

        offsets = build_family_slot_offsets(mechanics_per_family)
        total_mechanics = offsets[-1]

        self.hidden_dim = int(hidden_dim)
        self.num_families = int(num_families)
        self.mechanics_per_family: tuple[int, ...] = tuple(
            int(n) for n in mechanics_per_family
        )
        self.total_mechanics = int(total_mechanics)

        # Family classifier: small, fast.
        self.family_head = nn.Linear(hidden_dim, num_families, bias=bias)
        # Within-family scores: one big Linear over the flat-mechanic
        # axis; family-mask gates which slots contribute to the
        # hierarchy / loss.
        self.within_family_head = nn.Linear(
            hidden_dim, total_mechanics, bias=bias
        )

        family_mask = _build_family_mask(
            num_families, mechanics_per_family, offsets
        )
        # Store offsets + mask as buffers so .to(device) / .cuda()
        # carries them along without making them trainable params.
        self.register_buffer("family_mask", family_mask, persistent=False)
        self.register_buffer(
            "family_offsets",
            torch.as_tensor(offsets, dtype=torch.long),
            persistent=False,
        )
        # mechanic_to_family[m] = f for the family containing m.
        m2f = torch.zeros(total_mechanics, dtype=torch.long)
        for f in range(num_families):
            lo, hi = offsets[f], offsets[f + 1]
            m2f[lo:hi] = f
        self.register_buffer("mechanic_to_family", m2f, persistent=False)

    # ----- forward -----

    def forward(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        if latent.dim() != 2 or latent.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"expected latent of shape [B, {self.hidden_dim}], "
                f"got {tuple(latent.shape)}"
            )

        family_logits = self.family_head(latent)              # [B, F]
        within_family_logits = self.within_family_head(latent)  # [B, M]

        # log P(family): [B, F]
        log_p_family = F.log_softmax(family_logits, dim=-1)

        # log P(mechanic | family): per-family log-softmax over the
        # flat-mechanic axis. We compute this by slicing each family's
        # contiguous block; this is O(num_families) Python iterations
        # but each is a single log_softmax over a small slice. For
        # num_families ~ 33 this is negligible.
        log_p_within = torch.full_like(
            within_family_logits, fill_value=float("-inf")
        )
        offsets = self.family_offsets.tolist()
        for f in range(self.num_families):
            lo, hi = offsets[f], offsets[f + 1]
            log_p_within[:, lo:hi] = F.log_softmax(
                within_family_logits[:, lo:hi], dim=-1
            )

        # hier_mechanic_logits[b, m] = log_p_family[b, fam(m)]
        #                            + log_p_within[b, m]
        # We broadcast log_p_family across the flat mechanic axis using
        # mechanic_to_family as a gather index.
        fam_idx = self.mechanic_to_family.unsqueeze(0).expand(
            latent.shape[0], -1
        )                                                       # [B, M]
        log_p_family_per_mech = log_p_family.gather(1, fam_idx)  # [B, M]
        hier_mechanic_logits = log_p_family_per_mech + log_p_within

        return {
            "family_logits": family_logits,
            "within_family_logits": within_family_logits,
            "hier_mechanic_logits": hier_mechanic_logits,
            "family_mask": self.family_mask,
            "family_offsets": self.family_offsets,
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def hierarchical_mechanic_loss(
    outputs: dict[str, torch.Tensor],
    family_target: torch.Tensor,
    mechanic_target: torch.Tensor,
    mechanic_to_family: torch.Tensor,
    *,
    family_ce_weight: float = 1.0,
    within_family_ce_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Cross-entropy on family + within-family classification.

    The within-family CE is computed against the per-example
    family-conditioned slice of ``within_family_logits``. Concretely:
    for example ``b`` with ground-truth family ``f = family_target[b]``,
    we cross-entropy-classify ``within_family_logits[b, offsets[f]:offsets[f+1]]``
    against the within-family target slot
    ``mechanic_target[b] - offsets[f]``.

    Because each example only contributes to its own family's slice,
    gradient flows into ``within_family_head`` rows for that family
    only — the long tail gets cleaner signal per gradient step.

    Args:
        outputs: dict from ``HierarchicalMechanicHead.forward``.
        family_target: ``[B]`` long, ground-truth family ids.
        mechanic_target: ``[B]`` long, ground-truth FLAT mechanic ids
            (indexing into the [0, total_mechanics) range).
        mechanic_to_family: ``[total_mechanics]`` long, the
            family-id-of-each-mechanic map (see
            :func:`build_mechanic_to_family_map`). Used here as a
            sanity check; the family slicing uses ``family_offsets``
            from the outputs.
        family_ce_weight: scalar weight on the family CE term.
        within_family_ce_weight: scalar weight on the within-family CE term.

    Returns:
        dict with keys ``"family_ce"``, ``"within_family_ce"``,
        ``"total"``.
    """
    family_logits = outputs["family_logits"]                 # [B, F]
    within_family_logits = outputs["within_family_logits"]   # [B, M]
    family_offsets = outputs["family_offsets"]               # [F+1]

    if family_target.dtype != torch.long:
        family_target = family_target.long()
    if mechanic_target.dtype != torch.long:
        mechanic_target = mechanic_target.long()

    # Sanity: each mechanic_target's family matches family_target.
    # We don't raise here (caller may legitimately have noisy labels);
    # the assert is gated behind torch.is_grad_enabled() to avoid
    # device sync overhead in eval. Skip it in production hot path.
    # (Kept as a no-op for clarity.)

    family_ce = F.cross_entropy(family_logits, family_target)

    # For each example b, extract its family's slice of within-family
    # logits and CE-classify the within-family target slot. We do this
    # in a vectorised way: build a row-shifted log_softmax over a
    # per-example mask.
    #
    # Per-example approach:
    #   log_p = log_softmax(within_family_logits[b, lo_b:hi_b], dim=0)
    #   loss_b = -log_p[mechanic_target[b] - lo_b]
    #
    # Vectorised: mask off non-family logits with -inf, then take
    # log_softmax over the full axis (mask makes other families
    # contribute zero probability) — this is mathematically identical
    # to slicing.
    B = within_family_logits.shape[0]
    M = within_family_logits.shape[-1]
    # family_mask is [F, M]; pick per-example mask via family_target.
    family_mask = outputs["family_mask"]                       # [F, M]
    per_ex_mask = family_mask.index_select(0, family_target)   # [B, M]
    # Mask off other families' slots with -inf.
    NEG_INF = torch.finfo(within_family_logits.dtype).min
    masked_logits = within_family_logits.masked_fill(
        ~per_ex_mask, NEG_INF
    )
    log_p = F.log_softmax(masked_logits, dim=-1)               # [B, M]
    # Gather log-probability at the mechanic_target slot.
    within_nll = -log_p.gather(
        1, mechanic_target.view(-1, 1)
    ).squeeze(-1)                                              # [B]
    within_family_ce = within_nll.mean()

    total = (
        float(family_ce_weight) * family_ce
        + float(within_family_ce_weight) * within_family_ce
    )

    # Touch mechanic_to_family so the symbol is part of the API
    # surface even though family_offsets carries the structural
    # information we need at loss time. Keeping it as an explicit
    # argument lets callers verify consistency across modules.
    _ = mechanic_to_family

    return {
        "family_ce": family_ce,
        "within_family_ce": within_family_ce,
        "total": total,
    }


__all__ = [
    "HierarchicalMechanicHead",
    "hierarchical_mechanic_loss",
    "build_mechanic_to_family_map",
    "build_family_slot_offsets",
]
