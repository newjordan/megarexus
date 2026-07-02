"""Additive composition of the flat mechanic head with the hier head.

Motivation
----------
``HierarchicalMechanicHead`` decomposes ``P(mechanic)`` as
``P(family) * P(mechanic | family)`` — great for long-tail signal but
also a *replacement* of the flat 3043-way head. That coupling is risky
in early smoke: if hier supervision underperforms (bad family-id maps,
sparse within-family slots), the flat path is gone.

This module keeps the flat mechanic_head training normally **and**
adds the hier head's family-decomposed score as a soft auxiliary:

    combined_logits[b, c] = flat_logits[b, c]
                          + w * hier_mechanic_logits[b, concept_to_flat[c]]

When ``w == 0`` the hier head is silent and the flat path is
mathematically unchanged. When ``w > 0`` the hier head contributes a
log-probability bias that nudges the joint argmax. The flat head's CE
keeps training the flat head as before; the hier CE keeps training
the hier head as before. The two are coupled only through the trunk's
gradients.

Two helpers:

* :func:`compute_hier_additive_logits` — combine flat + hier logits
  (re-permuting hier to concept-axis via ``concept_to_flat``).
* :func:`compute_hier_additive_loss` — flat-CE + weighted hier-CE
  ensemble, returning every sub-term in a dict so the caller can log
  / weight further.

Both work on the public output dicts of the existing heads — no model
surgery required.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F

from .hierarchical_heads import hierarchical_mechanic_loss


def compute_hier_additive_logits(
    flat_mechanic_logits: torch.Tensor,
    hier_outputs: Mapping[str, torch.Tensor],
    hier_mechanic_weight: float,
    *,
    concept_to_flat: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return ``flat_logits + w * hier_logits_in_concept_axis``.

    Args:
        flat_mechanic_logits: ``[B, total_mechanics]`` — the flat
            ``mechanic_head``'s raw logits, indexed by *concept* axis
            (the dataset-native ordering).
        hier_outputs: dict from
            :meth:`HierarchicalMechanicHead.forward`. Must contain
            ``hier_mechanic_logits`` (``[B, total_mechanics]`` on
            family-contiguous flat axis).
        hier_mechanic_weight: scalar ``w`` blending in the hier signal.
            ``w=0`` returns the flat logits unchanged.
        concept_to_flat: optional ``[total_mechanics]`` long tensor
            mapping concept_idx → flat_idx. If None, identity is
            assumed (concepts already family-contiguous).

    Returns:
        ``[B, total_mechanics]`` combined logits on the concept axis.
        Shape matches ``flat_mechanic_logits`` exactly.
    """
    if "hier_mechanic_logits" not in hier_outputs:
        raise KeyError("hier_outputs must contain 'hier_mechanic_logits'")
    hier_flat = hier_outputs["hier_mechanic_logits"]
    if hier_flat.shape != flat_mechanic_logits.shape:
        raise ValueError(
            f"hier shape {tuple(hier_flat.shape)} must match "
            f"flat shape {tuple(flat_mechanic_logits.shape)}"
        )

    w = float(hier_mechanic_weight)
    if w == 0.0:
        # Avoid spurious dependence on hier outputs when caller dials weight to 0
        return flat_mechanic_logits

    if concept_to_flat is None:
        hier_in_concept = hier_flat
    else:
        c2f = concept_to_flat.to(device=hier_flat.device, dtype=torch.long)
        if c2f.shape != (hier_flat.shape[-1],):
            raise ValueError(
                f"concept_to_flat shape {tuple(c2f.shape)} must equal "
                f"({hier_flat.shape[-1]},)"
            )
        # hier_in_concept[b, c] = hier_flat[b, concept_to_flat[c]]
        hier_in_concept = hier_flat.index_select(1, c2f)

    return flat_mechanic_logits + w * hier_in_concept


def compute_hier_additive_loss(
    flat_logits: torch.Tensor,
    hier_outputs: Mapping[str, torch.Tensor],
    mechanic_target: torch.Tensor,
    family_target: torch.Tensor,
    concept_to_flat: torch.Tensor,
    mechanic_to_family: torch.Tensor,
    *,
    flat_weight: float = 1.0,
    hier_weight: float = 1.0,
    hier_family_ce_weight: float = 1.0,
    hier_within_family_ce_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute flat CE on flat_logits + hier CE on hier outputs.

    The flat head is trained against ``mechanic_target`` (concept-axis
    ids) via vanilla cross-entropy on ``flat_logits``. The hier head is
    trained against ``family_target`` + a flat-axis-remapped
    ``mechanic_target`` via
    :func:`lewm.hierarchical_heads.hierarchical_mechanic_loss`.

    The combined total is::

        total = flat_weight * flat_ce
              + hier_weight * (
                  hier_family_ce_weight * hier_family_ce
                + hier_within_family_ce_weight * hier_within_family_ce
                )

    Dialing ``hier_weight=0`` reduces ``total`` to the flat CE term —
    a safe rollback if the hier supervision turns out to be noisy.

    Args:
        flat_logits: ``[B, total_mechanics]`` flat-head raw logits on
            the concept axis.
        hier_outputs: dict from :class:`HierarchicalMechanicHead.forward`.
        mechanic_target: ``[B]`` long, concept-axis mechanic ids.
        family_target: ``[B]`` long, family ids.
        concept_to_flat: ``[total_mechanics]`` long, concept → flat
            permutation. Used to remap ``mechanic_target`` for the hier
            within-family CE.
        mechanic_to_family: ``[total_mechanics]`` long, the family of
            each FLAT-axis mechanic id (passed straight through to
            ``hierarchical_mechanic_loss``).
        flat_weight: scalar weight on the flat CE term.
        hier_weight: scalar weight on the (combined) hier CE term.
        hier_family_ce_weight / hier_within_family_ce_weight: passed
            through to ``hierarchical_mechanic_loss``. Note these are
            *inside* the hier_weight multiplier — the hier_weight
            scales the whole hier subsystem uniformly.

    Returns:
        dict with tensors ``flat_ce``, ``hier_family_ce``,
        ``hier_within_family_ce``, ``total``.
    """
    if mechanic_target.dtype != torch.long:
        mechanic_target = mechanic_target.long()
    if family_target.dtype != torch.long:
        family_target = family_target.long()

    # Flat CE (concept-axis, standard nn.CrossEntropyLoss over flat_logits)
    flat_ce = F.cross_entropy(flat_logits, mechanic_target)

    # Hier CE — flatten the concept-axis target through concept_to_flat.
    c2f = concept_to_flat.to(device=mechanic_target.device, dtype=torch.long)
    flat_mechanic_target = c2f.gather(0, mechanic_target)
    hier_losses = hierarchical_mechanic_loss(
        hier_outputs,
        family_target,
        flat_mechanic_target,
        mechanic_to_family,
        family_ce_weight=hier_family_ce_weight,
        within_family_ce_weight=hier_within_family_ce_weight,
    )

    hier_family_ce = hier_losses["family_ce"]
    hier_within_family_ce = hier_losses["within_family_ce"]
    hier_total = (
        float(hier_family_ce_weight) * hier_family_ce
        + float(hier_within_family_ce_weight) * hier_within_family_ce
    )

    total = float(flat_weight) * flat_ce + float(hier_weight) * hier_total

    return {
        "flat_ce": flat_ce,
        "hier_family_ce": hier_family_ce,
        "hier_within_family_ce": hier_within_family_ce,
        "total": total,
    }


__all__ = [
    "compute_hier_additive_logits",
    "compute_hier_additive_loss",
]
