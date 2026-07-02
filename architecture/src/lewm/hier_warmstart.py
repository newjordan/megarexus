"""Warmstart a HierarchicalMechanicHead from trained flat-head weights.

A typical Lewis run trains the flat 3043-way ``mechanic_head`` + flat
33-way ``family_head`` first, then we want to "upgrade" to the
hierarchical head without throwing away the flat head's learned class
prototypes. Copying the flat rows into the hier head's analogous slots
is the cheapest possible warmstart — the within-family classifier
starts where the flat mechanic head left off, just re-indexed by
family-contiguous slot order.

Two helpers:

* :func:`init_hier_from_flat` — direct tensor-level mutator. Use this
  when you already have all the relevant flat-head tensors in hand
  (e.g. inside a custom training script that just swapped heads).
* :func:`load_hier_warmstart_from_checkpoint` — convenience that
  parses the multiple checkpoint formats Lewis writes
  (``model_state_dict`` / ``model_state`` / ``state_dict`` / bare),
  extracts the relevant mechanic & family head tensors, and calls
  ``init_hier_from_flat``.

Neither helper touches the trunk or the family-mask buffers — only
the hier head's two ``nn.Linear``s.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch


def init_hier_from_flat(
    hier_head,
    flat_mechanic_weight: torch.Tensor,
    flat_mechanic_bias: Optional[torch.Tensor],
    flat_family_weight: torch.Tensor,
    flat_family_bias: Optional[torch.Tensor],
    concept_to_flat: torch.Tensor,
) -> None:
    """Mutate ``hier_head`` so its rows reproduce the flat heads.

    The family head is a 1:1 copy (same axis order). The within-family
    head's rows are *permuted*: flat row ``c`` (concept id ``c``) is
    copied to within-family slot ``concept_to_flat[c]``.

    Args:
        hier_head: ``HierarchicalMechanicHead`` instance to mutate.
        flat_mechanic_weight: ``[total_mechanics, hidden_dim]`` — the
            trained flat ``mechanic_head.weight``.
        flat_mechanic_bias: ``[total_mechanics]`` or None — the trained
            flat ``mechanic_head.bias``. May be None if either the flat
            head was bias-free or the hier head was constructed with
            ``bias=False`` (in that case nothing is copied for bias).
        flat_family_weight: ``[num_families, hidden_dim]`` — flat
            ``family_head.weight``.
        flat_family_bias: ``[num_families]`` or None — flat
            ``family_head.bias``.
        concept_to_flat: ``[total_mechanics]`` long — permutation
            mapping concept_idx → flat (family-contiguous) idx.

    The function is in-place and returns None; ``hier_head`` is
    mutated. No gradient state is touched.
    """
    # --- family head: direct copy ---
    fam_w = hier_head.family_head.weight
    if flat_family_weight.shape != fam_w.shape:
        raise ValueError(
            f"flat_family_weight shape {tuple(flat_family_weight.shape)} "
            f"must match hier_head.family_head.weight shape "
            f"{tuple(fam_w.shape)}"
        )
    with torch.no_grad():
        fam_w.copy_(flat_family_weight.to(device=fam_w.device, dtype=fam_w.dtype))
        if (
            flat_family_bias is not None
            and hier_head.family_head.bias is not None
        ):
            fam_b = hier_head.family_head.bias
            if flat_family_bias.shape != fam_b.shape:
                raise ValueError(
                    f"flat_family_bias shape {tuple(flat_family_bias.shape)} "
                    f"must match hier_head.family_head.bias shape "
                    f"{tuple(fam_b.shape)}"
                )
            fam_b.copy_(
                flat_family_bias.to(device=fam_b.device, dtype=fam_b.dtype),
            )

    # --- within-family head: permuted copy ---
    wfam_w = hier_head.within_family_head.weight  # [total_mech, hidden_dim]
    total_mech, hidden_dim = wfam_w.shape
    if flat_mechanic_weight.shape != wfam_w.shape:
        raise ValueError(
            f"flat_mechanic_weight shape {tuple(flat_mechanic_weight.shape)} "
            f"must match hier_head.within_family_head.weight shape "
            f"{tuple(wfam_w.shape)}"
        )
    c2f = concept_to_flat.to(device=wfam_w.device, dtype=torch.long)
    if c2f.shape != (total_mech,):
        raise ValueError(
            f"concept_to_flat shape {tuple(c2f.shape)} must equal "
            f"({total_mech},)"
        )

    src_w = flat_mechanic_weight.to(device=wfam_w.device, dtype=wfam_w.dtype)
    with torch.no_grad():
        # Row c of src goes to row c2f[c] of dst.
        # Equivalent: dst[c2f] = src.
        wfam_w.index_copy_(0, c2f, src_w)

        if (
            flat_mechanic_bias is not None
            and hier_head.within_family_head.bias is not None
        ):
            wfam_b = hier_head.within_family_head.bias
            if flat_mechanic_bias.shape != wfam_b.shape:
                raise ValueError(
                    f"flat_mechanic_bias shape {tuple(flat_mechanic_bias.shape)} "
                    f"must match hier_head.within_family_head.bias shape "
                    f"{tuple(wfam_b.shape)}"
                )
            src_b = flat_mechanic_bias.to(
                device=wfam_b.device, dtype=wfam_b.dtype,
            )
            wfam_b.index_copy_(0, c2f, src_b)


def _extract_state_dict(loaded) -> dict:
    """Pluck the state-dict out of the multiple checkpoint envelopes Lewis writes.

    Mirrors :func:`lewm.goal_property.from_pretrained_add_study_tags_heads`
    so we accept the same set of formats: ``model_state_dict``,
    ``model_state``, ``state_dict``, or a bare state-dict.
    """
    if isinstance(loaded, dict):
        for key in ("model_state_dict", "model_state", "state_dict"):
            if key in loaded:
                return loaded[key]
    return loaded


def load_hier_warmstart_from_checkpoint(
    hier_head,
    checkpoint_path: str | Path,
    concept_to_flat: torch.Tensor,
    *,
    mechanic_weight_key: str = "mechanic_head.weight",
    mechanic_bias_key: str = "mechanic_head.bias",
    family_weight_key: str = "family_head.weight",
    family_bias_key: str = "family_head.bias",
) -> None:
    """Read flat-head tensors out of a checkpoint + call ``init_hier_from_flat``.

    Args:
        hier_head: ``HierarchicalMechanicHead`` to mutate.
        checkpoint_path: path to a ``torch.save`` artefact. Accepts the
            usual envelopes (``model_state_dict`` / ``model_state`` /
            ``state_dict`` / bare).
        concept_to_flat: ``[total_mechanics]`` long permutation.
        mechanic_weight_key / mechanic_bias_key / family_weight_key /
        family_bias_key: keys in the state_dict to read from. Defaults
        match :class:`GoalPropertyModel`'s flat-head names.
    """
    loaded = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False,
    )
    sd = _extract_state_dict(loaded)
    if not isinstance(sd, dict):
        raise RuntimeError(
            f"could not find a state-dict inside checkpoint "
            f"{checkpoint_path}; got {type(loaded)}"
        )

    if mechanic_weight_key not in sd:
        raise KeyError(
            f"checkpoint missing '{mechanic_weight_key}' — "
            f"is the checkpoint actually a flat-mechanic-head Lewis run?"
        )
    if family_weight_key not in sd:
        raise KeyError(
            f"checkpoint missing '{family_weight_key}' — "
            f"is the checkpoint actually a flat-family-head Lewis run?"
        )

    flat_mechanic_weight = sd[mechanic_weight_key]
    flat_mechanic_bias = sd.get(mechanic_bias_key, None)
    flat_family_weight = sd[family_weight_key]
    flat_family_bias = sd.get(family_bias_key, None)

    init_hier_from_flat(
        hier_head=hier_head,
        flat_mechanic_weight=flat_mechanic_weight,
        flat_mechanic_bias=flat_mechanic_bias,
        flat_family_weight=flat_family_weight,
        flat_family_bias=flat_family_bias,
        concept_to_flat=concept_to_flat,
    )


__all__ = [
    "init_hier_from_flat",
    "load_hier_warmstart_from_checkpoint",
]
