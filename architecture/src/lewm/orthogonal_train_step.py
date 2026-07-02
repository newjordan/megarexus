"""Safe PCGrad-style train-step helper.

``PairedGradientOrthogonalizer.project`` does N backward passes (one per
head, each preceded by ``model.zero_grad``) with ``retain_graph=True``.
A side-effect: head-LOCAL parameter gradients get clobbered every
iteration, so after the loop only the last head's local grads survive.
This module wraps the orthogonaliser so that head-local gradients are
correctly preserved and the trunk grads are correctly projected.

The procedure is:

  1. ``model.zero_grad(set_to_none=False)``
  2. ``total_loss.backward(retain_graph=True)`` — produces the correct
     head-local grads (each head's loss only touches its own head
     params) and the un-projected summed trunk grads.
  3. Stash a clone of every head-local ``param.grad`` keyed by name.
  4. Call ``orthogonalizer.project(head_losses)`` — runs N retain-graph
     backwards and returns the flat combined trunk grad. Side-effect:
     head-local grads now contain garbage (last head only).
  5. Use ``assign_flat_trunk_grads`` to write the projected trunk grads.
  6. Restore the stashed head-local grads — each param's ``.grad`` is
     overwritten with the stash's clone.

After this returns, ``optimizer.step()`` applies PCGrad on the trunk
and standard SGD/Adam on every head-local parameter.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch

from .relational_gradients import (
    PairedGradientOrthogonalizer,
    assign_flat_trunk_grads,
)


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Mirror ``PropertyTrainer._partition_trainable``: peel DDP / compile."""
    base = model
    if hasattr(base, "module"):
        base = base.module
    if hasattr(base, "_orig_mod"):
        base = base._orig_mod
    return base


def _stash_head_local_grads(
    model: torch.nn.Module,
    trunk_prefixes: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Clone ``.grad`` for every parameter whose name does NOT start with
    any trunk prefix. Returned dict is keyed by parameter name and holds
    detached clones (safe to overwrite later)."""
    base = _unwrap(model)
    prefixes = tuple(trunk_prefixes)
    stash: dict[str, torch.Tensor] = {}
    for name, p in base.named_parameters():
        if name.startswith(prefixes):
            continue
        if p.grad is None:
            continue
        stash[name] = p.grad.detach().clone()
    return stash


def _restore_head_local_grads(
    model: torch.nn.Module,
    stash: Mapping[str, torch.Tensor],
    trunk_prefixes: Sequence[str],
) -> None:
    """Write each stashed grad back into its parameter's ``.grad`` field.

    Trunk params are NOT touched here — the caller is responsible for
    having written projected trunk grads via ``assign_flat_trunk_grads``
    BEFORE calling restore. Head-local params whose grad was None in the
    stash are explicitly set to None so the optimiser ignores them.
    """
    base = _unwrap(model)
    prefixes = tuple(trunk_prefixes)
    for name, p in base.named_parameters():
        if name.startswith(prefixes):
            continue
        if name in stash:
            saved = stash[name].to(device=p.device, dtype=p.dtype)
            if p.grad is None:
                p.grad = saved.clone()
            else:
                p.grad.copy_(saved)
        else:
            # Parameter had no grad after total_loss.backward — clear
            # whatever the orthogonaliser scratch left behind.
            p.grad = None


def orthogonal_step(
    model: torch.nn.Module,
    head_losses: Mapping[str, torch.Tensor],
    total_loss: Optional[torch.Tensor],
    trunk_prefixes: Sequence[str],
    orthogonalizer: PairedGradientOrthogonalizer,
) -> None:
    """Compute correct head-local + PCGrad-projected trunk grads.

    Args:
        model: GoalPropertyModel (optionally DDP / torch.compile wrapped).
        head_losses: per-head loss tensors. May be a subset of all heads
            (heads with disabled losses can be absent).
        total_loss: scalar tensor combining all losses. If ``None``, the
            legacy path is used (just project + assign, head-local grads
            are not preserved — caller asked for this).
        trunk_prefixes: parameter-name prefixes that identify trunk
            parameters (everything else is treated as head-local).
        orthogonalizer: pre-built ``PairedGradientOrthogonalizer``.

    After return, every trainable parameter has the correct ``.grad``
    populated so ``optimizer.step()`` does PCGrad-trunk + plain
    head-local updates. The function does NOT call ``optimizer.step()``.
    """
    # --- Legacy / fast path: caller skipped the total-loss backward ----
    if total_loss is None:
        combined = orthogonalizer.project(head_losses)
        assign_flat_trunk_grads(model, combined, trunk_prefixes)
        return

    # --- Step 1: zero grads (set_to_none=False so .grad tensors exist) -
    model.zero_grad(set_to_none=False)

    # --- Step 2: correct head-local + summed trunk grads from total ----
    total_loss.backward(retain_graph=True)

    # --- Step 3: stash head-local grads BEFORE the orthogonaliser
    # clobbers them with its N retain_graph backwards.
    head_stash = _stash_head_local_grads(model, trunk_prefixes)

    # --- Step 4: PCGrad projection (this internally zero_grad's every
    # iteration and leaves head-local grads in the last head's state).
    combined = orthogonalizer.project(head_losses)

    # --- Step 5: write projected trunk grads back into the model ------
    assign_flat_trunk_grads(model, combined, trunk_prefixes)

    # --- Step 6: restore head-local grads from the stash --------------
    _restore_head_local_grads(model, head_stash, trunk_prefixes)
