"""Operator-facing preflight summary for the Lewis training pathway.

The job: tell the operator, in plain text, whether ANY loss term has a
zero effective starting weight or any param-group has a zero LR. This
is the "is anything frozen?" / "are our numbers going up?" sanity
check, run once at trainer construction time, before a single step.

Reads the live :class:`PropertyTrainer` after ``__init__`` so it sees
the real param groups, the real balancer (including grad-aware
wrapper), and the real model surface. Returns a string for the caller
to log; never raises.
"""

from __future__ import annotations

from typing import Iterable

import torch


_LOSS_TERMS = (
    "agent_weight",
    "goal_type_weight",
    "keys_weight",
    "delta_px_weight",
    "mechanic_weight",
    "family_weight",
    "subclass_weight",
    "stage_weight",
    "cond_mechanic_weight",
    "goal_family_weight",
    "role_buckets_weight",
    "capability_weight",
    "topology_weight",
    "contrastive_weight",
    "sigreg_weight",
    "hier_family_ce_weight",
    "hier_within_family_ce_weight",
)


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    base = model
    if hasattr(base, "module"):
        base = base.module
    if hasattr(base, "_orig_mod"):
        base = base._orig_mod
    return base


def _format_weight(name: str, value: float) -> str:
    flag = ""
    if value == 0.0:
        flag = "  ← ZERO (silent)"
    elif value < 0.0:
        flag = "  ← NEGATIVE (likely bug)"
    return f"    {name:<30} {value:>10.4f}{flag}"


def _format_lr(group: dict) -> str:
    name = group.get("name", "default")
    lr = float(group.get("lr", 0.0))
    mult = float(group.get("lr_mult", 1.0))
    n_params = sum(p.numel() for p in group["params"])
    flag = ""
    if lr == 0.0:
        flag = "  ← ZERO LR (frozen)"
    elif mult == 0.0:
        flag = "  ← ZERO MULT (frozen)"
    return (
        f"    {name:<22} lr={lr:.2e}  mult={mult:.3f}  "
        f"params={n_params:>10,}{flag}"
    )


def _count_frozen(model: torch.nn.Module) -> tuple[int, int, list[str]]:
    """Return (n_frozen_params, n_total_params, sample_frozen_names)."""
    n_frozen = 0
    n_total = 0
    frozen_names: list[str] = []
    base = _unwrap(model)
    for name, p in base.named_parameters():
        n_total += p.numel()
        if not p.requires_grad:
            n_frozen += p.numel()
            if len(frozen_names) < 5:
                frozen_names.append(name)
    return n_frozen, n_total, frozen_names


def _format_hier_state(model: torch.nn.Module, config) -> list[str]:
    base = _unwrap(model)
    lines: list[str] = []
    if getattr(base, "hier_mechanic_head", None) is None:
        return ["  Hier head: OFF (flat mechanic_head path only)"]
    additive = bool(getattr(base, "hier_additive_mode", False))
    w = float(getattr(base, "hier_mechanic_weight", 0.0))
    anneal = int(getattr(config, "hier_mechanic_weight_anneal_steps", 0))
    lines.append(
        f"  Hier head: ON ({'additive' if additive else 'replace'} mode)"
    )
    lines.append(
        f"    starting hier_mechanic_weight = {w:.4f}"
        + ("  (anneal active)" if anneal > 0 else "")
    )
    if anneal > 0:
        target = float(getattr(config, "hier_mechanic_weight", 0.0))
        lines.append(
            f"    anneal: 0 → {target:.4f} over {anneal} global steps"
        )
    if additive and w == 0.0 and anneal == 0:
        lines.append(
            "    ⚠ hier_mechanic_weight=0 with no anneal — hier signal "
            "is silent at inference (trains but never affects argmax)"
        )
    return lines


def _format_balancer_state(trainer) -> list[str]:
    bal = trainer.adaptive_balancer
    if bal is None:
        return ["  Adaptive balancer: OFF"]
    lines = [f"  Adaptive balancer: ON ({type(bal).__name__})"]
    eff = bal.effective_weights() if hasattr(bal, "effective_weights") else {}
    if eff:
        lines.append("    effective starting weights:")
        for task in sorted(eff):
            v = float(eff[task])
            flag = "  ← ZERO" if v == 0.0 else ""
            lines.append(f"      {task:<24} {v:>10.4f}{flag}")
    return lines


def _format_optional_components(trainer, config) -> list[str]:
    lines: list[str] = []
    has_contrastive = trainer.contrastive_augmenter is not None
    cw = float(getattr(config, "contrastive_weight", 0.0))
    if has_contrastive and cw == 0.0:
        lines.append(
            "  ⚠ contrastive augmenter loaded but contrastive_weight=0 — "
            "partner episodes loaded each step but loss zero'd (CPU waste)"
        )
    elif has_contrastive:
        lines.append(f"  Contrastive: ON (weight={cw:.4f})")
    else:
        lines.append("  Contrastive: OFF")
    ortho = trainer.orthogonalizer is not None
    if ortho:
        kind = type(trainer.orthogonalizer).__name__
        lines.append(f"  PCGrad orthogonalizer: ON ({kind})")
    if getattr(trainer, "live_probe", None) is not None:
        lines.append(
            f"  Live relational probe: ON  "
            f"warmup={trainer.live_probe.warmup_steps} "
            f"threshold={trainer.live_probe.conflict_threshold:.3f}"
        )
    return lines


def summarize(trainer) -> str:
    """Produce a single multi-line string summarising the training surface."""
    config = trainer.config
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("PROPERTY TRAINER PREFLIGHT — is anything silently zero?")
    lines.append("=" * 70)

    # Param groups + LRs
    lines.append("\n[param groups]")
    for g in trainer.optimizer.param_groups:
        lines.append(_format_lr(g))

    # Frozen counts
    n_frozen, n_total, frozen_names = _count_frozen(trainer.model)
    lines.append(
        f"\n[trainable] {n_total - n_frozen:,} of {n_total:,} params "
        f"({100.0 * (n_total - n_frozen) / max(1, n_total):.1f}% trainable)"
    )
    if n_frozen > 0:
        lines.append(
            f"  frozen sample: " + ", ".join(frozen_names)
            + (" ..." if len(frozen_names) >= 5 else "")
        )

    # Loss term static weights (from config — these are the BASE weights,
    # the balancer's effective_weights() shows the runtime-adjusted values)
    lines.append("\n[loss-term base weights]")
    for name in _LOSS_TERMS:
        if hasattr(config, name):
            lines.append(_format_weight(name, float(getattr(config, name))))

    # Hier head specifics
    lines.append("\n[hier head]")
    lines.extend(_format_hier_state(trainer.model, config))

    # Adaptive balancer
    lines.append("\n[balancer]")
    lines.extend(_format_balancer_state(trainer))

    # Other optional components
    lines.append("\n[other components]")
    lines.extend(_format_optional_components(trainer, config))

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


__all__ = ["summarize"]
