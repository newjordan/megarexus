"""Transcript-derived proposal macros (hypothesis priors, not labels).

Sources: SP80 narration usefulness + Loom algorithm confirmations themes.
Env still grades — these only bias propose_logits / conceptual proposals.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Named macros from human play language (SP80 + cross-game algorithms)
MACROS: list[dict[str, Any]] = [
    {
        "id": "isolation_motion",
        "hint": "identify movable vs fixed; prefer probes on changeable components",
        "bias_actions": {1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15, 6: 0.25},
    },
    {
        "id": "route_then_pour",
        "hint": "arrange routing geometry before ACTION5 pour",
        "bias_actions": {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.35},
    },
    {
        "id": "conserve_fuel",
        "hint": "tight pour budget — avoid thrash; fewer random ACTION5",
        "bias_actions": {5: -0.2, 6: 0.1},
    },
    {
        "id": "faucet_align",
        "hint": "align under spout / vertical then side-link",
        "bias_actions": {2: 0.1, 4: 0.1, 5: 0.15},
    },
    {
        "id": "symmetry_gap",
        "hint": "complete U/gap symmetry before pour",
        "bias_actions": {1: 0.08, 2: 0.08, 3: 0.08, 4: 0.08},
    },
    {
        "id": "irreversible_gate",
        "hint": "treat pour as costly; prefer placement probes first",
        "bias_actions": {5: -0.15, 6: 0.2},
    },
]


def macro_action_bias(
    available_actions: list[int] | None = None,
    n_actions: int = 8,
) -> np.ndarray:
    """Return additive logit bias [n_actions] from all macros (sum)."""
    bias = np.zeros(n_actions, dtype=np.float32)
    avail = set(int(a) for a in (available_actions or list(range(1, 7))))
    for m in MACROS:
        for a, v in (m.get("bias_actions") or {}).items():
            ai = int(a)
            if ai < n_actions and (not avail or ai in avail):
                bias[ai] += float(v)
    return bias


def macro_packet(map_components: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """JSON-serializable macro summary for dream_summary."""
    n_comp = len(map_components or [])
    # simple geometry cues from component count
    active = [m["id"] for m in MACROS]
    if n_comp <= 2:
        active = ["isolation_motion", "irreversible_gate"]
    elif n_comp >= 6:
        active = ["route_then_pour", "symmetry_gap", "conserve_fuel"]
    return {
        "name": "transcript_macros",
        "status": "ok",
        "hypothesis_only": True,
        "scores": {
            "n_macros": float(len(MACROS)),
            "n_active": float(len(active)),
            "n_components_cue": float(n_comp),
        },
        "active_macros": active,
        "messages": [f"macros:{','.join(active)}"],
        "admitted": False,
    }


def apply_macro_bias_to_logits(
    logits,  # torch.Tensor [1,A] or [A]
    available_actions: list[int] | None = None,
):
    import torch

    bias = macro_action_bias(available_actions, n_actions=int(logits.shape[-1]))
    b = torch.as_tensor(bias, device=logits.device, dtype=logits.dtype)
    if logits.dim() == 1:
        return logits + b
    return logits + b.unsqueeze(0)
