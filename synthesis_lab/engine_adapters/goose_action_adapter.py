"""STOCHASTIC_GOOSE_RUNTIME adapter: stochastic action sampling.

Mirrors Stochastic Goose's ``_sample_from_combined_output``
(stochastic_goose/custom_agents/action.py) over a 4101-logit vector:
sigmoid on logits, mask by available actions, divide the 4096 click-cell
probabilities by 4096 so "click" competes as one action type, renormalize,
then sample. Used two ways:

1. During data collection (orchestration/collect_sp80_rollouts.py) with
   ``logits=None`` — pure uniform stochastic proposals, Goose's cold-start
   behavior before its model has learned anything.
2. At inference over the MegarexusCNN's ``change_logits`` head — Goose's
   own "prefer actions predicted to change the frame" heuristic.

The adapter proposes actions; it never claims to own a solved level.
"""

from __future__ import annotations

import numpy as np

GRID = 64
NUM_SIMPLE = 5  # ACTION1..ACTION5
HEAD_DIM = NUM_SIMPLE + GRID * GRID  # 4101
CLICK_ACTION_ID = 6


def sample_goose_action(
    available_actions: list[int],
    logits: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
    temperature: float = 1.0,
) -> tuple[int, int, int]:
    """Sample (action_id, x, y) Goose-style.

    ``available_actions`` holds ARC action ids (subset of 1..6 here; RESET
    is handled by the caller on GAME_OVER/NOT_PLAYED, as in Goose).
    ``logits`` is the 4101-dim head output, or None for uniform sampling.
    """
    rng = rng or np.random.default_rng()
    if logits is None:
        probs = np.ones(HEAD_DIM, dtype=np.float64)
    else:
        if logits.shape != (HEAD_DIM,):
            raise ValueError(f"expected logits shape ({HEAD_DIM},), got {logits.shape}")
        probs = 1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64) / temperature))

    mask = np.zeros(HEAD_DIM, dtype=np.float64)
    for aid in available_actions:
        if 1 <= aid <= NUM_SIMPLE:
            mask[aid - 1] = 1.0
        elif aid == CLICK_ACTION_ID:
            mask[NUM_SIMPLE:] = 1.0
    probs = probs * mask
    # Goose: down-weight each click cell so "click" acts as one action type
    probs[NUM_SIMPLE:] /= GRID * GRID
    total = probs.sum()
    if total <= 0:
        raise ValueError(f"no available actions to sample from: {available_actions}")
    probs /= total

    idx = int(rng.choice(HEAD_DIM, p=probs))
    if idx < NUM_SIMPLE:
        return idx + 1, 0, 0
    cell = idx - NUM_SIMPLE
    return CLICK_ACTION_ID, cell % GRID, cell // GRID
