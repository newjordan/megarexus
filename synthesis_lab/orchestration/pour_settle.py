"""Single source of truth for reading a pour's SETTLED outcome frame.

NIGHT 2026-07-05 root cause: harvested pour trajectories run PAST the pour's
settle point — on a win the level flashes then RESETS back to the arrangement
(targets 13->11, paint cleared), so steps[-1] is post-reset garbage with zero
satisfied targets. Every imaginer this session (v38 global, v39-42 CA, v43
one-shot) was trained on that post-reset frame and so never saw a single win
transition (11->13). That is a LABEL bug, not a structural wall.

Observed dynamics (L0, verified on 1600 trajs):
  * paint (6) grows monotonically +16/tick after the ACTION5 pour, then plateaus
    when the spill can no longer advance (win: filled into targets; fail: blocked).
  * on a COMPLETE win every target 11 flips to 13 (n11->0) AT that plateau and is
    held for several frames, THEN the episode resets.
  * color 14 is STRUCTURAL border geometry (present in the arrangement, ~62 cells)
    that merely flickers during the win-flash. It is NOT a wall-hit/fail marker.

Correct settled frame = FIRST frame at max paint mass after the last pour. This
lands on the held win/fail plateau for both outcomes. Win = all targets satisfied
(some 13, no remaining 11). Legal: pure forward-physics reading of env transitions.
"""

from __future__ import annotations

import numpy as np

PAINT, SOURCE, TARGET, FILLED, BORDER = 6, 4, 11, 13, 14


def _count(frame, color):
    return int((np.asarray(frame) == color).sum())


def settle_index(steps):
    """Index of the settled pour-outcome frame: first frame reaching the maximum
    paint(6) mass on/after the last ACTION5 pour. Robust for both win and fail
    (paint plateaus at the outcome, then any reset reverts paint to ~16)."""
    acts = [int(s.get("action_id", -1)) for s in steps]
    pour = max((i for i, a in enumerate(acts) if a == 5), default=0)
    n6 = np.array([_count(s["frame"], PAINT) for s in steps])
    rel = int(np.argmax(n6[pour:]))  # first occurrence of the plateau max
    return pour + rel


def settle_frame(steps):
    return np.asarray(steps[settle_index(steps)]["frame"], dtype=np.int8)


def pour_arrangement(steps):
    """The pre-pour arrangement frame = the ACTION5 frame (the last one)."""
    acts = [int(s.get("action_id", -1)) for s in steps]
    pour = max((i for i, a in enumerate(acts) if a == 5), default=0)
    return np.asarray(steps[pour]["frame"], dtype=np.int8)


def is_win(frame):
    """Complete win: every target satisfied — some FILLED(13) and no TARGET(11)
    left. Color 14 (structural border) is irrelevant to win/fail."""
    f = np.asarray(frame)
    return bool((f == FILLED).any()) and not bool((f == TARGET).any())


def targets_left(frame):
    return _count(frame, TARGET)


def filled(frame):
    return _count(frame, FILLED)
