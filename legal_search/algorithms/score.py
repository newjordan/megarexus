"""Scoring signals that are always legal — env fields + pixel diffs only."""

from __future__ import annotations

from typing import Any

import numpy as np

from legal_search.algorithms.components import frame_delta_count


def env_score_signals(
    *,
    levels_before: int,
    levels_after: int,
    frame_before: np.ndarray | None = None,
    frame_after: np.ndarray | None = None,
    state_after: str = "",
) -> dict[str, Any]:
    levels_up = int(levels_after) > int(levels_before)
    delta = 0
    if frame_before is not None and frame_after is not None:
        delta = frame_delta_count(frame_before, frame_after)
    return {
        "levels_before": int(levels_before),
        "levels_after": int(levels_after),
        "levels_up": bool(levels_up),
        "frame_delta_pixels": int(delta),
        "state_after": str(state_after),
        "win_token_present": "WIN" in str(state_after).upper(),
        "judge": "environment_fields_and_pixels_only",
        "solve_truth_admitted": False,
    }


def proxy_search_score(
    *,
    frame_delta_pixels: int,
    levels_up: bool,
    program_len: int,
) -> float:
    """Cheap stand-in when imagination model is absent.

    Prefers level-ups, then change, penalizes long programs.
    Not a win predictor — exploration bias only.
    """
    score = 0.0
    if levels_up:
        score += 100.0
    score += min(50.0, frame_delta_pixels / 8.0)
    score -= 0.05 * program_len
    return float(score)
