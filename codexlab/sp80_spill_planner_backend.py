"""Fail-closed tombstone for the removed SP80 white-box spill planner backend.

This backend previously depended on source-derived SP80 spill planning. It is
kept only as an import-compatible quarantine boundary so old policy names cannot
silently run or produce score evidence.
"""

from __future__ import annotations

from typing import Any


PLANNER_SCHEMA = "codexlab.sp80_spill_planner_removed.v1"
REMOVAL_REASON = (
    "sp80_spill_planner_backend removed from active use: white-box "
    "engine/source-derived SP80 planning is not allowed for leaderboard or "
    "baseline scoring in this lane."
)


def rotation_prior(_rotation_k: int) -> dict[str, tuple[int, int]]:
    raise RuntimeError(REMOVAL_REASON)


def search_winning_layout(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(REMOVAL_REASON)


class SpillPlannerBrain:
    """Import-compatible fail-closed placeholder."""

    removal_reason = REMOVAL_REASON

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError(REMOVAL_REASON)


__all__ = ["SpillPlannerBrain", "search_winning_layout", "rotation_prior", "PLANNER_SCHEMA"]
