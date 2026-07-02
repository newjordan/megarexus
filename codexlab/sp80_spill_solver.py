#!/usr/bin/env python3
"""Fail-closed tombstone for the removed SP80 white-box spill solver.

This module previously loaded local SP80 engine internals and derived layouts
from source-level mechanics. That is not an acceptable ARCPrize submission path
or baseline path for this lane.
"""

from __future__ import annotations

from typing import Any


REMOVAL_REASON = (
    "sp80_spill_solver removed from active use: white-box engine/source-derived "
    "planning is not allowed for leaderboard or baseline scoring in this lane."
)


def _removed(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError(REMOVAL_REASON)


plan_layout = _removed
solve_level = _removed
groundhog_metrics = _removed


def main() -> int:
    raise SystemExit(REMOVAL_REASON)


if __name__ == "__main__":
    main()
