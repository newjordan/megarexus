"""Canonical grid/coordinate helpers for Crucible transition rows.

SP80 harness state coordinates are `[x, y]`. Saved `.npy` grids are indexed in
that same order as `grid[x, y]`. Crucible's display coordinate is the canonical
coordinate returned by `canonicalize_coord()`; consumers should call this module
instead of unpacking coordinate pairs ad hoc.
"""

from __future__ import annotations

from typing import Any

import numpy as np


SUPPORTED_ROTATION_QUARTERS = {0, 2}


def normalize_coord(coord: Any) -> tuple[int, int] | None:
    if not isinstance(coord, (list, tuple)) or len(coord) != 2:
        return None
    a0, a1 = coord
    if not isinstance(a0, (int, float)) or not isinstance(a1, (int, float)):
        return None
    return int(a0), int(a1)


def coord_xy_to_grid_index(coord: Any) -> tuple[int, int] | None:
    """Return the `grid[x, y]` index for an SP80 `[x, y]` coordinate."""
    return normalize_coord(coord)


def grid_index_to_coord_xy(index: Any) -> list[int] | None:
    """Return an SP80 `[x, y]` coordinate from a `grid[x, y]` index."""
    pair = normalize_coord(index)
    if pair is None:
        return None
    return [pair[0], pair[1]]


def canonicalize_grid(grid: np.ndarray, rotation_quarters: int) -> np.ndarray:
    """Return grid in display orientation for supported SP80 rotations."""
    rq = int(rotation_quarters)
    if rq not in SUPPORTED_ROTATION_QUARTERS:
        raise ValueError(f"unsupported rotation_quarters={rotation_quarters!r}")
    arr = np.asarray(grid)
    return arr if rq == 0 else np.rot90(arr, 2)


def canonicalize_coord(coord: Any, shape: tuple[int, int], rotation_quarters: int) -> tuple[int, int] | None:
    """Map raw `[x, y]` into the same canonical frame as canonicalize_grid()."""
    pair = normalize_coord(coord)
    if pair is None:
        return None
    a0, a1 = pair
    size0, size1 = int(shape[0]), int(shape[1])
    if a0 < 0 or a1 < 0 or a0 >= size0 or a1 >= size1:
        return None
    rq = int(rotation_quarters)
    if rq == 0:
        return a0, a1
    if rq == 2:
        return size0 - 1 - a0, size1 - 1 - a1
    raise ValueError(f"unsupported rotation_quarters={rotation_quarters!r}")


def display_coord(coord: Any, shape: tuple[int, int], rotation_quarters: int) -> tuple[int, int] | None:
    """Return the display coordinate; display is canonical for Crucible rows."""
    return canonicalize_coord(coord, shape, rotation_quarters)


def coord_delta(before: tuple[int, int] | None, after: tuple[int, int] | None) -> tuple[int, int] | None:
    if before is None or after is None:
        return None
    return int(after[0] - before[0]), int(after[1] - before[1])
