#!/usr/bin/env python3
"""Frame-only visible-object perception for SP80 spill boards.

This module reads rendered 64x64 SP80 frames, not engine sprite state. It
recovers the visible core spill objects needed to begin replacing privileged
``extract_state(game)`` reads in the ARCPrize harness path.

Boundary: this is not a complete simulator state. The SP80 border, padding, and
wall hazards share color 1 in rendered frames; when they overlap, a single frame
does not identify which cells are true wall sprites versus board/border pixels.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Any, Iterable

import numpy as np


FRAME_SPILL_PERCEPTION_SCHEMA = "codexlab.sp80_frame_spill_perception.v1"
FRAME_SPILL_STATE_SCHEMA = "codexlab.sp80_frame_spill_state.v1"
CORE_VISIBLE_KINDS = {"source", "marker", "platform", "bucket", "diverter"}
VISIBLE_SPILL_VALUES = {4, 6, 8, 9, 11, 13, 15}
LIMITATIONS = [
    "wall_border_same_color_overlap_not_disambiguated_from_single_frame",
    "color1_barriers_are_conservative_wall_candidates",
    "visible_core_spill_objects_only",
    "not_solve_truth_or_complete_simulator_state",
]


def _as_frame(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    if arr.ndim != 2:
        raise ValueError(f"expected 2d SP80 frame, got shape {arr.shape}")
    if arr.shape[0] < 64 or arr.shape[1] < 64:
        raise ValueError(f"expected at least 64x64 SP80 frame, got shape {arr.shape}")
    return arr[:64, :64].astype(np.int16, copy=False)


def infer_grid_size(frame: Any) -> int:
    """Infer the SP80 logical grid size for the known 16x16/20x20 levels.

    The 20x20 render is letterboxed into 64 pixels at scale 3, leaving a 2-pixel
    color-1 pad before the board. The 16x16 render uses scale 4 with no pad.
    """
    arr = _as_frame(frame)
    return 20 if int(np.count_nonzero(arr[:, :2] == 1)) >= 96 else 16


def _block_mode(block: np.ndarray) -> int:
    values = [int(v) for v in block.reshape(-1)]
    non_ui = [v for v in values if v != 14]
    sample = non_ui or values
    counts = Counter(sample)
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def display_grid_from_frame(frame: Any, grid_size: int | None = None) -> np.ndarray:
    """Downsample a rendered frame into display-oriented logical cells."""
    arr = _as_frame(frame)
    n = int(grid_size or infer_grid_size(arr))
    if n not in (16, 20):
        raise ValueError(f"unsupported SP80 grid size {n}")
    scale = 64 // n
    offset = (64 - n * scale) // 2
    grid = np.zeros((n, n), dtype=np.int16)
    for y in range(n):
        for x in range(n):
            y0 = offset + y * scale
            x0 = offset + x * scale
            grid[y, x] = _block_mode(arr[y0 : y0 + scale, x0 : x0 + scale])
    return grid


def infer_rotation_quarters(display_grid: Any) -> int:
    """Infer frame rotation k where display = np.rot90(engine, k)."""
    dg = np.asarray(display_grid)
    best_k = 0
    best_score = None
    for k in range(4):
        engine = np.rot90(dg, k=(-k) % 4)
        score = _rotation_score(engine)
        if best_score is None or score > best_score:
            best_score = score
            best_k = k
    return best_k


def _rotation_score(engine_grid: np.ndarray) -> int:
    n = int(engine_grid.shape[0])
    score = 0
    source_y = np.argwhere(engine_grid == 6)[:, 0].tolist()
    marker_y = np.argwhere(engine_grid == 4)[:, 0].tolist()
    bucket_y = np.argwhere((engine_grid == 11) | (engine_grid == 13))[:, 0].tolist()
    movable = np.count_nonzero((engine_grid == 8) | (engine_grid == 9) | (engine_grid == 15))

    score += 12 * sum(1 for y in source_y if y <= 2)
    score += 6 * sum(1 for y in marker_y if y <= 2)
    score += 3 * sum(1 for y in bucket_y if y >= n // 2)
    score += int(movable > 0)
    score -= 10 * sum(1 for y in source_y if y > n // 2)
    score -= 4 * sum(1 for y in bucket_y if y < 3)
    return int(score)


def engine_grid_from_frame(
    frame: Any,
    grid_size: int | None = None,
    rotation_quarters: int | None = None,
) -> np.ndarray:
    display = display_grid_from_frame(frame, grid_size=grid_size)
    k = infer_rotation_quarters(display) if rotation_quarters is None else int(rotation_quarters)
    return np.rot90(display, k=(-k) % 4).copy()


def engine_cell_to_display_cell(
    x: int,
    y: int,
    grid_size: int,
    rotation_quarters: int,
) -> tuple[int, int]:
    """Map an engine/logical grid cell to the rotated display-grid cell."""
    n = int(grid_size)
    gx, gy = int(x), int(y)
    k = int(rotation_quarters) % 4
    if k == 0:
        return gx, gy
    if k == 1:
        return gy, n - 1 - gx
    if k == 2:
        return n - 1 - gx, n - 1 - gy
    return n - 1 - gy, gx


def display_click_for_engine_cell(
    x: int,
    y: int,
    grid_size: int,
    rotation_quarters: int,
) -> list[int]:
    """Return a 64x64 display click for an engine/logical grid cell."""
    n = int(grid_size)
    if n not in (16, 20):
        raise ValueError(f"unsupported SP80 grid size {n}")
    dx, dy = engine_cell_to_display_cell(x, y, n, rotation_quarters)
    scale = 64 // n
    offset = (64 - n * scale) // 2
    return [int(offset + dx * scale + scale // 2), int(offset + dy * scale + scale // 2)]


def _display_bbox_for_cells(
    cells: list[list[int]],
    grid_size: int,
    rotation_quarters: int,
) -> list[int]:
    n = int(grid_size)
    scale = 64 // n
    offset = (64 - n * scale) // 2
    boxes = []
    for x, y in cells:
        dx, dy = engine_cell_to_display_cell(int(x), int(y), n, rotation_quarters)
        boxes.append((offset + dx * scale, offset + dy * scale, offset + (dx + 1) * scale - 1, offset + (dy + 1) * scale - 1))
    return [
        int(min(b[0] for b in boxes)),
        int(min(b[1] for b in boxes)),
        int(max(b[2] for b in boxes)),
        int(max(b[3] for b in boxes)),
    ]


def frame_object_click_xy(
    obj: dict[str, Any],
    grid_size: int,
    rotation_quarters: int,
) -> list[int]:
    """Return a frame-derived display click for a perceived object."""
    cells = [tuple(int(v) for v in cell) for cell in obj.get("occ", [])]
    if not cells:
        raise ValueError("object has no occupied cells")
    holes = {tuple(int(v) for v in cell) for cell in obj.get("holes", [])}
    cx = sum(x for x, _ in cells) / len(cells)
    cy = sum(y for _, y in cells) / len(cells)
    candidates = sorted(
        cells,
        key=lambda cell: (
            cell in holes,
            abs(cell[0] - cx) + abs(cell[1] - cy),
            cell[1],
            cell[0],
        ),
    )
    return display_click_for_engine_cell(candidates[0][0], candidates[0][1], grid_size, rotation_quarters)


def _with_display_geometry(
    objects: list[dict[str, Any]],
    grid_size: int,
    rotation_quarters: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for obj in objects:
        annotated = dict(obj)
        annotated["display_bbox_xyxy"] = _display_bbox_for_cells(
            obj["occ"],
            grid_size,
            rotation_quarters,
        )
        annotated["display_click_xy"] = frame_object_click_xy(obj, grid_size, rotation_quarters)
        annotated["display_geometry_source"] = "frame_grid_rotation_geometry"
        out.append(annotated)
    return out


def connected_components(grid: Any, values: Iterable[int]) -> list[set[tuple[int, int]]]:
    arr = np.asarray(grid)
    allowed = {int(v) for v in values}
    seen: set[tuple[int, int]] = set()
    components: list[set[tuple[int, int]]] = []
    height, width = arr.shape
    for y in range(height):
        for x in range(width):
            if (x, y) in seen or int(arr[y, x]) not in allowed:
                continue
            comp: set[tuple[int, int]] = set()
            q: deque[tuple[int, int]] = deque([(x, y)])
            seen.add((x, y))
            while q:
                cx, cy = q.popleft()
                comp.add((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if not (0 <= nx < width and 0 <= ny < height):
                        continue
                    if (nx, ny) in seen or int(arr[ny, nx]) not in allowed:
                        continue
                    seen.add((nx, ny))
                    q.append((nx, ny))
            components.append(comp)
    return components


def visible_spill_objects_from_engine_grid(grid: Any) -> list[dict[str, Any]]:
    arr = np.asarray(grid)
    objects: list[dict[str, Any]] = []

    for comp in connected_components(arr, {6, 4}):
        values = {int(arr[y, x]) for x, y in comp}
        if values & {6}:
            for sub in connected_components(_mask_grid(arr, comp, {6}), {6}):
                objects.append(_object("source", sub, arr))
            for sub in connected_components(_mask_grid(arr, comp, {4}), {4}):
                objects.append(_object("marker", sub, arr, holes=sub))
        elif values == {4} and not _touches_values(arr, comp, {8, 9}):
            objects.append(_object("marker", comp, arr, holes=comp))

    for comp in connected_components(arr, {11, 13}):
        objects.append(_object("bucket", comp, arr))

    for comp in connected_components(arr, {15}):
        objects.append(_object("diverter", comp, arr, movable=True))

    for comp in connected_components(arr, {8, 4}):
        values = {int(arr[y, x]) for x, y in comp}
        if 8 in values:
            holes = {(x, y) for x, y in comp if int(arr[y, x]) == 4}
            objects.append(_object("platform", comp, arr, holes=holes, movable=True))

    for comp in connected_components(arr, {9, 4}):
        values = {int(arr[y, x]) for x, y in comp}
        if 9 not in values:
            continue
        if _looks_like_selected_diverter(comp, {9}):
            objects.append(_object("diverter", comp, arr, movable=True))
        else:
            holes = {(x, y) for x, y in comp if int(arr[y, x]) == 4}
            objects.append(_object("platform", comp, arr, holes=holes, movable=True))
    return sorted(objects, key=lambda item: (item["y"], item["x"], item["kind"], item["width"], item["height"]))


def _mask_grid(grid: np.ndarray, comp: set[tuple[int, int]], keep: set[int]) -> np.ndarray:
    masked = np.full(grid.shape, -1, dtype=np.int16)
    for x, y in comp:
        if int(grid[y, x]) in keep:
            masked[y, x] = int(grid[y, x])
    return masked


def _touches_values(grid: np.ndarray, comp: set[tuple[int, int]], values: set[int]) -> bool:
    height, width = grid.shape
    wanted = {int(v) for v in values}
    for x, y in comp:
        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
            if 0 <= nx < width and 0 <= ny < height and int(grid[ny, nx]) in wanted:
                return True
    return False


def _looks_like_selected_diverter(comp: set[tuple[int, int]], values: set[int]) -> bool:
    if values != {9} or len(comp) != 3:
        return False
    xs = [x for x, _ in comp]
    ys = [y for _, y in comp]
    return max(xs) - min(xs) == 1 and max(ys) - min(ys) == 1


def _object(
    kind: str,
    cells: set[tuple[int, int]],
    grid: np.ndarray,
    *,
    holes: set[tuple[int, int]] | None = None,
    movable: bool = False,
) -> dict[str, Any]:
    xs = [x for x, _ in cells]
    ys = [y for _, y in cells]
    hole_cells = sorted(holes or set())
    hist = Counter(int(grid[y, x]) for x, y in cells)
    return {
        "kind": kind,
        "x": int(min(xs)),
        "y": int(min(ys)),
        "width": int(max(xs) - min(xs) + 1),
        "height": int(max(ys) - min(ys) + 1),
        "occ": [[int(x), int(y)] for x, y in sorted(cells)],
        "holes": [[int(x), int(y)] for x, y in hole_cells],
        "movable": bool(movable),
        "color_histogram": {str(k): int(v) for k, v in sorted(hist.items())},
        "source": "rendered_frame",
    }


def perceive_frame_spill_state(
    frame: Any,
    grid_size: int | None = None,
    rotation_quarters: int | None = None,
) -> dict[str, Any]:
    display = display_grid_from_frame(frame, grid_size=grid_size)
    k = infer_rotation_quarters(display) if rotation_quarters is None else int(rotation_quarters)
    engine = np.rot90(display, k=(-k) % 4).copy()
    objects = _with_display_geometry(
        visible_spill_objects_from_engine_grid(engine),
        int(engine.shape[0]),
        int(k),
    )
    return {
        "schema": FRAME_SPILL_PERCEPTION_SCHEMA,
        "status": "visible_core_recovered",
        "grid": [int(engine.shape[1]), int(engine.shape[0])],
        "rotation_quarters": int(k),
        "frame_only": True,
        "privileged_state_used": False,
        "simulation_ready": False,
        "objects": objects,
        "limitations": list(LIMITATIONS),
        "truth_boundary": "visible core spill objects only; not solve truth or complete simulator state",
    }


def _barrier_occupants_from_engine_grid(grid: np.ndarray, start_idx: int) -> list[dict[str, Any]]:
    occupants: list[dict[str, Any]] = []
    for offset, comp in enumerate(connected_components(grid, {1})):
        cells = {(int(x), int(y)) for x, y in comp}
        xs = [x for x, _ in cells]
        ys = [y for _, y in cells]
        occupants.append(
            {
                "idx": int(start_idx + offset),
                "kind": "wall",
                "name": f"frame_color1_barrier_{offset:02d}",
                "x": int(min(xs)),
                "y": int(min(ys)),
                "occ": cells,
                "holes": set(),
                "movable": False,
                "source": "rendered_frame_color1_conservative_barrier",
            }
        )
    return occupants


def frame_spill_model_state_from_frame(
    frame: Any,
    grid_size: int | None = None,
    rotation_quarters: int | None = None,
    *,
    include_color1_barriers: bool = True,
) -> dict[str, Any]:
    """Convert rendered-frame perception into the spill-model state shape.

    The returned object intentionally uses synthetic frame ids instead of engine
    sprite indices. It is suitable for comparing visible geometry and for future
    frame-derived planning work, but not for final overflow truth because wall
    hazard sprites are not disambiguated from same-color borders/padding.
    """
    packet = perceive_frame_spill_state(
        frame,
        grid_size=grid_size,
        rotation_quarters=rotation_quarters,
    )
    occupants = []
    for idx, obj in enumerate(packet["objects"]):
        occ = {tuple(int(v) for v in cell) for cell in obj["occ"]}
        holes = {tuple(int(v) for v in cell) for cell in obj["holes"]}
        occupants.append(
            {
                "idx": int(idx),
                "kind": obj["kind"],
                "name": f"frame_{obj['kind']}_{idx:02d}",
                "x": int(obj["x"]),
                "y": int(obj["y"]),
                "occ": occ,
                "holes": holes,
                "movable": bool(obj["movable"]),
                "source": "rendered_frame",
            }
        )
    if include_color1_barriers:
        engine_grid = engine_grid_from_frame(
            frame,
            grid_size=grid_size,
            rotation_quarters=packet["rotation_quarters"],
        )
        occupants.extend(_barrier_occupants_from_engine_grid(engine_grid, len(occupants)))
    return {
        "schema": FRAME_SPILL_STATE_SCHEMA,
        "source_schema": packet["schema"],
        "grid": tuple(int(v) for v in packet["grid"]),
        "rotation_k": int(packet["rotation_quarters"]),
        "occupants": occupants,
        "frame_only": True,
        "privileged_state_used": False,
        "simulation_ready": False,
        "limitations": list(packet["limitations"]),
        "truth_boundary": packet["truth_boundary"],
        "idx_source": "frame_synthetic_sorted_visible_objects",
        "barrier_mode": "conservative_color1_as_wall" if include_color1_barriers else "none",
    }


def frame_spill_dims_from_state(state: dict[str, Any]) -> dict[int, tuple[int, int]]:
    """Return occupant extents for a frame-derived spill-model state."""
    dims: dict[int, tuple[int, int]] = {}
    for obj in state["occupants"]:
        cells = list(obj["occ"])
        xs = [int(x) for x, _ in cells]
        ys = [int(y) for _, y in cells]
        dims[int(obj["idx"])] = (int(max(xs) - min(xs) + 1), int(max(ys) - min(ys) + 1))
    return dims


__all__ = [
    "CORE_VISIBLE_KINDS",
    "FRAME_SPILL_PERCEPTION_SCHEMA",
    "FRAME_SPILL_STATE_SCHEMA",
    "connected_components",
    "display_grid_from_frame",
    "display_click_for_engine_cell",
    "engine_grid_from_frame",
    "engine_cell_to_display_cell",
    "frame_spill_dims_from_state",
    "frame_spill_model_state_from_frame",
    "frame_object_click_xy",
    "infer_grid_size",
    "infer_rotation_quarters",
    "perceive_frame_spill_state",
    "visible_spill_objects_from_engine_grid",
]
