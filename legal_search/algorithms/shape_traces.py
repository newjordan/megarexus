"""Shape-trace geometry — mined from codexlab/contra_head/vision.py.

KEEP: masks, boundary, perimeter, geometry hints, thin-line/field classes.
STRIP: semantic bucket labels (floor/wall/goal names as truth).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from legal_search.algorithms.components import connected_components, dominant_background


def boundary_mask_from_cells(h: int, w: int, cells: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    for y, x in cells:
        if 0 <= y < h and 0 <= x < w:
            mask[y, x] = True
    boundary = np.zeros((h, w), dtype=np.uint8)
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys, xs):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = int(y) + dy, int(x) + dx
            if ny < 0 or nx < 0 or ny >= h or nx >= w or not mask[ny, nx]:
                boundary[y, x] = 1
                break
    return boundary


def orthogonal_perimeter(h: int, w: int, cells: list[tuple[int, int]]) -> int:
    cell_set = set(cells)
    total = 0
    for y, x in cells:
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (y + dy, x + dx) not in cell_set:
                total += 1
    return total


def geometry_hint(area_fraction: float, width: int, height: int, edge_touch_count: int) -> str:
    """Geometry class label — not a game role."""
    if area_fraction >= 0.30:
        return "large_field"
    if height == 1 or width == 1:
        return "thin_line"
    if area_fraction <= 0.0015:
        return "single_or_tiny_pixel_group"
    if edge_touch_count >= 2:
        return "edge_connected_group"
    return "local_pixel_group"


def component_cells_from_frame(
    frame: np.ndarray,
    component: dict[str, Any],
) -> list[tuple[int, int]]:
    """Recover cells inside bbox matching component color (approx re-flood)."""
    grid = np.asarray(frame)
    h, w = grid.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in component["bbox_xyxy"]]
    color = int(component["color"])
    cells: list[tuple[int, int]] = []
    for y in range(max(0, y0), min(h, y1 + 1)):
        for x in range(max(0, x0), min(w, x1 + 1)):
            if int(grid[y, x]) == color:
                cells.append((y, x))
    return cells


def build_shape_trace(frame: np.ndarray, component: dict[str, Any]) -> dict[str, Any]:
    """One shape-trace hypothesis for a component — no semantic roles."""
    grid = np.asarray(frame)
    h, w = grid.shape[:2]
    cells = component_cells_from_frame(grid, component)
    area = len(cells) or int(component.get("area") or 0)
    x0, y0, x1, y1 = [int(v) for v in component["bbox_xyxy"]]
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    edge_touches = {
        "left": x0 == 0,
        "right": x1 >= w - 1,
        "top": y0 == 0,
        "bottom": y1 >= h - 1,
    }
    edge_touch_count = int(sum(1 for v in edge_touches.values() if v))
    peri = orthogonal_perimeter(h, w, cells) if cells else int(component.get("perimeter") or 0)
    area_frac = float(area) / float(max(1, h * w))
    fill = float(area) / float(max(1, bw * bh))
    boundary = boundary_mask_from_cells(h, w, cells) if cells else None
    boundary_count = int(np.count_nonzero(boundary)) if boundary is not None else peri
    return {
        "component_id": component["component_id"],
        "color": int(component["color"]),
        "bbox_xyxy": [x0, y0, x1, y1],
        "center_xy": list(component["center_xy"]),
        "area": area,
        "area_fraction": area_frac,
        "width": bw,
        "height": bh,
        "fill_ratio": fill,
        "perimeter": peri,
        "boundary_cell_count": boundary_count,
        "edge_touches": edge_touches,
        "edge_touch_count": edge_touch_count,
        "geometry_hint": geometry_hint(area_frac, bw, bh, edge_touch_count),
        "compactness": float(area) / float(max(1, peri)),
        "admitted": False,
        "semantic_role": None,  # never assign floor/wall/goal
        "evidence_status": "shape_trace_geometry_only",
    }


def color_group_rows(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group components by raw color — not semantic roles (vision color groups)."""
    by_color: dict[int, list[dict[str, Any]]] = {}
    for c in components:
        by_color.setdefault(int(c["color"]), []).append(c)
    rows = []
    for color, members in sorted(by_color.items(), key=lambda kv: -sum(int(m.get("area") or 0) for m in kv[1])):
        area = sum(int(m.get("area") or m.get("area_pixels") or 0) for m in members)
        boxes = [m["bbox_xyxy"] for m in members]
        ub = [
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        ]
        rows.append(
            {
                "color": color,
                "n_components": len(members),
                "total_area": area,
                "component_ids": [m["component_id"] for m in members],
                "union_bbox_xyxy": ub,
                "semantic_role": None,
                "admitted": False,
            }
        )
    return rows


def mine_shape_traces(frame: np.ndarray, *, max_traces: int = 16) -> dict[str, Any]:
    comps = connected_components(frame)[:max_traces]
    traces = [build_shape_trace(frame, c) for c in comps]
    by_hint: dict[str, int] = {}
    for t in traces:
        by_hint[t["geometry_hint"]] = by_hint.get(t["geometry_hint"], 0) + 1
    groups = color_group_rows(comps)
    return {
        "traces": traces,
        "n_traces": len(traces),
        "by_geometry_hint": by_hint,
        "background": dominant_background(frame),
        "thin_line_count": by_hint.get("thin_line", 0),
        "large_field_count": by_hint.get("large_field", 0),
        "edge_connected_count": by_hint.get("edge_connected_group", 0),
        "color_groups": groups,
        "n_color_groups": len(groups),
        "semantic_labels_assigned": False,
    }
