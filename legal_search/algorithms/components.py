"""Generic connected components and frame diffs — mined from CONTRA vision.

Also mines: contact graph, component geometry features, change-region bboxes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def frame_changed(a: np.ndarray, b: np.ndarray) -> bool:
    return not np.array_equal(np.asarray(a), np.asarray(b))


def frame_delta_count(a: np.ndarray, b: np.ndarray) -> int:
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.shape != bb.shape:
        return int(aa.size)
    return int(np.count_nonzero(aa != bb))


def dominant_background(frame: np.ndarray) -> int:
    arr = np.asarray(frame).reshape(-1)
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def connected_components(
    frame: np.ndarray,
    *,
    background: int | None = None,
    min_area: int = 3,
    max_area: int = 800,
    exclude_full_span: bool = True,
    with_label_map: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], np.ndarray]:
    """4-connected same-color components. No game roles assigned.

    If with_label_map=True, also returns an int map (0=bg/ignore, else label_index+1).
    """
    grid = np.asarray(frame)
    h, w = grid.shape[:2]
    if background is None:
        background = dominant_background(grid)
    visited = np.zeros((h, w), dtype=bool)
    label_map = np.zeros((h, w), dtype=np.int32)
    out: list[dict[str, Any]] = []
    cid = 0
    for y in range(h):
        for x in range(w):
            if visited[y, x] or int(grid[y, x]) == background:
                continue
            color = int(grid[y, x])
            stack = [(y, x)]
            visited[y, x] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and int(grid[ny, nx]) == color:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            area = len(cells)
            if area < min_area or area > max_area:
                continue
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            y0, y1 = min(ys), max(ys)
            x0, x1 = min(xs), max(xs)
            if exclude_full_span and ((x1 - x0) >= w - 4 or (y1 - y0) >= h - 4):
                continue
            bw = x1 - x0 + 1
            bh = y1 - y0 + 1
            fill = float(area) / float(max(1, bw * bh))
            label_index = cid + 1
            for cy, cx in cells:
                label_map[cy, cx] = label_index
            # perimeter approx: cells with orthogonal neighbor outside component
            cell_set = set(cells)
            peri = 0
            for cy, cx in cells:
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if (ny, nx) not in cell_set:
                        peri += 1
            out.append(
                {
                    "component_id": f"c{cid}",
                    "label_index": label_index,
                    "color": color,
                    "area": area,
                    "area_pixels": area,
                    "bbox_xyxy": [x0, y0, x1, y1],
                    "center_xy": [float(np.mean(xs)), float(np.mean(ys))],
                    "n_cells": area,
                    "bbox_w": bw,
                    "bbox_h": bh,
                    "fill_ratio": fill,
                    "perimeter": peri,
                    "aspect": float(bw) / float(max(1, bh)),
                    "raw_pixel_value": color,
                }
            )
            cid += 1
    out.sort(key=lambda r: -int(r["area"]))
    if with_label_map:
        return out, label_map
    return out


def edge_cells_for_bbox(bbox_xyxy: list[int], side: str) -> list[list[int]]:
    """Port-like edge cells of a bbox — algorithm mined from PortBand ports."""
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    if side == "top":
        return [[x, y0] for x in range(x0, x1 + 1)]
    if side == "bottom":
        return [[x, y1] for x in range(x0, x1 + 1)]
    if side == "left":
        return [[x0, y] for y in range(y0, y1 + 1)]
    if side == "right":
        return [[x1, y] for y in range(y0, y1 + 1)]
    raise ValueError(f"unknown side {side}")


def contact_graph(components: list[dict[str, Any]], label_map: np.ndarray) -> dict[str, Any]:
    """Orthogonal contact graph — mined from vision.contact_graph."""
    height, width = label_map.shape[:2]
    id_by_label = {
        int(c["label_index"]): str(c["component_id"])
        for c in components
        if "label_index" in c
    }
    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    neighbor_ids: dict[int, set[int]] = defaultdict(set)
    neighbor_contact_pixels: dict[int, int] = defaultdict(int)

    for y in range(height):
        for x in range(width):
            here = int(label_map[y, x])
            if here == 0:
                continue
            for dy, dx in ((1, 0), (0, 1)):
                ny, nx = y + dy, x + dx
                if ny >= height or nx >= width:
                    continue
                there = int(label_map[ny, nx])
                if there == 0 or there == here:
                    continue
                pair = (min(here, there), max(here, there))
                pair_counts[pair] += 1
                neighbor_ids[here].add(there)
                neighbor_ids[there].add(here)
                neighbor_contact_pixels[here] += 1
                neighbor_contact_pixels[there] += 1

    contacts = []
    for (a, b), count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0])):
        if a not in id_by_label or b not in id_by_label:
            continue
        contacts.append(
            {
                "component_a": id_by_label[a],
                "component_b": id_by_label[b],
                "contact_edge_count": int(count),
            }
        )
    return {
        "contact_pair_count": len(contacts),
        "contacts": contacts[:32],
        "neighbor_ids_by_label": {k: list(v) for k, v in neighbor_ids.items()},
        "neighbor_contact_pixels_by_label": dict(neighbor_contact_pixels),
    }


def component_features(
    component: dict[str, Any],
    *,
    frame_shape: tuple[int, int],
    graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mined from vision.component_features — geometry ratios only, no roles."""
    height, width = frame_shape
    x0, y0, x1, y1 = [int(v) for v in component["bbox_xyxy"]]
    area = int(component.get("area") or component.get("area_pixels") or 0)
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    label_index = int(component.get("label_index") or 0)
    neighbor_ids: list = []
    neighbor_contact_pixels = 0
    if graph:
        neighbor_ids = list(graph.get("neighbor_ids_by_label", {}).get(label_index, []) or [])
        neighbor_contact_pixels = int(
            graph.get("neighbor_contact_pixels_by_label", {}).get(label_index, 0) or 0
        )
    left_edge = 1.0 if x0 == 0 else 0.0
    right_edge = 1.0 if x1 == width - 1 else 0.0
    top_edge = 1.0 if y0 == 0 else 0.0
    bottom_edge = 1.0 if y1 == height - 1 else 0.0
    edge_contact_count = int(left_edge + right_edge + top_edge + bottom_edge)
    return {
        "component_id": component["component_id"],
        "color": int(component.get("color") or 0),
        "area_ratio": float(area) / float(max(1, height * width)),
        "bbox_width_ratio": float(bw) / float(max(1, width)),
        "bbox_height_ratio": float(bh) / float(max(1, height)),
        "bbox_fill_ratio": float(area) / float(max(1, bw * bh)),
        "center_x_ratio": float(component["center_xy"][0]) / float(max(1, width - 1)),
        "center_y_ratio": float(component["center_xy"][1]) / float(max(1, height - 1)),
        "bottom_contact": bottom_edge,
        "top_contact": top_edge,
        "left_contact": left_edge,
        "right_contact": right_edge,
        "border_contact": max(left_edge, right_edge, top_edge, bottom_edge),
        "edge_contact_count": edge_contact_count,
        "neighbor_component_count": len(neighbor_ids),
        "neighbor_contact_pixels": neighbor_contact_pixels,
        "neighbor_contact_ratio": float(neighbor_contact_pixels) / float(max(1, area)),
        "perimeter": int(component.get("perimeter") or 0),
        "aspect": float(component.get("aspect") or (bw / max(1, bh))),
    }


def enrich_components(frame: np.ndarray) -> dict[str, Any]:
    """Components + label map + contact graph + per-component features."""
    comps, label_map = connected_components(frame, with_label_map=True)  # type: ignore[misc]
    comps = list(comps)
    graph = contact_graph(comps, label_map)
    h, w = np.asarray(frame).shape[:2]
    feats = [component_features(c, frame_shape=(h, w), graph=graph) for c in comps[:20]]
    return {
        "components": comps,
        "label_map_nonzero": int(np.count_nonzero(label_map)),
        "contact_graph": {
            "contact_pair_count": graph["contact_pair_count"],
            "contacts": graph["contacts"],
        },
        "component_features": feats,
        "n_components": len(comps),
        "background": dominant_background(frame),
    }


def change_regions(before: np.ndarray, after: np.ndarray, *, max_regions: int = 8) -> dict[str, Any]:
    """Changed-pixel connected regions — mined idea from DeltaBand surface events."""
    aa = np.asarray(before)
    bb = np.asarray(after)
    if aa.shape != bb.shape:
        return {"regions": [], "n_changed": int(aa.size), "changed_fraction": 1.0}
    mask = aa != bb
    n_changed = int(np.count_nonzero(mask))
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    regions: list[dict[str, Any]] = []
    rid = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            ys = [c[0] for c in cells]
            xs = [c[1] for c in cells]
            regions.append(
                {
                    "region_id": f"d{rid}",
                    "area": len(cells),
                    "bbox_xyxy": [min(xs), min(ys), max(xs), max(ys)],
                    "center_xy": [float(np.mean(xs)), float(np.mean(ys))],
                    "colors_before": sorted({int(aa[cy, cx]) for cy, cx in cells[:64]}),
                    "colors_after": sorted({int(bb[cy, cx]) for cy, cx in cells[:64]}),
                }
            )
            rid += 1
    regions.sort(key=lambda r: -r["area"])
    return {
        "regions": regions[:max_regions],
        "n_regions": len(regions),
        "n_changed": n_changed,
        "changed_fraction": float(n_changed) / float(max(1, h * w)),
    }
