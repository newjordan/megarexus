"""Spatial relation queues — mined from codexlab/contra_head/geoband.py.

KEEP: bearings, distances, directional gaps, lateral offsets, pair queues.
STRIP: forced route / admitted actor-target solve path.
"""

from __future__ import annotations

import math
from typing import Any


def bearing_degrees(from_xy: tuple[float, float] | list[float], to_xy: tuple[float, float] | list[float]) -> float:
    dx = float(to_xy[0]) - float(from_xy[0])
    dy = float(to_xy[1]) - float(from_xy[1])
    return float(math.degrees(math.atan2(dy, dx)) % 360.0)


def euclidean_distance(a: tuple[float, float] | list[float], b: tuple[float, float] | list[float]) -> float:
    return float(math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1])))


def manhattan_distance(a: tuple[float, float] | list[float], b: tuple[float, float] | list[float]) -> float:
    return abs(float(b[0]) - float(a[0])) + abs(float(b[1]) - float(a[1]))


def directional_gap(origin: tuple[float, float] | list[float], direction: str, box: list[int]) -> float:
    """Mined from GeoBand `directional_gap`."""
    x, y = float(origin[0]), float(origin[1])
    x0, y0, x1, y1 = [float(v) for v in box]
    if direction == "up":
        return max(0.0, y - y1)
    if direction == "down":
        return max(0.0, y0 - y)
    if direction == "left":
        return max(0.0, x - x1)
    if direction == "right":
        return max(0.0, x0 - x)
    return 64.0


def lateral_offset(origin: tuple[float, float] | list[float], direction: str, box: list[int]) -> float:
    """Mined from GeoBand `lateral_offset`."""
    x, y = float(origin[0]), float(origin[1])
    x0, y0, x1, y1 = [float(v) for v in box]
    if direction in {"up", "down"}:
        if x0 <= x <= x1:
            return 0.0
        return min(abs(x - x0), abs(x - x1))
    if direction in {"left", "right"}:
        if y0 <= y <= y1:
            return 0.0
        return min(abs(y - y0), abs(y - y1))
    return 64.0


def primary_direction(from_xy: list[float], to_xy: list[float]) -> str:
    dx = float(to_xy[0]) - float(from_xy[0])
    dy = float(to_xy[1]) - float(from_xy[1])
    if abs(dx) >= abs(dy):
        return "right" if dx >= 0 else "left"
    return "down" if dy >= 0 else "up"


def relationship_edge(
    a: dict[str, Any],
    b: dict[str, Any],
    *,
    edge_kind: str = "spatial_pair",
) -> dict[str, Any]:
    """Mined from GeoBand `relationship_edge` — hypothesis only."""
    ca = a["center_xy"]
    cb = b["center_xy"]
    direction = primary_direction(ca, cb)
    return {
        "edge_kind": edge_kind,
        "from_id": a["component_id"],
        "to_id": b["component_id"],
        "from_color": int(a.get("color") or 0),
        "to_color": int(b.get("color") or 0),
        "bearing_degrees": bearing_degrees(ca, cb),
        "distance_euclid": euclidean_distance(ca, cb),
        "distance_manhattan": manhattan_distance(ca, cb),
        "direction": direction,
        "gap": directional_gap(ca, direction, b["bbox_xyxy"]),
        "lateral": lateral_offset(ca, direction, b["bbox_xyxy"]),
        "admitted": False,
        "evidence_status": "spatial_relation_hypothesis",
    }


def mine_geo_relations(
    components: list[dict[str, Any]],
    *,
    max_pairs: int = 16,
) -> dict[str, Any]:
    """Budgeted spatial pair queue — legal GeoBand mine."""
    comps = components[:12]
    edges: list[dict[str, Any]] = []
    for i, a in enumerate(comps):
        for b in comps[i + 1 :]:
            edges.append(relationship_edge(a, b))
    edges.sort(key=lambda e: (e["distance_manhattan"], e["lateral"]))
    edges = edges[:max_pairs]
    # nearest-neighbor per component
    nearest: list[dict[str, Any]] = []
    for a in comps:
        cands = [e for e in edges if e["from_id"] == a["component_id"] or e["to_id"] == a["component_id"]]
        if not cands:
            # compute directly if truncated
            others = [relationship_edge(a, b) for b in comps if b["component_id"] != a["component_id"]]
            others.sort(key=lambda e: e["distance_manhattan"])
            cands = others[:1]
        if cands:
            best = min(cands, key=lambda e: e["distance_manhattan"])
            nearest.append(
                {
                    "component_id": a["component_id"],
                    "nearest_id": best["to_id"] if best["from_id"] == a["component_id"] else best["from_id"],
                    "manhattan": best["distance_manhattan"],
                    "bearing": best["bearing_degrees"],
                }
            )
    mean_dist = float(sum(e["distance_manhattan"] for e in edges) / max(1, len(edges)))
    return {
        "pairs": edges,
        "nearest": nearest,
        "n_pairs": len(edges),
        "mean_manhattan": mean_dist,
        "min_manhattan": float(edges[0]["distance_manhattan"]) if edges else 0.0,
        "n_components": len(comps),
    }
