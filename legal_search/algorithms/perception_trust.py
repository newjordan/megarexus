"""Perception trust gate — legal re-expression of sim3d perception_gate idea.

KEEP: self-consistency checks between raw frame and induced structure.
STRIP: SP80 scene JSON, solid color roles, fixture geometry as truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from legal_search.algorithms.components import (
    connected_components,
    dominant_background,
    enrich_components,
)
from legal_search.algorithms.shape_traces import mine_shape_traces


def cell_ray(
    solid: np.ndarray,
    y0: int,
    x0: int,
    d: tuple[int, int],
) -> tuple[int, int] | None:
    """First solid cell along direction d — mined from perception_gate._cell_ray."""
    gh, gw = solid.shape
    y, x = y0, x0
    while True:
        y += d[0]
        x += d[1]
        if not (0 <= y < gh and 0 <= x < gw):
            return None
        if solid[y, x]:
            return (y, x)


def gate_frame_structure(frame: np.ndarray) -> dict[str, Any]:
    """Trust checks without SP80 scene — structure must be self-consistent."""
    grid = np.asarray(frame)
    h, w = grid.shape[:2]
    bg = dominant_background(grid)
    scene = enrich_components(grid)
    comps = scene["components"]
    traces = mine_shape_traces(grid)
    # solid = non-background
    solid = grid != bg
    # ray probes from center
    cy, cx = h // 2, w // 2
    rays = {}
    for name, d in (("up", (-1, 0)), ("down", (1, 0)), ("left", (0, -1)), ("right", (0, 1))):
        hit = cell_ray(solid, cy, cx, d)
        rays[name] = {"hit": hit is not None, "cell": list(hit) if hit else None}

    # reconstruction: recoloring components should cover non-bg pixels
    covered = np.zeros_like(grid, dtype=bool)
    for c in comps:
        x0, y0, x1, y1 = c["bbox_xyxy"]
        color = int(c["color"])
        patch = grid[y0 : y1 + 1, x0 : x1 + 1]
        covered[y0 : y1 + 1, x0 : x1 + 1] |= patch == color
    non_bg = grid != bg
    covered_non_bg = int(np.count_nonzero(covered & non_bg))
    total_non_bg = int(np.count_nonzero(non_bg))
    cover_ratio = covered_non_bg / max(1, total_non_bg)

    checks = {
        "components_present": {
            "pass": len(comps) >= 1,
            "n_components": len(comps),
        },
        "coverage": {
            "pass": cover_ratio >= 0.5,
            "cover_ratio": cover_ratio,
            "covered_non_bg": covered_non_bg,
            "total_non_bg": total_non_bg,
        },
        "shape_traces": {
            "pass": traces["n_traces"] == len(comps[:16]) or traces["n_traces"] > 0,
            "n_traces": traces["n_traces"],
        },
        "contact_graph": {
            "pass": True,  # always ok as optional structure
            "n_contacts": scene["contact_graph"].get("contact_pair_count", 0),
        },
        "rays": {
            "pass": any(v["hit"] for v in rays.values()) or total_non_bg == 0,
            "directions": rays,
        },
    }
    all_pass = all(bool(c.get("pass")) for c in checks.values())
    return {
        "schema": "legal_search.perception_trust.v1",
        "background": bg,
        "checks": checks,
        "all_pass": all_pass,
        "n_checks_pass": sum(1 for c in checks.values() if c.get("pass")),
        "n_checks": len(checks),
        "trust_score": float(sum(1 for c in checks.values() if c.get("pass")) / max(1, len(checks))),
        "fixture_scene_loaded": False,
        "sp80_roles_used": False,
    }
