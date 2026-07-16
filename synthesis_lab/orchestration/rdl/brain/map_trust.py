"""frame → (optional sim3d scene + perception_gate) → KnownMapView.

Fail closed: trusted=False unless gate reports trusted.
Always provides a component list via connected components on non-bg colors.
Never imports dropletsim / spill_model / native_scene as solver.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from synthesis_lab.orchestration.rdl.brain.types import KnownMapView

# orchestration/sim3d (for optional gate path)
_SIM3D = Path(__file__).resolve().parents[2] / "sim3d"
_REPO = Path(__file__).resolve().parents[4]


def _dominant_background(frame: np.ndarray) -> int:
    arr = np.asarray(frame).reshape(-1)
    if arr.size == 0:
        return 0
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _local_connected_components(
    frame: np.ndarray,
    *,
    background: int | None = None,
    min_area: int = 3,
    max_area: int = 800,
    exclude_full_span: bool = True,
) -> list[dict[str, Any]]:
    """4-connected same-color components (non-background). Local fallback."""
    grid = np.asarray(frame)
    if grid.ndim > 2:
        grid = grid[..., 0]
    h, w = grid.shape[:2]
    if background is None:
        background = _dominant_background(grid)
    visited = np.zeros((h, w), dtype=bool)
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
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and not visited[ny, nx]
                        and int(grid[ny, nx]) == color
                    ):
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
            out.append(
                {
                    "component_id": f"c{cid}",
                    "label_index": cid + 1,
                    "color": color,
                    "area": area,
                    "area_pixels": area,
                    "bbox_xyxy": [x0, y0, x1, y1],
                    "center_xy": [float(np.mean(xs)), float(np.mean(ys))],
                    "n_cells": area,
                    "bbox_w": bw,
                    "bbox_h": bh,
                    "fill_ratio": float(area) / float(max(1, bw * bh)),
                    "aspect": float(bw) / float(max(1, bh)),
                    "raw_pixel_value": color,
                }
            )
            cid += 1
    out.sort(key=lambda r: -int(r["area"]))
    return out


def extract_components(frame: np.ndarray) -> tuple[list[dict[str, Any]], int, str]:
    """Prefer legal_search.algorithms.components; else local CC.

    Returns (components, background, source).
    """
    grid = np.asarray(frame)
    if grid.ndim > 2:
        grid = grid[..., 0]
    bg = _dominant_background(grid)
    try:
        if str(_REPO) not in sys.path:
            sys.path.insert(0, str(_REPO))
        from legal_search.algorithms.components import (  # noqa: WPS433
            connected_components,
            dominant_background,
        )

        bg = int(dominant_background(grid))
        comps = connected_components(grid, background=bg)
        if isinstance(comps, tuple):
            comps = comps[0]
        return list(comps), bg, "legal_search.algorithms.components"
    except Exception:
        comps = _local_connected_components(grid, background=bg)
        return comps, bg, "local_cc"


def _scene_to_summary(scene: Any) -> dict[str, Any]:
    return {
        "n_spouts": len(getattr(scene, "spouts", []) or []),
        "n_targets": len(getattr(scene, "targets", []) or []),
        "floor_rows": list(getattr(scene, "floor_rows", []) or []),
        "cell": int(getattr(scene, "cell", 0) or 0),
        "flipped": bool(getattr(scene, "flipped", False)),
        "emit_volume": float(getattr(scene, "emit_volume", 0.0) or 0.0),
    }


def _scene_physics_fields(scene: Any) -> dict[str, Any]:
    """Extract WP-A lattice fields from a sim3d Scene when available."""
    spouts = tuple((int(y), int(x)) for y, x in (getattr(scene, "spouts", []) or []))
    mouths = tuple(
        tuple((int(y), int(x)) for y, x in (getattr(t, "mouth", []) or []))
        for t in (getattr(scene, "targets", []) or [])
    )
    glyphs = []
    for t in getattr(scene, "targets", []) or []:
        glyph = getattr(t, "glyph", None)
        if glyph is not None:
            try:
                ys, xs = np.nonzero(np.asarray(glyph))
                glyphs.append(tuple((int(y), int(x)) for y, x in zip(ys, xs)))
            except Exception:
                glyphs.append(())
        else:
            glyphs.append(())
    caps = tuple(float(getattr(t, "capacity", 0.0) or 0.0) for t in (getattr(scene, "targets", []) or []))
    solid = getattr(scene, "solid", None)
    if solid is not None:
        solid = np.asarray(solid, dtype=bool).copy()
    return {
        "solid": solid,
        "spouts": spouts,
        "mouths": mouths,
        "floor_rows": tuple(int(r) for r in (getattr(scene, "floor_rows", []) or [])),
        "glyph_cells": tuple(glyphs),
        "capacities": caps,
        "emit_volume": float(getattr(scene, "emit_volume", 0.0) or 0.0),
    }


def _try_sim3d_gate(frame: np.ndarray) -> dict[str, Any]:
    """Attempt sim3d extract_scene + gate_scene. Never raises."""
    result: dict[str, Any] = {
        "trusted": False,
        "gate_failures": ["sim3d_unavailable"],
        "checks": {},
        "scene_summary": {},
        "status": "degraded",
        "error": None,
        "physics": {},
    }
    try:
        if str(_SIM3D) not in sys.path:
            sys.path.insert(0, str(_SIM3D))
        from perception_gate import gate_scene  # noqa: WPS433
        from scene import extract_scene  # noqa: WPS433

        scene = extract_scene(np.asarray(frame, dtype=np.int8))
        verdict = gate_scene(frame, scene)
        trusted = bool(verdict.get("trusted"))
        failures = list(verdict.get("gate_failures") or [])
        if not trusted and not failures:
            failures = ["gate_untrusted"]
        result.update(
            {
                "trusted": trusted,
                "gate_failures": failures if not trusted else [],
                "checks": dict(verdict.get("checks") or {}),
                "scene_summary": _scene_to_summary(scene),
                "status": "ok" if trusted else "degraded",
                "diagnostics": dict(verdict.get("diagnostics") or {}),
                "physics": _scene_physics_fields(scene),
            }
        )
    except Exception as exc:  # fail closed
        result["trusted"] = False
        result["gate_failures"] = [f"sim3d_gate_error:{type(exc).__name__}"]
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["status"] = "degraded"
    return result


def frame_to_map(frame: np.ndarray, *, try_sim3d: bool = True) -> KnownMapView:
    """Build a KnownMapView from a raw color-index frame.

    Fail closed on trust. Components always filled from non-bg CC.
    """
    grid = np.asarray(frame)
    if grid.ndim > 2:
        grid = grid[..., 0]
    comps, bg, comp_source = extract_components(grid)

    trusted = False
    gate_failures: list[str] = ["gate_not_run"]
    scene_summary: dict[str, Any] = {}
    checks: dict[str, Any] = {}
    status = "degraded"
    gate_source = "none"
    physics: dict[str, Any] = {}

    if try_sim3d:
        gate = _try_sim3d_gate(grid)
        trusted = bool(gate.get("trusted"))
        gate_failures = list(gate.get("gate_failures") or ["untrusted"])
        if trusted:
            gate_failures = []
        scene_summary = dict(gate.get("scene_summary") or {})
        checks = dict(gate.get("checks") or {})
        status = "ok" if trusted else "degraded"
        gate_source = "sim3d.perception_gate"
        physics = dict(gate.get("physics") or {})
        if gate.get("error"):
            scene_summary["gate_error"] = gate["error"]
    else:
        gate_failures = ["sim3d_skipped"]
        gate_source = "skipped"

    # Fail closed: never mark trusted without explicit gate pass.
    if not trusted:
        trusted = False
        status = "degraded"

    # solid mask: prefer scene lattice; else non-bg frame mask
    solid = physics.get("solid")
    if solid is None:
        solid = np.asarray(grid) != bg

    provenance = {
        "source": gate_source,
        "trusted": trusted,
        "component_source": comp_source,
        "n_components": len(comps),
    }

    return KnownMapView(
        frame=grid,
        solid=solid,
        solid_mask=solid,
        spouts=tuple(physics.get("spouts") or ()),
        mouths=tuple(physics.get("mouths") or ()),
        floor_rows=tuple(physics.get("floor_rows") or ()),
        glyph_cells=tuple(physics.get("glyph_cells") or ()),
        capacities=tuple(physics.get("capacities") or ()),
        emit_volume=float(physics.get("emit_volume") or 0.0),
        trusted=trusted,
        meta=dict(provenance),
        gate_failures=gate_failures,
        components=comps,
        background=bg,
        status=status,
        provenance=provenance,
        scene_summary=scene_summary,
        checks=checks,
    )


def build_trusted_map(frame: np.ndarray, **kwargs: Any) -> KnownMapView:
    """Alias for hemisphere_i compatibility."""
    return frame_to_map(frame, **kwargs)


# Aliases matching build-spec wording
map_frame = frame_to_map
frame_to_trusted_map = frame_to_map
