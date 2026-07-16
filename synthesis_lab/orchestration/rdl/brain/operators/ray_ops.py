"""RayOp — 8-dir rays + polarity (mouth+/floor−).

Prefers sim3d imagination_kernel.cast_rays when KnownMap/SimState can be built
simply; otherwise pure-numpy solid-mask rays with the same polarity convention.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from synthesis_lab.orchestration.rdl.brain.types import (
    KnownMapView,
    OperatorResult,
    TruthFlags,
)

# .../orchestration/rdl/brain/operators/ray_ops.py → parents[3] = orchestration
_SIM3D = Path(__file__).resolve().parents[3] / "sim3d"

EIGHT_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))

_POLARITY = {
    "mouth": ("beneficial", 1.0),
    "floor": ("adversarial", -1.0),
    "glyph": ("structural", -0.25),
    "solid": ("structural", -0.25),
    "open": ("neutral", 0.0),
}


def _ensure_sim3d_path() -> bool:
    if not _SIM3D.is_dir():
        return False
    p = str(_SIM3D)
    if p not in sys.path:
        sys.path.insert(0, p)
    try:
        import _paths  # noqa: F401
        return True
    except Exception:
        return False


def _solid_of(map_view: KnownMapView) -> np.ndarray | None:
    s = map_view.get_solid() if hasattr(map_view, "get_solid") else getattr(map_view, "solid", None)
    if s is None:
        return None
    return np.asarray(s, dtype=bool)


def _try_kernel_cast_rays(map_view: KnownMapView) -> list[dict] | None:
    """Build a minimal KnownMap + SimState and call cast_rays if importable."""
    solid = _solid_of(map_view)
    if solid is None:
        return None
    if not _ensure_sim3d_path():
        return None
    try:
        from imagination_kernel import KnownMap, SimState, cast_rays  # type: ignore
    except Exception:
        return None
    try:
        mouths = tuple(
            tuple((int(c[0]), int(c[1])) for c in cells)
            for cells in (map_view.mouths or ())
        )
        glyphs = tuple(
            tuple((int(c[0]), int(c[1])) for c in cells)
            for cells in (map_view.glyph_cells or ())
        )
        if not glyphs and mouths:
            glyphs = mouths
        caps = tuple(float(c) for c in (map_view.capacities or ()))
        if len(caps) < len(mouths):
            caps = caps + tuple(1.0 for _ in range(len(mouths) - len(caps)))
        spouts = tuple((int(y), int(x)) for y, x in (map_view.spouts or ()))
        floor_rows = tuple(int(r) for r in (map_view.floor_rows or ()))
        kmap = KnownMap(
            solid=solid,
            floor_rows=floor_rows,
            spouts=spouts,
            mouths=mouths,
            glyph_cells=glyphs,
            capacities=caps,
            emit_volume=float(map_view.emit_volume or 0.0),
            provenance={"source": "KnownMapView", "trusted": bool(map_view.trusted)},
        )
        state = SimState(kmap=kmap)
        hits = cast_rays(state, dirs=EIGHT_DIRS)
        clean = []
        for h in hits:
            clean.append({
                "anchor": h.get("anchor"),
                "origin": tuple(int(v) for v in h.get("origin", (0, 0))),
                "dir": tuple(int(v) for v in h.get("dir", (0, 0))),
                "path_len": int(h.get("path_len", 0)),
                "contact": h.get("contact"),
                "object_id": str(h.get("object_id", "")),
                "network": h.get("network"),
                "signed_potential_y": float(h.get("signed_potential_y", 0.0)),
            })
        return clean
    except Exception:
        return None


def _numpy_cast_rays(map_view: KnownMapView) -> list[dict]:
    """Local 8-dir rays on solid mask; polarity mouth+/floor− generic."""
    solid = _solid_of(map_view)
    assert solid is not None
    h, w = solid.shape
    floor_set = set(int(r) for r in (map_view.floor_rows or ()))
    owner: dict[tuple[int, int], int] = {}
    for i, cells in enumerate(map_view.mouths or ()):
        for c in cells:
            owner[(int(c[0]), int(c[1]))] = i
    glyph_owner: dict[tuple[int, int], int] = {}
    for i, cells in enumerate(map_view.glyph_cells or ()):
        for c in cells:
            glyph_owner[(int(c[0]), int(c[1]))] = i

    anchors: list[tuple[str, tuple[int, int]]] = [
        ("spout", (int(y), int(x))) for y, x in (map_view.spouts or ())
    ]
    for cells in (map_view.mouths or ()):
        if cells:
            top = min(cells, key=lambda c: int(c[0]))
            anchors.append(("mouth_top", (int(top[0]), int(top[1]))))
    if not anchors:
        anchors = [("spout", (0, w // 2))]

    hits: list[dict] = []
    for kind, (ay, ax) in anchors:
        for dy, dx in EIGHT_DIRS:
            y, x, path = ay, ax, 0
            contact_kind = "open"
            ident: int | None = None
            while True:
                y += dy
                x += dx
                path += 1
                if not (0 <= y < h and 0 <= x < w):
                    contact_kind, ident = "open", None
                    break
                if (y, x) in owner:
                    contact_kind, ident = "mouth", owner[(y, x)]
                    break
                if y in floor_set:
                    contact_kind, ident = "floor", None
                    break
                if solid[y, x]:
                    if (y, x) in glyph_owner:
                        contact_kind, ident = "glyph", glyph_owner[(y, x)]
                    else:
                        contact_kind, ident = "solid", None
                    break
            network, signed_y = _POLARITY[contact_kind]
            hits.append({
                "anchor": kind,
                "origin": (ay, ax),
                "dir": (dy, dx),
                "path_len": path,
                "contact": contact_kind,
                "object_id": f"{contact_kind}:{ident}:{y}:{x}",
                "network": network,
                "signed_potential_y": float(signed_y),
            })
    return hits


def _summarize(hits: list[dict]) -> dict[str, Any]:
    if not hits:
        return {
            "n_hits": 0.0,
            "mean_signed_potential": 0.0,
            "n_beneficial": 0.0,
            "n_adversarial": 0.0,
            "n_structural": 0.0,
            "n_neutral": 0.0,
            "mean_path_len": 0.0,
        }
    signed = [float(h["signed_potential_y"]) for h in hits]
    nets = [h.get("network") for h in hits]
    return {
        "n_hits": float(len(hits)),
        "mean_signed_potential": float(sum(signed) / len(signed)),
        "n_beneficial": float(sum(1 for n in nets if n == "beneficial")),
        "n_adversarial": float(sum(1 for n in nets if n == "adversarial")),
        "n_structural": float(sum(1 for n in nets if n == "structural")),
        "n_neutral": float(sum(1 for n in nets if n == "neutral")),
        "mean_path_len": float(sum(int(h["path_len"]) for h in hits) / len(hits)),
    }


class RayOp:
    """8-direction ray bundle with network polarity."""

    name = "ray"

    def run(self, map_view: Any, context: dict | None = None) -> OperatorResult:
        context = dict(context or {})
        if not isinstance(map_view, KnownMapView):
            map_view = _coerce_map_view(map_view)

        if _solid_of(map_view) is None:
            return OperatorResult(
                name=self.name,
                status="failed",
                scores=_summarize([]),
                payload={"error": "no_solid"},
                summary={"error": "no_solid"},
                messages=["no solid mask"],
                truth_flags=TruthFlags(),
                hypothesis_only=True,
            )

        hits = _try_kernel_cast_rays(map_view)
        if hits is not None:
            summary = {
                "hits": hits[:64],
                "n_hits_full": len(hits),
                "backend": "imagination_kernel.cast_rays",
            }
            return OperatorResult(
                name=self.name,
                status="ok",
                scores=_summarize(hits),
                payload=dict(summary),
                summary=summary,
                messages=["sim3d cast_rays ok"],
                truth_flags=TruthFlags(),
                hypothesis_only=True,
            )

        hits = _numpy_cast_rays(map_view)
        summary = {
            "hits": hits[:64],
            "n_hits_full": len(hits),
            "backend": "numpy_solid_mask_rays",
        }
        return OperatorResult(
            name=self.name,
            status="degraded",
            scores=_summarize(hits),
            payload=dict(summary),
            summary=summary,
            messages=["kernel cast_rays unavailable; numpy 8-dir fallback"],
            truth_flags=TruthFlags(),
            hypothesis_only=True,
        )


def _coerce_map_view(obj: Any) -> KnownMapView:
    if obj is None:
        return KnownMapView()
    if isinstance(obj, dict):
        keys = KnownMapView.__dataclass_fields__
        return KnownMapView(**{k: v for k, v in obj.items() if k in keys})
    return KnownMapView(
        solid=getattr(obj, "solid", None),
        solid_mask=getattr(obj, "solid_mask", None),
        spouts=tuple(getattr(obj, "spouts", ()) or ()),
        mouths=tuple(getattr(obj, "mouths", ()) or ()),
        floor_rows=tuple(getattr(obj, "floor_rows", ()) or ()),
        glyph_cells=tuple(getattr(obj, "glyph_cells", ()) or ()),
        capacities=tuple(getattr(obj, "capacities", ()) or ()),
        emit_volume=float(getattr(obj, "emit_volume", 0.0) or 0.0),
        frame=getattr(obj, "frame", None),
        trusted=bool(getattr(obj, "trusted", False)),
        components=list(getattr(obj, "components", []) or []),
        background=int(getattr(obj, "background", 0) or 0),
        gate_failures=list(getattr(obj, "gate_failures", []) or []),
        status=str(getattr(obj, "status", "degraded") or "degraded"),
        meta=dict(getattr(obj, "meta", {}) or {}),
    )
