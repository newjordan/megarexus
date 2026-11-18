"""BodyOp — stream/sphere body plugs via sim3d hydraulic / imagine_pour.

Wraps Lane R (HydraulicSim via pour3d.imagine_pour) when importable; otherwise
returns a pure-numpy gravity stub with the same score keys (status: degraded).

BANS: dropletsim, native_scene as solver, spill_model, _invalid.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from synthesis_lab.orchestration.rdl.brain.brain_types import (
    KnownMapView,
    OperatorResult,
    TruthFlags,
)

# .../orchestration/rdl/brain/operators/body.py → parents[3] = orchestration
_SIM3D = Path(__file__).resolve().parents[3] / "sim3d"


def _ensure_sim3d_path() -> bool:
    """Add sim3d package dir (and its _paths bootstrap) if present."""
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


def _try_imagine_pour(frame: np.ndarray, lane: str, context: dict) -> dict | None:
    """Return imagine_pour dict or None on any failure."""
    if not _ensure_sim3d_path():
        return None
    try:
        from pour3d import imagine_pour  # type: ignore
    except Exception:
        return None
    try:
        kwargs: dict[str, Any] = {"lane": lane}
        if "emit_volume" in context:
            kwargs["emit_volume"] = context["emit_volume"]
        if "depth" in context:
            kwargs["depth"] = context["depth"]
        return imagine_pour(np.asarray(frame), **kwargs)
    except Exception:
        return None


def _solid_of(map_view: KnownMapView) -> np.ndarray | None:
    s = map_view.get_solid() if hasattr(map_view, "get_solid") else getattr(map_view, "solid", None)
    if s is None:
        return None
    return np.asarray(s, dtype=bool)


def _numpy_gravity_stub(map_view: KnownMapView, context: dict) -> OperatorResult:
    """Cheap column drop from spouts: count mouth vs floor first-hits."""
    solid = _solid_of(map_view)
    if solid is None:
        return OperatorResult(
            name="body",
            status="degraded",
            scores={
                "margin": 0.0,
                "winner_ok": 0.0,
                "n_mouth_hits": 0.0,
                "n_floor_hits": 0.0,
            },
            payload={"backend": "numpy_gravity_stub"},
            summary={"error": "no_solid", "backend": "numpy_gravity_stub"},
            messages=["no solid mask; empty stub"],
            truth_flags=TruthFlags(),
            hypothesis_only=True,
        )
    h, w = solid.shape
    floor_set = set(int(r) for r in (map_view.floor_rows or ()))
    if not floor_set and h > 0:
        floor_set = {h - 1}

    owner: dict[tuple[int, int], int] = {}
    for i, cells in enumerate(map_view.mouths or ()):
        for c in cells:
            owner[(int(c[0]), int(c[1]))] = i

    spouts = list(map_view.spouts or ())
    if not spouts and "spout" in context:
        spouts = [tuple(context["spout"])]
    if not spouts:
        spouts = [(0, w // 2)]

    mouth_hits = 0
    floor_hits = 0
    paths: list[dict[str, Any]] = []
    for sy, sx in spouts:
        y, x = int(sy), int(sx)
        path_len = 0
        contact = "open"
        while True:
            y += 1
            path_len += 1
            if y >= h:
                contact = "open"
                break
            if (y, x) in owner:
                contact = "mouth"
                mouth_hits += 1
                break
            if y in floor_set:
                contact = "floor"
                floor_hits += 1
                break
            if solid[y, x]:
                contact = "solid"
                break
        paths.append({
            "origin": (int(sy), int(sx)),
            "path_len": path_len,
            "contact": contact,
        })

    n = max(len(spouts), 1)
    fill_score = mouth_hits / n
    lava_score = floor_hits / n
    margin = float(fill_score - lava_score)
    return OperatorResult(
        name="body",
        status="degraded",
        scores={
            "margin": margin,
            "winner_ok": float(mouth_hits > 0 and floor_hits == 0),
            "n_mouth_hits": float(mouth_hits),
            "n_floor_hits": float(floor_hits),
            "fill_proxy": float(fill_score),
            "lava_proxy": float(lava_score),
        },
        payload={"paths": paths, "backend": "numpy_gravity_stub"},
        summary={"paths": paths, "backend": "numpy_gravity_stub"},
        messages=["sim3d imagine_pour unavailable or failed; numpy stub"],
        truth_flags=TruthFlags(),
        hypothesis_only=True,
    )


class BodyOp:
    """Body physics operator: prefer sim3d hydraulic stream pour, else stub."""

    name = "body"

    def __init__(self, lane: str = "stream") -> None:
        # "stream" = HydraulicSim (Lane R); "particle" = PourSim spheres (Lane P)
        self.lane = lane

    def run(self, map_view: Any, context: dict | None = None) -> OperatorResult:
        context = dict(context or {})
        if not isinstance(map_view, KnownMapView):
            map_view = _coerce_map_view(map_view)

        frame = map_view.frame
        if frame is None and context.get("frame") is not None:
            frame = context["frame"]

        lane = str(context.get("lane", self.lane))
        if frame is not None:
            out = _try_imagine_pour(np.asarray(frame), lane=lane, context=context)
            if out is not None:
                scores: dict[str, Any] = {
                    "margin": float(out.get("margin", 0.0)),
                    "winner_ok": float(bool(out.get("winner_ok", False))),
                    "n_frames": float(out.get("n_frames", 0)),
                }
                native = out.get("native") or {}
                if native:
                    scores["ticks"] = float(native.get("ticks", 0))
                    scores["lava"] = float(bool(native.get("lava", False)))
                    mf = native.get("mouth_fill") or []
                    scores["mouth_fill_sum"] = float(sum(float(v) for v in mf))
                summary = {
                    "backend": f"imagine_pour:{lane}",
                    "reasons": list(out.get("reasons") or []),
                    "native": {
                        k: (list(v) if isinstance(v, (list, tuple)) else v)
                        for k, v in (native or {}).items()
                        if isinstance(v, (int, float, bool, list, tuple, str, type(None)))
                    },
                }
                return OperatorResult(
                    name=self.name,
                    status="ok",
                    scores=scores,
                    payload=dict(summary),
                    summary=summary,
                    messages=["sim3d imagine_pour ok"],
                    truth_flags=TruthFlags(),
                    hypothesis_only=True,
                )

        return _numpy_gravity_stub(map_view, context)


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
