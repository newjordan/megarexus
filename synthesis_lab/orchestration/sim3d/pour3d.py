"""Public API: imagine one pour with the 3D simulator (either lane).

    from sim3d.pour3d import imagine_pour
    out = imagine_pour(pre_frame, lane="stream")     # Lane R (fast ranker)
    out = imagine_pour(pre_frame, lane="particle")   # Lane P (sphere physics)
    out["margin"]      # min_t(fill_t - lavan_t) — the §7 decisive score
    out["winner_ok"]   # per-target winner predicate on the SIMULATED rollout

Drop-in shape for imagine_search_gate-style loops: rank arrangements by margin,
pour the top-K for real, env grades. The sim NEVER decides a headline (C1)."""
from __future__ import annotations

import json

import _paths  # noqa: F401
from _paths import PKG

from scene import extract_scene
from engine import PourSim, SimParams
from raycast import HydraulicSim, HydroParams
from metrics import score_rollout


def load_params(lane):
    """Calibrated params if calibrate.py has written them, else defaults."""
    f = PKG / f"params_{lane}.json"
    if f.exists():
        d = json.loads(f.read_text())
        return SimParams(**d) if lane == "particle" else HydroParams(**d)
    return SimParams() if lane == "particle" else HydroParams()


def imagine_pour(frame, lane="stream", params=None, depth=1, emit_volume=None):
    scene = extract_scene(frame, depth=depth, emit_volume=emit_volume)
    params = params or load_params(lane)
    if lane == "particle":
        result = PourSim(scene, params).run()
        per_cell = params.per_cell
    elif lane == "stream":
        result = HydraulicSim(scene, params).run()
        per_cell = params.cell_cap
    else:
        raise ValueError(f"unknown lane {lane!r}")
    out = score_rollout(scene, result, per_cell=per_cell)
    out["lane"] = lane
    out["scene"] = scene
    out["result"] = result
    return out
