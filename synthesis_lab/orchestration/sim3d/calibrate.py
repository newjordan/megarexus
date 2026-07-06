"""Calibration lane (SIM3D stage S1) — fit SimParams/HydroParams on LEGAL data only.

CORPUS = fixtures/calib_l0_pours.npz (55 real L0 pours incl. 5 wins and the
filled-but-lost overshoot pours 5/11/17...) + sp80_variants harvests you add with
--harvest. The five t0 canonical arrangements are the TEST SET — calibrating on
them is answer-injection-adjacent and voids the decisive test. Don't.

    python3 calibrate.py --measure            # emission volume/rate from real pours
    python3 calibrate.py --fit --lane stream  # coarse grid search -> params_stream.json
    python3 calibrate.py --report --lane both # per-pour sim-vs-real label table
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, PKG

from pertarget_labels import pertarget_labels  # noqa: E402
from scene import extract_scene, PAINT  # noqa: E402
from engine import SimParams  # noqa: E402
from raycast import HydroParams  # noqa: E402
from pour3d import imagine_pour  # noqa: E402


def _pours():
    z = np.load(FIXTURES / "calib_l0_pours.npz")
    lens = z["lens"]; starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    for i in range(len(lens)):
        s, l = int(starts[i]), int(lens[i])
        yield i, z["pre"][i], z["frames_flat"][s:s + l], bool(z["wins"][i])


def measure():
    """Emission volume (macro cells) + rate from the real paint(6) timelines."""
    vols, rates = [], []
    for i, pre, frames, win in _pours():
        n6 = np.array([int((f == PAINT).sum()) for f in frames])
        base = n6[0]
        peak = int(n6.max())
        vols.append((peak - base) / 16.0)                    # px -> macro cells
        d = np.diff(n6)
        grow = d[d > 0]
        if len(grow):
            rates.append(float(np.median(grow)) / 16.0)
    print(f"emission volume: median {np.median(vols):.1f} macro cells "
          f"(min {min(vols):.1f} max {max(vols):.1f})")
    print(f"emission rate:   median {np.median(rates):.2f} cells/tick")
    return float(np.median(vols))


def _loss_one(pre, frames, lane, params, emit_volume):
    """Per-target label distance between sim and the real animation."""
    real = pertarget_labels(pre, frames)
    sim = imagine_pour(pre, lane=lane, params=params, emit_volume=emit_volume)["labels"]
    if real["n_targets"] != sim["n_targets"]:
        return 2.0 * max(real["n_targets"], 1)
    err = 0.0
    for rt, st in zip(real["targets"], sim["targets"]):
        err += abs(rt["fill"] - st["fill"]) + abs(rt["lavan"] - st["lavan"])
    return err / max(real["n_targets"], 1)


def fit(lane, emit_volume, max_pours=30):
    """Coarse grid search over the few params that matter. Extend the grids as the
    fidelity report demands — this is deliberately a skeleton, not a final tuner."""
    if lane == "stream":
        grid = {
            "drain_per_tick": [0.5, 1.0, 2.0],
            "spread_per_tick": [1, 2],
            "cell_cap": [1.0],
        }
        base = HydroParams
    else:
        grid = {
            "radius": [0.30, 0.35, 0.45],
            "per_cell": [2.0, 3.0],
            "friction": [0.4, 0.55, 0.7],
        }
        base = SimParams
    keys = list(grid)
    best, best_loss = None, np.inf
    corpus = list(_pours())[:max_pours]
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = base(**dict(zip(keys, combo)))
        losses = [_loss_one(pre, frames, lane, params, emit_volume)
                  for _, pre, frames, _ in corpus]
        loss = float(np.mean(losses))
        print(f"  {dict(zip(keys, combo))} -> loss {loss:.4f}")
        if loss < best_loss:
            best, best_loss = params, loss
    out = PKG / f"params_{lane}.json"
    out.write_text(json.dumps(best.to_json(), indent=1))
    print(f"best {lane} loss {best_loss:.4f} -> {out}")
    return best, best_loss


def report(lanes, emit_volume):
    rows = []
    for i, pre, frames, win in _pours():
        real = pertarget_labels(pre, frames)
        rm = min((t["fill"] - t["lavan"] for t in real["targets"]), default=0.0)
        line = f"pour {i:2d} win={int(win)} real_margin={rm:+.2f}"
        for lane in lanes:
            sim = imagine_pour(pre, lane=lane, emit_volume=emit_volume)
            line += f" | {lane}: margin={sim['margin']:+.2f} ok={int(sim['winner_ok'])}"
            rows.append((i, win, lane, rm, sim["margin"]))
        print(line)
    for lane in lanes:
        sub = [(rm, sm, w) for i, w, l, rm, sm in rows if l == lane]
        agree = np.mean([(sm > 0) == w for rm, sm, w in sub])
        print(f"{lane}: sim-margin-positive agrees with env win on "
              f"{agree * 100:.0f}% of {len(sub)} pours")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--lane", default="stream", choices=["stream", "particle", "both"])
    ap.add_argument("--emit-volume", type=float, default=None)
    args = ap.parse_args()
    vol = args.emit_volume
    if args.measure or vol is None:
        vol = measure()
    lanes = ["stream", "particle"] if args.lane == "both" else [args.lane]
    if args.fit:
        for lane in lanes:
            fit(lane, vol)
    if args.report:
        report(lanes, vol)
