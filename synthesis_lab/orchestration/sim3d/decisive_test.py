"""THE decisive test (HANDOFF_SIMULATOR §7) — run this before ANY tuning/training.

Rolls the verified L1 winner + v52's 4 false positives (fixtures/t0_arrangements.npz,
all EVAL-ONLY) through the simulator, reads per-target (fill, lavan) from the
SIMULATED rollout via pertarget_labels, and checks:

  PASS  =  winner margin  min_t(fill_t - lavan_t)  is POSITIVE
           AND strictly ABOVE every false positive's margin.

Real labels (t0_pertarget_20260705.json): winner ≈ +1.0, every FP ≈ 0 or negative
(lava-saturated). A simulator that reproduces that ordering has beaten every neural
imaginer built so far (v38-v56). Minutes to run; no training; no env pours.

    python3 decisive_test.py --lane stream
    python3 decisive_test.py --lane particle
    python3 decisive_test.py --lane both        # default

HONESTY: this is a fidelity DIAGNOSTIC on recorded arrangements + recorded labels.
It pours nothing for real and claims no headline (C1). Never calibrate on these five.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, LAB

from pour3d import imagine_pour


def real_margins(t0):
    out = {}
    rows = [("winner", t0["winner"])] + [(f"fp{i}", fp) for i, fp in
                                         enumerate(t0["false_positives"])]
    for name, rec in rows:
        out[name] = min(t["fill"] - t["lavan"] for t in rec["targets"])
    return out


def run_lane(lane, pres, names, reals):
    print(f"\n=== lane: {lane} ===")
    margins = {}
    for name, pre in zip(names, pres):
        out = imagine_pour(pre, lane=lane)
        margins[name] = out["margin"]
        t = out["labels"]["targets"]
        per_t = " ".join(f"U{i}(f={x['fill']:.2f},l={x['lavan']:.2f})"
                         for i, x in enumerate(t))
        print(f"  {name:7s} sim_margin={out['margin']:+.3f}  real={reals[name]:+.3f}  "
              f"native_fill={['%.2f' % v for v in out['native']['mouth_fill']]} "
              f"native_lava={out['native']['lava']}  | {per_t}")
    w = margins["winner"]
    fps = [v for k, v in margins.items() if k != "winner"]
    passed = w > 0 and all(w > v for v in fps)
    print(f"  winner_margin={w:+.3f}  max_fp_margin={max(fps):+.3f}  "
          f"-> {'PASS' if passed else 'FAIL'}")
    return {"lane": lane, "margins": margins, "pass": bool(passed)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", default="both", choices=["stream", "particle", "both"])
    ap.add_argument("--out", default=None, help="write JSON result here")
    args = ap.parse_args()

    z = np.load(FIXTURES / "t0_arrangements.npz")
    pres, names = z["pre"], [str(n) for n in z["names"]]
    t0 = json.loads((LAB / "experiments" / "megarexus_cnn" / "runs" /
                     "t0_pertarget_20260705.json").read_text())
    reals = real_margins(t0)
    print("real margins (ground truth, env-graded):",
          {k: round(v, 3) for k, v in reals.items()})

    lanes = ["stream", "particle"] if args.lane == "both" else [args.lane]
    results = [run_lane(lane, pres, names, reals) for lane in lanes]

    print("\nSTAGE S0 RESULT — decisive 5-arrangement test")
    for r in results:
        print(f"  {r['lane']}: {'PASS' if r['pass'] else 'FAIL'} "
              f"(winner {r['margins']['winner']:+.3f} vs FPs "
              f"{[round(v, 3) for k, v in r['margins'].items() if k != 'winner']})")
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(json.dumps(
            [{k: (v if k != "margins" else {m: float(x) for m, x in v.items()})
              for k, v in r.items()} for r in results], indent=1))
    return results


if __name__ == "__main__":
    main()
