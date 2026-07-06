"""DW round-3 — re-select pour dynamics on ALL observed data, incl. the flywheel.

Round 2's holdout "perfection" hid a horizon inconsistency: fitting evaluated
outcomes at the observed animation length (~20-28 ticks) while dw_solve rolls
to quiescence (200), so mouth overflow that the truncated fit never saw became
systematic imagined lava on real winners at solve time (the 5-seed plateau).

Round 3 fixes the semantics honestly and lets observation re-select:
  - physics family: round-2 knobs x absorb {cap, unlimited}  (192 rollout cfgs)
  - outcome semantics evaluated post-hoc per rollout, 4 combos:
      rule  'fill_cap' | 'entered'   win predicate (capacity vs first-entry)
      stop  'volume'   | 'win'       end-state at quiescence vs win-moment
    win-moment semantics is horizon-independent: a pour wins iff the predicate
    fires with ZERO floor touches so far; a loss reports FINAL floor count.
  - corpora: calib L0 (55) + calib L1 (61) + r3 solve-time flywheel (31, incl.
    the 4 sweep winners). L2 corpus stays a zero-shot exam, untouched.
  - split: per-corpus i%4==3 holdout; select on train by (outcome+lava, IoU);
    gates on holdout: outcome >= 0.90, median first-div >= 10. Win-side and
    solve-time subset accuracies reported (the sweep's failure mode).

    python3 r3_fit.py
Exit 0 if holdout gates pass, 2 otherwise (negative branch: escalate).
"""
from __future__ import annotations

import itertools
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES

from dynamics_fit import load_pours, GATE_OUTCOME, GATE_FIRSTDIV  # noqa: E402
from imagination_kernel import FallingMaterial, run, win_moment  # noqa: E402
from scene import M  # noqa: E402

CORPORA = [("l0_calib", "calib_l0_pours.npz"),
           ("l1_calib", "calib_l1_pours_20260706.npz"),
           ("solvetime", "r3_solvetime_pours_20260706.npz")]
SEMANTICS = list(itertools.product(["fill_cap", "entered"], ["volume", "win"]))


def rollout_eval(cfg, pour):
    """One rollout to quiescence -> tick metrics + the 4 semantics verdicts."""
    states = run(pour["kmap"], FallingMaterial(**cfg), max_ticks=200)
    n = min(pour["n_ticks"], len(states))
    tick_iou, div = [], n
    for t in range(n):
        s = states[t]
        m = (s.settled > 0).copy()
        for (y, x) in s.moving:
            m[y, x] = True
        b = pour["obs_masks"][t]
        inter, union = int((m & b).sum()), int((m | b).sum())
        tick_iou.append(inter / union if union else 1.0)
        if div == n and not (m == b).all():
            div = t
    caps = pour["kmap"].capacities
    final = states[-1]
    verdicts = {}
    for rule, stop in SEMANTICS:
        if stop == "volume":
            req = caps if rule == "fill_cap" else [1.0] * len(caps)
            w = (len(caps) > 0 and all(f >= r for f, r in zip(final.fills, req))
                 and final.floor_touch == 0)
            lava = final.floor_touch > 0
        else:
            tick, floor_at = win_moment(states, caps, rule=rule)
            w = tick is not None and floor_at == 0
            lava = floor_at > 0
        verdicts[(rule, stop)] = (bool(w), bool(lava))
    return {"iou": float(np.mean(tick_iou)) if tick_iou else 0.0,
            "fdiv": div, "verdicts": verdicts}


def score(rows, pours, rule, stop):
    out_ok = [r["verdicts"][(rule, stop)][0] == p["obs_win"]
              for r, p in zip(rows, pours)]
    lava_ok = [r["verdicts"][(rule, stop)][1] == p["obs_lava"]
               for r, p in zip(rows, pours)]
    wins = [i for i, p in enumerate(pours) if p["obs_win"]]
    return {"outcome_acc": float(np.mean(out_ok)) if out_ok else 0.0,
            "lava_acc": float(np.mean(lava_ok)) if lava_ok else 0.0,
            "mean_iou": float(np.mean([r["iou"] for r in rows])) if rows else 0.0,
            "median_first_div": float(np.median([r["fdiv"] for r in rows])) if rows else 0.0,
            "win_side_acc": float(np.mean([out_ok[i] for i in wins])) if wins else None,
            "n": len(rows), "n_wins": len(wins)}


def main():
    corpus = []                                     # (corpus_name, pour) with split tag
    for name, fx in CORPORA:
        pours, skipped = load_pours(FIXTURES / fx)
        for p in pours:
            p["corpus"] = name
            p["hold"] = (p["i"] % 4 == 3)
        corpus += pours
        print(f"[r3fit] {name}: {len(pours)} usable ({len(skipped)} gate-skipped), "
              f"{sum(p['obs_win'] for p in pours)} wins", flush=True)
    train = [p for p in corpus if not p["hold"]]
    hold = [p for p in corpus if p["hold"]]
    print(f"[r3fit] train {len(train)} ({sum(p['obs_win'] for p in train)} wins) / "
          f"holdout {len(hold)} ({sum(p['obs_win'] for p in hold)} wins)", flush=True)

    grid = [dict(zip(("deflect", "spread", "mouth", "edge", "cell_cap", "absorb"), v))
            for v in itertools.product(
                ["none", "dl", "dr", "both"], ["none", "lateral"], ["fill", "entry"],
                ["dispose", "halt"], [1.0, 2.0, 3.0], ["cap", "unlimited"])]
    table = []
    for gi, cfg in enumerate(grid):
        rows = [rollout_eval(cfg, p) for p in train]
        for rule, stop in SEMANTICS:
            table.append({"cfg": cfg, "rule": rule, "stop": stop,
                          "train": score(rows, train, rule, stop)})
        if (gi + 1) % 48 == 0:
            print(f"[r3fit] {gi + 1}/{len(grid)} rollout configs", flush=True)
    table.sort(key=lambda e: (e["train"]["outcome_acc"] + e["train"]["lava_acc"],
                              e["train"]["mean_iou"]), reverse=True)

    hdr = (f"{'deflect':>7} {'spread':>7} {'mouth':>6} {'edge':>7} {'cap':>4} "
           f"{'absorb':>9} {'rule':>8} {'stop':>6} | {'IoU':>6} {'fdiv':>5} "
           f"{'out':>5} {'lava':>5} {'winacc':>6}")
    print(hdr)
    for e in table[:14]:
        c, r = e["cfg"], e["train"]
        wa = "-" if r["win_side_acc"] is None else f"{r['win_side_acc']:.2f}"
        print(f"{c['deflect']:>7} {c['spread']:>7} {c['mouth']:>6} {c['edge']:>7} "
              f"{c['cell_cap']:>4.1f} {c['absorb']:>9} {e['rule']:>8} {e['stop']:>6} | "
              f"{r['mean_iou']:>6.3f} {r['median_first_div']:>5.1f} "
              f"{r['outcome_acc']:>5.2f} {r['lava_acc']:>5.2f} {wa:>6}")

    best = table[0]
    hrows = [rollout_eval(best["cfg"], p) for p in hold]
    hr = score(hrows, hold, best["rule"], best["stop"])
    st_idx = [i for i, p in enumerate(hold) if p["corpus"] == "solvetime"]
    st = score([hrows[i] for i in st_idx], [hold[i] for i in st_idx],
               best["rule"], best["stop"]) if st_idx else None
    gates = {"outcome": hr["outcome_acc"] >= GATE_OUTCOME,
             "firstdiv": hr["median_first_div"] >= GATE_FIRSTDIV,
             "win_side": hr["win_side_acc"] is None or hr["win_side_acc"] == 1.0}
    passed = all(gates.values())
    print(f"\nBEST {best['cfg']} rule={best['rule']} stop={best['stop']}")
    print(f"HOLDOUT {hr}")
    if st:
        print(f"HOLDOUT solve-time subset {st}")
    print(f"GATES {gates} -> {'PASS' if passed else 'FAIL (negative branch: escalate)'}")

    out = FIXTURES / "dynamics_round3_20260706.json"
    out.write_text(json.dumps({
        "protocol": "round-3: physics x absorb grid, outcome semantics post-hoc "
                    "(rule x stop), quiescence-horizon outcomes (fixes round-2 "
                    "truncation artifact); train-select, holdout-gate; corpora "
                    "L0+L1 calib + solve-time flywheel; observations only",
        "corpora": [c[1] for c in CORPORA],
        "selected": {"cfg": best["cfg"], "rule": best["rule"], "stop": best["stop"]},
        "train": best["train"], "holdout": hr, "holdout_solvetime": st,
        "table_top40": table[:40], "gates": gates, "pass": bool(passed)},
        indent=1, default=str))
    print(f"-> {out}", flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
