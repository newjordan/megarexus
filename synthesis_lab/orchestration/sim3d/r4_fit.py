"""DW round-4 — joint dynamics selection: FallingMaterial vs FlowTrail, four corpora.

Round-4 context (DW_LAB_LOG Phase A): the L3 corpus shows a liquid-film TRAIL
flow — persistent painted trails, spread-and-split at obstacle tops, both
housings emitting (uncharged leads by 1 tick) — behavior the FallingMaterial
family cannot express (its per-unit piling puts a unit in one place; the
observed streams split). FlowTrail (imagination_kernel) is the generic film-
flow family; every knob has alternatives and observation alone selects.

Pre-registered protocol (DW_LAB_LOG round-4 entry):
  - corpora: L0 calib (55) + L1 calib (61) + solve-time flywheel (31) + L3
    calib (40); per-corpus split, holdout = pour i%4==3. L2 stays a ZERO-SHOT
    exam (l2_zeroshot section of the fixture), untouched by selection.
  - families: FallingMaterial full 192-cfg grid (round-3 knobs x absorb) and
    FlowTrail 16-cfg grid (emitters x phase x mouth x volume); outcome
    semantics post-hoc per rollout (rule in {fill_cap, entered} x stop in
    {volume, win}), exactly as round 3.
  - tick alignment: FallingMaterial obs[t] <-> state[t]; FlowTrail
    obs[t] <-> state[t+1] (declared family convention: frame 0 is captured
    after the first flow tick — uncharged streams already visible there).
  - volume='measured' resolves to the corpus's median peak visible paint in
    lattice cells (pure observation; recorded in the fixture).
  - selection on pooled TRAIN by (outcome_acc + lava_acc, mean IoU); full
    table reported. GATES on HOLDOUT, per corpus:
      all corpora:   outcome >= 0.90 and lava >= 0.90
      L0/L1:         outcome == 1.00 and lava == 1.00 (no regression vs r3)
      win-side:      1.00 wherever holdout wins exist
      L3:            median first-div >= 10
    Negative branch: exit 2 and escalate with the table — do not widen the
    family blind.

    python3 r4_fit.py
"""
from __future__ import annotations

import itertools
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES

from dynamics_fit import load_pours, GATE_FIRSTDIV  # noqa: E402
from imagination_kernel import FallingMaterial, FlowTrail, run, win_moment  # noqa: E402
from scene import PAINT  # noqa: E402

CORPORA = [("l0_calib", "calib_l0_pours.npz"),
           ("l1_calib", "calib_l1_pours_20260706.npz"),
           ("solvetime", "r3_solvetime_pours_20260706.npz"),
           ("l3_calib", "calib_l3_pours_20260706.npz")]
L2_EXAM = "calib_l2_pours_20260706.npz"
SEMANTICS = list(itertools.product(["fill_cap", "entered"], ["volume", "win"]))


def measured_volume(fx):
    """Median peak visible paint per pour, in LATTICE cells (observation only)."""
    from scene import extract_scene
    z = np.load(FIXTURES / fx, allow_pickle=True)
    lens, ff, pre = z["lens"], z["frames_flat"], z["pre"]
    vols, off = [], 0
    for i in range(len(lens)):
        frames = ff[off:off + lens[i]]; off += lens[i]
        u = extract_scene(pre[i]).cell
        n6 = np.array([int((f == PAINT).sum()) for f in frames])
        vols.append((int(n6.max()) - int(n6[0])) / float(u * u))
    return float(np.median(vols))


def rollout_eval(entry, pour, vol_of):
    """One rollout to quiescence -> tick metrics + the 4 semantics verdicts."""
    if entry["family"] == "falling_material":
        hyp, off = FallingMaterial(**entry["cfg"]), 0
    else:
        cfg = dict(entry["cfg"])
        cfg["volume"] = vol_of[pour["corpus"]] if cfg["volume"] == "measured" else None
        hyp, off = FlowTrail(**cfg), 1
    states = run(pour["kmap"], hyp, max_ticks=200)
    n = pour["n_ticks"]
    tick_iou, div = [], n
    for t in range(n):
        s = states[min(t + off, len(states) - 1)]
        m = (s.settled > 0).copy()
        for tip in s.moving:
            m[tip[0], tip[1]] = True
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
    vol_of = {name: measured_volume(fx) for name, fx in CORPORA}
    print(f"[r4fit] measured volumes (lattice cells): {vol_of}", flush=True)

    corpus = []
    for name, fx in CORPORA:
        pours, skipped = load_pours(FIXTURES / fx)
        for p in pours:
            p["corpus"] = name
            p["hold"] = (p["i"] % 4 == 3)
        corpus += pours
        print(f"[r4fit] {name}: {len(pours)} usable ({len(skipped)} gate-skipped), "
              f"{sum(p['obs_win'] for p in pours)} wins", flush=True)
    train = [p for p in corpus if not p["hold"]]
    hold = [p for p in corpus if p["hold"]]
    print(f"[r4fit] train {len(train)} ({sum(p['obs_win'] for p in train)} wins) / "
          f"holdout {len(hold)} ({sum(p['obs_win'] for p in hold)} wins)", flush=True)

    grid = [{"family": "falling_material",
             "cfg": dict(zip(("deflect", "spread", "mouth", "edge", "cell_cap", "absorb"), v))}
            for v in itertools.product(
                ["none", "dl", "dr", "both"], ["none", "lateral"], ["fill", "entry"],
                ["dispose", "halt"], [1.0, 2.0, 3.0], ["cap", "unlimited"])]
    grid += [{"family": "flow_trail",
              "cfg": dict(zip(("emitters", "phase", "mouth", "volume", "merge"), v))}
             for v in itertools.product(
                 ["charged", "all"], ["sync", "uncharged_lead"],
                 ["feed_die", "feed_overflow"], ["unlimited", "measured"],
                 ["split", "die"])]

    table = []
    for gi, entry in enumerate(grid):
        rows = [rollout_eval(entry, p, vol_of) for p in train]
        for rule, stop in SEMANTICS:
            table.append({**entry, "rule": rule, "stop": stop,
                          "train": score(rows, train, rule, stop)})
        if (gi + 1) % 48 == 0:
            print(f"[r4fit] {gi + 1}/{len(grid)} rollout configs", flush=True)
    table.sort(key=lambda e: (e["train"]["outcome_acc"] + e["train"]["lava_acc"],
                              e["train"]["mean_iou"]), reverse=True)

    print(f"{'family':>16} {'cfg':<70} {'rule':>8} {'stop':>6} | "
          f"{'IoU':>6} {'fdiv':>5} {'out':>5} {'lava':>5} {'winacc':>6}")
    for e in table[:14]:
        r = e["train"]
        wa = "-" if r["win_side_acc"] is None else f"{r['win_side_acc']:.2f}"
        print(f"{e['family']:>16} {json.dumps(e['cfg']):<70.70} {e['rule']:>8} "
              f"{e['stop']:>6} | {r['mean_iou']:>6.3f} {r['median_first_div']:>5.1f} "
              f"{r['outcome_acc']:>5.2f} {r['lava_acc']:>5.2f} {wa:>6}")

    best = table[0]
    hrows = [rollout_eval(best, p, vol_of) for p in hold]
    by_corpus, gates = {}, {}
    for name, _ in CORPORA:
        idx = [i for i, p in enumerate(hold) if p["corpus"] == name]
        s = score([hrows[i] for i in idx], [hold[i] for i in idx],
                  best["rule"], best["stop"])
        by_corpus[name] = s
        ok = s["outcome_acc"] >= 0.90 and s["lava_acc"] >= 0.90
        if name in ("l0_calib", "l1_calib"):
            ok = ok and s["outcome_acc"] == 1.0 and s["lava_acc"] == 1.0
        if s["win_side_acc"] is not None:
            ok = ok and s["win_side_acc"] == 1.0
        if name == "l3_calib":
            ok = ok and s["median_first_div"] >= GATE_FIRSTDIV
        gates[name] = ok
    passed = all(gates.values())

    print(f"\nBEST {best['family']} {best['cfg']} rule={best['rule']} stop={best['stop']}")
    for name, s in by_corpus.items():
        print(f"HOLDOUT {name}: {s} -> {'PASS' if gates[name] else 'FAIL'}")
    print(f"GATES {gates} -> {'PASS' if passed else 'FAIL (negative branch: escalate)'}")

    # L2 zero-shot exam (report-only, not selection)
    l2, l2_skip = load_pours(FIXTURES / L2_EXAM)
    for p in l2:
        p["corpus"] = "l2_exam"
    vol_of["l2_exam"] = measured_volume(L2_EXAM)
    l2_rows = [rollout_eval(best, p, vol_of) for p in l2]
    l2_score = score(l2_rows, l2, best["rule"], best["stop"])
    print(f"L2 ZERO-SHOT (exam, not gated): {l2_score}")

    out = FIXTURES / "dynamics_round4_20260706.json"
    out.write_text(json.dumps({
        "protocol": "round-4: FallingMaterial(192) + FlowTrail(16) grids, semantics "
                    "post-hoc; corpora L0+L1+solvetime+L3, per-corpus i%4==3 holdout; "
                    "select pooled-train (outcome+lava, IoU); per-corpus holdout gates "
                    "(L0/L1 no-regression 1.00, L3 fdiv>=10); L2 zero-shot exam only; "
                    "observations only",
        "corpora": [c[1] for c in CORPORA],
        "measured_volumes": vol_of,
        "selected": {"family": best["family"], "cfg": best["cfg"],
                     "rule": best["rule"], "stop": best["stop"]},
        "train": best["train"], "holdout_by_corpus": by_corpus, "gates": gates,
        "l2_zeroshot": {"score": l2_score, "gate_skipped": len(l2_skip)},
        "table_top40": table[:40], "pass": bool(passed)}, indent=1, default=str))
    print(f"-> {out}", flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
