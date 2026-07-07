"""DW Phase-3 — induce SP80 pour dynamics from OBSERVED env pours only.

Fits the generic FallingMaterial hypothesis family (imagination_kernel) against
`fixtures/calib_l0_pours.npz` (55 observed L0 pours: full frame sequences +
env win labels — all legal interaction data). No engine internals; the Phase-0
hook enforces this mechanically.

Pre-registered protocol (DREAMWEAVER_INTEGRATION_PLAN_20260706.md §2 H2, §3.P3):
  - split: episode-level, holdout = pour indices with i % 4 == 3
  - full family fit table is reported, not just the winner
  - selection on TRAIN by (outcome_acc + lava_acc, mean IoU); gates on HOLDOUT:
    outcome agreement >= 0.90 and median first-divergence tick >= 10
  - negative branch: if no config passes, ESCALATE with the table (do not
    widen the family by peeking at anything but observations)

Observed conventions (verified empirically, see DW_LAB_LOG):
  - 1 LATTICE cell of paint appears per animation frame per stream tip -> tick
    alignment 1:1 (holds on both the 4px and 3px native lattices)
  - material mask = (frame == 6) per lattice cell (ANY) on the scene's padded
    canonical lattice, minus the pre-pour reservoir cells
  - lava = some pixel row NEWLY majority-14 (the strip flashes as a near-full-
    width band: 253-256 px on 4px levels, 179-180 px on L3's 3px strip — the
    old fixed >200 delta misread L3 as no-lava on all 40 flashing pours;
    round-4 Phase B replaced it after verifying label identity on every 4px
    corpus, see r4_phaseb fixture)
  - fill = any color-13 pixel inside a target's glyph region

    python3 dynamics_fit.py [--caps 1,2,3] [--out fixtures/dynamics_fit_l0_20260706.json]
Exit 0 if the holdout gates pass, 2 otherwise (negative branch).
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES

from scene import extract_scene, M, CELL, GRID, OPEN, PAINT, WALL, _lattice_any
from perception_gate import gate_scene
from imagination_kernel import KnownMap, FallingMaterial, run

GATE_OUTCOME = 0.90
GATE_FIRSTDIV = 10


def macro_any(mask64):
    return mask64.reshape(M, CELL, M, CELL).any(axis=(1, 3))


def _new_majority_lines(pre, frame, axis):
    pre_lines = (np.asarray(pre) == WALL).mean(axis=axis) > 0.5
    lines = (np.asarray(frame) == WALL).mean(axis=axis) > 0.5
    return int((lines & ~pre_lines).sum())


def row_flash(pre, frames):
    """Lava = >= 2 pixel wall-band lines NEWLY majority-WALL(14) in one
    frame. L0-L4 flash as row bands; L5 side-wall failures expose the same
    mechanic as column bands. Exactly 1 new line is the WIN/level-transition
    recolor of the outer HUD border row (final frame of winning animations);
    separation is regression-checked against the old rows-only detector on all
    five pre-r5 corpora."""
    for f in frames:
        new_rows = _new_majority_lines(pre, f, axis=1)
        new_cols = _new_majority_lines(pre, f, axis=0)
        if new_rows >= 2 or new_cols >= 2:
            return True
    return False


def load_pours(path):
    d = np.load(path, allow_pickle=True)
    pre, ff, lens, wins = d["pre"], d["frames_flat"], d["lens"], d["wins"]
    pours, off, skipped = [], 0, []
    for i in range(len(lens)):
        frames = ff[off:off + lens[i]]; off += lens[i]
        sc = extract_scene(pre[i])
        v = gate_scene(pre[i], sc)
        if not v["trusted"]:
            skipped.append({"pour": i, "gate_failures": v["gate_failures"]})
            continue
        kmap = KnownMap.from_scene(sc, v, pour=i)
        u, (py, px) = sc.cell, sc.pad
        gh, gw = sc.solid.shape

        def canon(f64):
            f = np.full((gh * u, gw * u), OPEN, dtype=np.int8)
            f[py:py + GRID, px:px + GRID] = np.flipud(f64) if sc.flipped else f64
            return f

        reservoir = _lattice_any(canon(pre[i]), [PAINT], u)
        obs_masks = [_lattice_any(canon(f), [PAINT], u) & ~reservoir for f in frames]
        pours.append({"i": i, "kmap": kmap, "obs_masks": obs_masks,
                      "obs_win": bool(wins[i]), "obs_lava": row_flash(pre[i], frames),
                      "n_ticks": int(lens[i])})
    return pours, skipped


def sim_masks(kmap, hyp, n_ticks):
    """obs[t] is the board after t animation frames -> compare against state[t]
    (state[0] = pre-motion, empty). Alignment verified empirically (DW_LAB_LOG)."""
    states = run(kmap, hyp, max_ticks=n_ticks)
    masks, finals = [], states[-1]
    for s in states[:n_ticks]:
        m = (s.settled > 0)
        for tip in s.moving:
            m = m.copy(); m[tip[0], tip[1]] = True
        masks.append(m)
    while len(masks) < n_ticks:
        masks.append(masks[-1] if masks else np.zeros((M, M), dtype=bool))
    return masks, finals


def eval_config(cfg, pours):
    hyp = FallingMaterial(**cfg)
    ious, firstdivs, out_ok, lava_ok = [], [], [], []
    for p in pours:
        sm, final = sim_masks(p["kmap"], FallingMaterial(**cfg), p["n_ticks"])
        n = min(len(sm), len(p["obs_masks"]))
        div = n
        tick_iou = []
        for t in range(n):
            a, b = sm[t], p["obs_masks"][t]
            inter, union = int((a & b).sum()), int((a | b).sum())
            tick_iou.append(inter / union if union else 1.0)
            if div == n and not (a == b).all():
                div = t
        ious.append(float(np.mean(tick_iou)) if tick_iou else 0.0)
        firstdivs.append(div)
        caps = p["kmap"].capacities
        sim_win = (all(f >= c for f, c in zip(final.fills, caps))
                   and final.floor_touch == 0)
        out_ok.append(sim_win == p["obs_win"])
        lava_ok.append((final.floor_touch > 0) == p["obs_lava"])
    return {"mean_iou": float(np.mean(ious)),
            "median_first_div": float(np.median(firstdivs)),
            "outcome_acc": float(np.mean(out_ok)),
            "lava_acc": float(np.mean(lava_ok)),
            "n": len(pours)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="calib_l0_pours.npz")
    ap.add_argument("--caps", default="1,2,3")
    ap.add_argument("--out", default="dynamics_fit_l0_20260706.json")
    a = ap.parse_args()

    pours, skipped = load_pours(FIXTURES / a.fixture)
    train = [p for p in pours if p["i"] % 4 != 3]
    hold = [p for p in pours if p["i"] % 4 == 3]
    n_win = sum(p["obs_win"] for p in pours)
    base_out = max(n_win, len(pours) - n_win) / len(pours)   # majority-class baseline
    print(f"pours: {len(pours)} usable ({len(skipped)} gate-skipped) "
          f"-> train {len(train)} / holdout {len(hold)}; wins {n_win} "
          f"(majority-class outcome baseline {base_out:.3f})")

    grid = list(itertools.product(
        ["none", "dl", "dr", "both"], ["none", "lateral"],
        ["fill", "entry"], ["dispose", "halt"],
        [float(c) for c in a.caps.split(",")]))
    table = []
    for deflect, spread, mouth, edge, cap in grid:
        cfg = {"deflect": deflect, "spread": spread, "mouth": mouth,
               "edge": edge, "cell_cap": cap}
        r = eval_config(cfg, train)
        table.append({"cfg": cfg, "train": r})
    table.sort(key=lambda e: (e["train"]["outcome_acc"] + e["train"]["lava_acc"],
                              e["train"]["mean_iou"]), reverse=True)

    print(f"{'deflect':>7} {'spread':>7} {'mouth':>6} {'edge':>7} {'cap':>4} | "
          f"{'IoU':>6} {'fdiv':>5} {'out':>5} {'lava':>5}")
    for e in table[:12]:
        c, r = e["cfg"], e["train"]
        print(f"{c['deflect']:>7} {c['spread']:>7} {c['mouth']:>6} {c['edge']:>7} "
              f"{c['cell_cap']:>4.1f} | {r['mean_iou']:>6.3f} {r['median_first_div']:>5.1f} "
              f"{r['outcome_acc']:>5.2f} {r['lava_acc']:>5.2f}")

    best = table[0]
    hr = eval_config(best["cfg"], hold)
    gates = {"holdout_outcome_acc": hr["outcome_acc"],
             "holdout_lava_acc": hr["lava_acc"],
             "holdout_median_first_div": hr["median_first_div"],
             "gate_outcome_pass": hr["outcome_acc"] >= GATE_OUTCOME,
             "gate_firstdiv_pass": hr["median_first_div"] >= GATE_FIRSTDIV}
    passed = gates["gate_outcome_pass"] and gates["gate_firstdiv_pass"]
    print(f"\nBEST {best['cfg']}\nHOLDOUT {hr}\nGATES {gates} -> "
          f"{'PASS' if passed else 'FAIL (negative branch: escalate)'}")

    (FIXTURES / a.out).write_text(json.dumps({
        "protocol": "pre-registered (plan §2 H2): train-select, holdout-gate; "
                    "full table reported; observations only",
        "split": {"train": [p["i"] for p in train], "holdout": [p["i"] for p in hold],
                  "gate_skipped": skipped},
        "majority_class_outcome_baseline": base_out, "n_wins": int(n_win),
        "table": table, "best": best, "holdout": hr, "gates": gates,
        "h2_pass": bool(passed)}, indent=1, default=str))
    print(f"-> {FIXTURES / a.out}")
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
