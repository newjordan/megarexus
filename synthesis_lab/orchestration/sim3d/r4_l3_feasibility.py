"""Round-4 L3 decision-branch probe — is a clean L3 win FEASIBLE in kernel space?

The Phase-D sweep + 6.6x probe show the r4 kernel imagines 4/4-fill L3
arrangements but always with 3-4 residual floor touches (surplus split legs
dying in the gap columns). Before investing in a constructive proposer — and
per the pre-registered decision branch (escalate if no positive-margin
arrangement exists) — this searches DIRECT bar placements in the KnownMap
(pure kernel space, no env, no reachability constraint): translate the
movable bars anywhere legal and ask whether ANY configuration yields
fills=all + floor_touch=0 under the round-4 induced model.

  - feasible (wins found)  -> exit 0; the proposer's job is then only to
    REALIZE a winning placement via legal clicks (env decides, as always)
  - infeasible             -> exit 2; escalation evidence for the
    "intended L3 mechanic" question (STOP, do not grind)

    python3 r4_l3_feasibility.py [--restarts 150] [--steps 250]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES

from scene import extract_scene, _pixel_components, MOVABLE, SELECTED  # noqa: E402
from perception_gate import gate_scene  # noqa: E402
from imagination_kernel import KnownMap, FlowTrail, run  # noqa: E402

MAX_ROW = None
TARGETS = None
ALLOW_ADJ = False
R4_CFG = {"emitters": "all", "phase": "uncharged_lead",
          "mouth": "feed_die", "volume": None, "merge": "die"}


def base_map():
    z = np.load(FIXTURES / "calib_l3_pours_20260706.npz")
    pre = z["pre"][0]
    sc = extract_scene(pre)
    v = gate_scene(pre, sc)
    km = KnownMap.from_scene(sc, v, source="l3_default_board")
    g = np.asarray(sc.g)
    bars = [sorted((int(y), int(x)) for y, x in zip(*np.nonzero(m)))
            for m in [np.zeros_like(sc.solid)]]  # placeholder replaced below
    mov = (g == MOVABLE) | (g == SELECTED)
    bars = [[(int(y), int(x)) for y, x in zip(*comp)]
            for comp in [(ys, xs) for ys, xs in _pixel_components(mov)]]
    return km, bars


def build_solid(km, bars, offsets):
    """None if any translated bar collides/out-of-bounds, else the new solid."""
    gh, gw = km.solid.shape
    hud = km.hud if km.hud is not None else np.zeros_like(km.solid)
    mouth_cells = {c for cells in km.mouths for c in cells}
    bar_mask = np.zeros_like(km.solid)
    for cells in bars:
        for (y, x) in cells:
            bar_mask[y, x] = True
    static = km.solid & ~bar_mask
    solid = static.copy()
    placed = []                     # per-bar translated cell sets
    for cells, (dy, dx) in zip(bars, offsets):
        cur = set()
        for (y, x) in cells:
            ny, nx = y + dy, x + dx
            if not (0 <= ny < gh and 0 <= nx < gw):
                return None
            if ny in km.floor_rows or solid[ny, nx] or hud[ny, nx] \
                    or (ny, nx) in mouth_cells:
                return None
            if MAX_ROW is not None and ny > MAX_ROW:
                return None
            solid[ny, nx] = True
            cur.add((ny, nx))
        placed.append(cur)
    # bars must stay 4-separated from each other (else same-color comps fuse
    # in the frame and closed-loop tracking/selection breaks at realization)
    if ALLOW_ADJ:
        return solid
    for i, ci in enumerate(placed):
        for j, cj in enumerate(placed):
            if i >= j:
                continue
            for (y, x) in ci:
                if any((y + dy, x + dx) in cj
                       for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))):
                    return None
    return solid


def score(km, solid):
    km2 = KnownMap(solid=solid, floor_rows=km.floor_rows, spouts=km.spouts,
                   mouths=km.mouths, glyph_cells=km.glyph_cells,
                   capacities=km.capacities, emit_volume=km.emit_volume,
                   all_spouts=km.all_spouts, spout_charged=km.spout_charged,
                   hud=km.hud, provenance={"derived": "placement_search"})
    states = run(km2, FlowTrail(**R4_CFG), max_ticks=200)
    end = states[-1]
    idx = TARGETS if TARGETS is not None else range(len(km.capacities))
    fill = float(np.mean([min(end.fills[i] / km.capacities[i], 1.0) for i in idx]))
    lava = end.floor_touch / max(float(end.emitted), 1.0)
    win = (all(end.fills[i] >= km.capacities[i] for i in idx)
           and end.floor_touch == 0)
    return {"fitness": fill - lava + (1.0 if win else 0.0), "fill": fill,
            "floor": int(end.floor_touch), "fills": list(map(int, end.fills)),
            "win": bool(win)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=150)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze", default="",
                    help="comma-separated bar indices that must stay at (0,0) "
                         "(e.g. env-probed unselectable comps)")
    ap.add_argument("--allow-adjacent", action="store_true",
                    help="skip the bar-separation constraint (realization must "
                         "then keep the adjacent bar SELECTED to stay trackable)")
    ap.add_argument("--targets", default=None,
                    help="comma-separated target indices: win = THESE mouths filled "
                         "with floor 0 (clean PARTIAL pour, multi-pour planning)")
    ap.add_argument("--out", default="r4_l3_feasibility_20260706.json")
    ap.add_argument("--max-row", type=int, default=None,
                    help="reject placements with any bar cell below this row "
                         "(realization recon: rows 16-17 proved env-unreachable)")
    a = ap.parse_args()
    global MAX_ROW, TARGETS, ALLOW_ADJ
    ALLOW_ADJ = a.allow_adjacent
    MAX_ROW = a.max_row
    TARGETS = (None if a.targets is None
               else sorted(int(t) for t in a.targets.split(",")))
    km, bars = base_map()
    frozen = {int(i) for i in a.freeze.split(",") if i != ""}
    free = [i for i in range(len(bars)) if i not in frozen]
    print(f"[l3feas] {len(bars)} bars, sizes {[len(b) for b in bars]}, "
          f"frozen {sorted(frozen)}", flush=True)
    rng = np.random.default_rng(a.seed)

    def rand_offs():
        return [(0, 0) if i in frozen else
                (int(rng.integers(-9, 10)), int(rng.integers(-12, 13)))
                for i in range(len(bars))]

    wins, best_overall = [], None
    for r in range(a.restarts):
        offs = [(0, 0)] * len(bars) if r == 0 else rand_offs()
        solid = build_solid(km, bars, offs)
        while solid is None:
            offs = rand_offs()
            solid = build_solid(km, bars, offs)
        best = (score(km, solid), offs)
        for _ in range(a.steps):
            i = free[int(rng.integers(len(free)))]
            d = [list(o) for o in best[1]]
            d[i][int(rng.integers(2))] += int(rng.choice([-3, -2, -1, 1, 2, 3]))
            cand = [tuple(o) for o in d]
            solid = build_solid(km, bars, cand)
            if solid is None:
                continue
            s = score(km, solid)
            if s["fitness"] > best[0]["fitness"]:
                best = (s, cand)
        if best[0]["win"]:
            wins.append({"offsets": [list(o) for o in best[1]], **best[0]})
        if best_overall is None or best[0]["fitness"] > best_overall[0]["fitness"]:
            best_overall = best
        if (r + 1) % 25 == 0:
            print(f"[l3feas] {r+1}/{a.restarts} restarts, wins {len(wins)}, "
                  f"best {best_overall[0]}", flush=True)

    out = FIXTURES / a.out
    out.write_text(json.dumps({
        "protocol": "kernel-space placement feasibility (no env, no reachability): "
                    "translate movable bars, r4 FlowTrail scores; decision-branch "
                    "evidence for the L3 lane",
        "targets": TARGETS, "n_bars": len(bars), "bar_sizes": [len(b) for b in bars],
        "restarts": a.restarts, "steps": a.steps,
        "wins_found": len(wins), "win_examples": wins[:10],
        "best": {"offsets": [list(o) for o in best_overall[1]], **best_overall[0]},
    }, indent=1))
    print(f"[l3feas] wins={len(wins)} best={best_overall[0]} -> {out}", flush=True)
    raise SystemExit(0 if wins else 2)


if __name__ == "__main__":
    main()
