"""P2b — direct GRID search for an L2 win with the exact droplet CA (dropletsim.pour_core).

The move-genome search moves each bar on one axis only; the win likely needs bars at 2D
positions. Here we place the 4 movable bars at arbitrary (row, col) on the macro grid (fixed
sources/mouths/wall from the real L2 scene), score with the exact, free droplet CA, and
hill-climb toward all-3-filled with ZERO wall-hits (a clean win: every excess branch shed off
the sides). No env in the loop; a found win is reconstructed and env-verified separately.

    python3 p2_grid_search.py --restarts 4000 --steps 300
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, LAB
from imagine_search_gate import _present, GAME_ID  # noqa: E402
from scene import extract_scene, M, MOVABLE, SELECTED  # noqa: E402
from dropletsim import pour_core, _typed  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402


def l2_scene():
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    env = arc.make(GAME_ID, seed=1234); _, g = _present(env, 2)
    sc = extract_scene(g)
    block, target, wall, tid = _typed(sc)
    # movable bar widths (canonical): find horizontal runs of block cells
    bars = []
    seen = np.zeros_like(block)
    for y in range(M):
        x = 0
        while x < M:
            if block[y, x] and not seen[y, x]:
                x1 = x
                while x1 + 1 < M and block[y, x1 + 1]:
                    x1 += 1
                bars.append({"w": x1 - x + 1, "row0": y, "col0": x})
                seen[y, x:x1 + 1] = True
                x = x1 + 1
            else:
                x += 1
    spouts = [(int(sy), int(sx)) for (sy, sx) in sc.spouts]
    return target, wall, tid, spouts, len(sc.targets), bars


def build_block(placements, widths):
    b = np.zeros((M, M), dtype=bool)
    for (row, col), w in zip(placements, widths):
        if 0 <= row < M and 0 <= col and col + w <= M:
            b[row, col:col + w] = True
    return b


def valid(placements, widths, target, wall, spouts):
    occ = np.zeros((M, M), dtype=bool)
    src_cols = {sx for _, sx in spouts}
    for (row, col), w in zip(placements, widths):
        if not (3 <= row <= 12 and 0 <= col and col + w <= M):
            return False
        if occ[row, col:col + w].any():
            return False                              # bars can't overlap
        seg = np.zeros(M, dtype=bool); seg[col:col + w] = True
        if (target[row] & seg).any() or (wall[row] & seg).any():
            return False
        occ[row, col:col + w] = True
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=4000)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    target, wall, tid, spouts, ntgt, bars = l2_scene()
    widths = [b["w"] for b in bars]
    nb = len(bars)
    rng = np.random.default_rng(a.seed)

    def score(placements):
        b = build_block(placements, widths)
        r = pour_core(b, target, wall, tid, spouts, ntgt)
        return r

    def rand_place():
        for _ in range(200):
            pl = [(int(rng.integers(3, 13)), int(rng.integers(0, M - w + 1))) for w in widths]
            if valid(pl, widths, target, wall, spouts):
                return pl
        return None

    def fitness(r):   # win first; else maximize fills, minimize wall-hits
        return (r["win"], r["n_filled"], -r["n_wall"])

    best_overall, best_fit, evals, win_cfg = None, (False, -1, -99), 0, None
    for _r in range(a.restarts):
        pl = rand_place()
        if pl is None:
            continue
        r = score(pl); evals += 1
        cur, cf = pl, fitness(r)
        for _s in range(a.steps):
            # mutate one bar's row or col
            i = int(rng.integers(nb)); pl2 = [list(p) for p in cur]
            if rng.integers(2):
                pl2[i][0] = int(np.clip(pl2[i][0] + rng.integers(-2, 3), 3, 12))
            else:
                pl2[i][1] = int(np.clip(pl2[i][1] + rng.integers(-3, 4), 0, M - widths[i]))
            pl2 = [tuple(p) for p in pl2]
            if not valid(pl2, widths, target, wall, spouts):
                continue
            r2 = score(pl2); evals += 1
            if fitness(r2) > cf:
                cur, cf = pl2, fitness(r2)
                if r2["win"]:
                    win_cfg = cur; break
        if cf > best_fit:
            best_fit, best_overall = cf, cur
        if win_cfg is not None:
            break

    rbest = score(win_cfg if win_cfg else best_overall)
    report = {"evals": evals, "n_targets": ntgt, "bar_widths": widths,
              "bars_default": [(b["row0"], b["col0"]) for b in bars],
              "best_fitness": {"win": rbest["win"], "n_filled": rbest["n_filled"],
                               "n_wall": rbest["n_wall"], "filled": rbest["filled"]},
              "win_config": [list(p) for p in win_cfg] if win_cfg else None,
              "best_config": [list(p) for p in (win_cfg or best_overall)],
              "verdict": "GRID WIN FOUND" if win_cfg else "no clean win found (grid search)"}
    print(json.dumps(report, indent=1))
    if a.out:
        from pathlib import Path
        out_p = Path(a.out)
        if not out_p.is_absolute():
            out_p = FIXTURES / out_p.name
        out_p.write_text(json.dumps(report, indent=1))
        print(f"[grid] wrote {out_p}")


if __name__ == "__main__":
    main()
