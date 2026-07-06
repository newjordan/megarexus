"""DW round-3 flywheel — solve-time pours become observation data.

The 5-seed sweep showed the round-2 kernel's fitness plateaus: it imagines
lava 0.76-0.84 on candidates that env-verify as CLEAN WINS, so top-K selection
is a coin flip among tied candidates. This script replays every sweep top-K
candidate at its own seed (legal env interaction) and compares the REAL pour
against the round-2 IMAGINED rollout, per candidate:

  real:     paint-cell trajectory, final paint location (in-mouth / elsewhere),
            lava flash (px14 growth above pre baseline), env win
  imagined: fills, floor_touch, disposed, settled in/out of mouths

The point is to let observation answer: where does the real excess paint go
on a winning pour that the kernel wrongly sends to the floor?

    python3 r3_flywheel.py
Writes fixtures/r3_flywheel_20260706.json + a per-pour npz corpus
(r3_solvetime_pours_20260706.npz, harvest schema + genomes/seeds/levels).
"""
from __future__ import annotations

import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, LAB

from imagine_search_gate import _present, GAME_ID  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from scene import extract_scene, PAINT, WALL, M, CELL  # noqa: E402
from perception_gate import gate_scene  # noqa: E402
from imagination_kernel import KnownMap, FallingMaterial, run  # noqa: E402
from harvest_pours import pour_once  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

ROUND2 = {"deflect": "none", "spread": "lateral", "mouth": "entry",
          "edge": "dispose", "cell_cap": 3.0}
SEEDS = [1234, 7, 42, 101, 2026]
LEVELS = [0, 1]


def macro_paint_cells(frame):
    """Per-macro-cell paint pixel count / 16 -> [16,16] float occupancy."""
    f = np.asarray(frame)
    m = (f == PAINT).astype(np.float32)
    return m.reshape(M, CELL, M, CELL).sum(axis=(1, 3)) / (CELL * CELL)


def analyze_real(sc, frames):
    """Observed pour: paint volume trajectory + final location + lava flash."""
    base14 = int((sc.pre == WALL).sum())
    cells = [float((np.asarray(f) == PAINT).sum()) / (CELL * CELL) for f in frames]
    lava = any(int((np.asarray(f) == WALL).sum()) - base14 >= 100 for f in frames)
    fin = np.flipud(frames[-1]) if sc.flipped else frames[-1]
    occ = macro_paint_cells(fin)
    mouth_cells = {c for t in sc.targets for c in t.mouth}
    in_mouth = float(sum(occ[y, x] for (y, x) in mouth_cells))
    total = float(occ.sum())
    per_target = [float(sum(occ[y, x] for (y, x) in t.mouth)) for t in sc.targets]
    return {"peak_cells": max(cells), "final_cells": total,
            "final_in_mouth": in_mouth, "final_elsewhere": total - in_mouth,
            "per_target_final": per_target, "lava_flash": bool(lava),
            "cells_trajectory": [round(c, 2) for c in cells]}


def analyze_imagined(sc, verdict):
    km = KnownMap.from_scene(sc, verdict)
    end = run(km, FallingMaterial(**ROUND2), max_ticks=200)[-1]
    mouth_cells = {c for t in sc.targets for c in t.mouth}
    settled = np.asarray(end.settled)
    in_mouth = float(sum(settled[y, x] for (y, x) in mouth_cells))
    return {"fills": list(end.fills), "capacities": list(km.capacities),
            "floor_touch": int(end.floor_touch), "disposed": int(end.disposed),
            "settled_in_mouth": in_mouth,
            "settled_elsewhere": float(settled.sum() - in_mouth),
            "emit_volume": float(km.emit_volume), "terminal": end.terminal}


def main():
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    rows, pres, frames_flat, lens, wins, provs = [], [], [], [], [], []
    for level in LEVELS:
        for seed in SEEDS:
            fx = FIXTURES / f"dw_solve_L{level}_s{seed}_20260706.json"
            if not fx.exists():
                continue
            d = json.loads(fx.read_text())
            env0 = arc.make(GAME_ID, seed=seed)
            _, c0 = _present(env0, level)
            centers = [(int(c["center_xy"][0]), int(c["center_xy"][1]))
                       for c in frame_components(c0)[:8]]
            for t in d["topk"]:
                genome = [tuple(g) for g in t["genome"]]
                r = pour_once(arc, seed, level, genome, centers)
                if r is None:
                    print(f"[r3] L{level} s{seed} r{t['rank']}: replay DEAD", flush=True)
                    continue
                pre, frames, won = r
                sc = extract_scene(pre)
                v = gate_scene(pre, sc)
                row = {"level": level, "seed": seed, "rank": t["rank"],
                       "fitness": t["fitness"], "env_win_sweep": t["env_win"],
                       "env_win_replay": bool(won), "trusted": bool(v["trusted"]),
                       "real": analyze_real(sc, frames),
                       "imagined": analyze_imagined(sc, v) if v["trusted"] else None}
                rows.append(row)
                pres.append(pre); frames_flat.extend(np.asarray(f, dtype=np.int8) for f in frames)
                lens.append(len(frames)); wins.append(bool(won))
                provs.append({"level": level, "seed": seed, "rank": t["rank"],
                              "genome": [list(g) for g in genome]})
                im = row["imagined"]
                print(f"[r3] L{level} s{seed} r{t['rank']} win={won} "
                      f"lava_real={row['real']['lava_flash']} "
                      f"peak={row['real']['peak_cells']:.1f} "
                      f"in_mouth={row['real']['final_in_mouth']:.1f} "
                      f"elsewhere={row['real']['final_elsewhere']:.1f} | "
                      f"img floor={im['floor_touch'] if im else '?'} "
                      f"disp={im['disposed'] if im else '?'} "
                      f"in_mouth={im['settled_in_mouth'] if im else '?'}", flush=True)

    out = FIXTURES / "r3_flywheel_20260706.json"
    out.write_text(json.dumps({
        "protocol": "replay of sweep top-K candidates at their own seeds; real-vs-"
                    "imagined comparison; observation data for round-3 selection",
        "round2": ROUND2, "rows": rows}, indent=1))
    np.savez_compressed(
        FIXTURES / "r3_solvetime_pours_20260706.npz",
        pre=np.stack(pres), frames_flat=np.stack(frames_flat),
        lens=np.asarray(lens, dtype=np.int32), wins=np.asarray(wins, dtype=bool))
    (FIXTURES / "r3_solvetime_pours_20260706.provenance.json").write_text(
        json.dumps({"pours": provs, "provenance": "legal env replay of sweep "
                    "top-K genomes (solve-time flywheel observation data)"}, indent=1))
    n_win = sum(1 for r in rows if r["env_win_replay"])
    print(f"[r3] {len(rows)} pours captured ({n_win} wins) -> {out}", flush=True)


if __name__ == "__main__":
    main()
