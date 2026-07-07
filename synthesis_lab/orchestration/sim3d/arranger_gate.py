"""Arranger gate — hold virtual_arranger to BYTE-IDENTICAL (modulo the budget
ring) against real env realization before any continuous-episode runner may
trust it.

Protocol (pre-registered):
  1. per (level, seed): present frame, centers[:10] (dw_solve convention);
     N random genomes from dw_solve's search distribution, action-cost-capped
     so real realization survives the level's action budget
  2. realize each genome REAL (fresh env, offline validation lane — resets are
     legal here, this is observation) and VIRTUAL (virtual_arranger from the
     same present frame); compare all non-ring cells
  3. kernel-equivalence subsample: r4 kernel_score fitness on real vs virtual
     candidate frames must agree (the search must rank identically)
  4. post-pour phase: arrange + failed pour, then a SECOND genome from the
     observed post-pour live frame, real-vs-virtual (re-plan states). For L5
     overlap states, the virtual board carries unit-mask continuity from the
     pre-pour arrangement and rebases on the live frame; hidden 8/15 cells are
     not cold-parseable from pixels alone.
PASS = zero mismatches (refusals are allowed and counted).

    python3 arranger_gate.py --levels 0,1,2,3 --seeds 555,1234,42 --n 40
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, LAB

from imagine_search_gate import _present, _grid, _apply_real, GAME_ID, ACTION_OF_ID, game_live  # noqa: E402
from imag_search_solve import actions  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from virtual_arranger import VirtualBoard, realize_virtual, frames_equal, gauge_mask  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

MOVES = (1, 2, 3, 4)
# observed per-level action budgets (probe_costs 2026-07-06): present tick
# included; GAME_OVER when the ring gauge empties
BUDGETS = {0: 30, 1: 45, 2: 100, 3: 120}


def rand_genome(rng, n_centers, max_genes=8, budget=None):
    """dw_solve search distribution, cost-capped to survive realization."""
    while True:
        k = int(rng.integers(1, max_genes + 1))
        g = [(int(rng.integers(n_centers)), int(rng.choice(MOVES)),
              int(rng.integers(1, 9))) for _ in range(k)]
        cost = sum(1 + m for (_, _, m) in g)
        if budget is None or cost <= budget:
            return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="0,1,2,3")
    ap.add_argument("--seeds", default="555,1234,42")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--kernel-sub", type=int, default=6,
                    help="kernel-equivalence checks per (level, seed)")
    ap.add_argument("--postpour", type=int, default=4,
                    help="post-pour re-plan checks per (level, seed)")
    ap.add_argument("--out", default="arranger_gate_20260706.json")
    a = ap.parse_args()

    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)

    import dw_solve
    fx = json.loads((FIXTURES / dw_solve.MODEL_FIXTURES["r4"]).read_text())
    dw_solve.MODEL.update(
        family=fx["selected"].get("family", "falling_material"),
        cfg=fx["selected"]["cfg"], rule=fx["selected"]["rule"],
        stop=fx["selected"]["stop"],
        measured_volumes=fx.get("measured_volumes", {}))

    report = {"protocol": __doc__.split("\n\n")[1],
              "cells_compared": "all except budget-gauge border sides", "results": {}}
    total_mm = 0
    for level in [int(x) for x in a.levels.split(",")]:
        for seed in [int(x) for x in a.seeds.split(",")]:
            env0 = arc.make(GAME_ID, seed=seed)
            _, f0 = _present(env0, level)
            centers = [(int(c["center_xy"][0]), int(c["center_xy"][1]))
                       for c in frame_components(f0)[:10]]
            rng = np.random.default_rng(seed * 7 + level)
            # arrangement actions only; leave room for the present tick + margin
            act_budget = BUDGETS.get(level, 30) - 6

            stats = {"n": 0, "match": 0, "mismatch": 0, "virt_refused": 0,
                     "real_dead": 0, "kernel_agree": 0, "kernel_disagree": 0,
                     "postpour_match": 0, "postpour_mismatch": 0,
                     "postpour_refused": 0, "mismatch_examples": [],
                     "postpour_mismatch_examples": []}
            for i in range(a.n):
                g = rand_genome(rng, len(centers), budget=act_budget)
                seq = actions(g, centers)
                env = arc.make(GAME_ID, seed=seed)
                fd, cur = _present(env, level)
                fd, cur, _ = _apply_real(env, fd, cur, seq)
                if not game_live(fd):
                    stats["real_dead"] += 1
                    continue
                virt = realize_virtual(f0, seq)
                stats["n"] += 1
                if virt is None:
                    stats["virt_refused"] += 1
                    continue
                if frames_equal(cur, virt):
                    stats["match"] += 1
                    if stats["kernel_agree"] + stats["kernel_disagree"] < a.kernel_sub:
                        sr = dw_solve.kernel_score(cur.astype(np.int8), None, level=level)
                        sv = dw_solve.kernel_score(virt.astype(np.int8), None, level=level)
                        fr = None if sr is None else round(sr["fitness"], 6)
                        fv = None if sv is None else round(sv["fitness"], 6)
                        stats["kernel_agree" if fr == fv else "kernel_disagree"] += 1
                else:
                    stats["mismatch"] += 1
                    keep = ~(gauge_mask(cur) | gauge_mask(virt))
                    diff = int((np.asarray(cur)[keep]
                                != np.asarray(virt)[keep]).sum())
                    if len(stats["mismatch_examples"]) < 5:
                        stats["mismatch_examples"].append(
                            {"genome": [list(x) for x in g], "diff_cells": diff})

            # post-pour re-plan states
            rng2 = np.random.default_rng(seed * 13 + level + 1)
            for i in range(a.postpour):
                g1 = rand_genome(rng2, len(centers), budget=max(6, act_budget // 2))
                g2 = rand_genome(rng2, len(centers), budget=max(6, act_budget // 3))
                env = arc.make(GAME_ID, seed=seed)
                fd, cur = _present(env, level)
                base = int(getattr(fd, "levels_completed", 0) or 0)
                vpre = VirtualBoard.parse(cur)
                if vpre is not None:
                    try:
                        vpre.apply(actions(g1, centers))
                    except Exception:  # noqa: BLE001
                        vpre = None
                fd, cur, _ = _apply_real(env, fd, cur, actions(g1, centers))
                if not game_live(fd):
                    continue
                try:
                    fd = env.step(ACTION_OF_ID[5])
                except Exception:  # noqa: BLE001
                    continue
                if fd is None or not game_live(fd):
                    continue
                if int(getattr(fd, "levels_completed", 0) or 0) > base:
                    continue  # accidental win — not a re-plan state
                live = _grid(fd)
                seq2 = actions(g2, centers)
                fd, cur, _ = _apply_real(env, fd, live, seq2)
                if not game_live(fd):
                    continue
                vbase = vpre.rebase(live) if (level >= 5 and vpre is not None) else live
                virt = realize_virtual(vbase, seq2)
                if virt is None:
                    stats["postpour_refused"] += 1
                    continue
                key = "postpour_match" if frames_equal(cur, virt) else "postpour_mismatch"
                stats[key] += 1
                if key == "postpour_mismatch" and len(stats["postpour_mismatch_examples"]) < 5:
                    keep = ~(gauge_mask(cur) | gauge_mask(virt))
                    diff = int((np.asarray(cur)[keep]
                                != np.asarray(virt)[keep]).sum())
                    stats["postpour_mismatch_examples"].append({
                        "pre_pour_genome": [list(x) for x in g1],
                        "replan_genome": [list(x) for x in g2],
                        "diff_cells": diff,
                    })

            total_mm += stats["mismatch"] + stats["postpour_mismatch"] + stats["kernel_disagree"]
            report["results"][f"L{level}_s{seed}"] = stats
            print(f"[gate] L{level} s{seed}: {stats['match']}/{stats['n']} match, "
                  f"{stats['mismatch']} mismatch, {stats['virt_refused']} refused, "
                  f"{stats['real_dead']} real-dead | kernel {stats['kernel_agree']}✓"
                  f"/{stats['kernel_disagree']}✗ | postpour "
                  f"{stats['postpour_match']}✓/{stats['postpour_mismatch']}✗", flush=True)

    report["verdict"] = "PASS" if total_mm == 0 else f"FAIL ({total_mm} disagreements)"
    out = FIXTURES / a.out
    out.write_text(json.dumps(report, indent=1))
    print(f"[gate] {report['verdict']} -> {out}", flush=True)
    raise SystemExit(0 if total_mm == 0 else 2)


if __name__ == "__main__":
    main()
