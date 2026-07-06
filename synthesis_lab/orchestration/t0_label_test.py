"""T0 — DECISIVE LABEL TEST (STAGE_PT_SPEC §5.T0). Local, no training.

Prove the PER-TARGET ground truth (pertarget_labels §3) separates the verified L1
winner from v52's false positives, BEFORE spending any training compute. A clean
FAIL, written up, is a successful delivery — the decomposition hypothesis would be
dead at the labels (a valid, valuable result).

Flow:
  0. Reproduce the verified L1 winner genome (env-confirmed) and its per-target
     labels. HARD GATE: it must env-win here, else the seed/centers are wrong.
  1. Regenerate v52's top-4 false positives via imag_search_solve's own machinery
     (v52 scoring, hill-climb), REAL-pour each, confirm env LOSS, per-target labels.
  2. PASS iff: winner fill_t>=0.95 AND lavan_t<=0.2 for ALL targets; AND every FP
     violates one of those on >=1 target. Sanity: winner n_targets==3.

    python3 synthesis_lab/orchestration/t0_label_test.py \
        --ckpt <v52>/best.pt --level 1 --seed 1234 --restarts 40 --steps 80 \
        --out synthesis_lab/experiments/megarexus_cnn/runs/t0_pertarget_<date>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_LAB = Path(__file__).resolve().parents[1]
_REPO = _LAB.parent
for _p in (_LAB, _LAB / "orchestration", _LAB / "experiments" / "megarexus_cnn"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from imagine_search_gate import _present, _apply_real, _grid, GAME_ID, ACTION_OF_ID, game_live  # noqa: E402
from imag_search_solve import V48, actions, frame_of, MOVES  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from pertarget_labels import pertarget_labels, winner_ok  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

# §2 ground fact — env-confirmed verified L1 winner genome (240/240 filled, lava=1).
WINNER_GENOME = [(1, 1, 2), (6, 4, 2), (4, 1, 3), (1, 1, 3), (3, 4, 3), (5, 1, 1), (7, 2, 3)]


def pour_and_label(arc, seed, level, genome, centers, r=3):
    """Real-pour a genome and extract §3 per-target labels. Single ACTION5 press =
    whole animation (fd.frame is the frame LIST, mirrors harvest_variants)."""
    env = arc.make(GAME_ID, seed=seed)
    fd, cur = _present(env, level)
    base = int(getattr(fd, "levels_completed", 0) or 0)
    fd, cur, _ = _apply_real(env, fd, cur, actions(genome, centers))
    if not game_live(fd):
        return None
    pre = cur.astype(np.int8)
    fd = env.step(ACTION_OF_ID[5])
    if fd is None:
        return None
    won = int(getattr(fd, "levels_completed", 0) or 0) > base
    f = fd.frame
    frames = [np.asarray(x, np.int8) for x in f] if isinstance(f, list) else [np.asarray(f, np.int8)]
    lab = pertarget_labels(pre, frames, r=r)
    lab["env_win"] = bool(won)
    lab["genome"] = [list(g) for g in genome]
    return lab


def search_v52_fps(arc, v, seed, level, centers, restarts, steps, max_genes, topk):
    """Reproduce imag_search_solve's global top-K by v52 imagined score (the FPs)."""
    rng = np.random.default_rng(seed + level)

    def rand_g():
        k = int(rng.integers(1, max_genes + 1))
        return [(int(rng.integers(len(centers))), int(rng.choice(MOVES)), int(rng.integers(1, 4))) for _ in range(k)]

    def mutate(g):
        g = [list(x) for x in g]; op = rng.integers(4)
        if op == 0 and g: i = rng.integers(len(g)); g[i][1] = int(rng.choice(MOVES))
        elif op == 1 and g: i = rng.integers(len(g)); g[i][2] = int(rng.integers(1, 4))
        elif op == 2 and len(g) < max_genes: g.append([int(rng.integers(len(centers))), int(rng.choice(MOVES)), int(rng.integers(1, 4))])
        elif g and len(g) > 1: del g[int(rng.integers(len(g)))]
        return [tuple(x) for x in g]

    found = []
    for r in range(restarts):
        cur = rand_g()
        cf = frame_of(arc, seed, level, cur, centers)
        cs = v.score_frame(cf) if cf is not None else -1e9
        for _ in range(steps):
            cand = mutate(cur)
            cff = frame_of(arc, seed, level, cand, centers)
            if cff is None:
                continue
            s = v.score_frame(cff)
            if s > cs:
                cur, cs = cand, s
        found.append((cs, cur))
        print(f"[t0] restart {r+1}/{restarts} best_score={cs:.4f}", flush=True)
    found.sort(key=lambda x: -x[0])
    seen, topk_out = set(), []
    for s, g in found:
        key = tuple(sorted(g))
        if key in seen:
            continue
        seen.add(key); topk_out.append((float(s), g))
        if len(topk_out) >= topk:
            break
    return topk_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--max-genes", type=int, default=8)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--r", type=int, default=3, help="§3 lava dilation radius (ablation: 2 or 5)")
    ap.add_argument("--fill-min", type=float, default=0.95)
    ap.add_argument("--lavan-max", type=float, default=0.2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = a.device
    v = V48(a.ckpt, dev, a.lam)
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((_REPO / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)

    env0 = arc.make(GAME_ID, seed=a.seed)
    _, c0 = _present(env0, a.level)
    comps = frame_components(c0)
    centers = [(int(c["center_xy"][0]), int(c["center_xy"][1])) for c in comps[:8]]
    print(f"[t0] ckpt={a.ckpt} level={a.level} seed={a.seed} R={a.r} "
          f"n_components={len(comps)} centers={len(centers)}", flush=True)

    result = {"level": a.level, "seed": a.seed, "R": a.r, "lam": a.lam,
              "fill_min": a.fill_min, "lavan_max": a.lavan_max, "ckpt": a.ckpt}

    # --- 0. Verified winner (HARD GATE) --------------------------------------
    win = pour_and_label(arc, a.seed, a.level, WINNER_GENOME, centers, r=a.r)
    if win is None:
        result["error"] = "winner arrangement not game_live — seed/centers mismatch"
        Path(a.out).write_text(json.dumps(result, indent=2))
        print(f"[t0] FATAL: {result['error']}", flush=True)
        print("[t0] VERDICT=FATAL", flush=True)
        return
    result["winner"] = win
    print(f"[t0] WINNER env_win={win['env_win']} n_targets={win['n_targets']}", flush=True)
    for i, t in enumerate(win["targets"]):
        print(f"[t0]   winner U{i}: fill={t['fill']:.3f} lavan={t['lavan']:.3f} "
              f"sus={t['sus']} cells={t['cells']} lava={t['lava']}", flush=True)
    if not win["env_win"]:
        result["error"] = "winner genome did NOT env-win at this seed — cannot proceed"
        Path(a.out).write_text(json.dumps(result, indent=2))
        print(f"[t0] FATAL: {result['error']}", flush=True)
        print("[t0] VERDICT=FATAL", flush=True)
        return

    # --- 1. v52 false positives ----------------------------------------------
    print(f"[t0] searching v52 top-{a.topk} FPs ({a.restarts}x{a.steps} imagined)...", flush=True)
    fps = search_v52_fps(arc, v, a.seed, a.level, centers, a.restarts, a.steps, a.max_genes, a.topk)
    print(f"[t0] v52 top-{a.topk} imagined scores = {[round(s,3) for s,_ in fps]}", flush=True)
    result["fp_scores"] = [s for s, _ in fps]
    fp_records = []
    for rank, (s, g) in enumerate(fps):
        lab = pour_and_label(arc, a.seed, a.level, g, centers, r=a.r)
        if lab is None:
            fp_records.append({"rank": rank + 1, "score": s, "skipped": True})
            continue
        lab["score"] = s
        lab["rank"] = rank + 1
        ok, reasons = winner_ok(lab, a.fill_min, a.lavan_max)
        lab["passes_winner_predicate"] = ok  # a real FP must be FALSE here
        lab["violations"] = reasons
        fp_records.append(lab)
        print(f"[t0] FP#{rank+1} score={s:.3f} env_win={lab['env_win']} "
              f"n_targets={lab['n_targets']} passes_winner_pred={ok}", flush=True)
        for i, t in enumerate(lab["targets"]):
            print(f"[t0]    U{i}: fill={t['fill']:.3f} lavan={t['lavan']:.3f}", flush=True)
    result["false_positives"] = fp_records

    # --- 2. PASS/FAIL --------------------------------------------------------
    w_ok, w_reasons = winner_ok(win, a.fill_min, a.lavan_max)
    result["winner_passes"] = w_ok
    result["winner_violations"] = w_reasons
    result["winner_n_targets_ok"] = (win["n_targets"] == 3)

    # every FP that is a genuine env-loss must FAIL the winner predicate
    real_fp = [r for r in fp_records if not r.get("skipped") and not r.get("env_win", False)]
    fp_separated = all(not r["passes_winner_predicate"] for r in real_fp)
    any_fp_is_win = any(r.get("env_win", False) for r in fp_records if not r.get("skipped"))
    result["n_real_fps"] = len(real_fp)
    result["fps_separated"] = fp_separated
    result["any_fp_actually_won"] = any_fp_is_win

    verdict = "PASS" if (w_ok and result["winner_n_targets_ok"] and fp_separated and not any_fp_is_win) else "FAIL"
    result["verdict"] = verdict
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"[t0] winner_passes={w_ok} (violations={w_reasons}) n_targets_ok={result['winner_n_targets_ok']}", flush=True)
    print(f"[t0] fps_separated={fp_separated} n_real_fps={len(real_fp)} any_fp_won={any_fp_is_win}", flush=True)
    print(f"[t0] wrote {a.out}", flush=True)
    print(f"[t0] VERDICT={verdict}", flush=True)


if __name__ == "__main__":
    main()
