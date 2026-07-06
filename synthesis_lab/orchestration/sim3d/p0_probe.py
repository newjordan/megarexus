"""P0 diagnostic (multi-source fill fidelity) — reproduce Correction B, quantify the fix.

For each L2 arrangement, records the FULL per-target picture so we can (a) confirm the
env fills targets the sim scores 0, and (b) see whether a transient droplet visits the
enclosed mouth cell exactly when the env credits the fill:

  env_fill[i]   : pertarget_labels(pre, REAL pour frames) — ground truth (13-recolor).
  sim_fill[i]   : pertarget_labels(pre, SIM rendered frames) = score_rollout labels
                  (SAME grader, SAME frame_components(pre) target order -> aligned).
  sim_native[i] : raw settled-based mouth_fill from raycast (the thing under test).
  peak_occ[i]   : max over ticks of occupancy (settled+falling) AT the enclosed mouth
                  cell — the transient-visit signal the enclosed-entry fix will latch on.

EVAL-ONLY real level; every pour env-graded; no answer injection. Mirrors proposer_probe.

    python3 p0_probe.py --level 2 --n 200 --out fixtures/p0_l2_diag.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, LAB

from imagine_search_gate import (_present, _candidates, _apply_real,  # noqa: E402
                                 GAME_ID, ACTION_OF_ID, game_live)
from pertarget_labels import pertarget_labels  # noqa: E402
from pour3d import imagine_pour  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

FILL_THRESH = 0.85   # the render/13-recolor threshold; env fill is ~binary at the glyph


def _peak_occ_at_mouths(out):
    """max occupancy ever seen at each target's enclosed mouth cell, sorted by mouth x
    (matches pertarget target order, which is frame_components scan order by x)."""
    scene = out["scene"]; snaps = out["result"]["snapshots"]
    tgts = sorted(scene.targets, key=lambda t: (min(x for _, x in t.mouth) if t.mouth else 0))
    peaks = []
    for t in tgts:
        if not t.mouth:
            peaks.append(0.0); continue
        p = 0.0
        for (occ, _mf, _lava) in snaps:
            for (my, mx) in t.mouth:
                p = max(p, float(occ[my, mx]))
        peaks.append(p)
    return peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    env = arc.make(GAME_ID, seed=a.seed)
    fd, g = _present(env, a.level)
    rng = np.random.default_rng(a.seed)
    cands = _candidates(g, 6, (1, 2, 3), a.n, rng)
    idx = rng.choice(len(cands), size=min(a.n, len(cands)), replace=False)

    rows = []   # per arrangement: dict with env/sim/native/peak per target + win
    poured = 0
    for k, ai in enumerate(idx):
        arrangement = cands[int(ai)]
        env2 = arc.make(GAME_ID, seed=a.seed)
        fd2, cur = _present(env2, a.level)
        base = int(getattr(fd2, "levels_completed", 0) or 0)
        fd2, cur, _ = _apply_real(env2, fd2, cur, arrangement)
        if not game_live(fd2):
            continue
        pre = cur.astype(np.int8)
        fd2 = env2.step(ACTION_OF_ID[5])
        won = int(getattr(fd2, "levels_completed", 0) or 0) > base
        f = fd2.frame
        frames = ([np.asarray(x, dtype=np.int8) for x in f] if isinstance(f, list)
                  else [np.asarray(f, dtype=np.int8)])
        env_lab = pertarget_labels(pre, frames, r=3)
        env_fill = [t["fill"] for t in env_lab["targets"]]

        out = imagine_pour(pre, lane="stream")
        sim_fill = [t["fill"] for t in out["labels"]["targets"]]
        sim_native = list(out["native"]["mouth_fill"])
        peak = _peak_occ_at_mouths(out)
        poured += 1
        rows.append({
            "env_fill": env_fill, "sim_fill": sim_fill, "sim_native": sim_native,
            "peak_occ": peak, "won": bool(won),
            "env_lava": int(env_lab["global"]["lava"]), "sim_lava": bool(out["native"]["lava"]),
        })
        if (k + 1) % 50 == 0:
            print(f"[p0] L{a.level} {k+1}/{len(idx)}: {poured} poured", flush=True)

    # --- per-(arrangement,target) fill agreement (GATE P0.2) ---
    n_pairs = env_filled_pairs = agree = agree_on_envfilled = 0
    sim_says0_env_says1 = []   # the Correction-B cells
    for r in rows:
        for i, ef in enumerate(r["env_fill"]):
            if i >= len(r["sim_fill"]):
                continue
            sf = r["sim_fill"][i]
            efb, sfb = ef >= FILL_THRESH, sf >= FILL_THRESH
            n_pairs += 1
            agree += int(efb == sfb)
            if efb:
                env_filled_pairs += 1
                agree_on_envfilled += int(sfb)
                if not sfb:
                    sim_says0_env_says1.append(
                        {"env": round(ef, 3), "sim": round(sf, 3),
                         "native": round(r["sim_native"][i] if i < len(r["sim_native"]) else -1, 3),
                         "peak_occ": round(r["peak_occ"][i] if i < len(r["peak_occ"]) else -1, 3)})
    wins = sum(int(r["won"]) for r in rows)
    report = {
        "level": a.level, "poured": poured, "wins": wins,
        "n_target_pairs": n_pairs, "env_filled_pairs": env_filled_pairs,
        "fill_agreement_overall": agree / max(n_pairs, 1),
        "fill_agreement_on_envfilled": agree_on_envfilled / max(env_filled_pairs, 1),
        "correction_b_cells": len(sim_says0_env_says1),
        "correction_b_examples": sim_says0_env_says1[:12],
    }
    print(json.dumps(report, indent=1))
    if a.out:
        from pathlib import Path
        out_p = Path(a.out)
        if not out_p.is_absolute():
            out_p = FIXTURES / out_p.name
        np.savez_compressed(out_p, rows=json.dumps(rows), report=json.dumps(report))
        print(f"[p0] wrote {out_p}")


if __name__ == "__main__":
    main()
