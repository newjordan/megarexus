"""Proposer diagnostic (Stage-C prep): does a real level have FINDABLE winners in the
`_candidates` reachable arrangement space, and does the (post-iter2) sim rank them on top?

Distinguishes the two walls (handoff §5):
  * ranker wall  -> winners exist but sim mis-ranks them (S1 territory)
  * PROPOSER wall -> ~no winners are reachable at all (Stage C territory)

Harvests N real env-graded arrangements for a level, saves any winners (pre-frames +
move sequences = seeds for a proposer), reports win-rate + ranker precision/recall.
EVAL-ONLY real level; every pour env-graded; no answer injection.

    python3 proposer_probe.py --level 2 --n 200 --out fixtures/proposer_l2.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, LAB

from imagine_search_gate import (_present, _candidates, _apply_real, _grid,  # noqa: E402
                                 GAME_ID, ACTION_OF_ID, game_live)
from pertarget_labels import pertarget_labels  # noqa: E402
from pour3d import imagine_pour  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402


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
    # sample up to n distinct candidates
    idx = rng.choice(len(cands), size=min(a.n, len(cands)), replace=False)

    wins, floods, poured = 0, 0, 0
    sim_m, env_w = [], []
    winner_frames, winner_moves = [], []
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
        lab = pertarget_labels(pre, frames, r=3)
        fills = [t["fill"] for t in lab["targets"]]; lava = lab["global"]["lava"]
        poured += 1; wins += int(won)
        floods += int(bool(fills and min(fills) >= 0.85 and lava))
        sim = imagine_pour(pre, lane="stream")
        sim_m.append(float(sim["margin"])); env_w.append(bool(won))
        if won:
            winner_frames.append(pre)
            winner_moves.append([list(m) for m in arrangement])
        if (k + 1) % 50 == 0:
            print(f"[probe] L{a.level} {k+1}/{len(idx)}: {poured} poured, {wins} wins, {floods} fill+flood", flush=True)

    sim_m = np.array(sim_m); env_w = np.array(env_w, dtype=bool)
    pred = sim_m > 0
    tp = int((pred & env_w).sum()); fp = int((pred & ~env_w).sum()); fn = int((~pred & env_w).sum())
    verdict = ("PROPOSER-blocked (≈0 reachable winners)" if wins == 0 else
               ("RANKER issue (winners exist, sim misses/mis-ranks)" if fn > 0 or (tp and fp > tp)
                else "reachable + rankable (search should solve with enough evals)"))
    report = {
        "level": a.level, "poured": poured, "wins": wins, "win_rate": wins / max(poured, 1),
        "fill_and_flood": floods,
        "ranker": {"tp": tp, "fp": fp, "fn": fn,
                   "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)},
        "verdict": verdict,
    }
    print(json.dumps(report, indent=1))
    if a.out:
        from pathlib import Path
        out = Path(a.out)
        if not out.is_absolute():
            out = FIXTURES / out.name
        np.savez_compressed(
            out,
            winner_frames=np.stack(winner_frames) if winner_frames else np.zeros((0, 64, 64), np.int8),
            winner_moves=json.dumps(winner_moves), sim_margin=sim_m, env_win=env_w,
            report=json.dumps(report))
        print(f"[probe] wrote {out} ({len(winner_frames)} winners saved)")


if __name__ == "__main__":
    main()
