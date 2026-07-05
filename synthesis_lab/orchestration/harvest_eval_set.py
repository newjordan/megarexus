"""Harvest a reusable (arrangement, env_win, sustained_fill, level) eval set for the
shape-alignment / spatial-probe generalisation tests (NIGHT FINDING 22-23).

Per-candidate: apply a structured+random arrangement, ONE step(5) (the whole pour
animation is fd.frame as a LIST), record env_win (levels_completed) and a
sequence-aware sustained_fill = max n13 HELD >=3 consecutive frames / #targets
(the honest progress signal; a transient overshoot peak scores low). Legal: env-graded.

    python3 synthesis_lab/orchestration/harvest_eval_set.py \
        --levels 0,1 --seeds 40 --random-cands 40 --out <data>/eval_shape_l0l1.pt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch

_LAB = Path(__file__).resolve().parents[1]; _REPO = _LAB.parent
for p in (_LAB, _LAB / "orchestration", _LAB / "experiments" / "megarexus_cnn"):
    sys.path.insert(0, str(p))
from imagine_search_gate import (_present, _candidates, _apply_real, _grid,
                                 GAME_ID, ACTION_OF_ID, game_live)  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402


def sustained_fill(frames, ntarg, K=3):
    n13 = [int((x == 13).sum()) for x in frames]
    best = 0
    for i in range(len(n13) - K + 1):
        best = max(best, min(n13[i:i + K]))
    return best / max(ntarg, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="0,1")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=8000)
    ap.add_argument("--random-cands", type=int, default=40)
    ap.add_argument("--max-comps", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((_REPO / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    recs = []
    for level in [int(x) for x in a.levels.split(",") if x != ""]:
        nwin = 0
        for si in range(a.seeds):
            s = a.base_seed + level * 1000 + si
            env = arc.make(GAME_ID, seed=s)
            _, c0 = _present(env, level)
            ntarg = int((c0 == 11).sum())
            rng = np.random.default_rng(s)
            cands = _candidates(c0, a.max_comps, (1, 2, 3), a.random_cands, rng)
            for arr_seq in cands:
                er = arc.make(GAME_ID, seed=s)
                fdr, cur = _present(er, level)
                base = int(getattr(fdr, "levels_completed", 0) or 0)
                fdr, cur, _ = _apply_real(er, fdr, cur, arr_seq)
                if not game_live(fdr):
                    continue
                arr = cur.astype(np.int8)
                base_wall = int((c0 == 14).sum())
                fdr = er.step(ACTION_OF_ID[5])
                won = int(getattr(fdr, "levels_completed", 0) or 0) > base
                f = fdr.frame
                frames = [np.asarray(x, dtype=np.int8) for x in f] if isinstance(f, list) \
                    else [np.asarray(f, dtype=np.int8)]
                lava = max(int((x == 14).sum()) for x in frames) - base_wall
                recs.append({"arrangement": arr, "env_win": bool(won),
                             "sustained_fill": float(sustained_fill(frames, ntarg)),
                             "lava": int(lava), "n_targets": ntarg, "level": level})
                nwin += int(won)
            if (si + 1) % 10 == 0:
                print(f"[eval] L{level} seed {si+1}/{a.seeds}: {len(recs)} recs, L{level} wins so far {nwin}", flush=True)
        print(f"[eval] level {level}: {nwin} env wins", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(recs, a.out)
    print(f"[eval] done. {len(recs)} recs -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
