"""Harvest an ON-DISTRIBUTION pour shard for the Stage-B win-probe.

FINDING 18: the win-probe trained on the old harvested shard hits 0.79 held-out but
0.58 on FRESH arrangements — it never saw the proposer's arrangement distribution.
Fix: collect pours drawn from the SAME _candidates generator (structured shifts +
random clicks) the gate/CEM uses, env-grade each, and save (arrangement, settled,
env_win). Training the probe on this distribution is the honest fix for the shift.

Legal: arrangements are random/structured, outcomes are graded by the real env
(levels_completed), no answers or route lineage injected.

    python3 synthesis_lab/orchestration/harvest_pour_shard.py \
        --levels 0 --seeds 60 --random-cands 40 --out <data>/pour_ondist_l0.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_LAB = Path(__file__).resolve().parents[1]
_REPO = _LAB.parent
for p in (_LAB, _LAB / "orchestration", _LAB / "experiments" / "megarexus_cnn"):
    sys.path.insert(0, str(p))

from imagine_search_gate import (  # noqa: E402
    _present, _candidates, _apply_real, _grid,
    GAME_ID, ACTION_OF_ID, game_live,
)
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

PAINT = 6


def _settle_from_pour(env_r, fdr, curr, base_lv, max_ticks=60):
    """Pour for real, recording frames; return (settled_frame, env_win). Settled =
    the max-paint plateau frame (same rule as pour_settle.settle_index)."""
    frames = [curr]
    win = False
    if game_live(fdr):
        for _ in range(max_ticks):
            fdr = env_r.step(ACTION_OF_ID[5])
            if fdr is None:
                break
            g = _grid(fdr)
            frames.append(g)
            if int(getattr(fdr, "levels_completed", 0) or 0) > base_lv:
                win = True
                break
            if not game_live(fdr):
                break
    n6 = np.array([int((f == PAINT).sum()) for f in frames])
    settled = frames[int(np.argmax(n6))]
    return settled.astype(np.int8), win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="0")
    ap.add_argument("--seeds", type=int, default=60)
    ap.add_argument("--base-seed", type=int, default=5000)
    ap.add_argument("--random-cands", type=int, default=40)
    ap.add_argument("--max-comps", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    mags = (1, 2, 3)
    arcade = Arcade(arc_api_key="anonymous",
                    environments_dir=str((_REPO / "environment_files").resolve()),
                    operation_mode=OperationMode.OFFLINE)
    records = []
    nwin = 0
    for level in [int(x) for x in a.levels.split(",") if x != ""]:
        for si in range(a.seeds):
            s = a.base_seed + int(level) * 1000 + si
            env = arcade.make(GAME_ID, seed=s)
            _, cur0 = _present(env, level)
            rng = np.random.default_rng(s)
            cands = _candidates(cur0, a.max_comps, mags, a.random_cands, rng)
            for arrangement in cands:
                env_i = arcade.make(GAME_ID, seed=s)
                fdi, curi = _present(env_i, level)
                fdi, arr, _ = _apply_real(env_i, fdi, curi, arrangement)
                if not game_live(fdi):
                    continue
                env_r = arcade.make(GAME_ID, seed=s)
                fdr, curr = _present(env_r, level)
                base_lv = int(getattr(fdr, "levels_completed", 0) or 0)
                fdr, curr, _ = _apply_real(env_r, fdr, curr, arrangement)
                settled, win = _settle_from_pour(env_r, fdr, curr, base_lv)
                records.append({"arrangement": np.asarray(arr, dtype=np.int8),
                                "settled": settled, "env_win": bool(win),
                                "level": int(level)})
                nwin += int(win)
            if (si + 1) % 10 == 0:
                print(f"[harvest] level {level} seed {si+1}/{a.seeds}: "
                      f"{len(records)} pours, {nwin} wins", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(records, a.out)
    print(f"[harvest] done. {len(records)} pours, {nwin} wins "
          f"({100*nwin/max(len(records),1):.1f}%) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
