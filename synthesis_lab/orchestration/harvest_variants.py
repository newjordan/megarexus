"""Harvest (arrangement, fill, lava, win) across MANY randomised SP80 variant boards
(NIGHT FINDING 29 — user's "move the obj blocks" idea) to train a shape-based, lava-aware
imagination that learns transferable routing SKILL instead of memorising one layout.

For each variant: build a randomised valid board, generate structured+random arrangements
(the proposer distribution), pour each with real CA physics, record env-graded outcomes.
Legal: varied PROBLEMS, no injected answers; every pour env-graded; real levels held out.

    python3 synthesis_lab/orchestration/harvest_variants.py \
        --variants 400 --arr-per 8 --out <data>/pour_variants.pt
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch

_LAB = Path(__file__).resolve().parents[1]; _REPO = _LAB.parent
for p in (_LAB / "orchestration", _LAB / "experiments" / "megarexus_cnn"):
    sys.path.insert(0, str(p))
from imagine_search_gate import _candidates, _apply_real, _grid, ACTION_OF_ID, game_live  # noqa: E402
from sp80_variants import make_env_variant, SRC_TAG  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402


def pour_record(env, cur, base_wall, ntarg):
    fd = env.step(ACTION_OF_ID[5])
    if fd is None:
        return None
    won = int(getattr(fd, "levels_completed", 0) or 0) > 0
    f = fd.frame
    frames = [np.asarray(x, dtype=np.int8) for x in f] if isinstance(f, list) else [np.asarray(f, dtype=np.int8)]
    n13 = [int((x == 13).sum()) for x in frames]
    sus = 0
    for i in range(len(n13) - 2):
        sus = max(sus, min(n13[i:i + 3]))
    lava = max(int((x == 14).sum()) for x in frames) - base_wall
    return sus / max(ntarg, 1), int(lava), bool(won)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=400)
    ap.add_argument("--arr-per", type=int, default=8)
    ap.add_argument("--base-seed", type=int, default=20000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((_REPO / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    recs, nwin, nvar = [], 0, 0
    for v in range(a.variants):
        # per-variant deterministic rng so the outer scan and each per-pour rebuild
        # produce the SAME layout (make_env_variant consumes rng to place objects).
        env, g = make_env_variant(arc, seed=a.base_seed + v, rng=np.random.default_rng(a.base_seed + v))
        if env is None:
            continue
        nvar += 1
        ntarg = int((g == 11).sum()); base_wall = int((g == 14).sum())
        crng = np.random.default_rng(a.base_seed + v)
        cands = _candidates(g, 6, (1, 2, 3), a.arr_per, crng)
        # cap arrangements per variant to keep breadth over variants
        for arrangement in cands[:a.arr_per + 3]:
            try:
                # fresh variant env per pour (rebuild the same layout deterministically)
                env2, g2 = make_env_variant(arc, seed=a.base_seed + v, rng=np.random.default_rng(a.base_seed + v))
                if env2 is None:
                    break
                fd = env2.step(ACTION_OF_ID[1]); cur = _grid(fd)
                fd, cur, _ = _apply_real(env2, fd, cur, arrangement)
                if not game_live(fd):
                    continue
                arr = cur.astype(np.int8)
                out = pour_record(env2, cur, base_wall, ntarg)
            except (ValueError, AttributeError, RuntimeError):
                continue  # degenerate variant (e.g. pour exceeds frame budget) -> skip
            if out is None:
                continue
            fill, lava, won = out
            recs.append({"arrangement": arr, "fill": float(fill), "lava": int(lava),
                         "env_win": bool(won), "n_targets": ntarg, "variant": v})
            nwin += int(won)
        if (v + 1) % 50 == 0:
            print(f"[var] {v+1}/{a.variants} variants ({nvar} valid): {len(recs)} pours, {nwin} wins", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(recs, a.out)
    print(f"[var] done. {len(recs)} pours over {nvar} variants, {nwin} wins "
          f"({100*nwin/max(len(recs),1):.1f}%) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
