"""Multi-pour semantics probe (handoff §6 open question): do FILLED targets persist across
ACTION5 presses, enabling incremental multi-pour filling within the ~4-pour budget?

Method: sim-pick an arrangement that cleanly fills the MOST targets (no flood), realize it,
then press ACTION5 repeatedly WITHOUT rearranging, recording per-target filled(13) counts of
the settled board between pours. If the 13s persist/accumulate -> multi-pour is real and the
solve approach must model pour COUNT. If each pour resets -> single-pour only (L2 truly needs
a rare one-shot winner). EVAL-ONLY real level.

    python3 mp_probe.py --level 2
"""
from __future__ import annotations

import argparse

import numpy as np

import _paths  # noqa: F401
from _paths import LAB

from imagine_search_gate import (_present, _candidates, _apply_real, _grid,  # noqa: E402
                                 GAME_ID, ACTION_OF_ID, game_live)
from pour3d import imagine_pour  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402


def clean_fill_score(pre):
    """sim: number of targets that fill cleanly (fill>=0.85, lavan<0.5)."""
    out = imagine_pour(pre, lane="stream")
    t = out["labels"]["targets"]
    return sum(1 for x in t if x["fill"] >= 0.85 and x["lavan"] < 0.5), out["margin"]


def settled_grid(fd):
    f = fd.frame
    frames = ([np.asarray(x, dtype=np.int8) for x in f] if isinstance(f, list)
              else [np.asarray(f, dtype=np.int8)])
    return frames[-1], frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sample", type=int, default=150)
    a = ap.parse_args()
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    env = arc.make(GAME_ID, seed=a.seed)
    fd, g = _present(env, a.level)
    rng = np.random.default_rng(a.seed)
    cands = _candidates(g, 6, (1, 2, 3), a.sample, rng)
    idx = rng.choice(len(cands), size=min(a.sample, len(cands)), replace=False)

    # sim-pick the arrangement that cleanly fills the most targets
    best = None
    for ai in idx:
        env2 = arc.make(GAME_ID, seed=a.seed); fd2, cur = _present(env2, a.level)
        fd2, cur, _ = _apply_real(env2, fd2, cur, cands[int(ai)])
        if not game_live(fd2):
            continue
        pre = cur.astype(np.int8)
        nf, m = clean_fill_score(pre)
        if best is None or (nf, m) > (best[0], best[1]):
            best = (nf, m, cands[int(ai)])
    nf, m, arrangement = best
    ntgt_px = int((g == 11).sum())
    print(f"L{a.level}: best sim clean-fill = {nf} targets (margin {m:+.1f}); "
          f"total target-px {ntgt_px} (~{ntgt_px//16} macro). Now real multi-pour:")

    # realize + repeated ACTION5 WITHOUT rearranging; track filled(13) between pours
    env3 = arc.make(GAME_ID, seed=a.seed); fd3, cur = _present(env3, a.level)
    fd3, cur, _ = _apply_real(env3, fd3, cur, arrangement)
    base = int(getattr(fd3, "levels_completed", 0) or 0)
    prev13 = int((cur == 13).sum())
    for p in range(1, 6):
        if not game_live(fd3):
            print(f"  pour {p}: not alive (budget exhausted)"); break
        fd3 = env3.step(ACTION_OF_ID[5])
        if fd3 is None:
            print(f"  pour {p}: None"); break
        won = int(getattr(fd3, "levels_completed", 0) or 0) > base
        last, frames = settled_grid(fd3)
        peak13 = max(int((x == 13).sum()) for x in frames)
        settle13 = int((last == 13).sum())
        print(f"  pour {p}: won={won} | peak filled13={peak13} settled filled13={settle13} "
              f"(prev settled {prev13})  -> {'PERSISTS/ACCUMULATES' if settle13 > prev13 else 'reset (no carry)'}")
        prev13 = settle13
        if won:
            print(f"    -> WON after {p} pours"); break


if __name__ == "__main__":
    main()
