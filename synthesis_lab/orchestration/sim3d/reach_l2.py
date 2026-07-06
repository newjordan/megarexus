"""Existential probe: does a REACHABLE L2 winner exist in the click-move action space?
Bigger, RICHER real harvest (2-10 clicks, magnitude 1-3) than proposer_probe's short
genomes. 0 wins in N rich arrangements => L2's solution likely isn't expressible from the
presented state (action-space limit, not search effort). Any win => seed + proof. EVAL-ONLY.

    python3 reach_l2.py --level 2 --n 600
"""
from __future__ import annotations
import argparse, json
import numpy as np
import _paths  # noqa: F401
from _paths import FIXTURES, LAB
from imagine_search_gate import _present, _grid, GAME_ID, ACTION_OF_ID, game_live  # noqa: E402
from imag_search_solve import frame_of, real_win, MOVES  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from pour3d import imagine_pour  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-clicks", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    env0 = arc.make(GAME_ID, seed=a.seed); _, c0 = _present(env0, a.level)
    centers = [(int(c["center_xy"][0]), int(c["center_xy"][1]))
               for c in frame_components(c0)[:8]]
    rng = np.random.default_rng(a.seed)

    def rich_g():
        k = int(rng.integers(2, a.max_clicks + 1))
        return [(int(rng.integers(len(centers))), int(rng.choice(MOVES)),
                 int(rng.integers(1, 4))) for _ in range(k)]

    wins, poured = 0, 0
    win_frames, win_genomes, win_sim = [], [], []
    for i in range(a.n):
        g = rich_g()
        cf = frame_of(arc, a.seed, a.level, g, centers)
        if cf is None:
            continue
        poured += 1
        won = real_win(arc, a.seed, a.level, g, centers)
        if won:
            wins += 1
            win_frames.append(cf.astype(np.int8))
            win_genomes.append([list(x) for x in g])
            win_sim.append(float(imagine_pour(cf.astype(np.int8), lane="stream")["margin"]))
        if (i + 1) % 100 == 0:
            print(f"[reach] L{a.level} {i+1}/{a.n}: {poured} poured, {wins} WINS", flush=True)
    print(json.dumps({"level": a.level, "poured": poured, "wins": wins,
                      "win_sim_margins": win_sim,
                      "verdict": ("reachable winner EXISTS" if wins else
                                  "no reachable winner in N rich arrangements")}, indent=1))
    if a.out and win_frames:
        from pathlib import Path
        out = Path(a.out); out = out if out.is_absolute() else FIXTURES / out.name
        np.savez_compressed(out, win_frames=np.stack(win_frames),
                            win_genomes=json.dumps(win_genomes), win_sim=np.array(win_sim))
        print(f"[reach] wrote {out} ({wins} winners)")


if __name__ == "__main__":
    main()
