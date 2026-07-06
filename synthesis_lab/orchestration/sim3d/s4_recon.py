"""SIM3D stage S4 — level reconnaissance (execution plan §5.S4).

For each real level L0..L5: extract the scene, run the DEFAULT pour for real
(env-graded) and in the sim, and check the S4 gate — default-pour sim margin sign
matches the env outcome. Flags geometry the extractor mishandles (color-15 redirect
blocks, sidewall HUD). EVAL-ONLY reads of real levels; no answer injection.

    python3 s4_recon.py            # table over all levels
    python3 s4_recon.py --show 3   # + ASCII macro grid of level 3's present frame
"""
from __future__ import annotations

import argparse

import numpy as np

import _paths  # noqa: F401
from _paths import LAB

from imagine_search_gate import _present, _grid, GAME_ID, ACTION_OF_ID, game_live  # noqa: E402
from scene import extract_scene, to_macro  # noqa: E402
from pertarget_labels import pertarget_labels  # noqa: E402
from pour3d import imagine_pour  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

SYM = {0: ".", 12: ".", 1: "_", 4: "S", 6: "o", 11: "U", 13: "#", 14: "X",
       8: "B", 9: "b", 15: "R"}


def show_macro(mac, title):
    print(title)
    for row in mac:
        print("".join(SYM.get(int(c), "?") for c in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--levels", default="0,1,2,3,4,5")
    ap.add_argument("--show", type=int, default=None, help="also ASCII-dump this level's present")
    a = ap.parse_args()
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)

    print(f"{'lvl':>3} {'orient':>10} {'flip':>4} {'ntgt':>4} {'mouths':>16} {'floor':>6} "
          f"{'c15':>4} | {'env':>10} {'sim_m':>5} {'sim_ok':>6} {'sign?':>5}")
    for lvl in [int(x) for x in a.levels.split(",")]:
        env = arc.make(GAME_ID, seed=a.seed)
        fd, cur = _present(env, lvl)
        g = cur
        c15 = int((g == 15).sum())
        # extract
        try:
            sc = extract_scene(g)
            mouths = [len(t.mouth) for t in sc.targets]
            flip = sc.flipped; ntgt = len(sc.targets); floor = sc.floor_rows
            extract_ok = True
        except Exception as e:  # noqa: BLE001
            print(f"L{lvl}: extract_scene FAILED: {type(e).__name__}: {e}")
            continue
        src_rows = np.nonzero((g == 4).any(axis=1))[0]; tgt_rows = np.nonzero((g == 11).any(axis=1))[0]
        orient = "up/rot" if src_rows.mean() > tgt_rows.mean() else "down"
        # default pour for real
        env2 = arc.make(GAME_ID, seed=a.seed)
        fd2, cur2 = _present(env2, lvl)
        base = int(getattr(fd2, "levels_completed", 0) or 0)
        fd2 = env2.step(ACTION_OF_ID[5])
        won = int(getattr(fd2, "levels_completed", 0) or 0) > base
        f = fd2.frame
        frames = ([np.asarray(x, dtype=np.int8) for x in f] if isinstance(f, list)
                  else [np.asarray(f, dtype=np.int8)])
        rlab = pertarget_labels(cur2.astype(np.int8), frames, r=3)
        rmarg = min((t["fill"] - t["lavan"] for t in rlab["targets"]), default=0.0)
        env_desc = f"{'WIN' if won else 'loss'} m={rmarg:+.1f}"
        # default pour in sim
        sim = imagine_pour(cur2.astype(np.int8), lane="stream")
        sign = "OK" if (sim["margin"] > 0) == won else "MISS"
        print(f"{lvl:>3} {orient:>10} {str(flip):>4} {ntgt:>4} {str(mouths):>16} {str(floor):>6} "
              f"{c15:>4} | {env_desc:>10} {sim['margin']:>+5.1f} {str(sim['winner_ok']):>6} {sign:>5}")

    if a.show is not None:
        env = arc.make(GAME_ID, seed=a.seed)
        fd, cur = _present(env, a.show)
        show_macro(to_macro(cur), f"\n--- L{a.show} present (macro) ---")


if __name__ == "__main__":
    main()
