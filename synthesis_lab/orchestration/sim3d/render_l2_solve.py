"""Render the env-confirmed L2 winning pour to an MP4: replay the block moves, then the pour."""
from __future__ import annotations

import json
import numpy as np
import cv2

import _paths  # noqa: F401
from _paths import LAB, FIXTURES
from imagine_search_gate import _present, _grid, GAME_ID, ACTION_OF_ID, game_live
from imag_search_solve import actions
from engine_adapters.portband_glove_adapter import frame_components
from arc_agi import Arcade
from arc_agi.base import OperationMode

PAL = np.array([
    [0, 0, 0], [64, 64, 72], [128, 0, 128], [0, 128, 128], [230, 180, 40],
    [200, 200, 200], [90, 160, 255], [255, 255, 255], [160, 110, 60],
    [240, 140, 40], [120, 200, 120], [60, 60, 200], [24, 24, 28],
    [80, 220, 100], [220, 60, 60], [255, 0, 255],
], dtype=np.uint8)

SCALE = 12
OUT = str(FIXTURES / "L2_solve.mp4")


def rgb(f):
    return PAL[np.clip(np.asarray(f, np.int16), 0, 15)]


def up(f):
    img = rgb(f)
    img = np.kron(img, np.ones((SCALE, SCALE, 1), np.uint8))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def main():
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    genome = [tuple(x) for x in json.load(open(FIXTURES / "p2_droplet_solve.json"))["genome"]]
    env = arc.make(GAME_ID, seed=1234)
    fd, c0 = _present(env, 2)
    centers = [(int(c["center_xy"][0]), int(c["center_xy"][1])) for c in frame_components(c0)[:8]]
    base = int(getattr(fd, "levels_completed", 0) or 0)

    seq = []                                   # list of (frame, n_hold)
    seq.append((_grid(fd), 18))                # initial board
    for (aid, x, y) in actions(genome, centers):
        if not game_live(fd):
            break
        data = {"x": int(x), "y": int(y)} if aid == 6 else None
        fd = env.step(ACTION_OF_ID[aid], data=data)
        seq.append((_grid(fd), 2 if aid != 6 else 4))   # linger on selects
    seq.append((_grid(fd), 20))                # arranged board, held

    fd = env.step(ACTION_OF_ID[5])             # POUR
    won = int(getattr(fd, "levels_completed", 0) or 0) > base
    pf = fd.frame
    pour = ([np.asarray(x, np.int8) for x in pf] if isinstance(pf, list)
            else [np.asarray(pf, np.int8)])
    # the win frame = most filled mouths with no floor-flash (drop the post-win reset tail)
    win_i = max(range(len(pour)),
                key=lambda i: (int((pour[i] == 13).sum()), -int((pour[i] == 14).sum())))
    for i, f in enumerate(pour[:win_i + 1]):
        seq.append((f, 3))
    seq.append((pour[win_i], 36))              # hold the WIN (all 3 mouths filled, no flood)

    h, w = 64 * SCALE, 64 * SCALE
    vw = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), 15, (w, h))
    total = 0
    for f, hold in seq:
        img = up(f)
        for _ in range(hold):
            vw.write(img); total += 1
    vw.release()
    print(f"env_win={won}  frames={total}  wrote {OUT}")


if __name__ == "__main__":
    main()
