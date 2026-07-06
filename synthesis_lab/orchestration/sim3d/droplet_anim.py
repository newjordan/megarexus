"""Animate the droplet-CA SIMULATION (not the env): time-step the block-flow mechanic so you can
watch droplets fall, fork at blocks (x-1/x+1), enter buckets (turn green), shed off the open side
edges, and flash the floor red on a fail-touch. Renders on the native sprite grid. MP4 out.

    python3 droplet_anim.py --level 2 --genome-json fixtures/p2_droplet_solve.json --out fixtures/L2_sim.mp4
    python3 droplet_anim.py --level 3 --out fixtures/L3_sim.mp4
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import cv2

import _paths  # noqa: F401
from _paths import LAB, FIXTURES
from imagine_search_gate import _present, GAME_ID, ACTION_OF_ID, game_live
from imag_search_solve import actions
from engine_adapters.portband_glove_adapter import frame_components
from native_scene import native_grid
from scene import extract_scene
from dropletsim import _typed
from imagine_search_gate import _grid
from arc_agi import Arcade
from arc_agi.base import OperationMode


def grids_from_frame(frame):
    """Typed grid dict from the 16x16 frame-macro extractor (faithful for L0/L1/L2)."""
    sc = extract_scene(frame)
    block, target, wall, tid = _typed(sc)
    return {"block": block, "target": target, "wall": wall, "tid": tid,
            "spouts": list(sc.spouts or [sc.spout]), "n_targets": len(sc.targets)}

SCALE = 22
BG = (20, 18, 24)
C_BLOCK = (110, 80, 55)     # BGR
C_WALL = (40, 40, 150)      # floor/wall (dark red)
C_WALLFLASH = (60, 60, 255)
C_TGT = (150, 90, 40)       # unfilled bucket (blue)
C_TGT_FILL = (90, 210, 110) # filled bucket (green)
C_DROP = (245, 190, 90)     # paint droplet (cyan-ish)
C_SRC = (60, 190, 235)      # source (gold)


def kind_fn(block, target, wall, H, W):
    def kind(y, x):
        if x < 0 or x >= W or y < 0 or y >= H:
            return None
        if block[y, x]:
            return "block"
        if target[y, x]:
            return "target"
        if wall[y, x]:
            return "wall"
        return None
    return kind


def simulate(ng, max_ticks=160, emit_ticks=70):
    block, target, wall, tid = ng["block"], ng["target"], ng["wall"], ng["tid"]
    spouts, ntgt = ng["spouts"], ng["n_targets"]
    H, W = block.shape
    kind = kind_fn(block, target, wall, H, W)
    filled, droplets = set(), set()
    snaps, flash = [], 0
    for t in range(max_ticks):
        if t < emit_ticks:
            for s in spouts:
                droplets.add((int(s[0]), int(s[1])))
        nxt_set = set()
        for (y, x) in droplets:
            k = kind(y + 1, x)
            if k is None:
                if y + 1 < H:
                    nxt_set.add((y + 1, x))          # fall (off bottom -> disposed)
            elif k == "wall":
                flash = 4                            # fail flash
            elif k == "block":
                for nx in (x - 1, x + 1):
                    if kind(y, nx) is None:
                        nxt_set.add((y, nx))
            elif k == "target":
                le, ri = kind(y, x - 1), kind(y, x + 1)
                if le == "target" and ri == "target":
                    filled.add(int(tid[y + 1, x]))
                else:
                    for nx in (x - 1, x + 1):
                        if kind(y, nx) is None:
                            nxt_set.add((y, nx))
        droplets = nxt_set
        snaps.append((set(droplets), set(filled), flash > 0))
        flash = max(0, flash - 1)
        if not droplets and t >= emit_ticks:
            break
    win = (len([f for f in filled if f >= 0]) >= ntgt) and not any(s[2] for s in snaps[-3:])
    return snaps, filled, ntgt


def render(ng, snaps, out):
    block, target, wall, tid = ng["block"], ng["target"], ng["wall"], ng["tid"]
    H, W = block.shape
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), 18, (W * SCALE, H * SCALE))

    def draw(drops, filled, flash):
        img = np.zeros((H, W, 3), np.uint8); img[:] = BG
        img[wall] = C_WALLFLASH if flash else C_WALL
        img[block] = C_BLOCK
        for y in range(H):
            for x in range(W):
                if target[y, x]:
                    img[y, x] = C_TGT_FILL if int(tid[y, x]) in filled else C_TGT
        big = np.kron(img, np.ones((SCALE, SCALE, 1), np.uint8))
        for (y, x) in drops:                          # draw droplets as circles
            cv2.circle(big, (x * SCALE + SCALE // 2, y * SCALE + SCALE // 2),
                       SCALE // 2 - 2, C_DROP, -1)
        return big

    hold0 = snaps[0]
    for _ in range(10):
        vw.write(draw(*hold0))
    for s in snaps:
        vw.write(draw(*s))
    last = snaps[-1]
    for _ in range(30):
        vw.write(draw(*last))
    vw.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--genome-json", default=None)
    ap.add_argument("--native", action="store_true", help="use native-sprite extractor (L3+)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    env = arc.make(GAME_ID, seed=1234)
    fd, c0 = _present(env, a.level)
    if a.genome_json:
        genome = [tuple(x) for x in json.load(open(a.genome_json))["genome"]]
        centers = [(int(c["center_xy"][0]), int(c["center_xy"][1])) for c in frame_components(c0)[:10]]
        for (aid, x, y) in actions(genome, centers):
            if not game_live(fd):
                break
            fd = env.step(ACTION_OF_ID[aid], data={"x": int(x), "y": int(y)} if aid == 6 else None)
    ng = native_grid(env._game) if a.native else grids_from_frame(_grid(fd))
    snaps, filled, ntgt = simulate(ng)
    out = a.out or str(FIXTURES / f"L{a.level}_sim.mp4")
    render(ng, snaps, out)
    print(f"L{a.level} sim: buckets filled={sorted(filled)}/{ntgt}  fail={any(s[2] for s in snaps[-3:])}  "
          f"ticks={len(snaps)}  wrote {out}")


if __name__ == "__main__":
    main()
