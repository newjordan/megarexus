"""Round-5 Phase-A observation over the harvested L4/L5 pour corpora:
find stream contacts with the color-15 movable blocker and dump local
animation windows so the contact semantics are read from frames, not assumed.

Usage:
    python3 r5_l4_observe.py --level 4              # contact scan, all pours
    python3 r5_l4_observe.py --level 4 --pour 19 \
        --ticks 2,3,4,5,6,7,9,11 --window 35,59,12,34   # frame dump

Findings so far (2026-07-06, pour 19 deep read — see DW_LAB_LOG round-5
Phase-A entry; VERIFY on the remaining contact pours before fitting):
  * L4 flow RISES (sources at bottom, 1 cell/tick), film-trail persists
  * mouths at top face DOWN, fill=13 on entry; fail = color-14 growth at the
    TOP wall band + flash alternation (win/fail semantics mirrored)
  * color-15 contact: stream is BLOCKED, spreads laterally beneath the block,
    splits and rises around both clear sides — generic static-obstacle
    film-flow semantics; no pass-through, no absorption observed
  * the 15-structure is MOVABLE: arranged positions vary across pours;
    selected it renders 9 (whole L footprint), unselected it renders 15
"""
from __future__ import annotations

import argparse

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES

TRAIL = 6
BLOCKER = 15


def load(level):
    d = np.load(FIXTURES / f"calib_l{level}_pours_20260706.npz", allow_pickle=True)
    lens = d["lens"]
    offs = np.concatenate([[0], np.cumsum(lens)])
    return d, offs


def contact_scan(level):
    d, offs = load(level)
    ff, lens = d["frames_flat"], d["lens"]
    for p in range(len(lens)):
        frames = ff[offs[p]:offs[p + 1]]
        f0 = frames[0]                      # arranged board, pre-flow
        m15 = f0 == BLOCKER
        if not m15.any():
            print(f"pour {p:2d}: NO color-15 on arranged board")
            continue
        ys, xs = np.nonzero(m15)
        dil = np.zeros_like(m15)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            dil[max(0, dy):64 + min(0, dy), max(0, dx):64 + min(0, dx)] |= \
                m15[max(0, -dy):64 + min(0, -dy), max(0, -dx):64 + min(0, -dx)]
        dil &= ~m15
        first = next((t for t in range(len(frames))
                      if ((frames[t] == TRAIL) & (f0 != TRAIL) & dil).any()), None)
        print(f"pour {p:2d}: 15@x{xs.min()}-{xs.max()},y{ys.min()}-{ys.max()}"
              f"  first_trail_contact={'t' + str(first) if first is not None else 'NONE'}"
              f"  frames={lens[p]} win={bool(d['wins'][p])}")


def dump(level, pour, ticks, window):
    d, offs = load(level)
    frames = d["frames_flat"][offs[pour]:offs[pour + 1]]
    y0, y1, x0, x1 = window
    print(f"=== L{level} pour {pour} ({len(frames)} frames) window y{y0}-{y1} x{x0}-{x1}")
    for t in ticks:
        if t >= len(frames):
            break
        print(f"t{t}:")
        for y in range(y0, y1):
            print("   ", "".join(f"{frames[t][y, x]:X}" for x in range(x0, x1)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=4)
    ap.add_argument("--pour", type=int, default=None)
    ap.add_argument("--ticks", default="0,2,4,6,8,10,12,14")
    ap.add_argument("--window", default="0,64,0,64")
    a = ap.parse_args()
    if a.pour is None:
        contact_scan(a.level)
    else:
        dump(a.level, a.pour, [int(t) for t in a.ticks.split(",")],
             [int(v) for v in a.window.split(",")])
