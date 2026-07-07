"""Round-5 step-1 verifier: auto-classify stream/color-15 contact semantics
on every L4 contact pour, frames-only.

Per pour reports:
  entered_15   -- any trail pixel ever inside the 15 mask (pass/absorb check)
  sides        -- first tick trail touches the blocker from below/left/right/above
  rose_L/R     -- trail present above blocker top on each flank (post-contact)
  over_top     -- trail directly adjacent above a 15 cell (pass-over/re-merge)
  notch        -- trail inside the L's open quadrant (bounding box minus mask)

CAVEAT: rose_L/R and over_top are presence checks in fixed windows — they do
NOT attribute the trail to the contacting stream (unrelated streams trip
them). Verified 2026-07-06 by eyeballing dumps (r5_l4_observe.py --pour):
result = entered_15 False on all 16 contact pours; every class reduces to
generic static-obstacle block semantics (see DW_LAB_LOG round-5 step-1 entry).
"""
from __future__ import annotations

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES

TRAIL = 6
BLOCKER = 15

d = np.load(FIXTURES / "calib_l4_pours_20260706.npz", allow_pickle=True)
lens = d["lens"]
offs = np.concatenate([[0], np.cumsum(lens)])
ff = d["frames_flat"]

# side name = where the trail sits relative to the blocker (y grows downward;
# flow rises toward smaller y, so "below" = trail underneath the blocker)
SIDES = {"below": (1, 0), "above": (-1, 0), "left": (0, -1), "right": (0, 1)}

YY, XX = np.mgrid[0:64, 0:64]


def shift(mask, dy, dx):
    out = np.zeros_like(mask)
    out[max(0, dy):64 + min(0, dy), max(0, dx):64 + min(0, dx)] = \
        mask[max(0, -dy):64 + min(0, -dy), max(0, -dx):64 + min(0, -dx)]
    return out


for p in range(len(lens)):
    frames = ff[offs[p]:offs[p + 1]]
    f0 = frames[0]
    m15 = f0 == BLOCKER
    if not m15.any():
        continue
    ys, xs = np.nonzero(m15)
    ymin, ymax, xmin, xmax = ys.min(), ys.max(), xs.min(), xs.max()
    bbox = np.zeros_like(m15)
    bbox[ymin:ymax + 1, xmin:xmax + 1] = True
    notch_mask = bbox & ~m15
    above_top = YY < ymin
    left_flank = above_top & (XX >= xmin - 3) & (XX <= xmin - 1)
    right_flank = above_top & (XX >= xmax + 1) & (XX <= xmax + 3)

    entered = False
    side_first = {}
    contact_t = None
    rose_left = rose_right = over_top = notch = False
    for t in range(len(frames)):
        trail = (frames[t] == TRAIL) & (f0 != TRAIL)
        if not trail.any():
            continue
        if (trail & m15).any():
            entered = True
        for name, (dy, dx) in SIDES.items():
            adj = shift(m15, dy, dx) & ~m15 & trail
            if adj.any() and name not in side_first:
                side_first[name] = t
                if contact_t is None:
                    contact_t = t
        if contact_t is not None:
            rose_left |= bool((trail & left_flank).any())
            rose_right |= bool((trail & right_flank).any())
            over_top |= bool((shift(m15, -1, 0) & ~m15 & trail).any())
            notch |= bool((trail & notch_mask).any())
    if contact_t is None:
        continue
    order = {k: f"t{v}" for k, v in sorted(side_first.items(), key=lambda kv: kv[1])}
    print(f"pour {p:2d}: 15@x{xmin}-{xmax},y{ymin}-{ymax}"
          f"  entered_15={entered}  sides={order}"
          f"  rose_L={rose_left} rose_R={rose_right}"
          f" over_top={over_top} notch={notch}")
