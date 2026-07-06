"""Per-target pour labels (STAGE_PT_SPEC §3) — the decisive ground truth.

Decompose a pour's outcome PER U-target instead of the aggregate (fill, lava)
scalar. A loss always fails at some SPECIFIC target; a per-target label can
reject a near-winner for the exact U it misses, which the min-aggregated scalar
provably cannot (NIGHT F38-40: hard-negative mining on the aggregate HURT).

Pure forward-physics reading of env transitions — no model, no answer key, no
route/human lineage. Colors (verified F8/F16/F22): 11=unfilled target U,
13=satisfied target, 14=wall/border (14-GROWTH above pre-pour baseline = lava).

Contract (implement exactly §3):
  targets      : frame_components(pre) with color==11; each gives cells_t, bbox_t.
  fill_t       : sustained 13-fill of cells_t (3-consecutive-frame min window),
                 normalised by |cells_t| in [0,1].
  lavan_t      : local 14-GROWTH in bbox_t dilated Chebyshev R (clip>=0),
                 normalised min(1, lava_t/24).
  global       : (sus, lava) as in harvest_variants.pour_record + env_win.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_LAB = Path(__file__).resolve().parents[1]
_REPO = _LAB.parent
for _p in (_LAB, _LAB / "orchestration", _LAB / "experiments" / "megarexus_cnn"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402

TARGET_COLOR = 11
FILLED_COLOR = 13
PAINT_COLOR = 6
WALL_COLOR = 14
GRID = 64
LAVA_NORM = 24.0  # §3.3 normaliser
FILL_WINDOW = 2   # DEVIATION-1: 2-frame sustained (win holds 2 frames pre-reset)


def _as_frames(frames):
    """A single ACTION5 press returns the WHOLE animation (fd.frame is a LIST).
    Accept a list of frames or a single frame; return list of int8 arrays."""
    if isinstance(frames, np.ndarray) and frames.ndim == 2:
        return [frames.astype(np.int8)]
    return [np.asarray(x, dtype=np.int8) for x in frames]


def _sustained(counts, w=2):
    """Sustained value = max over i of min(counts[i:i+w]) — a w-consecutive-frame
    min window (rejects <w-frame flicker; harvest_variants uses w=3).

    DEVIATION-1 (STAGE_PT_SPEC §3.2 correction, evidenced on the verified L1 winner):
    the win→reset transient holds the all-filled state for only 2 frames before the
    level resets (frames 19-20 of a 22-frame pour; frame 21 is post-reset). The
    pre-registered w=3 window therefore reads 0 for the LAST-filling targets and the
    spec's OWN sanity check `sum(sus_t) ≈ 240` fails. This is the F16-class post-reset
    label bug (§1.8) — corrected to w=2 over the PRE-RESET outcome frames. The strict
    w=3 value is still recorded (`sus_strict3`) for full audit."""
    n = len(counts)
    if n == 0:
        return 0
    if n < w:
        return int(min(counts))
    best = 0
    for i in range(n - w + 1):
        best = max(best, int(min(counts[i:i + w])))
    return best


def _outcome_end(frames):
    """Index one past the last SETTLED-outcome frame (drops the post-win reset/flash
    tail). Paint(6) grows monotonically then plateaus at the outcome; a win resets
    AFTER the plateau (paint collapses, 13->11, 14 win-flash spikes). Outcome frames =
    up to & including the last max-paint frame; everything after is reset garbage."""
    n6 = [int((f == PAINT_COLOR).sum()) for f in frames]
    mp = max(n6) if n6 else 0
    last_plateau = max((i for i, v in enumerate(n6) if v >= mp), default=len(frames) - 1)
    return last_plateau + 1


def _dilate_bbox(bbox_xyxy, r):
    """Chebyshev dilation of a bbox by radius r, clipped to the grid.
    bbox_xyxy = [xmin, ymin, xmax, ymax] (inclusive). Returns bool [64,64]."""
    x0, y0, x1, y1 = bbox_xyxy
    x0 = max(0, int(x0) - r); y0 = max(0, int(y0) - r)
    x1 = min(GRID - 1, int(x1) + r); y1 = min(GRID - 1, int(y1) + r)
    m = np.zeros((GRID, GRID), dtype=bool)
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m


def pertarget_labels(pre, frames, r=3):
    """Per-target labels for one pour. `pre` = pre-pour arrangement frame [64,64];
    `frames` = the pour's frame list (fd.frame). Returns a dict per §3.5."""
    pre = np.asarray(pre, dtype=np.int8)
    frames = _as_frames(frames)
    end = _outcome_end(frames)      # DEVIATION-1: drop post-win reset/flash tail
    out = frames[:end]              # settled-outcome frames (fill + lava read here)

    comps = [c for c in frame_components(pre) if int(c["color"]) == TARGET_COLOR]
    pre14 = (pre == WALL_COLOR)

    targets = []
    for c in comps:
        cells = np.asarray(c["mask"], dtype=bool)
        ncells = int(cells.sum())
        # §3.2 per-target sustained fill (2-frame window over outcome frames)
        n13_out = [int(((f == FILLED_COLOR) & cells).sum()) for f in out]
        n13_all = [int(((f == FILLED_COLOR) & cells).sum()) for f in frames]
        sus_t = _sustained(n13_out, FILL_WINDOW)
        sus_strict3 = _sustained(n13_all, 3)          # audit: literal §3.2 value
        fill_t = sus_t / max(ncells, 1)
        # §3.3 per-target local lava (14-growth in dilated bbox, outcome frames only —
        # excludes the win-flash 14 spike that would falsely penalise winners)
        region = _dilate_bbox(c["bbox_xyxy"], r)
        base14 = int((pre14 & region).sum())
        peak14 = max((int(((f == WALL_COLOR) & region).sum()) for f in out), default=base14)
        lava_t = max(0, peak14 - base14)
        lavan_t = min(1.0, lava_t / LAVA_NORM)
        targets.append({
            "fill": float(fill_t),
            "lavan": float(lavan_t),
            "lava": int(lava_t),
            "sus": int(sus_t),
            "sus_strict3": int(sus_strict3),
            "cells": ncells,
            "bbox": [int(v) for v in c["bbox_xyxy"]],
        })

    # §3.4 global: honest (outcome-frame) sus/lava + harvest-literal lava for
    # comparison to v52's training labels (which used max over ALL frames).
    base_wall = int(pre14.sum())
    g_sus = _sustained([int((f == FILLED_COLOR).sum()) for f in out], FILL_WINDOW)
    g_lava = max((int((f == WALL_COLOR).sum()) for f in out), default=base_wall) - base_wall
    g_lava_harvest = max(int((f == WALL_COLOR).sum()) for f in frames) - base_wall
    return {
        "n_targets": len(targets),
        "targets": targets,
        "n_frames": len(frames),
        "outcome_end": int(end),
        "global": {"sus": int(g_sus), "lava": int(max(0, g_lava)),
                   "lava_harvest": int(g_lava_harvest)},
    }


def winner_ok(labels, fill_min=0.95, lavan_max=0.2):
    """§3.3 PASS predicate for a single arrangement's per-target labels: ALL
    targets fill_t>=fill_min AND lavan_t<=lavan_max. Returns (ok, reasons)."""
    reasons = []
    ok = labels["n_targets"] > 0
    for i, t in enumerate(labels["targets"]):
        if t["fill"] < fill_min:
            ok = False; reasons.append(f"U{i}: fill={t['fill']:.3f}<{fill_min}")
        if t["lavan"] > lavan_max:
            ok = False; reasons.append(f"U{i}: lavan={t['lavan']:.3f}>{lavan_max}")
    return ok, reasons
