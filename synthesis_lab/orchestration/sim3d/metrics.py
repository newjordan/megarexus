"""Sim rollout -> rendered frames -> the PROVEN per-target scorer.

The one non-negotiable design rule: the simulator is graded by the SAME function
that grades real pours — pertarget_labels (STAGE_PT ground truth, T0 PASS). We
render each sim snapshot back into a 64x64 SP80-colored frame (paint=6, filled
U=13, floor-lava flash=14) and hand the sequence to pertarget_labels verbatim.
No parallel scoring path that could quietly diverge from the env's semantics.
"""
from __future__ import annotations

import numpy as np

import _paths  # noqa: F401
from scene import M, CELL, GRID, PAINT, FILLED, WALL, HUD

from pertarget_labels import pertarget_labels, winner_ok  # noqa: E402

OCC_RENDER = 0.5      # occupancy (fraction of a cell's fluid capacity) that renders as paint
FILL_RENDER = 0.85    # mouth fill fraction at which the U recolors 11->13


def render_frames(scene, snapshots, per_cell=1.0, occ_render=OCC_RENDER,
                  fill_render=FILL_RENDER):
    """snapshots: [(occ [16,16], mouth_fill [n_t], lava bool)] in CANONICAL coords
    (both lanes emit this). per_cell: occupancy units per full macro cell (Lane P:
    SimParams.per_cell; Lane R: HydroParams.cell_cap). Returns frames [T,64,64] int8
    in ORIGINAL orientation."""
    base = np.flipud(scene.pre).copy() if scene.flipped else scene.pre.copy()
    glyphs = [np.flipud(t.cells_px) if scene.flipped else t.cells_px
              for t in scene.targets]
    strip_rows = [r for row in scene.floor_rows
                  for r in range(row * CELL, row * CELL + CELL)]
    frames = []
    for occ, mouth_fill, lava in snapshots:
        f = base.copy()
        paint_cells = np.asarray(occ) >= occ_render * per_cell
        px = np.kron(paint_cells, np.ones((CELL, CELL), dtype=bool))
        f[px & (base == 12)] = PAINT                      # paint only over open space
        for ti, g in enumerate(glyphs):
            if ti < len(mouth_fill) and mouth_fill[ti] >= fill_render:
                f[g] = FILLED
        if lava:
            f[strip_rows, :] = WALL                       # the game's floor-is-lava flash
        frames.append(np.flipud(f) if scene.flipped else f)
    return np.stack(frames) if frames else np.zeros((0, GRID, GRID), dtype=np.int8)


def score_rollout(scene, result, per_cell=1.0, r=3):
    """Full scoring of one sim pour: rendered frames -> pertarget_labels -> the
    §7 decisive margin  min_t(fill_t - lavan_t)  plus the winner predicate."""
    frames = render_frames(scene, result["snapshots"], per_cell=per_cell)
    labels = pertarget_labels(scene.pre, frames, r=r)
    margins = [t["fill"] - t["lavan"] for t in labels["targets"]] or [0.0]
    ok, reasons = winner_ok(labels)
    return {
        "labels": labels,
        "margin": float(min(margins)),
        "winner_ok": bool(ok),
        "reasons": reasons,
        "native": {
            "mouth_fill": [float(v) for v in result["mouth_fill"]],
            "lava": bool(result["lava"]),
            "ticks": int(result["ticks"]),
        },
        "n_frames": int(len(frames)),
    }
