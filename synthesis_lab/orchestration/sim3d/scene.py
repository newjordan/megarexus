"""Board -> 3D scene extraction (SIM3D stage S-A).

Turns a 64x64 SP80 frame into a semantic 3D scene the physics/raycast lanes pour
into. All geometry lives on the game's own 16x16 MACRO-GRID (4px cells — verified:
emission = +16px/tick = exactly 1 macro cell of paint per tick; streams are 1 macro
cell wide). One macro cell = one unit cube in the 3D scene (extruded DEPTH deep).

CANONICALIZATION (the rotation unlock): the scene is always flipped so the pour
flows DOWN (+y). Real L1 is rotated 180° (source at bottom, pours up) — the exact
out-of-distribution rotation that broke v48 (NIGHT F35). After the flip L1 looks
like L0 to the simulator; results are mapped back before rendering. Level-invariance
by construction, not by training.

Semantic color map (verified on present_l0/l1 fixtures + pour animations):
  4  = source housing (solid)          6  = paint reservoir (the fluid, pre-pour)
  11 = U-target glyph (solid; recolors 11->13 when its MOUTH fills)
  8/9 = movable bar, unselected/selected (solid — these are what arrangements move)
  14 = wall bar behind the source (solid; also the game's flash color)
  1  = boundary strip BEHIND THE TARGETS (rows 60-63 on L0, 0-3 on L1): solid AND
       the LAVA SENSOR — overflow paint touching it flashes it to 14 ("the floor is
       lava", verified pour 5: n14 53->309 = +256 = the whole 64x4 strip).
  12 = open space.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRID = 64
M = 16                    # macro-grid side
CELL = GRID // M          # 4 px per macro cell

SOURCE, PAINT, HUD, OPEN = 4, 6, 1, 12
TARGET, FILLED, WALL = 11, 13, 14
MOVABLE, SELECTED = 8, 9
SOLID_COLORS = (SOURCE, TARGET, FILLED, WALL, MOVABLE, SELECTED)


def to_macro(frame):
    """64x64 -> 16x16 by per-block majority vote (wall bars are 62px, not 4-aligned,
    so exact block equality can't be asserted)."""
    f = np.asarray(frame)
    blocks = f.reshape(M, CELL, M, CELL).transpose(0, 2, 1, 3).reshape(M, M, CELL * CELL)
    out = np.zeros((M, M), dtype=np.int8)
    for y in range(M):
        for x in range(M):
            vals, counts = np.unique(blocks[y, x], return_counts=True)
            out[y, x] = vals[np.argmax(counts)]
    return out


def _macro_mask_any(frame, colors):
    """16x16 bool: block contains ANY pixel of the given colors (catches the 62px
    wall bar whose end blocks are minority-14)."""
    f = np.asarray(frame)
    m = np.isin(f, np.asarray(colors, dtype=f.dtype))
    return m.reshape(M, CELL, M, CELL).any(axis=(1, 3))


@dataclass
class Target:
    """One U-target. Coordinates are CANONICAL macro-grid (flow = down)."""
    cells_px: np.ndarray          # [64,64] bool — the color-11 glyph, ORIGINAL frame coords
    glyph: np.ndarray             # [16,16] bool — glyph cells, canonical macro
    mouth: list                   # [(y,x)] canonical macro cells of the hollow interior
    capacity: float = 0.0         # fluid volume (macro cells) the mouth holds


@dataclass
class Scene:
    pre: np.ndarray               # original 64x64 int8 frame (pre-pour arrangement)
    flipped: bool                 # vertical flip applied to canonicalize flow-down?
    g: np.ndarray                 # [16,16] canonical macro color grid
    solid: np.ndarray             # [16,16] bool static solids (incl. floor strip)
    floor_rows: list = field(default_factory=list)   # canonical macro rows of the lava strip
    spout: tuple = (0, 0)         # (y, x) canonical macro cell where fluid enters the field
    emit_volume: float = 37.0     # macro cells of fluid one ACTION5 emits (calibrate.py)
    targets: list = field(default_factory=list)
    depth: int = 1                # slab depth in macro cells (3D extrusion)

    def uncanon_rows(self, rows):
        return [M - 1 - r for r in rows] if self.flipped else list(rows)


def _components(mask):
    """4-connected components of a bool [16,16] mask -> list of index arrays."""
    lab = np.full(mask.shape, -1, dtype=np.int32)
    comps = []
    for y, x in zip(*np.nonzero(mask)):
        if lab[y, x] >= 0:
            continue
        stack, cells = [(y, x)], []
        lab[y, x] = len(comps)
        while stack:
            cy, cx = stack.pop()
            cells.append((cy, cx))
            for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                if 0 <= ny < M and 0 <= nx < M and mask[ny, nx] and lab[ny, nx] < 0:
                    lab[ny, nx] = len(comps)
                    stack.append((ny, nx))
        comps.append(cells)
    return comps


def extract_scene(frame, depth=1, emit_volume=None):
    """Pre-pour 64x64 frame -> Scene (canonical flow-down macro geometry)."""
    pre = np.asarray(frame, dtype=np.int8)
    src_rows = np.nonzero((pre == SOURCE).any(axis=1))[0]
    tgt_rows = np.nonzero((pre == TARGET).any(axis=1))[0]
    if len(src_rows) == 0 or len(tgt_rows) == 0:
        raise ValueError("scene needs a source(4) and targets(11)")
    flipped = bool(src_rows.mean() > tgt_rows.mean())   # source below targets -> pour flows UP
    f = np.flipud(pre) if flipped else pre

    g = to_macro(f)
    solid = _macro_mask_any(f, SOLID_COLORS)

    # Lava strip = the color-1 boundary band BEHIND the targets (below them, canonical).
    tgt_rows_c = np.nonzero((f == TARGET).any(axis=1))[0]
    strip_px_rows = [r for r in np.nonzero((f == HUD).all(axis=1) | ((f == HUD).sum(axis=1) > GRID // 2))[0]
                     if r > tgt_rows_c.max()]
    floor_rows = sorted({r // CELL for r in strip_px_rows})
    for r in floor_rows:
        solid[r, :] = True

    # Emitter: fluid enters just below the paint reservoir (color 6 block under the
    # source). If an arrangement shoved a bar right there, lift the spout back up.
    p_ys, p_xs = np.nonzero(f == PAINT)
    sy, sx = int(p_ys.max()) // CELL + 1, int(round(p_xs.mean())) // CELL
    while sy > 0 and solid[sy, sx]:
        sy -= 1
    spout = (sy, sx)

    # Targets: decompose each U glyph into solid glyph cells + hollow MOUTH cells
    # (bbox interior cells that are open and reachable from the bbox's top row).
    targets = []
    glyph_all = _macro_mask_any(f, (TARGET,))
    for cells in _components(glyph_all):
        ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
        y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
        gm = np.zeros((M, M), dtype=bool)
        for c in cells:
            gm[c] = True
        mouth = [(y, x) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)
                 if not solid[y, x]]
        cpx_canon = np.zeros((GRID, GRID), dtype=bool)
        for (y, x) in cells:
            blk = f[y*CELL:(y+1)*CELL, x*CELL:(x+1)*CELL]
            cpx_canon[y*CELL:(y+1)*CELL, x*CELL:(x+1)*CELL] = (blk == TARGET)
        cells_px = np.flipud(cpx_canon) if flipped else cpx_canon
        targets.append(Target(cells_px=cells_px, glyph=gm, mouth=mouth,
                              capacity=float(len(mouth))))
    targets.sort(key=lambda t: min(x for _, x in t.mouth) if t.mouth else 0)

    vol = float(emit_volume) if emit_volume is not None else 37.0  # measured L0 (calibrate.py)
    return Scene(pre=pre, flipped=flipped, g=g, solid=solid, floor_rows=floor_rows,
                 spout=spout, emit_volume=vol, targets=targets, depth=depth)


def solid_boxes(scene):
    """Static colliders as 3D AABBs [N,6] (y0,x0,z0,y1,x1,z1) in macro units,
    greedily merged along x then y for fewer boxes. z spans the slab depth."""
    solid = scene.solid.copy()
    boxes = []
    for y in range(M):
        x = 0
        while x < M:
            if not solid[y, x]:
                x += 1; continue
            x1 = x
            while x1 + 1 < M and solid[y, x1 + 1]:
                x1 += 1
            boxes.append([y, x, 0, y + 1, x1 + 1, scene.depth])
            x = x1 + 1
    merged = []
    for b in sorted(boxes, key=lambda b: (b[1], b[4], b[0])):
        if merged and merged[-1][1] == b[1] and merged[-1][4] == b[4] and merged[-1][3] == b[0]:
            merged[-1][3] = b[3]
        else:
            merged.append(list(b))
    return np.asarray(merged, dtype=np.float64)
