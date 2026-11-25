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

# ---- PALETTE MANIFEST — the ONLY game-specific input to this machinery ----
# H4 contract (DREAMWEAVER_INTEGRATION_PLAN §2): everything below this block is
# game-agnostic. Each role↔color binding was made and is continually verified
# from OBSERVED frames + env outcomes (perception gate per frame; pour fits per
# dynamics claim). On a different game, this manifest is the single thing that
# must be re-induced; the machinery is unchanged.
PALETTE = {
    "open": 12, "boundary": 1,            # background / HUD boundary band
    "source": 4, "paint": 6,              # emitter housing / fluid
    "target": 11, "filled": 13,           # U-glyph / filled indication
    "wall": 14, "movable": 8, "selected": 9,
    # blocker: r5 Phase-A live extension (de-grey protocol). Induced from
    # frames+outcomes only: 16 L4 + 40 L5 contact pours, trail NEVER enters a
    # 15 cell (no pass/absorb/turn observed); the structure is MOVABLE
    # (arranged positions vary; SELECTED it renders 9 over its footprint,
    # unselected stays 15, not 8). Its flow semantics are NOT fixed here —
    # FlowTrail's blocker_contact knob family decides by observation
    # (r5 fit). Evidence: DW_LAB_LOG_20260706 round-5 step-1/step-2 entries.
    "blocker": 15,
}
SOURCE, PAINT, HUD, OPEN = (PALETTE["source"], PALETTE["paint"],
                            PALETTE["boundary"], PALETTE["open"])
TARGET, FILLED, WALL = PALETTE["target"], PALETTE["filled"], PALETTE["wall"]
MOVABLE, SELECTED = PALETTE["movable"], PALETTE["selected"]
BLOCKER = PALETTE["blocker"]
# BLOCKER is deliberately NOT in SOLID_COLORS: its collision semantics are a
# dynamics KNOB (FlowTrail.blocker_contact), not a settled fact. Scene carries
# it as a separate mask; coverage checks account for it explicitly.
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


def _lattice_majority(f, u):
    """[H,W] frame (H,W multiples of u) -> [H/u,W/u] per-block majority color."""
    h, w = f.shape[0] // u, f.shape[1] // u
    blocks = f.reshape(h, u, w, u).transpose(0, 2, 1, 3).reshape(h, w, u * u)
    out = np.zeros((h, w), dtype=np.int8)
    for y in range(h):
        for x in range(w):
            vals, counts = np.unique(blocks[y, x], return_counts=True)
            out[y, x] = vals[np.argmax(counts)]
    return out


def _lattice_any(f, colors, u):
    """[H,W] frame -> [H/u,W/u] bool: block contains ANY pixel of the colors."""
    m = np.isin(f, np.asarray(colors, dtype=f.dtype))
    h, w = f.shape[0] // u, f.shape[1] // u
    return m.reshape(h, u, w, u).any(axis=(1, 3))


def _pixel_components(mask):
    """4-connected components of a bool pixel mask -> list of (ys, xs) arrays."""
    mask = np.asarray(mask, dtype=bool)
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
                if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                        and mask[ny, nx] and lab[ny, nx] < 0):
                    lab[ny, nx] = len(comps)
                    stack.append((ny, nx))
        ys = np.asarray([c[0] for c in cells]); xs = np.asarray([c[1] for c in cells])
        comps.append((ys, xs))
    return comps


def detect_lattice(f):
    """Native pixel lattice (unit, oy, ox) INDUCED from target-glyph runs.

    The U glyphs are the finest reliable geometry in frame: wall thickness and
    interior gap share the piece unit. unit = gcd of every horizontal+vertical
    color-11 run length; origin = the modal run-start residue per axis.
    L0-L2 measure (4, 0, 0) — the historical fixed macro grid. L3-L5 measure
    (3, 2, 2), which the fixed grid could not represent (merged targets,
    vanished mouths). Pixels only; degenerate frames fall back to (CELL, 0, 0)
    and the perception gate passes judgement downstream.
    """
    from functools import reduce
    from math import gcd
    mask = (np.asarray(f) == TARGET)
    lens, hstarts, vstarts = [], [], []
    for arr, starts in ((mask, hstarts), (mask.T, vstarts)):
        for row in arr:
            idx = np.nonzero(row)[0]
            if len(idx) == 0:
                continue
            for run in np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1):
                lens.append(len(run)); starts.append(int(run[0]))
    if not lens:
        return CELL, 0, 0
    unit = reduce(gcd, lens)
    if unit < 2:
        return CELL, 0, 0
    oy = int(np.bincount([s % unit for s in vstarts]).argmax()) if vstarts else 0
    ox = int(np.bincount([s % unit for s in hstarts]).argmax()) if hstarts else 0
    return int(unit), oy, ox


@dataclass
class Target:
    """One U-target. Coordinates are CANONICAL lattice cells (flow = down)."""
    cells_px: np.ndarray          # [64,64] bool — the color-11 glyph, ORIGINAL frame coords
    glyph: np.ndarray             # [gh,gw] bool — glyph cells, canonical lattice
    mouth: list                   # [(y,x)] canonical lattice cells of the hollow interior
    capacity: float = 0.0         # fluid volume (lattice cells) the mouth holds
    opening: str = None           # single open side of the mouth: up/down/left/right
                                  # (L4/L5 carry side-facing mouths; None = ambiguous,
                                  # the perception gate fails such targets)


@dataclass
class Scene:
    pre: np.ndarray               # original 64x64 int8 frame (pre-pour arrangement)
    flipped: bool                 # vertical flip applied to canonicalize flow-down?
    g: np.ndarray                 # [gh,gw] canonical lattice color grid
    solid: np.ndarray             # [gh,gw] bool static solids (incl. floor strip)
    floor_rows: list = field(default_factory=list)   # canonical lattice rows of the lava strip
    spout: tuple = (0, 0)         # (y, x) canonical lattice cell where fluid enters (= spouts[0])
    spouts: list = field(default_factory=list)  # one per SOURCE housing (multi-source: L2=3)
    emit_volume: float = 37.0     # lattice cells of fluid one ACTION5 emits (measured on L0's
                                  # 4px lattice; other lattices need per-level calibration)
    targets: list = field(default_factory=list)
    depth: int = 1                # slab depth in lattice cells (3D extrusion)
    cell: int = CELL              # native lattice pitch in pixels (induced; L0-L2=4, L3-L5=3)
    pad: tuple = (0, 0)           # (pad_y, pad_x) top-left padding aligning lattice origin
    spout_charged: list = field(default_factory=list)  # per spout: housing has a reservoir?
    blocker: np.ndarray = None    # [gh,gw] bool movable-blocker (color 15) cells; collision
                                  # semantics decided by FlowTrail.blocker_contact (r5 knob)

    def uncanon_rows(self, rows):
        gh = self.solid.shape[0]
        return [gh - 1 - r for r in rows] if self.flipped else list(rows)


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
    """Pre-pour 64x64 frame -> Scene (canonical flow-down geometry on the
    frame's NATIVE lattice).

    The lattice (pitch + origin) is INDUCED from target-glyph pixel runs
    (detect_lattice): L0-L2 sit on a 4px origin-0 lattice — identical to the
    historical fixed macro grid, so those scenes are unchanged — while L3-L5
    sit on a 3px origin-2 lattice the fixed grid could not represent (merged
    targets, vanished mouths: the two perception-gate failure classes). The
    canonical frame is padded top-left so the lattice origin coincides with
    the array origin; all Scene coordinates are lattice cells of that padded
    canonical frame. Target.cells_px stays in ORIGINAL frame pixels.
    Pixels only — no engine internals.
    """
    pre = np.asarray(frame, dtype=np.int8)
    src_rows = np.nonzero((pre == SOURCE).any(axis=1))[0]
    tgt_rows = np.nonzero((pre == TARGET).any(axis=1))[0]
    if len(src_rows) == 0 or len(tgt_rows) == 0:
        raise ValueError("scene needs a source(4) and targets(11)")
    flipped = bool(src_rows.mean() > tgt_rows.mean())   # source below targets -> pour flows UP
    f0 = np.flipud(pre) if flipped else pre

    u, oy, ox = detect_lattice(f0)
    pad_y, pad_x = (u - oy) % u, (u - ox) % u
    gh, gw = -(-(GRID + pad_y) // u), -(-(GRID + pad_x) // u)
    f = np.full((gh * u, gw * u), OPEN, dtype=np.int8)
    f[pad_y:pad_y + GRID, pad_x:pad_x + GRID] = f0

    g = _lattice_majority(f, u)
    solid = _lattice_any(f, SOLID_COLORS, u)
    blocker = _lattice_any(f, [BLOCKER], u)

    # Lava strip = the color-1 boundary band BEHIND the targets (below them, canonical).
    W = gw * u
    tgt_px_rows = np.nonzero((f == TARGET).any(axis=1))[0]
    strip_px_rows = [r for r in np.nonzero((f == HUD).all(axis=1)
                                           | ((f == HUD).sum(axis=1) > W // 2))[0]
                     if r > tgt_px_rows.max()]
    floor_rows = sorted({r // u for r in strip_px_rows})
    for r in floor_rows:
        solid[r, :] = True

    # Emitter(s): one spout per SOURCE housing (color 4), positioned by the
    # housing's paint reservoir (color 6) when it has one — `charged` — else by
    # the housing itself (L3/L5 carry an uncharged mid-board catch/redirect
    # housing; whether it re-emits is a DYNAMICS question left to observation).
    # Single-source levels yield the identical old reservoir-pixel computation.
    def _spout_of(ys, xs):
        # The cell just below the paint can land one row past the lattice
        # when the reservoir touches the bottom of the padded frame.
        sy = int(np.max(ys)) // u + 1
        sx = int(round(float(np.mean(xs)))) // u
        sy = min(max(sy, 0), solid.shape[0] - 1)
        sx = min(max(sx, 0), solid.shape[1] - 1)
        while sy > 0 and solid[sy, sx]:
            sy -= 1
        return (sy, sx)

    housings = _pixel_components(f == SOURCE)
    pools = _pixel_components(f == PAINT)
    pairs = []
    if housings:
        used = set()
        for hys, hxs in housings:
            res = next((j for j, (pys, pxs) in enumerate(pools)
                        if j not in used
                        and int(pxs.min()) <= int(hxs.max())
                        and int(pxs.max()) >= int(hxs.min())), None)
            if res is not None:
                used.add(res)
                pairs.append((_spout_of(*pools[res]), True))
            else:
                pairs.append((_spout_of(hys, hxs), False))
    else:                                   # sp80_variants boards: reservoir only
        pairs = [(_spout_of(pys, pxs), True) for pys, pxs in pools]
    if not pairs:
        raise ValueError("scene needs a source(4) or reservoir(6) to place the spout")
    seen = {}
    for s, c in pairs:
        seen[s] = seen.get(s, False) or c
    spouts = sorted(seen)
    spout_charged = [seen[s] for s in spouts]
    spout = spouts[0]

    # Targets: one per color-11 PIXEL component (macro components merged
    # adjacent L3-L5 targets). Mouth = open bbox-interior cells; the single
    # open side of the mouth is recorded (L4/L5 mouths face sideways).
    targets = []
    for ys, xs in _pixel_components(f == TARGET):
        cs = sorted({(int(y) // u, int(x) // u) for y, x in zip(ys, xs)})
        gm = np.zeros((gh, gw), dtype=bool)
        for c in cs:
            gm[c] = True
        y0, y1 = min(c[0] for c in cs), max(c[0] for c in cs)
        x0, x1 = min(c[1] for c in cs), max(c[1] for c in cs)
        mouth = [(y, x) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)
                 if not solid[y, x]]
        mset = set(mouth)
        opens = set()
        for (my, mx) in mouth:
            for dy, dx, name in ((-1, 0, "up"), (1, 0, "down"),
                                 (0, -1, "left"), (0, 1, "right")):
                ny, nx = my + dy, mx + dx
                if (ny, nx) in mset:
                    continue
                if 0 <= ny < gh and 0 <= nx < gw and not solid[ny, nx]:
                    opens.add(name)
        opening = opens.pop() if len(opens) == 1 else None
        cpx = np.zeros((gh * u, gw * u), dtype=bool)
        cpx[ys, xs] = True
        cpx = cpx[pad_y:pad_y + GRID, pad_x:pad_x + GRID]
        cells_px = np.flipud(cpx) if flipped else cpx
        targets.append(Target(cells_px=cells_px, glyph=gm, mouth=mouth,
                              capacity=float(len(mouth)), opening=opening))
    targets.sort(key=lambda t: min(x for _, x in t.mouth) if t.mouth else 0)

    vol = float(emit_volume) if emit_volume is not None else 37.0  # measured L0 (calibrate.py)
    return Scene(pre=pre, flipped=flipped, g=g, solid=solid, floor_rows=floor_rows,
                 spout=spout, spouts=spouts, emit_volume=vol, targets=targets,
                 depth=depth, cell=u, pad=(pad_y, pad_x), spout_charged=spout_charged,
                 blocker=blocker)


def solid_boxes(scene):
    """Static colliders as 3D AABBs [N,6] (y0,x0,z0,y1,x1,z1) in macro units,
    greedily merged along x then y for fewer boxes. z spans the slab depth."""
    solid = scene.solid.copy()
    gh, gw = solid.shape
    boxes = []
    for y in range(gh):
        x = 0
        while x < gw:
            if not solid[y, x]:
                x += 1; continue
            x1 = x
            while x1 + 1 < gw and solid[y, x1 + 1]:
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
