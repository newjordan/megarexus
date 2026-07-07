"""Virtual arrangement operator — realize click-move candidate frames from the
LIVE frame with zero env interaction. This is the continuous-episode search
realizer: dw_solve's hill-climb realized candidates with fresh-env resets
(frame_of), which a ranked one-episode run cannot do.

Model induced from OBSERVATION only (probe_arrange / probe_select /
probe_arena / probe_clearance / probe_objobj / probe_composite, 2026-07-06 —
legal frame diffs; no game source, answer-key ban):
  * movable UNITS are rigid bodies with an unselected base color: 8 for bars,
    15 for r5 blockers; selected units render as 9. Units may carry embedded
    color-4 bridge cells (L3 housing "88 4 88" selects and moves as one;
    clicking the bridge selects the unit)
  * L0-L3 bars are exactly one lattice cell tall; a wider {8,9} component
    taller than one cell is stacked TOUCHING bars — split into cell-height row
    bands, then horizontal runs (post-pour re-plan frames, 2026-07-06).
    R5 extends this without regressing the split: one-cell-wide vertical 8-bars
    and L-shaped color-15 blockers remain single rigid units.
  * exactly one unit is selected; ACTION6 on a unit selects it (old one
    deselects); clicks anywhere else change nothing; on OVERLAPPING units
    the SMALLEST-footprint unit wins the hit-test, ties to the latest unit
    in original row-major scan order (selection state / placement irrelevant)
  * ACTION1..4 = up/down/left/right translate the selected unit by exactly
    one lattice CELL (4 px on L0-L2, 3 px on L3 — inferred per frame)
  * units pass freely THROUGH each other (selected renders on top; hidden
    cells re-emerge); vacated cells restore to field color 12
  * a move is BLOCKED (silent no-op) iff it would leave the grid, land any
    unit cell on non-field static (anything but 12), or put any unit cell in
    the FORBIDDEN ROW BAND = rows within one cell of any row containing
    source/target structure (colors 6/11/13) — a whole-row rule, verified on
    objects with no structure above their own columns. R5 side-mouth levels
    use the same one-cell margin locally around side-wall structure instead
    of banning the whole row; L4 observed bars can pass through side-mouth
    rows when far from the wall mouth. The L5 vertical 8-bar additionally
    obeys top/bottom edge shelves for units taller than one cell, induced from
    edge source/target components: one move into the shelf is legal, the next
    is a no-op.
  * one border side is a depleting action-budget GAUGE (cells in {14,0};
    GAME_OVER when it empties) — gauge sides are copied through unchanged and
    excluded from every comparison
Refusal-safe: parse() returns None on frames that violate the model's
assumptions; callers skip, never guess.

Gate: arranger_gate.py holds this file to byte-identical (modulo gauge)
against real env realization before any runner may trust it.
"""
from __future__ import annotations

import numpy as np

FIELD = 12
UNSEL = 8
SEL = 9
BLOCKER = 15
BRIDGE = 4
GRID = 64
FLOW_STRUCT = (6, 11, 13)      # sources, targets, filled targets
MOVABLE_COLORS = (UNSEL, SEL, BLOCKER)
PAINT = 6


def gauge_sides(frame):
    """Border sides whose cells are all in {14, 0} — the budget gauge."""
    f = np.asarray(frame)
    sides = {"row0": f[0, :], "row63": f[GRID - 1, :],
             "col0": f[:, 0], "col63": f[:, GRID - 1]}
    return [k for k, v in sides.items() if set(np.unique(v)) <= {0, 14}]


def gauge_mask(frame):
    m = np.zeros((GRID, GRID), dtype=bool)
    for k in gauge_sides(frame):
        if k == "row0":
            m[0, :] = True
        elif k == "row63":
            m[GRID - 1, :] = True
        elif k == "col0":
            m[:, 0] = True
        else:
            m[:, GRID - 1] = True
    return m


def frames_equal(a, b):
    """Board equality modulo the budget gauge (union of both frames' sides)."""
    a, b = np.asarray(a), np.asarray(b)
    keep = ~(gauge_mask(a) | gauge_mask(b))
    return bool(np.array_equal(a[keep], b[keep]))


def _components(mask):
    from scipy.ndimage import label
    lab, n = label(mask)
    return [lab == i for i in range(1, n + 1)]


def _infer_base(mask, frame, cell, allow_blocker=False):
    vals = set(int(v) for v in np.unique(np.asarray(frame)[mask]))
    if BLOCKER in vals:
        return BLOCKER
    ys, xs = np.nonzero(mask)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    # A selected blocker renders as 9. The r5 blocker is an L-shaped 2x2-cell
    # footprint; bars are straight one-cell-wide/tall rectangles.
    if allow_blocker and SEL in vals and h > cell and w > cell:
        bbox_cells = max(1, (h // cell) * (w // cell))
        filled_cells = mask.sum() / float(cell * cell)
        if filled_cells < bbox_cells:
            return BLOCKER
    return UNSEL


def _band_split(comps, frame, cell, allow_vertical=False):
    """Split only the old stacked-horizontal-bar case. Vertical bars and
    color-15 L-blockers are single rigid units."""
    out = []
    for m in comps:
        ys = np.nonzero(m.any(axis=1))[0]
        xs = np.nonzero(m.any(axis=0))[0]
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        base = _infer_base(m, frame, cell, allow_blocker=allow_vertical)
        if base == BLOCKER or h == cell or (allow_vertical and w == cell):
            out.append({"mov": m, "base": base})
            continue
        if h % cell:
            raise ValueError(f"component height {h} not a multiple of cell {cell}")
        for k in range(h // cell):
            band = np.zeros_like(m)
            band[ys.min() + k * cell: ys.min() + (k + 1) * cell, :] = True
            sub = m & band
            if sub.any():
                out.extend({"mov": c, "base": base} for c in _components(sub))
    return out


def _merge_bridged(units0, frame, cell):
    """Rigid units: merge movable comps joined by an all-color-4 gap of
    <= 2*cell px with matching extent on the cross axis (both axes probed)."""
    units = [{"mov": u["mov"].copy(), "bridge": np.zeros_like(u["mov"]),
              "base": int(u["base"])} for u in units0]
    merged = True
    while merged:
        merged = False
        for i in range(len(units)):
            for j in range(i + 1, len(units)):
                if units[i]["base"] != units[j]["base"]:
                    continue
                b = _bridge_between(units[i], units[j], frame, cell)
                if b is not None:
                    units[i]["mov"] |= units[j]["mov"]
                    units[i]["bridge"] |= units[j]["bridge"] | b
                    del units[j]
                    merged = True
                    break
            if merged:
                break
    return units


def _bridge_between(u, v, frame, cell):
    """All-4 gap between two comps (same rows, horizontal gap, or same cols,
    vertical gap) -> bridge mask, else None."""
    f = np.asarray(frame)
    for axis in (1, 0):
        am = u["mov"] | u["bridge"]
        bm = v["mov"] | v["bridge"]
        ai = np.nonzero(am.any(axis=axis))[0]
        bi = np.nonzero(bm.any(axis=axis))[0]
        if not (len(ai) and len(bi) and ai.min() == bi.min() and ai.max() == bi.max()):
            continue
        aj = np.nonzero(am.any(axis=1 - axis))[0]
        bj = np.nonzero(bm.any(axis=1 - axis))[0]
        lo, hi = (aj.max(), bj.min()) if aj.max() < bj.min() else (bj.max(), aj.min())
        gap = hi - lo - 1
        if not (0 < gap <= 2 * cell):
            continue
        sl = (slice(ai.min(), ai.max() + 1), slice(lo + 1, hi)) if axis == 1 \
            else (slice(lo + 1, hi), slice(bi.min(), bi.max() + 1))
        if (f[sl] == BRIDGE).all():
            b = np.zeros_like(am)
            b[sl] = True
            return b
    return None


class VirtualBoard:
    """Arrangement state parsed once from a live frame; cheap to clone."""

    def __init__(self, frame, cell=None):
        f = np.asarray(frame, dtype=np.int8).copy()
        self._gauge = gauge_mask(f)
        comps = _components(np.isin(f, MOVABLE_COLORS))
        if not comps:
            raise ValueError("no movable units")
        if cell is None:
            cell = min(min(int(m.any(0).sum() and (np.ptp(np.nonzero(m)[k]) + 1))
                           for k in (0, 1)) for m in comps)
        self.cell = int(cell)
        units = _band_split(comps, f, self.cell, allow_vertical=bool((f == BLOCKER).any()))
        self.units = _merge_bridged(units, f, self.cell)
        sel = [i for i, u in enumerate(self.units) if (f[u["mov"]] == SEL).any()]
        if len(sel) != 1:
            raise ValueError(f"expected 1 selected unit, saw {len(sel)}")
        self.selected = sel[0]
        self.static = f.copy()
        for u in self.units:
            self.static[u["mov"] | u["bridge"]] = FIELD
        self._refresh_constraints()
        self._frame0 = f

    def _refresh_constraints(self):
        struct = np.isin(self.static, FLOW_STRUCT) & ~self._gauge
        struct_rows = np.nonzero(struct.any(axis=1))[0]
        self.forbidden_rows = np.zeros(GRID, dtype=bool)
        for s in struct_rows:
            lo, hi = max(0, s - self.cell), min(GRID, s + self.cell + 1)
            self.forbidden_rows[lo:hi] = True
        self.forbidden_mask = np.repeat(self.forbidden_rows[:, None], GRID, axis=1)
        self.forbidden_rects = []
        self.vertical_top_limit = None
        self.vertical_bottom_limit = None
        sx = np.nonzero(struct.any(axis=0))[0]
        has_blocker = any(u["base"] == BLOCKER for u in self.units)
        side_struct = (has_blocker and len(sx)
                       and (sx.min() < 2 * self.cell or sx.max() >= GRID - 2 * self.cell))
        if side_struct:
            self.forbidden_mask = np.zeros((GRID, GRID), dtype=bool)
            ys, xs = np.nonzero(struct)
            for y, x in zip(ys, xs):
                y0, y1 = max(0, y - self.cell), min(GRID, y + self.cell + 1)
                x0, x1 = max(0, x - self.cell), min(GRID, x + self.cell + 1)
                self.forbidden_mask[y0:y1, x0:x1] = True
            for comp in _components(struct):
                cy, _ = np.nonzero(comp)
                cx = np.nonzero(comp.any(axis=0))[0]
                rect = (max(0, int(cy.min()) - self.cell),
                        min(GRID - 1, int(cy.max()) + self.cell),
                        max(0, int(cx.min()) - self.cell),
                        min(GRID - 1, int(cx.max()) + self.cell))
                self.forbidden_rects.append(rect)
                vals = set(int(v) for v in np.unique(self.static[comp]))
                if PAINT not in vals:
                    continue
                if cy.min() < 2 * self.cell:
                    lim = min(GRID, cy.max() + self.cell + 1)
                    self.vertical_top_limit = lim if self.vertical_top_limit is None else max(
                        self.vertical_top_limit, lim)
                if cy.max() >= GRID - 2 * self.cell:
                    lim = max(0, cy.min() - self.cell - 1)
                    self.vertical_bottom_limit = lim if self.vertical_bottom_limit is None else min(
                        self.vertical_bottom_limit, lim)

    @classmethod
    def parse(cls, frame, cell=None):
        try:
            return cls(frame, cell=cell)
        except Exception:  # noqa: BLE001
            return None

    def clone(self):
        vb = object.__new__(VirtualBoard)
        vb.cell = self.cell
        vb.units = [{"mov": u["mov"].copy(), "bridge": u["bridge"].copy(),
                     "base": u["base"]}
                    for u in self.units]
        vb.selected = self.selected
        vb.static = self.static
        vb.forbidden_rows = self.forbidden_rows
        vb.forbidden_mask = self.forbidden_mask
        vb.forbidden_rects = list(self.forbidden_rects)
        vb.vertical_top_limit = self.vertical_top_limit
        vb.vertical_bottom_limit = self.vertical_bottom_limit
        vb._gauge = self._gauge
        vb._frame0 = self._frame0
        return vb

    def rebase(self, frame):
        """Carry the known unit masks into a new live frame.

        L5 post-pour frames can contain overlapped 8/15 units whose hidden
        cells are not cold-parseable from pixels alone. Continuity from the
        pre-pour arrangement preserves those masks while refreshing static
        paint/fill/gauge state from the live frame.
        """
        vb = self.clone()
        f = np.asarray(frame, dtype=np.int8).copy()
        vb._gauge = gauge_mask(f)
        vb.static = f.copy()
        for u in vb.units:
            vb.static[u["mov"] | u["bridge"]] = FIELD
        sel = [i for i, u in enumerate(vb.units) if (f[u["mov"]] == SEL).any()]
        if len(sel) == 1:
            vb.selected = sel[0]
        vb._frame0 = f
        vb._refresh_constraints()
        return vb

    def click(self, x, y):
        # overlap hit-test: SMALLEST-footprint unit wins; ties -> latest in
        # original scan order. 11 gate evidence rows + 4-case constructed
        # tie probe, 2026-07-06 (selection state and left/right placement
        # proven irrelevant; width vs cell-count degenerate at 1-cell-tall)
        if not (0 <= x < GRID and 0 <= y < GRID):
            return
        hits = [(0 if u["base"] == UNSEL else 1,
                 int((u["mov"] | u["bridge"]).sum()), i)
                for i, u in enumerate(self.units)
                if (u["mov"] | u["bridge"])[int(y), int(x)]]
        if hits:
            self.selected = min(hits, key=lambda t: (t[0], t[1], -t[2]))[2]

    def move(self, direction):
        c = self.cell
        dy, dx = {1: (-c, 0), 2: (c, 0), 3: (0, -c), 4: (0, c)}[int(direction)]
        u = self.units[self.selected]
        full = u["mov"] | u["bridge"]
        ys, xs = np.nonzero(full)
        ny, nx = ys + dy, xs + dx
        if ny.min() < 0 or ny.max() >= GRID or nx.min() < 0 or nx.max() >= GRID:
            return
        if dy < 0 and self.vertical_top_limit is not None:
            h = ys.max() - ys.min() + 1
            w = xs.max() - xs.min() + 1
            if h > self.cell and ny.min() < self.vertical_top_limit:
                return
        if dy > 0 and self.vertical_bottom_limit is not None:
            h = ys.max() - ys.min() + 1
            if h > self.cell and ny.max() > self.vertical_bottom_limit:
                return
        by0, by1, bx0, bx1 = int(ny.min()), int(ny.max()), int(nx.min()), int(nx.max())
        for ry0, ry1, rx0, rx1 in self.forbidden_rects:
            if by0 <= ry1 and by1 >= ry0 and bx0 <= rx1 and bx1 >= rx0:
                return
        if self.forbidden_mask[ny, nx].any():
            return
        shifted = np.zeros_like(full)
        shifted[ny, nx] = True
        if (self.static[shifted & ~full] != FIELD).any():
            return
        u["mov"] = self._shift(u["mov"], dy, dx)
        u["bridge"] = self._shift(u["bridge"], dy, dx)

    @staticmethod
    def _shift(m, dy, dx):
        out = np.zeros_like(m)
        ys, xs = np.nonzero(m)
        out[ys + dy, xs + dx] = True
        return out

    def apply(self, action_seq):
        """(aid, x, y) tuples in imagine_search_gate._apply_real convention."""
        for (aid, x, y) in action_seq:
            if aid == 6:
                self.click(x, y)
            elif aid in (1, 2, 3, 4):
                self.move(aid)
            else:
                raise ValueError(f"virtual arranger cannot apply action {aid}")

    def frame(self):
        f = self.static.copy()
        order = sorted((i for i in range(len(self.units)) if i != self.selected),
                       key=lambda i: self.units[i]["base"])
        order.append(self.selected)
        for i in order:
            u = self.units[i]
            f[u["mov"]] = SEL if i == self.selected else u["base"]
            f[u["bridge"]] = BRIDGE
        f[self._gauge] = self._frame0[self._gauge]
        return f


def realize_virtual(frame_or_board, action_seq, cell=None):
    """Candidate frame for an action sequence, or None (refused).
    Accepts a raw frame or a pre-parsed VirtualBoard (cloned, not mutated)."""
    if isinstance(frame_or_board, VirtualBoard):
        vb = frame_or_board.clone()
    else:
        vb = VirtualBoard.parse(frame_or_board, cell=cell)
        if vb is None:
            return None
    try:
        vb.apply(action_seq)
    except Exception:  # noqa: BLE001
        return None
    return vb.frame()
