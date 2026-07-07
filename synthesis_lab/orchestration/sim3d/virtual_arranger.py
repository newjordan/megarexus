"""Virtual arrangement operator — realize click-move candidate frames from the
LIVE frame with zero env interaction. This is the continuous-episode search
realizer: dw_solve's hill-climb realized candidates with fresh-env resets
(frame_of), which a ranked one-episode run cannot do.

Model induced from OBSERVATION only (probe_arrange / probe_select /
probe_arena / probe_clearance / probe_objobj / probe_composite, 2026-07-06 —
legal frame diffs; no game source, answer-key ban):
  * movable UNITS are rigid bodies of {8 unselected, 9 selected} cells,
    possibly with embedded color-4 bridge cells (L3 housing "88 4 88" selects
    and moves as one; clicking the bridge selects the unit)
  * every unit is exactly ONE lattice cell tall (observed L0-L3); a {8,9}
    component taller than one cell is stacked TOUCHING units — split into
    cell-height row bands, then horizontal runs (post-pour re-plan frames,
    2026-07-06); non-multiple heights are refused
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
    objects with no structure above their own columns
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
BRIDGE = 4
GRID = 64
FLOW_STRUCT = (6, 11, 13)      # sources, targets, filled targets


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


def _band_split(comps, cell):
    """Units are one cell tall: split taller components into cell-height row
    bands, then horizontal runs. Non-multiple heights -> ValueError (refuse)."""
    out = []
    for m in comps:
        ys = np.nonzero(m.any(axis=1))[0]
        h = ys.max() - ys.min() + 1
        if h == cell:
            out.append(m)
            continue
        if h % cell:
            raise ValueError(f"component height {h} not a multiple of cell {cell}")
        for k in range(h // cell):
            band = np.zeros_like(m)
            band[ys.min() + k * cell: ys.min() + (k + 1) * cell, :] = True
            sub = m & band
            if sub.any():
                out.extend(_components(sub))
    return out


def _merge_bridged(comps, frame, cell):
    """Rigid units: merge movable comps joined by an all-color-4 gap of
    <= 2*cell px with matching extent on the cross axis (both axes probed)."""
    units = [{"mov": c.copy(), "bridge": np.zeros_like(c)} for c in comps]
    merged = True
    while merged:
        merged = False
        for i in range(len(units)):
            for j in range(i + 1, len(units)):
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
        comps = _components(np.isin(f, (UNSEL, SEL)))
        if not comps:
            raise ValueError("no movable units")
        if cell is None:
            cell = min(min(int(m.any(0).sum() and (np.ptp(np.nonzero(m)[k]) + 1))
                           for k in (0, 1)) for m in comps)
        self.cell = int(cell)
        comps = _band_split(comps, self.cell)
        self.units = _merge_bridged(comps, f, self.cell)
        sel = [i for i, u in enumerate(self.units) if (f[u["mov"]] == SEL).any()]
        if len(sel) != 1:
            raise ValueError(f"expected 1 selected unit, saw {len(sel)}")
        self.selected = sel[0]
        self.static = f.copy()
        for u in self.units:
            self.static[u["mov"] | u["bridge"]] = FIELD
        struct_rows = np.nonzero(
            (np.isin(self.static, FLOW_STRUCT) & ~self._gauge).any(axis=1))[0]
        self.forbidden_rows = np.zeros(GRID, dtype=bool)
        for s in struct_rows:
            lo, hi = max(0, s - self.cell), min(GRID, s + self.cell + 1)
            self.forbidden_rows[lo:hi] = True
        self._frame0 = f

    @classmethod
    def parse(cls, frame, cell=None):
        try:
            return cls(frame, cell=cell)
        except Exception:  # noqa: BLE001
            return None

    def clone(self):
        vb = object.__new__(VirtualBoard)
        vb.cell = self.cell
        vb.units = [{"mov": u["mov"].copy(), "bridge": u["bridge"].copy()}
                    for u in self.units]
        vb.selected = self.selected
        vb.static = self.static
        vb.forbidden_rows = self.forbidden_rows
        vb._gauge = self._gauge
        vb._frame0 = self._frame0
        return vb

    def click(self, x, y):
        # overlap hit-test: SMALLEST-footprint unit wins; ties -> latest in
        # original scan order. 11 gate evidence rows + 4-case constructed
        # tie probe, 2026-07-06 (selection state and left/right placement
        # proven irrelevant; width vs cell-count degenerate at 1-cell-tall)
        if not (0 <= x < GRID and 0 <= y < GRID):
            return
        hits = [(int((u["mov"] | u["bridge"]).sum()), i)
                for i, u in enumerate(self.units)
                if (u["mov"] | u["bridge"])[int(y), int(x)]]
        if hits:
            self.selected = min(hits, key=lambda t: (t[0], -t[1]))[1]

    def move(self, direction):
        c = self.cell
        dy, dx = {1: (-c, 0), 2: (c, 0), 3: (0, -c), 4: (0, c)}[int(direction)]
        u = self.units[self.selected]
        full = u["mov"] | u["bridge"]
        ys, xs = np.nonzero(full)
        ny, nx = ys + dy, xs + dx
        if ny.min() < 0 or ny.max() >= GRID or nx.min() < 0 or nx.max() >= GRID:
            return
        if self.forbidden_rows[ny].any():
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
        order = [i for i in range(len(self.units)) if i != self.selected]
        order.append(self.selected)
        for i in order:
            u = self.units[i]
            f[u["mov"]] = SEL if i == self.selected else UNSEL
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
