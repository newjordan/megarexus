"""Lane R — hydraulic stream tracer + ray queries (the cheap, exact cousin of Lane P).

The observed pour CA (fixtures/calib_l0_pours.npz, NIGHT-verified): paint is emitted
1 macro cell/tick, falls as 1-cell-wide streams 1 cell/tick, POOLS sideways when it
lands on support (surface flow: paint on top of full paint equalizes laterally and
spills over pool/bar edges), DRAINS into holes below (U mouths), and touching the
boundary strip behind the targets flashes it to 14 AND ENDS THE POUR ("the floor is
lava" — verified: blocked pours plateau at 13 emitted cells because the flash stops
emission). Volume-conserving, deterministic, microseconds per pour, zero drift.

Use Lane R as the fast RANKER over thousands of arrangements; Lane P (engine.py,
sphere physics) is the shape-general DECIDER substrate. Same Scene in, same
snapshot/metrics contract out (metrics.render_frames handles both).

Also: ray_aabb / drop_column — straight-line queries against the 3D scene boxes
("raytracing solutions" — pathing checks like 'which support does column x hit').
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from scene import M


@dataclass
class HydroParams:
    emit_per_tick: float = 1.0     # macro cells of fluid per tick (measured: exactly 1)
    fall_per_tick: int = 1         # cells a stream descends per tick
    flow_per_tick: float = 1.0     # surface/lateral volume moved per cell per tick
    cell_cap: float = 1.0          # settled volume a cell holds
    stop_on_lava: int = 2          # ticks after first floor contact before the pour ends
    stop_on_win: int = 4           # ticks after all mouths full before the pour ends
    max_ticks: int = 300
    settle_grace: int = 6          # quiet ticks (nothing moved) before stopping

    def to_json(self):
        return asdict(self)


class HydraulicSim:
    def __init__(self, scene, params=None):
        self.sc = scene
        self.p = params or HydroParams()

    def run(self, record_frames=True):
        p, sc = self.p, self.sc
        cap = p.cell_cap
        falling = np.zeros((M, M))
        settled = np.zeros((M, M))
        remaining = float(sc.emit_volume)
        floor_set = set(sc.floor_rows)
        snapshots = []
        lava_tick = None
        win_tick = None
        quiet = 0
        tick = 0

        def open_cell(y, x):
            return 0 <= y < M and 0 <= x < M and not sc.solid[y, x]

        for tick in range(p.max_ticks):
            moved = False
            # 1) emit while volume remains and no fail flash yet. Backpressure pause
            #    at 2*cap, not cap: a pool cell AT the spout (bar right under the
            #    source, e.g. the real L0 win) hovers near cap while conducting flow
            #    away — only genuine dead-end backup stacks past 2*cap.
            if remaining > 1e-9 and lava_tick is None and settled[sc.spout] < 2 * cap - 1e-9:
                q = min(p.emit_per_tick, remaining)
                falling[sc.spout] += q
                remaining -= q
                moved = True

            # 2) falling volume descends; blocked volume joins the surface below it
            for _ in range(p.fall_per_tick):
                nf = np.zeros_like(falling)
                for y in range(M - 1, -1, -1):
                    for x in range(M):
                        vol = falling[y, x]
                        if vol <= 1e-9:
                            continue
                        ny = y + 1
                        if open_cell(ny, x) and settled[ny, x] < cap - 1e-9:
                            nf[ny, x] += vol            # keep falling into open space
                        else:
                            settled[y, x] += vol        # land on solid or on full paint
                        moved = True
                falling = nf

            # 3) COLUMN-BASE SHEDDING — the micro-rule that matches every observed
            #    real pour (pour 5's early leak, the winner's leak-free backup, the
            #    under-bar pocket route): lateral flow happens ONLY at a paint
            #    column's SOLID-supported base cell; the column above is a reservoir
            #    feeding it (base keeps cap). A column based in a hole (mouth) first
            #    drains, then just STACKS — it never spreads over the prong tops,
            #    which is exactly why the verified winner backs up without lava.
            for x in range(M):
                y = M - 1
                while y >= 0:
                    if settled[y, x] <= 1e-9:
                        y -= 1; continue
                    y_base = y                          # deepest cell of this column run
                    y_top = y
                    while y_top - 1 >= 0 and settled[y_top - 1, x] > 1e-9:
                        y_top -= 1
                    ny = y_base + 1
                    if open_cell(ny, x) and settled[ny, x] < cap - 1e-9:
                        q = min(settled[y_base, x], p.flow_per_tick, cap - settled[ny, x])
                        settled[y_base, x] -= q
                        falling[ny, x] += q             # drain down (mouths, gaps)
                        moved = True
                    elif ny >= M or sc.solid[ny, x]:    # solid base: shed laterally
                        total = settled[y_top:y_base + 1, x].sum()
                        avail = min(p.flow_per_tick, total - cap)
                        # full solid-based cells CONDUCT: walk each direction through
                        # the filled run and deliver at the first non-full open cell
                        outs = []
                        for dx in (-1, 1):
                            nx = x + dx
                            while (open_cell(y_base, nx)
                                   and settled[y_base, nx] >= cap - 1e-9
                                   and y_base + 1 < M and sc.solid[y_base + 1, nx]):
                                nx += dx
                            if open_cell(y_base, nx) and settled[y_base, nx] < cap - 1e-9:
                                outs.append(nx)
                        if avail > 1e-9 and outs:
                            share = avail / len(outs)
                            for nx in outs:
                                supported = (not open_cell(y_base + 1, nx)
                                             or settled[y_base + 1, nx] >= cap - 1e-9)
                                if supported:
                                    settled[y_base, nx] += share   # pool widens
                                else:
                                    falling[y_base, nx] += share   # spills over the edge
                            rem = avail                 # draw from the TOP of the column
                            yy = y_top
                            while rem > 1e-9 and yy <= y_base:
                                take = min(settled[yy, x], rem)
                                settled[yy, x] -= take
                                rem -= take
                                yy += 1
                            moved = True
                    y = y_top - 1

            # 4) events: floor-lava flash / all-mouths-full win
            if lava_tick is None:
                for (fy, fx) in zip(*np.nonzero(settled > 1e-9)):
                    if fy in floor_set or (fy + 1) in floor_set:
                        lava_tick = tick
                        break
            occ, mouth_fill = self._occupancy(settled, falling)
            if win_tick is None and len(mouth_fill) and (mouth_fill > 0.999).all():
                win_tick = tick
            if record_frames:
                snapshots.append((occ, mouth_fill, lava_tick is not None))

            quiet = 0 if moved else quiet + 1
            if lava_tick is not None and tick - lava_tick >= p.stop_on_lava:
                break
            if win_tick is not None and tick - win_tick >= p.stop_on_win:
                break
            if remaining <= 1e-9 and falling.sum() <= 1e-9 and quiet >= p.settle_grace:
                break

        occ, mouth_fill = self._occupancy(settled, falling)
        return {
            "settled": settled, "ticks": tick + 1, "emitted": sc.emit_volume - remaining,
            "occupancy": occ, "mouth_fill": mouth_fill,
            "lava_ticks": [] if lava_tick is None else [lava_tick],
            "lava": lava_tick is not None,
            "snapshots": snapshots, "params": p.to_json(),
        }

    def _occupancy(self, settled, falling):
        occ = settled + falling
        fills = []
        for t in self.sc.targets:
            if not t.mouth:
                fills.append(0.0); continue
            got = sum(float(settled[c]) for c in t.mouth)
            fills.append(min(1.0, got / max(t.capacity * self.p.cell_cap, 1e-9)))
        return occ, np.asarray(fills)


# ---- ray queries over the 3D scene -------------------------------------------

def ray_aabb(origin, direction, boxes):
    """First-hit slab test: origin/direction [3] vs boxes [N,6] (lo3,hi3).
    Returns (t, box_index) or (inf, -1)."""
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(direction, dtype=np.float64)
    d = np.where(np.abs(d) < 1e-12, 1e-12, d)
    lo, hi = boxes[:, :3], boxes[:, 3:]
    t1 = (lo - o) / d
    t2 = (hi - o) / d
    tmin = np.minimum(t1, t2).max(axis=1)
    tmax = np.maximum(t1, t2).min(axis=1)
    hit = (tmax >= np.maximum(tmin, 0.0))
    if not hit.any():
        return np.inf, -1
    tm = np.where(hit, np.maximum(tmin, 0.0), np.inf)
    i = int(np.argmin(tm))
    return float(tm[i]), i


def drop_column(scene, x_col):
    """Where does a straight drop in macro column x land? -> (row, kind) where kind
    is 'mouth:<i>' / 'floor' / 'solid'. The zero-tick pathing primitive."""
    for y in range(M):
        for ti, t in enumerate(scene.targets):
            if (y, x_col) in t.mouth:
                return y, f"mouth:{ti}"
        if scene.solid[y, x_col]:
            return y, ("floor" if y in scene.floor_rows or (y - 1) in scene.floor_rows
                       else "solid")
    return M - 1, "floor"
