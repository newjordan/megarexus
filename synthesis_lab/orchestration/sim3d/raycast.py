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
    backpressure_cap: float = 2.0  # spout stacks past this*cap -> emission pauses (2.0 = validated)
    floor_spill_multisource: bool = True  # P1 (2026-07-06): allow floor-gated ledge spill on
                                   # MULTI-source too. It was single-source-only because pre-latch
                                   # it suppressed mouth fills; the enclosed-entry latch now credits
                                   # fills on entry (independent of pool drainage), so floor-spill
                                   # can model multi-source overshoot floods without eating fills —
                                   # the fix for the S2-MS precision FPs (129/133 = env floods the
                                   # sim misses). Default True (S2-MS: precision 0.587->0.675,
                                   # recall 1.0 unchanged; P0 multi-source gates re-verified intact).
    ledge_spill_floor: bool = True  # S1 iter2 (ADOPTED 2026-07-06): a bar-lip pool dribbles
                                   # over an open ledge AT REST *only when the straight drop
                                   # below reaches the FLOOR strip* (not a mouth). Targets the
                                   # overshoot-flood path without accelerating mouth-fills.
                                   # Full S2 (11,810 pours, both seed batches): precision
                                   # 0.736->0.807, recall 0.992 unchanged, argmax nontrivial
                                   # 0/2->1/1, S0 still PASS. (Contrast the reverted blanket
                                   # ledge_spill: 0.736->0.447.) Set False for pre-iter2 behavior.
    # NOTE (S1, 2026-07-05): a `ledge_spill` variant (bar-lip pools dribble over an open
    # ledge even at rest) was tried to catch the S2 overshoot-FPs. It fixed 5/8 of board
    # 336's FPs and kept S0 PASS, but REGRESSED the full S2 corpus (+17 FPs elsewhere,
    # precision 0.559->0.447, argmax 8/9->6/9). Reverted. See SIM3D_S1_RESULT_20260705.md:
    # the correct fix is a WIN-vs-FLOOD timing race, not a blanket spill.

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
        spouts = sc.spouts if getattr(sc, "spouts", None) else [sc.spout]
        remaining = {sp: float(sc.emit_volume) for sp in spouts}   # per-source volume
        # ledge_spill_floor was calibrated on SINGLE-source S2 variants; on multi-source
        # levels (L2/L4) it cascades the combined flow to the wall too early (over-floods,
        # suppressing mouth fills). Gate it to single-source; multi-source keeps only the
        # basic settled-on-floor lava detection. Single-source path stays byte-identical.
        floor_spill_active = p.ledge_spill_floor and (len(spouts) == 1 or p.floor_spill_multisource)
        floor_set = set(sc.floor_rows)
        # Enclosed-entry mouth fill (game source L775, multi-source fidelity fix 2026-07-06):
        # the game recolours the WHOLE target to 13 the tick a droplet ENTERS a mouth cell
        # that is flanked by that target's own glyph in the flow-forward direction AND on
        # both perpendicular sides (`owyqnbmyij is faksfmaezd is lxmalbxucn`) — a TRANSIENT
        # entry event, permanent once set (added to `cevwbinfgl`), NOT settle-to-capacity.
        # The single-source settled proxy misses this when multi-source routing delivers a
        # droplet late (after the paint plateau -> _outcome_end trims it) or thin (below the
        # 0.85 render threshold). GATED to multi-source: single-source (L0/L1) keeps its
        # byte-identical settled path — single-cell enclosed mouths settle the tick the
        # droplet lands, so the proxy is exact there and GATE P0.1 stays pristine.
        enclosed_active = len(spouts) > 1
        enclosed = []
        for t in sc.targets:
            cells = [(my, mx) for (my, mx) in t.mouth
                     if my + 1 < M and t.glyph[my + 1, mx]
                     and mx - 1 >= 0 and t.glyph[my, mx - 1]
                     and mx + 1 < M and t.glyph[my, mx + 1]]
            enclosed.append(cells)
        latched = [False] * len(sc.targets)
        latch_tick = [None] * len(sc.targets)
        snapshots = []
        lava_tick = None
        win_tick = None
        quiet = 0
        tick = 0

        def open_cell(y, x):
            return 0 <= y < M and 0 <= x < M and not sc.solid[y, x]

        def drops_to_floor(y_from, x):
            """Straight drop from (y_from, x) downward: True iff the first solid it hits is
            the floor strip (or it falls off the board). False if the path feeds a mouth or
            hits a non-floor solid (a block it would pool on)."""
            y = y_from
            while y < M:
                for t in sc.targets:
                    if (y, x) in t.mouth:
                        return False           # feeds a mouth -> legitimate fill, don't force
                if sc.solid[y, x]:
                    return (y in floor_set) or ((y - 1) in floor_set)
                y += 1
            return True                        # off the bottom = floor

        for tick in range(p.max_ticks):
            moved = False
            # 1) emit while volume remains and no fail flash yet. Backpressure pause
            #    at 2*cap, not cap: a pool cell AT the spout (bar right under the
            #    source, e.g. the real L0 win) hovers near cap while conducting flow
            #    away — only genuine dead-end backup stacks past 2*cap.
            if lava_tick is None:
                for sp in spouts:              # emit from EACH source (multi-source levels)
                    if remaining[sp] > 1e-9 and settled[sp] < p.backpressure_cap * cap - 1e-9:
                        q = min(p.emit_per_tick, remaining[sp])
                        falling[sp] += q
                        remaining[sp] -= q
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
                        # S1 iter2: if a ledge out drains straight to the FLOOR (not a mouth),
                        # a bar-lip pool dribbles over it even at rest -> the overshoot flood.
                        if floor_spill_active and any(
                                (open_cell(y_base + 1, nx) and settled[y_base + 1, nx] < cap - 1e-9
                                 and drops_to_floor(y_base + 1, nx)) for nx in outs):
                            avail = min(p.flow_per_tick, total)
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
            # enclosed-entry latch (game L775): a droplet occupying an enclosed mouth cell
            # credits the whole target now and forever (multi-source only; see run() head).
            if enclosed_active:
                for ti, cells in enumerate(enclosed):
                    if not latched[ti] and any(settled[cy, cx] + falling[cy, cx] > 1e-9
                                               for (cy, cx) in cells):
                        latched[ti] = True
                        latch_tick[ti] = tick
            occ, mouth_fill = self._occupancy(settled, falling, latched)
            if win_tick is None and len(mouth_fill) and (mouth_fill > 0.999).all():
                win_tick = tick
            if record_frames:
                snapshots.append((occ, mouth_fill, lava_tick is not None))

            quiet = 0 if moved else quiet + 1
            if lava_tick is not None and tick - lava_tick >= p.stop_on_lava:
                break
            if win_tick is not None and tick - win_tick >= p.stop_on_win:
                break
            if sum(remaining.values()) <= 1e-9 and falling.sum() <= 1e-9 and quiet >= p.settle_grace:
                break

        occ, mouth_fill = self._occupancy(settled, falling, latched)
        return {
            "settled": settled, "ticks": tick + 1,
            "emitted": sc.emit_volume * len(spouts) - sum(remaining.values()),
            "occupancy": occ, "mouth_fill": mouth_fill,
            "lava_ticks": [] if lava_tick is None else [lava_tick],
            "lava": lava_tick is not None,
            "lava_tick": lava_tick, "win_tick": win_tick,
            "latch_tick": latch_tick,
            "snapshots": snapshots, "params": p.to_json(),
        }

    def _occupancy(self, settled, falling, latched=None):
        occ = settled + falling
        fills = []
        for ti, t in enumerate(self.sc.targets):
            if latched is not None and latched[ti]:
                fills.append(1.0); continue      # enclosed-entry latch (game L775)
            if not t.mouth:
                fills.append(0.0); continue
            # Settled-occupancy fill: the single-source proxy (exact for single-cell enclosed
            # mouths, which settle the tick a droplet lands). Multi-source targets are handled
            # ABOVE by the enclosed-entry latch — the faithful L775 rule — which fires on the
            # transient entry BEFORE the settle, so a late/thin routed droplet is still credited
            # (the reverted "count settled+falling everywhere" heuristic over-filled via pass-
            # through and cost single-source AUC; the latch is precise: fp=0 on the L2 corpus).
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
