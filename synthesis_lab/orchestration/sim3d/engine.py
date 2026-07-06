"""Lane P — deterministic 3D sphere-pour physics (the user's imagination substrate).

We literally pour spheres out of the bucket: fluid = spheres of radius r in a slab
[0..16]x[0..16]x[0..depth] (macro units), static colliders = the scene's extruded
AABBs. Position-Based Dynamics: predict under gravity, then iterate projections
(sphere-vs-AABB push-out, sphere-vs-sphere separation via spatial hash). No RNG in
the physics — emission jitter comes from a seeded, fixed table, so identical inputs
give identical rollouts (the whole point vs the drifting neural CA: a hand-coded
simulator CANNOT compound learned error).

Everything tunable lives in SimParams — that is the ENTIRE calibration surface
(calibrate.py fits it on the L0+variants corpus; NEVER on the t0 canonical five).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from scene import M


@dataclass
class SimParams:
    radius: float = 0.35          # sphere radius, macro units (0.5 = one cell wide)
    per_cell: float = 2.0         # spheres per macro cell of fluid volume
    emit_rate: float = 2.0        # spheres spawned per macro tick (game: 1 cell/tick)
    gravity: float = 30.0         # macro units / s^2
    dt: float = 0.02              # physics step (s)
    substeps: int = 1
    iters: int = 6                # PBD projection iterations per step
    damping: float = 0.94         # velocity retention per step (paint ~ inelastic)
    friction: float = 0.55        # tangential velocity retention on contact
    jitter: float = 0.06          # emission x/z jitter amplitude (deterministic table)
    seed: int = 0                 # seeds the fixed jitter table only
    sleep_v: float = 0.15         # |v| below which a sphere can sleep
    sleep_ticks: int = 6          # consecutive slow ticks before sleeping
    max_ticks: int = 400          # macro ticks (render granularity)
    steps_per_tick: int = 8       # physics steps per macro tick
    settle_ticks: int = 25        # stop after this many ticks with all spheres asleep

    def to_json(self):
        return asdict(self)


class PourSim:
    """Simulate one pour into a Scene. Usage:
        sim = PourSim(scene, params); result = sim.run()
    result: dict with sphere rest positions, per-tick occupancy snapshots, mouth
    fill fractions per target, and lava events (floor-strip contacts)."""

    def __init__(self, scene, params=None):
        from scene import solid_boxes
        self.sc = scene
        self.p = params or SimParams()
        self.boxes = solid_boxes(scene)          # [N,6] y0,x0,z0,y1,x1,z1
        self.n_total = int(round(scene.emit_volume * self.p.per_cell))
        rng = np.random.default_rng(self.p.seed)
        self._jit = (rng.random((self.n_total, 2)) - 0.5) * 2 * self.p.jitter
        # mouth cell lookup: cell -> target index
        self.mouth_of = {}
        for ti, t in enumerate(scene.targets):
            for c in t.mouth:
                self.mouth_of[c] = ti

    # -- geometry helpers -------------------------------------------------
    def _project_boxes(self, x):
        """Push spheres out of every static AABB (and the domain walls)."""
        r = self.p.radius
        x[:, 0] = np.clip(x[:, 0], r - 2.0, M - r)          # allow spawn above field
        x[:, 1] = np.clip(x[:, 1], r, M - r)
        x[:, 2] = np.clip(x[:, 2], r, self.sc.depth - r)
        for b in self.boxes:
            lo = b[:3]; hi = b[3:]
            c = np.maximum(lo, np.minimum(x, hi))            # closest point in box
            d = x - c
            dist2 = (d * d).sum(1)
            inside = dist2 < r * r
            if not inside.any():
                continue
            idx = np.nonzero(inside)[0]
            for i in idx:
                if dist2[i] > 1e-12:                          # outside face: push along d
                    n = d[i] / np.sqrt(dist2[i])
                    x[i] = c[i] + n * r
                else:                                         # center inside: min-axis pop-out
                    gaps = np.concatenate([x[i] - lo + r, hi - x[i] + r])
                    ax = int(np.argmin(gaps))
                    if ax < 3:
                        x[i, ax] = lo[ax] - r
                    else:
                        x[i, ax - 3] = hi[ax - 3] + r
        return x

    def _separate(self, x):
        """Sphere-sphere positional separation via spatial hash (deterministic order)."""
        r2 = 2 * self.p.radius
        n = len(x)
        if n < 2:
            return x
        cell = np.floor(x / r2).astype(np.int64)
        key = (cell[:, 0] + 64) * 4096 * 4096 + (cell[:, 1] + 64) * 4096 + (cell[:, 2] + 64)
        order = np.argsort(key, kind="stable")
        ks = key[order]
        # neighbor candidates: same-or-adjacent hash cells via offset re-hash
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    off = (dy) * 4096 * 4096 + (dx) * 4096 + (dz)
                    pos = np.searchsorted(ks, key + off)
                    for i in range(n):
                        j0 = pos[i]
                        while j0 < n and ks[j0] == key[i] + off:
                            j = order[j0]; j0 += 1
                            if j <= i:
                                continue
                            d = x[i] - x[j]
                            dist2 = float(d @ d)
                            if 1e-12 < dist2 < r2 * r2:
                                dist = np.sqrt(dist2)
                                push = 0.5 * (r2 - dist) * (d / dist)
                                x[i] += push; x[j] -= push
        return x

    # -- main loop ---------------------------------------------------------
    def run(self, record_frames=True):
        p, sc = self.p, self.sc
        sy, sx = sc.spout
        x = np.zeros((0, 3)); v = np.zeros((0, 3)); slow = np.zeros(0, dtype=np.int32)
        emitted = 0
        lava_ticks = []       # macro ticks at which some sphere touched the floor strip
        snapshots = []        # (occupancy [16,16] int counts, mouth_fill [n_targets])
        floor_y = min(sc.floor_rows) if sc.floor_rows else M - 1
        asleep_since = None

        for tick in range(p.max_ticks):
            # emit
            k = min(int(round(p.emit_rate)), self.n_total - emitted)
            if k > 0:
                jit = self._jit[emitted:emitted + k]
                nx = np.column_stack([
                    np.full(k, sy - 0.6) - 0.2 * np.arange(k),
                    np.full(k, sx + 0.5) + jit[:, 0],
                    np.full(k, sc.depth * 0.5) + jit[:, 1] * sc.depth,
                ])
                x = np.vstack([x, nx])
                v = np.vstack([v, np.zeros((k, 3))])
                slow = np.concatenate([slow, np.zeros(k, dtype=np.int32)])
                emitted += k
            if len(x) == 0:
                continue

            awake = slow < p.sleep_ticks
            for _ in range(p.steps_per_tick * p.substeps):
                x_prev = x.copy()
                v[awake, 0] += p.gravity * p.dt
                x[awake] += v[awake] * p.dt
                for _ in range(p.iters):
                    x = self._project_boxes(x)
                    x = self._separate(x)
                moved = x - x_prev
                v = np.where(awake[:, None], moved / p.dt * p.damping, 0.0)
                v[:, 1] *= p.friction; v[:, 2] *= p.friction
            speed = np.linalg.norm(v, axis=1)
            slow = np.where(speed < p.sleep_v, slow + 1, 0)

            occ, mouth_fill = self._occupancy(x)
            touching_floor = (x[:, 0] + p.radius >= floor_y)
            if touching_floor.any():
                lava_ticks.append(tick)
            if record_frames:
                snapshots.append((occ, mouth_fill, bool(touching_floor.any())))

            if emitted >= self.n_total and (slow >= p.sleep_ticks).all():
                asleep_since = asleep_since if asleep_since is not None else tick
                if tick - asleep_since >= p.settle_ticks:
                    break
            else:
                asleep_since = None

        occ, mouth_fill = self._occupancy(x)
        return {
            "x": x, "emitted": emitted, "ticks": tick + 1,
            "occupancy": occ, "mouth_fill": mouth_fill,
            "lava_ticks": lava_ticks, "lava": len(lava_ticks) > 0,
            "snapshots": snapshots, "params": p.to_json(),
        }

    def _occupancy(self, x):
        """Macro-cell sphere counts + per-target mouth fill fraction (0..1)."""
        occ = np.zeros((M, M), dtype=np.int32)
        if len(x):
            iy = np.clip(x[:, 0].astype(np.int64), 0, M - 1)
            ix = np.clip(x[:, 1].astype(np.int64), 0, M - 1)
            np.add.at(occ, (iy, ix), 1)
        fills = []
        for t in self.sc.targets:
            if not t.mouth:
                fills.append(0.0); continue
            got = sum(int(occ[c]) for c in t.mouth)
            cap = t.capacity * self.p.per_cell
            fills.append(min(1.0, got / max(cap, 1e-9)))
        return occ, np.asarray(fills)
