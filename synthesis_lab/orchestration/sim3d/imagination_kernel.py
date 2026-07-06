"""Imagination kernel — Dream Kernel contract ported to sim3d (DW plan Phase 2).

Ported concepts from dreamweaver `dream_kernel` (docs/dream_kernel_v001.md):
  - KnownMap with PROVENANCE: "unknown perception can feed the kernel only after
    it has been converted into explicit known-map cells with provenance" —
    here that means a Scene may enter ONLY with a passing perception_gate
    verdict; `KnownMap.from_scene` raises UntrustedSceneError otherwise.
  - Deterministic, inspectable state transitions (SimState sequence is the
    renderable evidence trail).
  - RayBundle with object identity + network polarity (beneficial/adversarial/
    structural/neutral) and a signed potential per contact. Rays are a
    geometry/survival signal, not the full gameplay answer.
  - Pluggable transition rule (DynamicsHypothesis). The kernel does not know
    SP80's dynamics; Phase 3 SELECTS a hypothesis purely by tick-matching
    observed env pours. `GravityFall` below is reference machinery to prove
    the kernel, NOT a claim about SP80 physics.

Legality: inputs are gate-trusted Scenes (observed pixels) + generic
falling-material primitives. No engine internals anywhere in this lane
(mechanically enforced by the Phase-0 hook).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from scene import M

EIGHT_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


class UntrustedSceneError(RuntimeError):
    """A Scene without a passing perception-gate verdict may not enter the kernel."""


@dataclass(frozen=True)
class KnownMap:
    """Explicit spatial cells + provenance (canonical macro coords, flow down)."""
    solid: np.ndarray                      # [16,16] bool, static colliders incl. floor
    floor_rows: tuple                      # lava strip rows (adversarial)
    spouts: tuple                          # ((y,x), ...) material entry cells
    mouths: tuple                          # per target: ((y,x), ...) goal cells
    glyph_cells: tuple                     # per target: ((y,x), ...) structural cells
    capacities: tuple                      # per target: mouth capacity (cells)
    emit_volume: float                     # material units one pour emits
    provenance: dict = field(default_factory=dict, hash=False, compare=False)

    @classmethod
    def from_scene(cls, scene, gate_verdict, **meta):
        if not gate_verdict.get("trusted"):
            raise UntrustedSceneError(
                "scene rejected: perception gate failures "
                f"{gate_verdict.get('gate_failures')} — fix the extractor and re-gate; "
                "the kernel refuses unproven maps (Dream Kernel provenance rule)")
        return cls(
            solid=scene.solid.copy(),
            floor_rows=tuple(scene.floor_rows),
            spouts=tuple((int(y), int(x)) for y, x in scene.spouts),
            mouths=tuple(tuple((int(y), int(x)) for y, x in t.mouth) for t in scene.targets),
            glyph_cells=tuple(tuple((int(y), int(x)) for y, x in zip(*np.nonzero(t.glyph)))
                              for t in scene.targets),
            capacities=tuple(float(t.capacity) for t in scene.targets),
            emit_volume=float(scene.emit_volume),
            provenance={"source": "perception_gate", "trusted": True,
                        "gate_failures": [], **meta},
        )

    def mouth_owner(self):
        return {c: i for i, cells in enumerate(self.mouths) for c in cells}


@dataclass
class SimState:
    """Kernel state at one tick. The run() sequence is the renderable record."""
    kmap: KnownMap
    tick: int = 0
    settled: np.ndarray = None             # [16,16] float, at-rest material
    moving: list = field(default_factory=list)   # [(y,x)] in-flight units
    emitted: int = 0
    fills: list = field(default_factory=list)    # per-target units in mouth
    floor_touch: int = 0                   # adversarial contacts (lava events)
    disposed: int = 0                      # units that left the map
    terminal: str = ""                     # "", "quiescent"

    def __post_init__(self):
        if self.settled is None:
            self.settled = np.zeros((M, M), dtype=np.float32)
        if not self.fills:
            self.fills = [0] * len(self.kmap.mouths)

    def clone(self):
        s = SimState(kmap=self.kmap, tick=self.tick, settled=self.settled.copy(),
                     moving=list(self.moving), emitted=self.emitted,
                     fills=list(self.fills), floor_touch=self.floor_touch,
                     disposed=self.disposed, terminal=self.terminal)
        return s

    def signature(self):
        return (self.tick, self.settled.tobytes(), tuple(self.moving), self.emitted,
                tuple(self.fills), self.floor_touch, self.disposed, self.terminal)


class DynamicsHypothesis:
    """Pluggable transition rule. Subclasses mutate `state` for one tick.

    Phase-3 contract: hypotheses are GENERIC falling-material primitives with
    parameters; which one describes SP80 is decided ONLY by tick-matching
    observed env pours (dynamics_fit.py), never by reading the game.
    """
    name = "base"
    params: dict = {}

    def step(self, state: SimState) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class GravityFall(DynamicsHypothesis):
    """Reference machinery hypothesis: emit, fall straight, settle on contact.

    Deliberately naive — exists to prove kernel determinism/rendering, and as
    the simplest member of the Phase-3 family. NOT a claim about SP80 physics.
    """
    name = "gravity_fall"

    def __init__(self, emit_per_tick_per_spout=1):
        self.params = {"emit_per_tick_per_spout": int(emit_per_tick_per_spout)}

    def step(self, state: SimState) -> None:
        km = state.kmap
        owner = km.mouth_owner()
        # move first (stable order = deterministic); newly emitted units enter at
        # the spout and sit there until next tick — matches the observed animation
        # convention where the stream column starts AT the spout cell (obs[t] is
        # the board after t frames; alignment verified empirically, DW_LAB_LOG).
        nxt = []
        for (y, x) in state.moving:
            ny = y + 1
            if ny >= M:
                state.disposed += 1
                continue
            if ny in km.floor_rows:
                state.floor_touch += 1
                continue
            if km.solid[ny, x] or state.settled[ny, x] >= 1.0:
                state.settled[y, x] += 1.0
                if (y, x) in owner:
                    state.fills[owner[(y, x)]] += 1
                continue
            nxt.append((ny, x))
        state.moving = nxt
        for _ in range(self.params["emit_per_tick_per_spout"]):
            for (sy, sx) in km.spouts:
                if state.emitted >= km.emit_volume:
                    break
                if not km.solid[sy, sx]:
                    state.moving.append((sy, sx))
                    state.emitted += 1
        if state.emitted >= km.emit_volume and not state.moving:
            state.terminal = "quiescent"


class FallingMaterial(DynamicsHypothesis):
    """The Phase-3 GENERIC falling-material family (classic falling-sand space).

    Parameters (all generic physics knobs, no game knowledge):
      deflect  'none' | 'dl' | 'dr' | 'both'   diagonal try when down is blocked
               ('both' alternates preference by unit index — deterministic)
      spread   'none' | 'lateral'              sideways try when diagonals blocked
      mouth    'fill' | 'entry'                goal cells absorb on settle vs on entry
      edge     'dispose' | 'halt'              off-board lateral motion vanishes vs blocks
      cell_cap float                           material units one cell holds at rest
      absorb   'cap' | 'unlimited'             mouth cells saturate at cell_cap vs
                                               swallow every arriving unit (round-3
                                               knob; the flywheel winners suggest
                                               mouths never back up in a clean pour)

    Which configuration (if any) describes SP80 is decided ONLY by
    dynamics_fit.py / r3_fit.py tick-matching observed env pours.
    """
    name = "falling_material"

    def __init__(self, deflect="none", spread="none", mouth="fill",
                 edge="dispose", cell_cap=1.0, emit_per_tick_per_spout=1,
                 absorb="cap"):
        self.params = {"deflect": deflect, "spread": spread, "mouth": mouth,
                       "edge": edge, "cell_cap": float(cell_cap),
                       "emit_per_tick_per_spout": int(emit_per_tick_per_spout),
                       "absorb": absorb}

    def _moves(self, y, x, idx):
        p = self.params
        cand = [(y + 1, x)]
        if p["deflect"] == "dl":
            cand += [(y + 1, x - 1)]
        elif p["deflect"] == "dr":
            cand += [(y + 1, x + 1)]
        elif p["deflect"] == "both":
            first = -1 if idx % 2 == 0 else 1
            cand += [(y + 1, x + first), (y + 1, x - first)]
        if p["spread"] == "lateral":
            first = -1 if idx % 2 == 0 else 1
            cand += [(y, x + first), (y, x - first)]
        return cand

    def step(self, state: SimState) -> None:
        km, p = state.kmap, self.params
        cap = p["cell_cap"]
        owner = km.mouth_owner()
        # move-then-emit: newly emitted units occupy the spout cell until the
        # next tick (matches observed animation convention; DW_LAB_LOG).
        nxt = []
        for idx, (y, x) in enumerate(state.moving):
            placed = False
            for (ny, nx) in self._moves(y, x, idx):
                if ny >= M:
                    state.disposed += 1; placed = True; break
                if not (0 <= nx < M):
                    if p["edge"] == "dispose":
                        state.disposed += 1; placed = True; break
                    continue                       # halt: this move blocked, try next
                if ny in km.floor_rows:
                    state.floor_touch += 1; placed = True; break
                if km.solid[ny, nx] or state.settled[ny, nx] >= cap:
                    continue
                if p["mouth"] == "entry" and (ny, nx) in owner:
                    state.fills[owner[(ny, nx)]] += 1
                    if p["absorb"] == "cap":
                        state.settled[ny, nx] += 1.0   # unlimited: swallowed, no pile-up
                    placed = True; break
                nxt.append((ny, nx)); placed = True; break
            if not placed:                          # nowhere to go: settle in place
                if (y, x) in owner:
                    state.fills[owner[(y, x)]] += 1
                    if p["absorb"] == "cap":
                        state.settled[y, x] += 1.0
                else:
                    state.settled[y, x] += 1.0
        state.moving = nxt
        for _ in range(p["emit_per_tick_per_spout"]):
            for (sy, sx) in km.spouts:
                if state.emitted >= km.emit_volume:
                    break
                if not km.solid[sy, sx]:
                    state.moving.append((sy, sx))
                    state.emitted += 1
        if state.emitted >= km.emit_volume and not state.moving:
            state.terminal = "quiescent"


def run(kmap: KnownMap, hyp: DynamicsHypothesis, max_ticks: int = 200):
    """Deterministic rollout -> list of SimState (tick 0 .. terminal/max)."""
    states = [SimState(kmap=kmap)]
    while states[-1].tick < max_ticks and not states[-1].terminal:
        s = states[-1].clone()
        s.tick += 1
        hyp.step(s)
        states.append(s)
    return states


def win_moment(states, capacities, rule="fill_cap"):
    """First tick where the win predicate holds, + floor touches AT that tick.

    rule 'fill_cap': every target's fills >= its capacity (round-2 end-state
    predicate, evaluated as a moment).  rule 'entered': every mouth has been
    entered by at least one unit — the round-3 hypothesis induced from the
    flywheel observation that winning animations terminate early, right after
    the last mouth is first reached.

    Returns (tick or None, floor_touch_at_moment_or_final).  A pour whose
    predicate never fires reports the FINAL floor count (loss diagnosis is
    horizon-honest: no truncation can hide overflow).
    """
    for s in states:
        req = capacities if rule == "fill_cap" else [1.0] * len(capacities)
        if capacities and all(f >= r for f, r in zip(s.fills, req)):
            return s.tick, int(s.floor_touch)
    return None, int(states[-1].floor_touch)


def cast_rays(state: SimState, dirs=EIGHT_DIRS):
    """RayBundle from spouts + mouth-top anchors over solids ∪ settled material.

    Contacts carry object identity, network polarity, and signed potential:
    mouth=beneficial(+1), floor=adversarial(-1), other blockers=structural(-0.25),
    open exit=neutral(0). (Goals positive / hazards negative is the generic
    dream_kernel convention; mouth-as-goal and floor-as-hazard are OBSERVED
    facts — win recolor and lava flash — not engine knowledge.)
    """
    km = state.kmap
    blocked = km.solid | (state.settled >= 1.0)
    owner = km.mouth_owner()
    glyph_owner = {c: i for i, cells in enumerate(km.glyph_cells) for c in cells}
    anchors = [("spout", s) for s in km.spouts]
    anchors += [("mouth_top", min(cells, key=lambda c: c[0]))
                for cells in km.mouths if cells]
    hits = []
    for kind, (ay, ax) in anchors:
        for (dy, dx) in dirs:
            y, x, path = ay, ax, 0
            contact = None
            while True:
                y += dy; x += dx; path += 1
                if not (0 <= y < M and 0 <= x < M):
                    contact = ("open", None); break
                if (y, x) in owner:
                    contact = ("mouth", owner[(y, x)]); break
                if y in km.floor_rows:
                    contact = ("floor", None); break
                if blocked[y, x]:
                    contact = ("glyph", glyph_owner[(y, x)]) if (y, x) in glyph_owner \
                        else ("solid", None)
                    break
            kind_c, ident = contact
            network, signed_y = {
                "mouth": ("beneficial", 1.0), "floor": ("adversarial", -1.0),
                "glyph": ("structural", -0.25), "solid": ("structural", -0.25),
                "open": ("neutral", 0.0),
            }[kind_c]
            hits.append({"anchor": kind, "origin": (ay, ax), "dir": (dy, dx),
                         "path_len": path, "contact": kind_c,
                         "object_id": f"{kind_c}:{ident}:{y}:{x}",
                         "network": network, "signed_potential_y": signed_y})
    return hits


def render_state(state: SimState):
    """SimState -> [16,16] int8 macro grid in SP80 palette colors (for viz)."""
    km = state.kmap
    g = np.full((M, M), 12, dtype=np.int8)
    g[km.solid] = 8
    for r in km.floor_rows:
        g[r, :] = 1
    for i, cells in enumerate(km.glyph_cells):
        col = 13 if (km.capacities[i] and state.fills[i] >= km.capacities[i]) else 11
        for (y, x) in cells:
            g[y, x] = col
    g[state.settled >= 0.5] = 6
    for (y, x) in state.moving:
        g[y, x] = 6
    for (sy, sx) in km.spouts:
        if g[sy, sx] == 12:
            g[sy, sx] = 4
    return g
