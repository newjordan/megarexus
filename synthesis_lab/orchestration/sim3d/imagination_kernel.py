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

EIGHT_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


class UntrustedSceneError(RuntimeError):
    """A Scene without a passing perception-gate verdict may not enter the kernel."""


@dataclass(frozen=True)
class KnownMap:
    """Explicit spatial cells + provenance (canonical lattice coords, flow down)."""
    solid: np.ndarray                      # [gh,gw] bool, static colliders incl. floor
    floor_rows: tuple                      # lava strip rows (adversarial)
    spouts: tuple                          # ((y,x), ...) CHARGED material entry cells
    mouths: tuple                          # per target: ((y,x), ...) goal cells
    glyph_cells: tuple                     # per target: ((y,x), ...) structural cells
    capacities: tuple                      # per target: mouth capacity (cells)
    emit_volume: float                     # material units one pour emits
    mouth_openings: tuple = ()             # per target: up/down/left/right/None
    all_spouts: tuple = ()                 # every housing spout, charged or not
    spout_charged: tuple = ()              # per all_spouts entry: has a reservoir?
    hud: np.ndarray = None                 # [gh,gw] bool boundary/HUD cells (non-solid
                                           # walls for film flow; FallingMaterial ignores)
    blocker: np.ndarray = None             # [gh,gw] bool movable-blocker (color 15) cells;
                                           # semantics = FlowTrail.blocker_contact knob
                                           # (FallingMaterial treats them as solid)
    provenance: dict = field(default_factory=dict, hash=False, compare=False)

    @classmethod
    def from_scene(cls, scene, gate_verdict, **meta):
        if not gate_verdict.get("trusted"):
            raise UntrustedSceneError(
                "scene rejected: perception gate failures "
                f"{gate_verdict.get('gate_failures')} — fix the extractor and re-gate; "
                "the kernel refuses unproven maps (Dream Kernel provenance rule)")
        # Emit only from CHARGED spouts (housing with an observed reservoir).
        # Uncharged housings (L3/L5 mid-board catch/redirect nodes) are geometry,
        # not emitters — whether they re-emit is left to dynamics induction.
        charged = list(scene.spout_charged) if getattr(scene, "spout_charged", None) else []
        if len(charged) == len(scene.spouts) and any(charged):
            emitters = [s for s, c in zip(scene.spouts, charged) if c]
        else:
            emitters = list(scene.spouts)
            charged = [True] * len(scene.spouts)
        from scene import PALETTE
        hud = (np.asarray(scene.g) == PALETTE["boundary"]) & ~scene.solid
        blk = (scene.blocker.copy() if getattr(scene, "blocker", None) is not None
               else np.zeros_like(scene.solid))
        return cls(
            solid=scene.solid.copy(),
            floor_rows=tuple(scene.floor_rows),
            spouts=tuple((int(y), int(x)) for y, x in emitters),
            mouths=tuple(tuple((int(y), int(x)) for y, x in t.mouth) for t in scene.targets),
            glyph_cells=tuple(tuple((int(y), int(x)) for y, x in zip(*np.nonzero(t.glyph)))
                              for t in scene.targets),
            capacities=tuple(float(t.capacity) for t in scene.targets),
            mouth_openings=tuple(getattr(t, "opening", None) for t in scene.targets),
            emit_volume=float(scene.emit_volume),
            all_spouts=tuple((int(y), int(x)) for y, x in scene.spouts),
            spout_charged=tuple(bool(c) for c in charged),
            hud=hud,
            blocker=blk,
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
    settled: np.ndarray = None             # [gh,gw] float, at-rest material
    moving: list = field(default_factory=list)   # [(y,x)] in-flight units
    emitted: int = 0
    fills: list = field(default_factory=list)    # per-target units in mouth
    floor_touch: int = 0                   # adversarial contacts (lava events)
    disposed: int = 0                      # units that left the map
    terminal: str = ""                     # "", "quiescent"

    def __post_init__(self):
        if self.settled is None:
            self.settled = np.zeros(self.kmap.solid.shape, dtype=np.float32)
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
        gh = km.solid.shape[0]
        owner = km.mouth_owner()
        # move first (stable order = deterministic); newly emitted units enter at
        # the spout and sit there until next tick — matches the observed animation
        # convention where the stream column starts AT the spout cell (obs[t] is
        # the board after t frames; alignment verified empirically, DW_LAB_LOG).
        nxt = []
        for (y, x) in state.moving:
            ny = y + 1
            if ny >= gh:
                state.disposed += 1
                continue
            if ny in km.floor_rows:
                state.floor_touch += 1
                continue
            blocked = km.solid[ny, x] or (km.blocker is not None and km.blocker[ny, x])
            if blocked or state.settled[ny, x] >= 1.0:
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
        gh, gw = km.solid.shape
        cap = p["cell_cap"]
        owner = km.mouth_owner()
        # blocker cells (color 15) are plain solids for this family — the
        # contact-knob alternatives are film-flow semantics (FlowTrail only)
        solid = km.solid if km.blocker is None else (km.solid | km.blocker)
        # move-then-emit: newly emitted units occupy the spout cell until the
        # next tick (matches observed animation convention; DW_LAB_LOG).
        nxt = []
        for idx, (y, x) in enumerate(state.moving):
            placed = False
            for (ny, nx) in self._moves(y, x, idx):
                if ny >= gh:
                    state.disposed += 1; placed = True; break
                if not (0 <= nx < gw):
                    if p["edge"] == "dispose":
                        state.disposed += 1; placed = True; break
                    continue                       # halt: this move blocked, try next
                if ny in km.floor_rows:
                    state.floor_touch += 1; placed = True; break
                if solid[ny, nx] or state.settled[ny, nx] >= cap:
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
                if not solid[sy, sx]:
                    state.moving.append((sy, sx))
                    state.emitted += 1
        if state.emitted >= km.emit_volume and not state.moving:
            state.terminal = "quiescent"


class FlowTrail(DynamicsHypothesis):
    """Round-4 GENERIC liquid-film / trail-flow family (induced from the L3
    corpus observations, DW_LAB_LOG r4 Phase A — falling-sand piling cannot
    express the observed spread-and-split behavior).

    Fixed rules (all directly observed on the L3 corpus):
      - each emitter starts ONE stream; a stream tip advances 1 cell/tick and
        leaves a PERSISTENT painted trail (visible paint never drains)
      - a tip blocked below becomes a lateral run BOTH ways along its row,
        1 cell/side/tick, descending at the first open-below cell on each side
        (streams SPLIT at both overhang edges)
      - painted cells block flow like solids (merging streams die, no double
        paint); HUD/boundary cells block laterally and are never painted
      - mouth cells absorb on ENTRY: the tip becomes a FEED, +1 fill/tick up
        to the mouth capacity (fill cells painted bottom-up)
      - floor rows are adversarial: attempted entry = floor_touch, tip dies

    Knobs (selected ONLY by observation, r4_fit.py):
      emitters 'charged' | 'all'          do uncharged housings emit?
      phase    'sync' | 'uncharged_lead'  charged streams spend 1 tick draining
                                          through the housing (uncharged lead —
                                          the observed frame-0 mid stream);
                                          sync = all streams paint at tick 1
      mouth    'feed_die' | 'feed_overflow'  full mouth: stream ends quietly vs
                                          remaining volume spills (floor_touch)
      volume   None | float               unlimited (flow to saturation) vs
                                          total trail budget measured per level
      merge    'split' | 'die'            a falling tip landing on an EXISTING
                                          painted column splits like on solid vs
                                          merges into the stream (dies quietly)
      blocker_contact                     collision semantics of movable-blocker
                                          (color 15) cells — the r5 contact-knob
                                          family; observation selects (r5 fit):
               'block'   solid-like in every direction (falling tip splits on
                         top of it; lateral runs stop at its face)
               'turn_l'  falling contact deflects into a single LEFT run;
                         lateral contact stops (solid-like sideways)
               'turn_r'  mirror of turn_l
               'split'   falling contact splits (solid-like from above) but
                         lateral runs TUNNEL through blocker cells sideways
               'pass'    fully permeable (blocker invisible to flow)
               Levels without blocker cells are invariant to this knob.
      wrap     'immediate' | 'run_to_end'
                                          lateral film at an obstacle edge
                                          either descends immediately through
                                          the first open-below cell (r4 rule) or
                                          continues laterally until a row event;
                                          r5 L4 pours 9/12 vs 25 discriminate.
      side_mouth
               'lateral_only' | 'adjacent_feed'
                                          side-facing mouths are fed only by
                                          lateral tips (old r4/r5 behavior) or
                                          by a falling tip adjacent to the
                                          mouth's open side. The latter captures
                                          L4 side-mouth fills seen in real
                                          env replay without affecting up-mouths.

    GRAVITY is not a knob here by design: scene extraction canonicalizes every
    board so flow is DOWN (extract_scene flips when sources sit below targets
    — inference from source/target geometry, verified on L1-rot and L4-rise).

    Tick-alignment convention (declared, uniform across corpora): FlowTrail
    rollouts compare obs[t] <-> state[t+1] — frame 0 is captured after the
    first flow tick (uncharged streams already 1 cell long there).
    """
    name = "flow_trail"

    def __init__(self, emitters="all", phase="uncharged_lead",
                 mouth="feed_die", volume=None, merge="split",
                 blocker_contact="block", wrap="immediate",
                 side_mouth="lateral_only"):
        self.params = {"emitters": emitters, "phase": phase, "mouth": mouth,
                       "volume": None if volume is None else float(volume),
                       "merge": merge, "blocker_contact": blocker_contact,
                       "wrap": wrap, "side_mouth": side_mouth}

    def _paint(self, state, y, x):
        state.settled[y, x] = 1.0
        state.emitted += 1

    def _budget_left(self, state):
        v = self.params["volume"]
        return True if v is None else state.emitted < v

    def step(self, state: SimState) -> None:
        km, p = state.kmap, self.params
        gh, gw = km.solid.shape
        hud = km.hud if km.hud is not None else np.zeros_like(km.solid)
        blk = km.blocker if km.blocker is not None else np.zeros_like(km.solid)
        bc = p["blocker_contact"]
        owner = km.mouth_owner()
        openings = tuple(km.mouth_openings or ())

        def open_cell(y, x):
            return (0 <= y < gh and 0 <= x < gw and not km.solid[y, x]
                    and not hud[y, x] and not (bc != "pass" and blk[y, x])
                    and state.settled[y, x] < 1.0)

        if not self._budget_left(state):
            state.moving = []

        def enter_mouth(y, x, nxt):
            t = owner[(y, x)]
            state.fills[t] += 1
            state.emitted += 1
            if state.settled[y, x] < 1.0:
                state.settled[y, x] = 1.0
            nxt.append((y, x, "f"))

        def adjacent_side_mouth(y, x):
            if p.get("side_mouth", "lateral_only") != "adjacent_feed":
                return None
            for dy, dx, need_open in ((0, -1, "right"), (0, 1, "left")):
                c = (y + dy, x + dx)
                t = owner.get(c)
                if t is None:
                    continue
                opening = openings[t] if t < len(openings) else None
                if opening == need_open:
                    return c
            return None

        def lateral_step(y, x, mode, nxt):
            """One lateral move for a run tip at (y,x); appends the survivor."""
            d = -1 if mode == "l" else 1
            nx = x + d
            if bc == "split":                                 # tunnel sideways
                while 0 <= nx < gw and blk[y, nx]:
                    nx += d
            if not (0 <= nx < gw):
                state.disposed += 1
                return
            if (y, nx) in owner:                              # side-facing mouth
                enter_mouth(y, nx, nxt)
                return
            if not open_cell(y, nx):                          # wall/merge: die
                return
            self._paint(state, y, nx)
            below = (y + 1, nx)
            wraps_now = (below[0] >= gh or below[0] in km.floor_rows
                         or below in owner or open_cell(*below))
            if wraps_now and p["wrap"] == "immediate":
                nxt.append((y, nx, "d"))                      # descend here
            else:
                nxt.append((y, nx, mode))

        nxt = []
        for tip in state.moving:
            if not self._budget_left(state):
                break
            y, x, mode = tip
            if mode == "f":                                   # feeding a mouth
                t = owner.get((y, x))
                if t is None or state.fills[t] >= km.capacities[t]:
                    if p["mouth"] == "feed_overflow" and self._budget_left(state):
                        state.floor_touch += 1                # excess spills
                    continue
                state.fills[t] += 1
                state.emitted += 1
                for (my, mx) in sorted(km.mouths[t], key=lambda c: (-c[0], c[1])):
                    if state.settled[my, mx] < 1.0:
                        state.settled[my, mx] = 1.0
                        break
                nxt.append((y, x, "f"))
                continue
            if mode == "d":                                   # falling
                side = adjacent_side_mouth(y, x)
                if side is not None:
                    enter_mouth(side[0], side[1], nxt)
                    continue
                ny = y + 1
                if ny >= gh:
                    state.disposed += 1
                    continue
                if ny in km.floor_rows:
                    state.floor_touch += 1
                    continue
                if (ny, x) in owner:                          # mouth entry
                    enter_mouth(ny, x, nxt)
                    continue
                if open_cell(ny, x):
                    self._paint(state, ny, x)
                    nxt.append((ny, x, "d"))
                    continue
                if (p["merge"] == "die" and not km.solid[ny, x]
                        and not (bc != "pass" and blk[ny, x])
                        and not hud[ny, x] and state.settled[ny, x] >= 1.0):
                    continue                                  # merge into the stream
                if blk[ny, x] and bc == "turn_l":             # blocker deflects
                    lateral_step(y, x, "l", nxt)
                    continue
                if blk[ny, x] and bc == "turn_r":
                    lateral_step(y, x, "r", nxt)
                    continue
                lateral_step(y, x, "l", nxt)                  # split: run both ways
                lateral_step(y, x, "r", nxt)                  # (same-tick, observed)
                continue
            lateral_step(y, x, mode, nxt)                     # lateral run
        state.moving = nxt

        # move-then-spawn (same convention as FallingMaterial): a stream's
        # first cell appears at its spawn tick and advances the NEXT tick.
        # Streams start at tick 1; charged emitters idle 1 tick when
        # phase='uncharged_lead' (drain-through), painting at tick 2.
        if state.tick in (1, 2):
            spouts = (list(zip(km.all_spouts, km.spout_charged))
                      if p["emitters"] == "all" and km.all_spouts
                      else [(s, True) for s in km.spouts])
            for (sy, sx), ch in spouts:
                lead = p["phase"] == "uncharged_lead"
                due = (2 if (ch and lead) else 1)
                if state.tick != due or not self._budget_left(state):
                    continue
                if open_cell(sy, sx):
                    self._paint(state, sy, sx)
                    state.moving.append((sy, sx, "d"))

        started = state.tick >= (2 if p["phase"] == "uncharged_lead" else 1)
        if started and not state.moving:
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
    if km.blocker is not None:
        blocked = blocked | km.blocker
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
                if not (0 <= y < km.solid.shape[0] and 0 <= x < km.solid.shape[1]):
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
    """SimState -> [gh,gw] int8 lattice grid in SP80 palette colors (for viz)."""
    km = state.kmap
    g = np.full(km.solid.shape, 12, dtype=np.int8)
    g[km.solid] = 8
    if km.blocker is not None:
        g[km.blocker] = 15
    for r in km.floor_rows:
        g[r, :] = 1
    for i, cells in enumerate(km.glyph_cells):
        col = 13 if (km.capacities[i] and state.fills[i] >= km.capacities[i]) else 11
        for (y, x) in cells:
            g[y, x] = col
    g[state.settled >= 0.5] = 6
    for tip in state.moving:
        g[tip[0], tip[1]] = 6
    for (sy, sx) in km.spouts:
        if g[sy, sx] == 12:
            g[sy, sx] = 4
    return g
