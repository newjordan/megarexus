# DREAMWEAVER INTEGRATION PLAN — trust-gated 3D perception + induced-dynamics imagination
**Date:** 2026-07-06 · **Branch:** night-shape-lava-imagination-20260705 · **Status:** ACTIVE (Phase 0/1 in progress)
**Lab log (append-only, feeds the final report):** `DW_LAB_LOG_20260706.md`

## 0. Mission

Bring the user's Dreamweaver architecture (`$HOME/dreamweaver`) into megarexus sim3d:
the **V031 perception trust gate** (frame → grid → 3D geometry → raymap → strict re-render/ray
accuracy gate), the **Dream Kernel contract** (deterministic imagination over provenance-tagged
known maps), and **cellwave potential proposals** — replacing the retracted source-derived
dropletsim with a **legitimately-induced** world model.

Standing context: `INVALID_DROPLETSIM_ANSWERKEY_20260706.md` retracted the droplet-CA line
(answer key transcribed from game source). Honest boundary at plan start: **CNN-alone = 1 level**.

## 1. Legality contract (binding for every phase)

**The provenance test:** could this exact system run on a private ARC-AGI-3 game — no game
source on disk, no human demos in existence? If no, it is an answer key and cannot be a solver
or headline.

**Allowed inputs:** observed frames; outcomes of env interactions we execute; generic
game-agnostic machinery (grids, 3D extrusion, raycasts, wave/potential fields, falling-material
CA families, search). **Banned:** reading game engine source (mechanically enforced, Phase 0);
human answer routes; `dropletsim.py` / `native_scene.py` / `sp80_spill_model.py` lineage in any
training, scoring, solving, or headline path (they remain quarantined; per the user's ruling they
may exist only as a private dev-time cross-check and Phase 1–4 do not touch them at all).

**Contamination disclosure (recorded up front, honesty over optics):** this lab has previously
seen source-derived facts (e.g. block-fork, edge-disposal, enclosed-entry fill). We cannot unknow
them. Legitimacy of the induced kernel therefore rests on *procedure*, pre-registered here: the
dynamics hypothesis family is declared generic (falling-material primitives), and selection +
calibration use ONLY tick-matching against observed env pour frames with held-out splits and a
pre-registered negative branch. Goodness-of-fit per hypothesis is reported for the full family,
not just the expected winner.

**Evidence discipline:** `honesty_audit.py` runs before any wrap/report and its summary leads the
report. Every phase writes artifacts under `sim3d/fixtures/` + an entry in the lab log with
timestamp, command, evidence path, and outcome (including negatives).

## 2. Pre-registered hypotheses

- **H1 (perception trust):** A V031-style round-trip gate — scene re-rendered and diffed against
  the observed frame at pixel level, plus structural ray checks — run on L0–L5 pre-pour frames
  will mechanically detect extractor↔frame mismatches with zero game-source knowledge.
  *Prediction:* it flags the two known historical bug classes if re-tested on the old extractor
  behavior (L3: found 2 of 4 targets; L2: 3 sources collapsed to 1 spout), and produces at least
  one new UNTRUSTED verdict or residual on current L0–L5 extraction (the L2-WIN
  native-vs-frame disagreement in the notes suggests residual extractor error exists).
- **H2 (induced dynamics):** A deterministic imagination kernel whose SP80 transition rule is
  *selected from a generic falling-material hypothesis family purely by tick-matching observed
  env pours* reaches ≥0.90 outcome agreement (fill-set/flood/win) on a held-out observed-pour
  split, with first-divergence tick ≥10 on tick-match fixtures. *Negative branch:* if no member
  of the family fits, ESCALATE to user with the fit table — do not widen the family by peeking
  at source.
- **H3 (honest solve):** propose (cellwave/portband) → imagine (induced kernel) → env decides
  (top-K real pours, first-exposure budget) solves ≥1 level beyond the honest boundary with
  honesty_audit-clean lineage. *This is the headline claim candidate; nothing below it is.*
- **H4 (game-agnostic machinery, stretch):** the identical perception+kernel machinery (no
  SP80-specific constants beyond a palette manifest) runs end-to-end on `sp80_variants` boards
  and the dynamics-fit harness re-runs from scratch on them — demonstrating the machinery, not
  the knowledge, is what ships.

## 3. Phases

### Phase 0 — Mechanical guard (do first; antidote to 3 prior incidents)
In-context rules failed 3×; make the hack impossible at the tool layer.
- PreToolUse hook blocking Read/Grep/Bash access to game engine source
  (`environment_files/**/*.py`), echoing the honesty ruling on block. Config via
  `.claude/settings.json` (update-config skill).
- Acceptance: a deliberate `Read environment_files/sp80/589a99af/sp80.py` attempt is refused;
  normal work (metadata.json, running the env via harness) unaffected.

### Phase 1 — V031 perception trust gate port (`sim3d/perception_gate.py`)
Port the *concepts* of `dreamweaver/src/chronometric_map_perception.py` (round-trip label/
geometry/ray accuracy with strict thresholds → `trusted` bool + `gate_failures`), adapted to
SP80's already-integer frames (no palette step needed; truth = the observed frame itself).

Checks (each strict, V031-style; all computed from frame + Scene only):
1. **target_components:** 4-connected color-11 pixel components in frame == `len(scene.targets)`
   (catches the L3 class).
2. **glyph_pixel_exact:** union of `target.cells_px` == frame's color-11 mask exactly, and
   per-target IoU 1.0 (catches partial/mis-sized glyphs from macro-vote at wrong native scale).
3. **source_spout_match:** color-4 pixel components == `len(scene.spouts)`; each spout open
   (non-solid) and column-aligned under its blob (catches the L2 multi-source collapse).
4. **mouth_validity:** every mouth cell open in the frame-derived macro solids; enclosed
   left/right/below; a ray cast down from above the mouth terminates inside it.
5. **ray_contacts:** raymap (8 directions from spouts + mouth-top anchors) computed on
   frame-derived macro solids vs on scene-object-derived solids (glyphs + floor + source blobs
   + solid remainder) — exact contact agreement (port of `evaluate_ray_accuracy`).
6. **floor_strip:** scene.floor_rows == HUD-band rows below targets recomputed independently
   from pixels; solid coverage: every SOLID_COLORS pixel lies in a solid macro cell.

Interface: `gate_scene(frame, scene) -> {trusted, gate_failures, metrics...}` +
`scripts` harness `sim3d/perception_gate_eval.py` writing
`fixtures/perception_gate_L{0..5}_20260706.json` + a summary. Exit 0 trusted-all / 2 otherwise
(V031 convention).

### Phase 2 — Imagination kernel substrate (`sim3d/imagination_kernel.py`)
Python port of the Dream Kernel *contract* (Rust stays reference):
- `KnownMap` built ONLY from gate-trusted scenes; carries provenance
  (`{"source": "perception_gate", "trusted": true, "gate_fixture": ...}`) — refuses untrusted.
- Deterministic `step`, pluggable `DynamicsHypothesis`, `RayBundle` casting, and a renderable
  state sequence (for imagination-vs-real MP4s).
- Acceptance: kernel replays a trusted L0 map, steps a null/movement hypothesis
  deterministically, emits render frames; unit tests.

### Phase 3 — Induced dynamics (`sim3d/dynamics_fit.py`) — the science
- Hypothesis family (generic falling-material primitives, parameterized): straight fall;
  deflection at blocked cell (none/left/right/both); lateral spreading conditions (never /
  at-rest / over-capacity); boundary behavior at open edges (halt / vanish); emission
  (rate, stop conditions); absorption at enclosed cavities (on-entry / on-fill / never);
  failure sensor (floor-touch / any-wall-touch / none). Full-family fit table reported.
- Data: observed env pour frame sequences ONLY — existing tick-match fixtures + fresh
  harvests via the offline harness; episode-level train/held-out split.
- Gates (pre-registered): held-out outcome agreement ≥0.90; first-div ≥10; report per-hypothesis
  fit. Negative branch: escalate with the table.

### Phase 4 — Propose → imagine → env decides
- Cellwave-style potential fields (port concepts of `cellwave_potential_field.py`) +
  portband-seeded proposals; kernel ranks by imagined outcome; env grades top-K within
  first-exposure budget. Solve claims = env-verified only, honesty_audit-clean.
- Renders: side-by-side imagination-vs-real MP4 per attempt.

### Phase 5 — Report
Build the report FROM the lab log: hypotheses → procedure → evidence paths → outcomes
(negatives included) → honest headline. honesty_audit summary leads.

## 4. Decision tree
- Gate flags current extraction (H1 true) → fix extractor legally (pixels only), re-gate; sim
  work proceeds only on trusted maps.
- H2 fails → escalate with fit table; consider AdaJEPA lane (see roadmap memory); do NOT touch
  quarantined sims for rules.
- H3 fails in budget → report honestly; the trusted-perception + induced-kernel machinery is
  still the deliverable (it is the ARC-intended shape).

## 5. Provenance of this plan
Dreamweaver surfaces referenced: `src/chronometric_map_perception.py` (V031),
`docs/chronometric_map_perception_v031.md`, `dream_kernel/src/lib.rs` +
`docs/dream_kernel_v001.md`, `src/cellwave_potential_field.py`,
`src/chronometric_grid_imagination.py`, `goals.md` hard boundaries. Reading the user's own
Dreamweaver code is legal (it is not game source); its boundaries were adopted, not weakened.
