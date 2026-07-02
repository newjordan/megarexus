# megarexus_cnn — SCOUT_EXPERIMENT results (2026-07-01)

All numbers are local, seeded, offline probes. No live scorecard was opened;
nothing here is a benchmark claim. Human corpus levels were completed by the
human player.

CORRECTION (2026-07-01, level probe): an earlier draft claimed random/flow
rollouts reached `levels_completed=0`. That was true of v0/v1 uniform-random
data but NOT of later shards — environment fields show flow-biased random
completed level 1 in 2/60 episodes and the catch-play curriculum in 10/50
(single-glove) and 6/40 (multi-glove). Catching paint is causally tied to
SP80's level objective.

## v0 — single-frame, single-step regime (`runs/scout_v0`)

Data: 1,177 human corpus transitions + 1,956 uniform-Goose rollout steps
(last-animation-frame only). Val = sp80 episode_005 + snake final level
(245 human transitions, episode/level-split).

| metric (val) | model | baseline |
|---|---|---|
| next-frame cell acc | 0.926 | 0.989 persistence |
| next-frame changed-cell acc | **0.636** | 0.000 persistence |
| inverse-dynamics action acc | 0.776 | 0.518 majority |
| BC action-type acc | 0.665 | 0.518 majority |
| click localization (inv) | 24.3 px | — |

Verdict: learns *something* about action effects but is regime-limited for
the catch-and-redirect mechanic: no velocity input, no animation frames,
random data almost never pours.

## v1 — flow regime (`runs/flow_v1`)

Regime changes (why v0 could never learn catch-and-redirect):
1. **ACTION5 returns ~20–22 animation frames per pour** — v0's collector
   kept only the last. v1 expands every animation frame into "tick"
   transitions (action_id 0 = time passes). The paint physics lives here.
2. **2-frame input stack** — single frames cannot represent paint velocity;
   catch/deflection is unlearnable without it.
3. **Flow-biased Goose collection** (p_flow=0.3, p_click=0.2) so pours are
   dense in the data (8,006 rollout steps, most of them flow ticks).

Same val split as v0:

| metric (val) | v0 | v1 |
|---|---|---|
| changed-cell acc | 0.636 | 0.638 |
| inverse-dynamics action acc | 0.776 | **0.918** |
| BC action-type acc | 0.665 | **0.967** |
| click localization | 24.3 px | **15.8 px** |

## Ghost overlay confirmation (`eval_ghost.py`)

The trained CNN plays the ghost's predictive half: imagine H steps ahead
autoregressively, overlay against reality, count confirmations. Truth
boundary preserved (no goal/solve/control admission).

**Fresh rollout physics** (unseen seeds, `ghost_rollout/`): the paint
mechanic hypothesis is CONFIRMED on clean flow —

| horizon | changed-cell acc | change IoU | cell acc | persistence |
|---|---|---|---|---|
| 1 | 1.000 | 0.976 | 0.9949 | 0.9367 |
| 3 | 0.790 | 0.756 | 0.9560 | 0.9092 |
| 5 | 0.778 | 0.752 | 0.9731 | 0.9455 |

Model beats persistence at every horizon; 5-step imagined pours keep
~0.75 IoU with reality. Overlays: `runs/flow_v1/ghost_rollout/overlay_*.png`.

**Human corpus flow segments** (`ghost/`): step-1 changed-cell acc 0.564,
IoU 0.272, degrading by h=5; cell acc slightly *below* persistence.
Honest read: human pours involve concurrent bar moves and deliberate
catches that flow-biased random data still under-covers. The physics is
learned; the *human-interaction distribution* is not yet.

## Lock/key asymmetry goal seeker probe

`antisymmetric_relationship_hypotheses` fired on **0/40 sp80 and 0/40
snake** held-out frames. Root cause (probed): the upstream
`component_asymmetry` stage produced 0 confident facing-direction
hypotheses out of 501 residual components on these frames — the pairing
logic never receives candidates. So "works across all games to an extent"
does not currently extend to sp80/s5i5 scenes with default thresholds
(confidence ≥ 0.20, score ≥ 0.45); its cross-game claim should be
re-measured per game before it is wired in as the goal provider.

## v2/v3 — catch-play regime (`runs/catch_v2`, `runs/catch_v3`)

The game of catch (user's framing): paint = ball, Portband moveable object
= glove. `orchestration/collect_catch_play.py` probes for gloves by
controlled interaction (click + move + observed displacement), sweeps them
side-to-side under real pours, and labels catches in hindsight from pixels
against Portband's real port edge cells (`_object_ports`). Train/val seeds
disjoint. Curricula: 183 single-glove pours (0.62 caught) + 111
multi-object pours with per-glove roster labels (0.50) train; 84 unseen-
seed pours val. Edge-adjoin trials (70, 0.26 adjoined — PortBand edge
locks made behavioral, `collect_edge_adjoin.py`) feed the world model.

v2 lesson: a global-mean-pool catch head cannot represent glove-vs-pour
geometry — it collapsed to always-predict-catch (0.393 = val label rate).
v3 fixed it spatially: glove-masked pooling + global pooling after conv.

v3 (19,307 train transitions, 2.95M params), val on unseen seeds:

| metric (val) | v3 | baseline |
|---|---|---|
| **catch prediction** (106 pours) | **1.000** | 0.528 base rate |
| next-frame cell acc | 0.991 | 0.971 persistence |
| changed-cell acc / flow | 0.987 / 0.987 | 0.000 persistence |
| inverse-dynamics action acc | 0.990 | 0.656 majority |
| BC action-type acc | 0.996 | 0.656 majority |
| ghost h=1 change IoU (fresh pours) | 0.960 | — |
| ghost h=5 cell acc | 0.972 | 0.937 persistence |

Caveat on the 1.000: given the pour-time frame (source position visible)
and the glove plane, the catch outcome is near-deterministic geometry —
perfect accuracy is achievable and means the CNN extracted the rule, not
that the task is unbounded. The honest claim: **the model predicts, before
ACTION5 fires, whether a given glove placement catches the pour, on
episodes it has never seen.** SP80 L1 has 2 controllable gloves; roster
labels cover credit assignment (which glove the stream reaches).

## Behavioral catch probe (`orchestration/behavioral_catch_probe.py`)

The closed loop: the LeWM-role dynamics head ghost-imagines the pour
(ACTION5 + ticks, pure imagination, no env access), the target column is
read off the imagined change mass, and the Portband glove is moved under
it with real ACTION3/4 moves. Strict catch definition: paint on the
glove's TOP port edge within the first 8 animation frames (liquid spread
touching a side later does not count).

Paired on identical unseen seeds (20 episodes, 80 pours per policy):

| policy | caught | rate |
|---|---|---|
| **model-guided (imagine → intercept)** | **77/80** | **0.96** |
| random placement (curriculum rule) | 30/80 | 0.38 |

Attribution per lab rules: the world model (LEWM_REX_ACTION_OWNER) owned
the placement decision; Portband defined the glove and its edges; the
catch head logged per-pour predictions but did not steer (its saturated
logits make plateaus — hill-climbing on it underperformed random, kept in
the report as a negative result). Imagined pour column matched the real
one to within ±2 cells in spot checks.

Probe hygiene findings (fixed during the probe, affect future collection):
- The interaction probe originally accepted a full-width HUD strip as the
  "glove" (gauge drain mimics displacement). Fixed: full-width/height
  bands excluded (same heuristic as lewm sprite filtering) and
  displacement must be direction-consistent with the issued move.
- Earlier curriculum shards may contain strip-gloves; their labels are
  still self-consistent ("would the pour touch THIS bbox") but a
  re-collection with the fixed probe is queued for v4.

## Level probe: how the system does against SP80 (`orchestration/level_probe.py`)

The question "how would the system do against sp80?" answered with the only
honest instrument: environment fields (`levels_completed`, `state`) from
local offline episodes. SCOUT_EXPERIMENT; no live scorecard.

Paired full episodes on identical unseen seeds (20 each), MAX_POURS=8,
glove re-found after each level transition:

| policy | episodes completing level 1 | total levels | strict catches / pours |
|---|---|---|---|
| **model-guided (imagine → intercept)** | **10/20** | 10 | 87/104 (0.84) |
| random placement | 5/20 | 5 | 38/95 (0.40) |

Human reference (corpus, evidence fields only): 6 episodes, every one
reached level 4+; one reached level 6 and ended WIN. So the system today is
a **level-1-in-half-of-episodes agent; the human plays the whole game.**

Measured failure modes, in order of impact:

1. **Level-1 move budget.** Episodes that fail level 1 die at GAME_OVER
   after ~5 pours despite 4/5 strict catches — catching is strongly
   correlated with completion (model doubles random's level rate) but a
   high catch rate alone does not guarantee it within the budget.
   Placement moves are spent by both the probe (3 actions/candidate) and
   the intercept walk.
2. **Level-2 probe miss.** 3/10 model episodes that beat level 1 stalled
   on level 2 with no glove found (NOT_FINISHED, not dead). Replay
   diagnostic: level 2 HAS a movable glove — the color-9 bar at
   (20,36)-(39,39) moves ±4 px under ACTION3/4 — but `find_gloves` probes
   only 6 shuffled candidates out of level 2's 8 components (~25% miss
   rate, matching the stall rate). Probe luck, not physics.
3. **Old-curriculum strip gloves confirmed.** The level-1 completions
   inside `catch_play_train.pt` used the pre-fix probe; a replayed episode
   shows its "glove" was the HUD strip `[0,0,40,0]`. Those completions
   were not glove-driven. Today's probes use the fixed detector, so the
   10/20-vs-5/20 result stands; v4 re-collection remains queued.

Report: `runs/catch_v3/level_probe.json`
(schema `synthesis_lab.level_probe.v0`).

### Gas budget: known implicitly, unused by the controller

The move budget ("gas") is the row-0 HUD gauge — the same strip behind the
fake-glove bug. Probed on unseen-seed val transitions (catch_v3 model):

- **Drain physics: learned.** On 261 move transitions with a gauge drain,
  the model predicted a drain every time and got 558/558 drained cells
  exactly right (persistence = 0.000 by definition).
- **Death transition: learned.** On the 8 val transitions where the state
  flips to GAME_OVER, the model predicted the death-screen change with
  208/208 changed cells correct (small sample, unseen seeds).

So the world model knows the gas budget end-to-end as pixel physics; what
does NOT exist is any explicit use of it — no moves-remaining scalar head,
and `model_place` walks up to 10 moves without consulting the gauge. This
is failure mode 1 of the level probe: the knowledge is in the imagination,
the controller just never reads it. Natural upgrade: budget-aware
placement — imagine the intercept walk first, read the predicted gauge
(or predicted death frame), and trade catch position against remaining
gas before spending real moves.

## Shape/map systems audit → training backlog (2026-07-01)

Full sweep of contra_head + architecture/src/lewm answered "how much of the
shape and map recognition systems is the world model trained on?" —
**almost none.** Training consumes exactly two low-level Contra functions
(`raw_pixel_components_with_labels`, `build_raw_color_component_packet` for
the background color) feeding the 6 generic input planes, plus Portband
`_object_ports` as the catch-label source. Everything semantic — floor/wall
scoring (the "centroid solver" = `score_floor`'s low_centroid term +
floor-first gate in `analyze_scene_surface_hypotheses`), checkerboard floor
patterns, contact graph, facings, compound objects, all 8 symmetry passes,
lock-key sensor, geoband relational map, portband alignments/edge-lock
ledger/lineup matrix, LeWM's wall/floor constancy masks and centroid player
tracker — never reaches training. The original LeWM pipeline also trained
on raw one-hot frames only; its map systems were MCTS planning-time.

Audit also found `contra_head/sp80_spill_model.py` — a hand-coded
SP80-specific mechanic model with the win condition written in. RULING
(user, 2026-07-01): **that is an answer key, not a teacher.** It is
excluded from training entirely — not as label source, not as informer,
not as agreement metric, not as ghost verifier. The boundary: general,
game-agnostic engines (Contra surface/symmetry/portband geometry, LeWM
map heuristics) are legitimate synthesis teachers; any hand-authored
model of a specific game's mechanics or goals is out of bounds. The
system must learn win semantics from its own experience: environment
fields (levels_completed, GAME_OVER) and pixels, hindsight only.

### Backlog: elements to add, one at a time, ablated like the catch head

Rules of engagement for every element: added as EITHER input plane, aux
prediction head (preferred — distillation), or hindsight label source;
teacher outputs are hypothesis-grade (truth flags preserved); each element
is kept only if unseen-seed val improves vs the previous checkpoint and
its own baseline (base rate / persistence / majority). One element per
training run so credit assignment stays clean.

Tier 1 — outcome semantics, learned from experience only (no mechanic
model anywhere in the loop):
- [ ] **Liquid-role head** (per-cell/per-component): roles derived
  behaviorally from pixels across many pours — "accumulator" = region
  where poured liquid persists across ticks; "boundary" = constant cells
  whose liquid contact precedes GAME_OVER; "redirector" = cells where
  imagined/observed flow changes direction. No role vocabulary imported
  from any game-specific code; names describe observed behavior.
- [x] **Level-up head** (outcome_v4): P(levels_completed increments |
  frame, pour) — hindsight from environment fields. This is how the
  system LEARNS the win condition instead of being told it. Val: 3/3 tp,
  0 fp; thin positives, grows with collection.
- [ ] **Death-by-liquid head**: P(GAME_OVER follows this pour) —
  hindsight env fields; pairs with the gas death head (below).
- [~] **Fill-progress / settle head** (outcome_v4: label ill-posed —
  liquid is transient, persistent diff ≈ gauge drain; median 2 cells).
  Redesign: multi-pour integration or bucket-fill delta.

Tier 2 — budget ("caring", from the gas audit):
- [ ] **Death head**: P(GAME_OVER within k actions | stack, action) —
  hindsight labels from rollouts; conditioned on candidate action.
- [ ] **Cost-to-intercept head**: moves needed to reach target column —
  hindsight count from probe episodes.

Tier 3 — geometry already queued, data in hand:
- [ ] **Edge-adjoin head**: `adjoined`/`gap_px` from (stack, glove plane,
  direction) — 70 trials collected (`edge_adjoin_train.pt`); PortBand
  `directional_gap`/`alignment_error_px` analogue.
- [x] **Controllability head** (outcome_v4): which component displaces
  under ACTION1-4 — labels from every move in the shards (14,663 train /
  1,848 val samples). Val 1.000 vs 0.528 base; ranks the level-2 movable
  bar +4.31 vs ≤−14 for fixed scenery with zero probe actions.

Tier 4 — Contra scene-surface distillation:
- [ ] **Floor plane/head**: predict `score_floor` winner mask (teacher =
  `analyze_scene_surface_hypotheses`; hypothesis-grade).
- [ ] **Wall plane/head**: `score_wall` winners; cross-check against the
  pixel-honest LeWM constancy rule (cells constant across episode).
- [ ] **Checkerboard floor membership**: per-cell pattern mask.
- [ ] **Immovability plane** (input): trajectory-constancy mask — fully
  pixel-derived, cheap, honest; LeWM `_wall_floor_masks` rule.
- [ ] **Facing head**: per-component facing direction (teacher =
  `component_asymmetry`) — this is the dead upstream of the lock/key
  seeker; a trained facing head with CNN change-filtered candidates is
  the planned recalibration path.
- [ ] **Contact-graph head**: predict component adjacency (teacher =
  `contact_graph`) — supports "adjoin" reasoning beyond the glove.

Tier 5 — relations/symmetry (after Tier 4 gives roles):
- [ ] **Symmetry/lock-key aux targets**: axis + polar-pair + lock/key
  assignments from `build_symmetry_hypotheses` (8 passes) as auxiliary
  classification targets; feeds the cross-game goal seeker.
- [ ] **Port alignment head**: predict `_alignment_row` status
  (exact/near/residual) for object pairs — portband edge-lock ledger and
  lineup matrix distillation; the "block adjoins wall/block" test at full
  Portband fidelity.

## outcome_v4 — first backlog run (`runs/outcome_v4`)

Backlog elements trained: level-up head + settle head (Tier 1),
controllability head (Tier 3). Labels: environment fields + hindsight
pixels only. Control rows are auxiliary duplicates gated out of the main
losses (move dynamics not over-weighted); level-completing pours are
excluded from the settle target (their "settled" frame is a new layout).
3.64M params, same data recipe as catch_v3, 40 epochs, 3735s.

Val (unseen seeds), v3 metrics all held:

| metric | v4 | v3 | baseline |
|---|---|---|---|
| catch prediction (106 pours) | 1.000 | 1.000 | 0.528 |
| next-frame cell / changed / flow | 0.991 / 0.986 / 0.986 | 0.991 / 0.987 / 0.987 | 0.971 persist |
| inverse dynamics | 0.997 | 0.990 | 0.656 majority |
| **controllability** (1,848 samples) | **1.000** | — | 0.528 base |
| **level-up** (84 pours) | **tp 3/3, fp 0** | — | 0.964 base (all-negative) |
| settle IoU | 0.214 | — | ill-posed label (below) |

- **Controllability head: kept.** Behavioral check: on the level-2 scene
  (reached by replaying a val episode) it ranks the movable color-9 bar
  +4.31 with every fixed component below −14 — one forward pass per
  candidate, zero probe actions. This replaces probe-luck glove finding
  (the 3/10 level-2 stalls). Caveat: SP80's level-2 layout is identical
  across seeds, so the ranking is a capability demo; generalization
  evidence is the 1,848 unseen-seed val samples.
- **Level-up head: kept, needs more positives.** All 3 val level-up
  pours detected, zero false alarms across 84. Only 18 train / 3 val
  positives — statistically thin; grows with every model-guided
  collection round.
- **Settle head: label sent back for redesign.** Diagnosis: true
  "persistent change" masks have median 2 cells — SP80 liquid is
  transient (drained/vanished by the settle frame), so the label mostly
  captures the gas-gauge drain, not accumulation. IoU 0.214 (0.274 at
  best threshold) reflects the label, not the head. Redesign candidates:
  multi-pour integration (cells repeatedly liquid-colored across pours)
  or bucket-fill delta once the accumulator region is discovered.

## Level probe v2: learned glove finder → level 1 solved 20/20

Same paired protocol and seeds as the first level probe, but the MODEL
policy finds its glove with the trained controllability head (rank all
components, click the winner — 1 action) instead of interaction probing
(up to 18 actions). Random baseline unchanged (reproduced exactly: 5/20).
Report: `runs/outcome_v4/level_probe_learned.json`.

| policy | level-1 completions | vs probe finder |
|---|---|---|
| **model + learned finder** | **20/20** | 10/20 |
| random + interaction probe | 5/20 | 5/20 (identical) |

Every episode: level 1 completed, level-2 bar found instantly (zero
stalls, previously 3), ~7 pours 6 catches, then GAME_OVER **on level 2**.
Two effects compounded: the head always selects the right glove, and the
~17 saved probe actions are gas that level 1 needed. The uniform level-2
endgame (identical counts across episodes) reflects SP80's deterministic
level-2 layout meeting a deterministic policy.

Attribution: controllability head (trained on hindsight displacement
labels) owned glove selection; world-model imagination owned placement;
environment fields owned the level evidence. Still SCOUT, still local.

Remaining blocker, now isolated: **level-2 gameplay** — catching alone
doesn't complete it before the gas runs out. Queued: budget-aware
placement (imagination dry-run before spending real moves) and the
redesigned fill/settle signal to learn what level 2 actually wants.

## Level-2 guidance from the user (2026-07-01) + level probe v3

The user supplied two pointers, given and recorded as *research
direction* (what to measure), not as training labels — the answer-key
ban applies to guidance too if it ever becomes a label source:

1. **"You can't miss" — critical.** Audit finding: we always *labeled*
   misses (hindsight catch labels) but never *acted* on them — the
   probe's `model_place` pours unconditionally even when the intercept
   walk was blocked or incomplete. That's the 6/7 catches per level-2
   episode: one unassured pour thrown anyway, wasting paint + gas.
2. **Level 2 is about overflow.** Some overflow must be blocked using
   the sidewalls: the glove "catches" at an orthogonal angle *toward*
   the paint flow, and that side-contact creates the block. In Portband
   vocabulary this is paint contacting the glove's **left/right port
   edges** — already expressible as `catch_label(sides=("left","right"))`
   with zero new machinery. Our catch training so far is top-edge only.

Level probe v3 (`level_probe.py` update, report
`runs/outcome_v4/level_probe_careful.json`) puts both into the
controller and the evidence stream:

- `careful_place`: imagine the pour column → imagine the intercept
  walk + pour and read the **predicted gauge drain** (learned drain
  physics, no env access) → decline if the plan would exhaust the
  row-0 gauge (present-frame read, controller informer) → walk for
  real → **pour only if the glove actually spans the imagined column**.
  Declined pours are recorded with reasons, never thrown.
- Every pour now also logs `side_block_contact` /
  `side_contact_cells` (full-animation left/right port contact) and
  gauge before/after vs predicted plan drain — the overflow-block
  evidence the fill-signal redesign needs.
- Random baseline untouched (still the original pour-always policy).

v3 results (`level_probe_careful.json`): model 20/20 level 1 (as v2),
random 5/20 reproduced. But zero declines fired and the level-2 endgame
was unchanged — which exposed that our level-2 story was wrong, see next
section.

## Level 2 decoded: it's a plumbing puzzle, and it is now completed

Chain of measurements (diagnostics in the session scratchpad; all local
offline environment experience + human-demonstration mining):

1. **Instrumentation was wrong twice.** The gauge drains color→color
   (14→0), so non-background counting reads 64 forever. And on level 2
   the gauge is **row 63** (row 0 becomes a color-1 band); it drains
   ~1.5 cells/action and is NOT what ends level-2 episodes (moves-only
   survives to 58/64 used).
2. **The level-2 "miss" never existed.** The fatal pour returns ONE
   frame — no paint falls. Level 2 grants **~4 animated pours; the 5th
   pour call is instant GAME_OVER** regardless of gauge or position.
   Pours are the scarce resource (the user's "you can't miss," stronger
   than expected).
3. **Glove position is irrelevant to the old policy's outcome.** Center
   / hard left / hard right / half offsets / all vertical parks (the
   bar also moves ±4 on actions 1/2): identical 4-pours-then-death.
4. **Level 2's flow is upward.** A bottom-center spout (x40-43) jets
   paint up; it splits on the bar's underside; risers climb both flanks
   and flood the top band. Three funnels with mouths (x16-19 / 32-35 /
   48-51) are the receivers.
5. **Human corpus mining** (6 episodes, all reach level 4+): every
   episode plays the second level as pour-FIRST (activate the tap),
   then live click-select + move of **three** objects — both 8-colored
   platforms AND the bar — steering the flow while it runs. The
   platforms are movable; our collectors never tried (component census:
   click+move displaces them ±4px like the bar).
6. **Verbatim human replay fails offline** — the offline env settles
   the whole flow inside the pour step, so steering-after-pour moves
   paint that is already gone.
7. **Arrange-then-pour completes level 2.** The human WIN episode's
   final configuration (platform 2 right ×2 → [36,24,47,27], platform 1
   right ×3 → [20,16,31,19], bar right ×1 → [24,36,43,39]), then ONE
   pour: `levels_completed=2` on the first pour, first level-2
   completion in this lab. Level 3 revealed: three spouts, three
   funnels, wider platforms — same mechanic, more streams.

Honesty note: (7) is a **demonstration-derived scripted rollout**, not
model capability — it must never be presented as the model solving
level 2, and the specific configuration must never be hardcoded as a
label source or reward. Its legitimate uses: (a) curriculum — a
plumbing-play collector that randomizes arrangements (click any movable,
move on all four axes, pour last), with demonstration replays as a few
seeds among many; (b) real `level_up=1` positives from environment
fields wherever arrangements complete levels; (c) controllability
samples for platforms and vertical axes. The model then has to learn
arrangement→outcome itself; the imagination planner searches
arrangements against its own learned heads.

Settle-label postmortem: level 2 has **no persistent fill state** —
settled frames differ only by gauge cells pour-to-pour. The fill-delta
redesign is dead; the outcome signal for "did this pour do what the
level wants" is `levels_completed` (level-up head), plus flow-path
dynamics the world model already learns.

## Next steps

1. Behavioral catch probe: model-guided glove placement (argmax of the
   catch head over candidate placements) vs random placement over seeded
   episodes; measure caught-paint fraction from environment pixels
   (still SCOUT, still local). The catch head is now good enough to drive
   this.
2. Waterfall control: extend multi-object play to simultaneous streams —
   per-glove catch heads already trained on roster labels; next is
   *choosing* which glove to command (control policy on top).
3. Edge-adjoin head: predict `adjoined`/`gap_px` from (frame stack, glove
   plane, direction) using the collected trials — PortBand alignment
   prediction, the "block adjoins wall/block" test.
4. Re-run ghost on human pour segments after catch-dense data closes the
   human-interaction gap.
5. Lock/key seeker: recalibrate `component_asymmetry` confidence on sp80/
   snake sprites, or feed it Contra components pre-filtered by the CNN's
   change head (moving vs static), then re-probe across games.
