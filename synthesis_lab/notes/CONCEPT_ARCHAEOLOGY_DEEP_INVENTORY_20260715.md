# Concept Archaeology — Deep Inventory (2026-07-15)

**Purpose:** Recover the multi-day conceptual stack across Dreamweaver, megarexus
(CONTRA / sim3d / legal_search), Loom (Rexus/LeWM), and transcript algorithm
mines — at the *idea* level, not the latest training run.

**Truth status:** Inventory / synthesis. Not scorecard evidence. Not a claim that
any single system is currently the live primary.

**Primary trees**

| Tree | Role in the grand design |
|---|---|
| `/home/frosty40/dreamweaver` | Chronometric body, Dream Kernel, A/B Q/A, CellWave, perception trust, DW-TJEPA plan |
| `/home/frosty40/megarexus` | CONTRA frame substrate, 3D pour world, legal mine, ONE LANE learned physics, wide model |
| `/home/frosty40/Loom` (+ Loom-gepa-bg) | Rexus core loop, LeWM chassis, GRAM/TIMR/Meta-Horizon, transcript algorithms |

---

## 0. The north-star philosophies (do not lose these)

### 0.1 CONTRA Charter — frame before agent

**Source:** `codexlab/docs/CONTRA_CHARTER.md` (2026-05-28)

- Job is to encode **how a human establishes a frame** on an unfamiliar board:
  figure/ground, actor vs scenery, goal-as-lock, key/fit, containment, support,
  symmetry-as-salience, change-under-action.
- As **deterministic ground-floor algorithms**, proven across ~25 local games.
- Agent comes **last**. An agent on a weak frame always starves.
- Fail-closed: `control_truth_admitted=False`, `goal_condition_admitted=False`,
  `action_choice_admitted=False`, `solve_condition_admitted=False`.
- Hypothesis ≠ truth. WIN only from engine GameState.
- Methodology: one brick at a time + **cross-game ablation regression**.

### 0.2 Dreamweaver A/B Q/A — open ontology

**Source:** `dreamweaver/docs/chronometric_ab_qa_overlay_v030.md`

Root interface is **not** trap/fuel/mirror/key enums:

| Symbol | Meaning |
|---|---|
| **A** | current self / controllable state estimate |
| **B** | desired outcome / objective estimate |
| **D** | dimensional gridspace for A, B, motion |
| **I** | internal imagination frame (2D/3D/latent drawable map) |
| **Q** | unresolved questions that can change A→B |
| **C** | confidence over the interpretation |

Gameplay = discovering objective modifiers via Q/A. Names may *emerge*; they are
not the starting ontology.

### 0.3 DW-TJEPA discipline — transition error is authority

**Source:** `dreamweaver/docs/leworldmodel_dreamweaver_rebuild_plan_2026-05-09.md`

Rejected pattern: council of advisors all say “plausible” → act.

Required pattern:

```text
transition model predicts S_next
branch score only through calibrated transition confidence
act
returned frame → prediction error E
E updates future confidence immediately
```

Copy LeWM discipline: **compact state + action-conditioned transition +
prediction-error feedback**. Do **not** blindly copy goal-latent distance as
the ARC decider.

### 0.4 Rexus core action loop — 12 steps

**Source:** `Loom/docs/REXUS_CORE_ACTION_LOOP.md`

Observe → encode topology → compare memory → generate candidates → predict
deltas → score (validity, goal_progress, information_gain, lattice_consistency,
action_cost) → prune → select → execute → observe → update memory → repeat.

Every selected action leaves an S/A/O trace. Fail closed. Lewis informs goals;
Rexus owns action; GameOps is transport only.

### 0.5 Honesty / provenance (shared across trees)

- Env / scorecard is the only grader for levels/WIN.
- Imagined outcomes are pre-action simulation; observed outcomes are calibration
  truth — **never mix**.
- Quarantine fences: spill_model, dropletsim/native_scene-as-solver, human routes
  as train labels, fixture win banks, source-derived answer keys.
- Provenance-tagged KnownMaps only after perception trust gates.

---

## 1. Chronometric / Dreamweaver foundation (deep math + trust)

### 1.1 4D event state (Contortion)

**Sources:** `chronometric_contortion_foundation.md`, foundation review v002

- Event state `x^μ = (t, x, y, z)` with **latent y = signed outcome axis**
  (not image y).
- Time-like velocity `u^μ`, momentum `p^μ`, phase-locked tensor `K(τ)`.
- Log-time phase `θ(τ)`; projected contortion force with hard projector
  `u_μ F_cont^μ = 0`.
- Potential-family basis (16 families) as inspectable evidence channels.
- Modes: audit / residual_once / residual_each / branch_rollout.
- `score_branch` / branch-library hotload + potential fallbacks.
- Long calibration arc V001–V055+ (goal_plan.md): signed-Y MAE driven near zero
  on bridge corpora; branch selection oracle match on heldout multi-action
  groups; ARC offline loop connectivity proven (ls20), solve not yet.

### 1.2 Dream Kernel contract

**Sources:** `dream_kernel_v001.md`, Rust `dream_kernel/src/lib.rs`, megarexus
`sim3d/imagination_kernel.py`

Owns:

- KnownMap (provenance-gated)
- SimState / Action / StepOutcome
- RayBundle + RayContact with stable object_id
- RayNetworkKind: beneficial / adversarial / structural / neutral
- signed_potential_y + potential_family per hit
- PotentialDatum with event_coord `(t, x, y_chrono, z)`
- ChronometricFrame overlay (event_mu, phase_theta, branch_direction_n, …)
- BranchMatrix, BranchPotential, ObjectLinkHypothesis
- FrameIntegrity hash chain
- imagined_vs_observed calibration slots
- NemoRelayPacket (optional semantic review)

Rule: unknown perception enters only after conversion to explicit known cells
with provenance.

### 1.3 Perception trust (V031)

- Labeled image → integer grid → 3D cell geometry → raymap → accuracy gate →
  `trusted` bool.
- Rays must not influence action selection unless gate passes
  (`goals.md` hard boundary).
- Ported SP80 checks in `sim3d/perception_gate.py` (targets, glyphs, spouts,
  mouths, rays, floor, coverage).

### 1.4 Sensory alignment (V032–V034)

- Visual map, 3D projection, ray trust, predicted next-state, observed next-state,
  outcome confirmation = one record per state/action.
- V033: imagined signed-Y is **pre-action**; observed is **post-action**.
- V034 human-eval smattering for isolated failure modes.

### 1.5 A/B Q/A overlay (V030)

See §0.2. Nemo = semantic Q/A reviser; NanoWM = constrained branch scorer.

### 1.6 CellWave multi-potential field (polarized attractors)

**Source:** `dreamweaver/src/cellwave_potential_field.py`

Schema: `arc_agi3.cellwave_multi_potential_field.v001`  
Policy: `sum_signed_potentials_then_follow_viable_positive_action_differential`  
Math: `newjordan_cellwave_advection_diffusion_emitters_currents_occluders`

| Component | Polarity | Source idea |
|---|---|---|
| goal_attraction_field | + | goal as negative sink / distance gradient |
| primary_goal_attractor | + | distance + heat-seeking wave |
| subgoal_attractor | + | intermediate objects as soft attractors |
| path_confirmation_current | + | optimal path + polarity-reversal cells |
| tunnel_flow | + | open-neighbor topology currents |
| wall_occluder_repulsion | − | walls |
| hazard_repulsion | − | hazards |
| bandit_feedback | ± | post-action Marco Polo |
| bayesian_belief | ± | belief filter |

Plus: velocity field from net potential; advection/diffusion propagation;
first-step candidates only if `pathing_confirmed`.

### 1.7 Heat-seeking goal rays

- Goal probability wave, hotspots, optimal_path_cells
- `direct_goal_ray_blocked`
- `wave_collision_polarity_reversal` (collision_cells, phase_locked_cells)
- `stop_ray_blaster` channel summaries
- Feeds CellWave; used in prize_runner scoring

### 1.8 Grid imagination raymap

**Source:** `chronometric_grid_imagination.py`

- Height map: playable height 0, raised blockers
- Non-wall objects = anchors emitting 8-dir rays
- GridRayHit path + contact; projects to RaytraceProbe imagination frames

### 1.9 DW-TJEPA rebuild targets

Compact latent transition model from legal B/A/A observations:

- predict after latent, no-op flag, action-effect class, route-progress class
- surprise score after return
- prediction error dominates branch ranking

---

## 2. Physics family (body ops on trusted maps)

### 2.1 Lane R — hydraulic stream tracer

**Source:** `sim3d/raycast.py`

- Volume-conserving deterministic streams
- Column-base shedding (core micro-rule)
- Ledge spill floor-gated (S1 iter2)
- Enclosed-entry latch for multi-source
- **Also:** `ray_aabb`, `drop_column` — geometric path/support queries
- Role: fast RANKER over thousands of arrangements

### 2.2 Lane P — 3D sphere PBD pour

**Source:** `sim3d/engine.py`

- Literal spheres out of the bucket
- Static colliders = extruded AABBs from Scene
- Deterministic jitter table; SimParams entire calibration surface
- Role: shape-general DECIDER substrate (user’s original “full 3D” idea)
- S0 particle uncalibrated baseline failed; stream lane carried L1

### 2.3 Induced dynamics families

**Source:** DW plan + imagination_kernel + dynamics_fit rounds

- FallingMaterial parameter grid
- FlowTrail family (selected by observation on SP80)
- Full-table fit, holdout gates, negative branch = escalate (don’t peek source)
- Grey residue: PALETTE bindings, hypothesis vocabulary bias, user search hints

### 2.4 Learned body plugs (megarexus later)

- ONE LANE dynamics UNet + win probe (env-only labels)
- WideWorldModel 25-game next-frame (no game id)
- D4 equivariance prior; held-out winner discipline
- Branch fleet as env-search teacher (not headline alone)

### 2.5 Unified physics operator view (synthesis)

On one trusted KnownMap:

| Operator class | Systems |
|---|---|
| Ray ops | Dream Kernel rays, grid raymap, heat rays, ray_aabb/drop |
| Field ops | CellWave polarized potentials, chronometric families, signed-Y |
| Body ops | spheres, streams, induced CA, learned dynamics |

Dreams rank; env grades.

---

## 3. CONTRA concept stack (frame substrate)

### 3.1 Concept index (Charter §4)

| # | Human frame idea | Detector family |
|---|---|---|
| 1 | Figure substrate (colors + components) | vision |
| 2 | Figure/ground surfaces | vision surface scores |
| 3 | Salience by (a)symmetry | symmetry / asymmetry |
| 4 | Actor affordance | oracle + calibration hypotheses |
| 5 | Goal/target affordance (lock/cavity) | oracle + calibration |
| 6 | Lock/key fit | oracle fit sensors |
| 7 | Support / action affordance | perception + StateCon |
| 8 | StateCon sticky relation groups | statecon |
| 9 | GeoBand relational geometry | geoband |
| 10 | Boundary passes / unresolved queues | boundary_pass |
| 11 | DeltaBand change-under-action | deltaband |
| 12 | Action-result transfer prior | match_calibration |
| 13 | Admission ladder | relationship_ladder |
| 14 | Oracle derivation head | oracle |

Plus: ExploreBand, ActionModBand, ViewSystem, TabulaRasa, Trainable levers,
Action semantics ledger, Match calibration stages 0/1.

### 3.2 PortBand structural graph (deep)

**Source:** `contra_head/portband.py` schemas/policies

Not “paint answers” — candidate geometry:

- Object **ports** and **port alignments**
- **Edge-lock** configurations (copy/compare/rank static shape locks)
- **Fleet** candidates / support combo matrices
- **Lineup matrix** (per-support route confidence — score not probability)
- **Route-flow lock** after single controlled activate_flow surface event
- **Action state tags** → interaction graph → interaction matrix
- Relative math profiles, temporal trajectory profiles
- Resource gauge profiles, motion/energy conservation assessment
- Solve-like candidate ledger (**clue not solve truth**)
- Object correspondence across frames

Boundary literals everywhere: not control/goal/action/solve truth.

### 3.3 Boundary-pass contract

**Source:** `BOUNDARY_PASS_SPEC.md`

- Inclusive primary relationship group first
- Refinement may exclude from subgroup but not erase from primary
- Every exclusion → unresolved boundary row with reason code
- Y objects: known shape, unknown attachment
- Boundary tests (not NL prompts): shape_equivalence_unknown,
  shape_plus_y_attachment, group_asymmetry, independent_vs_compound,
  actor_target_support_role_unknown, geometry_relationship_unresolved

### 3.4 Ghost relational memory

**Source:** `CONTRA_GHOST_RELATIONAL_MEMORY_ARCHITECTURE_2026-06-02.md`

- Ghost = **historical grounded** memory from controlled completion boundaries
- Oracle = **future speculative** sidecar
- Stores relation graphs at wins, not routes/coordinates
- Question: which current transforms make the board more like a prior
  **relationship graph**?
- Confluence scoring: one transform satisfying multiple relations

### 3.5 Crucible mechanic induction

**Source:** `CRUCIBLE_MECHANIC_INDUCTION_PLAN_2026-06-12.md`

- Mine (before, action, after) from admitted runs
- Induce local rules from anomalies; falsify against full corpus
- Promote zero-contradiction rules into forward-model **hypothesis ledger**
- Not control truth
- Activation identity transitions (not raw coordinates) as key insight
- Datasets: transitions, activation transitions, discrepancy, mechanics ledgers

### 3.6 Teacher / student signals

- Opus 4.8 public playthrough mining → reasoning corpus + teacher signals
- ActionSemanticsLedger as student packet producer
- Process supervision only, not solve truth

---

## 4. Rexus / Loom planning concepts

| Concept | Intent |
|---|---|
| LeWM | Encoder + action-conditioned latent predictor; SIGReg; CEM/MPC |
| Rex / Rexus | Concrete chassis + structured actions; owns action loop |
| Lewis / LeWMSight | Goal/context informer only — never action owner |
| GameOps | Signal transport / standardization |
| Goal packet | Probabilities over goal/mechanic/family/stage/tags; no actions |
| TIMR | Temporal inversion micro-reps: cycle consistency + irreversibility |
| GRAM | Internal recurrence + multi-trajectory width + LPRM process reward |
| LDT | Lattice constraints between recurrent passes; abstain if invalid |
| Meta-Horizon | Safeguard against short-horizon overfitting |
| LPRM / Q6 | Process reward / scorer — often training-blocked until real O_pred |
| State Prober | O_pred vs O_true deltas |
| S/A/O Bible | Provenance, identity, splits, quarantine for every training row |
| GRAM++ pretrain gate | Operating picture + path fan + lattice + leakage before train claims |

---

## 5. Human transcript algorithms (cross-game)

**Source:** Loom-gepa-bg `transcript_algorithms_confirmations_v1` +
per-game `*_algorithm_confirmations.json` (~25 games)

### 5.1 Cross-cutting algorithms (high value)

| Algorithm | Idea |
|---|---|
| **Control-locus probe** | First: what does each action actually control? Build effect table |
| **Irreversible-action gate** | Don’t spend/lose without on-goal prediction or first-class probe |
| **Non-controlled goal extractor** | Goals often fixed reference objects, not the avatar |
| **Transcript-corrected family router** | Correct over-generic family labels from play language |
| **Fit / resource planners** | Finite pieces, fuel, grow caps |
| **Curiosity loss ≠ budget drain** | Separate hazard/resource/time traps from slow exhaustion |

### 5.2 Recurring human first questions (from transcripts)

1. What is controllable vs fixed? (isolation of motion)
2. What is the goal object / reference (often non-controlled)?
3. What is irreversible / budget / fuel?
4. Route / topology vs local greed?
5. Symmetry / fit / gap completion?
6. Multi-entity / cooperative / adversarial timing?

### 5.3 Named mechanics mined per game (samples)

- **Liquid pour:** gravity_pour_action, piece_placement_routing,
  symmetry_gap_completion, isolated_flow_branching,
  vertical_close_horizontal_lock, faucet_down_then_side_link,
  fuel/pour conservation
- **Button wheel (lp85):** no_dpad_click_only, button_orbit_rotation,
  conveyor delivery, transfer lanes, convergence_control, timing phase
- **Gravity falls / routes:** obstruction-only control, gravity flip resources,
  hazard-default gates
- **Mirror cursors:** coupled multi-cursor transforms, wall isolation,
  pair convergence
- **Peg jump / chain links:** legal jump graph, trajectory collision,
  tight cluster heuristic
- **Wayfinder pool:** control-switch click, box-push delivery,
  corner-trap irreversibility
- **Cargo sort:** attribute_tuple_goal, fuel_conservation, wall pusher
- **Gravity well combine:** click-as-attractor agency
- **Code blocks:** bit-row opcode decode, program synthesis from guide priors
- **Key gate dockworkers:** attach/drop, autonomous mover timing, trap intercept

### 5.4 SP80 narration corpus (megarexus)

- 186 segments / 77 useful
- Themes: routing, isolation motion, faucet, conservation, multi-horizon planning,
  “assume all pieces then revise”

---

## 6. Legal_search mine (algorithms → features → policy)

25 systems consumed into 392-D v2_fullmine (gate signed). Residual debt:
PortBand Monte-Carlo parity, crucible grammar depth, DW multi-family online fit,
LeWM weight load, Goose CNN, match transfer_prior completeness.

Train labels legal only: did_change, weak taken-action, levels_up weight.
AdaJEPA online LoRA adapt path exists.
Path B overnight: synth levels yes; Arcade levels on best ckpt no.

---

## 7. Hard-won negative knowledge (keep)

| Falsified / banned | Why |
|---|---|
| Reactive CNN / BC / DAgger for SP80 complete | 0/18; deliberation gap |
| Static pre-pour outcome CNN as decider | Winner vs FP near-identical statics |
| Pixel-CA long rollout alone | Drift / overpaint (partially beaten by 3D sim) |
| dropletsim / native_scene solver | Answer key from game source — retracted |
| spill_model as reward | Privileged state + hardcoded win |
| Zero-shot multi-game dynamics transfer | Wide model: substrate shared, dynamics not |
| Wide pretrain helping SP80 | Architecture helps; weights don’t |
| Random L2 arrangement grind | 0/1400 without faithful multi-source imaginer |
| Multi-pour accumulation on SP80 | Fills reset every pour |
| AdaJEPA goal-latent distance as ARC decider | No honest goal image for L1+ |
| Promote on synth level rate alone | Path B: Arcade still thrash/no-op |

---

## 8. Concept → RDL slot map (how to reassemble)

| Deep concept | RDL layer |
|---|---|
| CONTRA frame stack + PortBand + boundary | L1 perception + L4 structure |
| A/B Q/A + open ontology | L5 controller / meta |
| V031 trust + KnownMap | L1 |
| Dream Kernel | L2 spine |
| CellWave + heat rays + polarity | L4 Field ops / L3 rank assist |
| Stream/sphere/induced/learned body | L2 Body plugs |
| Chronometric signed-Y + branch library | L3 ranker |
| Ghost + Crucible | Memory + mechanic induction |
| Rexus 12-step + S/A/O | Outer control loop |
| DW-TJEPA / LP / wide | Transition authority + multi-game body |
| Transcript algorithms A1–A6 + per-game | Proposer macros + irreversibility gates |
| TIMR / GRAM / Meta-Horizon | Training regime safeguards |
| legal_search consume | Multi-game feature/policy substrate |

---

## 9. Artifact index (where to reopen work)

### Dreamweaver
- `goals.md`, `goal_plan.md`, `goal_test_results.md`
- `docs/dream_kernel_v001.md`, `dream_kernel_ralph_async_loop.md`
- `docs/chronometric_*.md`, `chronometric_ab_qa_overlay_v030.md`
- `docs/leworldmodel_dreamweaver_rebuild_plan_2026-05-09.md`
- `src/cellwave_potential_field.py`, `chronometric_grid_imagination.py`,
  `chronometric_map_perception.py`, `dreamweaver_prize_runner.py`
- `dream_kernel/` Rust crate

### megarexus CONTRA / sim3d
- `codexlab/docs/CONTRA_CHARTER.md` + Ghost, Crucible, Oracle, Controller map
- `codexlab/contra_head/*` (esp. portband, boundary, ghost, deltaband)
- `synthesis_lab/orchestration/sim3d/*` + notes SIM3D_*, DW_*, ROADMAP_*
- `legal_search/docs/*`, `MINING_THEN_TRAIN.md`
- `synthesis_lab/notes/ONE_LANE_*`, `WIDE_MODEL_*`, `LP_TRAINING_*`

### Loom / transcripts
- `Loom/docs/REXUS_CORE_ACTION_LOOP.md`, `rex.md`, TIMR, GRAM, Meta-Horizon, LRS
- `Loom/docs/IDEAS_ROUND4_100.md` (100 planner/reward ideas)
- `Loom-gepa-bg/experiments/2026-05-21_transcript_algorithms_confirmations_v1/`
- Per-game algorithm_confirmations under incoming_audio_transcription_v1
- Human goal discovery calculus research hunts

---

## 10. Honest summary of the “lost” work

You did **not** only invent pour balls. Across May–July 2026 you built:

1. A **philosophy of frame-first abstraction** (CONTRA) with fail-closed truth.
2. A **chronometric world-model body** with signed outcome axis and branch scoring.
3. A **Dream Kernel** imagination substrate with polarized ray networks.
4. A **CellWave polarized attractor** navigation system with heat-seeking rays.
5. A **full 3D pour physics lab** (streams + spheres + induced CA + later learned).
6. A **PortBand / Ghost / Crucible** relational geometry and mechanic induction stack.
7. A **Rexus 12-step S/A/O action owner** with Lewis/GameOps role split.
8. A **cross-game transcript algorithm library** (control locus, irreversibility,
   family-specific planners).
9. A **legal multi-game mine** of 25 systems into blend features.
10. Training safeguards: TIMR, GRAM/LPRM, Meta-Horizon, honesty audits, quarantine.

What fragmented was the **integration spine**, not the depth of the ideas.
This inventory is the recovery map for reassembly under REXUS Dream Lattice
or any successor name.
