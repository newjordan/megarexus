# CONTRA Charter — The Frame-Building Substrate

> Durable north star. If a decision contradicts this doc, stop and reconcile before proceeding.
> Last set: 2026-05-28. Owner: human. Drafted with Claude, reviewed with Codex.
> Coordination contract: `claudelab/CODEX_CLAUDE_COMMS.md`.

## 1. The Goal (read this first, every time)

CONTRA's job is to **encode the concepts a human uses to establish a frame on an
unfamiliar board** — figure/ground, actor vs. scenery, goal-as-lock, key/fit,
containment, support, symmetry-as-salience, change-under-action — as
**deterministic ground-floor algorithms**, and to **prove they generalize across
the ~25 local ARC-AGI-3 games**.

That generalizing substrate is the *nutrition*. Only once frame-building
generalizes does a learning/reasoning agent get attached.

**Why the agent comes last.** Every prior attempt to bolt on an ad-hoc
reasoning/play system has *starved*: the agent has no principled way to
establish a frame, so its "reasoning" is ungrounded. The bottleneck is
**abstraction**, and abstraction must be **earned across games**, not assumed
per game. An agent on a weak frame always starves. We are building the frame
first, on purpose.

CONTRA is **not** the solver, not the agent, not the game motor. It is the
perception + abstraction + verification substrate.

## 2. Non-Negotiables (the honesty contract)

These exist because a controller that fabricates progress is the exact failure
this architecture is built to catch.

- **No layer promotes itself.** Every stage stays fail-closed:
  `control_truth_admitted=False`, `goal_condition_admitted=False`,
  `action_choice_admitted=False`, `solve_condition_admitted=False`.
- **Hypothesis ≠ truth.** Affordances, StateCon, GeoBand, symmetry, and Oracle
  outputs are labeled candidates. They never silently become control/goal/solve
  truth.
- **Match calibration & transfer priors reweight roles only.** Stage-0/stage-1
  init calibration and the action-result transfer prior adjust *current-frame
  role candidates*; they never admit actor truth, target truth, control truth,
  action choice, or solve truth.
- **WIN is engine-sourced only.** Any win/score signal comes from
  `arcengine.GameState` (`WIN / GAME_OVER / NOT_FINISHED / NOT_PLAYED`), never
  asserted by a hypothesis, a review page, the substrate, or (later) the agent.
- **No claim without a reproducible artifact.** A passing check means a command
  + output exists in the run, re-checkable from hashes by a verifier. If an
  artifact lacks the requested condition, mark it `not_run`/`stale`/`proxy`/
  `invalid` — do not patch wording around the gap.
- **No solve/beaten language** anywhere in the substrate phase.

## 3. Methodology — one brick at a time, with cross-field ablations

We build the system **one game at a time**, improving its ability to **derive**
the abstractions per game, cross-checking accuracy across every game already
integrated as we add the next, then human-calibrating/fine-tuning everything we
can. No stone unturned. Purpose-built.

### Per-game integration loop (repeat for each new game)

1. **Derive.** Run the CONTRA concept stack on the new game's frames. Record
   what each concept's detector produces *with no game-specific answer fed in*.
2. **Score against the answer key.** Compare derived abstractions to the
   human-reviewed expectations for that game (`answer_key_calibration`). Record
   per-concept derivation accuracy.
3. **Cross-game ablation regression.** Re-score **all already-integrated games**.
   A brick added for game N must not drop accuracy on games 1..N-1. For each new
   field/concept, **ablate it** (remove it, re-score across the whole integrated
   set) and record the marginal contribution. Dead weight gets pruned;
   regressions block the brick.
4. **Human calibrate / fine-tune.** Exhaustively. Fill the answer key, tighten
   thresholds, resolve boundary-queue items, sign off.
5. **Lock in.** The game joins the integrated set; its scores become part of the
   regression suite for every future brick.

## 4. Concept Index

Each concept = the human board-reading idea, its deterministic detector (real
symbols in the active `codexlab/` tree), and its status. Status reflects current
code, not aspiration.

> **Metric pattern for every concept:** derivation match-rate vs. answer key per
> game; must not regress across the integrated set; ablation delta recorded.

| # | Concept (human frame idea) | Deterministic detector | Status |
|---|---|---|---|
| 1 | **Figure substrate** — exact raw colors + connected components | `vision.raw_pixel_components_with_labels`, `vision.contact_graph`, `controller_head.build_raw_color_component_packet` | works on sp80; cross-game unproven |
| 2 | **Figure/ground** — floor / wall / background / checkerboard surfaces | `vision.score_floor`/`score_wall`/`score_background`/`analyze_scene_surface_hypotheses`, `vision.checkerboard_floor_pattern_hypotheses` | sp80 only |
| 3 | **Salience by (a)symmetry** — what stands out as non-scenery | `vision.component_asymmetry`, `symmetry.build_symmetry_hypotheses`, `vision.antisymmetric_relationship_hypotheses` | sp80 only |
| 4 | **Actor affordance** — action-origin / emitter region | `controller_head.calibration_actor_affordance_hypotheses` (seeded) + `controller_head.oracle_actor_affordance_hypotheses` (derived) | sp80 seeded; derivation embryonic |
| 5 | **Goal/target affordance** — interaction / open region (the lock/cavity) | `controller_head.calibration_target_affordance_hypotheses` (seeded) + `controller_head.oracle_target_affordance_hypotheses` (derived) | sp80 seeded; derivation embryonic |
| 6 | **Lock/key fit** — does the actor (key) fit the goal (lock) | `oracle._best_open_cavity_fit`, `oracle._best_key_fit_for_lock`, `symmetry` lock/key sensor | sp80 only |
| 7 | **Support / action affordance** — what enables or constrains the relation | `perception.score_support_candidate`/`score_compound_support_candidate`, `statecon.build_statecon_support_affordances` | sp80, manual/gated admission |
| 8 | **StateCon** — sticky relational condition group over actor/target/support | `statecon.build_state_condition_hypotheses`, `statecon.relation_features`, `statecon_identity_mlp` scaffold | sp80 only |
| 9 | **GeoBand** — relational geometry: bearings, distances, angular error/tolerance, lateral bandwidth, symmetry | `geoband.build_geoband_hypothesis`, `geoband.relationship_edge` + boundary passes | sp80; ar25 smoke |
| 10 | **Boundary passes / queues** — unresolved attachments, known-Y objects, role/geometry uncertainty | `boundary_pass.unresolved_boundary` + geoband boundary queue | sp80 only |
| 11 | **Change-under-action (DeltaBand)** — how relations shift across a controlled action boundary; null-action signal | `deltaband.build_deltaband_observation`/`build_deltaband_observation_set` | gate present, controlled runs partial |
| 12 | **Action-result transfer prior** — *temporal* context from observed action/result; per-function confidence math (`actor_origin`/`target_effect`/`lock_key_effect`), fast decay on misses, reweights current-frame roles | `match_calibration/transfer_prior.build_action_result_transfer_prior` (+ `stage0`, `stage1` init calibration) | new in `codexlab`; wired through controller_head/oversight/review page; in static sp80 00–03 report stays `no_prior_observations_supplied` (no controlled observations injected yet) |
| 13 | **Admission ladder** — the fail-closed contract tying all stages together | `relationship_ladder.build_contra_relationship_ladder` | works |
| 14 | **Oracle (derivation head)** — turns the priors into label-free goal/actor guesses to be *scored against* the answer key | `oracle.py` (goal candidate, actor candidate, lock/key, actor-facing geometry) | sp80; ar25 smoke; this is the generalization engine to grow |

Concepts 4–6, 12, and 14 are the heart of the bet: today they lean on
human-seeded calibration / no-prior-observation defaults, and Oracle + the
transfer prior are the embryo of label-free, temporally-informed derivation.
Maturing derivation here, game by game, is the central work.

### Two kinds of calibration (do not conflate)

- **`answer_key_calibration`** — human-reviewed expectations used to *score*
  derived abstractions. **Not a detector output.** Surfaces today:
  `manual_calibration_review.human_reviewed_expectations` (actor ids, target ids,
  support/action ids, connectors, known-Y objects, unresolved objects, image
  hash), `sp80_surface_calibration.json`, and the cross-game smoke calibration
  template (actor/target/support attachments).
- **`match_runtime_calibration`** — initialization and evidence-boundary signals
  gathered *during a match*: stage-0 init canvas click, stage-1 init-choice
  probe, DeltaBand controlled boundaries, and the action-result transfer prior.
  These reweight roles only; they never admit truth.

Calibration files are the **answer key the derivation is scored against**, not a
license to hand-label the answer and call the concept solved. The thing being
improved is the **derivation ability**; cross-game regression proves it
generalizes incrementally.

## 5. The Phase Gate (the thing that keeps getting missed)

**No agent. No reasoning/play loop. No action-choice admission. No generated
intent. No win-claiming.** Until the concept index above derives correctly
across the integrated game set and survives cross-game ablation + human
fine-tune.

The single-step `sp80_player_harness` control-fidelity check is allowed (it
tests the motor contract, not gameplay). It must stay demoted to a unit test and
must never be described as "playing."

**Calibration carveout.** Calibration clicks/probes — stage-0 init click,
stage-1 init-choice probe, DeltaBand controlled before/action/after boundaries —
are allowed **only** as initialization or evidence-boundary probes. They are
**not gameplay moves** and do **not** admit action choice.

When the gate is eventually opened, it opens as exactly **one** sanctioned door:
a future agent that *commits to an action under uncertainty* while CONTRA stays
honest about verified-vs-assumed — and even then, WIN is read from the engine.

## 6. Transfer Prior Guardrail (the level 03 lesson)

If levels 01 and 02 record controlled actor-origin success for the
purple/magenta paint head, then level 03 may carry a high-confidence
`actor_origin` transfer prior for the matching purple/magenta component.

That does exactly two things:

1. Reweights the purple/magenta component toward **primary actor-origin**.
2. Reweights other actor-like components toward **secondary relational** roles
   (key/lock, symmetry, anchor, or affected-object context).

It does **not**: delete the secondary object, treat it as an equal primary
actor, admit actor truth, admit action choice, infer the target, or claim a
solve. (In code: `nonprimary_due_to_primary_actor_prior` marks the demotion;
admission flags stay false.)

## 7. Current State (2026-05-28)

- **Integrated:** `sp80` (589a99af) — primary, worked, seed-calibrated; transfer
  prior wired across levels 00–03 (static, no injected observations yet).
- **Smoke only:** `ar25` (0c556536) — cross-game GeoBand/Oracle artifact exists;
  calibration template is `pending_human_review`, not integrated.
- **Untouched (23):** bp35, m0r0, ls20, dc22, vc33, lf52, ka59, tu93, cd82,
  sc25, g50t, wa30, lp85, cn04, tr87, sk48, ft09, s5i5, re86, r11l, su15, tn36,
  sb26.
- **Path boundary (per `CODEX_CLAUDE_COMMS.md`):** `codexlab/` is the **active
  committed implementation path** (Codex owns implementation; git-tracked, source
  of truth for code). `claudelab/` is the **review/planning + charter mirror** —
  NOT active implementation unless the human explicitly promotes it. This charter
  and the comms doc live in `claudelab/` (currently untracked). Modules are
  referenced by package-relative path (`contra/...`, `oracle.py`); resolve them
  against `codexlab/`. Never present a `claudelab/` result as implementation
  evidence unless the command was actually run from that path.

## 8. Open Questions To Resolve As We Go

- Where should controlled action-result observations be recorded for match
  memory (so the transfer prior leaves `no_prior_observations_supplied`)?
- Which game is brick #2 (first true cross-game derivation test)?
- The concrete per-concept scoring function + accuracy threshold per game.
- How the cross-game ablation regression is run and recorded (a regression
  harness over the integrated set).
- The exact answer-key schema once more than one game is integrated.
- Should `claudelab/CONTRA_CHARTER.md` + `CODEX_CLAUDE_COMMS.md` be git-tracked
  (charter-path tracking) or stay untracked review mirror? (human decision)
