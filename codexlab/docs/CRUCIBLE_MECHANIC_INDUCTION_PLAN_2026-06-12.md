# Crucible Mechanic Induction Loop for CONTRA

Date: 2026-06-12
Status: pinned implementation plan, operating contract, and live-work handoff
Scope: CONTRA/SP80 mechanic induction for L1 paint-control. This document is not solve evidence.

## Current Decision (2026-06-12)

The source-edge activation probe class is in `stale_loop`; see the source-edge
failure notes and stale-loop boundaries below. Coordinate/source/edge/target
distance is an insufficient activation outcome model.

Do not run L2 again until an activation transition model keyed by
selected-object identity and post-activation identity is verifier-green, or
Crucible rule induction changes the representation enough to explain the c0008
`[62,52]` contradiction.

Observation evidence, not solve evidence: the contradictory c0008 pair has the
same display coordinate `[62,52]`, raw coordinate `[1,11]` under
`rotation_quarters=2`, and different selected-object identity transitions. The
positive row in
`codexlab/runs/_system_run/paint_control_skill_level02_source_edge_activation_probe_20260612T_local/level_02/packets.jsonl`
action 38 changes `before_state.selected_name` from `plzwjbfyfli-5` to
`plzwjbfyfli-6` and has `distance_delta +0.5`. The negative row in
`codexlab/runs/_system_run/paint_control_skill_level02_source_edge_activation_ledger_20260612T_local/level_02/packets.jsonl`
action 31 changes `before_state.selected_name` from `plzwjbfyfli-6` to
`plzwjbfyfli-5` and has `distance_delta -2.5`.

Implementation order: activation row schema, mining over admitted runs, then a
narrower `source_edge_outcome_model`. These artifacts remain evidence only and
must carry all-false truth flags.

## Context

CONTRA's L1 blocker is the mechanic, not the scorer. The best paint-skill runs still show wrong-direction movement and blocked/stationary actions. Those rows are currently treated as negative feedback, but three hand-coded redirect attempts have not solved L1 and hand-coded mechanics do not generalize to other ARC-AGI-3 games.

Crucible closes the Ghost loop as a falsifiable mechanic-induction pipeline:

1. Mine `(before_grid, action, after_grid)` transitions from existing controlled run artifacts.
2. Induce local transition rules from prediction anomalies.
3. Falsify candidate rules against the full transition corpus.
4. Promote zero-contradiction rules into a forward model.
5. Feed the model into action selection, discriminating probes, and the loop harness.

The only solve claim remains `levels_completed_delta > 0` in a controlled run with a passing Angel/report verifier.

Current target failure bands from the paint-skill level-2 work:

- approximately `20%` wrong-direction moves in the best controlled paint-skill runs
- approximately `7.5%` blocked or stationary moves
- ranked fallback collapse when fallback fires without mechanic evidence
- exact activation alignment failures at the capture edge

These rows are transition evidence, not just negative reward. The product goal is to learn local mechanics from `(before_grid, action, after_grid)` evidence and feed the learned forward model into CONTRA without making the model an authority source.

## Authority Boundary

Crucible is a hypothesis-ledger and forward-model layer. It is not control truth, solve truth, goal truth, or Oracle truth.

- CONTRA remains the integrated system.
- `PaintControlSkillBrain` remains the action-authority backend for this mechanic while it is selected.
- PortBand supplies present-state geometry, source/edge candidates, and static hypothesis context only.
- DeltaBand supplies observed surface and state-change evidence.
- Ghost records verified calls, transitions, and outcomes only after recorded runs.
- Oracle supplies goal/review hypotheses only.
- ExploreBand probes remain inside the existing Oracle selection to executor validation chain.
- Admission remains gated by the existing loop predicate; Crucible cannot bypass `gate.ok`, `target_delta > 0`, or `gate.no_regression`.

All Crucible predictions must carry all-false truth flags and the boundary literal:

```text
mechanic_forward_model_is_hypothesis_ledger_composition_not_control_truth
```

## Durable Outputs

Crucible produces machine-checkable local artifacts before live wiring consumes them:

- Admitted-runs manifest: `codexlab/datasets/sp80_crucible_admitted_runs_v0/admitted_runs.json`
- Transition corpus: `codexlab/datasets/sp80_crucible_transitions_v0/{corpus.jsonl, manifest.json}`
- Activation transition corpus: `codexlab/datasets/sp80_crucible_activation_transitions_v0/{corpus.jsonl, manifest.json}`
- Source-edge outcome model: `codexlab/datasets/sp80_source_edge_outcome_v0/{ledger.json, manifest.json}`
- Discrepancy ledger: `codexlab/datasets/sp80_crucible_discrepancy_v0/{anomalies.jsonl, ledger.json}`
- Mechanic ledger: `codexlab/datasets/sp80_crucible_mechanics_v0/{mechanic_ledger.json, manifest.json}`
- Stage runs: `codexlab/runs/_crucible/crucible_<stage>_<UTCstamp>/`
- Verifier: `codexlab/contra_verify_crucible.py --artifact corpus|activation_corpus|source_edge_outcome|ledger|rules|mechanics|run`

Every consumer records the mechanic ledger sha when it uses one. Missing ledgers fail open unless a phase explicitly says to fail closed for offline verification.

## Verified Groundwork

- Existing artifacts contain about 100 `packets.jsonl` files, about 3,946 packets, and about 1,590 usable transitions with both `.npy` frames on disk.
- Grids are `64x64 int16`, palette `{1,4,6,8,9,11,12,14}`.
- `before_state.selected_coord` is `[x,y]`; saved `.npy` grids index `grid[x,y]`. `selected_display_coord` equals the canonical coordinate from `contra_head/crucible/canon.py`. For example, raw `[1,11]` under `rotation_quarters=2` on a `64x64` grid canonicalizes to display `[62,52]`. `contra_head/crucible/canon.py` is the only conversion authority.
- `Ghost.RealtimeMechanicMapper.observe()` exists historically but was not wired; its Wilson lower-bound reliability math should be copied with attribution from git `b77d0a43:codexlab/contra/ghost_mechanics.py`.
- `contra_loop/loop.py` Step 2 `DISCOVER-GAP` is a stub; the admission predicate stays untouched.
- ExploreBand currently has observation-only action probes; the Oracle selection to executor validation authority chain must remain unchanged.
- Reuse patterns from:
  - `codexlab/paint_control_skill_motion_effect_mine.py`
  - `codexlab/contra_verify_ghost.py`
  - `codexlab/contra_loop/keystones/sp80_l1_gravity_flip.py`
  - `codexlab/datasets/sp80_control_sft_v0/`

## Current Paint-Control Failure Basin

The live paint-control blocker is not only action scoring. The controller can often reduce target distance, but it still lacks a reliable mechanic model for:

- wrong-direction movement, including cases like issuing `move_left` and observing the actor move right
- stationary moves caused by blocked edges or object contact
- activation readiness at the final capture step
- separating harmful fallback actions from useful redirect probes

The current interpretation is:

- Wrong-direction and stationary rows are not just negative labels. They are transition evidence for the hidden mechanic.
- Goal-distance reduction is a useful diagnostic, but it is not solve evidence.
- Exact `levels_completed_delta > 0` in a controlled run remains the only solve claim.
- Broad hand-coded activation tolerance is unsafe unless a verifier-backed port or learned forward model says the edge is actually ready.

## Current Operating Objective

The next test is the first real Puff/Crucible control test, not another broad training run. The policy should still supply learned paint-control intent, but CONTRA must use its own lock/key, asymmetry, PortBand, DeltaBand, Ghost, and Oracle machinery to isolate which local mechanic and which goal/edge are actually valid.

Working hypothesis:

- If Puff can paint where directed but CONTRA still cannot get paint into the bucket, the blocker is goal isolation, source/edge authority, or mechanic induction, not raw policy capacity.
- If motion demotion correctly removes blocked or away moves and fallback then chooses harmful `select_object` or `activate_flow`, the missing representation is object/edge outcome prediction.
- Same-basin reruns are not progress. After repeated failures in the same representation class, the next valid step is a new observable state/action representation, not a new penalty constant or longer drive.

Immediate loop:

1. Keep using the controlled SP80 harness with `--backend paint_control_skill`.
2. Inspect `packets.jsonl`, `brain_calls.jsonl`, before/after frames, Angel report, verifier output, and feedback export after each run.
3. Prefer small representation changes backed by transition evidence.
4. Commit and push source changes only after verifier gates pass and the run either improves controlled metrics or changes the failure basin.
5. Do not claim solve unless `levels_completed_delta > 0` in a controlled run with a passing review/verifier surface.

## PortBand and Controller Fidelity Workstream

Crucible depends on the controller seeing the right geometry and feedback. These fixes can land before or alongside the offline induction phases.

### A. Goal Analysis and Capture

A1. Replace hardcoded goal fallbacks with PortBand-derived targets.

- `paint_control_skill.py` currently has fallback target constants for missing goal labels.
- PortBand already computes target ports with open direction and bucket geometry.
- Route those target rows through to the skill so the controller never aims at guessed coordinates.

A2. Add goal-completion awareness to capture.

- Target rows should carry per-target fill state from DeltaBand surface events.
- Capture logic should skip already-satisfied goals instead of relocking them.

A3. Audit Monte Carlo fleet sampling for goal blind spots.

- `_build_portband_fleet()` samples only a subset of support combinations.
- Add sampled/total counters to the fleet packet.
- On SP80 L1, verify the known solution geometry survives sampling.
- If not, raise the budget or bias sampling toward the current target's ports.

A4. Make asymmetric alignment tolerances cell-size derived.

- Single-port and dual-port projection tolerances should be derived from grid/display scale rather than hardcoded pixels.
- Regression-test with the L1 staircase geometry.

### B. Feedback Messaging to the Controller Head

B1. Fix visible-alignment truncation before feedback.

- `visible_alignments = deduped[:32]` can hide alignments before surface feedback has a chance to rerank them.
- Apply feedback before truncation or keep a feedback-eligible reserve pool.
- Regression: generate more than 50 alignments and prove a hidden alignment with strong dynamic evidence surfaces.

B2. Re-score after feedback instead of freezing scores.

- Edge-lock quality and effective lineup confidence must be recomputed after feedback-adjusted sorting.
- The lineup matrix consumed by the head should reflect the final score order.
- Preserve confidence labeling as hypothesis scores, not probabilities or truth.

B3. Strengthen the source-key contract.

- Rows dropped for `active_portband_source_key` mismatch should emit an explicit `feedback_scope_mismatch` event.
- Regression: matched rows receive feedback, unmatched rows do not, and mismatch is observable.

B4. Close packet-regression gaps.

Add assertions for:

- truncation behavior
- edge-lock ledger `rows` versus `top_rows[:8]` consistency
- score immutability or recomputation after feedback, depending on the final contract
- effective-confidence formula

### C. Dynamic Motion, Fallback, and Activation

C1. Mine issued-direction versus observed-delta pairs.

- If deltas depend on board position or local contact geometry, treat them as mechanic evidence.
- That evidence belongs in Ghost/Crucible forward modeling, not as a permanent negative-feedback penalty.

C2. Rebuild ranked fallback as evidence-gated.

- Fallback should choose candidates with local observed support, not only candidates allowed by a reason ladder.
- Gate candidates on whether that action has moved that object in a useful direction near the current position.

C3. Treat adjacent activation as a model question, not a tolerance hack.

- `target_distance == 1` can be a coordinate-scale symptom, but activation attempts at distance one must require stronger evidence.
- Acceptable evidence sources are verifier-backed PortBand exact port alignment or a promoted Crucible rule.
- Grid distance alone is not sufficient.

C4. Establish a fixed accuracy benchmark before tuning.

- Use a standard harness run with fixed seed/config and at least 80 actions.
- Report positive-progress, wrong-direction, stationary, selection success, activation success, and level delta.
- `levels_completed_delta` remains the solve gate.

## Operating Loop

Use this loop for live paint-control work while Crucible is being built:

1. Inspect the latest controlled run and its `packets.jsonl`, `brain_calls.jsonl`, before/after frames, Angel report, verifier output, and feedback export.
2. Identify the current failure basin from the recorded transition evidence.
3. Make the smallest representation-level change aligned with that basin.
4. Add a focused regression for the contract being changed.
5. Run `paint_control_skill_regression.py`, `codexlab/dev_check.py`, and `sp80_integration_smoke.sh` as appropriate for the change.
6. Run a short controlled level-2 drive through the SP80 harness.
7. Export feedback, inspect Angel/verifier/summary/packet evidence, and compare against the fixed baseline.
8. Commit and push only when the basin changes or the controlled metrics improve without introducing a worse failure.
9. Stop after repeated same-basin failures and require a real representation change before launching more runs.

Do not launch hour-scale training as a substitute for missing transition representation. Training becomes useful after the action/outcome model is no longer stuck in the same hidden-mechanic basin.

## Flexible Watch-and-Refine Loop

For active test drives, use a short controlled loop that can change direction as evidence arrives:

1. Run one controlled level-2 drive with `--backend paint_control_skill`.
2. Export feedback labels and inspect packet/brain-call decisions before changing code.
3. Classify the basin as motion prediction, source selection, activation readiness, goal isolation, or fallback exhaustion.
4. Make one representation change only.
5. Add or update the smallest regression that pins that contract.
6. Run the local gate for that change.
7. Run one new controlled level-2 drive.
8. Compare level delta, positive-progress rate, wrong-direction rate, stationary rate, selected-source changes, activation attempts, and final failure reason.
9. Commit and push only if the basin changes, the controlled metrics improve, or the instrumentation exposes a missing state that was previously invisible.
10. Stop on two verified same-basin failures and record `stale_loop`.

The controller is being tested for whether it can use its lock/key and asymmetry machinery to isolate the right goal, source, and edge. If Puff can paint where directed but paint still does not reach the bucket, the next work item is CONTRA goal/mechanic isolation, not more raw policy training.

## Suggested Implementation Order

1. C1 direction-versus-outcome mining.
2. Stage A transition corpus miner and verifier.
3. Stage B discrepancy ledger and redirect-trigger histogram.
4. B1 and B2 feedback reranking fixes.
5. Stage C and Stage D induction, falsification, mechanic ledger, and keystone.
6. A1 target routing and C3 evidence-backed activation readiness.
7. F1 demotion-only paint-skill reranker.
8. F2 bounded scorer modifier.
9. Stage E discriminating probes.
10. Stage G loop harness wiring and admission attempt.

## Go/No-Go Milestones

- Stage A go: full corpus mined, manifest mismatch statistics in the expected band, corpus verifier green.
- Stage B go: discrepancy ledger built, v0 disagreement recomputes, redirect-trigger color histogram reviewed.
- Stage C go: at least one blocked and one redirect candidate induced under the grammar verifier.
- Stage D go: replay falsification promotes only zero-contradiction rules; mechanic verifier green.
- Keystone go: `sp80_l1_surface_redirect_chain.py` predicts win-bank L1 actor deltas better than a near-zero match rate.
- F1 go: demotion-only paint-skill integration is fail-open and regression-protected.
- F2 go: bounded scorer modifier proves all Crucible signal magnitudes stay within cap and do not stack with committed-rule bonuses.
- Stage E go: mechanic-discrimination probes preserve authority flags and cap at two per frame.
- Stage G go: loop admission attempt runs through the unchanged gate. Only `levels_completed_delta > 0` is a solve claim.

If the keystone reports near-zero actor-delta matches, stop live wiring and iterate offline grammar v1 with component or largest-object features before any consumer trusts Crucible predictions.

## Implementation Snapshot: 2026-06-12

This section records the branch state after the first Crucible implementation pass. It is a status note, not a solve claim.

Committed branch state:

- `8a6127a6 Add Crucible transition corpus miner`
- `e3dfd72e Add Crucible discrepancy ledger`
- `e12454a8 Use Crucible blocked predictions in paint skill`

Implemented:

- Stage A transition corpus miner and verifier path.
- Stage B discrepancy ledger and verifier path.
- A PRE-GRAMMAR BRIDGE from discrepancy-derived exact rules. It is not Stage C/D-falsified and is not the promoted mechanic ledger.
- F1-style paint-skill reranking for predicted blocked moves, demotion-only and fail-open when the ledger is missing or has no opinion.

Tracked artifacts:

- `codexlab/datasets/sp80_crucible_transitions_v0/corpus.jsonl`
- `codexlab/datasets/sp80_crucible_transitions_v0/manifest.json`
- `codexlab/datasets/sp80_crucible_discrepancy_v0/anomalies.jsonl`
- `codexlab/datasets/sp80_crucible_discrepancy_v0/ledger.json`

Current discrepancy-ledger headline:

- `transition_count: 708`
- `anomaly_count: 185`
- status counts: `anomaly: 185`, `matched_v0: 355`, `selection_changed_transition: 168`
- blocked counts: `move_down: 10`, `move_left: 1`, `move_right: 57`, `move_up: 109`
- currently useful promoted runtime rule from the minimal loader: `move_right` with forward-cell `OOB/-1` predicts blocked

Controlled level-2 evidence after F1 demotion:

- Run: `codexlab/runs/_system_run/paint_control_skill_level02_crucible_demote_penalty_20260612T_local`
- Category: controlled run.
- Angel verifier: passed.
- `levels_completed_delta_total: 0`.
- Feedback counts: `positive_goal_distance_reduced: 25`, `hard_negative_stationary_move: 6`, `negative_goal_distance_increased: 7`, `neutral_no_goal_progress: 2`.
- Forward model status in brain calls: `demoted: 10`, `no_opinion: 30`.

Interpretation:

- The forward model successfully demotes a repeated blocked `move_right` at the right boundary trap.
- The run still does not solve level 2.
- The remaining basin is not fixed by another rank tweak: at the trap, `move_up` is stationary, `move_right` is predicted blocked, and remaining fallback actions such as `move_left`, `move_down`, and broad `select_object` are harmful or unauthoritative.
- A scout that failed closed when no productive fallback remained ended the run with `brain_error`; do not reintroduce that guard as-is unless the desired behavior is explicit controlled failure rather than continued exploration.

Next representation-level work:

- Mine selection outcomes around the trap and compare negative trap selections with historically positive selections from `[55,56]`.
- Promote selection authority only if it is backed by source/target identity, PortBand key, or outcome evidence.
- Continue Stage C and Stage D induction so redirects and blocked moves are learned as falsifiable local rules instead of fallback penalties.
- Do not launch another level-2 drive from the same basin until the representation changes.

### Offline Activation Representation Pivot

This snapshot records the no-live-run pivot after the c0008 source-edge
contradiction. It is observation evidence only, not solve evidence, control
truth, action-choice truth, or prediction truth.

Implemented offline artifacts:

- Admitted-runs manifest:
  `codexlab/datasets/sp80_crucible_admitted_runs_v0/admitted_runs.json`
  with `default_policy: exclude`.
- Re-mined motion corpus:
  `codexlab/datasets/sp80_crucible_transitions_v0/manifest.json`
  has `row_count: 956`, `not_admitted_skipped_count: 45`, and baseline
  `mismatch_rate: 0.298172`.
- Activation transition corpus:
  `codexlab/datasets/sp80_crucible_activation_transitions_v0/manifest.json`
  has `row_count: 59`, field presence `focus_edge: 24`,
  `active_portband_source_key: 13`, `target_distance: 19`, and
  `before_selected_name/after_selected_name: 59`.
- Source-edge outcome model:
  `codexlab/datasets/sp80_source_edge_outcome_v0/manifest.json`
  has `entry_count: 39`, tier histogram `candidate: 6`, `observed: 33`, and
  zero promoted entries.

The required c0008 audit is verifier-green:

- `c0008_contradiction_audit.separated: true`
- `separating_fields: ["before_selected_identity"]`
- positive key before-selected identity: `plzwjbfyfli-5`
- negative key before-selected identity: `plzwjbfyfli-6`

No consumer wiring into `paint_control_skill.py` is part of this pivot.

### PRE-GRAMMAR BRIDGE: Local-Delta Reranker Update

This update records the next representation change after the blocked-only loader. It is still not solve evidence. The local-delta loader is a PRE-GRAMMAR BRIDGE from discrepancy-derived exact rules; it is not Stage C/D-falsified and is not the promoted mechanic ledger.

Committed in `fa0dd747 Use Crucible local delta predictions in paint skill`:

- `CrucibleModel` derives PRE-GRAMMAR BRIDGE exact 5x5-neighborhood local-delta rules from the discrepancy corpus when all matching examples have one observed delta and support is at least eight.
- `CrucibleModel.predict()` checks those local-delta rules before forward-cell blocked rules.
- `paint_control_skill.py` demotes predicted moves that are blocked or strictly away from the active goal, including moves along an axis whose goal delta is already aligned.
- `paint_control_skill_regression.py` covers predicted-away demotion and aligned-axis demotion.

Verification:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/paint_control_skill_regression.py` passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/dev_check.py` passed.
- `PYTHON=python3 bash codexlab/sp80_integration_smoke.sh` passed.

Controlled level-2 evidence after local-delta reranking:

- Run: `codexlab/runs/_system_run/paint_control_skill_level02_crucible_local_delta_same_axis_20260612T_local`
- Category: controlled run.
- Angel verifier: passed.
- Action count: `40`.
- `levels_completed_delta_total: 0`.
- Feedback counts: `positive_goal_distance_reduced: 23`, `hard_negative_stationary_move: 5`, `negative_goal_distance_increased: 5`, `neutral_no_goal_progress: 7`.

Interpretation:

- Harmful move labels improved versus the blocked-only run (`negative_goal_distance_increased` from `7` to `5`, stationary from `6` to `5`), and the action basin changed.
- This still does not solve level 2.
- The current blocker is selection authority after motion demotion: when the forward model correctly demotes the bad trap moves, fallback chooses broad `select_object` actions that can switch to unauthoritative or no-op sources.
- The next representation-level change should mine and gate `select_object` outcomes instead of adding another motion fallback tweak.

### Selection and Activation Outcome Addendum

This addendum records the follow-up controlled probes after the local-delta reranker. These are negative evidence, not solve evidence.

Controlled run artifacts:

- `codexlab/runs/_system_run/paint_control_skill_level02_selection_outcome_adjacent_activation_20260612T_local`
  - Status: `nominal`.
  - `levels_completed_delta_total: 0`.
  - Feedback counts: `positive_goal_distance_reduced: 25`, `hard_negative_stationary_move: 5`, `negative_goal_distance_increased: 6`, `neutral_no_goal_progress: 4`.
  - Interpretation: one-cell adjacent activation can fire, but it does not solve L2 and can move control into a worse source/edge state. Do not reintroduce adjacent activation tolerance as a standalone rule.
- `codexlab/runs/_system_run/paint_control_skill_level02_selection_outcome_geometry_20260612T_local`
  - Status: `failed`.
  - `levels_completed_delta_total: 0`.
  - Feedback counts: `positive_goal_distance_reduced: 10`, `hard_negative_stationary_move: 4`, `negative_goal_distance_increased: 3`, `fail_closed_fallback_stationary_move: 1`.
  - Interpretation: a stricter geometry-only selection guard can fail closed early at the trap. Do not reintroduce it without a learned or verifier-backed source/edge outcome contract.

Next representation target:

- Mine `select_object` and `activate_flow` transitions as object/edge outcome rows, not move-delta rows.
- For `select_object`, model selected-source change, source identity, click target, PortBand key, and subsequent target-distance outcome.
- For `activate_flow`, model focus edge, target distance, locked source key, post-activation source change, and level delta.
- Promote only falsifiable hypotheses with all-false truth flags and explicit boundary text.
- Integrate with `paint_control_skill` as fail-open, demotion-only guidance first. The model may reject or demote harmful source/edge actions; it must not claim control truth or force activation.
- Add regressions for missing ledger, no-opinion, harmful-selection demotion, harmful-activation demotion, and byte-identical behavior when the outcome model is unavailable.

### Outcome History Guard Update

Committed in `0fb0385c Add paint skill outcome history guard`:

- Added default outcome ledger `codexlab/datasets/sp80_paint_skill_outcomes_v0/ledger.json`.
- Added fail-open harmful-history checks for `select_object` and `activate_flow`.
- Added ranked fallback reasons `selection_outcome_harmful_history` and `activation_outcome_harmful_history`.
- Added regression coverage for rejecting harmful PortBand-source selection history.

Verification:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/paint_control_skill_regression.py` passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/dev_check.py` passed.
- `PYTHON=python3 bash codexlab/sp80_integration_smoke.sh` passed.

Controlled level-2 evidence after the committed outcome-history guard:

- Run: `codexlab/runs/_system_run/paint_control_skill_level02_outcome_history_20260612T_local`
- Category: controlled run.
- Angel verifier: passed.
- Episode verifier: passed.
- Action count: `40`.
- `levels_completed_delta_total: 0`.
- Feedback counts: `positive_goal_distance_reduced: 27`, `hard_negative_stationary_move: 8`, `negative_goal_distance_increased: 5`.

Interpretation:

- The guard changed the basin and improved positive goal-distance reductions versus the local-delta run, but it still did not solve level 2.
- The remaining blocker is not another source blacklist. Expanding the blacklist recreated an early no-safe-candidate basin in the v2 scout.
- The next representation-level change should expose fallback exhaustion and source/edge planning state, then move into Crucible rule induction rather than adding more harmful-source constants.

### Budget-Aware Selection Scout Addendum

This addendum records the next local scout after the outcome-history guard. It is negative evidence and not solve evidence.

Local source changes in this scout:

- The SP80 harness adds an `episode_budget` block to each brain request with action index, maximum actions, and remaining action budget.
- Brain-call requests snapshot `action_memory` at decision time instead of holding a live list reference.
- `paint_control_skill.py` estimates selection step cost and can mark broad source selection as `selection_budget_infeasible`.
- Target-scoped packet filtering preserves a zero-value PortBand lineup context row when the explicit target filter removes all PortBand action rows.
- Feedback export labels `selection_budget_infeasible` as a fail-closed reason.

Controlled run artifacts:

- `codexlab/runs/_system_run/paint_control_skill_level02_budget_selection_v2_20260612T_local`
  - Status: `nominal`.
  - Angel verifier: passed.
  - Episode verifier: passed.
  - Action count: `40`.
  - `levels_completed_delta_total: 0`.
  - Feedback counts: `positive_goal_distance_reduced: 24`, `hard_negative_stationary_move: 7`, `negative_goal_distance_increased: 6`, `neutral_no_goal_progress: 3`.
  - Interpretation: budget-aware selection changed the basin by skipping the late broad `c0008` selection pattern, but metrics worsened versus the previous guarded baseline and the level still did not solve.
- `codexlab/runs/_system_run/paint_control_skill_level02_budget_outcome_v3_20260612T_local`
  - Status: `failed`.
  - Failure mode: `brain_error` after ranked fallback exhausted.
  - `levels_completed_delta_total: 0`.
  - Feedback counts: `positive_goal_distance_reduced: 10`, `hard_negative_stationary_move: 4`, `negative_goal_distance_increased: 3`, `fail_closed_ranked_fallback_exhausted: 1`.
  - Interpretation: expanding harmful outcome history around the new basin recreated the known no-safe-candidate failure. Do not expand source blacklists without a positive source/edge outcome model.

Current boundary:

- Budget feasibility is useful provenance, but it is not sufficient control evidence.
- Negative outcome blacklists can prevent known bad actions, but they do not identify the positive `select_object` or `activate_flow` plan.
- The missing representation is a falsifiable source/edge outcome model keyed by source component, selected coordinate, focus edge, target distance, and expected post-activation flow or level delta.
- The next valid work item is Crucible rule induction or an explicit source/edge forward model, not another same-class fallback penalty.

### Activation Authority Scout Addendum

This addendum records the follow-up activation-authority scouts after budget-aware source selection. These are negative controlled-run results and not solve evidence.

Local source changes in these scouts:

- Generic harmful `activate_flow` outcome rules now apply even when the active selected slot has a PortBand source key.
- Distance-one adjacent activation no longer becomes action authority from saturation evidence alone; saturation rows remain audit evidence, but activation now requires exact alignment unless a future source/edge model promotes it.

Controlled run artifacts:

- `codexlab/runs/_system_run/paint_control_skill_level02_unscoped_activation_guard_20260612T_local`
  - Status: `failed`.
  - Episode status: `budget_exhausted`.
  - Action count: `40`.
  - `levels_completed_delta_total: 0`.
  - Feedback counts: `positive_goal_distance_reduced: 21`, `hard_negative_stationary_move: 7`, `negative_goal_distance_increased: 6`, `neutral_no_goal_progress: 6`.
  - Interpretation: blocking the known harmful `[63,52]` activation displaced the failure to another distance-one activation at `[62,52]`, which moved the selected object to `[62,55]` and increased target distance.
- `codexlab/runs/_system_run/paint_control_skill_level02_exact_activation_only_20260612T_local`
  - Status: `failed`.
  - Episode status: `budget_exhausted`.
  - Action count: `40`.
  - `levels_completed_delta_total: 0`.
  - Feedback counts: `positive_goal_distance_reduced: 20`, `hard_negative_stationary_move: 7`, `negative_goal_distance_increased: 7`, `neutral_no_goal_progress: 6`.
  - Interpretation: removing distance-one activation as authority prevented the harmful activation class, but did not expose a positive capture plan. The controller cycled between `[62,52]` and `[63,52]`, then used late source selection without solving.

Stale-loop boundary:

- Controlled feedback exports currently show no positive `activate_flow` examples for level 2. Activation rows are neutral or harmful, so broad adjacent activation is not justified by the evidence.
- The failure basin is no longer "one missing blacklist." The missing representation is positive source/edge capture prediction: which selected source, focus edge, selected coordinate, and target port configuration should produce post-activation flow toward the bucket.
- Stop same-class reruns until Crucible induction or an explicit source/edge forward model can emit positive, falsifiable capture hypotheses. More fallback penalties, coordinate blacklists, or budget thresholds remain stale-loop work.

### Source/Edge Feedback Export Addendum

The feedback exporter now emits source/edge activation context in every row and a `source_edge_outcome_summary` block in `paint_control_skill_feedback_summary.json`.

Latest refreshed summaries:

- `codexlab/runs/_system_run/paint_control_skill_level02_exact_activation_only_20260612T_local/paint_control_skill_feedback/paint_control_skill_feedback_summary.json`
  - Activation candidates: `29`.
  - Executed activation rows: `0`.
  - Positive level activations: `0`.
  - Positive distance activations: `0`.
  - Fail-closed activation candidates: `29`.
  - Dominant buckets: `source_component:c0005`, `focus_edge=0`, target distances `1..3`, selected coordinates on the `[53..63,52]` approach line.
- `codexlab/runs/_system_run/paint_control_skill_level02_unscoped_activation_guard_20260612T_local/paint_control_skill_feedback/paint_control_skill_feedback_summary.json`
  - Activation candidates: `29`.
  - Executed activation rows: `1`.
  - Positive level activations: `0`.
  - Positive distance activations: `0`.
  - Negative distance activations: `1`.
  - Fail-closed activation candidates: `28`.

Interpretation:

- The controller can now expose which activation candidates it considered and why they failed closed.
- The current run evidence still contains no positive source/edge capture target for level 2.
- The next representation step should mine or probe for positive capture candidates, not relax distance-one activation or add more no-op penalties.

### Selection Outcome Feedback Export Addendum

The feedback exporter now emits selection identity and selection outcome context for every row:

- `selection_identity`
- `selection_outcome_context`
- `selection_outcome_summary`

These fields preserve the controller-side selected-source identity that was previously only visible inside `brain_calls.jsonl`:

- native action id and object slot
- PortBand task key
- source component ids
- source object id
- click target
- selected object name/shape before and after
- selected display coordinate before and after

Latest refreshed summary:

- `codexlab/runs/_system_run/paint_control_skill_level02_exact_activation_only_20260612T_local/paint_control_skill_feedback/paint_control_skill_feedback_summary.json`
  - Selection rows: `2`.
  - Positive level selections: `0`.
  - Positive distance selections: `0`.
  - Negative distance selections: `2`.
  - Repeated harmful bucket: `c0005_v08_00`, native action `12`, click `[16,22]`, before `[63,53]`, after `[53,53]`.
  - Late unknown-source bucket: native action `6`, click `[50,30]`, before `[62,53]`, after `[62,55]`.

Representative identity-aware mining over the latest run, best guarded baseline, and recent scouts found:

- Executed activations in that slice: `2`, both negative.
- Selection rows in that slice: `7`.
- Repeatable harmful selection bucket: `c0005_v08_00`, native action `12`, click `[16,22]`, before `[63,53]`, after `[53,53]`, negative in three recent scouts.
- One positive selection bucket in the slice: `c0007_v08_01`, native action `6`, click `[50,30]`, before `[53,52]`, after `[62,55]`, positive distance only in the best guarded baseline.

Interpretation:

- Selection identity is now visible enough for a future source/outcome model to distinguish `c0005`, `c0007`, `c0008`, native action id, click target, and before/after selected-source state.
- This is still not a positive capture model. The evidence does not justify another blacklist-only control change or a new controlled level-2 drive from the same basin.
- The next control change should consume this identity as part of a falsifiable source/edge outcome model, or continue Crucible induction toward a promoted rule, before rerunning L2.

### Negative Scout: C0005 63,53 Outcome Guard

Controlled run:

- `codexlab/runs/_system_run/paint_control_skill_level02_c0005_6353_outcome_guard_20260612T_local`
  - Status: `failed`.
  - Episode status: `brain_error`.
  - Action count: `18`.
  - `levels_completed_delta_total: 0`.
  - Angel verifier: passed.
  - Feedback counts: `positive_goal_distance_reduced: 10`, `hard_negative_stationary_move: 4`, `negative_goal_distance_increased: 3`, `fail_closed_ranked_fallback_exhausted: 1`.

Scout change:

- Temporarily promoted the repeated harmful bucket `c0005_v08_00`, native action `12`, click `[16,22]`, before `[63,53]`, after `[53,53]`, as a harmful selection rule.

Observed result:

- The rule correctly rejected the repeated harmful `c0005` source selection.
- The controller then selected `move_right` at `[63,53]`, which was stationary.
- One action later it returned to `[63,52]` and exhausted ranked fallback: move candidates were blocked, away, stationary, or oscillatory, and source selections were rejected by existing outcome/budget guards.

Interpretation:

- This recreates the known no-safe-candidate basin from earlier `budget_outcome_v3` style scouts.
- The temporary `c0005_v08_00 @ [63,53]` promoted rule was removed from the active outcome ledger after the controlled run because it worsened the basin and did not expose a positive source/edge capture plan.
- Do not retry blacklist-only selection expansion from this basin. The missing representation is still positive source/edge capture prediction or a discriminating probe that can discover one.

### Next Instrumentation: Fallback Exhaustion Candidate Matrix

The immediate next implementation step is not another controlled level-2 drive. The latest negative scout proves that blacklist-only selection expansion can remove a known bad action and still leave the controller with no safe candidate. The next artifact should make that no-safe-candidate state machine-checkable.

Add a `fallback_exhaustion_summary` block to `paint_control_skill_feedback_summary.json`.

Rows:

- include only decisions whose label or raw fallback status is `fail_closed_ranked_fallback_exhausted` / `ranked_fallback_exhausted`
- parse the raw fallback rejected-candidate list from `brain_calls.jsonl`
- preserve current selected display coordinate, selected object name/shape, target coordinate, and original fail reason
- record each rejected candidate's primitive, native action id, object slot, PortBand source key when present, candidate score, rejection reason, matched outcome rule id when present, forward-model demotion status, activation readiness status, and selection-budget status

Summary groups:

- current coordinate and selected object
- primitive
- rejection reason
- PortBand source key or `unknown_source`
- matched outcome rule id or `no_rule`
- forward-model status

Required counters:

- exhausted decision count
- total rejected candidates
- rejected candidates by primitive
- rejected candidates by reason
- viable candidate count after each filter family
- zero-viable terminal states

HTML review surface:

- add a compact "Fallback Exhaustion" table to `paint_control_skill_feedback/index.html`
- show the terminal selected coordinate first
- show candidate rows grouped by primitive and rejection reason
- link back to the source action index and brain-call row

Regression:

- synthetic exhausted fallback row produces the expected grouped summary
- non-exhausted fallback rows are ignored
- missing rejected-candidate list fails closed into an `unavailable_rejected_candidates` bucket instead of silently reporting zero candidates
- summary JSON and HTML remain valid when there are no exhausted rows

Boundary:

- This audit is not a control change and does not justify a new L2 drive by itself.
- A new controlled L2 run is valid only after the candidate matrix is consumed by a representation-level change: a positive source/edge outcome model, a Crucible promoted rule, or a bounded discriminating probe.

### Fallback Exhaustion and Source-Diversity Probe Addendum

This addendum records the next local instrumentation/control pass after the fallback-exhaustion candidate matrix. It is negative controlled-run evidence and not solve evidence.

Local source changes in this pass:

- `paint_control_skill_feedback_export.py` now emits `fallback_exhaustion_summary`, rejected fallback candidates, and viable fallback candidates in the feedback JSON and HTML review surface.
- `paint_control_skill.py` now prefers distinct PortBand source keys when mapping live request objects into paint-skill slots.
- When ranked fallback is exhausted, `paint_control_skill.py` can issue a bounded `paint_control_skill_source_diversity_probe` selection from lower-ranked PortBand rows if the candidate passes source-selection authority, selection-lock feedback, selection-budget feedback, and outcome-history feedback.
- Regressions cover fallback-exhaustion summary generation, distinct live PortBand source slots, and the source-diversity probe after ranked fallback exhaustion.

Verified local gates for the source-diversity pass:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/paint_control_skill_regression.py` passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/dev_check.py` passed.
- `PYTHON=python3 bash codexlab/sp80_integration_smoke.sh` passed.

Controlled run artifact:

- `codexlab/runs/_system_run/paint_control_skill_level02_source_diversity_probe_20260612T_local`
  - Category: controlled run.
  - Status: `nominal`.
  - Episode status: `budget_exhausted`.
  - Action count: `40`.
  - `levels_completed_delta_total: 0`.
  - Angel verifier: passed with `--strict-png-ledger --require-gameplay-visual-contract`.
  - Feedback counts: `positive_goal_distance_reduced: 24`, `hard_negative_stationary_move: 9`, `negative_goal_distance_increased: 7`.
  - Fallback exhaustion summary: `exhausted_decision_count: 0`; the source-diversity probe escaped the prior 18-action `ranked_fallback_exhausted` / `brain_error` basin.
  - Selection outcome summary: one source-diversity selection bucket, `c0008_v08_02|native_-1|before_63,52|after_55,56`, with one `negative_goal_distance_increased` label and no positive level or distance outcome.
  - Source/edge summary: `activation_row_count: 0`, `activation_candidate_count: 16`, `activation_fail_closed_count: 16`; active `source_component:c0008` activation candidates all fail closed with `activation_readiness_target_misaligned` at target distance `1`.
  - Final state: selected display coordinate `[63,53]`, selected object `plzwjbfyfli-6`, selected shape `[6,1]`, state `NOT_FINISHED`, `steps_remaining: 60`.

Representative controlled packet:

- Action `17` selected `paint_control_skill_source_diversity_probe`.
- Before selected display coordinate: `[63,52]`.
- After selected display coordinate: `[55,56]`.
- PortBand task key: `source_component:c0008`.
- Source component: `c0008_v08_02`.
- Click target: `[20,34]`.
- Goal-distance delta: `-12.0`, so the selection increased distance and is negative feedback.

Interpretation:

- Source-slot diversity and source-diversity probing are real representation changes because they exposed a different source/edge basin instead of ending at ranked-fallback exhaustion.
- The run still did not solve level 2 and did not improve the controlled metrics enough to commit/push as a working control improvement.
- The new failure basin is source-edge capture under active `source_component:c0008`: all distance-one activation candidates failed closed, and there is still no positive activation/capture model.
- Do not continue by adding another broad source blacklist. A blacklist-only response would likely return to the no-safe-candidate basin.

Next valid representation step:

- Add a source-scoped activation-discrimination probe or a positive source/edge outcome model before any further level-2 drive.
- The probe must be one-shot, source-scoped, and audit-labeled; it must not reintroduce broad adjacent activation tolerance.
- It should require active source identity, no matching harmful activation outcome rule, no prior bad activation for the same source/coordinate, and repeated recorded fail-closed activation candidates for that source.
- Raw decisions must mark the action as a discrimination probe with all-false truth boundaries.

### Source-Edge Activation Probe and Stale-Loop Addendum

This addendum records the source-scoped activation-discrimination pass after the c0008 source-diversity basin. These are controlled-run results and not solve evidence.

Local source changes in this pass:

- `paint_control_skill.py` adds `source_edge_activation_probe_feedback()`.
- Distance-one `activate_flow` can be upgraded from fail-closed to `source_edge_discrimination_probe` only when:
  - the selected source has a PortBand source key
  - the selected source was reached through `paint_control_skill_source_diversity_probe`
  - adjacent saturation evidence exists at the current source/coordinate
  - no harmful activation outcome rule matches
  - no prior bad activation at the same selected display coordinate is in the recent action memory
- Ranked fallback gives a bonus only to activation candidates that carry this explicit probe readiness.
- The outcome ledger adds a source-scoped harmful activation rule for `source_component:c0008`, focus edge `0`, selected coordinate `[61,52]`, target distance `1`, from the first probe run.

Regressions added:

- source-edge activation probe can emit one distance-one activation after source-diversity selection
- repeat activation after a bad activation row fails closed
- the c0008 harmful activation outcome rule blocks only the matching coordinate and leaves `[62,52]` eligible

Verified local gates:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/paint_control_skill_regression.py` passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/dev_check.py` passed.
- `PYTHON=python3 bash codexlab/sp80_integration_smoke.sh` passed.

Controlled run: source-edge activation probe:

- `codexlab/runs/_system_run/paint_control_skill_level02_source_edge_activation_probe_20260612T_local`
  - Category: controlled run.
  - Status: `nominal`.
  - Episode status: `budget_exhausted`.
  - Action count: `40`.
  - `levels_completed_delta_total: 0`.
  - Angel verifier: passed.
  - Episode verifier: passed.
  - Feedback counts: `positive_goal_distance_reduced: 25`, `hard_negative_stationary_move: 8`, `negative_goal_distance_increased: 7`.
  - Activation rows: `2`.
  - c0008 `[61,52]` activation at action `29`: negative, distance delta `-1.5`, selected object changed from `plzwjbfyfli-6` to `plzwjbfyfli-5`.
  - c0008 `[62,52]` activation at action `38`: positive distance only, distance delta `0.5`, no level completion.

Controlled run: source-edge activation ledger:

- `codexlab/runs/_system_run/paint_control_skill_level02_source_edge_activation_ledger_20260612T_local`
  - Category: controlled run.
  - Status: `nominal`.
  - Episode status: `budget_exhausted`.
  - Action count: `40`.
  - `levels_completed_delta_total: 0`.
  - Angel verifier: passed.
  - Episode verifier: passed.
  - Feedback counts: `positive_goal_distance_reduced: 25`, `hard_negative_stationary_move: 8`, `negative_goal_distance_increased: 7`.
  - Activation rows: `1`.
  - The c0008 `[61,52]` harmful ledger rule blocked the first probe opportunity.
  - c0008 `[62,52]` activation at action `31`: negative, distance delta `-2.5`, selected object changed from `plzwjbfyfli-6` to `plzwjbfyfli-5`.

Interpretation:

- The source-edge probe is a real representation change. It turns invisible fail-closed activation candidates into controlled transition evidence.
- It still does not solve level 2.
- Coordinate/source/edge/target-distance alone is not a sufficient activation outcome model: c0008 `[62,52]` produced positive distance in one controlled run and negative distance in the next.
- The missing representation is hidden activation state: selected object identity/shape, post-activation selected object identity, source component version, edge/port geometry, and whether activation changes the source object rather than pouring into the target bucket.

Stale-loop boundary:

- Do not launch another level-2 run in the same source-edge activation-probe class.
- The next valid work item is an activation transition model keyed by selected source object identity and post-activation selected-object transition, or a Crucible rule induction pass that can explain why the same c0008 coordinate produces different outcomes.
- Adding more coordinate-only harmful activation rules is stale-loop work and risks blacklisting the only visible capture probes without learning the capture mechanic.

## Unified Interface Contract

Package:

```text
codexlab/contra_head/crucible/
  __init__.py
  canon.py
  corpus.py
  forward_v0.py
  ledger.py
  grammar.py
  falsify.py
  model.py
```

Durable artifacts:

```text
codexlab/datasets/sp80_crucible_transitions_v0/
  corpus.jsonl
  manifest.json

codexlab/datasets/sp80_crucible_mechanics_v0/
  mechanic_ledger.json
  manifest.json

codexlab/runs/_crucible/crucible_<stage>_<UTCstamp>/
```

Public API in `crucible/model.py`, re-exported from `__init__.py`:

```python
CrucibleModel.load(ledger_path=DEFAULT) -> CrucibleModel | None
predict(before_grid, action_primitive, actor_coord, rotation_quarters=0) -> CruciblePrediction
disagreements(before_grid, actor_coord, rotation_quarters, reachable_actions) -> list[dict]
```

`CruciblePrediction` fields:

- `predicted_canon_delta`: `(dr, dc) | None`
- `predicted_blocked`: `bool`
- `effect_kind`
- `matched_rule_id | None`
- `confidence`
- `opinion`: `"prediction" | "no_opinion"`
- all truth flags false
- boundary: `mechanic_forward_model_is_hypothesis_ledger_composition_not_control_truth`

`disagreements()` rows:

- `disagreement_id`
- `action_primitive`
- `disagreeing_rule_ids` with at least two ids
- `rule_predictions`
- `posterior_entropy`

Coordinate convention:

- Canonical frame is display orientation.
- Coordinates are `[x,y]`; saved `.npy` grids index `grid[x,y]`.
- `move_up=(0,-1)`, `move_down=(0,1)`, `move_left=(-1,0)`, `move_right=(1,0)`.
- `contra_head/crucible/canon.py` owns `canonicalize_grid()`, `canonicalize_coord()`, and display-coordinate conversion for `rotation_quarters in {0,2}`.
- Unknown rotation is skipped with a reason.
- Per-row invariant: `canon_grid[actor] == 1`.

Stage A implementation note:

- The live SP80 harness currently emits `selected_coord` as `[x,y]` and the saved grid is indexable in that same order.
- The transition miner stores raw/display/canonical packet coordinates explicitly and uses the harness display-delta convention for the v0 expected-delta summary.
- If future phases require another coordinate convention, make that conversion only in `canon.py` and update the verifier and mismatch baseline together.

Verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/contra_verify_crucible.py --artifact corpus|ledger|rules|mechanics|run
```

## Phase 1: Stage A Transition Corpus Miner

Files:

- `codexlab/contra_head/crucible/canon.py`
- `codexlab/contra_head/crucible/corpus.py`
- `codexlab/crucible_mine_transitions.py`

Requirements:

- Walk `codexlab/runs/**/packets.jsonl`.
- Reuse walking/manifest idioms from `paint_control_skill_motion_effect_mine.py`.
- Extract schema `codexlab.crucible_transition.v1`.
- `transition_id = stable_id("crucible_transition", before_npy_sha, action_sig, after_npy_sha)`.
- Include action, before/after grid refs, actor coords raw/display/canon, selection_changed, observed delta/moved/blocked, expected_canon_delta, visual_delta, goal_progress, levels_completed_delta, provenance, and truth-boundary block.
- Re-hash `.npy` grids from disk. Mismatch skips and increments a count.
- Dedup key: `(before_npy_sha, action_sig, after_npy_sha)`.
- Manifest schema: `codexlab.crucible_transition_corpus.v1`.
- Manifest includes skip-reason histograms, primitive histograms, rotation histograms, per-level counts, source hashes, and baseline mismatch summary.
- Baseline mismatch summary should reproduce roughly the current wrong-direction and stationary bands.

Verifier `--artifact corpus`:

- recomputes ids
- sampled sha re-hash, with `--strict` checking all rows
- actor-color invariant
- dedup uniqueness
- truth flags all false
- forbidden-authority-key walk patterned after `contra_verify_ghost.py`

Gate:

- Miner runs over full `codexlab/runs/`.
- Manifest mismatch stats in `[0.15, 0.35]`.
- Verifier green.

## Phase 2: Stage B Discrepancy Ledger

Files:

- `codexlab/contra_head/crucible/forward_v0.py`
- `codexlab/contra_head/crucible/ledger.py`
- `codexlab/crucible_discrepancy_ledger.py`

Requirements:

- `predict_v0(primitive)` is naive translation: moves map to unit deltas; select and activate map to `(0,0)`.
- Version string: `naive_translation_v0`.
- Emit anomaly schema `codexlab.crucible_anomaly.v1`.
- Anomaly kinds: `blocked`, `redirect_left`, `redirect_right`, `reverse`, `slide`, `off_lattice`, `moved_when_static_expected`, `other`.
- Exclude `selection_changed` transitions from motion analysis but count them.
- Local context includes 5x5 canonical neighborhood with OOB `-1`, three-cell forward ray in issued direction, actor color, shape, and axis.
- Ledger artifact includes `anomalies.jsonl` and `ledger.json` with anomaly kind by primitive counts plus forward-cell color histogram per kind.

Verifier `--artifact ledger`:

- ids resolve
- `predict_v0` disagreement recomputes
- context shape checks

Gate:

- Ledger built.
- Redirect-trigger color histogram reviewed before Phase 3.

## Phase 3: Stage C Rule Grammar and Induction

Files:

- `codexlab/contra_head/crucible/grammar.py`
- `codexlab/crucible_induce_rules.py`

Grammar v0:

- Schema `codexlab.crucible_rule.v1`.
- `grammar_version: "crucible_grammar.v0"`.
- Selector: `per_primitive` or `direction_generic`.
- Condition: one or two atoms.
- Atom: `cell_color(offset, C)` where offset is one of `ahead1`, `ahead2`, `left1`, `right1`, `behind1`, `ahead_left1`, `ahead_right1`, and `C` is in the palette or `OOB`; plus `actor_axis(h|v)`.
- Effect: `blocked`, `forward`, `redirect_left`, `redirect_right`, `reverse`, `slide_until_blocked`.
- Enumeration: exhaustive single-atom rules plus data-driven two-atom conjunctions instantiated only from anomaly contexts they explain.
- `rule_id = stable_id("crucible_rule", selector, canonical_condition, effect)`.

Verifier `--artifact rules`:

- id recompute
- grammar-bound checks
- sampled `induced_from` anomalies satisfy condition and effect

## Phase 4: Stage D Replay Falsification, Mechanic Ledger, and Model

Files:

- `codexlab/contra_head/crucible/falsify.py`
- `codexlab/contra_head/crucible/model.py`
- `codexlab/crucible_falsify_rules.py`

Requirements:

- Copy Wilson lower-bound reliability math from `b77d0a43:codexlab/contra/ghost_mechanics.py` with attribution.
- For every candidate rule over every corpus move-transition, if condition fires, compare predicted vs observed and record support/contradiction.
- Store contradicting transition ids, capped at 20.
- `reliability = wilson_lower_bound(support, support+contradictions, z=1.0)`.
- Promotion:
  - contradictions `== 0`
  - support `>= 8`
  - reliability `>= 0.75`
  - support spans at least two distinct run dirs
- Redundancy pruning: more general subset wins.
- Exact conflict ties keep both rules conflicted.
- Mechanic ledger schema: `codexlab.crucible_mechanic_ledger.v1`.
- Store at `codexlab/datasets/sp80_crucible_mechanics_v0/`.
- Include per-rule accuracy, composed-model residual mismatch rate versus v0 baseline, and truth-boundary block.
- `CrucibleModel` predicts actor delta only; paint-flow grid effects are out of scope.

Verifier `--artifact mechanics`:

- recompute support and contradictions for sampled promoted rules
- Wilson recompute
- promotion invariant checks

Regression:

- `codexlab/crucible_regression.py` over two or three pinned run dirs.
- Baseline band.
- At least one blocked and one redirect candidate induced.
- All verifier modes green.
- Residual no worse than baseline.
- Register in `codexlab/dev_check.py` as CONTRA gate L1.

Keystone go/no-go:

- New `contra_loop/keystones/sp80_l1_surface_redirect_chain.py`.
- Clone `sp80_l1_gravity_flip.py` contract.
- `admission_status: "discovery_only"`.
- Applies `CrucibleModel.predict()` to all L1 move transitions in the win bank and reports exact actor-delta matches.
- If near zero matches, stop and iterate offline grammar v1 with connected-component/largest-object features before live wiring.

## Phase 5: Stage F1 Paint-Skill Predicted-Outcome Reranker

File:

- `codexlab/paint_control_skill.py`

Requirements:

- New init kwarg `forward_model_ledger_path`.
- Lazy load; missing ledger permanently fail-open.
- Between checkpoint inference and translation, query `predict()` for each move-kind native in `ranked_action_ids`.
- Demote to back, never delete, natives predicted blocked or strictly distance-increasing with confidence `>= 0.70`.
- If argmax is demoted, set fail reason `forward_model_predicted_blocked_or_away`.
- Add that reason to `RANKED_FALLBACK_REASONS`.
- No promotion in the skill.
- Add `forward_model_rerank` audit block with ledger sha, demoted ids, predictions, and status.

Regressions:

- demotion triggers fallback with new reason
- no-opinion is byte-identical to golden
- missing ledger is identical plus status `unavailable`

Gate:

- Standard fixed harness accuracy run with at least 80 actions.
- Report positive-progress, wrong-direction, and stationary rates from `paint_control_skill_feedback_export.py`.

## Phase 6: Stage F2 Bounded Scorer Modifier

File:

- `codexlab/sp80_player_harness.py`

Requirements:

- Add `crucible_forward_model_modifier`.
- Penalty no more than `0.20` for moves predicted blocked/away.
- Promotion no more than `0.10` only if row lacks `surface_orthogonal_flow_redirect_bonus` and `relay_lock_hold_bonus`.
- `MAX_CRUCIBLE_BONUS = 0.25` hard cap.
- Never touches `activate_flow` rows.
- Fail-open without ledger.
- Signals record ledger sha and rule ids.

Verifier `--artifact run`:

- scan run `packets.jsonl`
- all Crucible signal magnitudes `<= 0.25`
- no co-occurrence with committed-rule bonuses
- truth flags all false on prediction blocks

## Phase 7: Stage E Discriminating Probes in ExploreBand

Files:

- `codexlab/contra_head/exploreband.py`
- `codexlab/sp80_player_harness.py`
- `codexlab/contra_head/controller_head.py`
- `codexlab/contra_verify_exploreband.py`
- `codexlab/contra_exploreband_mechanic_probe_regression.py`

Requirements:

- New candidate kind `mechanic_discrimination_probe`.
- All existing authority and admitted-false fields preserved.
- Proposed action primitive is one of four moves and probe class is `single_step_direction_probe`.
- `mechanic_discrimination` includes disagreement id, at least two disagreeing rule ids, rule predictions, posterior entropy, and ledger ref.
- EIG formula: `clamp01(0.70*H(p)/log(k) + 0.16*mean(reliability) + 0.14*(1-danger))`.
- Legacy `proposed_observation_target` remains populated.
- `inject_mechanic_discrimination_probes()` appends, re-sorts by EIG, re-truncates, recomputes ranks/flags, cap two per frame.
- Called in controller head after budget resolution.
- Fed by `disagreements()`, fail-open empty.
- Consumer branch in `portband_signal_candidates`.
- If actor unselected, select object first.
- Oscillation gate and emit-but-mark zeroing stay untouched.
- Evidence refs include probe and disagreement ids.

Verifier and regression:

- schema checks
- EIG monotone in entropy
- cap
- fail-open
- authority statuses unchanged

## Phase 8: Stage G Loop Harness Wiring and Admission Attempt

Files:

- `codexlab/contra_loop/loop.py`
- `codexlab/contra_loop/crucible_redirect_planner.py`
- `codexlab/contra_loop/solver_registry.json`

Requirements:

- Replace `DISCOVER-GAP` stub.
- `gap_surface.json` v2 includes gap finder results, mechanic ledger summary, discrepancy ledger ref, unresolved disagreement count, and fail-open loading.
- Boundary string preserved.
- `crucible_redirect_planner` implements `SolverProtocol`.
- `evidence_tier=1`.
- Deterministic bounded beam search in imagination over move sequences through `predict()`.
- Depth `<=16`, nodes `<=5000`.
- Emits `ScoredCandidate` for first plan step only.
- Candidate kind: `crucible_redirect_plan_step`.
- Full plan in evidence refs.
- Empty list on failure.
- Register in default registry and inline producer in `build_contra_recommendation_packet`.
- Ceiling-capped like Oracle/Ghost rows.
- Add kind to `RELAY_LOCK_KINDS`.
- F1 reranker prefers, but does not force, the plan next step among non-demoted natives.
- Extend loop backend choices with `paint_control_skill` if missing.

Dry-run then real command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/contra.py loop iterate --target sp80:1 --mechanic crucible_redirect_planner --backend paint_control_skill --max-actions 40
```

Admission predicate, five-layer gate, `sp80_verify_episode_run.py`, and solver registration remain untouched.

## End-to-End Verification

1. Each phase requires its verifier mode plus regression green.
2. `python codexlab/contra.py check` and `codexlab/dev_check.py` stay green before the next phase.
3. Keystone go/no-go after Phase 4.
4. Accuracy benchmark after Phases 5 and 6 with fixed config and at least 80 actions.
5. Final admission attempt through `contra loop iterate`.
6. Only `levels_completed_delta > 0` in the gated run artifact counts as solving L1.

## Risks

- `rotation_quarters=0` canonicalization is less inspected than `2`; actor-color invariant skips bad rows.
- Color-only grammar may miss object-class keyed redirects; grammar v1 adds connected-component features if residual anomalies remain high.
- Hidden paint/flow state divergence should be killed by zero-contradiction promotion.
- Authority and truth boundaries require all prediction packets to carry false truth flags and the boundary literal.
- Paint-skill integration must remain demotion-only and fail-open.
- Bonus stacking is guarded with no-stack checks, caps, and run verifier assertions.

## Pinned Addendum: Detailed Mechanic-Induction Contract

This addendum pins the exact contract for the next implementation pass. It supersedes loose phase wording above only where it is more specific.

### Source Evidence Snapshot

- Corpus target: 100 `packets.jsonl` files, about 3,946 packets, and about 1,590 usable transitions with both `.npy` frames present.
- Grid contract: `64x64 int16`, palette `{1,4,6,8,9,11,12,14}`.
- Actor convention: `before_state.selected_coord = [x,y]`; saved `.npy` grids index `grid[x,y]`; actor color is `1`. `selected_display_coord` equals the canonical coordinate from `contra_head/crucible/canon.py`, e.g. raw `[1,11]` under `rotation_quarters=2` on `64x64` canonicalizes to `[62,52]`.
- Rotation convention: `rotation_quarters in {0,2}`. Unknown rotation skips with reason.
- Existing Ghost mechanic mapper is not wired. Copy Wilson lower-bound reliability from git `b77d0a43:codexlab/contra/ghost_mechanics.py` with attribution.
- `contra_loop/loop.py` Step 2 `DISCOVER-GAP` is a stub and must become a real gap surface.
- `ExploreBand` has no specific-action probe interface today; any probe addition must preserve oracle selection to executor validation.

### Package and Artifact Contract

Library package:

```text
codexlab/contra_head/crucible/
  __init__.py
  canon.py
  corpus.py
  forward_v0.py
  ledger.py
  grammar.py
  falsify.py
  model.py
```

Durable corpus:

```text
codexlab/datasets/sp80_crucible_transitions_v0/
  corpus.jsonl
  manifest.json
```

Durable mechanic ledger:

```text
codexlab/datasets/sp80_crucible_mechanics_v0/
  mechanic_ledger.json
  manifest.json
```

Per-stage run artifacts:

```text
codexlab/runs/_crucible/crucible_<stage>_<UTCstamp>/
```

Every live consumer records `ledger_sha256`.

### Model API

`codexlab/contra_head/crucible/model.py`, re-exported from `__init__.py`:

```python
CrucibleModel.load(ledger_path=DEFAULT) -> CrucibleModel | None
```

`None` means fail open for live consumers.

```python
predict(before_grid, action_primitive, actor_coord, rotation_quarters=0) -> CruciblePrediction
```

Prediction fields:

- `predicted_canon_delta: tuple[int, int] | None`
- `predicted_blocked: bool`
- `effect_kind: str`
- `matched_rule_id: str | None`
- `confidence: float`
- `opinion: "prediction" | "no_opinion"`
- all truth flags false
- boundary literal: `mechanic_forward_model_is_hypothesis_ledger_composition_not_control_truth`

```python
disagreements(before_grid, actor_coord, rotation_quarters, reachable_actions) -> list[dict]
```

Disagreement rows contain `disagreement_id`, `action_primitive`, `disagreeing_rule_ids` with at least two ids, `rule_predictions`, and `posterior_entropy`.

Canonical frame is display orientation, `[x,y]`:

- `move_up = (0,-1)`
- `move_down = (0,1)`
- `move_left = (-1,0)`
- `move_right = (1,0)`

Per-row invariant:

```text
canon_grid[actor] == 1
```

### Unified Verifier Contract

Single verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/contra_verify_crucible.py --artifact corpus|ledger|rules|mechanics|run
```

Required verifier behavior:

- `corpus`: id recompute, sampled SHA rehash, actor-color invariant, dedup uniqueness, all-false truth flags, forbidden-authority-key walk.
- `ledger`: anomaly ids resolve, `predict_v0` disagreements recompute, local-context shapes are valid.
- `rules`: rule ids recompute, grammar bounds hold, sampled `induced_from` anomalies satisfy condition and effect.
- `mechanics`: support and contradiction counts recompute for sampled promoted rules, Wilson reliability recomputes, promotion invariants hold.
- `run`: Crucible signal magnitudes are capped, no bonus stacking with committed-rule bonuses, prediction truth flags remain false.

### Stage A: Transition Miner Gate

CLI:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/crucible_mine_transitions.py
```

Rows use schema `codexlab.crucible_transition.v1`.

`transition_id`:

```text
_stable_id("crucible_transition", before_npy_sha, action_sig, after_npy_sha)
```

Dedup key:

```text
(before_npy_sha, action_sig, after_npy_sha)
```

Manifest schema: `codexlab.crucible_transition_corpus.v1`.

Gate:

- miner runs over full `codexlab/runs/`
- baseline mismatch stats land in `[0.15,0.35]`
- corpus verifier green

### Stage B: Discrepancy Ledger Gate

CLI:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/crucible_discrepancy_ledger.py
```

`predict_v0` version string:

```text
naive_translation_v0
```

Anomaly schema: `codexlab.crucible_anomaly.v1`.

Allowed `anomaly_kind`:

- `blocked`
- `redirect_left`
- `redirect_right`
- `reverse`
- `slide`
- `off_lattice`
- `moved_when_static_expected`
- `other`

Local context includes a 5x5 canonical neighborhood with OOB `-1`, a 3-cell forward ray, actor color, actor shape, and actor axis.

Gate:

- `anomalies.jsonl` and `ledger.json` exist
- redirect-trigger color histogram is reviewed before Stage C
- ledger verifier green

### Stage C: Rule Grammar Gate

CLI:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/crucible_induce_rules.py
```

Rule schema: `codexlab.crucible_rule.v1`.

Grammar version:

```text
crucible_grammar.v0
```

Grammar v0:

- selectors: `per_primitive` or `direction_generic`
- one or two condition atoms
- condition atoms: `cell_color(offset, C)` and `actor_axis(h|v)`
- offsets: `ahead1`, `ahead2`, `left1`, `right1`, `behind1`, `ahead_left1`, `ahead_right1`
- effects: `blocked`, `forward`, `redirect_left`, `redirect_right`, `reverse`, `slide_until_blocked`
- exhaustive single-atom enumeration plus data-driven two-atom conjunctions from explaining anomaly contexts

`rule_id`:

```text
_stable_id("crucible_rule", selector, canonical_condition, effect)
```

Gate:

- rules verifier green

### Stage D: Falsification and Mechanic Ledger Gate

CLI:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/crucible_falsify_rules.py
```

For each candidate rule over each corpus move-transition:

- if condition fires, compare predicted against observed
- record support
- record contradictions
- store contradicting `transition_id`s capped at 20

Reliability:

```text
wilson_lower_bound(support, support + contradictions, z=1.0)
```

Promotion:

- `contradictions == 0`
- `support >= 8`
- `reliability >= 0.75`
- support spans at least two distinct run directories

Mechanic ledger schema:

```text
codexlab.crucible_mechanic_ledger.v1
```

The composed model reports residual mismatch versus the v0 baseline. Target direction is about `27.5%` residual toward single digits; lower residual is not a solve claim.

Regression:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/crucible_regression.py
```

Gate:

- at least one blocked candidate induced
- at least one redirect candidate induced
- all verifier modes green
- residual not worse than baseline
- registered in `codexlab/dev_check.py`

### Keystone Go/No-Go

Add:

```text
codexlab/contra_loop/keystones/sp80_l1_surface_redirect_chain.py
```

Contract:

- clone `sp80_l1_gravity_flip.py` shape
- `HYPOTHESIS` dict
- pinned win-bank SHA
- `--dry-run`
- `--json-only`
- output schema `codexlab.contra_loop.keystone_result.v1`
- `admission_status: "discovery_only"`
- report `exact_actor_delta_matches N/M`

If matches are near `0/N`, stop before live wiring and iterate the offline grammar, starting with connected-component and largest-object features.

### Live Wiring Gates

Paint-skill F1:

- add `forward_model_ledger_path`
- lazy load ledger
- missing ledger fails open
- demote, never delete, move-kind natives predicted blocked or strictly distance-increasing with confidence at least `0.70`
- if argmax is demoted, use fail reason `forward_model_predicted_blocked_or_away`
- add that reason to `RANKED_FALLBACK_REASONS`
- no promotion inside the skill
- raw decision includes `forward_model_rerank`

Paint-skill regressions:

- demotion triggers fallback with the new reason
- no-opinion is byte-identical to golden
- missing ledger is identical except `status: "unavailable"`

Scorer F2:

- `crucible_forward_model_modifier`
- penalty no more than `0.20`
- promotion no more than `0.10`
- `MAX_CRUCIBLE_BONUS = 0.25`
- no bonus stacking with `surface_orthogonal_flow_redirect_bonus` or `relay_lock_hold_bonus`
- never touch `activate_flow`

ExploreBand probes:

- candidate kind `mechanic_discrimination_probe`
- probe primitive class `single_step_direction_probe`
- cap two probes per frame
- EIG formula remains `clamp01(0.70*H(p)/log(k) + 0.16*mean(reliability) + 0.14*(1-danger))`
- executed probe ids flow back through ordinary `packets.jsonl`

Loop harness:

- `gap_surface.json` v2 includes gap-finder results, mechanic-ledger summary, discrepancy-ledger ref, unresolved disagreement count, and the boundary string.
- `crucible_redirect_planner` implements `SolverProtocol`.
- imagination beam search is deterministic, depth at most 16, and at most 5,000 nodes.
- candidate kind is `crucible_redirect_plan_step`.
- F1 may prefer the planner's next step among non-demoted natives, but must not force it.

Dry run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/contra.py loop iterate --target sp80:1 --mechanic crucible_redirect_planner --backend paint_control_skill --max-actions 40 --dry-run
```

Real run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/contra.py loop iterate --target sp80:1 --mechanic crucible_redirect_planner --backend paint_control_skill --max-actions 40
```

Unchanged gates:

- admission predicate
- five-layer gate
- `sp80_verify_episode_run.py`
- solver registration checks

Rejections must be preserved in `rejections.jsonl` as honest negatives.

### Final Rule

Every phase requires its verifier and regression before the next phase. The final solve claim requires a gated run artifact with `levels_completed_delta > 0`. No progress-rate, residual-rate, model-match, or probe-success metric is a solve claim.
