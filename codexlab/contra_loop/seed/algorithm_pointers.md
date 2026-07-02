# CONTRA Seed Solvers — Algorithm Pointers (mined from verified wins)

These are **named, threshold-precise algorithmic procedures** reconstructed from CONTRA's
own *successful* SP80 playthroughs. They are the seed of the `contra_loop` solver library.

**Cardinal rule honored:** every VERIFIED pointer below traces to a run_dir whose
`result.json` has `levels_completed_delta == 1` and `status == "completed"`, with a clean
16-packet `packets.jsonl` decision trail. Anything not traceable to `levels_completed_delta > 0`
is parked under WANTED MECHANICS and excluded from seed solvers.

**Reproducibility proof:** the SP80-L0 win is **bit-identical across 5 independent run_dirs**
(same 16-action sequence, same per-packet `candidate_kind` and `value_score`). That is strong
evidence the win is *algorithmic, not luck*. A 6th, older 14-action run (different harness)
corroborates the same terminal mechanic. Note `informer_candidate_counts.ghost_relation_memory == 0`
on every L0 packet — the win is re-derived fresh each run from current-frame informers (deterministic),
not replayed from crystallized memory.

---

## PROVENANCE LEDGER

| Pointer | Proving run_dir (level_00) | Packet idx | delta |
|---|---|---|---|
| PORTBAND_EDGE_LOCK_ALIGN | contra_sp80_episode_levels_00_01_20260608T175142Z | 0–4 | 1 |
| FLOW_DIRECTION_DESCENT | contra_sp80_episode_levels_00_01_20260608T175142Z | 6–12 | 1 |
| MULTI_SOURCE_GOAL_ADJACENCY | contra_sp80_episode_levels_00_01_20260608T175142Z | 13–14 | 1 |
| ACTIVATE_FLOW_ON_FULL_ALIGNMENT (capstone mechanic) | contra_sp80_episode_levels_00_01_20260608T175142Z | 15 (WIN) | 1 |

Identical-sequence corroborating run_dirs (all delta=1, status completed, 16 actions):
`_20260608T173344Z`, `_20260608T165116Z`, `_20260604T171457Z`, `runs/_system_run/level_00`.
Cross-harness corroboration (delta=1, 14 actions, same `activate_flow` vs=0.48 terminal after
align): `runs/sp80_ghost_capture_level_00_20260602T041500Z`.

Ghost crystallization of the win:
`contra_sp80_episode_levels_00_01_20260608T175142Z/level_00/ghost.json` →
8 `confirmed_relationship_rows`: rows 0–6 `relation_family="edge_alignment"`
(`edge_alignment_kind="centered_dual_port_projection"`), row 7 `relation_family="activation_survived"`,
`round_index=15`, `confidence_score=1.0`, `confirmed_by=["deltaband","actionmodband","levels_completed_delta"]`,
`levels_completed_delta=1`, `survived_activation=true`. `completion_proof.terminal_packet_stop_reason="levels_completed_increased"`.

---

## The win, in one paragraph (L0, `pour_into_buckets`, `goal_object_count=2`)

CONTRA selects the **source actor** (BLUE platform — confirmed: `selected_anchor.shape=[5,1]`,
`target_label="bucket"`), drives it to **edge-align** with a target-port pair using a PortBand
**transform-hypothesis displacement**, follows a **downward flow-direction descent** toward the
buckets, brings a **second independent source adjacent** to its bucket, and finally issues
**`activate_flow` (ACTION5)** which pours the liquid (magenta) into all buckets and completes the
level (`levels_completed 0→1`). The premature `activate_flow` at idx 5 (identical vs=0.48) did
**not** win — the only difference is board alignment state, which is the heart of the mechanic.

---

## POINTER 1 — PORTBAND_EDGE_LOCK_ALIGN

- **applies_when:** a selectable *source* object and one or more *target ports* exist; an
  `edge_lock` / port-alignment configuration is derivable (`edge_lock_status ==
  "has_ranked_edge_lock_configuration"`); goal is a fill/route/connect condition. General predicate:
  *"move object A so its edge projects onto target port(s) B by a known displacement vector."*
- **procedure (observed thresholds):**
  1. Compute the PortBand lineup matrix; pick the row with `lineup_matrix_score == 1.0`,
     `alignment_kind == "centered_dual_port_projection"`, `match_status == "exact_alignment"`,
     `edge_lock_adjusted_confidence_score == 1.0`. Its `transform_hypothesis_xy` is the
     source→port displacement (L0 idx0: `[12.0, 36.0]`).
  2. `select_object` at the source's `target_override.click_xy` (idx0: `(22,18)`,
     `component_id=c0004_v09_00`). value_score ≈ **0.818**.
  3. While not aligned, emit motion along `portband_preferred_axis` first
     (`portband_axis_priority_policy = "centered_dual_port_projection_uses_horizontal_transform_before_dominant_abs_axis"`):
     `base_transform_primitive=move_right` → then orthogonal `move_down`
     (`orthogonal_axis_policy = "same_portband_task_loss_switches_to_remaining_transform_axis"`).
     Observed: idx2 move_right (vs 0.82), idx3 move_down (vs 0.821).
  4. Stop pushing a task once `task_saturated == true` /
     `task_saturation_positive_move_count >= 1` (penalty `task_saturation_penalty≈0.52`,
     `round_completed_task_penalty≈0.24`); vs collapses ~0.82 → **0.63** (idx4). That collapse is
     the *"alignment achieved, move on"* signal.
- **mechanic:** edge-aligning a source to its target port-pair is the precondition that makes the
  later flow activation succeed.
- **evidence_sources_used:** `portband.lineup_matrix.rows`.
- **value_score_signature:** ~**0.82** while aligning (with `edge_lock_adjusted_confidence_score=1.0`,
  `lineup_matrix_score=1.0`), dropping to ~0.63 on `task_saturated=true`.
- **provenance:** run_dir `…175142Z/level_00`, packets **idx 0–4**, delta=1.
  Ghost rows 0–4 (`relation_family="edge_alignment"`, `shared_transform_xy` ∈ {[12,36],[8,36],[8,32]}).
- **generalization_hypothesis:** General = "align object to a target by a measured displacement
  vector and stop on saturation." SP80-specific = the `centered_dual_port_projection` geometry and
  horizontal-first axis order; a new game would supply its own `alignment_kind`/transform.
- **crystallization_target:** `contra_loop.register(name="PORTBAND_EDGE_LOCK_ALIGN",
  trigger="edge_lock_status==has_ranked_edge_lock_configuration", emit="select+axis_moves(transform_hypothesis_xy)",
  stop="task_saturated", vs≈0.82, evidence="portband.lineup_matrix.rows")`.

---

## POINTER 2 — FLOW_DIRECTION_DESCENT

- **applies_when:** a source has been selected/aligned and the target (bucket) lies along a single
  dominant axis; a flow-direction hypothesis exists. General predicate: *"after coarse alignment,
  march the source along the hypothesized flow axis until display-distance to the target stops
  decreasing."*
- **procedure (observed):**
  1. Take the directional-motion probe whose primitive matches the flow-direction hypothesis
     (`directional_probe_policy = "flow_direction_hypothesis_maps_to_matching_move_primitive"`).
     On L0 this is `move_down`.
  2. Repeat the move while `goal_progress.distance_reduced == true` (each step
     `distance_delta == +1.0`). Observed idx6→11: bucket_00 display-manhattan 10.5 → 4.5 over 6
     monotone `move_down`s. value_score ≈ **0.61–0.69**.
  3. Stop when `distance_delta == 0.0` (idx12: 4.5→4.5, blocked) — the descent is exhausted; hand
     off to the next phase.
- **mechanic:** liquid/source travels along the flow axis toward the bucket; closing display
  distance along that axis sets up the pour.
- **evidence_sources_used:** `action_semantics_ledger`, `action_semantics_ledger.directional_motion_probe`.
- **value_score_signature:** flat ~**0.69** band of repeated single-axis motion probes
  (`semantic_probe_family` directional), no PortBand row competing above it.
- **provenance:** run_dir `…175142Z/level_00`, packets **idx 6–12**, delta=1.
- **generalization_hypothesis:** General = greedy single-axis descent gated on monotone
  distance-reduction. SP80-specific = downward gravity-like flow; another game may hypothesize a
  different axis. **Caveat (honest):** the flow *direction* is currently a probe, not crystallized —
  ghost rows record `flow_direction="unknown"`, `flow_axis="unknown"`. This descent works by
  distance-feedback, not by a confirmed direction model.
- **crystallization_target:** `contra_loop.register(name="FLOW_DIRECTION_DESCENT",
  trigger="aligned_source && distance_reduced", emit="repeat(flow_axis_move)",
  stop="distance_delta==0", vs≈0.69, evidence="action_semantics_ledger.directional_motion_probe")`.

---

## POINTER 3 — MULTI_SOURCE_GOAL_ADJACENCY

- **applies_when:** multiple *independent* source→target pairs exist
  (`independent_source_count >= 2`, `independent_target_count >= 2`) and the primary PortBand task
  is saturated (`all_eligible_portband_tasks_saturated == true`). General predicate: *"when several
  objects must each reach their own goal, pick the source with the fewest steps to goal-adjacency
  next."*
- **procedure (observed):**
  1. Build `portband.multi_object_goal_adjacency_plan`; choose the row with
     `optimum_signal_type == "multi_object_independent_goal_adjacency_least_steps"` and the highest
     `aggregate_signal_score` (idx13: **0.750048**, `positive_signal_count=2`, `total_step_cost=24`,
     `step_efficiency_score≈0.571`).
  2. `select_object` the next source at `target_override.click_xy` (idx13: `(30,46)`,
     `component_id=c0005_v09_00`). value_score ≈ **0.75**.
  3. Emit the planned move toward adjacency (idx14: `move_right`, `transform_hypothesis_xy=[4.0,8.0]`),
     value ≈ 0.75.
- **mechanic:** every source must sit adjacent to its own bucket before the single global
  `activate_flow` can pour into *all* buckets (`goal_object_count == 2`).
- **evidence_sources_used:** `portband.lineup_matrix.rows`, `portband.multi_object_goal_adjacency_plan`.
- **value_score_signature:** ~**0.75** with `confidence_model = "multi_object_positive_signal_per_step_not_probability"`,
  active only after `all_eligible_portband_tasks_saturated=true`.
- **provenance:** run_dir `…175142Z/level_00`, packets **idx 13–14**, delta=1.
  Ghost rows 5–6 (`shared_transform_xy=[4.0,8.0]`).
- **generalization_hypothesis:** General = least-steps scheduling over independent sub-goals
  (a tiny assignment/relay planner). SP80-specific = nothing structural; this is the most
  transferable pointer.
- **crystallization_target:** `contra_loop.register(name="MULTI_SOURCE_GOAL_ADJACENCY",
  trigger="independent_source_count>=2 && all_portband_tasks_saturated",
  emit="select(least_steps_source)+adjacency_move", vs≈0.75,
  evidence="portband.multi_object_goal_adjacency_plan")`.

---

## POINTER 4 — ACTIVATE_FLOW_ON_FULL_ALIGNMENT  (capstone mechanic — the actual win step)

- **applies_when:** all source objects are edge-aligned/adjacent to their target ports
  (every PortBand task saturated AND multi-object adjacency plan satisfied) and an `activate`/flow
  primitive (ACTION5) exists. General predicate: *"once every source is in position, fire the
  global state-change action."*
- **procedure (observed):**
  1. Emit `activate_flow` (ACTION5). On the winning packet idx15 it was selected as a generic
     `action_semantics_ledger:unknown_action_family_probe`, `ledger_base_score=0.48`,
     `directional_probe_mapped_primitive="activate_flow"`.
  2. **Effect:** `levels_completed 0→1`, `total_levels_completed_delta=1`,
     `selected_display_coord [6,11]→[57,57]` (board reset to next level),
     `steps_remaining 15→45`. `stop_reason="levels_completed_increased"`.
- **mechanic (one sentence):** *With every source aligned adjacent to its bucket, `activate_flow`
  pours the magenta liquid into all buckets and completes the level.*
- **proof it is alignment-gated, not the action alone:** the **identical** `activate_flow`
  (same vs=0.48, same `unknown_action_family_probe`) at **idx 5** did **NOT** win
  (`levels_completed_delta=0`, distance unchanged) — only board alignment state differed. Same
  contrast holds in the corroborating 14-action run (`sp80_ghost_capture…041500Z`): `activate_flow`
  vs=0.48 wins only at the terminal idx 13 after the move_right×3 / move_down×9 alignment march.
- **evidence_sources_used:** `action_semantics_ledger`, `action_semantics_ledger.unknown_action_family_probe`.
- **value_score_signature:** **0.48** (`ledger_base_score=0.48`). NOTE: this is a *low-confidence
  probe score*, identical for winning and losing activations — CONTRA does **not** yet score the
  aligned activation any higher than a blind probe (see WANTED MECHANIC #2).
- **provenance:** run_dir `…175142Z/level_00`, packet **idx 15 (WIN, delta=1)**; negative control
  packet **idx 5 (delta=0)**. Ghost row 7 `relation_family="activation_survived"`,
  `round_index=15`, `confidence_score=1.0`, `confirmed_by` includes `levels_completed_delta`,
  `survived_activation=true`.
- **generalization_hypothesis:** General = "commit the global effect action once preconditions are
  met." SP80-specific = the flow/pour semantics. Transfers to any game with a deferred global
  trigger gated on a configuration.
- **crystallization_target:** `contra_loop.register(name="ACTIVATE_FLOW_ON_FULL_ALIGNMENT",
  trigger="all_sources_aligned_and_adjacent", emit="activate_flow",
  confirm="levels_completed_delta>0", evidence="action_semantics_ledger", note="trigger currently a 0.48 probe — needs a confidence gate")`.

---

## WANTED MECHANICS (loop targets — NOT seed solvers; no delta>0 artifact)

These are clearly evidenced failures/gaps. They are excluded from the verified seed set.

### WANTED #1 (biggest L1 blocker) — STABLE_MULTI_SOURCE_RELAY_LOCK
- **Evidence:** L1 (`contra_sp80_episode_levels_00_01_20260608T175142Z/level_01`,
  `…173344Z`, `…165116Z`) fails identically: `status="budget_exhausted"`,
  `levels_completed_delta=0`, 20 actions. Decision trail **thrashes across 9 distinct
  `candidate_kind`s** (multi_object_goal_adjacency ×5, symmetry_sensor ×3, portband_lineup ×3,
  directional ×2, deferred_like_control ×4, support_surface, geoband_static) and re-selects objects
  with distance swings of ±6 to ±8 per packet. Two `activate_flow`s (idx11, idx13) fire prematurely
  and *increase* distance.
- **Contrast with the win:** L0 holds a single stable edge-lock — `top_value_score=0.82` and
  `candidate_kind=portband_lineup_matrix_row` for idx0–3 with no competing informer. L1 never
  establishes that lock (no row reaches the L0 0.82 edge_lock band), so it cannot commit to an
  alignment task long enough to reach the pour.
- **What the loop must build:** a multi-source *sequencing/relay* planner that (a) commits to one
  source→target lock at a time, (b) suppresses re-selection thrash, and (c) defers `activate_flow`
  until *all* sources are locked — the L1 board needs an ordered multi-block relay, not a single
  align-and-pour. (Matches the standing note that L1/L2 is a fuel-efficient multi-block relay.)

### WANTED #2 — ACTIVATION_COMPLETE_PREDICTOR (confidence gate for POINTER 4)
- **Evidence:** the winning `activate_flow` (idx15) and the losing one (idx5) carry the **same**
  `value_score=0.48` `unknown_action_family_probe`. CONTRA wins by *probing*, not by *predicting*.
- **What the loop must build:** a predicate that scores `activate_flow` high **only** when every
  PortBand task is saturated AND the multi-object adjacency plan is satisfied — turning the blind
  0.48 probe into a confident, gated trigger. This also crystallizes the missing
  `flow_direction`/`flow_axis` (ghost currently records both as `"unknown"`).

### WANTED #3 — CLOSE_THE_GHOST_LOOP
- **Evidence:** every L0 packet shows `informer_candidate_counts.ghost_relation_memory == 0`.
  The crystallized `ghost.json` (edge_alignment + activation_survived rows) is captured but **not
  fed back** into the next episode's informers (`ghost_verifier="pending"`). The win is re-derived
  from scratch each run.
- **What the loop must build:** wire confirmed ghost relation rows back in as a tier-2 informer so
  a solved mechanic accelerates the next level instead of being re-discovered.
