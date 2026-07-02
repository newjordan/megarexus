# CodexLab SP80 Control Harness

`codexlab/` is the canonical CONTRA/SP80 implementation lane for this repo.
See `docs/PACKAGE_GUIDE.md` for the package boundary, import convention, and
cleanup rules. The current lab scripts use the legacy flat-lab import
convention; run from the repository root with `PYTHONPATH=codexlab` unless a
script states a different command.

## Use CONTRA — the cockpit

`codexlab/contra.py` is the one headline tool you invoke. It IS the integrator
that holds the system together; reach everything else through it.

```bash
PYTHONPATH=codexlab python codexlab/contra.py check    # is the WHOLE system attached? (Oracle/Ghost->CONTRA->scorer->harness->reporting)
PYTHONPATH=codexlab python codexlab/contra.py serve    # launch the informer-DAG HTTP API
PYTHONPATH=codexlab python codexlab/contra.py run      # SP80 episode pipeline   (--levels 0,1 ...)
PYTHONPATH=codexlab python codexlab/contra.py report   # build a gameplay report (--run-dir ... --out-dir ...)
PYTHONPATH=codexlab python codexlab/contra.py gate     # full local green gate (dev_check)
PYTHONPATH=codexlab python codexlab/contra.py view     # run CONTRA's view on one frame
```

`contra check` is the flagship: if any subsystem has detached it fails loud and
names the broken edge — no more hand re-attaching. Subcomponents (reach them via
the cockpit, not directly): `contra_head/` (controller-head internals) and
`contra_service/` (the informer-DAG HTTP API).

Fast readiness check:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/dev_check.py
```

This check parses source without writing bytecode, runs the focused controller
packet regressions, and import-smokes the current package split.

Fast integration smoke:

```bash
PYTHON=/home/frosty40/miniconda3/bin/python3 bash codexlab/sp80_integration_smoke.sh
```

This drives the live SP80 harness with `tiny_mlp`, checks rendered frames, runs
`sp80_verify_run.py`, and removes its temporary run directory unless `KEEP=1`.

Run CONTRA on SP80 levels 0 and 1:

```bash
PYTHONPATH=codexlab /home/frosty40/miniconda3/bin/python3 -B codexlab/sp80_run_contra_levels.py --levels 0,1 --backend tiny_mlp --scale 8 --max-actions 20
```

This is the canonical gameplay path for "run CONTRA on L0 and L1." It runs
`sp80_player_harness.py --mode episode` on each requested ARCPrize SP80 level,
verifies each persistent episode with `sp80_verify_episode_run.py`, then builds
the Angel `contraview_report.json` surface through `gameplay_report_harness.py`.
The report opens through `?tool=contra-oracle&stage=portband&data=/data/<run_id>/contraview_report.json`.
The report verifier is strict for gameplay: `angel_report_verify.py` must pass
with `--strict-png-ledger --require-gameplay-visual-contract`.
That contract requires first-class Angel image frames for every recorded action
packet: every click overlay, every recorded gameplay before/after frame,
returned action-frame sequences, Config, CONTRA pass, View System passes, raw
colors, control isolation, scene surfaces, symmetry, actor/target/support
affordances, StateCon, GeoBand boundary queues, ExploreBand, action semantics,
layer handoffs, debug summary, admission, Oracle, Ghost, PortBand, relationship
graph, relationship ladder, DeltaBand, ActionModBand, and episode memory. It
also requires level summary images for Oracle, Ghost, relationship graph,
relationship ladder, and PortBand. The report is image-first: CONTRA mechanics
are not reportable as tables or JSON alone. Recorded clicks must be visible in
the action overlay image and machine-checkable through matching `click_xy` and
`has_click_marker` metadata in `contraview_report.json`. Returned action-frame
sequences must also be machine-checkable: each action timeline row records
`returned_frame_count` and `expected_gameplay_frame_count`, and the verifier
checks those counts against first-class Angel `gameplay_frame` rows. Each action
timeline row also records `contra_layer_visual_map`; every non-hidden
`layer_output_packets.outputs[].layer_id` must map to a visual stage for that
same action. The report-local `angel_assets/` directory must exactly match the
current `contraview_report.json` image references; stale unreferenced image files
are verifier failures.

Do not use `sp80_sweep_levels.py` as a gameplay result. That script is a
calibration/action-set sweep and its pass status is not level completion.

Contra:

- `Contra` is the control head: `intent + observation -> one allowed primitive`.
- `Contra v0.1` may read a `goal_condition` made from semantic labels and local relation fields.
- For SP80, buckets are labeled from live sprites tagged `repwkzbkhxl` and exposed as the `pour_into_buckets` goal condition.
- It must receive `view_state.control_isolation_packet` from a raw frame before every action.
- `contra/vision.py` builds that packet from raw pixels as a shape-trace-first view: shape traces are the primary control units, and exact color groups are supporting boundary evidence; this is Contra view structure, not a semantic bucket label or route.
- Every action packet must record a post-action view and a visual correlation check tying the chosen primitive to the observed state/frame delta.
- It is not the solver, not Qwen, and not the game motor.
- It cannot emit coordinates, routes, multiple actions, or gameplay-completion claims.
- Current backends: `tiny_mlp`, `lora_adapter`, and `qwen_http` behind the same primitive-only contract.
- Package coordinator: `contra/contra.py`; stable API: `contra/api.py`; controller head: `contra/controller_head.py`; control contract: `contra/controls.py`; primary visual sense: `contra/vision.py`.

Control path note: basic controls remain the first path. The object viewer is
an observation/label support layer for later goal conditioning; it is not a
replacement for movement/select/activate primitive checks.

Player architecture:

- `players/base.py`: shared primitive registry, player descriptor, brain-decision packet, and packet recorder.
- `players/sp80/registry.py`: SP80 primitive set, intent cases, aliases, and verifier expectations.
- `players/sp80/controller.py`: concrete SP80 player controller entrypoint.
- `sp80_player_harness.py`: runs the player contract against the live SP80 environment and writes packets.
- `sp80_verify_run.py`: verifies the recorded run from files and hashes, including the shape-trace-first control-isolation contract.
- `contra/api.py`: stable programmatic entrypoint for other systems. Call `contra.run_contra_view(...)` once per raw frame.
- `contra/controller_head.py`: request/response schema and orchestration for raw colors, scene surfaces, actor/target/support hypotheses, StateCon, GeoBand, DeltaBand gate, debug summary, and fail-closed admission status.
- `contra/handoff.py`: layer communication envelopes. Every controller-stage handoff records complete, empty/fail-closed, or recovered-error status so callers get a packet instead of an uncaught layer stop.
- `contra/vision.py`: deterministic raw-pixel shape-trace control isolation and scene-surface hypotheses used inside Contra view state.
- `contra/perception.py`: unified perception pipeline that wires vision, actor/target/support affordances, StateCon, GeoBand, and the relationship ladder.

This is a control player, not a gameplay solver. It tests whether a small brain
can choose one primitive and whether the game-specific motor can execute and
verify the primitive.

Goal-conditioned SP80 check:

- `approach_nearest_bucket` asks Contra for one local primitive toward the nearest labeled bucket.
- The request contains `goal_condition`, not bucket coordinates or a route.
- The packet records `goal_labels` provenance separately from the Contra request.
- The verifier requires `goal_progress.distance_reduced == true` for that local primitive.

Pixel-grounded object label viewer:

- Raw frame pixels are the observation ground truth.
- Controlled object masks come from engine sprite pixels projected onto the visible frame.
- Buckets are one object class, not the whole viewer; other visible SP80 objects are recorded too.
- SAM is not configured in this run. Future SAM rows must be marked as hypotheses until verified.

Build the viewer:

```bash
python codexlab/sp80_object_label_viewer.py --levels all --out-dir codexlab/runs/sp80_object_label_viewer --scale 8
python codexlab/sp80_verify_object_labels.py codexlab/runs/sp80_object_label_viewer
```

Viewer outputs:

- `index.html`: local review page with raw frame and controlled overlay.
- `object_labels.jsonl`: one pixel-mask object label row per visible object.
- `label_review_manifest.json`: command, source hashes, run condition, and file hashes.

SAM hypothesis viewer:

- `sp80_sam_hypothesis_viewer.py` runs local SAM2 over SP80 raw-frame renders and writes mask hypotheses.
- `sp80_sam_scoring.py` is the modular scorer. Its current modules check mask bounds, area, controlled-mask overlap, bucket overlap, geometry hint, and raw-pixel color coherence.
- `sp80_verify_sam_hypotheses.py` re-checks masks, hashes, score packets, and the no-admission boundary from disk.
- SAM rows remain `evidence_status: hypothesis`; even confirmed overlap rows are not admitted as controller truth or goal-condition truth without a later explicit promotion step.

Run the SAM2 hypothesis viewer:

```bash
python codexlab/sp80_sam_hypothesis_viewer.py --levels all --out-dir codexlab/runs/sp80_sam_hypothesis_viewer --grid 2 --max-masks 12 --scale 8
python codexlab/sp80_verify_sam_hypotheses.py codexlab/runs/sp80_sam_hypothesis_viewer
```

SAM outputs:

- `sam_hypotheses.jsonl`: one SAM mask hypothesis row per retained mask.
- `sam_scorecard.json`: modular scoring summary and confirmation counts.
- `controlled_object_labels.jsonl`: controlled engine-mask rows used as confirmation surfaces.
- `index.html`: raw frame, controlled overlay, and SAM-hypothesis overlay review page.

SAM acceptance gate:

- `sp80_sam_acceptance_gate.py` converts SAM hypothesis rows into a fail-closed review ledger.
- Without an explicit review decision file, every SAM row is written as `blocked`.
- Accepted rows become object-hypothesis review candidates only; this patch still records zero admission into control truth, goal conditions, Contra view state, or action choice.
- `sp80_verify_sam_acceptance.py` checks source hashes, row coverage, accepted-row provenance, and zero-admission counts.

Build and verify the default fail-closed acceptance ledger:

```bash
python codexlab/sp80_sam_acceptance_gate.py codexlab/runs/sp80_sam_hypothesis_viewer
python codexlab/sp80_verify_sam_acceptance.py codexlab/runs/sp80_sam_hypothesis_viewer
```

Optional manual review decisions use JSONL rows with this shape:

```json
{"schema":"codexlab.sp80_sam_review_decision.v1","hypothesis_id":"sam_l04_m002","review_decision":"accept","reviewer":"human","reviewed_at_utc":"2026-05-26T00:00:00Z","review_note":"Confirmed against raw frame and controlled bucket mask."}
```

An `accept` decision is still blocked unless the source SAM row is confirmed against a controlled mask and the review row carries reviewer/time/note provenance.

Build the one-row gate probe and per-slice isolation review page:

```bash
python codexlab/sp80_sam_acceptance_gate.py codexlab/runs/sp80_sam_hypothesis_viewer --review-decisions codexlab/runs/sp80_sam_hypothesis_viewer/sam_review_decisions.jsonl
python codexlab/sp80_verify_sam_acceptance.py codexlab/runs/sp80_sam_hypothesis_viewer
python codexlab/sp80_sam_iso_review.py codexlab/runs/sp80_sam_hypothesis_viewer
python codexlab/sp80_verify_sam_iso_review.py codexlab/runs/sp80_sam_hypothesis_viewer
```

The gate probe accepts `sam_l04_m002` only as an object-hypothesis review candidate. Admission counts for control truth, goal conditions, Contra view state, and action choice remain zero.

SAM-to-Contra admission gate:

- `sp80_sam_contra_admission_gate.py` is the second gate after acceptance.
- It admits only rows with `acceptance_status: accepted`, recorded review provenance, and `contra_object_view_eligible: true`.
- Admitted rows may enter the Contra object view only. Control truth, goal conditions, and action choice remain fail-closed and must stay zero.
- `contra.build_control_isolation_packet(..., admitted_sam_objects=...)` can then make admitted SAM masks the packet primary view while preserving raw shape traces as deterministic audit evidence.

Build and verify the fail-closed Contra admission ledger:

```bash
python codexlab/sp80_sam_contra_admission_gate.py codexlab/runs/sp80_sam_hypothesis_viewer
python codexlab/sp80_verify_sam_contra_admission.py codexlab/runs/sp80_sam_hypothesis_viewer
```

Human visual hint:

- The current recorded hint says bucket targets are the yellow U-shaped pixel areas in the review pictures.
- This hint is recorded in `sam_visual_hints.json`.
- It was not part of the existing SAM2 mask-generation condition, so it is shown in the isolation review page and carried as a future run/scoring cue only.

Rerun SAM masks with the yellow U-shaped bucket hint:

```bash
python codexlab/sp80_sam_hypothesis_viewer.py --levels all --out-dir codexlab/runs/sp80_sam_hypothesis_viewer_yellow_u --prompt-sources visual_hint_yellow_u --visual-hints codexlab/runs/sp80_sam_hypothesis_viewer/sam_visual_hints.json --max-masks 24 --scale 8
python codexlab/sp80_verify_sam_hypotheses.py codexlab/runs/sp80_sam_hypothesis_viewer_yellow_u
```

Color-first SAM prompts:

- `raw_color_components` looks only at the rendered raw frame grid.
- It records a raw color inventory with exact pixel values, counts, dominant-background candidate, and connected components.
- It creates SAM positive-point prompts from each raw color component. It does not use semantic object names, controlled centers, human hints, routes, or replays for prompt selection.

```bash
python codexlab/sp80_sam_hypothesis_viewer.py --levels all --out-dir codexlab/runs/sp80_sam_color_components --prompt-sources raw_color_components --raw-color-min-component-area 1 --max-masks 64 --scale 8
python codexlab/sp80_verify_sam_hypotheses.py codexlab/runs/sp80_sam_color_components
```

Color-board scene-surface hypotheses:

- `sp80_color_board_viewer.py` also scores raw color components into `floor_hypothesis`, `wall_hypotheses`, and `background_hypotheses`.
- Unassigned components are reported as `residual_component_hypotheses`; this is intentionally non-semantic and replaces the old misleading foreground wording.
- Every raw component now carries an `asymmetry` hypothesis with mirror-difference metrics and a fail-closed `facing_direction_hypothesis`.
- Residual components can also produce `antisymmetric_relationship_hypotheses` when similar same-color components have opposed facing hypotheses.
- SP80 calibration records the user-supplied yellow U-shape answer as `bucket_or_goal_region`: raw v11 U components are manual `open_target` evidence with explicit open directions.
- The viewer promotes those answers into first-class `target_affordance_hypotheses`. A target affordance means candidate interaction region, not goal-condition truth and not a controller action target.
- SP80 calibration also records the user-supplied actor answer as `actor_region`: raw v06 purple/magenta blocks grouped with raw v04 near-black orientation keys are manual candidate-liquid-emitter evidence.
- The viewer promotes those answers into first-class `actor_affordance_hypotheses`. An actor affordance means candidate action-origin/emitter region, not player identity and not controller truth. The flow-direction hypothesis is the vector from the near-black orientation key toward the purple/magenta actor block.
- The viewer also emits `state_condition_hypotheses` as sticky relational groups over actor affordances, target affordances, and relation features. These StateCon rows are scored by a deterministic untrained `statecon_identity_mlp` scaffold so the schema is MLP-ready without pretending a trained model exists.
- A StateCon means candidate condition gate over a small group in flux; it is not one object, not an event, not an action, and not a solve condition.
- `relation_descriptor_score` is the preferred StateCon score name. Existing `event_enable_score` rows are kept only as a legacy compatibility alias and must carry `score_semantics: descriptive_relation_score_not_prediction`.
- StateCon support members are admitted only through manual calibration or an explicit future gate. Level 0 currently records the raw v09 blue horizontal bar as a user-supplied `primary_action_solver` support member.
- StateCon visuals now use a branch/node graph: an amber StateCon bounding container, actor/target nodes, typed support-group nodes, and support leaf nodes. The macro StateCon anchor is recorded as `relation_bbox_center_not_object_anchor` and is not rendered as an object dot.
- Each StateCon now carries a fail-closed `geoband_hypothesis`: a relational affordance network over actors, goal/target attachments, support groups, actor-to-goal edges, actor-to-support edges, support-to-goal edges, actor-to-actor edges, bearings, distances, angular error/tolerance, lateral bandwidth, and symmetry score. A GeoBand is static relationship evidence only, not an action plan or solve condition.
- GeoBand now uses first-class boundary passes. Pass 1, `geoband_primary_relationship_group`, includes all visible relationship objects available to the StateCon. Pass 2, `geoband_subgroup_refinement`, records local subgroups without erasing primary objects. Boundary tests are not natural-language prompts; they are fail-closed unresolved tests over shape equivalence, shape-plus-Y attachments, group asymmetry, independence vs compound objects, role uncertainty, and geometry uncertainty.
- Every StateCon carries a `geoband_boundary_queue`. Objects excluded from a refined subgroup must remain in the primary GeoBand field and must carry an unresolved boundary reason. `Y` objects are known objects with an unresolved attachment and must name `base_shape_family` plus `unknown_attachment_basis`.
- `contra_relationship_ladder.v1` is the explicit admission ladder inside `contra/relationship_ladder.py` for this stack. Every stage records `stage_name`, `input_stage_ids`, `required_inputs`, `missing_inputs`, `completeness_status`, `admission_status`, `not_prediction`, `not_control_truth`, and `not_solve_truth`.
- GeoBand completeness now requires actor, goal/target, and support/action attachments. Levels without support/action attachments must be marked `incomplete_missing_support_action_attachment` and rendered as dim/incomplete networks.
- `contra/deltaband.py` defines `deltaband_observations`: structured before/after changes in GeoBand descriptors across a controlled action boundary. DeltaBand is not a single scalar and not a prediction mechanism. Scalar summaries are derived review signals only.
- DeltaBand records are invalid unless they carry pre-action ladder id, action invocation id, post-action ladder id, and before/after raw frame hashes.
- A null DeltaBand/action observation is first-class data: if raw pixels, component topology, and matched GeoBand descriptors do not change after an action, the record must carry `null_action_signal: true` and a repeat-null-action guard for the controller/search layer.
- These rows are scene-surface/actor/target/StateCon hypotheses only. They are not admitted as control truth, goal conditions, solve conditions, or action-choice evidence.
- The scoring starts from exact raw colors and a component contact graph, reserves dominant/full-frame background first, requires broad bottom-contact floor candidates, then scores walls and residual components.
- Multi-component visual patterns now live in CONTRA vision as scene pattern hypotheses. Checkerboard floors are grouped from repeated two-color tile components instead of being forced into the single-component `floor_hypothesis` slot.
- `sp80_verify_color_board_surfaces.py` loads `codexlab/sp80_surface_calibration.json` by default for SP80 manifests. That calibration is an agent-seeded regression artifact, not human-reviewed semantic ground truth.
- Each SP80 view now carries `manual_calibration_review`: a fail-closed human-review table with seed candidates and empty `human_reviewed_expectations` for actor ids, target ids, support/action ids, connectors, known-Y objects, unresolved objects, and image hash.
- Each view also carries `deltaband_controlled_event_requirement`: the required before/action/after artifact list for a future controlled run. Static viewers must keep this status as not-run.
- If no floor candidate clears the fail-closed threshold, `floor_hypothesis` is `null`.

Working terms:

| Term | Meaning in this harness | Admission boundary |
| --- | --- | --- |
| `actor_affordance` | Candidate action-origin/emitter region, currently from calibrated purple/magenta blocks and orientation keys. | Hypothesis only. Not player identity or controller truth. |
| `target_affordance` | Candidate interaction/open region, currently from calibrated yellow U-shape open targets. | Hypothesis only. Not goal-condition truth. |
| `scene_pattern_hypothesis` | Multi-component visual surface pattern, currently including checkerboard floor candidates from repeated alternating tile components. | Hypothesis only. Not semantic floor truth. |
| `state_condition_hypothesis` / StateCon | Sticky relational condition group over actor, target, support, and relation features. | Hypothesis only. Not an event, action, or solve condition. |
| `statecon_support_affordance` | Candidate support member that shapes or constrains a StateCon, such as a primary action solver support. | Manual or gated only. Not swept in from residuals. |
| `statecon_branch_graph` | Review graph with StateCon container, actor, target, support-group, and support-leaf nodes. Object nodes must anchor on object/component centers; macro anchors are relationship centers only. | Visualization/review structure only. |
| `contra_relationship_ladder` | Ordered contract tying raw components, surfaces, actor/target/support affordances, StateCon, GeoBand, and DeltaBand into one fail-closed admission ladder. | Contract/checking artifact only. No stage may silently promote itself. |
| `geoband_hypothesis` / GeoBand | Relational affordance network: actors, goal attachments, supports, peer actors, relationship edges, bearings, distances, bandwidth, and symmetry around a StateCon. | Static relationship hypothesis only. Not an action plan. |
| `geoband_boundary_pass` | Ordered boundary test pass. The first pass includes all visible relationship objects; later passes refine subgroups. | Contract/checking artifact only. Refinement cannot erase primary objects. |
| `geoband_boundary_queue` | Unresolved boundary tests such as shape equivalence, shape-plus-Y, group asymmetry, independence, role, or geometry uncertainty. | Hypothesis only. Requires review or DeltaBand evidence to resolve. |
| `manual_calibration_review` | Per-level human-review table for the SP80 calibration ids and image hash. | Required oversight gate. Seed candidates are not truth. |
| `deltaband_controlled_event_requirement` | Required artifact checklist for before/action/after controlled event evidence. | Not-run gate until a controlled action is recorded. |
| `deltaband_observations` / DeltaBand | Structured observed change in GeoBand descriptors across a controlled before/after action boundary. | Observation metadata only. Not prediction, not action choice. |
| `event_hypothesis` | Possible state transition if a future action occurs under a StateCon. | Not admitted by this viewer. |
| `solve_condition` | Terminal solved evidence from a controlled run or explicit future verifier. | Never inferred from these static hypotheses. |

Why these stages exist:

- Color components exist to keep the first visual pass grounded in exact raw pixels before any semantic naming.
- Actor, target, and support affordances exist to name candidate interaction roles without admitting them as truth.
- StateCon exists to keep small relational groups sticky across review and future before/after comparisons.
- GeoBand exists to describe static relational geometry and available affordance bandwidth. It informs later analysis; it does not guess outcomes.
- DeltaBand exists only after a controlled action boundary. It records how GeoBand relationships changed under `n(x)`, including null-action records that should stop repeated no-op actions in the same state.
- Event and solve systems must remain separate because they require controlled temporal evidence, not static labels.

```bash
python codexlab/sp80_color_board_viewer.py --frames-dir codexlab/runs/sp80_sam_color_components/frames --out-dir codexlab/runs/sp80_color_board_viewer --scale 12 --min-component-area 1
python codexlab/sp80_verify_color_board_surfaces.py codexlab/runs/sp80_color_board_viewer
python codexlab/cross_game_color_board_viewer.py --source-manifest codexlab/runs/cross_game_isolation_spotcheck/manifest.json --out-dir codexlab/runs/cross_game_color_board_viewer --scale 12 --min-component-area 1
python codexlab/sp80_verify_color_board_surfaces.py codexlab/runs/cross_game_color_board_viewer
python codexlab/sp80_manual_calibration_table.py --source-manifest codexlab/runs/contra_v001_sp80_boundary_pass_review/manifest.json --out-dir codexlab/runs/contra_v001_sp80_manual_calibration_table
python codexlab/sp80_verify_manual_calibration_table.py codexlab/runs/contra_v001_sp80_manual_calibration_table
python codexlab/sp80_manual_calibration_signoff_template.py --source-table codexlab/runs/contra_v001_sp80_manual_calibration_table/manual_calibration_table.json --out codexlab/runs/contra_v001_sp80_manual_calibration_table/human_signoff_template.json
python codexlab/sp80_apply_manual_calibration_signoff.py --signoff codexlab/runs/contra_v001_sp80_manual_calibration_table/human_signoff_template.json --out-dir codexlab/runs/contra_v001_sp80_manual_calibration_reviewed
python codexlab/sp80_verify_manual_calibration_table.py codexlab/runs/contra_v001_sp80_manual_calibration_table --require-reviewed
python codexlab/sp80_deltaband_controlled_run.py --levels all --out-dir codexlab/runs/contra_v001_sp80_deltaband_controlled_run --scale 12
python codexlab/sp80_verify_deltaband_run.py codexlab/runs/contra_v001_sp80_deltaband_controlled_run
```

Build a concise CONTRA debug summary from one or more viewer manifests:

```bash
python codexlab/contra_debug_summary.py codexlab/runs/contra_v001_sp80_boundary_pass_review/manifest.json codexlab/runs/contra_v001_cross_game_boundary_pass_review/manifest.json --out-dir codexlab/runs/contra_debug_summary
```

Debug summary outputs:

- `contra_debug_summary.json`: machine-readable per-view counts, stage status, missing inputs, reason codes, and source manifest hashes.
- `contra_debug_summary.md`: compact table report for review in the repo.
- `index.html`: network-viewable report with sheet links. It separates `CONTRA ran`, `StateCon active`, and `GeoBand active` so a complete raw-color pass cannot be mistaken for a complete relationship pass.

Unified CONTRA API smoke:

```bash
python codexlab/contra_api_smoke.py --raw-grid codexlab/runs/sp80_sam_color_components/frames/level_00_raw.npy --level-index 0 --out codexlab/runs/contra_api_smoke/result.json
```

CONTRA API v0.01 verifier/matrix:

```bash
python codexlab/contra_api_matrix.py --out-dir codexlab/runs/contra_api_v001_matrix
python codexlab/contra_api_verify.py codexlab/runs/contra_api_v001_matrix --require-probes
```

One-game cross-game smoke calibration setup:

```bash
python codexlab/contra_cross_game_smoke_calibration.py --out-dir codexlab/runs/contra_cross_game_smoke_calibration
python codexlab/contra_api_matrix.py --out-dir codexlab/runs/contra_api_v001_matrix --cross-game-smoke-calibration codexlab/runs/contra_cross_game_smoke_calibration/ar25_level_05/smoke_calibration_template.json
python codexlab/contra_api_verify.py codexlab/runs/contra_api_v001_matrix --require-probes
```

The smoke calibration template starts as `pending_human_review`. The matrix will
include it as `cross_game_smoke_pending_calibration` without supplying it to
CONTRA. After human review changes the template to `calibration_status:
user_supplied_answer` and fills actor, target, and support attachments, the
same matrix command will run it as `cross_game_smoke_calibrated`; the verifier
then requires actor, target, support, StateCon, and GeoBand activation while
still requiring zero control/goal/action/solve truth admission.

`Contra v0.01` means:

- One call to `contra.run_contra_view(...)` fires the whole CONTRA network.
- Normal bad input returns a fail-closed handoff packet instead of an uncaught hard stop.
- Layers that cannot compute emit explicit skipped output packets with missing inputs and reason codes.
- Static hypotheses never admit control truth, goal truth, action-choice truth, or solve truth.
- Inactive layers explain missing inputs or empty fail-closed outputs.
- Outputs are inspectable as JSON plus markdown/HTML debug summaries.

The matrix output is an API execution report, not a gameplay result. It records
which layers activated on calibrated paint frames, which layers stayed inactive
on no-calibration cross-game frames, and which layers recovered fail-closed in
the probe rows.

Oracle assumption head:

- `oracle.py` is not CONTRA. It is a separate assumption head that can read a
  CONTRA packet and emit review-only assumptions as a sidecar.
- Oracle assumptions are not predictions, not control truth, not goal truth,
  not action-choice truth, and not solve evidence.
- Oracle records one small missing-element assumption at a time from skipped
  CONTRA layer outputs, so a caller gets a review object without pretending the
  skipped layer computed.
- When the missing element is `goal_attachment`, the current Oracle rule is:
  if goal is unknown, assume the goal candidate for review is the largest
  colorful, asymmetrical, non-centroid component outside existing scene-surface
  roles.
- When actor/player is unknown, Oracle can use the goal candidate as an
  exclusion and assume the actor candidate for review is the asymmetrical
  non-surface component closest to centroid.
- Actor/goal geometry can override raw goal asymmetry: a colorful non-centroid
  non-surface goal candidate may be selected when it sits in the actor-facing
  vector with sufficient separation. This is a review-only Oracle rule, not
  goal truth.
- Lock/key traced-shape fit outranks plain actor-facing geometry. In shape-fit
  terms, the GOAL is the lock and the ACTOR is the key. Oracle compares local
  component shape masks and can select the open/cavity component as the goal
  lock, then select the fitting component as the actor key. Size terms recorded
  include lock shape, key shape, lock open-space pixels, key area pixels,
  fit score, open-space coverage, key-into-open-space ratio, collision ratio,
  and offset.
- Components grouped into scene floor patterns are excluded from Oracle
  actor/goal selection; separate key/action-looking objects remain separate
  candidates instead of being folded into the floor.
- Oracle output may be attached beside a CONTRA packet for review, but it must
  not be fed back as admitted CONTRA truth without a later explicit gate.

```bash
python codexlab/oracle.py --contra-packet codexlab/runs/contra_api_v001_matrix/packets/cross_game_smoke_pending_calibration_ar25_level_05.json --out codexlab/runs/oracle_smoke/ar25_level_05_oracle_assumptions.json --assumption-index 0 --verify
```

Programmatic entrypoint:

```python
import contra

result = contra.run_contra_view(
    raw_frame=frame,
    game_id="sp80-589a99af",
    level_index=0,
    calibration=level_calibration,
    mode="review",
)
```

The returned packet includes raw color/component data, the shape-trace control isolation packet, scene surfaces, actor/target/support hypotheses, StateCon, GeoBand, boundary queues, the DeltaBand controlled-event gate, debug summary, and fail-closed admission reasons. Other systems should call this API instead of individual CONTRA side modules.

The response also includes `layer_handoffs`. A layer handoff is the communication record between stages:

- `complete`: layer produced its expected packet.
- `complete_empty_fail_closed`: layer ran but had no admitted inputs or rows to pass forward.
- `complete_fail_closed_waiting_for_controlled_boundary`: layer is healthy but correctly waiting for a required controlled event, such as DeltaBand.
- `recovered_error_fail_closed`: layer raised an exception, the API recorded the error, returned an empty/not-admitted packet for downstream layers, and kept the response inspectable.

No handoff promotes control truth, goal truth, action choice, or solve truth. Graceful handoff means the system remains reviewable and composable; it does not mean the missing layer succeeded.

Run the local tiny-brain action-set harness:

```bash
python codexlab/sp80_player_harness.py --backend tiny_mlp --level-index 0 --out-dir codexlab/runs/sp80_player_harness_tiny_mlp --strict
```

Terminology note: `level_index` is the SP80 environment API name. Reports call these checks `calibration slices` because they are internal control checks, not gameplay-completion claims.

Outputs:

- `result.json`: run condition, source hashes, action set, case status.
- `packets.jsonl`: one packet per primitive/control test, including `view_before`, `view_after`, and `action_visual_correlation`.
- `brain_calls.jsonl`: request/decision records at the brain boundary, including the same pre-action `view_state.control_isolation_packet` sent to Contra.
- `frames/`: raw grid `.npy` frames and unannotated PNG renders.

Verify a recorded run:

```bash
python codexlab/sp80_verify_run.py codexlab/runs/sp80_player_harness_tiny_mlp
```

Sweep all SP80 calibration slices with the same harness and verifier:

```bash
python codexlab/sp80_sweep_levels.py --backend tiny_mlp --levels all --out-dir codexlab/runs/sp80_player_harness_tiny_mlp_levels --strict
```

Diagnose control-space drift and gameplay levers from a sweep:

```bash
python codexlab/sp80_control_drift_tracker.py codexlab/runs/sp80_player_harness_tiny_mlp_levels
```

Export verified primitive-control rows for Qwen/LoRA SFT:

```bash
python codexlab/sp80_export_control_dataset.py codexlab/runs/sp80_player_harness_tiny_mlp_levels --out-dir codexlab/datasets/sp80_control_sft_v0 --heldout-levels last
```

Validate the exported dataset before any LoRA dry-run:

```bash
python codexlab/sp80_validate_dataset_rows.py codexlab/datasets/sp80_control_sft_v0
python codexlab/sp80_check_dataset_leakage.py codexlab/datasets/sp80_control_sft_v0
```

Run the first offline LoRA mechanics dry-run:

```bash
python codexlab/sp80_lora_dry_run.py --dataset-dir codexlab/datasets/sp80_control_sft_v0 --out-dir codexlab/runs/sp80_lora_dry_run_v0 --strict
```

This dry-run uses a frozen local classifier plus trainable low-rank adapter
matrices. It is not a Qwen LoRA claim; it verifies the primitive-only adapter
path before a real Qwen-family base is attached.

Run the trained adapter as the harness brain:

```bash
python codexlab/sp80_player_harness.py --backend lora_adapter --lora-adapter codexlab/runs/sp80_lora_dry_run_v0/adapter.pt --level-index 0 --out-dir codexlab/runs/sp80_player_harness_lora_adapter --strict
```

Build one debug/tracking report from the sweep and dataset:

```bash
python codexlab/sp80_debug_report.py --sweep-dir codexlab/runs/sp80_player_harness_tiny_mlp_levels --dataset-dir codexlab/datasets/sp80_control_sft_v0 --out-dir codexlab/runs/sp80_debug
```

When Qwen has been smoke-tested, include its calibration-slice sweep in the same report:

```bash
python codexlab/sp80_debug_report.py --sweep-dir codexlab/runs/sp80_player_harness_tiny_mlp_levels --dataset-dir codexlab/datasets/sp80_control_sft_v0 --qwen-sweep-dir codexlab/runs/sp80_player_harness_qwen_levels --out-dir codexlab/runs/sp80_debug
```

Qwen/LoRA backend shape:

```bash
QWEN_HARNESS_URL=http://127.0.0.1:8000/decide \
python codexlab/sp80_player_harness.py --backend qwen_http --qwen-mode simple_json --out-dir codexlab/runs/sp80_player_harness_qwen
```

The brain returns one primitive such as `move_right`; the SP80 motor grounds and executes the real game action. The harness does not accept route scripts, replay coordinates, or human traces as controller success.
