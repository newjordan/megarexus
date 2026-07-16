# CONTRA SP80 Frame-Planning Handoff - 2026-06-28

## Status

Current work improved the local CONTRA/SP80 spill-planner path from privileged
engine-state planning toward rendered-frame-derived planning. Local controlled
episode evidence now includes a level 3 completion with a full Angel report, but
this is not ARCPrize3 online scorecard proof.

Do not report full ARCPrize3 success yet. The online scorecard runner is still
disconnected because `codexlab/arc_sp80_contra_solver_online.py` is deleted in
the worktree and no active SP80 runner currently wires into ARC online
`open_scorecard` / `close_scorecard`.

## Main Code Changes

- `codexlab/sp80_spill_planner_backend.py`
  - Uses rendered `before_frame` as the preferred planning state source when
    available.
  - Labels planner evidence as `planning_state_source:
    rendered_before_frame` and `privileged_state_used_for_action_choice: false`
    on that path.
  - Adds frame-derived selection click overrides and blocks fallback from
    frame synthetic ids to engine sprite-order ids.
  - Adds persistent frame-track assignment for synthetic movable ids.
  - Adds frame-selected anchor correction from blue-control pixel perception.

- `codexlab/sp80_frame_spill_perception.py`
  - New frame-only visible-object recovery for SP80 spill boards.
  - Recovers core visible spill objects and conservative color-1 barriers from
    64x64 frames.

- New focused helpers/regressions:
  - `codexlab/sp80_frame_spill_perception_regression.py`
  - `codexlab/sp80_frame_backend_planning_regression.py`
  - `codexlab/sp80_frame_spill_planner_regression.py`
  - `codexlab/sp80_frame_backend_live_debug.py`

- `codexlab/dev_check.py`
  - Imports the new frame perception module.
  - Runs the frame perception and backend planning regressions.

## Verified Local Results

Core checks run during this work:

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/dev_check.py`
  - Passed.
- `PYTHON=python3 bash codexlab/sp80_integration_smoke.sh`
  - Passed.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/sp80_frame_backend_planning_regression.py`
  - Passed 3/3 checks.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/sp80_frame_spill_perception_regression.py`
  - Passed 6/6 checks.
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/sp80_frame_spill_planner_regression.py --time-budget 8`
  - Passed 6/6 predicted-win frame plans.

Controlled local episode artifacts:

- `codexlab/runs/_goal_sp80_frame_planning_live_l0_l1_20260627T000000Z`
  - Levels 0 and 1 completed.
  - Episode verifier passed.
  - No full readable report surface; classify as partial controlled episode.

- `codexlab/runs/_goal_sp80_frame_planning_live_l2_retry3_20260627T000000Z`
  - Level 2 completed in 14 actions.
  - Episode verifier passed.
  - No full readable report surface; classify as partial controlled episode.

- `codexlab/runs/_goal_sp80_frame_planning_live_l3_retry2_20260628T000000Z`
  - Level 3 completed in 64 actions.
  - Episode verifier passed.
  - Compact artifact has Angel report files but report verification failed
    because compact mode omitted required per-action gameplay-frame contract
    entries. Classify report as invalid.

- `codexlab/runs/_goal_sp80_frame_planning_live_l3_fullreport_20260628T000000Z`
  - Level 3 completed in 64 actions.
  - Episode verifier passed.
  - Angel report exists under `angel_report/`.
  - `angel_report/report_manifest.json` reports:
    - `report_status: controlled_run`
    - `action_count: 64`
    - `slide_count: 2009`
    - `verification.verifier_status: passed`
  - Large artifact: about 6.4 GB.

## Boundaries

- The local SP80 harness is connected:
  - `codexlab/sp80_run_contra_levels.py`
  - `codexlab/sp80_player_harness.py`
  - `codexlab/contra.py` routing into local SP80 runs.

- The ARCPrize3 online proof path is disconnected:
  - `codexlab/arc_sp80_contra_solver_online.py` is deleted.
  - Current local SP80 runners do not open or close live ARC scorecards.
  - Historical online run dirs exist, but they depend on a missing runner and
    should not be used as current proof.

- Frame-planning caveat:
  - The frame state remains a visible-core reconstruction, not a complete
    simulator state.
  - Current action choice uses that frame state when `before_frame` exists.
  - This is explicitly labeled in packet evidence, but it needs more regression
    coverage around selected-anchor correction and synthetic-id drift.

## Recommended Next Steps

1. Restore or replace `codexlab/arc_sp80_contra_solver_online.py`.
2. Wire the restored online runner to ARC lifecycle:
   `open_scorecard()` -> live gameplay -> `close_scorecard()`.
3. Bind the improved frame-sourced `spill_planner` backend into that online
   runner.
4. Add a compact-but-verifier-valid Angel report mode; the current non-compact
   report can exceed 6 GB for one level.
5. Add targeted regressions for:
   - frame selected-anchor correction,
   - synthetic-id track swaps,
   - malformed or privileged-fallback `selected_coord_source`,
   - report verifier behavior for compact episode artifacts.
6. Only after that, run a live ARCPrize3 scorecard smoke and report scorecard
   ids/status alongside local frames/logs.

