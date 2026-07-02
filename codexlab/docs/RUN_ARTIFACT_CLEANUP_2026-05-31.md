# Run Artifact Cleanup Ledger

Date: 2026-05-31
Scope: `/home/frosty40/rexus`
Status: capture-before-delete ledger for factory-floor cleanup.

## Boundary

This ledger records cleanup of historical generated artifacts only. It is not a
new experiment result and does not promote any run to control truth, goal truth,
action-choice evidence, or solve evidence.

Protected surfaces:

- `codexlab/` source outside `runs/` and `datasets/`
- `codexlab/runs/sp80_color_board_viewer/`
- `codexlab/runs/contra_v001_sp80_deltaband_controlled_run/`
- `codexlab/runs/contra_api_smoke/`
- `codexlab/runs/sp80_lora_dry_run_v0/`
- `codexlab/runs/contra_exploreband_deployment_00_03_20260530T_smoke/`
- `codexlab/runs/contra_post_revert_visual_proof_l00_20260530T/`
- `codexlab/runs/contra_sp80_portband_imagination_level_00_20260529T1723Z/`
- `codexlab/runs/contra_sp80_portband_integrity_audit_00_03_20260529T1834Z/`
- `codexlab/runs/sp80_color_board_viewer_portband_integrity_smoke_1834Z/`
- `codexlab/runs/sp80_color_board_viewer_portband_smoke_final/`
- `corpus/s5i5_snake_modulo/`
- `environment/environment_files/`

## Deleted Generated Run Families

The following paths were historical generated duplicates or stale probe outputs.
They had already produced local `index.html`, `manifest.json`, `summary.json`,
or `result.json` surfaces where applicable; this ledger preserves the path names
and cleanup reason before removal.

DeltaBand/oracle rollout duplicates:

- `codexlab/runs/contra_sp80_deltaband_rollout_00_03_geoband_deltaband_frames_20260529T155514Z`
- `codexlab/runs/contra_sp80_deltaband_rollout_00_03_geoband_deltaband_frames_20260529T155642Z`
- `codexlab/runs/contra_sp80_deltaband_rollout_00_03_oracle_goal_20260529T153421Z`
- `codexlab/runs/contra_sp80_deltaband_rollout_00_03_oracle_goal_20260529T153547Z`
- `codexlab/runs/contra_sp80_deltaband_rollout_00_03_oracle_goal_frames_first_20260529T154114Z`

PortBand rollout and micro-probe duplicates:

- `codexlab/runs/contra_sp80_portband_deltaband_rollout_00_03_20260529T1652Z`
- `codexlab/runs/contra_sp80_portband_deltaband_rollout_00_03_20260529T1657Z`
- `codexlab/runs/contra_sp80_portband_deltaband_rollout_00_03_20260529T_portband`
- `codexlab/runs/contra_sp80_portband_edge_cells_level_00_20260529T1742Z`
- `codexlab/runs/contra_sp80_portband_edge_cells_level_00_20260529T1750Z`
- `codexlab/runs/contra_sp80_portband_edge_precision_level_00_20260529T1732Z`
- `codexlab/runs/contra_sp80_portband_imagination_level_00_20260529T1710Z`
- `codexlab/runs/contra_sp80_portband_imagination_level_00_20260529T1714Z`
- `codexlab/runs/contra_sp80_portband_imagination_level_00_20260529T1718Z`
- `codexlab/runs/contra_sp80_portband_integrity_audit_00_03_20260529T1800Z`

PortBand viewer smoke duplicates:

- `codexlab/runs/sp80_color_board_viewer_portband_edge_cells_smoke`
- `codexlab/runs/sp80_color_board_viewer_portband_edge_precision_smoke`
- `codexlab/runs/sp80_color_board_viewer_portband_imagination_smoke`
- `codexlab/runs/sp80_color_board_viewer_portband_integrity_smoke`
- `codexlab/runs/sp80_color_board_viewer_portband_smoke`

ExploreBand stale probe/debug family:

- `codexlab/runs/sp80_exploreband_probe_l00_20260530T_actionspace_fix`
- `codexlab/runs/sp80_exploreband_probe_l00_20260530T_actionspace_move_fix`
- `codexlab/runs/sp80_exploreband_probe_l00_20260530T_portband_guard_v1`
- `codexlab/runs/sp80_exploreband_probe_l00_20260530T_portband_guard_v3`
- `codexlab/runs/sp80_exploreband_probe_l00_20260530T_portfleet_complex`
- `codexlab/runs/sp80_exploreband_probe_l01_20260530T_actionspace_fix`
- `codexlab/runs/sp80_exploreband_probe_l01_20260530T_actionspace_move_fix`

Generated scratch:

- `codexlab/runs/contra_api_smoke_revert_check`
- `codexlab/runs/playwright_isolation_audit`
- `codexlab/runs/sp80_sam_color_components_probe`
- `codexlab/runs/sp80_sam_hypothesis_viewer_yellow_u_framecoords`

## Deleted Sidecar Generated Outputs

- `test_view/dist/`
- `claudelab/` after promoting unique docs into `codexlab/docs/`
- `deepseek_lab/s5i5_checkpoint.pt`
- `deepseek_lab/s5i5_control_classifier.pt`
- `deepseek_lab/s5i5_control_history.json`
- `deepseek_lab/s5i5_history.json`
- `deepseek_lab/s5i5_random_trajectories.pt`
- `deepseek_lab/s5i5_trajectory.pt`
- `kaggle_labs/output/*.onnx`
- `angel/` after promoting the SOP into `codexlab/docs/`

Promoted from `claudelab/` before deletion:

- `codexlab/docs/CONTRA_CHARTER.md`
- `codexlab/docs/CODEX_CLAUDE_COMMS.md`
- `codexlab/docs/CONTRA_FRAME_DEBUG_TEMPLATE_HANDOFF.md`
- `codexlab/docs/CONTRA_GAMEPLAY_REPORTING_SOP.md`

Kaggle data/source/notebooks and DeepSeek source scripts were left in place.

## Remaining Cleanup Debt

- Modified tracked run artifacts remain frozen in the working tree. They should
  be reviewed as intentional refresh vs noise before any revert or commit.
- Untracked `corpus/s5i5_snake_modulo/*review*` files remain unadmitted corpus
  review notes.
- Remaining untracked `codexlab/runs` directories are not deleted by this
  ledger because they are either selected keep surfaces or need a separate
  admission/archive decision.
