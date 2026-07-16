# CONTRA Code Reduction Prep

Date: 2026-05-31
Working tree: `$REPO`

## Cleanup Completed

- Removed Python `__pycache__` directories from the repo tree.
- Removed `test_view/node_modules`.
- Added root `.gitignore` entries for Python caches, local envs, node modules, and frontend build output.
- Added root `.gitignore` entries for local agent config and sidecar generated outputs.
- Added `codexlab/docs/PROJECT_NOTES_INDEX.md` and `codexlab/runs/README.md` to make notes/results boundaries explicit.
- Added `codexlab/__init__.py` and `codexlab/docs/PACKAGE_GUIDE.md` to mark the canonical package boundary and document the legacy `PYTHONPATH=codexlab` import convention.
- Added `codexlab/dev_check.py` as the fast no-bytecode readiness check.
- Searched for `autoloop`, `auto_loop`, `auto loop`, and loop-named directories/files; none were present in this checkout.
- Follow-up search found the failed loop family in `$HOME/CODEX_GOAL_RUN/codexlab/runs/sp80_transcript_prior*`; deleted `94` generated run directories after confirming `$HOME/CODEX_GOAL_RUN/codexlab/SP80_TRANSCRIPT_ALGORITHM_LOOP_HANDOFF_20260530.md` remains.

## Current Size And LOC Baseline

- `codexlab/runs`: about `1.3G`, `105` top-level run artifact directories.
- `claudelab/runs`: about `290M`, `51` top-level run artifact directories, scratch/secondary lane.
- `test_view`: about `1.8M` after removing `node_modules`.
- `codexlab` Python files outside `runs/` and `datasets/`: `94`.
- `codexlab` Python LOC outside `runs/` and `datasets/`: `38,308`.
- `claudelab` Python LOC: `30,324`.
- `codexlab/contra` Python LOC: `12,250`.

## Largest CONTRA Code Hotspots

- `codexlab/contra/vision.py` - `1,492` LOC.
- `codexlab/contra/portband.py` - `1,031` LOC.
- `codexlab/contra/controller_head.py` - `890` LOC.
- `codexlab/contra/symmetry.py` - `879` LOC.
- `codexlab/contra/match_calibration/stage0.py` - `747` LOC.
- `codexlab/contra/statecon.py` - `716` LOC.
- `codexlab/contra/perception.py` - `626` LOC.
- `codexlab/contra/geoband.py` - `623` LOC.

## Largest Codexlab Script Hotspots

- `codexlab/sp80_color_board_viewer.py` - `1,811` LOC.
- `codexlab/contra_cross_game_smoke_calibration.py` - `1,603` LOC.
- `codexlab/sp80_pixel_group_viewer.py` - `1,589` LOC.
- `codexlab/sp80_verify_color_board_surfaces.py` - `1,428` LOC.
- `codexlab/sp80_deltaband_controlled_run.py` - `1,288` LOC.
- `codexlab/contra_sp80_review_page.py` - `1,284` LOC.
- `codexlab/sp80_verify_pixel_groups.py` - `1,112` LOC.
- `codexlab/sp80_base_controller.py` - `1,031` LOC.

## Code Reduction Rules

- Keep the current canonical implementation in `codexlab`; treat `claudelab` as scratch/reference until manually promoted.
- Preserve review-surface and provenance behavior for run artifacts.
- Do not preserve failed same-basin scout code just because it produced artifacts.
- Do not call scheduler/counter/ranker changes a new architecture if the represented search space did not change.
- Prefer deleting duplicate runners and collapsing viewer/verifier boilerplate before touching core CONTRA APIs.

## First-Pass Candidates

- Deduplicate `claudelab` copies against `codexlab`; archive only unique notes or proven useful scripts.
- Completed: reviewed and deleted `sp80_subagent_autonomous_attempt.py`, repeated `sp80_actionmodband_autonomous_attempt_*` runs, and the `sp80_subagent_autonomous_attempt_*` run after capturing `codexlab/docs/SP80_AUTONOMOUS_ATTEMPT_HANDOFF_2026-05-31.md`.
- Started: extracted shared review helpers used by `sp80_color_board_viewer.py`, `sp80_pixel_group_viewer.py`, and `contra_sp80_review_page.py` into `codexlab/review_helpers.py`.
- Completed first controller-head split: captured `codexlab/docs/CONTROLLER_HEAD_API_MAP_2026-05-31.md` and extracted pure geometry/contact/digest helpers into `codexlab/contra/controller_geometry.py`.
- Completed second controller-head split: added `codexlab/contra_affordance_packet_regression.py` and extracted oracle/calibration/empty affordance builders into `codexlab/contra/controller_affordances.py`.
- Completed third controller-head split: added `codexlab/contra_controller_packet_regression.py`, extracted recovered/empty packet factories into `codexlab/contra/controller_error_packets.py`, and extracted layer-output packet construction into `codexlab/contra/controller_layer_outputs.py`.
- Next reduction candidates: consolidate review helpers across the largest viewer/review scripts, consolidate verifier boilerplate, then reduce `vision.py` and `portband.py`.
- Treat all historical run directories as retained artifacts until their summary/handoff has been captured.
