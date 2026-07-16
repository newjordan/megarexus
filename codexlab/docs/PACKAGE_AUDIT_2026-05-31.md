# CodexLab Package Audit

Date: 2026-05-31
Scope: `codexlab/` canonical CONTRA/SP80 work folder.

## Status

`codexlab/` is now organized as the main code package plus historical artifact lane.

- Source packages: `contra/`, `players/`
- Lab entrypoints: top-level `contra_*.py`, `sp80_*.py`, `cross_game_*.py`, `oracle.py`
- Data: `datasets/`
- Historical artifacts: `runs/`
- Project docs and handoffs: `docs/`

## Verification

Commands run from `$REPO`:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/dev_check.py
```

Result: syntax parsing, controller packet regressions, and import smoke passed.

## Current Footprint

- Source Python files outside `runs/` and `datasets/`: `94`
- Source Python LOC outside `runs/` and `datasets/`: `38,308`
- Reusable package Python files under `contra/` and `players/`: `39`
- Reusable package LOC under `contra/` and `players/`: `12,705`
- Top-level run artifact directories under `codexlab/runs`: `105`
- `codexlab/runs` size: about `1.3G`

## Cleanliness Boundary

The code package is syntactically clean and has an explicit package marker. It still intentionally uses a legacy flat-lab import style for many scripts, so run scripts from the repo root with `PYTHONPATH=codexlab` or by invoking the script path directly.

Run artifacts are retained as historical evidence/review surfaces, not as package code. Failed same-basin transcript-prior loop outputs were removed from the copied `CODEX_GOAL_RUN` workspace; remaining `codexlab/runs` artifacts should be pruned only after their summary or handoff is captured.

`codexlab/dev_check.py` is now the fast readiness entrypoint. Prefer it over a broad `compileall` pass during normal development because it uses AST parsing and `-B`/`PYTHONDONTWRITEBYTECODE=1` to avoid leaving bytecode behind.

## Remaining Reduction Targets

- Consolidate repeated HTML/review-page helpers across `sp80_color_board_viewer.py`, `sp80_pixel_group_viewer.py`, and `contra_sp80_review_page.py`.
- Split or reduce `contra/controller_head.py` only after mapping active imports and stable API symbols.
- Convert legacy flat imports to package-relative imports in a separate pass if we decide `python -m codexlab...` should become the primary execution mode.

## Reduction Completed In Follow-Up Pass

- Extracted shared review helpers into `codexlab/review_helpers.py`.
- Removed `codexlab/sp80_subagent_autonomous_attempt.py` after capturing `codexlab/docs/SP80_AUTONOMOUS_ATTEMPT_HANDOFF_2026-05-31.md`.
- Deleted the four failed autonomous-attempt run directories after preserving the latest screenshot in the PortBand failure diagnosis directory.
- Mapped `controller_head.py` callers/API in `codexlab/docs/CONTROLLER_HEAD_API_MAP_2026-05-31.md`.
- Extracted pure controller geometry/contact/digest helpers into `codexlab/contra/controller_geometry.py`, reducing `controller_head.py` from `1,878` to `1,769` LOC while keeping the public API stable.
- Added `codexlab/contra_affordance_packet_regression.py` for focused calibration/oracle/empty affordance packet-shape checks.
- Extracted oracle, calibration, and empty affordance builders into `codexlab/contra/controller_affordances.py`, reducing `controller_head.py` to `1,274` LOC while preserving its imported API symbols.
- Added `codexlab/contra_controller_packet_regression.py` for normal and invalid `run_contra_view` response packet-shape checks.
- Extracted recovered/empty packet factories into `codexlab/contra/controller_error_packets.py` and layer-output packet construction into `codexlab/contra/controller_layer_outputs.py`, reducing `controller_head.py` to `890` LOC.
- Added `codexlab/dev_check.py` as the package readiness check for future development sessions.
