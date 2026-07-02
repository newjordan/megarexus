# CONTRA Dev Factory Floor

Date: 2026-05-31
Scope: `/home/frosty40/rexus`, canonical lane `codexlab/`.

## Ready Check

Use this before new CONTRA/SP80 development:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/dev_check.py
```

Current result: passed. It parses `94` Python source files outside `runs/` and
`datasets/`, runs the two focused controller packet regressions, import-smokes
the controller helper split, and leaves no `__pycache__` or `.pyc` files.

Use this for the live SP80 integration smoke:

```bash
PYTHON=/home/frosty40/miniconda3/bin/python3 bash codexlab/sp80_integration_smoke.sh
```

Current result: passed. It drives the `tiny_mlp` SP80 harness, checks rendered
frames, runs `sp80_verify_run.py`, and deletes its temporary `_smoke_*` run
directory unless `KEEP=1`.

## Canonical Lane

- `codexlab/` is the active implementation lane.
- `codexlab/contra/` and `codexlab/players/` are reusable source packages.
- Top-level `codexlab/contra_*.py`, `sp80_*.py`, and `cross_game_*.py` files are lab runners, viewers, verifiers, and harnesses.
- `codexlab/runs/` is historical evidence/review output, not package source.
- `claudelab/` remains scratch/reference unless a file is intentionally promoted.

## Cleanup Boundary

Keep as evidence or source:

- Modified tracked canonical run outputs until a human decides whether they are intentional evidence refreshes or noise.
- `codexlab/runs/sp80_color_board_viewer/`, `contra_v001_sp80_deltaband_controlled_run/`, `contra_api_smoke/`, and `sp80_lora_dry_run_v0/` until reverified or summarized.
- `corpus/s5i5_snake_modulo/`, `environment/environment_files/`, and other shared source catalogs.
- Unique charter/SOP notes until promoted into `codexlab/docs/` or intentionally archived.

Capture before delete/archive:

- Repeated SP80 DeltaBand/oracle rollout families under `codexlab/runs/contra_sp80_deltaband_rollout_*`.
- Repeated PortBand rollout, edge, precision, imagination, and integrity probe families.
- Repeated ExploreBand probe families from `20260530T`.
- Duplicate `sp80_color_board_viewer_portband_*` smoke outputs.
- `claudelab/runs/` after any unique summaries are preserved.

Likely generated scratch:

- `codexlab/runs/contra_api_smoke_revert_check/`
- `codexlab/runs/playwright_isolation_audit/`
- `codexlab/runs/sp80_sam_color_components_probe/`
- `codexlab/runs/sp80_sam_hypothesis_viewer_yellow_u_framecoords/`
- local sidecar model/build outputs such as `deepseek_lab/*.pt`, `kaggle_labs/output/*.onnx`, and `test_view/dist/`

Do not blanket-delete `codexlab/runs/` or shared corpus/source catalogs. Runs are
cleaned only after their result boundary is captured.

## Current Code-Reduction State

- `controller_head.py` is down to `890` LOC after geometry, affordance, recovered-packet, and layer-output helper extraction.
- Focused regressions now lock affordance packet shape and normal/error controller packet shape.
- `sp80_integration_smoke.sh` is the repo-local integration smoke entrypoint.
- Next source reductions should start outside `controller_head.py` unless new regression coverage justifies another move.

Recommended next targets:

1. Consolidate review/page helpers across `sp80_color_board_viewer.py`, `sp80_pixel_group_viewer.py`, and `contra_sp80_review_page.py`.
2. Consolidate verifier boilerplate between `sp80_verify_color_board_surfaces.py` and `sp80_verify_pixel_groups.py`.
3. Reduce `contra/vision.py` with strict control-isolation regression coverage.
4. Reduce `contra/portband.py` after the PortBand artifact boundary is captured.
