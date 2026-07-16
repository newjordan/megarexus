# CodexLab Package Guide

Date: 2026-05-31
Scope: canonical CONTRA/SP80 work folder.

## Package Boundary

`codexlab/` is the canonical implementation lane.

- `contra/` is the reusable CONTRA control package.
- `players/` is the reusable per-game player contract package.
- `sp80_*.py`, `contra_*.py`, `cross_game_*.py`, and `oracle.py` are lab runners, verifiers, viewers, and harness scripts.
- `datasets/` contains structured training/eval data.
- `runs/` contains historical run artifacts and review surfaces, not source code.

## Import Convention

The current codebase uses a legacy flat-lab import style. Run scripts from the repository root with `PYTHONPATH=codexlab`, for example:

```bash
PYTHONPATH=codexlab python codexlab/contra_api_smoke.py
PYTHONPATH=codexlab python codexlab/sp80_verify_run.py codexlab/runs/<run-dir>
```

Do not mix new code into `runs/`. New reusable control logic belongs under `contra/` or `players/`; new one-off review or verifier entrypoints belong as explicit top-level lab scripts with a clear `main()`.

## Cleanliness Rules

- Keep `__pycache__`, local envs, package installs, and frontend build outputs out of the tree.
- Use `PYTHONDONTWRITEBYTECODE=1 python -B codexlab/dev_check.py` for the fast dev check so verification does not leave bytecode behind.
- Use `PYTHON=python3 bash codexlab/sp80_integration_smoke.sh` for the live SP80 harness smoke.
- Keep failed same-basin scout outputs only long enough to capture a durable handoff.
- Preserve review-surface and provenance fields for historical runs that may be cited.
- Treat `claudelab/` as scratch/reference unless a file is intentionally promoted into `codexlab/`.
- Prefer deleting duplicate lab runners or consolidating shared viewer/verifier helpers before adding new top-level scripts.

## Current Verification

As of 2026-05-31:

- `codexlab/dev_check.py` passes.
- `codexlab/sp80_integration_smoke.sh` passes with the conda Python runtime.
- Python syntax parsing passes for `codexlab` source outside `runs/` and `datasets/`.
- Core legacy imports pass with `PYTHONPATH=codexlab` for `contra`, `players`, `oracle`, and `sp80_base_controller`.
- No Python cache directories or bytecode files should remain after cleanup.
