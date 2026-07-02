#!/usr/bin/env python3
"""Fast local readiness check for the codexlab CONTRA lane."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODEXLAB = ROOT / "codexlab"
SKIP_DIRS = {"runs", "datasets", "__pycache__"}


def _source_files() -> list[Path]:
    files: list[Path] = []
    for path in CODEXLAB.rglob("*.py"):
        relative_parts = path.relative_to(CODEXLAB).parts
        if any(part in SKIP_DIRS for part in relative_parts):
            continue
        files.append(path)
    return sorted(files)


def _check_syntax() -> None:
    failures: list[str] = []
    for path in _source_files():
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path}: {exc}")
    if failures:
        raise RuntimeError("syntax check failed:\n" + "\n".join(failures))
    print(f"ok syntax: {len(_source_files())} Python files")


def _run(label: str, args: list[str]) -> None:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(CODEXLAB) if not existing else f"{CODEXLAB}{os.pathsep}{existing}"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    print(f"run {label}: {' '.join(args)}")
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def _import_smoke() -> None:
    code = "\n".join(
        [
            "import codexlab",
            "import contra",
            "import contra_head",
            "import contra_service",
            "import players",
            "import oracle",
            "import sp80_base_controller",
            "import sp80_frame_spill_perception",
            "from contra_head.api import run_contra_view",
            "from contra_head.api import CONTRA_LAYER_OUTPUT_SCHEMA",
            "from contra_head.api import CONTRA_LAYER_OUTPUT_SET_SCHEMA",
            "from contra_head.controller_head import ContraControllerRequest",
            "from contra_head.controller_head import CONTRA_RECOVERED_ERROR_PACKET_SCHEMA",
            "import contra_head.controller_affordances",
            "import contra_head.controller_error_packets",
            "import contra_head.controller_geometry",
            "import contra_head.controller_layer_outputs",
            "print('ok imports')",
        ]
    )
    _run("import-smoke", [sys.executable, "-B", "-c", code])


def main() -> int:
    checks = [
        ("syntax", _check_syntax),
        (
            "affordance-packet-regression",
            lambda: _run(
                "affordance-packet-regression",
                [sys.executable, "-B", "codexlab/contra_affordance_packet_regression.py"],
            ),
        ),
        (
            "controller-packet-regression",
            lambda: _run(
                "controller-packet-regression",
                [sys.executable, "-B", "codexlab/contra_controller_packet_regression.py"],
            ),
        ),
        (
            "cross-game-action-interaction-regression",
            lambda: _run(
                "cross-game-action-interaction-regression",
                [sys.executable, "-B", "codexlab/contra_cross_game_action_interaction_regression.py"],
            ),
        ),
        (
            "sp80-recommendation-optimum-regression",
            lambda: _run(
                "sp80-recommendation-optimum-regression",
                [sys.executable, "-B", "codexlab/sp80_recommendation_optimum_regression.py"],
            ),
        ),
        (
            "paint-control-skill-regression",
            lambda: _run(
                "paint-control-skill-regression",
                [sys.executable, "-B", "codexlab/paint_control_skill_regression.py"],
            ),
        ),
        (
            "sp80-goal-localization-regression",
            lambda: _run(
                "sp80-goal-localization-regression",
                [sys.executable, "-B", "codexlab/sp80_goal_localization_regression.py"],
            ),
        ),
        (
            "sp80-frame-spill-perception-regression",
            lambda: _run(
                "sp80-frame-spill-perception-regression",
                [sys.executable, "-B", "codexlab/sp80_frame_spill_perception_regression.py"],
            ),
        ),
        (
            "sp80-frame-backend-planning-regression",
            lambda: _run(
                "sp80-frame-backend-planning-regression",
                [sys.executable, "-B", "codexlab/sp80_frame_backend_planning_regression.py"],
            ),
        ),
        (
            "crucible-canon-regression",
            lambda: _run(
                "crucible-canon-regression",
                [sys.executable, "-B", "codexlab/crucible_canon_regression.py"],
            ),
        ),
        (
            "crucible-activation-regression",
            lambda: _run(
                "crucible-activation-regression",
                [sys.executable, "-B", "codexlab/crucible_activation_regression.py"],
            ),
        ),
        (
            "crucible-source-edge-outcome-regression",
            lambda: _run(
                "crucible-source-edge-outcome-regression",
                [sys.executable, "-B", "codexlab/crucible_source_edge_outcome_regression.py"],
            ),
        ),
        (
            "crucible-discrepancy-regression",
            lambda: _run(
                "crucible-discrepancy-regression",
                [sys.executable, "-B", "codexlab/crucible_discrepancy_regression.py"],
            ),
        ),
        (
            "crucible-mechanic-regression",
            lambda: _run(
                "crucible-mechanic-regression",
                [sys.executable, "-B", "codexlab/crucible_regression.py"],
            ),
        ),
        (
            "contra-service-contract",
            lambda: _run(
                "contra-service-contract",
                [sys.executable, "-B", "codexlab/contra_service_verify.py"],
            ),
        ),
        (
            "contra-check",
            lambda: _run(
                "contra-check",
                [sys.executable, "-B", "codexlab/contra.py", "check"],
            ),
        ),
        ("import-smoke", _import_smoke),
    ]
    for label, check in checks:
        try:
            check()
        except Exception as exc:
            print(f"FAIL {label}: {exc}", file=sys.stderr)
            return 1
    print("codexlab dev check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
