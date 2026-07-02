#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from typing import Any

MATCH_CALIBRATION_ADMISSION_FLAGS = {
    "admission_status": "not_admitted",
    "control_truth_admitted": False,
    "goal_condition_admitted": False,
    "action_choice_admitted": False,
    "solve_condition_admitted": False,
    "gameplay_move_admitted": False,
}


def calibration_id(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha1("|".join(str(part) for part in parts if part is not None).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{digest}"


def admission_flags() -> dict[str, Any]:
    return dict(MATCH_CALIBRATION_ADMISSION_FLAGS)


def missing_artifact_fields(required_artifacts: list[str]) -> dict[str, Any]:
    return {
        "required_artifacts": list(required_artifacts),
        "present_artifacts": [],
        "missing_artifacts": list(required_artifacts),
    }


def complete_artifact_fields(required_artifacts: list[str]) -> dict[str, Any]:
    return {
        "required_artifacts": list(required_artifacts),
        "present_artifacts": list(required_artifacts),
        "missing_artifacts": [],
    }


def ladder_id(relationship_ladder: dict[str, Any] | None) -> str | None:
    if not isinstance(relationship_ladder, dict):
        return None
    value = relationship_ladder.get("ladder_id")
    return str(value) if value is not None else None
