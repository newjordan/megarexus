"""Hard gate: training is blocked until mining is complete.

See legal_search/MINING_THEN_TRAIN.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
GATE_PATH = REPO / "legal_search" / "MINING_COMPLETE.json"
GATE_SCHEMA = "legal_search.mining_complete.v1"
REQUIRED_CHECKLIST = (
    "registry_smoke",
    "features_25_present",
    "depth_not_stub",
    "docs_synced",
    "legal_audit",
)


class MiningIncompleteError(RuntimeError):
    """Raised when training is attempted before mining is complete."""


def gate_path() -> Path:
    return GATE_PATH


def read_gate() -> dict[str, Any] | None:
    path = GATE_PATH
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MiningIncompleteError(f"invalid gate JSON at {path}: {exc}") from exc


def mining_is_complete() -> bool:
    data = read_gate()
    if not data:
        return False
    if data.get("schema") != GATE_SCHEMA:
        return False
    if data.get("complete") is not True:
        return False
    checklist = data.get("checklist") or {}
    if not isinstance(checklist, dict):
        return False
    for key in REQUIRED_CHECKLIST:
        if checklist.get(key) is not True:
            return False
    return True


def assert_mining_complete(*, allow_ungated_scout: bool = False) -> dict[str, Any]:
    """Call at the start of any train/collect-for-train entrypoint.

    If allow_ungated_scout is True, returns a scout receipt instead of raising.
    Scout mode is not gated training.
    """
    if allow_ungated_scout:
        return {
            "gated": False,
            "scout": True,
            "complete": False,
            "message": (
                "UNGATED SCOUT — mining not complete; outputs must go under "
                "legal_search/runs/_ungated_scout/ and must not be claimed as training."
            ),
            "gate_path": str(GATE_PATH),
            "doc": "legal_search/MINING_THEN_TRAIN.md",
        }

    data = read_gate()
    if data is None:
        raise MiningIncompleteError(
            "MINING INCOMPLETE — training blocked.\n"
            f"Missing gate file: {GATE_PATH}\n"
            "Finish JOB 1 (mining) in legal_search/MINING_THEN_TRAIN.md, "
            "then write MINING_COMPLETE.json with complete:true.\n"
            "Do not train until mining is complete."
        )

    if data.get("schema") != GATE_SCHEMA:
        raise MiningIncompleteError(
            f"MINING INCOMPLETE — bad schema {data.get('schema')!r}, want {GATE_SCHEMA!r}.\n"
            "See legal_search/MINING_THEN_TRAIN.md"
        )

    if data.get("complete") is not True:
        raise MiningIncompleteError(
            "MINING INCOMPLETE — gate file has complete != true.\n"
            f"Gate: {GATE_PATH}\n"
            "See legal_search/MINING_THEN_TRAIN.md"
        )

    checklist = data.get("checklist") or {}
    missing = [k for k in REQUIRED_CHECKLIST if checklist.get(k) is not True]
    if missing:
        raise MiningIncompleteError(
            "MINING INCOMPLETE — checklist not all true: "
            + ", ".join(missing)
            + f"\nGate: {GATE_PATH}\nSee legal_search/MINING_THEN_TRAIN.md"
        )

    return {
        "gated": True,
        "scout": False,
        "complete": True,
        "gate": data,
        "gate_path": str(GATE_PATH),
        "doc": "legal_search/MINING_THEN_TRAIN.md",
    }
