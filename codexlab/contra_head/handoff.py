#!/usr/bin/env python3
"""Graceful layer handoff packets for CONTRA."""

from __future__ import annotations

from typing import Any

CONTRA_LAYER_HANDOFF_SCHEMA = "codexlab.contra_layer_handoff.v1"
CONTRA_LAYER_HANDOFF_SET_SCHEMA = "codexlab.contra_layer_handoff_set.v1"


def layer_handoff(
    *,
    layer_id: str,
    status: str,
    input_refs: list[str] | None = None,
    output_refs: list[str] | None = None,
    next_layers: list[str] | None = None,
    reason_codes: list[str] | None = None,
    counts: dict[str, int] | None = None,
    recoverable: bool = True,
    handoff_policy: str = "continue_fail_closed",
    error: BaseException | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema": CONTRA_LAYER_HANDOFF_SCHEMA,
        "layer_id": str(layer_id),
        "status": str(status),
        "input_refs": list(input_refs or []),
        "output_refs": list(output_refs or []),
        "next_layers": list(next_layers or []),
        "reason_codes": sorted(set(str(value) for value in (reason_codes or []) if value)),
        "counts": dict(counts or {}),
        "recoverable": bool(recoverable),
        "handoff_policy": str(handoff_policy),
        "hard_stop_required": False if recoverable else True,
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }
    if error is not None:
        row["error_type"] = type(error).__name__
        row["error_message"] = str(error)
    return row


def handoff_set(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hard_stop_layers = [
        str(row.get("layer_id"))
        for row in rows
        if isinstance(row, dict) and row.get("hard_stop_required") is True
    ]
    recovered_layers = [
        str(row.get("layer_id"))
        for row in rows
        if isinstance(row, dict) and str(row.get("status")).startswith("recovered")
    ]
    fail_closed_layers = [
        str(row.get("layer_id"))
        for row in rows
        if isinstance(row, dict) and "fail_closed" in str(row.get("status"))
    ]
    return {
        "schema": CONTRA_LAYER_HANDOFF_SET_SCHEMA,
        "status": "hard_stop_required" if hard_stop_layers else "graceful_handoff",
        "layer_count": len(rows),
        "hard_stop_required": bool(hard_stop_layers),
        "hard_stop_layers": hard_stop_layers,
        "recovered_layer_count": len(recovered_layers),
        "recovered_layers": recovered_layers,
        "fail_closed_layer_count": len(fail_closed_layers),
        "fail_closed_layers": fail_closed_layers,
        "layers": rows,
    }
