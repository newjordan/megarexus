"""Layer-output packet construction for the CONTRA controller."""

from __future__ import annotations

from typing import Any

CONTRA_LAYER_OUTPUT_SCHEMA = "codexlab.contra_layer_output.v1"
CONTRA_LAYER_OUTPUT_SET_SCHEMA = "codexlab.contra_layer_output_set.v1"


def _stage_by_name(relationship_ladder: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(stage.get("stage_name")): stage
        for stage in relationship_ladder.get("stages", [])
        if isinstance(stage, dict)
    }


def _handoff_by_layer_id(handoffs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("layer_id")): row
        for row in handoffs
        if isinstance(row, dict) and row.get("layer_id")
    }


def _skip_output_reasons(stage: dict[str, Any] | None, handoff: dict[str, Any] | None) -> list[str]:
    reasons: list[str] = []
    if isinstance(stage, dict):
        for missing in stage.get("missing_inputs", []):
            reasons.append(f"missing_input:{missing}")
    if isinstance(handoff, dict):
        reasons.extend(str(value) for value in handoff.get("reason_codes", []) if value)
    return sorted(set(reasons))


def _layer_output_packet(
    *,
    layer_id: str,
    output_key: str,
    output_kind: str,
    row_count: int | None,
    stage: dict[str, Any] | None = None,
    handoff: dict[str, Any] | None = None,
    forced_status: str | None = None,
    data_ref: str | None = None,
) -> dict[str, Any]:
    handoff_status = str(handoff.get("status")) if isinstance(handoff, dict) else ""
    stage_status = str(stage.get("completeness_status")) if isinstance(stage, dict) else ""
    recovered = handoff_status.startswith("recovered")
    waiting = "waiting_for_controlled_boundary" in handoff_status or stage_status.startswith("not_run_requires_controlled")
    has_rows = row_count is None or row_count > 0
    computed = not recovered and not waiting and stage_status in {"", "complete"} and has_rows
    skipped = not computed and not recovered and not waiting
    status = forced_status
    if status is None:
        if recovered:
            status = "recovered_error_fail_closed_output"
        elif waiting:
            status = "skipped_waiting_for_controlled_boundary_output"
        elif skipped:
            status = "skipped_fail_closed_output"
        else:
            status = "computed_output"
    missing_inputs = []
    if isinstance(stage, dict):
        missing_inputs = [str(value) for value in stage.get("missing_inputs", []) if value]
    counts = dict(handoff.get("counts", {})) if isinstance(handoff, dict) and isinstance(handoff.get("counts"), dict) else {}
    if row_count is not None:
        counts.setdefault("rows", int(row_count))
    reason_codes = _skip_output_reasons(stage, handoff)
    if skipped and not reason_codes and not missing_inputs:
        reason_codes = ["output_not_computed_fail_closed"]
    return {
        "schema": CONTRA_LAYER_OUTPUT_SCHEMA,
        "layer_id": str(layer_id),
        "output_key": str(output_key),
        "output_kind": str(output_kind),
        "status": status,
        "computed": bool(computed),
        "skipped": bool(skipped or waiting),
        "waiting_for_controlled_boundary": bool(waiting),
        "recovered_error": bool(recovered),
        "missing_inputs": missing_inputs,
        "reason_codes": reason_codes,
        "counts": counts,
        "data_ref": data_ref or output_key,
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }


def build_layer_output_packets(
    *,
    raw_packet: dict[str, Any],
    control_isolation_packet: dict[str, Any],
    scene_surfaces: dict[str, Any],
    symmetry_hypotheses: dict[str, Any],
    actor_affordances: list[dict[str, Any]],
    target_affordances: list[dict[str, Any]],
    support_affordances: list[dict[str, Any]],
    statecons: list[dict[str, Any]],
    geobands: list[dict[str, Any]],
    boundary_queues: list[dict[str, Any]],
    relationship_ladder: dict[str, Any],
    deltaband_gate: dict[str, Any],
    exploreband_budget: dict[str, Any],
    action_semantics_ledger: dict[str, Any],
    tabula_rasa_notebook: dict[str, Any],
    episode_memory: dict[str, Any],
    debug_summary: dict[str, Any],
    admission: dict[str, Any],
    handoffs: list[dict[str, Any]],
) -> dict[str, Any]:
    stages = _stage_by_name(relationship_ladder)
    handoff_map = _handoff_by_layer_id(handoffs)
    outputs = [
        _layer_output_packet(
            layer_id="raw_color_components",
            output_key="raw_color_component_packet",
            output_kind="packet",
            row_count=int(raw_packet.get("color_count") or 0),
            stage=stages.get("raw_color_components"),
            handoff=handoff_map.get("raw_color_component_packet"),
        ),
        _layer_output_packet(
            layer_id="control_isolation",
            output_key="control_isolation_packet",
            output_kind="packet",
            row_count=int(control_isolation_packet.get("shape_trace_count") or 0)
            if isinstance(control_isolation_packet, dict)
            else 0,
            handoff=handoff_map.get("control_isolation_packet"),
        ),
        _layer_output_packet(
            layer_id="scene_surfaces",
            output_key="scene_surfaces",
            output_kind="packet",
            row_count=len(scene_surfaces.get("component_graph", {}).get("components", []))
            if isinstance(scene_surfaces, dict)
            else 0,
            stage=stages.get("scene_surface_hypotheses"),
            handoff=handoff_map.get("perception_pipeline"),
        ),
        _layer_output_packet(
            layer_id="symmetry_hypotheses",
            output_key="symmetry_hypotheses",
            output_kind="packet",
            row_count=int(symmetry_hypotheses.get("symmetry_relationship_group_count") or 0)
            if isinstance(symmetry_hypotheses, dict)
            else 0,
            stage=stages.get("symmetry_hypotheses"),
            handoff=handoff_map.get("perception_pipeline"),
        ),
        _layer_output_packet(
            layer_id="actor_affordances",
            output_key="actor_affordance_hypotheses",
            output_kind="row_list",
            row_count=len(actor_affordances),
            stage=stages.get("actor_affordances"),
            handoff=handoff_map.get("actor_affordance_builder"),
        ),
        _layer_output_packet(
            layer_id="target_affordances",
            output_key="target_affordance_hypotheses",
            output_kind="row_list",
            row_count=len(target_affordances),
            stage=stages.get("target_affordances"),
            handoff=handoff_map.get("target_affordance_builder"),
        ),
        _layer_output_packet(
            layer_id="support_action_affordances",
            output_key="support_action_affordance_hypotheses",
            output_kind="row_list",
            row_count=len(support_affordances),
            stage=stages.get("support_action_affordances"),
            handoff=handoff_map.get("perception_pipeline"),
        ),
        _layer_output_packet(
            layer_id="state_condition_hypotheses",
            output_key="state_condition_hypotheses",
            output_kind="row_list",
            row_count=len(statecons),
            stage=stages.get("state_condition_hypotheses"),
            handoff=handoff_map.get("perception_pipeline"),
        ),
        _layer_output_packet(
            layer_id="geoband_hypotheses",
            output_key="geoband_hypotheses",
            output_kind="row_list",
            row_count=len(geobands),
            stage=stages.get("geoband_hypotheses"),
            handoff=handoff_map.get("perception_pipeline"),
        ),
        _layer_output_packet(
            layer_id="geoband_boundary_queues",
            output_key="boundary_queues",
            output_kind="row_list",
            row_count=len(boundary_queues),
            stage=stages.get("geoband_hypotheses"),
            handoff=handoff_map.get("perception_pipeline"),
        ),
        _layer_output_packet(
            layer_id="deltaband_gate",
            output_key="deltaband_gate",
            output_kind="packet",
            row_count=None,
            stage=stages.get("deltaband_observations"),
            handoff=handoff_map.get("deltaband_gate"),
        ),
        _layer_output_packet(
            layer_id="exploreband_budget",
            output_key="exploreband_budget",
            output_kind="packet",
            row_count=int(exploreband_budget.get("candidate_probe_count") or 0)
            if isinstance(exploreband_budget, dict)
            else 0,
            handoff=handoff_map.get("exploreband_budget"),
        ),
        _layer_output_packet(
            layer_id="action_semantics_ledger",
            output_key="action_semantics_ledger",
            output_kind="packet",
            row_count=int(action_semantics_ledger.get("row_count") or 0)
            if isinstance(action_semantics_ledger, dict)
            else 0,
            handoff=handoff_map.get("action_semantics_ledger"),
        ),
        _layer_output_packet(
            layer_id="tabula_rasa_notebook",
            output_key="tabula_rasa_notebook",
            output_kind="packet",
            row_count=int(tabula_rasa_notebook.get("note_count") or 0),
            handoff=handoff_map.get("tabula_rasa_notebook"),
        ),
        _layer_output_packet(
            layer_id="episode_memory",
            output_key="episode_memory",
            output_kind="packet",
            row_count=int(episode_memory.get("frontier_count") or 0),
            handoff=handoff_map.get("episode_memory"),
        ),
        _layer_output_packet(
            layer_id="debug_summary",
            output_key="debug_summary",
            output_kind="packet",
            row_count=None,
            handoff=handoff_map.get("debug_summary"),
        ),
        _layer_output_packet(
            layer_id="admission",
            output_key="admission",
            output_kind="packet",
            row_count=None,
            handoff=handoff_map.get("admission"),
            forced_status="computed_fail_closed_output",
        ),
    ]
    skipped = [row for row in outputs if row.get("skipped") is True]
    recovered = [row for row in outputs if row.get("recovered_error") is True]
    return {
        "schema": CONTRA_LAYER_OUTPUT_SET_SCHEMA,
        "status": "complete_with_skipped_outputs" if skipped else "complete",
        "output_count": len(outputs),
        "computed_output_count": sum(1 for row in outputs if row.get("computed") is True),
        "skipped_output_count": len(skipped),
        "recovered_output_count": len(recovered),
        "skipped_output_keys": [str(row.get("output_key")) for row in skipped],
        "recovered_output_keys": [str(row.get("output_key")) for row in recovered],
        "outputs": outputs,
    }
