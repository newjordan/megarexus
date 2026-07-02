#!/usr/bin/env python3
"""Unified CONTRA controller head.

External systems call this layer once. The controller head then runs the
vision-first CONTRA network and returns one packet with fail-closed admission
metadata.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .controller_affordances import (
    ACTOR_AFFORDANCE_HYPOTHESIS_SCHEMA,
    TARGET_AFFORDANCE_HYPOTHESIS_SCHEMA,
    calibration_actor_affordance_hypotheses,
    calibration_target_affordance_hypotheses,
    empty_actor_affordance_hypotheses,
    empty_target_affordance_hypotheses,
    oracle_actor_affordance_hypotheses,
    oracle_target_affordance_hypotheses,
)
from .controller_geometry import (
    as_raw_grid as _as_raw_grid,
)
from .controller_error_packets import (
    CONTRA_RECOVERED_ERROR_PACKET_SCHEMA,
    RAW_COLOR_COMPONENT_PACKET_SCHEMA,
    empty_raw_color_component_packet as _empty_raw_color_component_packet,
    empty_scene_surfaces as _empty_scene_surfaces,
    empty_symmetry_hypotheses as _empty_symmetry_hypotheses,
    recovered_error_packet as _recovered_error_packet,
)
from .controller_layer_outputs import (
    CONTRA_LAYER_OUTPUT_SCHEMA,
    CONTRA_LAYER_OUTPUT_SET_SCHEMA,
    build_layer_output_packets as _build_layer_output_packets,
)
from .action_semantics_ledger import build_action_semantics_ledger
from .debug import summarize_view
from .episode_memory import build_initial_episode_memory
from .exploreband import build_exploreband_budget
from .ghost import build_ghost_pass
from .handoff import handoff_set, layer_handoff
from .oversight import build_deltaband_controlled_event_requirement
from .perception import build_contra_perception_pipeline
from .relationship_ladder import build_contra_relationship_ladder
from .tabula_rasa import build_tabula_rasa_notebook
from .view_system import (
    CONTRA_CONFIG_PASS_SCHEMA,
    CONTRA_ORACLE_PASS_SCHEMA,
    CONTRA_PASS_SCHEMA,
    CONTRA_VIEW_SYSTEM_PASS_SET_SCHEMA,
    build_config_pass,
    build_contra_pass,
    build_oracle_pass,
    build_view_system_passes,
)
from .vision import (
    CONTROL_ISOLATION_SCHEMA,
    PALETTE_NAMES,
    build_control_isolation_packet,
    raw_pixel_components_with_labels,
    sha256_raw_grid,
)

CONTRA_CONTROLLER_REQUEST_SCHEMA = "codexlab.contra_controller_request.v1"
CONTRA_CONTROLLER_RESPONSE_SCHEMA = "codexlab.contra_controller_response.v1"

ActorBuilder = Callable[..., list[dict[str, Any]]]
TargetBuilder = Callable[..., list[dict[str, Any]]]


def _is_white_loading_screen(frame: np.ndarray) -> bool:
    return bool(frame.size > 0 and np.all(frame.astype(int) == 0))


@dataclass(frozen=True)
class ContraControllerRequest:
    game_id: str
    level_index: int
    mode: str = "review"
    min_component_area: int = 1
    connectivity: int = 4
    shape_connectivity: int = 4

    def to_packet(self, *, raw_frame_sha256: str, calibration: dict[str, Any] | None) -> dict[str, Any]:
        return {
            "schema": CONTRA_CONTROLLER_REQUEST_SCHEMA,
            "game_id": self.game_id,
            "level_index": int(self.level_index),
            "mode": self.mode,
            "raw_frame_sha256": raw_frame_sha256,
            "min_component_area": int(self.min_component_area),
            "connectivity": int(self.connectivity),
            "shape_connectivity": int(self.shape_connectivity),
            "calibration_supplied": isinstance(calibration, dict),
        }


def build_raw_color_component_packet(raw_frame: Any, *, min_component_area: int = 1) -> dict[str, Any]:
    arr = _as_raw_grid(raw_frame)
    values, counts = np.unique(arr.astype(int), return_counts=True)
    background_value = int(values[int(np.argmax(counts))]) if len(values) else None
    components, _label_map = raw_pixel_components_with_labels(arr)
    colors: list[dict[str, Any]] = []
    for value, count in sorted(zip(values, counts), key=lambda item: int(item[0])):
        raw_value = int(value)
        color_components = [
            {
                "component_id": str(component["component_id"]),
                "component_index_for_color": int(component["component_index_for_color"]),
                "raw_pixel_value": int(component["raw_pixel_value"]),
                "area_pixels": int(component["area_pixels"]),
                "bbox_xyxy": [int(v) for v in component["bbox_xyxy"]],
                "center_xy": [float(v) for v in component["center_xy"]],
            }
            for component in components
            if int(component["raw_pixel_value"]) == raw_value
            and int(component["area_pixels"]) >= int(min_component_area)
        ]
        colors.append(
            {
                "raw_pixel_value": raw_value,
                "palette_name": PALETTE_NAMES.get(raw_value, f"value_{raw_value}"),
                "pixel_count": int(count),
                "component_count": len(color_components),
                "dominant_background_candidate": background_value == raw_value,
                "components": color_components,
            }
        )
    return {
        "schema": RAW_COLOR_COMPONENT_PACKET_SCHEMA,
        "source": "raw_frame_exact_pixel_values_and_4_connected_components",
        "frame_raw_grid_sha256": sha256_raw_grid(arr),
        "frame_shape": [int(v) for v in arr.shape],
        "min_component_area_pixels": int(min_component_area),
        "background_selection_policy": "dominant_raw_pixel_value",
        "background_raw_pixel_value": background_value,
        "color_count": len(colors),
        "colors": colors,
    }


def _select_builders(
    *,
    calibration: dict[str, Any] | None,
    actor_builder: ActorBuilder | None,
    target_builder: TargetBuilder | None,
) -> tuple[ActorBuilder, TargetBuilder, str]:
    if actor_builder is not None and target_builder is not None:
        return actor_builder, target_builder, "caller_supplied_builders"
    if isinstance(calibration, dict):
        return (
            actor_builder or calibration_actor_affordance_hypotheses,
            target_builder or calibration_target_affordance_hypotheses,
            "contra_controller_calibration_adapter",
        )
    return (
        actor_builder or oracle_actor_affordance_hypotheses,
        target_builder or oracle_target_affordance_hypotheses,
        "contra_controller_oracle_first_pass_adapter",
    )


def _wrap_actor_builder(builder: ActorBuilder, handoffs: list[dict[str, Any]]) -> ActorBuilder:
    def wrapped(*, scene_surfaces: dict[str, Any], actor_expectation: dict[str, Any] | None) -> list[dict[str, Any]]:
        try:
            rows = builder(scene_surfaces=scene_surfaces, actor_expectation=actor_expectation)
            handoffs.append(
                layer_handoff(
                    layer_id="actor_affordance_builder",
                    status="complete" if rows else "complete_empty_fail_closed",
                    input_refs=["scene_surface_hypotheses", "actor_expectation"],
                    output_refs=["actor_affordance_hypotheses"],
                    next_layers=["support_action_affordances", "state_condition_hypotheses"],
                    reason_codes=[] if rows else ["no_actor_affordance_rows"],
                    counts={"actors": len(rows)},
                )
            )
            return rows
        except Exception as exc:  # noqa: BLE001 - API boundary converts layer failures into handoffs.
            handoffs.append(
                layer_handoff(
                    layer_id="actor_affordance_builder",
                    status="recovered_error_fail_closed",
                    input_refs=["scene_surface_hypotheses", "actor_expectation"],
                    output_refs=["actor_affordance_hypotheses"],
                    next_layers=["support_action_affordances", "state_condition_hypotheses"],
                    reason_codes=["actor_builder_exception"],
                    counts={"actors": 0},
                    error=exc,
                )
            )
            return []

    return wrapped


def _wrap_target_builder(builder: TargetBuilder, handoffs: list[dict[str, Any]]) -> TargetBuilder:
    def wrapped(
        *,
        frame_shape: tuple[int, int],
        label_map: np.ndarray,
        component_lookup: dict[str, int],
        scene_surfaces: dict[str, Any],
        bucket_expectation: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        try:
            rows = builder(
                frame_shape=frame_shape,
                label_map=label_map,
                component_lookup=component_lookup,
                scene_surfaces=scene_surfaces,
                bucket_expectation=bucket_expectation,
            )
            handoffs.append(
                layer_handoff(
                    layer_id="target_affordance_builder",
                    status="complete" if rows else "complete_empty_fail_closed",
                    input_refs=["scene_surface_hypotheses", "target_expectation"],
                    output_refs=["target_affordance_hypotheses"],
                    next_layers=["support_action_affordances", "state_condition_hypotheses"],
                    reason_codes=[] if rows else ["no_target_affordance_rows"],
                    counts={"targets": len(rows)},
                )
            )
            return rows
        except Exception as exc:  # noqa: BLE001 - API boundary converts layer failures into handoffs.
            handoffs.append(
                layer_handoff(
                    layer_id="target_affordance_builder",
                    status="recovered_error_fail_closed",
                    input_refs=["scene_surface_hypotheses", "target_expectation"],
                    output_refs=["target_affordance_hypotheses"],
                    next_layers=["support_action_affordances", "state_condition_hypotheses"],
                    reason_codes=["target_builder_exception"],
                    counts={"targets": 0},
                    error=exc,
                )
            )
            return []

    return wrapped


def _extract_geobands(statecons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row["geoband_hypothesis"]
        for row in statecons
        if isinstance(row, dict) and isinstance(row.get("geoband_hypothesis"), dict)
    ]


def _extract_boundary_queues(statecons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row["geoband_boundary_queue"]
        for row in statecons
        if isinstance(row, dict) and isinstance(row.get("geoband_boundary_queue"), dict)
    ]


def _fail_closed_reasons(
    *,
    relationship_ladder: dict[str, Any],
    debug_summary: dict[str, Any],
    deltaband_gate: dict[str, Any],
) -> list[str]:
    reasons = [str(value) for value in debug_summary.get("reason_codes", []) if value]
    for stage_id in relationship_ladder.get("incomplete_stage_ids", []):
        reasons.append(f"incomplete_stage:{stage_id}")
    if deltaband_gate.get("controlled_run_status") != "complete":
        reasons.append("deltaband_requires_controlled_before_after_action_boundary")
    for row in relationship_ladder.get("controller_handoff_reasons", []):
        if row:
            reasons.append(str(row))
    return sorted(set(reasons))


def _admission_summary(
    *,
    relationship_ladder: dict[str, Any],
    debug_summary: dict[str, Any],
    deltaband_gate: dict[str, Any],
) -> dict[str, Any]:
    fail_closed_reasons = _fail_closed_reasons(
        relationship_ladder=relationship_ladder,
        debug_summary=debug_summary,
        deltaband_gate=deltaband_gate,
    )
    return {
        "schema": "codexlab.contra_controller_admission_summary.v1",
        "status": "fail_closed_not_admitted",
        "admission_status": "not_admitted",
        "fail_closed": True,
        "fail_closed_reasons": fail_closed_reasons,
        "contra_ran": debug_summary.get("contra_ran") is True,
        "statecon_active": debug_summary.get("statecon_active") is True,
        "geoband_reached": debug_summary.get("geoband_reached") is True,
        "geoband_active": debug_summary.get("geoband_active") is True,
        "relationship_ladder_complete": relationship_ladder.get("completeness_status") == "complete",
        "deltaband_status": deltaband_gate.get("status"),
        "prediction_status": "not_predictive",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }


def run_contra_view(
    *,
    raw_frame: Any,
    game_id: str,
    level_index: int,
    calibration: dict[str, Any] | None = None,
    mode: str = "review",
    ghost_context: dict[str, Any] | None = None,
    actor_builder: ActorBuilder | None = None,
    target_builder: TargetBuilder | None = None,
    min_component_area: int = 1,
    connectivity: int = 4,
    shape_connectivity: int = 4,
) -> dict[str, Any]:
    handoffs: list[dict[str, Any]] = []
    try:
        frame = _as_raw_grid(raw_frame)
        raw_frame_sha256 = sha256_raw_grid(frame)
        handoffs.append(
            layer_handoff(
                layer_id="raw_frame_validation",
                status="complete",
                input_refs=["raw_frame"],
                output_refs=["raw_frame_sha256"],
                next_layers=["raw_color_component_packet", "control_isolation_packet"],
                counts={"frame_height": int(frame.shape[0]), "frame_width": int(frame.shape[1])},
            )
        )
    except Exception as exc:  # noqa: BLE001 - public API must hand off invalid input cleanly.
        request = ContraControllerRequest(
            game_id=str(game_id),
            level_index=int(level_index),
            mode=str(mode),
            min_component_area=int(min_component_area),
            connectivity=int(connectivity),
            shape_connectivity=int(shape_connectivity),
        )
        handoffs.append(
            layer_handoff(
                layer_id="raw_frame_validation",
                status="recovered_error_fail_closed",
                input_refs=["raw_frame"],
                output_refs=[],
                next_layers=[],
                reason_codes=["invalid_raw_frame"],
                recoverable=True,
                error=exc,
            )
        )
        admission = {
            "schema": "codexlab.contra_controller_admission_summary.v1",
            "status": "fail_closed_not_admitted",
            "admission_status": "not_admitted",
            "fail_closed": True,
            "fail_closed_reasons": ["invalid_raw_frame"],
            "contra_ran": False,
            "statecon_active": False,
            "geoband_reached": False,
            "geoband_active": False,
            "relationship_ladder_complete": False,
            "deltaband_status": "not_run_invalid_raw_frame",
            "prediction_status": "not_predictive",
            "control_truth_admitted": False,
            "goal_condition_admitted": False,
            "action_choice_admitted": False,
            "solve_condition_admitted": False,
        }
        raw_packet = _empty_raw_color_component_packet(frame_shape=None, frame_sha256=None, error=exc)
        scene_surfaces = _empty_scene_surfaces(None, error=exc)
        symmetry_hypotheses = _empty_symmetry_hypotheses(None, error=exc)
        relationship_ladder = build_contra_relationship_ladder(
            level_index=int(level_index),
            source_raw_frame_sha256="invalid_raw_frame",
            raw_color_inventory=raw_packet,
            scene_surfaces=scene_surfaces,
            symmetry_hypotheses=symmetry_hypotheses,
            actor_affordances=[],
            target_affordances=[],
            support_affordances=[],
            statecons=[],
        )
        relationship_ladder["status"] = "not_run_invalid_raw_frame"
        deltaband_gate = _recovered_error_packet(stage="deltaband_gate", error=exc)
        debug_summary = summarize_view(
            {
                "game_id": str(game_id),
                "game_short": str(game_id).split("-")[0],
                "title": str(game_id),
                "level_index": int(level_index),
                "source_raw_frame_sha256": None,
                "colors": [],
                "scene_surface_hypotheses": scene_surfaces,
                "symmetry_hypotheses": symmetry_hypotheses,
                "actor_affordance_hypotheses": [],
                "target_affordance_hypotheses": [],
                "support_action_affordance_hypotheses": [],
                "state_condition_hypotheses": [],
                "contra_relationship_ladder": relationship_ladder,
            }
        )
        debug_summary["status"] = "not_run_invalid_raw_frame"
        debug_summary["reason_codes"] = sorted(set([*debug_summary.get("reason_codes", []), "invalid_raw_frame"]))
        tabula_rasa_notebook = build_tabula_rasa_notebook(
            game_id=str(game_id),
            level_index=int(level_index),
            source_raw_frame_sha256="invalid_raw_frame",
            actor_affordances=[],
            target_affordances=[],
            support_affordances=[],
            statecons=[],
            round_index=0,
        )
        episode_memory = build_initial_episode_memory(
            game_id=str(game_id),
            level_index=int(level_index),
            contra_packet={
                "packet_sha256_basis": "invalid_raw_frame",
                "state_condition_hypotheses": [],
                "tabula_rasa_notebook": tabula_rasa_notebook,
            },
        )
        exploreband_budget = build_exploreband_budget(
            level_index=int(level_index),
            source_raw_frame_sha256="invalid_raw_frame",
            macroband_object_ranking=None,
            stage0_init_calibration_requirement=None,
            previous_ledger=None,
            budget_config=None,
        )
        action_semantics_ledger = build_action_semantics_ledger(
            game_id=str(game_id),
            level_index=int(level_index),
            source_raw_frame_sha256="invalid_raw_frame",
            actor_affordances=[],
            target_affordances=[],
            support_affordances=[],
            statecons=[],
        )
        layer_output_packets = _build_layer_output_packets(
            raw_packet=raw_packet,
            control_isolation_packet=_recovered_error_packet(stage="control_isolation_packet", error=exc),
            scene_surfaces=scene_surfaces,
            symmetry_hypotheses=symmetry_hypotheses,
            actor_affordances=[],
            target_affordances=[],
            support_affordances=[],
            statecons=[],
            geobands=[],
            boundary_queues=[],
            relationship_ladder=relationship_ladder,
            deltaband_gate=deltaband_gate,
            exploreband_budget=exploreband_budget,
            action_semantics_ledger=action_semantics_ledger,
            tabula_rasa_notebook=tabula_rasa_notebook,
            episode_memory=episode_memory,
            debug_summary=debug_summary,
            admission=admission,
            handoffs=handoffs,
        )
        config_pass = build_config_pass(
            game_id=str(game_id),
            level_index=int(level_index),
            mode=str(mode),
            raw_frame_sha256=None,
            frame_shape=None,
            calibration=calibration,
            invalid_input=True,
        )
        oracle_pass = build_oracle_pass(
            level_index=int(level_index),
            raw_frame_sha256=None,
            scene_surfaces=None,
            frame_shape=None,
            label_map=None,
            component_lookup=None,
            invalid_input=True,
        )
        ghost_pass = build_ghost_pass(
            game_id=str(game_id),
            level_index=int(level_index),
            raw_frame_sha256=None,
            ghost_context=ghost_context,
            invalid_input=True,
        )
        contra_pass = build_contra_pass(
            level_index=int(level_index),
            raw_frame_sha256=None,
            layer_output_packets=layer_output_packets,
            relationship_ladder=relationship_ladder,
            deltaband_gate=deltaband_gate,
            admission=admission,
            config_pass=config_pass,
            oracle_pass=oracle_pass,
            ghost_pass=ghost_pass,
        )
        view_system_passes = build_view_system_passes(
            config_pass=config_pass,
            oracle_pass=oracle_pass,
            ghost_pass=ghost_pass,
            contra_pass=contra_pass,
        )
        return {
            "schema": CONTRA_CONTROLLER_RESPONSE_SCHEMA,
            "status": "invalid_input_fail_closed_handoff",
            "source": "contra_controller_head_run_contra_view",
            "game_id": str(game_id),
            "level_index": int(level_index),
            "mode": str(mode),
            "source_raw_frame_sha256": None,
            "request": request.to_packet(raw_frame_sha256="invalid_raw_frame", calibration=calibration),
            "builder_source": "not_selected_invalid_raw_frame",
            "config_pass": config_pass,
            "oracle_pass": oracle_pass,
            "ghost_pass": ghost_pass,
            "contra_pass": contra_pass,
            "view_system_passes": view_system_passes,
            "raw_color_component_packet": raw_packet,
            "control_isolation_packet_schema": CONTROL_ISOLATION_SCHEMA,
            "control_isolation_packet": _recovered_error_packet(stage="control_isolation_packet", error=exc),
            "scene_surfaces": scene_surfaces,
            "symmetry_hypotheses": symmetry_hypotheses,
            "actor_affordance_hypotheses": [],
            "target_affordance_hypotheses": [],
            "support_action_affordance_hypotheses": [],
            "state_condition_hypotheses": [],
            "geoband_hypotheses": [],
            "boundary_queues": [],
            "contra_relationship_ladder": relationship_ladder,
            "deltaband_gate": deltaband_gate,
            "exploreband_budget": exploreband_budget,
            "action_semantics_ledger": action_semantics_ledger,
            "tabula_rasa_notebook": tabula_rasa_notebook,
            "episode_memory": episode_memory,
            "debug_summary": debug_summary,
            "layer_output_packets": layer_output_packets,
            "layer_handoffs": handoff_set(handoffs),
            "admission": admission,
            "admission_status": admission["admission_status"],
            "fail_closed_reasons": admission["fail_closed_reasons"],
            "not_prediction": True,
            "not_control_truth": True,
            "not_solve_truth": True,
            "control_truth_admitted": False,
            "goal_condition_admitted": False,
            "action_choice_admitted": False,
            "solve_condition_admitted": False,
        }
    request = ContraControllerRequest(
        game_id=str(game_id),
        level_index=int(level_index),
        mode=str(mode),
        min_component_area=int(min_component_area),
        connectivity=int(connectivity),
        shape_connectivity=int(shape_connectivity),
    )
    try:
        raw_packet = build_raw_color_component_packet(frame, min_component_area=min_component_area)
        handoffs.append(
            layer_handoff(
                layer_id="raw_color_component_packet",
                status="complete",
                input_refs=["raw_frame"],
                output_refs=["raw_color_component_packet"],
                next_layers=["control_isolation_packet", "perception_pipeline"],
                counts={
                    "colors": len(raw_packet.get("colors", [])),
                    "components": sum(len(color.get("components", [])) for color in raw_packet.get("colors", [])),
                },
            )
        )
    except Exception as exc:  # noqa: BLE001
        raw_packet = _empty_raw_color_component_packet(
            frame_shape=[int(v) for v in frame.shape],
            frame_sha256=raw_frame_sha256,
            error=exc,
        )
        handoffs.append(
            layer_handoff(
                layer_id="raw_color_component_packet",
                status="recovered_error_fail_closed",
                input_refs=["raw_frame"],
                output_refs=["raw_color_component_packet"],
                next_layers=["control_isolation_packet", "perception_pipeline"],
                reason_codes=["raw_color_component_packet_exception"],
                counts={"colors": 0, "components": 0},
                error=exc,
            )
        )
    try:
        control_isolation_packet = build_control_isolation_packet(
            frame,
            game=str(game_id),
            phase=str(mode),
            case_id=f"{game_id}_level_{int(level_index):02d}",
            background_raw_pixel_value=raw_packet.get("background_raw_pixel_value"),
            connectivity=int(connectivity),
            min_area_pixels=int(min_component_area),
            shape_connectivity=int(shape_connectivity),
        )
        handoffs.append(
            layer_handoff(
                layer_id="control_isolation_packet",
                status="complete",
                input_refs=["raw_frame", "raw_color_component_packet"],
                output_refs=["control_isolation_packet"],
                next_layers=["perception_pipeline"],
                counts={
                    "shape_traces": int(control_isolation_packet.get("shape_trace_count") or 0),
                    "color_groups": int(control_isolation_packet.get("color_group_count") or 0),
                },
            )
        )
    except Exception as exc:  # noqa: BLE001
        control_isolation_packet = _recovered_error_packet(stage="control_isolation_packet", error=exc)
        handoffs.append(
            layer_handoff(
                layer_id="control_isolation_packet",
                status="recovered_error_fail_closed",
                input_refs=["raw_frame", "raw_color_component_packet"],
                output_refs=["control_isolation_packet"],
                next_layers=["perception_pipeline"],
                reason_codes=["control_isolation_packet_exception"],
                error=exc,
            )
        )
    config_pass = build_config_pass(
        game_id=str(game_id),
        level_index=int(level_index),
        mode=str(mode),
        raw_frame_sha256=raw_frame_sha256,
        frame_shape=[int(v) for v in frame.shape[:2]],
        calibration=calibration,
        first_loading_screen_detected=_is_white_loading_screen(frame),
    )
    selected_actor_builder, selected_target_builder, builder_source = _select_builders(
        calibration=calibration,
        actor_builder=actor_builder,
        target_builder=target_builder,
    )
    try:
        perception = build_contra_perception_pipeline(
            frame=frame,
            level_index=int(level_index),
            source_raw_frame_sha256=raw_frame_sha256,
            raw_color_inventory=raw_packet,
            calibration=calibration or {},
            actor_builder=_wrap_actor_builder(selected_actor_builder, handoffs),
            target_builder=_wrap_target_builder(selected_target_builder, handoffs),
        )
        relationship_ladder = perception["contra_relationship_ladder"]
        handoffs.append(
            layer_handoff(
                layer_id="perception_pipeline",
                status="complete",
                input_refs=["raw_color_component_packet", "control_isolation_packet"],
                output_refs=[
                    "scene_surfaces",
                    "symmetry_hypotheses",
                    "support_action_affordance_hypotheses",
                    "state_condition_hypotheses",
                    "contra_relationship_ladder",
                ],
                next_layers=["deltaband_gate", "debug_summary"],
                counts={
                    "symmetry_groups": perception.get("symmetry_hypotheses", {}).get("symmetry_relationship_group_count", 0)
                    if isinstance(perception.get("symmetry_hypotheses"), dict)
                    else 0,
                    "supports": len(perception.get("support_action_affordance_hypotheses", [])),
                    "statecons": len(perception.get("state_condition_hypotheses", [])),
                },
            )
        )
    except Exception as exc:  # noqa: BLE001
        scene_surfaces = _empty_scene_surfaces(frame, error=exc)
        symmetry_hypotheses = _empty_symmetry_hypotheses(frame, error=exc)
        relationship_ladder = build_contra_relationship_ladder(
            level_index=int(level_index),
            source_raw_frame_sha256=raw_frame_sha256,
            raw_color_inventory=raw_packet,
            scene_surfaces=scene_surfaces,
            symmetry_hypotheses=symmetry_hypotheses,
            actor_affordances=[],
            target_affordances=[],
            support_affordances=[],
            statecons=[],
        )
        relationship_ladder["controller_handoff_reasons"] = ["perception_pipeline_exception"]
        perception = {
            "schema": "codexlab.contra_perception_pipeline.v1",
            "status": "recovered_error_fail_closed",
            "scene_surface_hypotheses": scene_surfaces,
            "symmetry_hypotheses": symmetry_hypotheses,
            "actor_affordance_hypotheses": [],
            "target_affordance_hypotheses": [],
            "support_action_affordance_hypotheses": [],
            "state_condition_hypotheses": [],
            "contra_relationship_ladder": relationship_ladder,
        }
        handoffs.append(
            layer_handoff(
                layer_id="perception_pipeline",
                status="recovered_error_fail_closed",
                input_refs=["raw_color_component_packet", "control_isolation_packet"],
                output_refs=["empty_scene_surfaces", "contra_relationship_ladder"],
                next_layers=["deltaband_gate", "debug_summary"],
                reason_codes=["perception_pipeline_exception"],
                counts={"supports": 0, "statecons": 0},
                error=exc,
            )
        )
    try:
        deltaband_gate = build_deltaband_controlled_event_requirement(
            level_index=int(level_index),
            source_raw_frame_sha256=raw_frame_sha256,
            relationship_ladder=relationship_ladder,
            raw_color_component_packet=raw_packet,
            scene_surfaces=perception["scene_surface_hypotheses"],
            actor_affordances=perception["actor_affordance_hypotheses"],
            target_affordances=perception["target_affordance_hypotheses"],
            support_affordances=perception["support_action_affordance_hypotheses"],
            statecons=perception["state_condition_hypotheses"],
            symmetry_hypotheses=perception.get("symmetry_hypotheses", _empty_symmetry_hypotheses(frame)),
            prior_observations=(calibration or {}).get("match_action_result_prior_observations")
            if isinstance(calibration, dict)
            else None,
            exploration_budget_config=(calibration or {}).get("exploration_budget_config")
            if isinstance(calibration, dict)
            else None,
            exploration_budget_ledger=(calibration or {}).get("exploration_budget_ledger")
            if isinstance(calibration, dict)
            else None,
        )
        handoffs.append(
            layer_handoff(
                layer_id="deltaband_gate",
                status="complete_fail_closed_waiting_for_controlled_boundary",
                input_refs=["contra_relationship_ladder"],
                output_refs=["deltaband_gate"],
                next_layers=["exploreband_budget", "debug_summary", "admission"],
                reason_codes=["controlled_before_after_action_boundary_required"],
                counts={"missing_artifacts": len(deltaband_gate.get("missing_artifacts", []))},
            )
        )
    except Exception as exc:  # noqa: BLE001
        deltaband_gate = _recovered_error_packet(stage="deltaband_gate", error=exc)
        handoffs.append(
            layer_handoff(
                layer_id="deltaband_gate",
                status="recovered_error_fail_closed",
                input_refs=["contra_relationship_ladder"],
                output_refs=["deltaband_gate"],
                next_layers=["exploreband_budget", "debug_summary", "admission"],
                reason_codes=["deltaband_gate_exception"],
                error=exc,
            )
        )
    exploreband_budget = (
        deltaband_gate.get("exploreband_budget")
        if isinstance(deltaband_gate, dict) and isinstance(deltaband_gate.get("exploreband_budget"), dict)
        else build_exploreband_budget(
            level_index=int(level_index),
            source_raw_frame_sha256=raw_frame_sha256,
            macroband_object_ranking=None,
            stage0_init_calibration_requirement=None,
            previous_ledger=(calibration or {}).get("exploration_budget_ledger") if isinstance(calibration, dict) else None,
            budget_config=(calibration or {}).get("exploration_budget_config") if isinstance(calibration, dict) else None,
        )
    )
    handoffs.append(
        layer_handoff(
            layer_id="exploreband_budget",
            status=str(exploreband_budget.get("status") or "not_ready"),
            input_refs=["deltaband_gate", "macroband_object_ranking", "stage0_init_calibration_requirement"],
            output_refs=["exploreband_budget"],
            next_layers=["oracle_controller", "executor", "debug_summary"],
            reason_codes=["portband_observation_only_budgeted_probe_queue"],
            counts={
                "candidate_probe_count": int(exploreband_budget.get("candidate_probe_count") or 0),
                "remaining_probe_count": int((exploreband_budget.get("budget_policy") or {}).get("remaining_probe_count") or 0)
                if isinstance(exploreband_budget.get("budget_policy"), dict)
                else 0,
            },
        )
    )
    statecons = perception["state_condition_hypotheses"]
    view_for_debug = {
        "game_id": str(game_id),
        "game_short": str(game_id).split("-")[0],
        "title": str(game_id),
        "level_index": int(level_index),
        "source_raw_frame_sha256": raw_frame_sha256,
        "colors": raw_packet["colors"],
        "scene_surface_hypotheses": perception["scene_surface_hypotheses"],
        "symmetry_hypotheses": perception.get("symmetry_hypotheses", _empty_symmetry_hypotheses(frame)),
        "actor_affordance_hypotheses": perception["actor_affordance_hypotheses"],
        "target_affordance_hypotheses": perception["target_affordance_hypotheses"],
        "support_action_affordance_hypotheses": perception["support_action_affordance_hypotheses"],
        "state_condition_hypotheses": statecons,
        "contra_relationship_ladder": relationship_ladder,
    }
    try:
        debug_summary = summarize_view(view_for_debug)
        handoffs.append(
            layer_handoff(
                layer_id="debug_summary",
                status="complete",
                input_refs=["contra_view"],
                output_refs=["debug_summary"],
                next_layers=["admission"],
                counts=debug_summary.get("counts") if isinstance(debug_summary.get("counts"), dict) else {},
            )
        )
    except Exception as exc:  # noqa: BLE001
        debug_summary = {
            "schema": "codexlab.contra_debug_view_summary.v1",
            "status": "recovered_error_fail_closed",
            "contra_ran": False,
            "statecon_active": False,
            "geoband_reached": False,
            "geoband_active": False,
            "reason_codes": ["debug_summary_exception"],
            "counts": {},
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        handoffs.append(
            layer_handoff(
                layer_id="debug_summary",
                status="recovered_error_fail_closed",
                input_refs=["contra_view"],
                output_refs=["debug_summary"],
                next_layers=["admission"],
                reason_codes=["debug_summary_exception"],
                error=exc,
            )
        )
    admission = _admission_summary(
        relationship_ladder=relationship_ladder,
        debug_summary=debug_summary,
        deltaband_gate=deltaband_gate,
    )
    tabula_rasa_notebook = build_tabula_rasa_notebook(
        game_id=str(game_id),
        level_index=int(level_index),
        source_raw_frame_sha256=raw_frame_sha256,
        actor_affordances=perception["actor_affordance_hypotheses"],
        target_affordances=perception["target_affordance_hypotheses"],
        support_affordances=perception["support_action_affordance_hypotheses"],
        statecons=statecons,
        round_index=0,
    )
    packet_sha256_basis = hashlib.sha256(
        f"{game_id}|{level_index}|{mode}|{raw_frame_sha256}".encode("utf-8")
    ).hexdigest()
    episode_memory = build_initial_episode_memory(
        game_id=str(game_id),
        level_index=int(level_index),
        contra_packet={
            "packet_sha256_basis": packet_sha256_basis,
            "state_condition_hypotheses": statecons,
            "tabula_rasa_notebook": tabula_rasa_notebook,
            "exploreband_budget": exploreband_budget,
        },
    )
    action_semantics_ledger = build_action_semantics_ledger(
        game_id=str(game_id),
        level_index=int(level_index),
        source_raw_frame_sha256=raw_frame_sha256,
        actor_affordances=perception["actor_affordance_hypotheses"],
        target_affordances=perception["target_affordance_hypotheses"],
        support_affordances=perception["support_action_affordance_hypotheses"],
        statecons=statecons,
    )
    handoffs.append(
        layer_handoff(
            layer_id="tabula_rasa_notebook",
            status="complete_not_admitted",
            input_refs=["actor_affordance_hypotheses", "target_affordance_hypotheses", "support_action_affordance_hypotheses", "state_condition_hypotheses"],
            output_refs=["tabula_rasa_notebook"],
            next_layers=[],
            reason_codes=["llm_working_memory_not_evidence"],
            counts={"note_count": int(tabula_rasa_notebook.get("note_count") or 0)},
        )
    )
    handoffs.append(
        layer_handoff(
            layer_id="episode_memory",
            status="initialized_not_admitted",
            input_refs=["packet_sha256_basis", "state_condition_hypotheses", "tabula_rasa_notebook"],
            output_refs=["episode_memory"],
            next_layers=[],
            reason_codes=["episode_memory_working_state_not_truth"],
            counts={"frontier_count": int(episode_memory.get("frontier_count") or 0)},
        )
    )
    handoffs.append(
        layer_handoff(
            layer_id="action_semantics_ledger",
            status="complete_not_admitted",
            input_refs=[
                "actor_affordance_hypotheses",
                "target_affordance_hypotheses",
                "support_action_affordance_hypotheses",
                "state_condition_hypotheses",
            ],
            output_refs=["action_semantics_ledger"],
            next_layers=["actionmodband_controlled_probe"],
            reason_codes=["static_teacher_student_signal_not_control_truth"],
            counts={"rows": int(action_semantics_ledger.get("row_count") or 0)},
        )
    )
    handoffs.append(
        layer_handoff(
            layer_id="admission",
            status="complete_fail_closed",
            input_refs=["debug_summary", "contra_relationship_ladder", "deltaband_gate"],
            output_refs=["admission"],
            next_layers=[],
            reason_codes=admission["fail_closed_reasons"],
            counts={"fail_closed_reasons": len(admission["fail_closed_reasons"])},
        )
    )
    geobands = _extract_geobands(statecons)
    boundary_queues = _extract_boundary_queues(statecons)
    layer_output_packets = _build_layer_output_packets(
        raw_packet=raw_packet,
        control_isolation_packet=control_isolation_packet,
        scene_surfaces=perception["scene_surface_hypotheses"],
        symmetry_hypotheses=perception.get("symmetry_hypotheses", _empty_symmetry_hypotheses(frame)),
        actor_affordances=perception["actor_affordance_hypotheses"],
        target_affordances=perception["target_affordance_hypotheses"],
        support_affordances=perception["support_action_affordance_hypotheses"],
        statecons=statecons,
        geobands=geobands,
        boundary_queues=boundary_queues,
        relationship_ladder=relationship_ladder,
        deltaband_gate=deltaband_gate,
        exploreband_budget=exploreband_budget,
        action_semantics_ledger=action_semantics_ledger,
        tabula_rasa_notebook=tabula_rasa_notebook,
        episode_memory=episode_memory,
        debug_summary=debug_summary,
        admission=admission,
        handoffs=handoffs,
    )
    oracle_pass = build_oracle_pass(
        level_index=int(level_index),
        raw_frame_sha256=raw_frame_sha256,
        scene_surfaces=perception["scene_surface_hypotheses"],
        frame_shape=tuple(int(v) for v in frame.shape[:2]),
        label_map=perception.get("component_label_map"),
        component_lookup=perception.get("component_lookup") if isinstance(perception.get("component_lookup"), dict) else None,
    )
    ghost_pass = build_ghost_pass(
        game_id=str(game_id),
        level_index=int(level_index),
        raw_frame_sha256=raw_frame_sha256,
        ghost_context=ghost_context,
    )
    contra_pass = build_contra_pass(
        level_index=int(level_index),
        raw_frame_sha256=raw_frame_sha256,
        layer_output_packets=layer_output_packets,
        relationship_ladder=relationship_ladder,
        deltaband_gate=deltaband_gate,
        admission=admission,
        config_pass=config_pass,
        oracle_pass=oracle_pass,
        ghost_pass=ghost_pass,
    )
    view_system_passes = build_view_system_passes(
        config_pass=config_pass,
        oracle_pass=oracle_pass,
        ghost_pass=ghost_pass,
        contra_pass=contra_pass,
    )
    return {
        "schema": CONTRA_CONTROLLER_RESPONSE_SCHEMA,
        "status": "contra_view_hypothesis_packet",
        "source": "contra_controller_head_run_contra_view",
        "game_id": str(game_id),
        "level_index": int(level_index),
        "mode": str(mode),
        "source_raw_frame_sha256": raw_frame_sha256,
        "packet_sha256_basis": packet_sha256_basis,
        "request": request.to_packet(raw_frame_sha256=raw_frame_sha256, calibration=calibration),
        "builder_source": builder_source,
        "config_pass": config_pass,
        "oracle_pass": oracle_pass,
        "ghost_pass": ghost_pass,
        "contra_pass": contra_pass,
        "view_system_passes": view_system_passes,
        "raw_color_component_packet": raw_packet,
        "control_isolation_packet_schema": CONTROL_ISOLATION_SCHEMA,
        "control_isolation_packet": control_isolation_packet,
        "scene_surfaces": perception["scene_surface_hypotheses"],
        "symmetry_hypotheses": perception.get("symmetry_hypotheses", _empty_symmetry_hypotheses(frame)),
        "actor_affordance_hypotheses": perception["actor_affordance_hypotheses"],
        "target_affordance_hypotheses": perception["target_affordance_hypotheses"],
        "support_action_affordance_hypotheses": perception["support_action_affordance_hypotheses"],
        "state_condition_hypotheses": statecons,
        "geoband_hypotheses": geobands,
        "boundary_queues": boundary_queues,
        "contra_relationship_ladder": relationship_ladder,
        "deltaband_gate": deltaband_gate,
        "exploreband_budget": exploreband_budget,
        "action_semantics_ledger": action_semantics_ledger,
        "tabula_rasa_notebook": tabula_rasa_notebook,
        "episode_memory": episode_memory,
        "debug_summary": debug_summary,
        "layer_output_packets": layer_output_packets,
        "layer_handoffs": handoff_set(handoffs),
        "admission": admission,
        "admission_status": admission["admission_status"],
        "fail_closed_reasons": admission["fail_closed_reasons"],
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }
