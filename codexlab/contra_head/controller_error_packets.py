"""Recovered-error and empty packet factories for the CONTRA controller."""

from __future__ import annotations

from typing import Any

import numpy as np

from .symmetry import CONTRA_SYMMETRY_PACKET_SCHEMA

RAW_COLOR_COMPONENT_PACKET_SCHEMA = "codexlab.raw_frame_color_component_packet.v1"
CONTRA_RECOVERED_ERROR_PACKET_SCHEMA = "codexlab.contra_recovered_error_packet.v1"


def empty_raw_color_component_packet(
    *,
    frame_shape: list[int] | None,
    frame_sha256: str | None,
    error: BaseException | None,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "schema": RAW_COLOR_COMPONENT_PACKET_SCHEMA,
        "status": "recovered_error_empty_raw_color_packet",
        "source": "contra_controller_recovered_error_handoff",
        "frame_raw_grid_sha256": frame_sha256,
        "frame_shape": frame_shape or [],
        "min_component_area_pixels": 1,
        "background_selection_policy": "not_available",
        "background_raw_pixel_value": None,
        "color_count": 0,
        "colors": [],
    }
    if error is not None:
        packet["error_type"] = type(error).__name__
        packet["error_message"] = str(error)
    return packet


def empty_scene_surfaces(frame: np.ndarray | None, *, error: BaseException | None = None) -> dict[str, Any]:
    frame_shape = [int(v) for v in frame.shape[:2]] if frame is not None else []
    packet: dict[str, Any] = {
        "schema": "codexlab.scene_surface_hypotheses.v3",
        "status": "recovered_error_empty_scene_surfaces" if error is not None else "empty_scene_surfaces",
        "source": "contra_controller_recovered_error_handoff",
        "frame_shape": frame_shape,
        "component_graph": {
            "schema": "codexlab.raw_frame_color_component_graph.v1",
            "components": [],
            "contacts": [],
            "contact_pair_count": 0,
        },
        "floor_hypothesis": None,
        "floor_candidates": [],
        "wall_hypotheses": [],
        "wall_candidates": [],
        "background_hypotheses": [],
        "background_candidates": [],
        "residual_component_hypotheses": [],
        "residual_component_hypothesis_count": 0,
        "antisymmetric_relationship_hypotheses": [],
        "antisymmetric_relationship_hypothesis_count": 0,
    }
    if error is not None:
        packet["error_type"] = type(error).__name__
        packet["error_message"] = str(error)
    return packet


def empty_symmetry_hypotheses(frame: np.ndarray | None, *, error: BaseException | None = None) -> dict[str, Any]:
    pass_rows = [
        {
            "schema": "codexlab.contra_symmetry_pass.v1",
            "status": "not_run",
            "pass_order": order,
            "pass_id": pass_id,
            "definition": definition,
            "output_group_ids": [],
            "output_group_count": 0,
            "admission_status": "not_admitted",
            "not_prediction": True,
            "not_control_truth": True,
            "not_solve_truth": True,
            "control_truth_admitted": False,
            "goal_condition_admitted": False,
            "action_choice_admitted": False,
            "solve_condition_admitted": False,
        }
        for order, pass_id, definition in (
            (1, "surface_context_pass", "measure background/floor/wall reflection as context, not as mirror-object grouping"),
            (2, "center_divider_pass", "detect non-floor centered divider geometry that may split mirrored play zones"),
            (3, "playzone_mirror_pass", "group left and right non-floor play zones around detected center dividers"),
            (4, "component_self_symmetry_pass", "group non-floor self-symmetric component masks"),
            (5, "component_mirror_pair_pass", "group non-floor mirrored component-pair potentials, allowing color mismatch"),
            (6, "component_reverse_symmetry_pass", "group non-floor mirrored components with reversed fill or negative-space structure"),
            (7, "component_polar_symmetry_pass", "promote opposing-handed mirrored component pairs as polar symmetry clues for gameplay review"),
            (8, "symmetry_sensor_layer_pass", "derive static lock/key-style review sensors from polar symmetry groups"),
        )
    ]
    packet: dict[str, Any] = {
        "schema": CONTRA_SYMMETRY_PACKET_SCHEMA,
        "status": "recovered_error_empty_symmetry_hypotheses" if error is not None else "empty_symmetry_hypotheses",
        "definition": "static_reflection_geometry_clues_for_review_only",
        "frame_shape_hw": [int(v) for v in frame.shape[:2]] if frame is not None else [],
        "floor_component_ids_excluded_from_mirror_potential_groups": [],
        "symmetry_passes": pass_rows,
        "symmetry_pass_count": len(pass_rows),
        "surface_symmetry_hypotheses": [],
        "center_divider_hypotheses": [],
        "playzone_mirror_hypotheses": [],
        "component_self_symmetry_hypotheses": [],
        "component_mirror_pair_hypotheses": [],
        "component_reverse_symmetry_hypotheses": [],
        "component_polar_symmetry_hypotheses": [],
        "symmetry_sensor_hypotheses": [],
        "symmetry_sensor_count": 0,
        "lock_key_sensor_hypotheses": [],
        "lock_key_sensor_count": 0,
        "symmetry_relationship_groups": [],
        "symmetry_relationship_group_count": 0,
        "non_floor_symmetry_relationship_group_count": 0,
        "mirror_scenario_clue": False,
        "evidence_status": "not_run",
        "admission_status": "not_admitted",
        "not_prediction": True,
        "not_control_truth": True,
        "not_solve_truth": True,
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
        "provenance": "contra_controller_recovered_error_handoff",
    }
    if error is not None:
        packet["error_type"] = type(error).__name__
        packet["error_message"] = str(error)
    return packet


def recovered_error_packet(*, stage: str, error: BaseException) -> dict[str, Any]:
    return {
        "schema": CONTRA_RECOVERED_ERROR_PACKET_SCHEMA,
        "status": "recovered_error_fail_closed",
        "stage": str(stage),
        "error_type": type(error).__name__,
        "error_message": str(error),
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }
