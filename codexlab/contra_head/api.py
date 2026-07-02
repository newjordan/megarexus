#!/usr/bin/env python3
"""Stable programmatic CONTRA API."""

from __future__ import annotations

from .controller_head import (
    CONTRA_CONFIG_PASS_SCHEMA,
    CONTRA_CONTROLLER_REQUEST_SCHEMA,
    CONTRA_CONTROLLER_RESPONSE_SCHEMA,
    CONTRA_LAYER_OUTPUT_SCHEMA,
    CONTRA_LAYER_OUTPUT_SET_SCHEMA,
    CONTRA_ORACLE_PASS_SCHEMA,
    CONTRA_PASS_SCHEMA,
    CONTRA_VIEW_SYSTEM_PASS_SET_SCHEMA,
    RAW_COLOR_COMPONENT_PACKET_SCHEMA,
    build_raw_color_component_packet,
    calibration_actor_affordance_hypotheses,
    calibration_target_affordance_hypotheses,
    empty_actor_affordance_hypotheses,
    empty_target_affordance_hypotheses,
    run_contra_view,
)
from .ghost import CONTRA_GHOST_PASS_SCHEMA, CONTRA_GHOST_SCHEMA
from .exploreband import EXPLOREBAND_SCHEMA, build_exploreband_budget
from .handoff import CONTRA_LAYER_HANDOFF_SCHEMA, CONTRA_LAYER_HANDOFF_SET_SCHEMA
from .symmetry import (
    CONTRA_COMPONENT_POLAR_SYMMETRY_SCHEMA,
    CONTRA_SYMMETRY_LOCK_KEY_SENSOR_SCHEMA,
    CONTRA_SYMMETRY_PACKET_SCHEMA,
    build_symmetry_hypotheses,
)

__all__ = [
    "CONTRA_CONTROLLER_REQUEST_SCHEMA",
    "CONTRA_CONTROLLER_RESPONSE_SCHEMA",
    "CONTRA_CONFIG_PASS_SCHEMA",
    "CONTRA_ORACLE_PASS_SCHEMA",
    "CONTRA_PASS_SCHEMA",
    "CONTRA_VIEW_SYSTEM_PASS_SET_SCHEMA",
    "CONTRA_GHOST_SCHEMA",
    "CONTRA_GHOST_PASS_SCHEMA",
    "RAW_COLOR_COMPONENT_PACKET_SCHEMA",
    "CONTRA_LAYER_OUTPUT_SCHEMA",
    "CONTRA_LAYER_OUTPUT_SET_SCHEMA",
    "CONTRA_LAYER_HANDOFF_SCHEMA",
    "CONTRA_LAYER_HANDOFF_SET_SCHEMA",
    "EXPLOREBAND_SCHEMA",
    "CONTRA_SYMMETRY_PACKET_SCHEMA",
    "CONTRA_COMPONENT_POLAR_SYMMETRY_SCHEMA",
    "CONTRA_SYMMETRY_LOCK_KEY_SENSOR_SCHEMA",
    "build_raw_color_component_packet",
    "build_exploreband_budget",
    "build_symmetry_hypotheses",
    "calibration_actor_affordance_hypotheses",
    "calibration_target_affordance_hypotheses",
    "empty_actor_affordance_hypotheses",
    "empty_target_affordance_hypotheses",
    "run_contra_view",
]
