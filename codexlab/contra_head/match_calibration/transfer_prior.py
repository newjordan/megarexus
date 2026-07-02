#!/usr/bin/env python3
from __future__ import annotations

from typing import Any

from .common import admission_flags, calibration_id

MATCH_CALIBRATION_TRANSFER_PRIOR_SCHEMA = "codexlab.match_calibration.action_result_transfer_prior.v1"
MATCH_CALIBRATION_TRANSFER_PRIOR_ID = "match_action_result_transfer_prior"
MATCH_CALIBRATION_TRANSFER_PRIOR_STATUS_NO_OBS = "no_prior_observations_supplied"
MATCH_CALIBRATION_TRANSFER_PRIOR_CONTEXT_STATUS = "temporal_action_result_context_not_static_truth"

TRANSFER_PRIOR_FUNCTION_CONFIGS: dict[str, dict[str, float]] = {
    "actor_origin": {
        "success_gain": 0.52,
        "repeat_gain": 0.40,
        "miss_penalty": 0.68,
        "stale_level_penalty": 0.08,
        "high_confidence_threshold": 0.90,
    },
    "target_effect": {
        "success_gain": 0.34,
        "repeat_gain": 0.20,
        "miss_penalty": 0.58,
        "stale_level_penalty": 0.10,
        "high_confidence_threshold": 0.82,
    },
    "lock_key_effect": {
        "success_gain": 0.30,
        "repeat_gain": 0.18,
        "miss_penalty": 0.52,
        "stale_level_penalty": 0.10,
        "high_confidence_threshold": 0.78,
    },
}


def _component_rows(raw_color_component_packet: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not isinstance(raw_color_component_packet, dict):
        return rows
    for color in raw_color_component_packet.get("colors", []):
        if not isinstance(color, dict):
            continue
        for component in color.get("components", []):
            if isinstance(component, dict) and component.get("component_id"):
                row = dict(component)
                row["palette_name"] = color.get("palette_name")
                rows[str(component["component_id"])] = row
    return rows


def _observation_function_id(observation: dict[str, Any]) -> str:
    return str(observation.get("function_id") or observation.get("prior_function_id") or "actor_origin")


def _observation_success(observation: dict[str, Any]) -> bool:
    if observation.get("prediction_success") is not None:
        return bool(observation.get("prediction_success"))
    if observation.get("observed_action_origin") is not None:
        return bool(observation.get("observed_action_origin"))
    return str(observation.get("status") or "").endswith("success")


def _match_component_ids(
    *,
    observation: dict[str, Any],
    components: dict[str, dict[str, Any]],
) -> list[str]:
    explicit_ids = observation.get("current_component_ids")
    if isinstance(explicit_ids, list) and explicit_ids:
        return sorted(str(value) for value in explicit_ids if value is not None and str(value) in components)

    observed_id = observation.get("component_id") or observation.get("observed_component_id")
    if observed_id is not None and str(observed_id) in components:
        return [str(observed_id)]

    palette_name = observation.get("palette_name") or observation.get("observed_palette_name")
    raw_value = observation.get("raw_pixel_value") or observation.get("observed_raw_pixel_value")
    matches: list[str] = []
    for component_id, component in components.items():
        palette_match = palette_name is not None and str(component.get("palette_name")) == str(palette_name)
        raw_match = raw_value is not None and str(component.get("raw_pixel_value")) == str(raw_value)
        if palette_match or raw_match:
            matches.append(component_id)
    return sorted(matches)


def _confidence_for_summary(
    *,
    function_id: str,
    successes: int,
    misses: int,
    current_level_index: int,
    last_seen_level_index: int | None,
) -> float:
    cfg = TRANSFER_PRIOR_FUNCTION_CONFIGS.get(function_id, TRANSFER_PRIOR_FUNCTION_CONFIGS["actor_origin"])
    if successes <= 0:
        return 0.0
    confidence = cfg["success_gain"] + max(0, successes - 1) * cfg["repeat_gain"]
    confidence -= misses * cfg["miss_penalty"]
    if last_seen_level_index is not None:
        confidence -= max(0, int(current_level_index) - int(last_seen_level_index) - 1) * cfg["stale_level_penalty"]
    return round(max(0.0, min(0.99, confidence)), 6)


def build_action_result_transfer_prior(
    *,
    level_index: int,
    raw_color_component_packet: dict[str, Any] | None = None,
    prior_observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    components = _component_rows(raw_color_component_packet)
    observations = [row for row in prior_observations or [] if isinstance(row, dict)]
    if not observations:
        return {
            "schema": MATCH_CALIBRATION_TRANSFER_PRIOR_SCHEMA,
            "stage_id": MATCH_CALIBRATION_TRANSFER_PRIOR_ID,
            "status": MATCH_CALIBRATION_TRANSFER_PRIOR_STATUS_NO_OBS,
            "prior_context_status": MATCH_CALIBRATION_TRANSFER_PRIOR_CONTEXT_STATUS,
            "level_index": int(level_index),
            "function_configs": TRANSFER_PRIOR_FUNCTION_CONFIGS,
            "observation_count": 0,
            "transfer_prior_count": 0,
            "component_role_reweights": [],
            "primary_actor_prior_component_ids": [],
            "nonprimary_actor_like_component_ids": [],
            "prediction_status": "not_predictive",
            "evidence_status": "not_observed",
            **admission_flags(),
        }

    summaries: dict[tuple[str, str], dict[str, Any]] = {}
    for observation in observations:
        function_id = _observation_function_id(observation)
        role_family = str(observation.get("role_family") or observation.get("component_family") or "unknown_family")
        key = (function_id, role_family)
        summary = summaries.setdefault(
            key,
            {
                "transfer_prior_id": calibration_id("transfer_prior", function_id, role_family),
                "function_id": function_id,
                "role_family": role_family,
                "success_count": 0,
                "miss_count": 0,
                "last_seen_level_index": None,
                "source_observation_ids": [],
                "matched_current_component_ids": set(),
            },
        )
        if _observation_success(observation):
            summary["success_count"] += 1
        else:
            summary["miss_count"] += 1
        if observation.get("level_index") is not None:
            level_seen = int(observation["level_index"])
            last_seen = summary.get("last_seen_level_index")
            summary["last_seen_level_index"] = level_seen if last_seen is None else max(int(last_seen), level_seen)
        observation_id = observation.get("observation_id") or observation.get("calibration_id")
        if observation_id is not None:
            summary["source_observation_ids"].append(str(observation_id))
        summary["matched_current_component_ids"].update(_match_component_ids(observation=observation, components=components))

    transfer_priors: list[dict[str, Any]] = []
    component_reweights: dict[str, dict[str, Any]] = {}
    primary_actor_ids: set[str] = set()
    for summary in summaries.values():
        matched_ids = sorted(str(value) for value in summary.pop("matched_current_component_ids"))
        confidence = _confidence_for_summary(
            function_id=str(summary["function_id"]),
            successes=int(summary["success_count"]),
            misses=int(summary["miss_count"]),
            current_level_index=int(level_index),
            last_seen_level_index=summary.get("last_seen_level_index"),
        )
        cfg = TRANSFER_PRIOR_FUNCTION_CONFIGS.get(str(summary["function_id"]), TRANSFER_PRIOR_FUNCTION_CONFIGS["actor_origin"])
        confidence_band = "high" if confidence >= cfg["high_confidence_threshold"] else "medium" if confidence >= 0.50 else "low"
        prior = {
            **summary,
            "confidence": confidence,
            "confidence_band": confidence_band,
            "matched_current_component_ids": matched_ids,
            "prior_context_status": MATCH_CALIBRATION_TRANSFER_PRIOR_CONTEXT_STATUS,
            "admission_status": "not_admitted",
            "control_truth_admitted": False,
        }
        transfer_priors.append(prior)
        for component_id in matched_ids:
            row = component_reweights.setdefault(
                component_id,
                {
                    "component_id": component_id,
                    "role_reweights": [],
                    "max_actor_origin_confidence": 0.0,
                    "primary_actor_prior": False,
                },
            )
            row["role_reweights"].append(
                {
                    "transfer_prior_id": prior["transfer_prior_id"],
                    "function_id": prior["function_id"],
                    "role_family": prior["role_family"],
                    "confidence": confidence,
                    "confidence_band": confidence_band,
                }
            )
            if prior["function_id"] == "actor_origin":
                row["max_actor_origin_confidence"] = max(float(row["max_actor_origin_confidence"]), confidence)
                if confidence_band == "high":
                    row["primary_actor_prior"] = True
                    primary_actor_ids.add(component_id)

    for row in component_reweights.values():
        if primary_actor_ids and row["component_id"] not in primary_actor_ids:
            row["nonprimary_due_to_primary_actor_prior"] = True

    return {
        "schema": MATCH_CALIBRATION_TRANSFER_PRIOR_SCHEMA,
        "stage_id": MATCH_CALIBRATION_TRANSFER_PRIOR_ID,
        "status": "transfer_prior_ranked",
        "prior_context_status": MATCH_CALIBRATION_TRANSFER_PRIOR_CONTEXT_STATUS,
        "level_index": int(level_index),
        "function_configs": TRANSFER_PRIOR_FUNCTION_CONFIGS,
        "observation_count": len(observations),
        "transfer_prior_count": len(transfer_priors),
        "transfer_priors": sorted(transfer_priors, key=lambda row: (-float(row["confidence"]), str(row["transfer_prior_id"]))),
        "component_role_reweights": sorted(component_reweights.values(), key=lambda row: str(row["component_id"])),
        "primary_actor_prior_component_ids": sorted(primary_actor_ids),
        "nonprimary_actor_like_component_ids": [],
        "prediction_status": "prior_context_only_not_predictive",
        "evidence_status": "observed_prior_context" if observations else "not_observed",
        **admission_flags(),
    }
