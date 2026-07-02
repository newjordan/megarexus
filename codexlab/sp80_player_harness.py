#!/usr/bin/env python3
"""Swappable-brain harness for SP80/Liquid Pour basic controls.

The brain is allowed to choose only a control primitive. The motor/controller
grounds that primitive into the real SP80 action set and executes it against a
live game state. This keeps the future Qwen/LoRA player behind a narrow,
testable boundary instead of letting it emit coordinates or route scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from arcengine import GameAction

from contra_head import (
    BACKGROUND_SELECTION_POLICY,
    CONTROL_ISOLATION_SOURCE,
    build_actionmodband,
    build_deltaband_observation_set,
    build_episode_memory_after_round,
    build_initial_episode_memory,
    build_control_isolation_packet,
    build_object_action_response,
    contra_contract,
    contra_module_source_paths,
    contra_spec,
    run_contra_view,
)
from contra_head.ghost import load_ghost_context, write_ghost
from contra_head.crucible.canon import coord_delta as canon_coord_delta
from contra_head.crucible.canon import coord_xy_to_grid_index
from players.base import BrainDecision, PacketRecorder
from players.sp80.controller import SP80PlayerController
from players.sp80.player import SP80_PLAYER
from players.sp80.registry import CONTROL_PRIMITIVES, action_set_cases_for_harness, normalize_primitive
from sp80_base_controller import (
    ROOT,
    SP80_PATH,
    SP80BaseController,
    GroundedAction,
    SpriteRef,
    frame_delta,
    save_frame,
    save_frame_sequence,
    sha256_file,
    sha256_frame,
    primitive_toward_goal_condition,
)


ACTION_SET_CASES = action_set_cases_for_harness()
CONTRA_RECOMMENDATION_PACKET_SCHEMA = "codexlab.sp80_contra_recommendation_packet.v1"
CONTRA_EPISODE_RESULT_SCHEMA = "codexlab.sp80_contra_episode_harness.v1"


class BrainBackend(Protocol):
    name: str

    def decide(self, request: dict[str, Any]) -> BrainDecision:
        ...


class TinyMLPBrain:
    name = "tiny_mlp"

    def __init__(self, controller: SP80BaseController) -> None:
        self.controller = controller

    def decide(self, request: dict[str, Any]) -> BrainDecision:
        started = time.perf_counter()
        recommendation = first_recommended_primitive(request)
        if recommendation is not None:
            return BrainDecision(
                backend=self.name,
                primitive=recommendation["primitive"],
                confidence=float(recommendation.get("confidence") or 0.0),
                raw={
                    "policy_type": "contra_recommendation_packet",
                    "chosen_primitive": recommendation["primitive"],
                    "recommendation": recommendation,
                    "recommendation_packet_id": (
                        request.get("contra_recommendation_packet") or {}
                    ).get("recommendation_packet_id"),
                },
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        raw = self.controller.intent_policy.choose(
            str(request["intent"]),
            dict(request["observation"]["state"]),
            request.get("goal_condition"),
        )
        primitive = normalize_primitive(raw.get("chosen_primitive"))
        probs = raw.get("primitive_probs") or {}
        confidence = probs.get(primitive) if primitive is not None else None
        return BrainDecision(
            backend=self.name,
            primitive=primitive,
            confidence=float(confidence) if confidence is not None else None,
            raw=raw,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


class QwenHTTPBrain:
    name = "qwen_http"

    def __init__(
        self,
        *,
        url: str,
        timeout_s: float,
        mode: str,
        model: str | None,
    ) -> None:
        self.url = url
        self.timeout_s = float(timeout_s)
        self.mode = mode
        self.model = model

    def decide(self, request: dict[str, Any]) -> BrainDecision:
        started = time.perf_counter()
        try:
            payload = self._payload(request)
            req = urllib.request.Request(
                self.url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s) as response:
                text = response.read().decode("utf-8", errors="replace")
            raw = parse_jsonish(text)
            primitive, confidence = extract_primitive(raw)
            return BrainDecision(
                backend=self.name,
                primitive=primitive,
                confidence=confidence,
                raw=raw,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except (OSError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            return BrainDecision(
                backend=self.name,
                primitive=None,
                confidence=None,
                raw=None,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=repr(exc),
            )

    def _payload(self, request: dict[str, Any]) -> dict[str, Any]:
        instruction = (
            "Choose exactly one primitive from allowed_primitives. Return JSON "
            'only: {"primitive":"move_right","confidence":0.0,"reason":"short"}'
        )
        if self.mode == "openai_chat":
            return {
                "model": self.model or "qwen",
                "temperature": 0,
                "max_tokens": 96,
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": json.dumps(request, sort_keys=True)},
                ],
            }
        return {"instruction": instruction, "request": request}


class LoRAAdapterBrain:
    name = "lora_adapter"

    def __init__(self, adapter_path: Path) -> None:
        from sp80_lora_adapter import load_adapter_bundle, predict_request

        self.adapter_path = adapter_path
        self.bundle = load_adapter_bundle(adapter_path)
        self.predict_request = predict_request

    def decide(self, request: dict[str, Any]) -> BrainDecision:
        started = time.perf_counter()
        try:
            raw = self.predict_request(self.bundle, request)
            primitive = normalize_primitive(raw.get("primitive"))
            confidence = raw.get("confidence")
            return BrainDecision(
                backend=self.name,
                primitive=primitive,
                confidence=float(confidence) if confidence is not None else None,
                raw={
                    **raw,
                    "adapter_path": self.adapter_path.relative_to(ROOT).as_posix()
                    if self.adapter_path.is_relative_to(ROOT)
                    else str(self.adapter_path),
                },
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:
            return BrainDecision(
                backend=self.name,
                primitive=None,
                confidence=None,
                raw={"adapter_path": str(self.adapter_path)},
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error=repr(exc),
            )


def build_puffer_policy_brain(policy_path: Path, *, require_pufferlib: bool) -> BrainBackend:
    from sp80_puffer_policy import PufferPolicyBrain

    return PufferPolicyBrain(policy_path, require_pufferlib=require_pufferlib)


def build_paint_control_skill_brain(
    *,
    checkpoint_path: Path | None,
    policy_env: str,
    require_native: bool,
    focus_edge: int | None,
    forward_model_ledger_path: Path | None,
) -> BrainBackend:
    from paint_control_skill import PaintControlSkillBrain

    if checkpoint_path is not None and not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path
    if forward_model_ledger_path is not None and not forward_model_ledger_path.is_absolute():
        forward_model_ledger_path = ROOT / forward_model_ledger_path
    return PaintControlSkillBrain(
        checkpoint_path=checkpoint_path.resolve() if checkpoint_path is not None else None,
        policy_env=policy_env,
        require_native=require_native,
        focus_edge=focus_edge,
        forward_model_ledger_path=forward_model_ledger_path.resolve() if forward_model_ledger_path is not None else None,
    )


def build_spill_planner_brain(*, search_time_budget: float, search_restarts: int) -> BrainBackend:
    from sp80_spill_planner_backend import SpillPlannerBrain

    return SpillPlannerBrain(
        search_time_budget=float(search_time_budget),
        search_restarts=int(search_restarts),
    )


def parse_jsonish(text: str) -> Any:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"response is not JSON: {stripped[:200]}")
        return json.loads(stripped[start : end + 1])


def extract_primitive(raw: Any) -> tuple[str | None, float | None]:
    payload = raw
    if isinstance(raw, dict) and "choices" in raw:
        choice = raw["choices"][0]
        if "message" in choice:
            payload = parse_jsonish(str(choice["message"].get("content", "")))
        elif "text" in choice:
            payload = parse_jsonish(str(choice.get("text", "")))

    if not isinstance(payload, dict):
        return None, None
    candidate = (
        payload.get("primitive")
        or payload.get("chosen_primitive")
        or payload.get("action")
        or payload.get("action_name")
        or payload.get("action_id")
    )
    primitive = normalize_primitive(candidate)
    confidence = payload.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None
    return primitive, confidence_value


def first_recommended_primitive(request: dict[str, Any]) -> dict[str, Any] | None:
    packet = request.get("contra_recommendation_packet")
    if not isinstance(packet, dict):
        return None
    for row in packet.get("ranked_recommendations") or []:
        if not isinstance(row, dict):
            continue
        primitive = normalize_primitive(row.get("primitive"))
        if primitive in CONTROL_PRIMITIVES:
            return {**row, "primitive": primitive}
    return None


def selected_recommendation_from_decision(decision: BrainDecision) -> dict[str, Any]:
    raw = decision.raw if isinstance(decision.raw, dict) else {}
    recommendation = raw.get("recommendation")
    return recommendation if isinstance(recommendation, dict) else {}


def recommendation_row(
    *,
    rank: int,
    primitive: str,
    confidence: float,
    evidence_sources: list[str],
    reason: str,
    candidate_kind: str,
    value_score: float,
    signals: dict[str, Any] | None = None,
    evidence_refs: dict[str, Any] | None = None,
    target_override: dict[str, Any] | None = None,
    expected_effect_tags: list[str] | None = None,
    evidence_tier: int = 0,
) -> dict[str, Any]:
    row = {
        "rank": int(rank),
        "primitive": primitive,
        "confidence": round(max(0.0, min(1.0, float(confidence))), 6),
        "value_score": round(max(0.0, min(1.0, float(value_score))), 6),
        "evidence_tier": int(evidence_tier),
        "candidate_kind": candidate_kind,
        "evidence_sources": evidence_sources,
        "evidence_refs": evidence_refs or {},
        "signals": signals or {},
        "reason": reason,
        "expected_effect_tags": expected_effect_tags or [],
        "action_authority": "recommendation_to_existing_harness_brain_only",
        "motor_boundary": "existing_sp80_controller_executes_primitive",
    }
    if target_override:
        row["target_override"] = target_override
    return row


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def last_same_primitive_no_progress(action_memory: list[dict[str, Any]], primitive: str) -> bool:
    last = action_memory[-1] if action_memory else {}
    last_delta = (last.get("goal_progress") or {}).get("distance_delta")
    return (
        last.get("primitive") == primitive
        and (last.get("levels_completed_delta") or 0) <= 0
        and (last_delta is None or number(last_delta) <= 0.0)
    )


def repeated_no_progress_count(action_memory: list[dict[str, Any]], primitive: str) -> int:
    count = 0
    for row in reversed(action_memory):
        if row.get("primitive") != primitive:
            break
        progress = row.get("goal_progress") or {}
        if (row.get("levels_completed_delta") or 0) > 0 or number(progress.get("distance_delta")) > 0.0:
            break
        count += 1
    return count


def repeated_target_no_progress_count(
    action_memory: list[dict[str, Any]],
    *,
    primitive: str,
    component_id: str | None,
) -> int:
    if not component_id:
        return 0
    count = 0
    for row in list(reversed(action_memory))[:8]:
        rec = row.get("selected_recommendation") or {}
        target = rec.get("target_override") if isinstance(rec, dict) else {}
        if row.get("primitive") != primitive or not isinstance(target, dict) or target.get("component_id") != component_id:
            continue
        progress = row.get("goal_progress") or {}
        if (row.get("levels_completed_delta") or 0) > 0 or number(progress.get("distance_delta")) > 0.0:
            continue
        count += 1
    return count


def repeated_task_no_progress_count(
    action_memory: list[dict[str, Any]],
    *,
    primitive: str,
    task_key: str | None,
) -> int:
    if not task_key:
        return repeated_no_progress_count(action_memory, primitive)
    count = 0
    for row in reversed(action_memory):
        if row.get("primitive") != primitive:
            break
        rec = row.get("selected_recommendation")
        refs = rec.get("evidence_refs") if isinstance(rec, dict) else {}
        row_task_key = refs.get("portband_task_key") if isinstance(refs, dict) else None
        if str(row_task_key or "") != task_key:
            break
        progress = row.get("goal_progress") or {}
        if (row.get("levels_completed_delta") or 0) > 0 or number(progress.get("distance_delta")) > 0.0:
            break
        count += 1
    return count


def post_saturation_refinement_no_progress_count(action_memory: list[dict[str, Any]], task_key: str) -> int:
    if not task_key:
        return 0
    count = 0
    for row in reversed(action_memory[-8:]):
        if not str(row.get("primitive") or "").startswith("move_"):
            continue
        rec = row.get("selected_recommendation")
        if not isinstance(rec, dict):
            continue
        refs = rec.get("evidence_refs") if isinstance(rec.get("evidence_refs"), dict) else {}
        if str(refs.get("portband_task_key") or "") != task_key:
            continue
        signals = rec.get("signals") if isinstance(rec.get("signals"), dict) else {}
        refinement_bonus = number(
            signals.get("post_saturation_refinement_bonus")
            or signals.get("action_post_saturation_refinement_bonus")
        )
        if refinement_bonus <= 0.0:
            continue
        progress = row.get("goal_progress") or {}
        if (row.get("levels_completed_delta") or 0) > 0 or number(progress.get("distance_delta")) > 0.0:
            break
        count += 1
    return count


def orthogonal_axis_after_same_task_no_progress(
    action_memory: list[dict[str, Any]],
    *,
    primitive: str,
    task_key: str | None,
    dx: float,
    dy: float,
) -> str | None:
    if not task_key or repeated_task_no_progress_count(
        action_memory,
        primitive=primitive,
        task_key=task_key,
    ) < 1:
        return None
    if primitive in {"move_left", "move_right"} and abs(dy) > 0.0:
        return "move_down" if dy > 0.0 else "move_up"
    if primitive in {"move_up", "move_down"} and abs(dx) > 0.0:
        return "move_right" if dx > 0.0 else "move_left"
    return None


INVERSE_MOVES = {
    "move_left": "move_right",
    "move_right": "move_left",
    "move_up": "move_down",
    "move_down": "move_up",
}


def movement_sequence(action_memory: list[dict[str, Any]]) -> list[str]:
    sequence: list[str] = []
    for row in reversed(action_memory):
        primitive = str(row.get("primitive") or "")
        if primitive not in INVERSE_MOVES:
            break
        sequence.append(primitive)
    return list(reversed(sequence))


def oscillation_pair_key(left: str, right: str) -> str:
    return "|".join(sorted([left, right]))


def movement_oscillation_status(action_memory: list[dict[str, Any]], primitive: str) -> dict[str, Any]:
    sequence = movement_sequence(action_memory)
    pair_key = None
    prior_pair_count = 0
    if primitive in INVERSE_MOVES and sequence and INVERSE_MOVES[primitive] == sequence[-1]:
        pair_key = oscillation_pair_key(sequence[-1], primitive)
        for idx in range(1, len(sequence)):
            if oscillation_pair_key(sequence[idx - 1], sequence[idx]) == pair_key and INVERSE_MOVES[sequence[idx - 1]] == sequence[idx]:
                prior_pair_count += 1
    return {
        "movement_sequence": sequence,
        "movement_sequence_tail": sequence[-6:],
        "movement_oscillation_pair_key": pair_key,
        "movement_oscillation_prior_pair_count": prior_pair_count,
        "movement_oscillation_hard_limit": 1,
        "movement_oscillation_hard_blocked": bool(pair_key and prior_pair_count >= 1),
    }


def movement_oscillation_gate(
    *,
    action_memory: list[dict[str, Any]],
    primitive: str,
    candidate_kind: str,
    evidence_refs: dict[str, Any] | None,
    signals: dict[str, Any],
) -> tuple[bool, dict[str, Any], dict[str, Any]]:
    refs = dict(evidence_refs or {})
    enriched_signals = dict(signals)
    status = movement_oscillation_status(action_memory, primitive)
    refs["movement_sequence_scope"] = "contiguous_move_primitives"
    refs["movement_oscillation_pair_key"] = status["movement_oscillation_pair_key"]
    enriched_signals.update(status)
    task_key = refs.get("portband_task_key")
    stale_repetition_count = repeated_task_no_progress_count(
        action_memory,
        primitive=primitive,
        task_key=str(task_key) if task_key else None,
    )
    stale_repetition_blocked = stale_repetition_count >= 2
    refs["stale_repetition_scope"] = (
        "contiguous_same_primitive_same_portband_task_no_progress"
        if task_key
        else "contiguous_same_primitive_no_progress"
    )
    enriched_signals.update(
        {
            "stale_repetition_prior_count": stale_repetition_count,
            "stale_repetition_hard_limit": 2,
            "stale_repetition_hard_blocked": stale_repetition_blocked,
        }
    )
    return (
        not enriched_signals["movement_oscillation_hard_blocked"]
        and not stale_repetition_blocked
    ), refs, enriched_signals


RELAY_LOCK_KINDS = ("multi_object_goal_adjacency_plan", "portband_lineup_matrix_row")

# SURFACE_ORTHOGONAL_PAINT_REDIRECT rule (user-specified L1 mechanic): paint is captured at the
# top of the LARGEST action object and the ORTHOGONAL sides re-direct the paint flow -- a
# surface/orthogonal flow dictates paint direction. So when paint flows along one axis, the moves
# that re-direct it are the ones ORTHOGONAL to the flow axis, applied on the largest object.
_FLOW_ORTHOGONAL_MOVES = {
    "up": ("move_left", "move_right"), "down": ("move_left", "move_right"),
    "left": ("move_up", "move_down"), "right": ("move_up", "move_down"),
}


def _largest_object_flow_direction(contra_packet: dict[str, Any]) -> str | None:
    """Flow-direction hypothesis for the largest action object (top-of-object paint surface)."""
    traces = ((contra_packet.get("control_isolation_packet") or {}).get("closed_shape_traces")) or []
    largest_area = -1.0
    for t in traces:
        if isinstance(t, dict):
            largest_area = max(largest_area, number(t.get("area_pixels")))
    # actor / ledger flow hypothesis names the axis the paint travels along
    for src in (contra_packet.get("actor_affordance_hypotheses") or []):
        if isinstance(src, dict) and src.get("flow_direction_hypothesis"):
            return str(src["flow_direction_hypothesis"])
    rows = ((contra_packet.get("action_semantics_ledger") or {}).get("rows")) or []
    for r in rows:
        if isinstance(r, dict) and r.get("flow_direction_hypothesis"):
            return str(r["flow_direction_hypothesis"])
    return None


def surface_orthogonal_flow_redirect(rows: list[dict[str, Any]], contra_packet: dict[str, Any]) -> None:
    """Lock in the surface/orthogonal-flow rule: boost moves ORTHOGONAL to the paint-flow axis
    (these are the surface sides that re-direct the captured paint). Bonus-only; cannot introduce
    a new action. Marks the rows so the behaviour is auditable in the report."""
    flow = _largest_object_flow_direction(contra_packet)
    ortho = _FLOW_ORTHOGONAL_MOVES.get(str(flow or "").lower())
    if not ortho:
        return
    for row in rows:
        if str(row.get("primitive") or "") in ortho:
            boosted = clamp01(number(row.get("value_score")) + 0.18)
            row["value_score"] = round(boosted, 6)
            row["confidence"] = round(min(number(row.get("confidence")) or boosted, boosted), 6)
            if isinstance(row.get("signals"), dict):
                row["signals"]["surface_orthogonal_flow_redirect_bonus"] = 0.18
                row["signals"]["surface_orthogonal_flow_axis"] = flow
                row["signals"]["surface_orthogonal_flow_policy"] = (
                    "orthogonal_to_paint_flow_redirects_on_largest_object_surface")


def relay_lock_hold_bonus(rows: list[dict[str, Any]], action_memory: list[dict[str, Any]]) -> None:
    """STABLE_MULTI_SOURCE_RELAY_LOCK (HOLD rule, crystallized from the failing L1 run).

    Once a high-value relay leg is committed, add a persistence bonus to candidates that continue
    the SAME candidate_kind, so the controller does not abandon a still-closing leg and thrash
    across candidate_kinds (the observed L1 failure: lock abandoned at pkt5, 14 switches/20 pkts).
    Bonus-only: it can never introduce a new action, only re-rank an existing one. The lock is
    RELEASED (no bonus) once the committed kind made no forward progress twice in a row (stuck),
    which lets ADVANCE move on. Boosting the kinds L0 already wins with cannot regress L0.
    """
    if not action_memory:
        return
    last_kind = str((action_memory[-1].get("selected_recommendation") or {}).get("candidate_kind") or "")
    if not last_kind.startswith(RELAY_LOCK_KINDS):
        return
    noprog = 0
    for row in reversed(action_memory):
        k = str((row.get("selected_recommendation") or {}).get("candidate_kind") or "")
        if not k.startswith(RELAY_LOCK_KINDS):
            break
        gp = row.get("goal_progress") or {}
        if (row.get("levels_completed_delta") or 0) > 0 or number(gp.get("distance_delta")) > 0.0:
            break
        noprog += 1
    if noprog >= 2:
        return  # committed leg is stuck -> release the lock so the scorer can ADVANCE/switch
    for row in rows:
        if str(row.get("candidate_kind") or "").startswith(RELAY_LOCK_KINDS):
            boosted = clamp01(number(row.get("value_score")) + 0.20)
            row["value_score"] = round(boosted, 6)
            row["confidence"] = round(min(number(row.get("confidence")) or boosted, boosted), 6)
            if isinstance(row.get("signals"), dict):
                row["signals"]["relay_lock_hold_bonus"] = 0.20
                row["signals"]["relay_lock_hold_policy"] = "committed_relay_leg_persistence_until_stuck"


def semantic_probe_no_progress_count(
    action_memory: list[dict[str, Any]],
    *,
    primitive: str,
    family: str,
) -> int:
    if not family:
        return 0
    count = 0
    for row in reversed(action_memory[-8:]):
        if row.get("primitive") != primitive:
            continue
        rec = row.get("selected_recommendation")
        if not isinstance(rec, dict):
            continue
        candidate_kind = str(rec.get("candidate_kind") or "")
        if candidate_kind != f"action_semantics_ledger:{family}":
            continue
        progress = row.get("goal_progress") or {}
        if (row.get("levels_completed_delta") or 0) > 0 or number(progress.get("distance_delta")) > 0.0:
            continue
        count += 1
    return count


def semantic_probe_positive_count(
    action_memory: list[dict[str, Any]],
    *,
    primitive: str,
    family: str,
) -> int:
    if not family:
        return 0
    count = 0
    for row in reversed(action_memory[-4:]):
        if row.get("primitive") != primitive:
            continue
        rec = row.get("selected_recommendation")
        if not isinstance(rec, dict):
            continue
        candidate_kind = str(rec.get("candidate_kind") or "")
        if candidate_kind != f"action_semantics_ledger:{family}":
            continue
        progress = row.get("goal_progress") or {}
        if (row.get("levels_completed_delta") or 0) > 0 or number(progress.get("distance_delta")) > 0.0:
            count += 1
            continue
        break
    return count


def selected_object_stationary_move_count(action_memory: list[dict[str, Any]], primitive: str) -> int:
    if not primitive.startswith("move_"):
        return 0
    count = 0
    for row in reversed(action_memory[-6:]):
        if row.get("primitive") != primitive:
            continue
        delta = row.get("selected_display_coord_delta")
        if delta == [0, 0]:
            count += 1
            continue
        if delta is not None:
            break
    return count


def primitive_from_direction_hypothesis(direction: Any) -> str | None:
    return {
        "up": "move_up",
        "down": "move_down",
        "left": "move_left",
        "right": "move_right",
    }.get(str(direction or "").lower())


def collect_portband_lineup_matrix_rows(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("schema") == "codexlab.geoband_portband_lineup_matrix.v1":
            for row in value.get("rows") or []:
                if isinstance(row, dict):
                    rows.append(row)
        for child in value.values():
            rows.extend(collect_portband_lineup_matrix_rows(child))
    elif isinstance(value, list):
        for child in value:
            rows.extend(collect_portband_lineup_matrix_rows(child))
    return rows


def collect_goal_option_component_ids(value: Any) -> list[str]:
    if isinstance(value, dict):
        members = value.get("members")
        if isinstance(members, dict):
            component_ids = members.get("goal_option_component_ids")
            if isinstance(component_ids, list) and component_ids:
                return [str(component_id) for component_id in component_ids if component_id is not None]
        for child in value.values():
            found = collect_goal_option_component_ids(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = collect_goal_option_component_ids(child)
            if found:
                return found
    return []


def explicit_target_component_ids(goal_condition: dict[str, Any], contra_packet: dict[str, Any]) -> set[str]:
    target_index = goal_condition.get("target_bucket_index")
    if target_index is None:
        return set()
    try:
        index = int(target_index)
    except (TypeError, ValueError):
        return set()
    if index < 0:
        return set()
    goal_components = collect_goal_option_component_ids(contra_packet)
    if index >= len(goal_components):
        return set()
    return {canonical_portband_component_id(goal_components[index])}


def recommendation_uses_portband_target(row: dict[str, Any]) -> bool:
    if row.get("candidate_kind") in {"multi_object_goal_adjacency_plan", "portband_lineup_matrix_row"}:
        return True
    sources = row.get("evidence_sources")
    return isinstance(sources, list) and "portband.lineup_matrix.rows" in sources


def target_components_for_recommendation(row: dict[str, Any]) -> set[str]:
    refs = row.get("evidence_refs") if isinstance(row.get("evidence_refs"), dict) else {}
    components = refs.get("target_component_ids")
    if not isinstance(components, list):
        return set()
    return {canonical_portband_component_id(component_id) for component_id in components if component_id is not None}


def filter_recommendations_to_explicit_goal_target(
    rows: list[dict[str, Any]],
    *,
    goal_condition: dict[str, Any],
    contra_packet: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_components = explicit_target_component_ids(goal_condition, contra_packet)
    if not target_components:
        return rows, {
            "status": "not_applicable",
            "explicit_target_component_ids": [],
            "filtered_count": 0,
            "filtered_preview": [],
        }

    kept: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if not recommendation_uses_portband_target(row):
            kept.append(row)
            continue
        row_target_components = target_components_for_recommendation(row)
        if row_target_components and row_target_components.isdisjoint(target_components):
            filtered.append(row)
            continue
        kept.append(row)

    preview = []
    for row in filtered[:8]:
        refs = row.get("evidence_refs") if isinstance(row.get("evidence_refs"), dict) else {}
        preview.append(
            {
                "rank": row.get("rank"),
                "primitive": row.get("primitive"),
                "candidate_kind": row.get("candidate_kind"),
                "source_component_ids": refs.get("source_component_ids", []),
                "target_component_ids": refs.get("target_component_ids", []),
                "target_object_ids": refs.get("target_object_ids", []),
                "value_score": row.get("value_score"),
                "reason": "portband_target_components_do_not_match_explicit_goal_target",
            }
        )
    return kept, {
        "status": "applied",
        "explicit_target_component_ids": sorted(target_components),
        "filtered_count": len(filtered),
        "filtered_preview": preview,
        "boundary": "target filter scopes harness recommendations to the explicit bucket condition; PortBand hypotheses remain review-only",
    }


def target_scope_filtered_portband_context_row(
    contra_packet: dict[str, Any],
    target_scope_filter: dict[str, Any],
) -> dict[str, Any] | None:
    lineup_rows = collect_portband_lineup_matrix_rows(contra_packet)
    if not lineup_rows:
        return None
    preview = lineup_rows[0] if isinstance(lineup_rows[0], dict) else {}
    return recommendation_row(
        rank=9999,
        primitive="activate_flow",
        confidence=0.0,
        value_score=0.0,
        candidate_kind="target_scope_filtered_portband_lineup_context",
        evidence_sources=["portband.lineup_matrix.rows", "target_scope_filter.filtered_context"],
        evidence_refs={
            "lineup_matrix_row_count": len(lineup_rows),
            "filtered_count": target_scope_filter.get("filtered_count"),
            "explicit_target_component_ids": target_scope_filter.get("explicit_target_component_ids", []),
            "preview_lineup_matrix_row_id": preview.get("lineup_matrix_row_id"),
            "preview_source_component_ids": preview.get("source_component_ids", []),
            "preview_target_component_ids": preview.get("target_component_ids", []),
            "route_authority": "none_static_review_only",
        },
        signals={
            "target_scope_filtered_portband_context_score": 0.0,
            "lineup_matrix_row_count": len(lineup_rows),
            "filtered_count": target_scope_filter.get("filtered_count"),
            "action_choice_admitted": False,
            "control_truth_admitted": False,
            "goal_condition_admitted": False,
            "solve_condition_admitted": False,
        },
        reason="portband_lineup_rows_present_but_filtered_from_action_candidates_by_explicit_target_scope",
        expected_effect_tags=["review_context_only"],
    )


def lineage_key_for_lineup_row(row: dict[str, Any]) -> str:
    return "|".join(
        [
            ",".join(str(value) for value in row.get("source_component_ids", []) or []),
            ",".join(str(value) for value in row.get("target_component_ids", []) or []),
            ",".join(str(value) for value in row.get("target_port_ids", []) or []),
            str(row.get("alignment_kind") or (row.get("score_terms") or {}).get("alignment_kind") or ""),
            str(row.get("source_object_role") or ""),
        ]
    )


def route_key_for_lineup_row(row: dict[str, Any]) -> str:
    return "|".join(
        [
            ",".join(str(value) for value in row.get("target_object_ids", []) or []),
            ",".join(str(value) for value in row.get("target_port_ids", []) or []),
            str(row.get("alignment_kind") or (row.get("score_terms") or {}).get("alignment_kind") or ""),
            str(row.get("source_object_role") or ""),
        ]
    )


def active_portband_route_key(action_memory: list[dict[str, Any]]) -> str | None:
    if not action_memory:
        return None
    last = action_memory[-1]
    if (last.get("levels_completed_delta") or 0) > 0 or last.get("primitive") == "activate_flow":
        return None
    if last.get("primitive", "").startswith("move_") and last.get("selected_display_coord_delta") == [0, 0]:
        return None
    for memory_row in reversed(action_memory[-6:]):
        rec = memory_row.get("selected_recommendation") if isinstance(memory_row, dict) else None
        if not isinstance(rec, dict):
            continue
        if "portband.lineup_matrix.rows" not in (rec.get("evidence_sources") or []):
            continue
        refs = rec.get("evidence_refs") if isinstance(rec.get("evidence_refs"), dict) else {}
        route_key = refs.get("portband_route_key")
        if route_key:
            return str(route_key)
    return None


def blocked_portband_route_keys(action_memory: list[dict[str, Any]]) -> set[str]:
    blocked: set[str] = set()
    for memory_row in action_memory[-6:]:
        if not isinstance(memory_row, dict):
            continue
        if not str(memory_row.get("primitive") or "").startswith("move_"):
            continue
        if memory_row.get("selected_display_coord_delta") != [0, 0]:
            continue
        rec = memory_row.get("selected_recommendation")
        if not isinstance(rec, dict) or "portband.lineup_matrix.rows" not in (rec.get("evidence_sources") or []):
            continue
        refs = rec.get("evidence_refs") if isinstance(rec.get("evidence_refs"), dict) else {}
        route_key = refs.get("portband_route_key")
        if route_key:
            blocked.add(str(route_key))
    return blocked


def primitive_from_lineup_transform(row: dict[str, Any]) -> str:
    transform = row.get("transform_hypothesis_xy")
    if not isinstance(transform, list) or len(transform) != 2:
        return "activate_flow"
    dx = number(transform[0])
    dy = number(transform[1])
    alignment_kind = str(row.get("alignment_kind") or (row.get("score_terms") or {}).get("alignment_kind") or "")
    if alignment_kind == "centered_dual_port_projection" and abs(dx) > 0.0:
        return "move_right" if dx > 0.0 else "move_left"
    if abs(dy) > abs(dx) and abs(dy) > 0.0:
        return "move_down" if dy > 0.0 else "move_up"
    if abs(dx) > 0.0:
        return "move_right" if dx > 0.0 else "move_left"
    if abs(dy) > 0.0:
        return "move_down" if dy > 0.0 else "move_up"
    return "activate_flow"


def primitive_from_lineup_transform_with_memory(row: dict[str, Any], action_memory: list[dict[str, Any]]) -> str:
    primitive = primitive_from_lineup_transform(row)
    transform = row.get("transform_hypothesis_xy")
    if not isinstance(transform, list) or len(transform) != 2:
        return primitive
    dx = number(transform[0])
    dy = number(transform[1])
    task_key = portband_task_key_for_lineup_row(row)
    orthogonal = orthogonal_axis_after_same_task_no_progress(
        action_memory,
        primitive=primitive,
        task_key=task_key,
        dx=dx,
        dy=dy,
    )
    if orthogonal is not None:
        return orthogonal
    return primitive


def lineup_transform_step_cost(row: dict[str, Any]) -> float:
    transform = row.get("transform_hypothesis_xy")
    if not isinstance(transform, list) or len(transform) != 2:
        return 0.0
    return abs(number(transform[0])) + abs(number(transform[1]))


def lineup_has_motion_transform(row: dict[str, Any]) -> bool:
    return lineup_transform_step_cost(row) > 0.0


def lineup_match_status(row: dict[str, Any]) -> str:
    return str((row.get("score_terms") or {}).get("match_status") or "")


def task_keys_with_motion_transform(rows: list[dict[str, Any]]) -> set[str]:
    return {
        portband_task_key_for_lineup_row(row)
        for row in rows
        if isinstance(row, dict) and lineup_has_motion_transform(row)
    }


def lineup_step_efficiency(row: dict[str, Any]) -> float:
    return 1.0 / (1.0 + lineup_transform_step_cost(row) / 16.0)


def post_saturation_refinement_bonus(row: dict[str, Any], saturation_status: dict[str, Any]) -> float:
    if not saturation_status.get("all_eligible_portband_tasks_saturated"):
        return 0.0
    if not saturation_status.get("task_saturated") or not lineup_has_motion_transform(row):
        return 0.0
    match_status = lineup_match_status(row)
    if match_status not in {"exact_alignment", "residual_imagined_alignment"}:
        return 0.0
    return 0.56 if lineup_row_score(row) >= 0.85 else 0.34


def lineup_row_score(row: dict[str, Any]) -> float:
    score = (
        row.get("effective_lineup_confidence_score")
        if row.get("effective_lineup_confidence_score") is not None
        else row.get("action_state_adjusted_confidence_score")
        if row.get("action_state_adjusted_confidence_score") is not None
        else row.get("route_flow_adjusted_confidence_score")
        if row.get("route_flow_adjusted_confidence_score") is not None
        else row.get("confidence_score")
        if row.get("confidence_score") is not None
        else row.get("lineup_score")
    )
    return clamp01(number(score))


def target_signature_for_lineup_row(row: dict[str, Any]) -> str:
    target_objects = ",".join(str(value) for value in row.get("target_object_ids", []) or [])
    target_ports = ",".join(str(value) for value in row.get("target_port_ids", []) or [])
    return "|".join(
        [
            target_objects,
            target_ports,
            str(row.get("alignment_kind") or (row.get("score_terms") or {}).get("alignment_kind") or ""),
        ]
    )


def bbox_center_xy(bbox: Any) -> list[int] | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    return [int(round((x0 + x1) / 2.0)), int(round((y0 + y1) / 2.0))]


def selected_display_inside_source_bbox(before_state: dict[str, Any], row: dict[str, Any]) -> bool:
    selected = before_state.get("selected_display_coord")
    bbox = row.get("source_bbox_xyxy")
    if not isinstance(selected, list) or len(selected) != 2 or not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        sx, sy = float(selected[0]), float(selected[1])
        x0, y0, x1, y1 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return False
    return min(x0, x1) <= sx <= max(x0, x1) and min(y0, y1) <= sy <= max(y0, y1)


def source_selection_target_override(row: dict[str, Any]) -> dict[str, Any] | None:
    source_bbox = row.get("source_bbox_xyxy")
    click_xy = bbox_center_xy(source_bbox)
    if click_xy is None:
        return None
    source_components = [str(value) for value in row.get("source_component_ids", []) or [] if value is not None]
    return {
        "target_kind": "portband_source_object",
        "component_id": source_components[0] if source_components else None,
        "source_object_id": str(row.get("source_object_id") or ""),
        "source_component_ids": source_components,
        "source_bbox_xyxy": source_bbox,
        "click_xy": click_xy,
        "selection_basis": "portband_source_object_priority",
    }


def canonical_portband_component_id(value: Any) -> str:
    text = str(value or "")
    marker = "_v"
    if marker in text and text.startswith("c"):
        return text.split(marker, 1)[0]
    return text


def source_selection_key_from_target(target: dict[str, Any] | None) -> str | None:
    if not isinstance(target, dict):
        return None
    component_id = target.get("component_id")
    if component_id:
        return f"component:{canonical_portband_component_id(component_id)}"
    source_components = target.get("source_component_ids")
    if isinstance(source_components, list) and source_components:
        return f"component:{canonical_portband_component_id(source_components[0])}"
    source_object_id = target.get("source_object_id")
    if source_object_id:
        return f"source:{source_object_id}"
    click_xy = target.get("click_xy")
    if isinstance(click_xy, list) and len(click_xy) == 2:
        return f"click:{int(round(number(click_xy[0])))}:{int(round(number(click_xy[1])))}"
    return None


def source_selection_key_for_lineup_row(row: dict[str, Any]) -> str | None:
    return source_selection_key_from_target(source_selection_target_override(row))


def recently_selected_source_key(action_memory: list[dict[str, Any]]) -> str | None:
    for memory_row in reversed(action_memory[-4:]):
        if memory_row.get("primitive") != "select_object" or memory_row.get("status") != "executed":
            continue
        if memory_row.get("selected_display_coord_delta") == [0, 0]:
            continue
        rec = memory_row.get("selected_recommendation")
        target = rec.get("target_override") if isinstance(rec, dict) else None
        key = source_selection_key_from_target(target if isinstance(target, dict) else None)
        if key:
            return key
    return None


def recent_successful_source_selection_active(action_memory: list[dict[str, Any]], source_key: str | None) -> bool:
    if not source_key:
        return False
    no_progress_moves_after_selection = 0
    for memory_row in reversed(action_memory[-8:]):
        primitive = str(memory_row.get("primitive") or "")
        if primitive == "select_object":
            rec = memory_row.get("selected_recommendation")
            target = rec.get("target_override") if isinstance(rec, dict) else None
            recent_source_key = source_selection_key_from_target(target if isinstance(target, dict) else None)
            if recent_source_key is None:
                return False
            return (
                recent_source_key == source_key
                and memory_row.get("status") == "executed"
                and memory_row.get("selected_display_coord_delta") != [0, 0]
            )
        if primitive.startswith("move_"):
            progress = memory_row.get("goal_progress") or {}
            if (memory_row.get("levels_completed_delta") or 0) > 0 or number(progress.get("distance_delta")) > 0.0:
                no_progress_moves_after_selection = 0
                continue
            no_progress_moves_after_selection += 1
            if no_progress_moves_after_selection >= 2:
                return False
    return False


def blocked_source_selection_keys(action_memory: list[dict[str, Any]]) -> set[str]:
    blocked: set[str] = set()
    for memory_row in action_memory[-6:]:
        if memory_row.get("primitive") != "select_object" or memory_row.get("status") != "executed":
            continue
        if memory_row.get("selected_display_coord_delta") != [0, 0]:
            continue
        rec = memory_row.get("selected_recommendation")
        target = rec.get("target_override") if isinstance(rec, dict) else None
        key = source_selection_key_from_target(target if isinstance(target, dict) else None)
        if key:
            blocked.add(key)
    return blocked


def source_selection_status(before_state: dict[str, Any], row: dict[str, Any], action_memory: list[dict[str, Any]]) -> str:
    target_override = source_selection_target_override(row)
    source_key = source_selection_key_from_target(target_override)
    if target_override is None or source_key is None:
        return "source_object_unselectable_missing_bbox"
    if source_key in blocked_source_selection_keys(action_memory):
        return "source_selection_blocked_no_selection_change"
    if source_key == recently_selected_source_key(action_memory) or recent_successful_source_selection_active(action_memory, source_key):
        return "source_object_recently_selected_by_click"
    if before_state.get("selected") is not True:
        return "source_selection_required_no_selected_object"
    if selected_display_inside_source_bbox(before_state, row):
        return "source_object_already_selected"
    return "source_selection_required_selected_mismatch"


def primitive_and_target_for_lineup_row(
    row: dict[str, Any],
    *,
    before_state: dict[str, Any],
    action_memory: list[dict[str, Any]],
) -> tuple[str, dict[str, Any] | None, str]:
    status = source_selection_status(before_state, row, action_memory)
    target_override = source_selection_target_override(row)
    if status in {"source_selection_required_no_selected_object", "source_selection_required_selected_mismatch"} and target_override is not None:
        return "select_object", target_override, status
    return primitive_from_lineup_transform_with_memory(row, action_memory), None, status


def orthogonal_axis_correction_signal(
    row: dict[str, Any],
    *,
    primitive: str,
    action_memory: list[dict[str, Any]],
) -> dict[str, Any]:
    base_primitive = primitive_from_lineup_transform(row)
    transform = row.get("transform_hypothesis_xy")
    dx = number(transform[0]) if isinstance(transform, list) and len(transform) == 2 else 0.0
    dy = number(transform[1]) if isinstance(transform, list) and len(transform) == 2 else 0.0
    task_key = portband_task_key_for_lineup_row(row)
    repeated_count = repeated_task_no_progress_count(
        action_memory,
        primitive=base_primitive,
        task_key=task_key,
    )
    is_axis_switch = (
        primitive.startswith("move_")
        and base_primitive.startswith("move_")
        and primitive != base_primitive
    )
    return {
        "base_transform_primitive": base_primitive,
        "portband_axis_priority_policy": "centered_dual_port_projection_uses_horizontal_transform_before_dominant_abs_axis"
        if str(row.get("alignment_kind") or (row.get("score_terms") or {}).get("alignment_kind") or "") == "centered_dual_port_projection"
        and abs(dx) > 0.0
        else "dominant_abs_transform_axis_before_same_task_feedback",
        "portband_transform_abs_dx": abs(dx),
        "portband_transform_abs_dy": abs(dy),
        "portband_dominant_axis": "y" if abs(dy) > abs(dx) else "x" if abs(dx) > 0.0 else "none",
        "portband_preferred_axis": "x"
        if str(row.get("alignment_kind") or (row.get("score_terms") or {}).get("alignment_kind") or "") == "centered_dual_port_projection"
        and abs(dx) > 0.0
        else "y"
        if abs(dy) > abs(dx)
        else "x"
        if abs(dx) > 0.0
        else "none",
        "orthogonal_axis_after_same_task_no_progress": is_axis_switch,
        "orthogonal_axis_prior_no_progress_count": repeated_count,
        "orthogonal_axis_policy": "same_portband_task_loss_switches_to_remaining_transform_axis",
    }


def lineup_value_from_signals(row: dict[str, Any]) -> float:
    base = lineup_row_score(row)
    if base <= 0.0:
        return 0.0
    edge_score = number(row.get("edge_lock_quality_score"))
    return clamp01(0.50 * base + 0.22 * lineup_step_efficiency(row) + 0.14 * edge_score + 0.14 * lineup_definedness_score(row))


def lineup_definedness_score(row: dict[str, Any]) -> float:
    source_known = 1.0 if row.get("source_object_id") and row.get("source_component_ids") and source_selection_target_override(row) else 0.0
    target_known = 1.0 if row.get("target_object_ids") and row.get("target_port_ids") else 0.0
    transform_known = 1.0 if isinstance(row.get("transform_hypothesis_xy"), list) and len(row.get("transform_hypothesis_xy")) == 2 else 0.0
    edge_known = 1.0 if row.get("edge_lock_status") == "has_ranked_edge_lock_configuration" else 0.0
    return clamp01(0.38 * source_known + 0.34 * target_known + 0.18 * transform_known + 0.10 * edge_known)


def lineup_known_task_priority(row: dict[str, Any]) -> float:
    per_step_signal = clamp01(lineup_row_score(row) / max(1.0, lineup_transform_step_cost(row)))
    return clamp01(
        0.40 * lineup_row_score(row)
        + 0.24 * lineup_definedness_score(row)
        + 0.20 * lineup_step_efficiency(row)
        + 0.16 * per_step_signal
    )


def portband_task_key_for_lineup_row(row: dict[str, Any]) -> str:
    source_components = row.get("source_component_ids")
    if isinstance(source_components, list) and source_components:
        return f"source_component:{canonical_portband_component_id(source_components[0])}"
    source_id = row.get("source_object_id")
    if source_id:
        return f"source_object:{source_id}"
    return f"lineup_row:{row.get('lineup_matrix_row_id') or 'unknown'}"


def round_completed_portband_task_keys(action_memory: list[dict[str, Any]]) -> set[str]:
    completed: set[str] = set()
    for memory_row in action_memory[-8:]:
        primitive = str(memory_row.get("primitive") or "")
        if not primitive.startswith("move_"):
            continue
        progress = memory_row.get("goal_progress") or {}
        if (memory_row.get("levels_completed_delta") or 0) <= 0 and number(progress.get("distance_delta")) <= 0.0:
            continue
        rec = memory_row.get("selected_recommendation")
        refs = rec.get("evidence_refs") if isinstance(rec, dict) else None
        task_key = refs.get("portband_task_key") if isinstance(refs, dict) else None
        if task_key:
            completed.add(str(task_key))
    return completed


def portband_task_motion_summary(action_memory: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for idx, memory_row in enumerate(action_memory):
        primitive = str(memory_row.get("primitive") or "")
        if not primitive.startswith("move_"):
            continue
        rec = memory_row.get("selected_recommendation")
        refs = rec.get("evidence_refs") if isinstance(rec, dict) else None
        task_key = refs.get("portband_task_key") if isinstance(refs, dict) else None
        if not task_key:
            continue
        key = str(task_key)
        progress = memory_row.get("goal_progress") or {}
        distance_delta = number(progress.get("distance_delta"))
        levels_delta = number(memory_row.get("levels_completed_delta"))
        row = summary.setdefault(
            key,
            {
                "move_count": 0,
                "positive_move_count": 0,
                "no_progress_move_count": 0,
                "last_positive_action_index": None,
                "last_motion_action_index": None,
            },
        )
        row["move_count"] = int(row["move_count"]) + 1
        row["last_motion_action_index"] = idx
        if levels_delta > 0.0 or distance_delta > 0.0:
            row["positive_move_count"] = int(row["positive_move_count"]) + 1
            row["last_positive_action_index"] = idx
        else:
            row["no_progress_move_count"] = int(row["no_progress_move_count"]) + 1
    return summary


def saturated_portband_task_keys(action_memory: list[dict[str, Any]]) -> set[str]:
    return {
        task_key
        for task_key, summary in portband_task_motion_summary(action_memory).items()
        if int(summary.get("positive_move_count") or 0) > 0
    }


def recent_non_portband_probe_no_progress_count(action_memory: list[dict[str, Any]]) -> int:
    count = 0
    for memory_row in reversed(action_memory[-6:]):
        rec = memory_row.get("selected_recommendation")
        if not isinstance(rec, dict):
            continue
        candidate_kind = str(rec.get("candidate_kind") or "")
        evidence_sources = rec.get("evidence_sources") or []
        is_portband = (
            candidate_kind in {"multi_object_goal_adjacency_plan", "portband_lineup_matrix_row"}
            or "portband.lineup_matrix.rows" in evidence_sources
        )
        if is_portband:
            continue
        progress = memory_row.get("goal_progress") or {}
        if (memory_row.get("levels_completed_delta") or 0) > 0 or number(progress.get("distance_delta")) > 0.0:
            break
        count += 1
    return count


def portband_task_saturation_status(
    task_key: str,
    eligible_task_keys: set[str],
    action_memory: list[dict[str, Any]],
) -> dict[str, Any]:
    saturated_keys = saturated_portband_task_keys(action_memory)
    eligible_saturated = sorted(key for key in eligible_task_keys if key in saturated_keys)
    unsaturated_eligible = sorted(key for key in eligible_task_keys if key not in saturated_keys)
    is_saturated = task_key in saturated_keys
    alternatives_unsaturated = bool(unsaturated_eligible and task_key not in unsaturated_eligible)
    all_eligible_saturated = bool(eligible_task_keys and not unsaturated_eligible)
    post_saturation_probe_no_progress_count = recent_non_portband_probe_no_progress_count(action_memory)
    penalty = 0.0
    if is_saturated:
        penalty = 0.68 if alternatives_unsaturated else 0.52
        if post_saturation_probe_no_progress_count > 0:
            penalty = min(0.95, penalty + min(0.36, 0.18 * post_saturation_probe_no_progress_count))
    return {
        "task_saturation_policy": "post_positive_portband_task_revisit_penalty_without_route_replay",
        "task_saturated": is_saturated,
        "task_saturation_penalty": penalty,
        "task_saturation_positive_move_count": int(
            (portband_task_motion_summary(action_memory).get(task_key) or {}).get("positive_move_count") or 0
        ),
        "saturated_portband_task_keys": sorted(saturated_keys),
        "eligible_saturated_portband_task_keys": eligible_saturated,
        "eligible_unsaturated_portband_task_keys": unsaturated_eligible,
        "all_eligible_portband_tasks_saturated": all_eligible_saturated,
        "post_saturation_probe_no_progress_count": post_saturation_probe_no_progress_count,
        "post_saturation_reentry_policy": "recent_failed_non_portband_probe_increases_saturated_portband_reentry_penalty",
    }


def active_portband_task_key(action_memory: list[dict[str, Any]]) -> str | None:
    completed_task_keys = round_completed_portband_task_keys(action_memory)
    no_progress_moves = 0
    for memory_row in reversed(action_memory[-8:]):
        rec = memory_row.get("selected_recommendation")
        refs = rec.get("evidence_refs") if isinstance(rec, dict) else None
        task_key = refs.get("portband_task_key") if isinstance(refs, dict) else None
        if not task_key:
            continue
        primitive = str(memory_row.get("primitive") or "")
        if primitive.startswith("move_"):
            progress = memory_row.get("goal_progress") or {}
            if (memory_row.get("levels_completed_delta") or 0) > 0 or number(progress.get("distance_delta")) > 0.0:
                if str(task_key) in completed_task_keys:
                    return None
                return str(task_key)
            no_progress_moves += 1
            if no_progress_moves >= 2:
                return None
            continue
        if primitive == "select_object" and memory_row.get("status") == "executed" and memory_row.get("selected_display_coord_delta") != [0, 0]:
            if str(task_key) in completed_task_keys:
                return None
            return str(task_key)
    return None


def build_multi_object_goal_adjacency_plan(
    rows: list[dict[str, Any]],
    before_state: dict[str, Any],
    action_memory: list[dict[str, Any]],
) -> dict[str, Any] | None:
    active_task_key = active_portband_task_key(action_memory)
    completed_task_keys = round_completed_portband_task_keys(action_memory)

    def plan_metrics(selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
        row_scores = [lineup_row_score(row) for row in selected_rows]
        step_costs = [lineup_transform_step_cost(row) for row in selected_rows]
        edge_scores = [number(row.get("edge_lock_quality_score")) for row in selected_rows]
        sources = {str(row.get("source_object_id") or "") for row in selected_rows if row.get("source_object_id")}
        targets = {target_signature_for_lineup_row(row) for row in selected_rows if target_signature_for_lineup_row(row)}
        positive_count = sum(1 for score in row_scores if score > 0.0)
        total_steps = sum(step_costs)
        average_steps = total_steps / max(1, len(step_costs))
        step_efficiency = 1.0 / (1.0 + average_steps / 16.0)
        independent_ratio = min(len(sources), len(targets)) / max(len(selected_rows), 1)
        aggregate = clamp01(
            0.45 * (sum(row_scores) / max(1, len(row_scores)))
            + 0.25 * independent_ratio
            + 0.20 * step_efficiency
            + 0.10 * (sum(edge_scores) / max(1, len(edge_scores)))
        )
        return {
            "positive_signal_count": positive_count,
            "independent_source_count": len(sources),
            "independent_target_count": len(targets),
            "total_step_cost": total_steps,
            "average_step_cost": average_steps,
            "step_efficiency_score": step_efficiency,
            "independent_goal_adjacency_ratio": positive_count / max(1.0, total_steps),
            "aggregate_signal_score": aggregate,
        }

    eligible: list[dict[str, Any]] = []
    known_source_available = any(lineup_definedness_score(row) >= 0.9 for row in rows if isinstance(row, dict))
    for row in rows:
        if not isinstance(row, dict):
            continue
        if known_source_available and lineup_definedness_score(row) < 0.9:
            continue
        source_id = str(row.get("source_object_id") or "")
        target_key = target_signature_for_lineup_row(row)
        if not source_id or not target_key or lineup_row_score(row) <= 0.0:
            continue
        if row.get("source_object_role") not in {"support_group", "actor", "unknown"}:
            continue
        eligible.append(row)
    eligible_task_keys = {portband_task_key_for_lineup_row(row) for row in eligible}
    eligible.sort(
        key=lambda row: (
            0 if active_task_key and portband_task_key_for_lineup_row(row) == active_task_key else 1,
            1 if portband_task_key_for_lineup_row(row) in saturated_portband_task_keys(action_memory) else 0,
            1 if portband_task_key_for_lineup_row(row) in completed_task_keys else 0,
            -lineup_known_task_priority(row),
            -(lineup_row_score(row) / max(1.0, lineup_transform_step_cost(row))),
            lineup_transform_step_cost(row),
            -lineup_value_from_signals(row),
            str(row.get("source_object_id") or ""),
            str(row.get("lineup_matrix_row_id") or ""),
        )
    )

    selected: list[dict[str, Any]] = []
    used_sources: set[str] = set()
    used_targets: set[str] = set()
    for row in eligible:
        source_id = str(row.get("source_object_id") or "")
        target_key = target_signature_for_lineup_row(row)
        if source_id in used_sources or target_key in used_targets:
            continue
        candidate_selected = selected + [row]
        if len(candidate_selected) > 2:
            prior_score = float(plan_metrics(selected)["aggregate_signal_score"])
            candidate_score = float(plan_metrics(candidate_selected)["aggregate_signal_score"])
            if candidate_score < prior_score:
                continue
        selected = candidate_selected
        used_sources.add(source_id)
        used_targets.add(target_key)
        if len(selected) >= 4:
            break
    if len(selected) < 2:
        return None

    task_keys_with_motion = task_keys_with_motion_transform(eligible)
    selected = [
        row
        for row in selected
        if lineup_has_motion_transform(row) or portband_task_key_for_lineup_row(row) not in task_keys_with_motion
    ]
    if len(selected) < 2:
        return None

    metrics = plan_metrics(selected)
    if float(metrics["average_step_cost"]) > 24.0:
        return None
    aggregate_value = float(metrics["aggregate_signal_score"])

    action_rows: list[tuple[float, dict[str, Any], str, dict[str, Any]]] = []
    for row in selected:
        primitive, target_override, selection_status = primitive_and_target_for_lineup_row(
            row,
            before_state=before_state,
            action_memory=action_memory,
        )
        if primitive == "select_object":
            repeat_penalty = min(
                0.55,
                0.34
                * repeated_target_no_progress_count(
                    action_memory,
                    primitive=primitive,
                    component_id=str((target_override or {}).get("component_id") or ""),
                ),
            )
        else:
            repeat_penalty = min(0.35, 0.12 * repeated_no_progress_count(action_memory, primitive)) if primitive.startswith("move_") else 0.0
        selection_priority_bonus = 0.0
        task_key = portband_task_key_for_lineup_row(row)
        active_task_bonus = 0.18 if active_task_key and task_key == active_task_key else 0.0
        inactive_task_penalty = 0.16 if active_task_key and task_key != active_task_key else 0.0
        round_completed_task_penalty = 0.24 if task_key in completed_task_keys else 0.0
        saturation_status = portband_task_saturation_status(task_key, eligible_task_keys, action_memory)
        task_saturation_penalty = float(saturation_status["task_saturation_penalty"])
        saturation_refinement_bonus = post_saturation_refinement_bonus(row, saturation_status)
        refinement_no_progress_count = post_saturation_refinement_no_progress_count(action_memory, task_key)
        refinement_failure_penalty = 0.62 if saturation_refinement_bonus > 0.0 and refinement_no_progress_count >= 1 else 0.0
        unsaturated_queue_bonus = (
            0.14
            if completed_task_keys
            and not active_task_key
            and task_key not in completed_task_keys
            and not saturation_status["task_saturated"]
            else 0.0
        )
        undefined_penalty = clamp01(1.0 - lineup_definedness_score(row)) * 0.18
        action_rows.append(
            (
                clamp01(
                    lineup_value_from_signals(row)
                    + selection_priority_bonus
                    + active_task_bonus
                    + unsaturated_queue_bonus
                    + saturation_refinement_bonus
                    - repeat_penalty
                    - refinement_failure_penalty
                    - inactive_task_penalty
                    - round_completed_task_penalty
                    - task_saturation_penalty
                    - undefined_penalty
                ),
                row,
                primitive,
                {
                    "repeat_no_progress_penalty": repeat_penalty,
                    "selection_priority_bonus": selection_priority_bonus,
                    "active_task_bonus": active_task_bonus,
                    "unsaturated_queue_bonus": unsaturated_queue_bonus,
                    "post_saturation_refinement_bonus": saturation_refinement_bonus,
                    "post_saturation_refinement_no_progress_count": refinement_no_progress_count,
                    "post_saturation_refinement_failure_penalty": refinement_failure_penalty,
                    "inactive_task_penalty": inactive_task_penalty,
                    "round_completed_task_penalty": round_completed_task_penalty,
                    "task_saturation_penalty": task_saturation_penalty,
                    "task_saturation_status": saturation_status,
                    "undefined_task_penalty": undefined_penalty,
                    "target_override": target_override,
                    "selection_status": selection_status,
                },
            )
        )
    action_rows.sort(
        key=lambda item: (
            -item[0],
            lineup_transform_step_cost(item[1]),
            str(item[1].get("source_object_id") or ""),
            str(item[1].get("lineup_matrix_row_id") or ""),
        )
    )
    action_score, action_row, primitive, action_meta = action_rows[0]
    if action_score <= 0.0 or primitive == "activate_flow":
        return None
    aggregate_value = clamp01(action_score)
    return {
        "status": "ranked",
        "primitive": primitive,
        "value_score": aggregate_value,
        "action_row": action_row,
        "member_rows": selected,
        "positive_signal_count": metrics["positive_signal_count"],
        "independent_source_count": metrics["independent_source_count"],
        "independent_target_count": metrics["independent_target_count"],
        "total_step_cost": round(float(metrics["total_step_cost"]), 6),
        "average_step_cost": round(float(metrics["average_step_cost"]), 6),
        "step_efficiency_score": round(float(metrics["step_efficiency_score"]), 6),
        "independent_goal_adjacency_ratio": round(float(metrics["independent_goal_adjacency_ratio"]), 6),
        "aggregate_signal_score": round(aggregate_value, 6),
        "action_repeat_no_progress_penalty": action_meta["repeat_no_progress_penalty"],
        "action_selection_priority_bonus": action_meta["selection_priority_bonus"],
        "action_active_task_bonus": action_meta["active_task_bonus"],
        "action_unsaturated_queue_bonus": action_meta["unsaturated_queue_bonus"],
        "action_post_saturation_refinement_bonus": action_meta["post_saturation_refinement_bonus"],
        "action_post_saturation_refinement_no_progress_count": action_meta["post_saturation_refinement_no_progress_count"],
        "action_post_saturation_refinement_failure_penalty": action_meta["post_saturation_refinement_failure_penalty"],
        "action_inactive_task_penalty": action_meta["inactive_task_penalty"],
        "action_round_completed_task_penalty": action_meta["round_completed_task_penalty"],
        "action_task_saturation_penalty": action_meta["task_saturation_penalty"],
        "action_task_saturation_status": action_meta["task_saturation_status"],
        "action_undefined_task_penalty": action_meta["undefined_task_penalty"],
        "active_portband_task_key": active_task_key,
        "round_completed_portband_task_keys": sorted(completed_task_keys),
        "target_override": action_meta.get("target_override"),
        "selection_status": action_meta.get("selection_status"),
    }


def portband_lineup_matrix_candidates(
    contra_packet: dict[str, Any],
    *,
    before_state: dict[str, Any],
    action_memory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    rows = []
    active_route_key = active_portband_route_key(action_memory)
    blocked_routes = blocked_portband_route_keys(action_memory)
    for row in collect_portband_lineup_matrix_rows(contra_packet):
        row_id = str(row.get("lineup_matrix_row_id") or "")
        if not row_id or row_id in seen:
            continue
        if route_key_for_lineup_row(row) in blocked_routes:
            continue
        seen.add(row_id)
        rows.append(row)
    aggregate_plan = build_multi_object_goal_adjacency_plan(rows, before_state, action_memory)
    all_lineup_rows = list(rows)
    row_task_keys_with_motion = task_keys_with_motion_transform(all_lineup_rows)
    if active_route_key and any(route_key_for_lineup_row(row) == active_route_key for row in rows):
        rows = [row for row in rows if route_key_for_lineup_row(row) == active_route_key]

    candidates: list[dict[str, Any]] = []
    if aggregate_plan is not None:
        action_row = aggregate_plan["action_row"]
        admitted, evidence_refs, signals = movement_oscillation_gate(
            action_memory=action_memory,
            primitive=str(aggregate_plan["primitive"]),
            candidate_kind="multi_object_goal_adjacency_plan",
            evidence_refs={
                "lineup_matrix_row_id": action_row.get("lineup_matrix_row_id"),
                "portband_task_key": portband_task_key_for_lineup_row(action_row),
                "active_portband_task_key": aggregate_plan["active_portband_task_key"],
                "lineage_key": lineage_key_for_lineup_row(action_row),
                "portband_route_key": route_key_for_lineup_row(action_row),
                "active_portband_route_key": active_route_key,
                "blocked_portband_route_keys": sorted(blocked_routes),
                "source_component_ids": action_row.get("source_component_ids", []),
                "target_component_ids": action_row.get("target_component_ids", []),
                "target_object_ids": action_row.get("target_object_ids", []),
                "target_port_ids": action_row.get("target_port_ids", []),
                "aggregate_member_lineup_matrix_row_ids": [
                    row.get("lineup_matrix_row_id") for row in aggregate_plan["member_rows"]
                ],
                "aggregate_source_object_ids": [row.get("source_object_id") for row in aggregate_plan["member_rows"]],
                "aggregate_target_signatures": [
                    target_signature_for_lineup_row(row) for row in aggregate_plan["member_rows"]
                ],
                "route_authority": "none_static_review_only",
                "lineup_route_status": action_row.get("lineup_route_status"),
            },
            signals={
                "optimum_signal_type": "multi_object_independent_goal_adjacency_least_steps",
                "positive_signal_count": aggregate_plan["positive_signal_count"],
                "independent_source_count": aggregate_plan["independent_source_count"],
                "independent_target_count": aggregate_plan["independent_target_count"],
                "total_step_cost": aggregate_plan["total_step_cost"],
                "average_step_cost": aggregate_plan["average_step_cost"],
                "step_efficiency_score": aggregate_plan["step_efficiency_score"],
                "independent_goal_adjacency_ratio": aggregate_plan["independent_goal_adjacency_ratio"],
                "aggregate_signal_score": aggregate_plan["aggregate_signal_score"],
                "action_repeat_no_progress_penalty": aggregate_plan["action_repeat_no_progress_penalty"],
                "action_selection_priority_bonus": aggregate_plan["action_selection_priority_bonus"],
                "action_active_task_bonus": aggregate_plan["action_active_task_bonus"],
                "action_unsaturated_queue_bonus": aggregate_plan["action_unsaturated_queue_bonus"],
                "action_post_saturation_refinement_bonus": aggregate_plan["action_post_saturation_refinement_bonus"],
                "action_post_saturation_refinement_no_progress_count": aggregate_plan["action_post_saturation_refinement_no_progress_count"],
                "action_post_saturation_refinement_failure_penalty": aggregate_plan["action_post_saturation_refinement_failure_penalty"],
                "action_inactive_task_penalty": aggregate_plan["action_inactive_task_penalty"],
                "action_round_completed_task_penalty": aggregate_plan["action_round_completed_task_penalty"],
                "action_task_saturation_penalty": aggregate_plan["action_task_saturation_penalty"],
                **(
                    aggregate_plan["action_task_saturation_status"]
                    if isinstance(aggregate_plan.get("action_task_saturation_status"), dict)
                    else {}
                ),
                "action_undefined_task_penalty": aggregate_plan["action_undefined_task_penalty"],
                "lineup_definedness_score": lineup_definedness_score(action_row),
                "lineup_known_task_priority_score": lineup_known_task_priority(action_row),
                "active_portband_task_key": aggregate_plan["active_portband_task_key"],
                "round_completed_portband_task_keys": aggregate_plan["round_completed_portband_task_keys"],
                "source_selection_status": aggregate_plan.get("selection_status"),
                "source_selection_priority_sort": "dynamic_known_portband_task_queue_before_explore",
                "transform_hypothesis_xy": action_row.get("transform_hypothesis_xy"),
                "confidence_model": "multi_object_positive_signal_per_step_not_probability",
                "confidence_is_probability": False,
                **orthogonal_axis_correction_signal(
                    action_row,
                    primitive=str(aggregate_plan["primitive"]),
                    action_memory=action_memory,
                ),
            },
        )
        value = float(aggregate_plan["value_score"]) if admitted else 0.0
        candidates.append(
            recommendation_row(
                rank=len(candidates) + 1,
                primitive=str(aggregate_plan["primitive"]),
                confidence=value,
                value_score=value,
                candidate_kind="multi_object_goal_adjacency_plan",
                evidence_sources=["portband.lineup_matrix.rows", "portband.multi_object_goal_adjacency_plan"],
                evidence_refs=evidence_refs,
                signals=signals,
                target_override=aggregate_plan.get("target_override") if isinstance(aggregate_plan.get("target_override"), dict) else None,
                reason="multi_object_independent_goal_adjacency_least_steps_value"
                if admitted
                else "movement_oscillation_hard_blocked",
                expected_effect_tags=[
                    "multi_object_goal_adjacency",
                    "least_steps_positive_signal_ratio",
                    "relationship_matrix_row",
                ],
            )
        )
    for row in rows:
        row_id = str(row.get("lineup_matrix_row_id") or "")
        if not row_id:
            continue
        if any(lineup_definedness_score(candidate_row) >= 0.9 for candidate_row in all_lineup_rows) and lineup_definedness_score(row) < 0.9:
            continue
        score = lineup_row_score(row)
        value = lineup_value_from_signals(row)
        if value <= 0.0:
            continue
        primitive, target_override, selection_status = primitive_and_target_for_lineup_row(
            row,
            before_state=before_state,
            action_memory=action_memory,
        )
        lineage_key = lineage_key_for_lineup_row(row)
        route_key = route_key_for_lineup_row(row)
        repeat_penalty = 0.0
        selection_priority_bonus = 0.0
        active_task_key = active_portband_task_key(action_memory)
        completed_task_keys = round_completed_portband_task_keys(action_memory)
        row_task_keys = {portband_task_key_for_lineup_row(row) for row in all_lineup_rows}
        task_key = portband_task_key_for_lineup_row(row)
        active_task_bonus = 0.18 if active_task_key and task_key == active_task_key else 0.0
        inactive_task_penalty = 0.16 if active_task_key and task_key != active_task_key else 0.0
        round_completed_task_penalty = 0.24 if task_key in completed_task_keys else 0.0
        saturation_status = portband_task_saturation_status(task_key, row_task_keys, action_memory)
        task_saturation_penalty = float(saturation_status["task_saturation_penalty"])
        saturation_refinement_bonus = post_saturation_refinement_bonus(row, saturation_status)
        refinement_no_progress_count = post_saturation_refinement_no_progress_count(action_memory, task_key)
        refinement_failure_penalty = 0.62 if saturation_refinement_bonus > 0.0 and refinement_no_progress_count >= 1 else 0.0
        unsaturated_queue_bonus = (
            0.14
            if completed_task_keys
            and not active_task_key
            and task_key not in completed_task_keys
            and not saturation_status["task_saturated"]
            else 0.0
        )
        undefined_penalty = clamp01(1.0 - lineup_definedness_score(row)) * 0.18
        zero_transform_motion_alternative_penalty = (
            0.70
            if not lineup_has_motion_transform(row)
            and task_key in row_task_keys_with_motion
            else 0.0
        )
        if primitive == "select_object":
            repeat_penalty = min(
                0.55,
                0.34
                * repeated_target_no_progress_count(
                    action_memory,
                    primitive=primitive,
                    component_id=str((target_override or {}).get("component_id") or ""),
                ),
            )
        elif primitive.startswith("move_"):
            repeat_penalty = min(0.35, 0.12 * repeated_no_progress_count(action_memory, primitive))
        value = clamp01(
            value
            + selection_priority_bonus
            + active_task_bonus
            + unsaturated_queue_bonus
            + saturation_refinement_bonus
            - repeat_penalty
            - refinement_failure_penalty
            - inactive_task_penalty
            - round_completed_task_penalty
            - task_saturation_penalty
            - undefined_penalty
            - zero_transform_motion_alternative_penalty
        )
        evidence_refs = {
            "lineup_matrix_row_id": row_id,
            "portband_task_key": task_key,
            "active_portband_task_key": active_task_key,
            "lineage_key": lineage_key,
            "portband_route_key": route_key,
            "active_portband_route_key": active_route_key,
            "blocked_portband_route_keys": sorted(blocked_routes),
            "source_object_id": row.get("source_object_id"),
            "source_bbox_xyxy": row.get("source_bbox_xyxy"),
            "source_component_ids": row.get("source_component_ids", []),
            "target_component_ids": row.get("target_component_ids", []),
            "target_object_ids": row.get("target_object_ids", []),
            "target_port_ids": row.get("target_port_ids", []),
            "linked_alignment_ids": row.get("linked_alignment_ids", []),
            "linked_fleet_candidate_ids": row.get("linked_fleet_candidate_ids", []),
            "edge_lock_configuration_ids": row.get("edge_lock_configuration_ids", []),
            "edge_lock_top_configuration_id": row.get("edge_lock_top_configuration_id"),
            "edge_lock_rank": row.get("edge_lock_rank"),
            "edge_lock_kind": row.get("edge_lock_kind"),
            "edge_lock_policy": row.get("edge_lock_policy"),
            "route_authority": row.get("route_authority"),
            "lineup_route_status": row.get("lineup_route_status"),
        }
        signals = {
            "lineup_matrix_score": number(score),
            "lineup_step_cost": lineup_transform_step_cost(row),
            "lineup_step_efficiency_score": lineup_step_efficiency(row),
            "source_selection_status": selection_status,
            "edge_lock_status": row.get("edge_lock_status"),
            "edge_lock_quality_score": number(row.get("edge_lock_quality_score")),
            "edge_lock_adjusted_confidence_score": number(row.get("edge_lock_adjusted_confidence_score")),
            "potential_solution_signal_status": row.get("potential_solution_signal_status"),
            "transform_hypothesis_xy": row.get("transform_hypothesis_xy"),
            "confidence_model": row.get("confidence_model"),
            "confidence_is_probability": row.get("confidence_is_probability"),
            "alignment_kind": row.get("alignment_kind") or (row.get("score_terms") or {}).get("alignment_kind"),
            "match_status": (row.get("score_terms") or {}).get("match_status"),
            "repeat_no_progress_penalty": repeat_penalty,
            "selection_priority_bonus": selection_priority_bonus,
            "active_task_bonus": active_task_bonus,
            "unsaturated_queue_bonus": unsaturated_queue_bonus,
            "unsaturated_queue_policy": "post_positive_portband_progress_prioritizes_remaining_unsaturated_tasks_before_generic_probes",
            "post_saturation_refinement_bonus": saturation_refinement_bonus,
            "post_saturation_refinement_policy": "all_tasks_positive_allows_high_confidence_residual_portband_motion_before_generic_probes",
            "post_saturation_refinement_no_progress_count": refinement_no_progress_count,
            "post_saturation_refinement_failure_penalty": refinement_failure_penalty,
            "post_saturation_refinement_failure_policy": "repeated_no_progress_refinement_defers_same_portband_task",
            "inactive_task_penalty": inactive_task_penalty,
            "round_completed_task_penalty": round_completed_task_penalty,
            "task_saturation_penalty": task_saturation_penalty,
            "undefined_task_penalty": undefined_penalty,
            "zero_transform_motion_alternative_penalty": zero_transform_motion_alternative_penalty,
            "zero_transform_motion_alternative_policy": "same_source_motion_transform_rows_outrank_zero_transform_activate_flow",
            "lineup_definedness_score": lineup_definedness_score(row),
            "lineup_known_task_priority_score": lineup_known_task_priority(row),
            "round_completed_portband_task_keys": sorted(completed_task_keys),
            **saturation_status,
            "source_selection_priority_sort": "dynamic_known_portband_task_queue_before_explore",
            **orthogonal_axis_correction_signal(
                row,
                primitive=primitive,
                action_memory=action_memory,
            ),
        }
        admitted, evidence_refs, signals = movement_oscillation_gate(
            action_memory=action_memory,
            primitive=primitive,
            candidate_kind="portband_lineup_matrix_row",
            evidence_refs=evidence_refs,
            signals=signals,
        )
        if not admitted:
            value = 0.0
        candidates.append(
            recommendation_row(
                rank=len(candidates) + 1,
                primitive=primitive,
                confidence=value,
                value_score=value,
                candidate_kind="portband_lineup_matrix_row",
                evidence_sources=["portband.lineup_matrix.rows"],
                evidence_refs=evidence_refs,
                signals=signals,
                target_override=target_override,
                reason="movement_oscillation_hard_blocked" if not admitted else "portband_lineup_matrix_row_value",
                expected_effect_tags=[
                    "movement_oscillation_hard_blocked",
                    "portband_edge_alignment",
                    "relationship_matrix_row",
                ]
                if not admitted
                else ["portband_edge_alignment", "relationship_matrix_row"],
            )
        )
    return candidates


def portband_signal_candidates(contra_packet: dict[str, Any], action_memory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    budget = ((contra_packet.get("deltaband_gate") or {}).get("exploreband_budget") or {})
    queue = budget.get("candidate_probe_queue") or []
    candidates: list[dict[str, Any]] = []
    for idx, candidate in enumerate(queue):
        if not isinstance(candidate, dict):
            continue
        refs = candidate.get("source_evidence_refs") or {}
        scores = refs.get("signal_scores") or {}
        target = candidate.get("proposed_observation_target") or {}
        portband_score = number(scores.get("portband_score"))
        lineup_score = number(scores.get("portband_lineup_confidence_score"))
        route_flow_score = number(scores.get("portband_route_flow_confidence_score"))
        action_state_score = number(scores.get("portband_action_state_correlation_score"))
        support_score = number(scores.get("support_score") or candidate.get("support_score"))
        expected_gain = number(candidate.get("expected_information_gain"))
        curiosity = number(candidate.get("curiosity_score") or candidate.get("probe_value_score"))
        risk = number(candidate.get("risk_score"))
        value = clamp01(
            0.23 * portband_score
            + 0.18 * lineup_score
            + 0.18 * max(route_flow_score, action_state_score)
            + 0.14 * support_score
            + 0.16 * expected_gain
            + 0.11 * curiosity
            - 0.18 * risk
        )
        if value <= 0.0:
            continue
        click_xy = target.get("click_xy")
        component_id = str(target.get("component_id") or candidate.get("component_id") or "")
        primitive = "select_object" if click_xy is not None else "activate_flow"
        repeat_penalty = min(0.55, 0.34 * repeated_target_no_progress_count(
            action_memory,
            primitive=primitive,
            component_id=component_id,
        ))
        value = clamp01(value - repeat_penalty)
        target_override = {
            "target_kind": target.get("target_kind") or "component",
            "component_id": component_id or None,
            "click_xy": click_xy,
            "selection_basis": target.get("selection_basis") or "contra_value_scored_candidate",
        }
        signals = {
            "portband_score": portband_score,
            "portband_lineup_confidence_score": lineup_score,
            "portband_route_flow_confidence_score": route_flow_score,
            "portband_action_state_correlation_score": action_state_score,
            "support_score": support_score,
            "expected_information_gain": expected_gain,
            "curiosity_score": curiosity,
            "risk_penalty": risk,
            "repeat_no_progress_penalty": repeat_penalty,
        }
        evidence_refs, signals = dict(refs), dict(signals)
        admitted, evidence_refs, signals = movement_oscillation_gate(
            action_memory=action_memory,
            primitive=primitive,
            candidate_kind=str(candidate.get("probe_kind") or "contra_value_scored_probe"),
            evidence_refs=evidence_refs,
            signals=signals,
        )
        # Emit-but-mark a gated probe with a zeroed value instead of dropping it: it sinks to the
        # bottom of the value-score sort (never selected) but stays auditable, mirroring the
        # portband_lineup_matrix builder and keeping the probe-queue row present whenever the
        # budget surfaced a high-value probe (sp80_verify_episode_run requires the row to exist).
        if not admitted:
            value = 0.0
        blocked_reason = (
            "stale_repetition_hard_blocked"
            if signals.get("stale_repetition_hard_blocked")
            else "movement_oscillation_hard_blocked"
        )
        candidates.append(
            recommendation_row(
                rank=idx + 1,
                primitive=primitive,
                confidence=value,
                value_score=value,
                candidate_kind=str(candidate.get("probe_kind") or "contra_value_scored_probe"),
                evidence_sources=[
                    "deltaband_gate.exploreband_budget.candidate_probe_queue",
                    "deltaband_gate.macroband_object_ranking",
                    "portband_signal_scores",
                ],
                evidence_refs=evidence_refs,
                signals=signals,
                target_override=target_override,
                reason=blocked_reason if not admitted else "value_scored_contra_probe_candidate",
                expected_effect_tags=["probe_gate_blocked", "selection_or_state_change"]
                if not admitted
                else ["controlled_probe", "selection_or_state_change"],
            )
        )
    return candidates


def ledger_signal_candidates(
    *,
    ledger: dict[str, Any] | None,
    action_memory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = ledger.get("rows") if isinstance(ledger, dict) else []
    candidates: list[dict[str, Any]] = []
    for idx, row in enumerate(rows or []):
        if not isinstance(row, dict):
            continue
        family = str(row.get("action_family_hypothesis") or "")
        probe_names = [str(name) for name in (row.get("probe_action_names") or [])]
        if family == "support_surface_probe" or "ACTION5" in probe_names:
            primitive = "activate_flow"
            base = 0.48
        elif family == "directional_motion_probe":
            primitive = primitive_from_direction_hypothesis(row.get("flow_direction_hypothesis")) or "activate_flow"
            base = 0.31
        else:
            primitive = "activate_flow"
            base = 0.26
        repeat_penalty = min(0.35, 0.18 * repeated_no_progress_count(action_memory, primitive))
        semantic_no_progress_count = semantic_probe_no_progress_count(
            action_memory,
            primitive=primitive,
            family=family,
        )
        semantic_positive_count = semantic_probe_positive_count(
            action_memory,
            primitive=primitive,
            family=family,
        )
        semantic_probe_blocked = semantic_no_progress_count >= 1 and primitive == "activate_flow"
        semantic_feedback_penalty = 0.64 if semantic_probe_blocked else min(0.36, 0.24 * semantic_no_progress_count)
        post_saturation_probe_no_progress_count = recent_non_portband_probe_no_progress_count(action_memory)
        directional_recovery_bonus = (
            min(0.42, 0.32 * max(1, post_saturation_probe_no_progress_count))
            if family == "directional_motion_probe"
            and primitive.startswith("move_")
            and saturated_portband_task_keys(action_memory)
            and post_saturation_probe_no_progress_count > 0
            else 0.0
        )
        directional_continuation_bonus = (
            min(0.38, 0.30 * semantic_positive_count)
            if family == "directional_motion_probe"
            and primitive.startswith("move_")
            and semantic_positive_count > 0
            else 0.0
        )
        stationary_selected_count = selected_object_stationary_move_count(action_memory, primitive)
        stationary_selected_penalty = (
            min(0.72, 0.56 * stationary_selected_count)
            if family == "directional_motion_probe"
            and primitive.startswith("move_")
            else 0.0
        )
        value = clamp01(
            base
            + directional_recovery_bonus
            + directional_continuation_bonus
            - repeat_penalty
            - semantic_feedback_penalty
            - stationary_selected_penalty
        )
        evidence_refs = {
            "ledger_row_id": row.get("ledger_row_id"),
            "source_component_id": row.get("source_component_id"),
            "probe_action_names": probe_names,
            "flow_direction_hypothesis": row.get("flow_direction_hypothesis"),
            "semantic_probe_feedback_scope": "recent_same_action_semantics_family_same_primitive_no_progress",
            "selected_object_stationary_feedback_scope": "recent_same_primitive_selected_display_delta_zero",
        }
        signals = {
            "ledger_base_score": base,
            "repeat_no_progress_penalty": repeat_penalty,
            "directional_probe_policy": "flow_direction_hypothesis_maps_to_matching_move_primitive",
            "directional_probe_mapped_primitive": primitive,
            "directional_recovery_bonus": directional_recovery_bonus,
            "directional_continuation_bonus": directional_continuation_bonus,
            "directional_recovery_policy": "post_saturation_failed_probe_promotes_directional_motion",
            "directional_selected_object_stationary_count": stationary_selected_count,
            "directional_selected_object_stationary_penalty": stationary_selected_penalty,
            "directional_selected_object_stationary_policy": "same_primitive_selected_object_no_motion_suppresses_generic_directional_probe",
            "semantic_probe_prior_positive_count": semantic_positive_count,
            "post_saturation_probe_no_progress_count": post_saturation_probe_no_progress_count,
            "semantic_probe_family": family,
            "semantic_probe_prior_no_progress_count": semantic_no_progress_count,
            "semantic_probe_feedback_hard_limit": 1,
            "semantic_probe_feedback_hard_blocked": semantic_probe_blocked,
            "semantic_probe_feedback_penalty": semantic_feedback_penalty,
            "semantic_probe_feedback_policy": "noncontiguous_same_family_activate_flow_loss_blocks_next_flow_probe",
        }
        admitted, evidence_refs, signals = movement_oscillation_gate(
            action_memory=action_memory,
            primitive=primitive,
            candidate_kind=f"action_semantics_ledger:{family or 'unknown'}",
            evidence_refs=evidence_refs,
            signals=signals,
        )
        if semantic_probe_blocked or not admitted:
            value = 0.0
        candidates.append(
            recommendation_row(
                rank=idx + 1,
                primitive=primitive,
                confidence=value,
                value_score=value,
                candidate_kind=f"action_semantics_ledger:{family or 'unknown'}",
                evidence_sources=["action_semantics_ledger", f"action_semantics_ledger.{family or 'unknown'}"],
                evidence_refs=evidence_refs,
                signals=signals,
                reason="semantic_probe_feedback_hard_blocked" if semantic_probe_blocked else "value_scored_action_semantics_probe",
                expected_effect_tags=["controlled_probe"],
            )
        )
    return candidates


def goal_motion_candidate(
    *,
    goal_condition: dict[str, Any],
    action_memory: list[dict[str, Any]],
) -> dict[str, Any] | None:
    nearest = goal_condition.get("nearest_goal") or {}
    direction = nearest.get("relative_display_direction")
    distance = nearest.get("display_manhattan_distance")
    primitive = primitive_toward_goal_condition(goal_condition)
    if primitive is None:
        return None
    distance_score = 0.24
    if isinstance(distance, (int, float)):
        distance_score = 0.28 if float(distance) > 3.0 else 0.18
    repeat_penalty = min(0.42, 0.2 * repeated_no_progress_count(action_memory, primitive))
    value = clamp01(distance_score - repeat_penalty)
    if value <= 0.0 and last_same_primitive_no_progress(action_memory, primitive):
        return None
    evidence_refs = {"nearest_goal": nearest}
    signals = {
        "goal_motion_base_score": distance_score,
        "repeat_no_progress_penalty": repeat_penalty,
    }
    admitted, evidence_refs, signals = movement_oscillation_gate(
        action_memory=action_memory,
        primitive=primitive,
        candidate_kind="goal_relation_motion_fallback",
        evidence_refs=evidence_refs,
        signals=signals,
    )
    if not admitted:
        return None
    return recommendation_row(
        rank=1,
        primitive=primitive,
        confidence=value,
        value_score=value,
        candidate_kind="goal_relation_motion_fallback",
        evidence_sources=["goal_condition.nearest_goal.relative_display_direction"],
        evidence_refs=evidence_refs,
        signals=signals,
        reason=f"value_scored_motion_toward_nearest_goal_{direction}",
        expected_effect_tags=["nearest_bucket_distance_reduced"],
    )


_INFORMER_DIRECTION_PRIMITIVE = {
    "up": "move_up",
    "down": "move_down",
    "left": "move_left",
    "right": "move_right",
}


def _informer_direction_primitive(direction: Any) -> str | None:
    if not isinstance(direction, str):
        return None
    return _INFORMER_DIRECTION_PRIMITIVE.get(direction.strip().lower())


def oracle_review_candidates(
    contra_packet: dict[str, Any],
    *,
    before_state: dict[str, Any],
    action_memory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Oracle informer -> value-scored candidate priors (evidence_tier 1, never truth).

    Oracle hypotheses are review-only (``review_hypothesis_not_truth_source``); they
    enter here as a *control input* the harness may try, with execution + deltaband
    providing the truth.  The evidence-tier ordering keeps them strictly below
    verified current-frame signals, so a review hypothesis can never override an
    execution-derived candidate -- it only matters when the verified sources are
    empty.
    """
    oracle_pass = contra_packet.get("oracle_pass")
    if not isinstance(oracle_pass, dict):
        return []
    if oracle_pass.get("evidence_role") != "review_hypothesis_not_truth_source":
        return []
    pairs: list[tuple[Any, str, dict[str, Any]]] = []
    for actor in (oracle_pass.get("actor_review_hypotheses") or [])[:4]:
        if not isinstance(actor, dict):
            continue
        direction = actor.get("flow_direction_hypothesis") or actor.get("oracle_key_to_lock_direction")
        pairs.append((
            direction,
            "oracle_pass.actor_review_hypotheses[].flow_direction_hypothesis",
            {"actor_id": actor.get("actor_id"),
             "direction_agrees_with_raw_geometry": actor.get("direction_agrees_with_raw_geometry") is True},
        ))
    for target in (oracle_pass.get("target_review_hypotheses") or [])[:4]:
        if not isinstance(target, dict):
            continue
        pairs.append((
            target.get("key_to_lock_direction"),
            "oracle_pass.target_review_hypotheses[].key_to_lock_direction",
            {"target_id": target.get("target_id")},
        ))
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for direction, basis, refs in pairs:
        primitive = _informer_direction_primitive(direction)
        if primitive is None or primitive in seen:
            continue
        base = 0.30 if refs.get("direction_agrees_with_raw_geometry") else 0.26
        evidence_refs = {"oracle_direction": direction, "evidence_role": "review_hypothesis_not_truth_source", **refs}
        admitted, evidence_refs, signals = movement_oscillation_gate(
            action_memory=action_memory,
            primitive=primitive,
            candidate_kind="oracle_review_hypothesis",
            evidence_refs=evidence_refs,
            signals={"oracle_review_base_score": base},
        )
        if not admitted:
            continue
        repeat_penalty = min(0.2, 0.1 * repeated_no_progress_count(action_memory, primitive))
        value = clamp01(base - repeat_penalty)
        if value <= 0.0:
            continue
        seen.add(primitive)
        rows.append(
            recommendation_row(
                rank=1,
                primitive=primitive,
                confidence=value,
                value_score=value,
                candidate_kind="oracle_review_hypothesis",
                evidence_sources=[basis],
                evidence_refs=evidence_refs,
                signals=signals,
                reason=f"oracle_review_prior_motion_{direction}_not_truth_source",
                expected_effect_tags=["actor_moved_toward_oracle_goal_hypothesis"],
                evidence_tier=1,
            )
        )
    return rows


def ghost_relation_memory_candidates(
    contra_packet: dict[str, Any],
    *,
    before_state: dict[str, Any],
    action_memory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ghost informer -> value-scored fallback priors (evidence_tier 2, never truth).

    Confirmed relation memory from past controlled episodes (``relationship_memory_
    not_truth_source``).  Lowest priority: it only surfaces when verified current-
    frame signals AND Oracle review are absent (the L1 perception-starved case),
    keeping CONTRA proposing a geometry-consistent probe from a confirmed relation
    instead of the blind last-resort ``activate_flow``.
    """
    ghost_pass = contra_packet.get("ghost_pass")
    if not isinstance(ghost_pass, dict):
        return []
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in (ghost_pass.get("relation_memory_rows") or [])[:12]:
        if not isinstance(row, dict):
            continue
        geometry = row.get("geometry") if isinstance(row.get("geometry"), dict) else {}
        confirmation = row.get("confirmation") if isinstance(row.get("confirmation"), dict) else {}
        conf = clamp01(number(confirmation.get("confidence_score"), 0.5))
        family = str(row.get("relation_family") or "")
        direction = geometry.get("flow_direction")
        primitive = _informer_direction_primitive(direction)
        if primitive is None and (family == "activation_survived" or confirmation.get("survived_activation") is True):
            primitive = "activate_flow"
        if primitive is None or primitive in seen:
            continue
        base = clamp01(0.16 + 0.12 * conf)
        evidence_refs = {
            "ghost_relation_memory_id": row.get("relation_memory_id"),
            "relation_family": family,
            "flow_direction": direction,
            "confidence_score": conf,
            "boundary": "relationship_memory_not_action_authority",
        }
        signals = {"ghost_memory_base_score": base, "ghost_confidence_score": conf}
        if primitive.startswith("move_"):
            admitted, evidence_refs, signals = movement_oscillation_gate(
                action_memory=action_memory,
                primitive=primitive,
                candidate_kind="ghost_relation_memory",
                evidence_refs=evidence_refs,
                signals=signals,
            )
            if not admitted:
                continue
        repeat_penalty = min(0.16, 0.08 * repeated_no_progress_count(action_memory, primitive))
        value = clamp01(base - repeat_penalty)
        if value <= 0.0:
            continue
        seen.add(primitive)
        rows.append(
            recommendation_row(
                rank=1,
                primitive=primitive,
                confidence=value,
                value_score=value,
                candidate_kind="ghost_relation_memory",
                evidence_sources=["ghost_pass.relation_memory_rows[].geometry.flow_direction"],
                evidence_refs=evidence_refs,
                signals=signals,
                reason=f"ghost_confirmed_relation_prior_{primitive}_not_action_authority",
                expected_effect_tags=["geometry_consistent_probe_from_confirmed_relation"],
                evidence_tier=2,
            )
        )
    return rows


def build_contra_recommendation_packet(
    *,
    level_index: int,
    action_index: int,
    before_state: dict[str, Any],
    goal_condition: dict[str, Any],
    contra_packet: dict[str, Any],
    action_memory: list[dict[str, Any]],
) -> dict[str, Any]:
    """Convert CONTRA and local goal/memory signals into value-ranked primitives."""

    rows: list[dict[str, Any]] = []
    ledger = contra_packet.get("action_semantics_ledger")

    if before_state.get("selected") is not True:
        rows.append(
            recommendation_row(
                rank=1,
                primitive="select_object",
                confidence=0.66,
                value_score=0.66,
                candidate_kind="control_selection_establishment",
                evidence_sources=["live_selectable_sprite", "harness_target_selection"],
                evidence_refs={},
                signals={"selection_required_score": 0.66},
                reason="establish_controlled_selection_before_route_motion",
                expected_effect_tags=["selection_changed"],
            )
        )

    rows.extend(portband_lineup_matrix_candidates(contra_packet, before_state=before_state, action_memory=action_memory))
    rows.extend(portband_signal_candidates(contra_packet, action_memory))
    rows.extend(ledger_signal_candidates(ledger=ledger if isinstance(ledger, dict) else None, action_memory=action_memory))
    goal_candidate = goal_motion_candidate(goal_condition=goal_condition, action_memory=action_memory)
    if goal_candidate is not None:
        rows.append(goal_candidate)

    # Informer priors (evidence_tier 1/2): Oracle review + Ghost relation memory.
    # They are squeezed strictly below the lowest verified (tier-0) value via an
    # informer value ceiling, so a higher raw informer score can NEVER outrank a
    # verified candidate.  Combined with the pure value-score sort below this keeps
    # rows[0] == max(value) (required by sp80_verify_episode_run) while guaranteeing
    # informers only surface when the verified sources are empty (e.g. L1 perception
    # starvation), replacing the blind last-resort probe with a geometry-consistent
    # one rather than overriding execution-derived signals.
    verified_positive = [float(row.get("value_score") or 0.0) for row in rows if float(row.get("value_score") or 0.0) > 0.0]
    informer_ceiling = (min(verified_positive) - 1e-3) if verified_positive else None
    informer_rows: list[dict[str, Any]] = []
    informer_rows.extend(oracle_review_candidates(contra_packet, before_state=before_state, action_memory=action_memory))
    informer_rows.extend(ghost_relation_memory_candidates(contra_packet, before_state=before_state, action_memory=action_memory))
    for row in informer_rows:
        value = clamp01(float(row.get("value_score") or 0.0))
        if informer_ceiling is not None:
            value = min(value, informer_ceiling)
        if value <= 0.0:
            continue
        row["value_score"] = round(value, 6)
        row["confidence"] = round(min(float(row.get("confidence") or value), value), 6)
        if isinstance(row.get("signals"), dict):
            row["signals"]["informer_value_ceiling"] = informer_ceiling
        rows.append(row)

    if not rows or not any(float(row.get("value_score") or 0.0) > 0.0 for row in rows):
        rows.append(
            recommendation_row(
                rank=1,
                primitive="activate_flow",
                confidence=0.35,
                value_score=0.35,
                candidate_kind="last_resort_probe",
                evidence_sources=["contra_packet.no_value_scored_candidates"],
                evidence_refs={},
                signals={"last_resort_score": 0.35},
                reason="last_resort_probe_when_no_value_scored_candidate_available",
                expected_effect_tags=["flow_changed", "multi_frame_effect"],
            )
        )
    # STABLE_MULTI_SOURCE_RELAY_LOCK (HOLD): persist a committed relay leg before the final sort
    # so the controller stops thrashing across candidate_kinds on multi-source levels (L1).
    relay_lock_hold_bonus(rows, action_memory)
    surface_orthogonal_flow_redirect(rows, contra_packet)
    rows, target_scope_filter = filter_recommendations_to_explicit_goal_target(
        rows,
        goal_condition=goal_condition,
        contra_packet=contra_packet,
    )
    if not any("portband.lineup_matrix.rows" in (row.get("evidence_sources") or []) for row in rows):
        portband_context = target_scope_filtered_portband_context_row(contra_packet, target_scope_filter)
        if portband_context is not None:
            rows.append(portband_context)
    if not rows or not any(float(row.get("value_score") or 0.0) > 0.0 for row in rows):
        rows.append(
            recommendation_row(
                rank=1,
                primitive="activate_flow",
                confidence=0.35,
                value_score=0.35,
                candidate_kind="target_scoped_last_resort_probe",
                evidence_sources=["contra_packet.no_value_scored_candidates_after_target_scope_filter"],
                evidence_refs={},
                signals={
                    "last_resort_score": 0.35,
                    "target_scope_filter_status": target_scope_filter.get("status"),
                    "target_scope_filtered_count": target_scope_filter.get("filtered_count"),
                },
                reason="last_resort_probe_when_target_scoped_candidates_empty",
                expected_effect_tags=["flow_changed", "multi_frame_effect"],
            )
        )
    # Pure value-score ordering -- unchanged contract: rows[0] is the max-value row
    # (sp80_verify_episode_run requires this).  Informer priors were ceiling-capped
    # strictly below verified tier-0 above, so verified candidates always lead.
    rows = sorted(rows, key=lambda row: float(row.get("value_score") or 0.0), reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return {
        "schema": CONTRA_RECOMMENDATION_PACKET_SCHEMA,
        "recommendation_packet_id": f"sp80_l{level_index:02d}_a{action_index:03d}_contra_recommendations",
        "status": "ranked",
        "level_index": int(level_index),
        "action_index": int(action_index),
        "ranked_recommendations": rows,
        "top_primitive": rows[0]["primitive"],
        "ranking_policy": "value_scored_signal_modulation_no_static_family_priority",
        "target_scope_filter": target_scope_filter,
        "evidence_tier_policy": "verified_current_frame_tier0_gt_oracle_review_tier1_gt_ghost_memory_tier2",
        "top_evidence_tier": int(rows[0].get("evidence_tier") or 0),
        "informer_candidate_counts": {
            "oracle_review": sum(1 for row in rows if row.get("candidate_kind") == "oracle_review_hypothesis"),
            "ghost_relation_memory": sum(1 for row in rows if row.get("candidate_kind") == "ghost_relation_memory"),
        },
        "top_value_score": rows[0].get("value_score"),
        "source_contra_packet_sha256_basis": contra_packet.get("packet_sha256_basis"),
        "source_raw_frame_sha256": contra_packet.get("source_raw_frame_sha256"),
        "action_semantics_ledger_row_count": int(ledger.get("row_count") or 0)
        if isinstance(ledger, dict)
        else 0,
        "action_memory_count": len(action_memory),
        "recommendation_admitted_to_brain": True,
        "action_choice_admitted": False,
        "action_choice_admission_boundary": "recommendation_is_a_control_input_not_action_choice_truth",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "prediction_status": "not_predictive",
        "solve_condition_admitted": False,
    }


def save_raw_grid(frame: np.ndarray, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(frame, dtype=np.int16)
    np.save(path, arr)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "npy_sha256": sha256_file(path),
        "raw_frame_sha256": sha256_frame(arr),
        "shape": [int(v) for v in arr.shape],
        "dtype": str(arr.dtype),
    }


def save_frame_artifact(frame: np.ndarray, base_dir: Path, name: str, scale: int) -> dict[str, Any]:
    arr = np.asarray(frame)
    return {
        "png": save_frame(arr, base_dir / "png" / f"{name}.png", scale),
        "raw_grid": save_raw_grid(arr, base_dir / "raw_grid" / f"{name}.npy"),
    }


def coord_delta(before_state: dict[str, Any], after_state: dict[str, Any]) -> list[int] | None:
    before = coord_xy_to_grid_index(before_state.get("selected_coord"))
    after = coord_xy_to_grid_index(after_state.get("selected_coord"))
    delta = canon_coord_delta(before, after)
    if delta is None:
        return None
    return [delta[0], delta[1]]


def display_coord_delta(before_state: dict[str, Any], after_state: dict[str, Any]) -> list[int] | None:
    before = coord_xy_to_grid_index(before_state.get("selected_display_coord"))
    after = coord_xy_to_grid_index(after_state.get("selected_display_coord"))
    delta = canon_coord_delta(before, after)
    if delta is None:
        return None
    return [delta[0], delta[1]]


def resource_gauge_numeric_deltas(before_state: dict[str, Any], after_state: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for state_key, field_path, gauge_kind in (
        ("steps_remaining", "resource.fuel_gauge_steps_remaining", "fuel_gauge"),
    ):
        before_value = before_state.get(state_key)
        after_value = after_state.get(state_key)
        if not isinstance(before_value, (int, float)) or not isinstance(after_value, (int, float)):
            continue
        rows.append(
            {
                "field_path": field_path,
                "before_value": float(before_value),
                "after_value": float(after_value),
                "delta": round(float(after_value) - float(before_value), 6),
                "status": "observed_numeric_delta",
                "resource_gauge_kind": gauge_kind,
                "source_state_key": state_key,
            }
        )
    return rows


def goal_distance_for_object(goal_condition: dict[str, Any], object_id: str | None) -> float | None:
    if object_id is None:
        return None
    for obj in goal_condition.get("visible_goal_objects") or []:
        if obj.get("object_id") == object_id:
            distance = obj.get("display_manhattan_distance")
            return float(distance) if isinstance(distance, (int, float)) else None
    return None


def _paint_objective_distance(goal_condition: dict[str, Any]) -> float | None:
    paint = goal_condition.get("paint_objective")
    if not isinstance(paint, dict):
        return None
    for key in ("active_sink_paint_distance", "nearest_sink_paint_distance"):
        value = paint.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def goal_progress(before_goal: dict[str, Any], after_goal: dict[str, Any]) -> dict[str, Any]:
    target = before_goal.get("target_goal") or before_goal.get("nearest_goal") or {}
    target_object_id = target.get("object_id")
    before_distance = goal_distance_for_object(before_goal, target_object_id)
    after_distance = goal_distance_for_object(after_goal, target_object_id)
    distance_delta = None
    distance_reduced = None
    if before_distance is not None and after_distance is not None:
        distance_delta = float(before_distance - after_distance)
        distance_reduced = after_distance < before_distance
    # Measure-only payload->sink signal, emitted ALONGSIDE the effector->goal metric
    # above. The control/feedback objective is NOT flipped yet (measure-first).
    before_paint = _paint_objective_distance(before_goal)
    after_paint = _paint_objective_distance(after_goal)
    paint_distance_delta = None
    paint_distance_reduced = None
    if before_paint is not None and after_paint is not None:
        paint_distance_delta = float(before_paint - after_paint)
        paint_distance_reduced = after_paint < before_paint
    after_paint_obj = after_goal.get("paint_objective") if isinstance(after_goal.get("paint_objective"), dict) else {}
    return {
        "schema": "codexlab.sp80_goal_progress.v1",
        "condition": before_goal.get("condition"),
        "target_object_id": target_object_id,
        "target_bucket_index": before_goal.get("target_bucket_index"),
        "goal_selection_policy": before_goal.get("goal_selection_policy"),
        "before_display_manhattan_distance": before_distance,
        "after_display_manhattan_distance": after_distance,
        "distance_delta": distance_delta,
        "distance_reduced": distance_reduced,
        "paint_objective_measure": {
            "basis": "measure_only_payload_v6_to_sink_objective_flip_deferred",
            "before_paint_to_sink_distance": before_paint,
            "after_paint_to_sink_distance": after_paint,
            "paint_distance_delta": paint_distance_delta,
            "paint_distance_reduced": paint_distance_reduced,
            "spill_present_after": bool(after_paint_obj.get("spill_present")),
            "filled_sink_count_after": after_paint_obj.get("filled_sink_count"),
            "total_sink_count_after": after_paint_obj.get("total_sink_count"),
        },
    }


def build_view_state(
    *,
    phase: str,
    case_id: str,
    frame: np.ndarray,
    frame_artifact: dict[str, Any],
    state: dict[str, Any],
    goal_condition: dict[str, Any],
    goal_labels: dict[str, Any],
    target: SpriteRef,
) -> dict[str, Any]:
    selected = None
    if state.get("selected"):
        selected = {
            "semantic_label": "selected_control_object",
            "selected_name": state.get("selected_name"),
            "selected_coord": state.get("selected_coord"),
            "selected_display_coord": state.get("selected_display_coord"),
            "selected_shape": state.get("selected_shape"),
        }
    control_isolation_packet = build_control_isolation_packet(
        frame,
        game="sp80_liquid_pour",
        phase=phase,
        case_id=case_id,
        connectivity=4,
        min_area_pixels=1,
    )
    return {
        "schema": "codexlab.sp80_view_state.v1",
        "phase": phase,
        "case_id": case_id,
        "look_order": "before_action" if phase == "before" else "after_action",
        "raw_pixels_are_observation_ground_truth": True,
        "raw_frame": frame_artifact,
        "control_isolation_packet": control_isolation_packet,
        "control_isolation_required": True,
        "control_isolation_source": CONTROL_ISOLATION_SOURCE,
        "state_snapshot": state,
        "selected_object": selected,
        "target_object": {
            "semantic_label": "target_selectable_object",
            "name": target.name,
            "shape": [target.width, target.height],
            "tags": target.tags,
        },
        "goal_condition": goal_condition,
        "goal_labels": goal_labels,
        "object_view_source": "raw_frame_plus_contra_shape_trace_packet_plus_live_state_goal_relations",
        "sam_status": "not_run",
        "route_or_replay_used": False,
        "human_trace_used": False,
        "coordinate_script_used": False,
    }


def action_visual_correlation(
    *,
    case_id: str,
    expected_primitive: str | None,
    before_state: dict[str, Any],
    after_state: dict[str, Any],
    before_frame: np.ndarray,
    after_frame: np.ndarray,
    action_frame_count: int,
    before_goal: dict[str, Any],
    after_goal: dict[str, Any],
) -> dict[str, Any]:
    selected_delta = coord_delta(before_state, after_state)
    selected_display_delta = display_coord_delta(before_state, after_state)
    visual_delta = frame_delta(before_frame, after_frame)
    progress = goal_progress(before_goal, after_goal)
    expected_display_delta = {
        "move_up": [0, -1],
        "move_down": [0, 1],
        "move_left": [-1, 0],
        "move_right": [1, 0],
    }.get(str(expected_primitive))
    checks: list[dict[str, Any]] = [
        {"name": "look_before_recorded", "passed": True},
        {"name": "look_after_recorded", "passed": True},
        {"name": "raw_frame_shapes_match", "passed": visual_delta.get("shape_match") is True},
    ]
    if expected_display_delta is not None:
        checks.append(
            {
                "name": "selected_display_delta_matches_primitive",
                "passed": selected_display_delta == expected_display_delta,
                "expected": expected_display_delta,
                "observed": selected_display_delta,
            }
        )
        checks.append(
            {
                "name": "visible_pixels_changed",
                "passed": bool((visual_delta.get("changed_pixels") or 0) > 0),
                "changed_pixels": visual_delta.get("changed_pixels"),
            }
        )
    elif expected_primitive == "select_object":
        checks.append(
            {
                "name": "selection_visible_state_changed",
                "passed": after_state.get("selected") is True
                and after_state.get("selected_coord") is not None
                and bool((visual_delta.get("changed_pixels") or 0) > 0),
                "changed_pixels": visual_delta.get("changed_pixels"),
            }
        )
    elif expected_primitive == "activate_flow":
        checks.append(
            {
                "name": "flow_temporal_visual_sequence_recorded",
                "passed": action_frame_count > 1,
                "action_frame_count": action_frame_count,
            }
        )
    elif expected_primitive == "no_selected_motion":
        checks.append(
            {
                "name": "selected_display_delta_stationary",
                "passed": selected_display_delta == [0, 0],
                "expected": [0, 0],
                "observed": selected_display_delta,
            }
        )
    if case_id == "approach_nearest_bucket":
        checks.append(
            {
                "name": "nearest_bucket_distance_reduced",
                "passed": progress.get("distance_reduced") is True,
                "distance_delta": progress.get("distance_delta"),
            }
        )
    passed = all(bool(check.get("passed")) for check in checks)
    return {
        "schema": "codexlab.sp80_action_visual_correlation.v1",
        "status": "verified" if passed else "failed",
        "case_id": case_id,
        "primitive": expected_primitive,
        "view_before_required": True,
        "view_after_required": True,
        "control_isolation_before_required": True,
        "control_isolation_after_required": True,
        "visual_delta": visual_delta,
        "selected_coord_delta": selected_delta,
        "selected_display_coord_delta": selected_display_delta,
        "goal_progress": progress,
        "checks": checks,
    }


class SP80PlayerHarness:
    def __init__(
        self,
        *,
        controller: SP80BaseController,
        brain: BrainBackend,
        out_dir: Path,
        scale: int,
        strict: bool,
        target_bucket_index: int | None = None,
        compact_episode_artifacts: bool = False,
    ) -> None:
        self.controller = controller
        self.brain = brain
        self.out_dir = out_dir
        self.scale = int(scale)
        self.strict = bool(strict)
        self.target_bucket_index = target_bucket_index
        self.compact_episode_artifacts = bool(compact_episode_artifacts)
        self.frames_dir = out_dir / "frames"
        self.recorder = PacketRecorder(root=ROOT, out_dir=out_dir)
        self.packets_path = self.recorder.packets_path
        self.brain_calls_path = self.recorder.brain_calls_path

    def run_action_set(self) -> dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        packets: list[dict[str, Any]] = []
        brain_calls: list[dict[str, Any]] = []

        for case in ACTION_SET_CASES:
            packet, brain_call = self.run_case(case)
            packets.append(packet)
            brain_calls.append(brain_call)

        self.recorder.write_packets(packets)
        self.recorder.write_brain_calls(brain_calls)

        failures = [p for p in packets if p["status"] != "executed"]
        contra = contra_spec(
            backend=self.brain.name,
            game="sp80_liquid_pour",
            allowed_primitives=CONTROL_PRIMITIVES,
        )
        result = {
            "schema": "codexlab.sp80_player_harness.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "passed" if not failures else "failed",
            "failure_count": len(failures),
            "case_count": len(packets),
            "backend": self.brain.name,
            "player": SP80_PLAYER.to_packet(),
            "contra": contra,
            "condition": {
                "cwd": str(ROOT),
                "command": " ".join(sys.argv),
                "host": socket.gethostname(),
                "game_source": SP80_PATH.relative_to(ROOT).as_posix(),
                "game_source_sha256": sha256_file(SP80_PATH),
                "controller_source": "codexlab/sp80_base_controller.py",
                "controller_source_sha256": sha256_file(ROOT / "codexlab" / "sp80_base_controller.py"),
                "player_base_source": "codexlab/players/base.py",
                "player_base_source_sha256": sha256_file(ROOT / "codexlab" / "players" / "base.py"),
                "contra_source": "codexlab/contra_head/contra.py",
                "contra_source_sha256": sha256_file(ROOT / "codexlab" / "contra_head" / "contra.py"),
                "contra_module_sources": [
                    {"path": path, "sha256": sha256_file(ROOT / path)}
                    for path in contra_module_source_paths()
                ],
                "player_controller_source": SP80_PLAYER.controller_source,
                "player_controller_source_sha256": sha256_file(ROOT / SP80_PLAYER.controller_source),
                "player_registry_source": SP80_PLAYER.registry_source,
                "player_registry_source_sha256": sha256_file(ROOT / SP80_PLAYER.registry_source),
                "harness_source": "codexlab/sp80_player_harness.py",
                "harness_source_sha256": sha256_file(ROOT / "codexlab" / "sp80_player_harness.py"),
                "level_index": self.controller.level_index,
                "route_or_replay_used": False,
                "human_trace_used": False,
                "coordinate_script_used": False,
                "target_selection": "first live selectable sprite from game.fbrwmvzsym()",
                "goal_condition": "pour_into_buckets",
                "goal_label_source": "perceptual lock-goal color signature (yellow 11 / filled 13), verified against live game sprites tagged repwkzbkhxl",
                "goal_condition_target": "bucket",
                "look_first_before_every_action": True,
                "contra_control_isolation_required": True,
                "contra_control_isolation_source": CONTROL_ISOLATION_SOURCE,
                "contra_control_isolation_background_policy": BACKGROUND_SELECTION_POLICY,
                "look_after_every_action": True,
                "action_visual_correlation_required": True,
                "view_source": "raw frame artifact plus Contra control isolation packet plus visible state/object relation packets",
            },
            "allowed_primitives": CONTROL_PRIMITIVES,
            "sp80_action_set": self.controller.action_set(),
            "packets_jsonl": self.packets_path.relative_to(ROOT).as_posix(),
            "brain_calls_jsonl": self.brain_calls_path.relative_to(ROOT).as_posix(),
            "frames_dir": self.frames_dir.relative_to(ROOT).as_posix(),
            "case_status": {p["case_id"]: p["status"] for p in packets},
        }
        self.recorder.write_result(result)

        if failures and self.strict:
            failed_cases = ", ".join(p["case_id"] for p in failures)
            raise RuntimeError(f"harness failures under --strict: {failed_cases}")
        return result

    def run_episode(self, *, max_actions: int) -> dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        packets: list[dict[str, Any]] = []
        brain_calls: list[dict[str, Any]] = []
        action_memory: list[dict[str, Any]] = []
        game, obs = self.controller.new_game()
        ghost_context = load_ghost_context(
            run_dir=self.out_dir,
            root=ROOT,
            game_id="sp80-589a99af",
        )
        sprites = self.controller.selectable_sprites(game)
        if not sprites:
            raise RuntimeError("SP80 level has no selectable sprites")
        target = sprites[0]
        initial_state = self.controller.state(game, obs)
        initial_completed = int(initial_state.get("levels_completed") or 0)

        for action_index in range(max(1, int(max_actions))):
            full_packet, full_brain_call, obs = self.run_episode_action(
                game=game,
                obs=obs,
                target=target,
                action_index=action_index,
                max_actions=max(1, int(max_actions)),
                action_memory=action_memory,
                initial_completed=initial_completed,
                ghost_context=ghost_context,
            )
            packet = self.compact_episode_packet(full_packet) if self.compact_episode_artifacts else full_packet
            brain_call = self.compact_brain_call(full_brain_call) if self.compact_episode_artifacts else full_brain_call
            packets.append(packet)
            brain_calls.append(brain_call)
            action_memory.append(
                {
                    "action_index": action_index,
                    "primitive": full_packet.get("decision_primitive"),
                    "case_id": full_packet.get("case_id"),
                    "selected_recommendation": full_packet.get("selected_recommendation"),
                    "levels_completed_delta": full_packet.get("levels_completed_delta"),
                    "goal_progress": full_packet.get("goal_progress"),
                    "selected_coord_before": (full_packet.get("before_state") or {}).get("selected_coord")
                    if isinstance(full_packet.get("before_state"), dict)
                    else None,
                    "selected_coord_after": (full_packet.get("after_state") or {}).get("selected_coord")
                    if isinstance(full_packet.get("after_state"), dict)
                    else None,
                    "selected_display_coord_before": (full_packet.get("before_state") or {}).get("selected_display_coord")
                    if isinstance(full_packet.get("before_state"), dict)
                    else None,
                    "selected_display_coord_after": (full_packet.get("after_state") or {}).get("selected_display_coord")
                    if isinstance(full_packet.get("after_state"), dict)
                    else None,
                    "selected_coord_delta": full_packet.get("selected_coord_delta"),
                    "selected_display_coord_delta": full_packet.get("selected_display_coord_delta"),
                    "status": full_packet.get("status"),
                }
            )
            if full_packet.get("stop_reason") == "levels_completed_increased":
                break
            if full_packet.get("stop_reason") == "game_over":
                break
            if full_packet.get("status") != "executed":
                break

        self.recorder.write_packets(packets)
        self.recorder.write_brain_calls(brain_calls)
        final_state = initial_state
        for packet in packets:
            if packet.get("after_state") is not None:
                final_state = packet["after_state"]
        total_completed_delta = int((final_state or {}).get("levels_completed") or 0) - initial_completed
        terminal_status = packets[-1].get("status") if packets else "no_actions"
        terminal_reason = packets[-1].get("stop_reason") if packets else None
        result = {
            "schema": CONTRA_EPISODE_RESULT_SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "completed"
            if total_completed_delta > 0
            else "game_over"
            if terminal_reason == "game_over"
            else "budget_exhausted"
            if terminal_status == "executed"
            else str(terminal_status),
            "backend": self.brain.name,
            "mode": "episode",
            "level_index": self.controller.level_index,
            "max_actions": int(max_actions),
            "action_count": len(packets),
            "initial_state": initial_state,
            "final_state": final_state,
            "levels_completed_delta": total_completed_delta,
            "player": SP80_PLAYER.to_packet(),
            "contra": contra_spec(
                backend=self.brain.name,
                game="sp80_liquid_pour",
                allowed_primitives=CONTROL_PRIMITIVES,
            ),
            "condition": {
                "cwd": str(ROOT),
                "command": " ".join(sys.argv),
                "host": socket.gethostname(),
                "game_source": SP80_PATH.relative_to(ROOT).as_posix(),
                "game_source_sha256": sha256_file(SP80_PATH),
                "controller_source": "codexlab/sp80_base_controller.py",
                "controller_source_sha256": sha256_file(ROOT / "codexlab" / "sp80_base_controller.py"),
                "harness_source": "codexlab/sp80_player_harness.py",
                "harness_source_sha256": sha256_file(ROOT / "codexlab" / "sp80_player_harness.py"),
                "contra_source": "codexlab/contra_head/contra.py",
                "contra_source_sha256": sha256_file(ROOT / "codexlab" / "contra_head" / "contra.py"),
                "contra_module_sources": [
                    {"path": path, "sha256": sha256_file(ROOT / path)}
                    for path in contra_module_source_paths()
                ],
                "route_or_replay_used": False,
                "human_trace_used": False,
                "coordinate_script_used": False,
                "persistent_game_state": True,
                "contra_recommendation_packet_required": True,
                "brain_consumes_contra_recommendations": True,
                "compact_episode_artifacts": self.compact_episode_artifacts,
            },
            "allowed_primitives": CONTROL_PRIMITIVES,
            "sp80_action_set": self.controller.action_set(),
            "packets_jsonl": self.packets_path.relative_to(ROOT).as_posix(),
            "brain_calls_jsonl": self.brain_calls_path.relative_to(ROOT).as_posix(),
            "frames_dir": self.frames_dir.relative_to(ROOT).as_posix(),
        }
        ghost = write_ghost(
            result=result,
            packets=packets,
            run_dir=self.out_dir,
            root=ROOT,
            game_id="sp80-589a99af",
        )
        if ghost is not None:
            result["ghost_json"] = ghost["path"]
            result["ghost"] = ghost
        self.recorder.write_result(result)
        return result

    @staticmethod
    def compact_recommendation_packet(packet: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(packet, dict):
            return {}
        rows = [row for row in packet.get("ranked_recommendations") or [] if isinstance(row, dict)]
        return {
            "schema": packet.get("schema"),
            "recommendation_packet_id": packet.get("recommendation_packet_id"),
            "ranking_policy": packet.get("ranking_policy"),
            "top_primitive": packet.get("top_primitive"),
            "top_evidence_tier": packet.get("top_evidence_tier"),
            "top_value_score": packet.get("top_value_score"),
            "ranked_recommendations": rows[:1],
            "spill_planner_overlay": packet.get("spill_planner_overlay"),
            "compact_artifact": True,
            "boundary": "compact_episode_artifact_keeps_consumed_recommendation_not_full_static_contra_packet",
        }

    @classmethod
    def compact_brain_request(cls, request: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(request, dict):
            return {}
        observation = request.get("observation") if isinstance(request.get("observation"), dict) else {}
        return {
            "schema": request.get("schema"),
            "contra_schema": request.get("contra_schema"),
            "mode": request.get("mode"),
            "game": request.get("game"),
            "case_id": request.get("case_id"),
            "episode_budget": request.get("episode_budget"),
            "contra_recommendation_packet": cls.compact_recommendation_packet(
                request.get("contra_recommendation_packet")
            ),
            "action_memory": request.get("action_memory"),
            "allowed_primitives": request.get("allowed_primitives"),
            "sp80_action_set": request.get("sp80_action_set"),
            "observation": {
                "setup": observation.get("setup"),
                "state": observation.get("state"),
                "before_frame": observation.get("before_frame"),
                "target_sprite": observation.get("target_sprite"),
            },
            "compact_artifact": True,
        }

    @classmethod
    def compact_brain_call(cls, brain_call: dict[str, Any]) -> dict[str, Any]:
        request = cls.compact_brain_request(brain_call.get("request"))
        return {
            "schema": brain_call.get("schema"),
            "contra_schema": brain_call.get("contra_schema"),
            "case_id": brain_call.get("case_id"),
            "mode": brain_call.get("mode"),
            "contra": brain_call.get("contra"),
            "request": request,
            "decision": brain_call.get("decision"),
            "contra_request": request,
            "contra_decision": brain_call.get("contra_decision"),
            "status": brain_call.get("status"),
            "compact_artifact": True,
        }

    @classmethod
    def compact_episode_packet(cls, packet: dict[str, Any]) -> dict[str, Any]:
        before_contra = packet.get("contra_packet") if isinstance(packet.get("contra_packet"), dict) else {}
        after_contra = packet.get("after_contra_packet") if isinstance(packet.get("after_contra_packet"), dict) else {}
        keep_keys = [
            "schema",
            "contra_schema",
            "case_id",
            "mode",
            "status",
            "route_or_replay_used",
            "human_trace_used",
            "coordinate_script_used",
            "before_state",
            "after_state",
            "before_frame",
            "after_frame",
            "view_before",
            "view_after",
            "goal_condition",
            "after_goal_condition",
            "brain_decision",
            "contra_decision",
            "decision_primitive",
            "selected_recommendation",
            "action",
            "delta",
            "goal_progress",
            "selected_coord_delta",
            "selected_display_coord_delta",
            "action_frame_count",
            "action_png_sequence",
            "action_raw_grid_sequence",
            "deltaband_observations",
            "actionmodband_observations",
            "episode_memory_after_round",
            "levels_completed_delta",
            "total_levels_completed_delta",
            "stop_reason",
        ]
        compact = {key: packet.get(key) for key in keep_keys if key in packet}
        compact["contra_recommendation_packet"] = cls.compact_recommendation_packet(
            packet.get("contra_recommendation_packet")
        )
        compact["contra_packet"] = {
            "schema": before_contra.get("schema"),
            "compact_artifact": True,
            "omitted_fields": ["layer_output_packets", "view_system_passes", "portband_lineup_matrix"],
        }
        compact["after_contra_packet"] = {
            "schema": after_contra.get("schema"),
            "compact_artifact": True,
            "omitted_fields": ["layer_output_packets", "view_system_passes", "portband_lineup_matrix"],
        }
        compact["compact_artifact"] = True
        return compact

    def ground_episode_recommendation(
        self,
        *,
        game: Any,
        primitive: str,
        default_target: SpriteRef,
        target_override: dict[str, Any] | None,
    ) -> GroundedAction:
        if primitive == "select_object" and isinstance(target_override, dict):
            click_xy = target_override.get("click_xy")
            if (
                isinstance(click_xy, list)
                and len(click_xy) == 2
                and all(isinstance(v, (int, float)) for v in click_xy)
            ):
                data = {"x": int(round(float(click_xy[0]))), "y": int(round(float(click_xy[1])))}
                return GroundedAction(
                    primitive=primitive,
                    action=GameAction.ACTION6,
                    data=data,
                    selected_target=None,
                )
        return self.controller.ground_primitive(game, primitive, target=default_target)

    def run_episode_action(
        self,
        *,
        game: Any,
        obs: Any,
        target: SpriteRef,
        action_index: int,
        max_actions: int,
        action_memory: list[dict[str, Any]],
        initial_completed: int,
        ghost_context: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any], Any]:
        case_id = f"episode_action_{action_index:03d}"
        before_frame = np.asarray(obs.frame[-1])
        before_state = self.controller.state(game, obs)
        goal_labels = self.controller.bucket_goal_labels(game)
        goal_condition = self.controller.bucket_goal_condition(game, target_bucket_index=self.target_bucket_index)
        before_artifact = save_frame_artifact(
            before_frame,
            self.frames_dir / case_id,
            "before",
            self.scale,
        )
        view_before = build_view_state(
            phase="before",
            case_id=case_id,
            frame=before_frame,
            frame_artifact=before_artifact,
            state=before_state,
            goal_condition=goal_condition,
            goal_labels=goal_labels,
            target=target,
        )
        before_contra_packet = run_contra_view(
            raw_frame=before_frame,
            game_id="sp80-589a99af",
            level_index=int(self.controller.level_index),
            mode="episode_before_action",
            ghost_context=ghost_context,
        )
        recommendation_packet = build_contra_recommendation_packet(
            level_index=int(self.controller.level_index),
            action_index=action_index,
            before_state=before_state,
            goal_condition=goal_condition,
            contra_packet=before_contra_packet,
            action_memory=action_memory,
        )
        bind_episode = getattr(self.brain, "bind_episode", None)
        if callable(bind_episode):
            bind_episode(game=game, controller=self.controller, target=target, before_frame=before_frame)
        augment_recommendations = getattr(self.brain, "augment_recommendation_packet", None)
        if callable(augment_recommendations):
            recommendation_packet = augment_recommendations(
                recommendation_packet,
                before_state=before_state,
                action_memory=action_memory,
            )
        action_memory_snapshot = [dict(row) for row in action_memory if isinstance(row, dict)]
        remaining_action_budget = max(0, int(max_actions) - int(action_index))
        request = {
            "schema": "codexlab.sp80_brain_request.v1",
            "contra_schema": "codexlab.sp80_contra_request.v1",
            "mode": "episode",
            "game": "sp80_liquid_pour",
            "case_id": case_id,
            "episode_budget": {
                "schema": "codexlab.sp80_episode_budget.v1",
                "action_index": int(action_index),
                "max_actions": int(max_actions),
                "remaining_action_budget": int(remaining_action_budget),
                "boundary": "controlled_harness_budget_for_candidate_feasibility_not_goal_truth_or_solve_truth",
            },
            "intent": "Use CONTRA recommendation packet to choose the next primitive.",
            "view_state": view_before,
            "goal_condition": goal_condition,
            "contra_packet": before_contra_packet,
            "contra_recommendation_packet": recommendation_packet,
            "action_memory": action_memory_snapshot,
            "allowed_primitives": CONTROL_PRIMITIVES,
            "sp80_action_set": self.controller.action_set(),
            "observation": {
                "setup": "persistent_episode",
                "state": before_state,
                "before_frame": before_artifact,
                "view_before": view_before,
                "target_sprite": target.__dict__,
            },
            "brain_contract": contra_contract(),
            "contra_contract": contra_contract(),
        }
        decision = self.brain.decide(request)
        brain_call = {
            "schema": "codexlab.sp80_brain_call.v1",
            "contra_schema": "codexlab.sp80_contra_call.v1",
            "case_id": case_id,
            "mode": "episode",
            "contra": contra_spec(
                backend=self.brain.name,
                game="sp80_liquid_pour",
                allowed_primitives=CONTROL_PRIMITIVES,
            ),
            "request": request,
            "decision": decision.to_packet(),
            "contra_request": request,
            "contra_decision": decision.to_packet(),
            "goal_condition": goal_condition,
        }
        packet = {
            "schema": "codexlab.sp80_episode_harness_packet.v1",
            "status": "pending",
            "case_id": case_id,
            "mode": "episode",
            "action_index": int(action_index),
            "before_state": before_state,
            "before_frame": before_artifact,
            "view_before": view_before,
            "goal_condition": goal_condition,
            "goal_labels": goal_labels,
            "contra_packet": before_contra_packet,
            "contra_recommendation_packet": recommendation_packet,
            "brain_decision": decision.to_packet(),
            "contra_decision": decision.to_packet(),
            "decision_primitive": decision.primitive,
            "controlled_runtime": True,
            "route_or_replay_used": False,
            "human_trace_used": False,
            "coordinate_script_used": False,
        }
        if decision.error:
            packet["status"] = "brain_error"
            packet["error"] = decision.error
            brain_call["status"] = packet["status"]
            return packet, brain_call, obs
        if decision.primitive not in CONTROL_PRIMITIVES:
            packet["status"] = "invalid_primitive"
            packet["error"] = f"brain returned {decision.primitive!r}"
            brain_call["status"] = packet["status"]
            return packet, brain_call, obs

        try:
            selected_recommendation = selected_recommendation_from_decision(decision)
            target_override = selected_recommendation.get("target_override") if isinstance(selected_recommendation, dict) else None
            grounded = self.ground_episode_recommendation(
                game=game,
                primitive=str(decision.primitive),
                default_target=target,
                target_override=target_override if isinstance(target_override, dict) else None,
            )
            after_obs = self.controller.execute_grounded_action(game, grounded)
            after_frames = [np.asarray(frame) for frame in after_obs.frame]
            after_frame = after_frames[-1]
            after_artifact = save_frame_artifact(
                after_frame,
                self.frames_dir / case_id,
                "after",
                self.scale,
            )
            sequence_meta = None
            raw_sequence_meta = None
            if len(after_frames) > 1:
                sequence_meta = save_frame_sequence(
                    after_frames,
                    self.frames_dir / case_id / "png_sequence",
                    self.scale,
                )
                raw_sequence_meta = [
                    save_raw_grid(frame, self.frames_dir / case_id / "raw_grid_sequence" / f"frame_{idx:03d}.npy")
                    for idx, frame in enumerate(after_frames)
                ]
            after_state = self.controller.state(game, after_obs)
            after_goal_labels = self.controller.bucket_goal_labels(game)
            after_goal_condition = self.controller.bucket_goal_condition(game, target_bucket_index=self.target_bucket_index)
            view_after = build_view_state(
                phase="after",
                case_id=case_id,
                frame=after_frame,
                frame_artifact=after_artifact,
                state=after_state,
                goal_condition=after_goal_condition,
                goal_labels=after_goal_labels,
                target=target,
            )
            visual_correlation = action_visual_correlation(
                case_id=case_id,
                expected_primitive=str(decision.primitive),
                before_state=before_state,
                after_state=after_state,
                before_frame=before_frame,
                after_frame=after_frame,
                action_frame_count=len(after_frames),
                before_goal=goal_condition,
                after_goal=after_goal_condition,
            )
            after_contra_packet = run_contra_view(
                raw_frame=after_frame,
                game_id="sp80-589a99af",
                level_index=int(self.controller.level_index),
                mode="episode_after_action",
                ghost_context=ghost_context,
            )
            deltaband = build_deltaband_observation_set(
                before_statecons=before_contra_packet.get("state_condition_hypotheses", []),
                after_statecons=after_contra_packet.get("state_condition_hypotheses", []),
                action_invocation_id=f"{case_id}_{grounded.packet().get('name')}",
                action_primitive=decision.primitive,
                raw_pixel_delta_count=int(frame_delta(before_frame, after_frame).get("changed_pixels") or 0),
                component_delta_count=None,
                pre_action_ladder=before_contra_packet.get("contra_relationship_ladder"),
                post_action_ladder=after_contra_packet.get("contra_relationship_ladder"),
                before_raw_frame_sha256=sha256_frame(before_frame),
                after_raw_frame_sha256=sha256_frame(after_frame),
                before_raw_frame=before_frame,
                action_frame_sequence=after_frames,
                resource_gauge_numeric_deltas=resource_gauge_numeric_deltas(before_state, after_state),
            )
            levels_completed_delta = int(after_state.get("levels_completed") or 0) - int(before_state.get("levels_completed") or 0)
            action_response = build_object_action_response(
                before_state=before_state,
                after_state=after_state,
                action_taken=grounded.packet(),
                frame_delta=frame_delta(before_frame, after_frame),
                returned_frame_count=len(after_frames),
                action_invocation_id=f"{case_id}_{grounded.packet().get('name')}",
                action_index=action_index,
                downstream_result={
                    "levels_completed_delta": levels_completed_delta,
                    "action_frame_count": len(after_frames),
                    "goal_progress": goal_progress(goal_condition, after_goal_condition),
                },
            )
            actionmodband = build_actionmodband(
                action_responses=[action_response],
                action_invocation_id=f"{case_id}_{grounded.packet().get('name')}",
                levels_completed_delta=levels_completed_delta,
            )
            before_memory = build_initial_episode_memory(
                game_id="sp80-589a99af",
                level_index=int(self.controller.level_index),
                contra_packet=before_contra_packet,
            )
            episode_memory = build_episode_memory_after_round(
                game_id="sp80-589a99af",
                level_index=int(self.controller.level_index),
                round_index=action_index + 1,
                previous_memory=before_memory,
                before_contra_packet=before_contra_packet,
                after_contra_packet=after_contra_packet,
                action_taken=grounded.packet(),
                deltaband_result=deltaband,
                actionmodband_result=actionmodband,
            )
            total_delta = int(after_state.get("levels_completed") or 0) - int(initial_completed)
            stop_reason = None
            if total_delta > 0:
                stop_reason = "levels_completed_increased"
            elif str(after_state.get("state")) == "GAME_OVER":
                stop_reason = "game_over"
            packet.update(
                {
                    "status": "executed",
                    "action": grounded.packet(),
                    "selected_recommendation": selected_recommendation,
                    "after_state": after_state,
                    "after_frame": after_artifact,
                    "view_after": view_after,
                    "after_goal_condition": after_goal_condition,
                    "after_goal_labels": after_goal_labels,
                    "after_contra_packet": after_contra_packet,
                    "goal_progress": goal_progress(goal_condition, after_goal_condition),
                    "delta": frame_delta(before_frame, after_frame),
                    "action_visual_correlation": visual_correlation,
                    "selected_coord_delta": coord_delta(before_state, after_state),
                    "selected_display_coord_delta": display_coord_delta(before_state, after_state),
                    "action_frame_count": len(after_frames),
                    "action_png_sequence": sequence_meta,
                    "action_raw_grid_sequence": raw_sequence_meta,
                    "deltaband_observations": deltaband,
                    "episode_memory_after_round": episode_memory,
                    "actionmodband_observations": actionmodband,
                    "levels_completed_delta": levels_completed_delta,
                    "total_levels_completed_delta": total_delta,
                    "stop_reason": stop_reason,
                }
            )
            brain_call["status"] = packet["status"]
            return packet, brain_call, after_obs
        except Exception as exc:
            packet["status"] = "execution_error"
            packet["error"] = repr(exc)
            brain_call["status"] = packet["status"]
            return packet, brain_call, obs

    def run_case(self, case: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        case_id = str(case["case_id"])
        game, obs, target, setup_actions = self.prepare_case(str(case["setup"]))
        before_frame = np.asarray(obs.frame[-1])
        before_state = self.controller.state(game, obs)
        goal_labels = self.controller.bucket_goal_labels(game)
        goal_condition = self.controller.bucket_goal_condition(game)
        expected, expected_source = self.expected_primitive_for_case(case, goal_condition)
        before_artifact = save_frame_artifact(
            before_frame,
            self.frames_dir / case_id,
            "before",
            self.scale,
        )
        view_before = build_view_state(
            phase="before",
            case_id=case_id,
            frame=before_frame,
            frame_artifact=before_artifact,
            state=before_state,
            goal_condition=goal_condition,
            goal_labels=goal_labels,
            target=target,
        )
        request = self.build_request(
            case=case,
            before_state=before_state,
            before_frame_artifact=before_artifact,
            view_before=view_before,
            target=target,
            setup_actions=setup_actions,
            goal_condition=goal_condition,
        )
        decision = self.brain.decide(request)
        brain_call = {
            "schema": "codexlab.sp80_brain_call.v1",
            "contra_schema": "codexlab.sp80_contra_call.v1",
            "case_id": case_id,
            "contra": contra_spec(
                backend=self.brain.name,
                game="sp80_liquid_pour",
                allowed_primitives=CONTROL_PRIMITIVES,
            ),
            "request": request,
            "decision": decision.to_packet(),
            "contra_request": request,
            "contra_decision": decision.to_packet(),
            "goal_condition": goal_condition,
        }
        packet = self.base_packet(
            case=case,
            expected_primitive=expected,
            expected_primitive_source=expected_source,
            target=target,
            setup_actions=setup_actions,
            before_state=before_state,
            before_frame=before_artifact,
            view_before=view_before,
            decision=decision,
            goal_condition=goal_condition,
            goal_labels=goal_labels,
        )

        if expected is None:
            packet["status"] = "goal_condition_unresolved"
            packet["error"] = "could not derive a local primitive from the bucket goal condition"
            brain_call["status"] = packet["status"]
            return packet, brain_call
        if decision.error:
            packet["status"] = "brain_error"
            packet["error"] = decision.error
            brain_call["status"] = packet["status"]
            return packet, brain_call
        if decision.primitive not in CONTROL_PRIMITIVES:
            packet["status"] = "invalid_primitive"
            packet["error"] = f"brain returned {decision.primitive!r}"
            brain_call["status"] = packet["status"]
            return packet, brain_call
        if decision.primitive != expected:
            packet["status"] = "expected_mismatch"
            packet["error"] = f"brain returned {decision.primitive!r}, expected {expected!r}"
            brain_call["status"] = packet["status"]
            return packet, brain_call

        try:
            grounded = self.controller.ground_primitive(game, decision.primitive, target=target)
            after_obs = self.controller.execute_grounded_action(game, grounded)
            after_frames = [np.asarray(frame) for frame in after_obs.frame]
            after_frame = after_frames[-1]
            after_artifact = save_frame_artifact(
                after_frame,
                self.frames_dir / case_id,
                "after",
                self.scale,
            )
            sequence_meta = None
            if len(after_frames) > 1:
                sequence_meta = save_frame_sequence(
                    after_frames,
                    self.frames_dir / case_id / "png_sequence",
                    self.scale,
                )
                raw_sequence_meta = [
                    save_raw_grid(frame, self.frames_dir / case_id / "raw_grid_sequence" / f"frame_{idx:03d}.npy")
                    for idx, frame in enumerate(after_frames)
                ]
            else:
                raw_sequence_meta = None
            after_state = self.controller.state(game, after_obs)
            after_goal_labels = self.controller.bucket_goal_labels(game)
            after_goal_condition = self.controller.bucket_goal_condition(game)
            view_after = build_view_state(
                phase="after",
                case_id=case_id,
                frame=after_frame,
                frame_artifact=after_artifact,
                state=after_state,
                goal_condition=after_goal_condition,
                goal_labels=after_goal_labels,
                target=target,
            )
            visual_correlation = action_visual_correlation(
                case_id=case_id,
                expected_primitive=expected,
                before_state=before_state,
                after_state=after_state,
                before_frame=before_frame,
                after_frame=after_frame,
                action_frame_count=len(after_frames),
                before_goal=goal_condition,
                after_goal=after_goal_condition,
            )
            episode_memory = None
            episode_memory_error = None
            actionmodband = None
            try:
                before_contra_packet = run_contra_view(
                    raw_frame=before_frame,
                    game_id="sp80-589a99af",
                    level_index=int(self.controller.level_index),
                    mode="harness_before_action",
                )
                after_contra_packet = run_contra_view(
                    raw_frame=after_frame,
                    game_id="sp80-589a99af",
                    level_index=int(self.controller.level_index),
                    mode="harness_after_action",
                )
                deltaband = build_deltaband_observation_set(
                    before_statecons=before_contra_packet.get("state_condition_hypotheses", []),
                    after_statecons=after_contra_packet.get("state_condition_hypotheses", []),
                    action_invocation_id=f"{case_id}_{grounded.packet().get('name')}",
                    action_primitive=decision.primitive,
                    raw_pixel_delta_count=int(frame_delta(before_frame, after_frame).get("changed_pixels") or 0),
                    component_delta_count=None,
                    pre_action_ladder=before_contra_packet.get("contra_relationship_ladder"),
                    post_action_ladder=after_contra_packet.get("contra_relationship_ladder"),
                    before_raw_frame_sha256=sha256_frame(before_frame),
                    after_raw_frame_sha256=sha256_frame(after_frame),
                    before_raw_frame=before_frame,
                    action_frame_sequence=after_frames,
                    resource_gauge_numeric_deltas=resource_gauge_numeric_deltas(before_state, after_state),
                )
                action_invocation_id = f"{case_id}_{grounded.packet().get('name')}"
                action_response = build_object_action_response(
                    before_state=before_state,
                    after_state=after_state,
                    action_taken=grounded.packet(),
                    frame_delta=frame_delta(before_frame, after_frame),
                    returned_frame_count=len(after_frames),
                    action_invocation_id=action_invocation_id,
                    action_index=0,
                    downstream_result={
                        "levels_completed_delta": int(after_state.get("levels_completed") or 0) - int(before_state.get("levels_completed") or 0),
                        "action_frame_count": len(after_frames),
                        "goal_progress": goal_progress(goal_condition, after_goal_condition),
                    },
                )
                actionmodband = build_actionmodband(
                    action_responses=[action_response],
                    action_invocation_id=action_invocation_id,
                    levels_completed_delta=int(after_state.get("levels_completed") or 0) - int(before_state.get("levels_completed") or 0),
                )
                before_memory = build_initial_episode_memory(
                    game_id="sp80-589a99af",
                    level_index=int(self.controller.level_index),
                    contra_packet=before_contra_packet,
                )
                episode_memory = build_episode_memory_after_round(
                    game_id="sp80-589a99af",
                    level_index=int(self.controller.level_index),
                    round_index=1,
                    previous_memory=before_memory,
                    before_contra_packet=before_contra_packet,
                    after_contra_packet=after_contra_packet,
                    action_taken=grounded.packet(),
                    deltaband_result=deltaband,
                    actionmodband_result=actionmodband,
                )
            except Exception as exc:  # noqa: BLE001 - harness packet should record memory failure, not hide action result.
                episode_memory_error = {"error_type": type(exc).__name__, "error_message": str(exc)}
            packet.update(
                {
                    "status": "executed",
                    "action": grounded.packet(),
                    "after_state": after_state,
                    "view_after": view_after,
                    "after_goal_condition": after_goal_condition,
                    "after_goal_labels": after_goal_labels,
                    "goal_progress": goal_progress(goal_condition, after_goal_condition),
                    "after_frame": after_artifact,
                    "delta": frame_delta(before_frame, after_frame),
                    "action_visual_correlation": visual_correlation,
                    "selected_coord_delta": coord_delta(before_state, after_state),
                    "selected_display_coord_delta": display_coord_delta(before_state, after_state),
                    "action_frame_count": len(after_frames),
                    "action_png_sequence": sequence_meta,
                    "action_raw_grid_sequence": raw_sequence_meta,
                    "episode_memory_after_round": episode_memory,
                    "episode_memory_error": episode_memory_error,
                    "actionmodband_observations": actionmodband,
                }
            )
            brain_call["status"] = packet["status"]
            return packet, brain_call
        except Exception as exc:
            packet["status"] = "execution_error"
            packet["error"] = repr(exc)
            brain_call["status"] = packet["status"]
            return packet, brain_call

    def expected_primitive_for_case(
        self,
        case: dict[str, Any],
        goal_condition: dict[str, Any],
    ) -> tuple[str | None, str]:
        expected = str(case["expected_primitive"])
        if expected == "goal_nearest_bucket":
            return (
                primitive_toward_goal_condition(goal_condition),
                "goal_condition.nearest_goal.relative_display_direction",
            )
        return expected, "case.expected_primitive"

    def prepare_case(self, setup: str) -> tuple[Any, Any, SpriteRef, list[dict[str, Any]]]:
        game, obs = self.controller.new_game()
        sprites = self.controller.selectable_sprites(game)
        if not sprites:
            raise RuntimeError("SP80 level has no selectable sprites")
        target = sprites[0]
        setup_actions: list[dict[str, Any]] = []

        if setup in {"selected", "blocked_up"}:
            obs, click = self.controller.click_sprite(game, target)
            setup_actions.append({"primitive": "select_object", "action_name": "ACTION6", "data": click})
        if setup == "blocked_up":
            for _ in range(32):
                before_state = self.controller.state(game, obs)
                moved_obs = self.controller.move_selected(game, "up")
                after_state = self.controller.state(game, moved_obs)
                raw_delta = coord_delta(before_state, after_state)
                visible_delta = display_coord_delta(before_state, after_state)
                setup_actions.append(
                    {
                        "primitive": "move_up",
                        "action_name": "ACTION1",
                        "data": {},
                        "selected_coord_delta": raw_delta,
                        "selected_display_coord_delta": visible_delta,
                    }
                )
                obs = moved_obs
                if raw_delta == [0, 0] or visible_delta == [0, 0]:
                    break
            else:
                raise RuntimeError("could not find blocked ACTION1 state for no_selected_motion")
        if setup not in {"fresh", "selected", "blocked_up"}:
            raise RuntimeError(f"unknown setup {setup!r}")
        return game, obs, target, setup_actions

    def build_request(
        self,
        *,
        case: dict[str, Any],
        before_state: dict[str, Any],
        before_frame_artifact: dict[str, Any],
        view_before: dict[str, Any],
        target: SpriteRef,
        setup_actions: list[dict[str, Any]],
        goal_condition: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema": "codexlab.sp80_brain_request.v1",
            "contra_schema": "codexlab.sp80_contra_request.v1",
            "game": "sp80_liquid_pour",
            "case_id": case["case_id"],
            "intent": case["intent"],
            "view_state": view_before,
            "goal_condition": goal_condition,
            "allowed_primitives": CONTROL_PRIMITIVES,
            "expected_effect_tags": case.get("effect_tags", []),
            "sp80_action_set": self.controller.action_set(),
            "observation": {
                "setup": case["setup"],
                "setup_action_count": len(setup_actions),
                "state": before_state,
                "before_frame": before_frame_artifact,
                "view_before": view_before,
                "target_sprite": target.__dict__,
                "setup_actions": setup_actions,
            },
            "brain_contract": contra_contract(),
            "contra_contract": contra_contract(),
        }

    def base_packet(
        self,
        *,
        case: dict[str, Any],
        expected_primitive: str | None,
        expected_primitive_source: str,
        target: SpriteRef,
        setup_actions: list[dict[str, Any]],
        before_state: dict[str, Any],
        before_frame: dict[str, Any],
        view_before: dict[str, Any],
        decision: BrainDecision,
        goal_condition: dict[str, Any],
        goal_labels: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema": "codexlab.sp80_harness_packet.v1",
            "status": "pending",
            "case_id": case["case_id"],
            "intent": case["intent"],
            "case_expected_primitive": case["expected_primitive"],
            "expected_primitive": expected_primitive,
            "expected_primitive_source": expected_primitive_source,
            "expected_effect_tags": case.get("effect_tags", []),
            "setup": case["setup"],
            "target_sprite": target.__dict__,
            "setup_actions": setup_actions,
            "goal_condition": goal_condition,
            "goal_labels": goal_labels,
            "view_before": view_before,
            "brain_decision": decision.to_packet(),
            "contra": contra_spec(
                backend=self.brain.name,
                game="sp80_liquid_pour",
                allowed_primitives=CONTROL_PRIMITIVES,
            ),
            "contra_decision": decision.to_packet(),
            "before_state": before_state,
            "before_frame": before_frame,
            "controlled_runtime": True,
            "route_or_replay_used": False,
            "human_trace_used": False,
            "coordinate_script_used": False,
        }


def build_brain(args: argparse.Namespace, controller: SP80BaseController) -> BrainBackend:
    if args.backend == "tiny_mlp":
        return TinyMLPBrain(controller)
    if args.backend == "lora_adapter":
        if args.lora_adapter is None:
            raise RuntimeError("--lora-adapter is required for --backend lora_adapter")
        adapter_path = args.lora_adapter
        if not adapter_path.is_absolute():
            adapter_path = ROOT / adapter_path
        return LoRAAdapterBrain(adapter_path.resolve())
    if args.backend == "qwen_http":
        url = args.qwen_url or os.environ.get("QWEN_HARNESS_URL")
        if not url:
            raise RuntimeError("--qwen-url or QWEN_HARNESS_URL is required for --backend qwen_http")
        return QwenHTTPBrain(
            url=url,
            timeout_s=args.qwen_timeout_s,
            mode=args.qwen_mode,
            model=args.qwen_model,
        )
    if args.backend == "puffer_policy":
        if args.puffer_policy is None:
            raise RuntimeError("--puffer-policy is required for --backend puffer_policy")
        policy_path = args.puffer_policy
        if not policy_path.is_absolute():
            policy_path = ROOT / policy_path
        return build_puffer_policy_brain(
            policy_path.resolve(),
            require_pufferlib=bool(args.require_pufferlib),
        )
    if args.backend == "paint_control_skill":
        return build_paint_control_skill_brain(
            checkpoint_path=args.paint_skill_checkpoint,
            policy_env=args.paint_skill_env,
            require_native=bool(args.paint_skill_require_native),
            focus_edge=args.target_bucket_index,
            forward_model_ledger_path=args.paint_skill_forward_model_ledger,
        )
    if args.backend == "spill_planner":
        return build_spill_planner_brain(
            search_time_budget=float(args.spill_search_time_budget),
            search_restarts=int(args.spill_search_restarts),
        )
    raise RuntimeError(f"unknown backend {args.backend!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=[
            "tiny_mlp",
            "lora_adapter",
            "qwen_http",
            "puffer_policy",
            "paint_control_skill",
            "spill_planner",
        ],
        default="tiny_mlp",
    )
    parser.add_argument("--mode", choices=["action_set", "episode"], default="action_set")
    parser.add_argument("--level-index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("codexlab/runs/sp80_player_harness_tiny_mlp"))
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=20)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--qwen-url", default=None)
    parser.add_argument("--qwen-timeout-s", type=float, default=20.0)
    parser.add_argument("--qwen-mode", choices=["simple_json", "openai_chat"], default="simple_json")
    parser.add_argument("--qwen-model", default=None)
    parser.add_argument("--lora-adapter", type=Path, default=None)
    parser.add_argument("--puffer-policy", type=Path, default=None)
    parser.add_argument("--require-pufferlib", action="store_true")
    parser.add_argument("--target-bucket-index", type=int, default=0)
    parser.add_argument("--paint-skill-checkpoint", type=Path, default=None)
    parser.add_argument("--paint-skill-env", default="rexus_sp80_paintflow_l1_multi")
    parser.add_argument("--paint-skill-require-native", action="store_true")
    parser.add_argument(
        "--paint-skill-forward-model-ledger",
        type=Path,
        default=Path("codexlab/datasets/sp80_crucible_discrepancy_v0/ledger.json"),
    )
    parser.add_argument("--spill-search-time-budget", type=float, default=12.0)
    parser.add_argument("--spill-search-restarts", type=int, default=12)
    parser.add_argument(
        "--compact-episode-artifacts",
        action="store_true",
        help="write compact episode JSONL rows while preserving frame/action/result provenance",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    controller = SP80PlayerController(level_index=args.level_index)
    brain = build_brain(args, controller)
    harness = SP80PlayerHarness(
        controller=controller,
        brain=brain,
        out_dir=ROOT / args.out_dir,
        scale=max(1, int(args.scale)),
        strict=args.strict,
        target_bucket_index=int(args.target_bucket_index) if args.mode == "episode" else None,
        compact_episode_artifacts=bool(args.compact_episode_artifacts),
    )
    if args.mode == "episode":
        result = harness.run_episode(max_actions=max(1, int(args.max_actions)))
    else:
        result = harness.run_action_set()
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.mode == "episode":
        return 0 if result["status"] in {"completed", "budget_exhausted", "game_over"} else 1
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
