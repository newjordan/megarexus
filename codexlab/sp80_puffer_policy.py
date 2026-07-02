"""PufferLib-backed primitive policy adapter for the SP80 harness.

This module is intentionally narrow: it only chooses one existing SP80 control
primitive. The live SP80 controller still grounds and executes the primitive.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

from players.base import BrainDecision
from players.sp80.registry import CONTROL_PRIMITIVES, normalize_primitive
from sp80_base_controller import ROOT


POLICY_SCHEMA = "codexlab.sp80_puffer_primitive_policy.v1"


def inspect_pufferlib() -> dict[str, Any]:
    """Return import metadata without making PufferLib a hard dependency.

    The top-level package can import even when the native RL backend is absent.
    For this spike, ``available`` means the training/runtime backend imports.
    """

    started = time.perf_counter()
    try:
        module = importlib.import_module("pufferlib")
        try:
            version = importlib.metadata.version("pufferlib")
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", None)
        result = {
            "top_level_available": True,
            "version": version,
            "module_file": getattr(module, "__file__", None),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
    except Exception as exc:  # noqa: BLE001 - probe should report exact runtime failure.
        return {
            "available": False,
            "top_level_available": False,
            "pufferl_available": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
    try:
        pufferl = importlib.import_module("pufferlib.pufferl")
        result.update(
            {
                "available": True,
                "pufferl_available": True,
                "pufferl_file": getattr(pufferl, "__file__", None),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
    except Exception as exc:  # noqa: BLE001 - probe should report exact runtime failure.
        result.update(
            {
                "available": False,
                "pufferl_available": False,
                "pufferl_error_type": type(exc).__name__,
                "pufferl_error_message": str(exc),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
    return result


def load_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"{path}: expected schema {POLICY_SCHEMA!r}")
    probabilities = policy.get("primitive_probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        raise ValueError(f"{path}: primitive_probabilities must be a non-empty object")
    normalized: dict[str, float] = {}
    for key, value in probabilities.items():
        primitive = normalize_primitive(key)
        if primitive not in CONTROL_PRIMITIVES:
            raise ValueError(f"{path}: invalid primitive {key!r}")
        normalized[primitive] = float(value)
    total = sum(max(0.0, score) for score in normalized.values())
    if total <= 0.0:
        raise ValueError(f"{path}: primitive probabilities must contain positive mass")
    policy["primitive_probabilities"] = {key: max(0.0, value) / total for key, value in normalized.items()}
    overrides = policy.get("case_primitive_overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"{path}: case_primitive_overrides must be an object when present")
    normalized_overrides: dict[str, str] = {}
    for case_id, value in overrides.items():
        primitive = normalize_primitive(value)
        if primitive not in CONTROL_PRIMITIVES:
            raise ValueError(f"{path}: invalid override primitive {value!r} for {case_id!r}")
        normalized_overrides[str(case_id)] = primitive
    policy["case_primitive_overrides"] = normalized_overrides
    return policy


def request_features(request: dict[str, Any]) -> dict[str, Any]:
    observation = request.get("observation") if isinstance(request.get("observation"), dict) else {}
    state = observation.get("state") if isinstance(observation.get("state"), dict) else {}
    action_memory = request.get("action_memory") if isinstance(request.get("action_memory"), list) else []
    return {
        "intent": request.get("intent"),
        "mode": request.get("mode"),
        "allowed_primitive_count": len(request.get("allowed_primitives") or []),
        "setup_action_count": len(request.get("setup_actions") or []),
        "action_memory_count": len(action_memory),
        "has_goal_condition": isinstance(request.get("goal_condition"), dict),
        "state_keys": sorted(str(key) for key in state.keys())[:16],
    }


class PufferPolicyBrain:
    """Adapter for an exported primitive-probability policy from a PufferLib scout."""

    name = "puffer_policy"

    def __init__(self, policy_path: Path, *, require_pufferlib: bool = False) -> None:
        self.policy_path = policy_path
        self.policy = load_policy(policy_path)
        self.pufferlib = inspect_pufferlib()
        if require_pufferlib and not self.pufferlib["available"]:
            raise RuntimeError(f"pufferlib import failed: {self.pufferlib}")

    def decide(self, request: dict[str, Any]) -> BrainDecision:
        started = time.perf_counter()
        probabilities = dict(self.policy["primitive_probabilities"])
        allowed = {
            primitive
            for primitive in (normalize_primitive(row) for row in request.get("allowed_primitives", []))
            if primitive in CONTROL_PRIMITIVES
        }
        candidates = {
            primitive: score
            for primitive, score in probabilities.items()
            if not allowed or primitive in allowed
        }
        if not candidates:
            return BrainDecision(
                backend=self.name,
                primitive=None,
                confidence=None,
                raw={
                    "policy_schema": self.policy.get("schema"),
                    "policy_path": self._display_policy_path(),
                    "error": "no_allowed_policy_candidates",
                    "pufferlib": self.pufferlib,
                },
                latency_ms=(time.perf_counter() - started) * 1000.0,
                error="no_allowed_policy_candidates",
            )
        override_primitive = self._case_override(request)
        if override_primitive is not None and override_primitive in candidates:
            primitive = override_primitive
            confidence = candidates[primitive]
            selection_source = "case_primitive_overrides"
        else:
            primitive, confidence = max(candidates.items(), key=lambda row: (float(row[1]), row[0]))
            selection_source = "primitive_probabilities"
        return BrainDecision(
            backend=self.name,
            primitive=primitive,
            confidence=float(confidence),
            raw={
                "policy_schema": self.policy.get("schema"),
                "policy_path": self._display_policy_path(),
                "policy_source": self.policy.get("source"),
                "policy_kind": self.policy.get("policy_kind", "primitive_probability_export"),
                "chosen_primitive": primitive,
                "selection_source": selection_source,
                "primitive_probabilities": probabilities,
                "case_primitive_overrides": self.policy.get("case_primitive_overrides", {}),
                "request_features": request_features(request),
                "pufferlib": self.pufferlib,
                "truth_boundary": "adapter_compatibility_scout_not_trained_solve_evidence",
            },
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _display_policy_path(self) -> str:
        try:
            return self.policy_path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return str(self.policy_path)

    def _case_override(self, request: dict[str, Any]) -> str | None:
        overrides = self.policy.get("case_primitive_overrides") or {}
        return overrides.get(str(request.get("case_id")))
