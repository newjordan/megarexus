#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from dataclasses import asdict, dataclass


CONTRA_NAME = "Contra"
CONTRA_SCHEMA = "codexlab.contra_head.v1"
CONTRA_VERSION = "contra_v0.1"


@dataclass(frozen=True)
class ContraHeadSpec:
    name: str
    version: str
    backend: str
    game: str
    allowed_primitives: list[str]
    role: str
    output_contract: dict[str, Any]

    def to_packet(self) -> dict[str, Any]:
        return {
            "schema": CONTRA_SCHEMA,
            **asdict(self),
        }


def contra_contract() -> dict[str, Any]:
    return {
        "input": "intent plus view_state.control_isolation_packet plus optional goal_condition",
        "output": "one JSON primitive only",
        "must_receive_view_before_action": True,
        "must_receive_control_isolation_before_action": True,
        "must_be_verified_against_view_after_action": True,
        "must_verify_control_isolation_after_action": True,
        "may_read_view_state": True,
        "may_read_control_isolation": True,
        "may_read_goal_condition": True,
        "goal_condition_scope": "semantic goal labels and local relation fields only",
        "control_isolation_scope": "raw-pixel shape traces are deterministic audit evidence; admitted SAM masks may become Contra object-view units only through fail-closed review admission; neither path is goal truth or route truth",
        "may_choose_coordinates": False,
        "may_emit_route": False,
        "may_emit_multiple_actions": False,
        "may_claim_gameplay_completion": False,
        "motor_executes_actions": True,
    }


def contra_spec(*, backend: str, game: str, allowed_primitives: list[str]) -> dict[str, Any]:
    return ContraHeadSpec(
        name=CONTRA_NAME,
        version=CONTRA_VERSION,
        backend=backend,
        game=game,
        allowed_primitives=list(allowed_primitives),
        role="intent_plus_observation_to_one_control_primitive",
        output_contract=contra_contract(),
    ).to_packet()
