#!/usr/bin/env python3
"""Wire-contract registry for the CONTRA informer-DAG service.

This module is intentionally dependency-light: it re-exports the existing
versioned ``*_SCHEMA`` constants (the real inter-module contracts) and defines
the small set of request-envelope schemas the service adds on top.  It does NOT
import the heavy player harness; the harness/ghost-context schema *strings* are
re-declared here and a drift check in ``contra_service_verify`` asserts they
still equal their source-of-truth constants, so drift fails the green gate
rather than silently diverging.

Boundary principle: no subsystem in CONTRA-land admits truth.  Oracle and Ghost
*inform* CONTRA; CONTRA synthesises guidance (also not truth); only execution in
the live harness is a truth source.  Every response a handler returns must
satisfy :func:`assert_no_truth_admission`.
"""

from __future__ import annotations

from dataclasses import dataclass

from contra_head.api import (
    CONTRA_CONFIG_PASS_SCHEMA,
    CONTRA_CONTROLLER_REQUEST_SCHEMA,
    CONTRA_CONTROLLER_RESPONSE_SCHEMA,
    CONTRA_GHOST_PASS_SCHEMA,
    CONTRA_GHOST_SCHEMA,
    CONTRA_ORACLE_PASS_SCHEMA,
    CONTRA_PASS_SCHEMA,
    CONTRA_VIEW_SYSTEM_PASS_SET_SCHEMA,
)

# --- Schemas owned elsewhere, re-declared here to keep this module import-light.
# Source of truth: sp80_player_harness.CONTRA_RECOMMENDATION_PACKET_SCHEMA and the
# literal in contra_head.ghost.load_ghost_context.  contra_service_verify asserts these
# strings still match their owners, so drift fails the gate.
CONTRA_RECOMMENDATION_PACKET_SCHEMA = "codexlab.sp80_contra_recommendation_packet.v1"
CONTRA_GHOST_CONTEXT_SCHEMA = "codexlab.contra_ghost_context.v1"
CONTRA_SERVICE_HEALTH_SCHEMA = "codexlab.contra_service_health.v1"

# --- New request-envelope + meta schemas this service introduces.
CONTRA_SERVICE_VIEW_REQUEST_SCHEMA = "codexlab.contra_service_view_request.v1"
CONTRA_SERVICE_GHOST_PASS_REQUEST_SCHEMA = "codexlab.contra_service_ghost_pass_request.v1"
CONTRA_SERVICE_GHOST_CONTEXT_REQUEST_SCHEMA = "codexlab.contra_service_ghost_context_request.v1"
CONTRA_SERVICE_RECOMMENDATION_REQUEST_SCHEMA = "codexlab.contra_service_recommendation_request.v1"
CONTRA_SERVICE_FEEDBACK_INGEST_REQUEST_SCHEMA = "codexlab.contra_service_feedback_ingest_request.v1"
CONTRA_SERVICE_FEEDBACK_INGEST_RESULT_SCHEMA = "codexlab.contra_service_feedback_ingest_result.v1"
CONTRA_SERVICE_CONTRACTS_SCHEMA = "codexlab.contra_service_contracts.v1"

# --- Admission invariant -----------------------------------------------------
# The flags ApiVerifier.verify_no_truth_admission gates on.  A response may omit
# some (e.g. the recommendation packet carries no admission_status), but any flag
# it *does* carry must be False, and any admission_status must be not-admitted.
ADMISSION_TRUTH_FLAGS: tuple[str, ...] = (
    "control_truth_admitted",
    "goal_condition_admitted",
    "action_choice_admitted",
    "solve_condition_admitted",
)
ADMITTED_STATUS_OK: frozenset[str] = frozenset({"not_admitted", "fail_closed_not_admitted"})


def truth_admission_violations(packet, *, label: str = "response") -> list[str]:
    """Return admission-boundary violations for ``packet`` (empty list == clean)."""
    if not isinstance(packet, dict):
        return [f"{label}: not a dict"]
    violations: list[str] = []
    for flag in ADMISSION_TRUTH_FLAGS:
        if flag in packet and packet.get(flag) is not False:
            violations.append(f"{label}: {flag}={packet.get(flag)!r} (must be False)")
    status = packet.get("admission_status")
    if status is not None and status not in ADMITTED_STATUS_OK:
        violations.append(
            f"{label}: admission_status={status!r} (must be one of {sorted(ADMITTED_STATUS_OK)})"
        )
    return violations


def assert_no_truth_admission(packet, *, label: str = "response") -> None:
    """Raise if ``packet`` admits truth across any known flag/status."""
    violations = truth_admission_violations(packet, label=label)
    if violations:
        raise AssertionError("truth admission boundary violated:\n" + "\n".join(violations))


@dataclass(frozen=True)
class EndpointContract:
    """One edge of the informer DAG, expressed as a service endpoint."""

    method: str
    path: str
    handler: str
    response_schema: str
    admission: str
    summary: str
    request_schema: str | None = None
    enforces_no_truth_admission: bool = True


CONTRACTS: tuple[EndpointContract, ...] = (
    EndpointContract(
        method="POST",
        path="/v1/contra/view",
        handler="contra_view",
        request_schema=CONTRA_SERVICE_VIEW_REQUEST_SCHEMA,
        response_schema=CONTRA_CONTROLLER_RESPONSE_SCHEMA,
        admission="full_truth_boundary_not_admitted",
        summary="CONTRA synthesis: full controller view over a raw frame (Oracle+Ghost passes embedded).",
    ),
    EndpointContract(
        method="POST",
        path="/v1/contra/oracle/pass",
        handler="oracle_pass",
        request_schema=CONTRA_SERVICE_VIEW_REQUEST_SCHEMA,
        response_schema=CONTRA_ORACLE_PASS_SCHEMA,
        admission="review_hypothesis_not_truth_source",
        summary="Oracle informer: actor/target review hypotheses (never action truth).",
    ),
    EndpointContract(
        method="POST",
        path="/v1/contra/ghost/pass",
        handler="ghost_pass",
        request_schema=CONTRA_SERVICE_GHOST_PASS_REQUEST_SCHEMA,
        response_schema=CONTRA_GHOST_PASS_SCHEMA,
        admission="ghost_relationship_memory_not_truth_source",
        summary="Ghost informer: confirmed relation-memory rows from controlled episodes.",
    ),
    EndpointContract(
        method="POST",
        path="/v1/contra/ghost/context",
        handler="ghost_context",
        request_schema=CONTRA_SERVICE_GHOST_CONTEXT_REQUEST_SCHEMA,
        response_schema=CONTRA_GHOST_CONTEXT_SCHEMA,
        admission="ghost_context_truth_boundary",
        summary="Load Ghost memory context for a game/run dir (not-truth, not-replay).",
    ),
    EndpointContract(
        method="POST",
        path="/v1/contra/recommendation",
        handler="recommendation",
        request_schema=CONTRA_SERVICE_RECOMMENDATION_REQUEST_SCHEMA,
        response_schema=CONTRA_RECOMMENDATION_PACKET_SCHEMA,
        admission="recommendation_is_a_control_input_not_action_choice_truth",
        summary="CONTRA guidance: value-ranked primitives (a control input, not an action choice).",
    ),
    EndpointContract(
        method="POST",
        path="/v1/contra/feedback/ingest",
        handler="feedback_ingest",
        request_schema=CONTRA_SERVICE_FEEDBACK_INGEST_REQUEST_SCHEMA,
        response_schema=CONTRA_SERVICE_FEEDBACK_INGEST_RESULT_SCHEMA,
        admission="ghost_memory_not_future_truth",
        summary="Harness->Ghost return edge: persist relation memory only on verified controlled completion.",
    ),
    EndpointContract(
        method="GET",
        path="/v1/contracts",
        handler="contracts",
        request_schema=None,
        response_schema=CONTRA_SERVICE_CONTRACTS_SCHEMA,
        admission="discovery_document",
        summary="LLM-discoverable contract registry for all endpoints.",
        enforces_no_truth_admission=False,
    ),
    EndpointContract(
        method="GET",
        path="/v1/health",
        handler="health",
        request_schema=None,
        response_schema=CONTRA_SERVICE_HEALTH_SCHEMA,
        admission="liveness",
        summary="Liveness probe.",
        enforces_no_truth_admission=False,
    ),
)

CONTRACTS_BY_PATH: dict[str, EndpointContract] = {c.path: c for c in CONTRACTS}
CONTRACTS_BY_HANDLER: dict[str, EndpointContract] = {c.handler: c for c in CONTRACTS}


def contracts_document() -> dict:
    """The ``GET /v1/contracts`` payload: every endpoint + req/resp schema + boundary."""
    return {
        "schema": CONTRA_SERVICE_CONTRACTS_SCHEMA,
        "service": "contra_informer_dag",
        "informer_dag": "oracle+ghost -> contra(view/recommendation) -> harness(execution=truth) -> ghost_memory",
        "truth_boundary": "no_endpoint_admits_truth_execution_is_the_only_truth_source",
        "endpoints": [
            {
                "method": c.method,
                "path": c.path,
                "handler": c.handler,
                "request_schema": c.request_schema,
                "response_schema": c.response_schema,
                "admission_boundary": c.admission,
                "enforces_no_truth_admission": c.enforces_no_truth_admission,
                "summary": c.summary,
            }
            for c in CONTRACTS
        ],
    }


__all__ = [
    "CONTRA_CONFIG_PASS_SCHEMA",
    "CONTRA_CONTROLLER_REQUEST_SCHEMA",
    "CONTRA_CONTROLLER_RESPONSE_SCHEMA",
    "CONTRA_GHOST_PASS_SCHEMA",
    "CONTRA_GHOST_SCHEMA",
    "CONTRA_ORACLE_PASS_SCHEMA",
    "CONTRA_PASS_SCHEMA",
    "CONTRA_VIEW_SYSTEM_PASS_SET_SCHEMA",
    "CONTRA_RECOMMENDATION_PACKET_SCHEMA",
    "CONTRA_GHOST_CONTEXT_SCHEMA",
    "CONTRA_SERVICE_HEALTH_SCHEMA",
    "CONTRA_SERVICE_VIEW_REQUEST_SCHEMA",
    "CONTRA_SERVICE_GHOST_PASS_REQUEST_SCHEMA",
    "CONTRA_SERVICE_GHOST_CONTEXT_REQUEST_SCHEMA",
    "CONTRA_SERVICE_RECOMMENDATION_REQUEST_SCHEMA",
    "CONTRA_SERVICE_FEEDBACK_INGEST_REQUEST_SCHEMA",
    "CONTRA_SERVICE_FEEDBACK_INGEST_RESULT_SCHEMA",
    "CONTRA_SERVICE_CONTRACTS_SCHEMA",
    "ADMISSION_TRUTH_FLAGS",
    "ADMITTED_STATUS_OK",
    "EndpointContract",
    "CONTRACTS",
    "CONTRACTS_BY_PATH",
    "CONTRACTS_BY_HANDLER",
    "truth_admission_violations",
    "assert_no_truth_admission",
    "contracts_document",
]
