#!/usr/bin/env python3
"""Stable projections over the CONTRA response -- the "via internal API" layer.

Reporting and verification slice CONTRA internals THROUGH these contract-checked
projections instead of ad-hoc dict spelunking, so a layer reorg fails loudly
(schema assertion) rather than silently mis-rendering a report.  These are the
same functions the service handlers use, so "consume the report via the internal
API" == "read through these projections".
"""

from __future__ import annotations

from typing import Any

from .contracts import (
    CONTRA_CONTROLLER_RESPONSE_SCHEMA,
    CONTRA_GHOST_PASS_SCHEMA,
    CONTRA_ORACLE_PASS_SCHEMA,
    CONTRA_RECOMMENDATION_PACKET_SCHEMA,
)


def require_schema(packet: Any, expected: str, *, label: str) -> dict:
    """Return ``packet`` iff it is a dict carrying ``schema == expected``; else raise."""
    if not isinstance(packet, dict):
        raise ValueError(f"{label}: expected dict, got {type(packet).__name__}")
    got = packet.get("schema")
    if got != expected:
        raise ValueError(f"{label}: schema mismatch: expected {expected!r}, got {got!r}")
    return packet


def project_oracle_pass(response: dict) -> dict:
    """The Oracle informer pass out of a full CONTRA response."""
    require_schema(response, CONTRA_CONTROLLER_RESPONSE_SCHEMA, label="contra_response")
    return require_schema(response.get("oracle_pass"), CONTRA_ORACLE_PASS_SCHEMA, label="oracle_pass")


def project_ghost_pass(response: dict) -> dict:
    """The Ghost informer pass out of a full CONTRA response."""
    require_schema(response, CONTRA_CONTROLLER_RESPONSE_SCHEMA, label="contra_response")
    return require_schema(response.get("ghost_pass"), CONTRA_GHOST_PASS_SCHEMA, label="ghost_pass")


def project_recommendation(packet: dict) -> dict:
    """Validate and return a CONTRA recommendation/guidance packet."""
    return require_schema(packet, CONTRA_RECOMMENDATION_PACKET_SCHEMA, label="recommendation")


def oracle_review_hypotheses(response: dict) -> dict:
    """Oracle's actor/target review hypotheses (review-only; never action truth)."""
    op = project_oracle_pass(response)
    return {
        "evidence_role": op.get("evidence_role"),
        "actor_review_hypotheses": op.get("actor_review_hypotheses", []),
        "target_review_hypotheses": op.get("target_review_hypotheses", []),
        "actor_hypothesis_count": op.get("actor_hypothesis_count", 0),
        "target_hypothesis_count": op.get("target_hypothesis_count", 0),
    }


def ghost_relation_memory_rows(response: dict) -> list[dict]:
    """Ghost's confirmed relation-memory rows (not action authority)."""
    gp = project_ghost_pass(response)
    rows = gp.get("relation_memory_rows", [])
    return list(rows) if isinstance(rows, list) else []


def ranked_recommendations(packet: dict) -> list[dict]:
    """The value-ranked primitives out of a CONTRA guidance packet."""
    rec = project_recommendation(packet)
    rows = rec.get("ranked_recommendations", [])
    return list(rows) if isinstance(rows, list) else []
