"""Operator protocol for MEGAMIND physics / map / conceptual plugs."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from synthesis_lab.orchestration.rdl.brain.brain_types import OperatorResult

__all__ = ["Operator", "OperatorResult"]


@runtime_checkable
class Operator(Protocol):
    name: str

    def run(self, map_view: Any, context: dict | None = None) -> OperatorResult:
        ...
