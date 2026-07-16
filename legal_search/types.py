"""Shared types for the legal search / consume layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Action:
    """One env action. action_id 1-5 discrete, 6 = click with x,y."""

    action_id: int
    x: int | None = None
    y: int | None = None
    source: str = "unknown"

    def as_tuple(self) -> tuple[int, int | None, int | None]:
        return int(self.action_id), self.x, self.y

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Transition:
    """One experienced step — the only legal training atom."""

    game_id: str
    step: int
    frame_before: Any  # np.ndarray
    action: Action
    frame_after: Any
    levels_before: int
    levels_after: int
    state_after: str = ""
    frame_changed: bool = False
    levels_up: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "step": self.step,
            "action": self.action.to_dict(),
            "levels_before": self.levels_before,
            "levels_after": self.levels_after,
            "state_after": self.state_after,
            "frame_changed": self.frame_changed,
            "levels_up": self.levels_up,
        }


@dataclass
class Proposal:
    """A candidate action or short program — never solve truth."""

    actions: list[Action]
    score: float
    kind: str
    evidence: dict[str, Any] = field(default_factory=dict)
    algorithm_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [a.to_dict() for a in self.actions],
            "score": float(self.score),
            "kind": self.kind,
            "evidence": dict(self.evidence),
            "algorithm_id": self.algorithm_id,
        }


@dataclass
class Hypothesis:
    """Fail-closed perception / relation hypothesis."""

    kind: str
    payload: dict[str, Any]
    confidence: float = 0.0
    admitted: bool = False  # always False under legal contract
    source_system: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "payload": dict(self.payload),
            "confidence": float(self.confidence),
            "admitted": False,
            "source_system": self.source_system,
            "evidence_status": "hypothesis_not_truth",
        }


@dataclass
class LegalPacket:
    """Unified output of any consume adapter."""

    system_id: str
    algorithm_id: str
    hypotheses: list[Hypothesis] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    receipt: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "legal_search.packet.v1",
            "system_id": self.system_id,
            "algorithm_id": self.algorithm_id,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "proposals": [p.to_dict() for p in self.proposals],
            "features": dict(self.features),
            "receipt": dict(self.receipt),
        }
