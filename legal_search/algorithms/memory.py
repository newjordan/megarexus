"""Episodic memory — algorithm mined from CONTRA episode_memory + Ghost.

Legal: per-game only. Wipe on game_id change. Never admits solve truth.
Stores successful programs and transition fingerprints from *this* run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from legal_search.types import Action, Proposal, Transition


@dataclass
class EpisodicMemory:
    algorithm_id: str = "memory.episodic.v1"
    game_id: str | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)
    successful_programs: list[dict[str, Any]] = field(default_factory=list)
    relation_hints: list[dict[str, Any]] = field(default_factory=list)
    max_transitions: int = 512
    max_programs: int = 32

    def bind_game(self, game_id: str) -> None:
        if self.game_id is not None and self.game_id != game_id:
            self.wipe()
        self.game_id = game_id

    def wipe(self) -> None:
        self.transitions.clear()
        self.successful_programs.clear()
        self.relation_hints.clear()
        self.game_id = None

    def record_transition(self, tr: Transition) -> None:
        self.bind_game(tr.game_id)
        self.transitions.append(tr.to_dict())
        if len(self.transitions) > self.max_transitions:
            self.transitions = self.transitions[-self.max_transitions :]
        if tr.levels_up:
            # store recent action window as a successful program fragment
            window = self.transitions[-12:]
            self.successful_programs.append(
                {
                    "levels_after": tr.levels_after,
                    "window": window,
                    "evidence_status": "episodic_success_not_universal_truth",
                }
            )
            if len(self.successful_programs) > self.max_programs:
                self.successful_programs = self.successful_programs[-self.max_programs :]

    def record_relation_hint(self, hint: dict[str, Any]) -> None:
        """Ghost-like: relation evidence, never action authority."""
        row = dict(hint)
        row["admitted"] = False
        row["evidence_status"] = "relation_hypothesis_not_truth"
        self.relation_hints.append(row)
        if len(self.relation_hints) > 64:
            self.relation_hints = self.relation_hints[-64:]

    def proposals_from_memory(self) -> list[Proposal]:
        """Re-score candidates only as weak priors for *this* game."""
        out: list[Proposal] = []
        for i, row in enumerate(self.successful_programs[-5:]):
            actions: list[Action] = []
            for step in row.get("window") or []:
                a = step.get("action") or {}
                actions.append(
                    Action(
                        int(a.get("action_id", 1)),
                        a.get("x"),
                        a.get("y"),
                        source=self.algorithm_id,
                    )
                )
            if not actions:
                continue
            out.append(
                Proposal(
                    actions=actions,
                    score=0.1 + 0.05 * i,  # weak; real ranker is env/model
                    kind="episodic_success_prior",
                    evidence={"levels_after": row.get("levels_after"), "this_game_only": True},
                    algorithm_id=self.algorithm_id,
                )
            )
        return out

    def snapshot(self) -> dict[str, Any]:
        return {
            "algorithm_id": self.algorithm_id,
            "game_id": self.game_id,
            "n_transitions": len(self.transitions),
            "n_successful_programs": len(self.successful_programs),
            "n_relation_hints": len(self.relation_hints),
            "cross_game_memory_admitted": False,
        }
