"""Ghost → legal episodic relation memory (no routes, no action authority)."""

from __future__ import annotations

import numpy as np

from legal_search.algorithms.ghost_relations import mine_ghost
from legal_search.algorithms.memory import EpisodicMemory
from legal_search.consume.base import ConsumeAdapter
from legal_search.types import Hypothesis, LegalPacket, Transition


class GhostConsume(ConsumeAdapter):
    system_id = "ghost"
    algorithm_id = "memory.confirmed_relation_hints.v1"
    source_paths = (
        "codexlab/contra_head/ghost.py",
        "legal_search/algorithms/ghost_relations.py",
        "legal_search/algorithms/memory.py",
    )
    stripped_bindings = (
        "ghost_as_action_route",
        "ghost_as_action_authority",
        "cross_game_ghost_transfer_without_reinduction",
    )

    def __init__(self, memory: EpisodicMemory | None = None) -> None:
        self.memory = memory or EpisodicMemory()

    def consume(
        self,
        frame: np.ndarray,
        *,
        game_id: str = "",
        transitions: list[Transition] | None = None,
        available_actions: list[int] | None = None,
        levels_completed: int = 0,
    ) -> LegalPacket:
        if transitions is not None:
            self.memory.wipe()
            if game_id:
                self.memory.bind_game(game_id)
            for tr in transitions:
                self.memory.record_transition(tr)
                if tr.levels_up:
                    self.memory.record_relation_hint(
                        {
                            "kind": "level_up_context",
                            "action": tr.action.to_dict(),
                            "levels_after": tr.levels_after,
                        }
                    )
                if tr.frame_changed:
                    self.memory.record_relation_hint(
                        {
                            "kind": "change_under_action",
                            "action_id": int(tr.action.action_id),
                            "levels_after": tr.levels_after,
                        }
                    )
        elif game_id:
            self.memory.bind_game(game_id)

        mined = mine_ghost(frame, transitions, game_id=game_id)
        snap = self.memory.snapshot()
        proposals = self.memory.proposals_from_memory()
        receipt = self.receipt(
            allowed_outputs=[
                "relation_hints",
                "confirmed_relation_rows",
                "lineup_summary",
                "episodic_success_fragments",
            ],
            banned_outputs=["action_route_script", "solve_truth"],
            notes="Ghost mined as this-game confirmed relations + lineup summary only.",
        )
        return LegalPacket(
            system_id=self.system_id,
            algorithm_id=self.algorithm_id,
            hypotheses=[
                Hypothesis(
                    kind="ghost_snapshot",
                    payload={
                        **snap,
                        **mined,
                        "n_relation_hints": snap.get("n_relation_hints", 0)
                        + int(mined.get("n_confirmed") or 0),
                    },
                    confidence=0.55 if mined.get("n_transitions") else 0.35,
                    source_system=self.system_id,
                )
            ],
            proposals=proposals,
            features={"levels_completed": levels_completed},
            receipt=receipt.to_dict(),
        )
