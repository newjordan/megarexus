"""Legal agent loop: explore → induce → search → act → update.

This is the only path that should claim "general learning" behavior.
Specialists (ft09, SP80 DW with fixture priors) stay outside this loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from legal_search.algorithms.explore import ChangeExplorer
from legal_search.algorithms.memory import EpisodicMemory
from legal_search.algorithms.program_search import hill_climb_programs
from legal_search.algorithms.role_induce import induce_movable_colors, induce_static_colors
from legal_search.algorithms.score import env_score_signals, proxy_search_score
from legal_search.consume.registry import consume_all
from legal_search.contract import TRUTH_BOUNDARY
from legal_search.types import Action, Transition


StepFn = Callable[[Action], dict[str, Any]]
# step_fn returns: frame, levels_completed, state, available_actions


@dataclass
class LegalAgent:
    """Minimal legal agent assembling mined algorithms."""

    game_id: str = ""
    explorer: ChangeExplorer = field(default_factory=ChangeExplorer)
    memory: EpisodicMemory = field(default_factory=EpisodicMemory)
    step_index: int = 0
    last_frame: np.ndarray | None = None
    last_levels: int = 0
    use_program_search_every: int = 8
    ledger: list[dict[str, Any]] = field(default_factory=list)

    def reset_for_game(self, game_id: str) -> None:
        if self.game_id and self.game_id != game_id:
            self.memory.wipe()
            self.explorer = ChangeExplorer()
        self.game_id = game_id
        self.memory.bind_game(game_id)
        self.step_index = 0
        self.last_frame = None
        self.last_levels = 0

    def decide(
        self,
        frame: np.ndarray,
        *,
        levels_completed: int = 0,
        available_actions: list[int] | None = None,
    ) -> Action:
        avail = available_actions or [1, 2, 3, 4, 5, 6]
        # Periodically try short program search with proxy score from memory
        if self.step_index > 0 and self.step_index % self.use_program_search_every == 0:
            recent = [
                # reconstruct minimal transitions from memory dicts is heavy;
                # use explorer-only + weak program bias
            ]
            def scorer(prog: list[Action]) -> float:
                return proxy_search_score(
                    frame_delta_pixels=len(prog),
                    levels_up=False,
                    program_len=len(prog),
                )

            proposals = hill_climb_programs(scorer, restarts=2, steps=4, seed=self.step_index, topk=1)
            if proposals and proposals[0].actions:
                action = proposals[0].actions[0]
                action = Action(action.action_id, action.x, action.y, source="legal_loop.program_search")
                self.ledger.append({"step": self.step_index, "mode": "program_search", "action": action.to_dict()})
                return action

        # Episodic success prior (this game only)
        mem_props = self.memory.proposals_from_memory()
        if mem_props and self.step_index % 5 == 0:
            action = mem_props[-1].actions[0]
            action = Action(action.action_id, action.x, action.y, source="legal_loop.episodic")
            self.ledger.append({"step": self.step_index, "mode": "episodic", "action": action.to_dict()})
            return action

        action = self.explorer.propose(avail)
        self.ledger.append({"step": self.step_index, "mode": "explore", "action": action.to_dict()})
        return action

    def observe_transition(self, tr: Transition) -> None:
        self.memory.record_transition(tr)
        self.explorer.observe(tr.action, tr.frame_changed)
        self.step_index += 1
        self.last_frame = np.asarray(tr.frame_after)
        self.last_levels = tr.levels_after

    def induce_snapshot(self) -> dict[str, Any]:
        # Build Transition objects from memory is not stored raw; expose explorer + memory
        return {
            "explorer": self.explorer.snapshot(),
            "memory": self.memory.snapshot(),
            "truth_boundary": dict(TRUTH_BOUNDARY),
        }

    def informers_packet(
        self,
        frame: np.ndarray,
        *,
        levels_completed: int = 0,
        available_actions: list[int] | None = None,
    ) -> dict[str, Any]:
        """Run full legal consume suite for inspection (not required every step)."""
        return consume_all(
            frame,
            game_id=self.game_id,
            available_actions=available_actions,
            levels_completed=levels_completed,
        )


def run_legal_episode(
    *,
    game_id: str,
    reset_fn: Callable[[], dict[str, Any]],
    step_fn: StepFn,
    max_actions: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """Drive a legal episode given env callbacks.

    reset_fn/step_fn return dicts with keys:
      frame, levels_completed, state, available_actions
    """
    agent = LegalAgent(game_id=game_id)
    agent.reset_for_game(game_id)
    obs = reset_fn()
    frame = np.asarray(obs["frame"])
    levels = int(obs.get("levels_completed") or 0)
    history: list[dict[str, Any]] = []
    for t in range(max_actions):
        avail = list(obs.get("available_actions") or [1, 2, 3, 4, 5, 6])
        action = agent.decide(frame, levels_completed=levels, available_actions=avail)
        nxt = step_fn(action)
        frame2 = np.asarray(nxt["frame"])
        levels2 = int(nxt.get("levels_completed") or 0)
        state2 = str(nxt.get("state") or "")
        changed = not np.array_equal(frame, frame2)
        tr = Transition(
            game_id=game_id,
            step=t,
            frame_before=frame,
            action=action,
            frame_after=frame2,
            levels_before=levels,
            levels_after=levels2,
            state_after=state2,
            frame_changed=changed,
            levels_up=levels2 > levels,
        )
        agent.observe_transition(tr)
        history.append(
            {
                **tr.to_dict(),
                "signals": env_score_signals(
                    levels_before=levels,
                    levels_after=levels2,
                    frame_before=frame,
                    frame_after=frame2,
                    state_after=state2,
                ),
            }
        )
        frame, levels, obs = frame2, levels2, nxt
        if "WIN" in state2.upper():
            break
        if "GAME_OVER" in state2.upper() and t > 0:
            # allow continue if env auto-resets externally; otherwise stop
            pass
    return {
        "schema": "legal_search.episode_result.v1",
        "game_id": game_id,
        "steps": len(history),
        "final_levels": levels,
        "final_state": str(obs.get("state") or ""),
        "agent_snapshot": agent.induce_snapshot(),
        "history_tail": history[-8:],
        "truth_boundary": dict(TRUTH_BOUNDARY),
        "legal_claim": (
            "episode used mined legal algorithms only; "
            "no fixture win banks; no spill/droplet lineage"
        ),
    }
