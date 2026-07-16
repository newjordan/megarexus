"""Stochastic Goose → legal explore algorithm."""

from __future__ import annotations

import numpy as np

from legal_search.algorithms.explore import ChangeExplorer
from legal_search.consume.base import ConsumeAdapter
from legal_search.types import Action, Hypothesis, LegalPacket, Proposal, Transition


class GooseConsume(ConsumeAdapter):
    system_id = "stochastic_goose"
    algorithm_id = "explore.change_by_repetition.v1"
    source_paths = (
        "stochastic_goose/custom_agents/action.py",
        "synthesis_lab/engine_adapters/goose_action_adapter.py",
        "codexlab/arc_play_harness.py::tufa_action_policy",
    )
    stripped_bindings = (
        "external_/tmp/ARC3-solution_default_path",
        "checkpoint_side_channels_unless_online",
        "game_id_seed_map_as_hidden_prior",
    )

    def __init__(self) -> None:
        self.explorer = ChangeExplorer()

    def observe(self, action: Action, changed: bool) -> None:
        self.explorer.observe(action, changed)

    def consume(
        self,
        frame: np.ndarray,
        *,
        game_id: str = "",
        transitions: list[Transition] | None = None,
        available_actions: list[int] | None = None,
        levels_completed: int = 0,
    ) -> LegalPacket:
        # Pure mine path: rebuild explorer from the provided transition window.
        # Avoids singleton double-count when encode/consume is called every step.
        if transitions is not None:
            explorer = ChangeExplorer(rng=self.explorer.rng)
            for tr in transitions[-64:]:
                explorer.observe(tr.action, tr.frame_changed)
            self.explorer = explorer
        avail = available_actions or [1, 2, 3, 4, 5, 6]
        action = self.explorer.propose(avail)
        snap = self.explorer.snapshot()
        # Structured rates for features (not only present=1)
        rates = {}
        for aid_s, row in (snap.get("action_stats") or {}).items():
            trials = float(row.get("trials") or 0)
            if trials > 0:
                rates[str(aid_s)] = float(row["changes"]) / trials
        n_click_bins = len(snap.get("click_bins") or {})
        best_click = 0.0
        if snap.get("click_bins"):
            best_click = max(
                (float(v["changes"]) / float(v["trials"]) for v in snap["click_bins"].values() if float(v.get("trials") or 0) > 0),
                default=0.0,
            )
        receipt = self.receipt(
            allowed_outputs=["change_biased_action", "online_stats"],
            banned_outputs=["win_condition", "game_source_rules"],
            notes="Goose core: prefer actions that change frames; online only.",
        )
        return LegalPacket(
            system_id=self.system_id,
            algorithm_id=self.algorithm_id,
            hypotheses=[
                Hypothesis(
                    kind="action_change_prior",
                    payload={
                        **snap,
                        "rates": rates,
                        "n_click_bins": n_click_bins,
                        "best_click_rate": best_click,
                        "n_transition_window": len(transitions or []),
                    },
                    confidence=0.5,
                    source_system=self.system_id,
                )
            ],
            proposals=[
                Proposal(
                    actions=[action],
                    score=0.5 if not rates else float(np.mean(list(rates.values()))),
                    kind="explore_action",
                    evidence={"available": avail, "rates": rates},
                    algorithm_id=self.algorithm_id,
                )
            ],
            features={"levels_completed": levels_completed, "game_id": game_id},
            receipt=receipt.to_dict(),
        )
