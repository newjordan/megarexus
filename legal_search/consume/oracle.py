"""Oracle → legal missing-element assumption generator (never truth)."""

from __future__ import annotations

import numpy as np

from legal_search.algorithms.components import connected_components
from legal_search.consume.base import ConsumeAdapter
from legal_search.types import Hypothesis, LegalPacket, Transition


class OracleConsume(ConsumeAdapter):
    system_id = "oracle"
    algorithm_id = "hypothesis.missing_element_assumption.v1"
    source_paths = ("codexlab/oracle.py",)
    stripped_bindings = (
        "assumption_as_control_truth",
        "assumption_as_action_choice",
        "assumption_as_solve_evidence",
    )

    def consume(
        self,
        frame: np.ndarray,
        *,
        game_id: str = "",
        transitions: list[Transition] | None = None,
        available_actions: list[int] | None = None,
        levels_completed: int = 0,
    ) -> LegalPacket:
        comps = connected_components(frame)
        # One small assumption at a time: largest non-background-ish component
        # might be worth probing — review object only.
        assumption = None
        if comps:
            top = comps[0]
            assumption = {
                "missing_element": "probe_role_unknown",
                "focus_component_id": top["component_id"],
                "focus_center_xy": top["center_xy"],
                "rationale": "largest_component_unlabeled_role",
                "review_only": True,
            }
        receipt = self.receipt(
            allowed_outputs=["one_assumption_review_object"],
            banned_outputs=["admitted_goal", "admitted_action", "prediction_as_fact"],
            notes="Oracle mined as single-hypothesis proposal factory, not solver.",
        )
        hyps = []
        if assumption:
            hyps.append(
                Hypothesis(
                    kind="oracle_assumption",
                    payload=assumption,
                    confidence=0.3,
                    source_system=self.system_id,
                )
            )
        return LegalPacket(
            system_id=self.system_id,
            algorithm_id=self.algorithm_id,
            hypotheses=hyps,
            proposals=[],
            features={},
            receipt=receipt.to_dict(),
        )
