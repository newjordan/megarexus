"""Loom / LeWM / Rexus → legal world-model + candidate scoring interfaces."""

from __future__ import annotations

import numpy as np

from legal_search.algorithms.cem_plan import discrete_cem_plan
from legal_search.algorithms.worldmodel import mine_worldmodel
from legal_search.consume.base import ConsumeAdapter
from legal_search.types import Hypothesis, LegalPacket, Transition


class LoomLeWMConsume(ConsumeAdapter):
    system_id = "loom_lewm_rexus"
    algorithm_id = "worldmodel.candidate_predict_rank.v1"
    source_paths = (
        "architecture/src/lewm/",
        "architecture/src/loom_layers/",
        "architecture/docs/canon_architecture/",
        "lewmtrain/",
        "codexlab/lewm_bridge/",
        "legal_search/algorithms/worldmodel.py",
        "legal_search/algorithms/dynamics.py",
    )
    stripped_bindings = (
        "runtime_admission_without_evidence",
        "GRAM++_as_unlocked_by_default",
        "Q6_as_action_owner",
        "pretrain_claims_as_live_policy",
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
        mined = mine_worldmodel(
            frame,
            transitions=transitions,
            available_actions=available_actions,
        )
        cem = discrete_cem_plan(
            frame,
            available_actions=available_actions,
            transitions=transitions,
            seed=hash(game_id) % 10000,
        )
        proposals = list(mined.get("proposals") or [])
        proposals.extend(list(cem.get("proposals") or []))
        proposals.sort(key=lambda p: -p.score)
        receipt = self.receipt(
            allowed_outputs=[
                "candidate_rank_by_change_affordance",
                "multi_step_consistency",
                "dynamics_backed_imagine",
                "cem_discrete_plan",
            ],
            banned_outputs=["admitted_live_policy_without_fit", "game_specific_pretrained_head"],
            notes=(
                "LeWM/Loom: affordance rank + multi-step consistency + discrete CEM. "
                "Runtime admitted only with transition evidence. No frozen weight requirement."
            ),
        )
        return LegalPacket(
            system_id=self.system_id,
            algorithm_id=self.algorithm_id,
            hypotheses=[
                Hypothesis(
                    kind="wm_interface",
                    payload={
                        "interface": mined["interface"],
                        "affordance_objective": mined["affordance_objective"],
                        "affordance_rates": mined["affordance_rates"],
                        "multi_step_consistency": mined["multi_step_consistency"],
                        "dynamics": mined["dynamics"],
                        "cem": {
                            "best_sequence": cem.get("best_sequence"),
                            "best_score": cem.get("best_score"),
                            "horizon": cem.get("horizon"),
                            "num_iterations": cem.get("num_iterations"),
                            "requires_lewm_weights": False,
                            "scorer": cem.get("scorer"),
                        },
                        "runtime_admitted": mined["runtime_admitted"] or bool(cem.get("runtime_admitted")),
                        "runtime_requires_evidence": True,
                        "encoder_interface": mined["encoder_interface"],
                        "cem_interface": "discrete_cem_dynamics_scored",
                        "n_transitions": mined["n_transitions"],
                    },
                    confidence=0.6 if mined["n_transitions"] else 0.4,
                    source_system=self.system_id,
                )
            ],
            proposals=proposals[:16],
            features={"frame_shape": list(np.asarray(frame).shape)},
            receipt=receipt.to_dict(),
        )
