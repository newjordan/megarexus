"""DreamWeaver / sim3d → legal search + imagine interfaces."""

from __future__ import annotations

import numpy as np

from legal_search.algorithms.dynamics import mine_dynamics
from legal_search.algorithms.perception_trust import gate_frame_structure
from legal_search.algorithms.program_search import hill_climb_programs
from legal_search.algorithms.score import proxy_search_score
from legal_search.consume.base import ConsumeAdapter
from legal_search.types import Action, Hypothesis, LegalPacket, Proposal, Transition


class DreamWeaverConsume(ConsumeAdapter):
    system_id = "dreamweaver_sim3d"
    algorithm_id = "search.imagine_then_rank.v1"
    source_paths = (
        "synthesis_lab/orchestration/sim3d/dw_solve.py",
        "synthesis_lab/orchestration/sim3d/imagination_kernel.py",
        "synthesis_lab/orchestration/sim3d/virtual_arranger.py",
        "synthesis_lab/orchestration/sim3d/perception_gate.py",
        "codexlab/arc_play_harness.py::megarexus_dw_directgrid_policy",
        "legal_search/algorithms/dynamics.py",
        "legal_search/algorithms/program_search.py",
    )
    stripped_bindings = (
        "fixture_genome_prior_candidates",
        "fixture_direct_prior_candidates",
        "dynamics_round4/5_json_as_permanent_payload",
        "level_strategy_table_L0_L5",
        "PAINT/lava/mouth hardcoded roles",
        "sp80_GAME_ID constants in policy",
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
        recent = list(transitions or [])[-32:]
        dyn = mine_dynamics(frame, recent, available_actions=available_actions)
        trust = gate_frame_structure(frame)

        def scorer(prog: list[Action]) -> float:
            if not recent:
                return -0.05 * len(prog)
            ids = [a.action_id for a in prog]
            match = sum(1 for a in recent if a.action.action_id in ids and a.frame_changed)
            return proxy_search_score(
                frame_delta_pixels=match * 8,
                levels_up=any(t.levels_up for t in recent),
                program_len=len(prog),
            )

        climb = hill_climb_programs(
            scorer, restarts=4, steps=8, seed=hash(game_id) % 10000, topk=2
        )
        # merge imagine proposals + hill-climb
        proposals: list[Proposal] = list(climb)
        for p in dyn.get("proposals") or []:
            proposals.append(p)
        proposals.sort(key=lambda p: -p.score)
        proposals = proposals[:6]

        best = proposals[0] if proposals else None
        search_stats = {
            "n_proposals": len(proposals),
            "n_recent_transitions": len(recent),
            "best_score": float(best.score) if best else 0.0,
            "best_len": len(best.actions) if best else 0,
            "dynamics_hypotheses": dyn.get("hypotheses"),
            "best_dynamics": dyn.get("best_hypothesis"),
            "induced_blob_color": dyn.get("induced_blob_color"),
            "perception_trust": trust,
            "trust_score": trust.get("trust_score"),
            "fixture_priors_loaded": False,
            "shipped_dynamics_json_loaded": False,
            "fixture_scene_loaded": False,
            "stripped": list(self.stripped_bindings),
            "scorer": "transition_proxy_plus_dynamics_imagine",
            "interface": dyn.get("interface"),
        }
        receipt = self.receipt(
            allowed_outputs=[
                "program_proposals",
                "search_stats",
                "dynamics_hypothesis_scores",
                "imagine_rollouts",
                "perception_trust_gate",
            ],
            banned_outputs=[
                "fixture_win_priors",
                "source_derived_dynamics",
                "hardcoded_win_condition",
            ],
            notes=(
                "Legal DW: DynamicsHypothesis + perception trust gate + program search. "
                "No shipped SP80 JSON or fixture banks."
            ),
        )
        return LegalPacket(
            system_id=self.system_id,
            algorithm_id=self.algorithm_id,
            hypotheses=[
                Hypothesis(
                    kind="search_without_shipped_priors",
                    payload=search_stats,
                    confidence=0.55 if recent else 0.4,
                    source_system=self.system_id,
                )
            ],
            proposals=proposals,
            features={"levels_completed": levels_completed, "game_id": game_id},
            receipt=receipt.to_dict(),
        )
