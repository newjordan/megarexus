"""CONTRA controller stack → legal perception / hypothesis algorithms."""

from __future__ import annotations

import numpy as np

from legal_search.algorithms.components import enrich_components
from legal_search.algorithms.saliency import color_histogram, edge_map, symmetry_score_horizontal
from legal_search.consume.base import ConsumeAdapter
from legal_search.types import Hypothesis, LegalPacket, Transition


class ContraConsume(ConsumeAdapter):
    system_id = "contra_head"
    algorithm_id = "perception.components_edges_symmetry.v1"
    source_paths = (
        "codexlab/contra_head/",
        "codexlab/contra_head/controller_head.py",
        "codexlab/contra_head/vision.py",
        "codexlab/contra_head/perception.py",
        "synthesis_lab/engine_adapters/contra_perception_adapter.py",
        "legal_search/algorithms/components.py",
    )
    stripped_bindings = (
        "goal_condition pour_into_buckets",
        "calibration actor/target as admitted truth",
        "SP80 manual calibration tables as control truth",
        "any control_truth_admitted=True path",
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
        scene = enrich_components(frame)
        comps = scene["components"]
        contacts = scene["contact_graph"]
        feats = scene["component_features"]
        hyp = [
            Hypothesis(
                kind="components",
                payload={
                    "background": scene["background"],
                    "components": comps[:24],
                    "n_components": scene["n_components"],
                },
                confidence=0.7,
                source_system=self.system_id,
            ),
            Hypothesis(
                kind="contact_graph",
                payload=contacts,
                confidence=0.6 if contacts.get("contact_pair_count") else 0.3,
                source_system=self.system_id,
            ),
            Hypothesis(
                kind="component_features",
                payload={"features": feats[:12]},
                confidence=0.55,
                source_system=self.system_id,
            ),
            Hypothesis(
                kind="edge_density",
                payload={"mean_edge": float(edge_map(frame).mean())},
                confidence=0.6,
                source_system=self.system_id,
            ),
            Hypothesis(
                kind="symmetry_horizontal",
                payload={"score": symmetry_score_horizontal(frame)},
                confidence=0.5,
                source_system=self.system_id,
            ),
            Hypothesis(
                kind="color_histogram",
                payload={"hist": color_histogram(frame).tolist()},
                confidence=0.6,
                source_system=self.system_id,
            ),
        ]
        receipt = self.receipt(
            allowed_outputs=[
                "components",
                "contact_graph",
                "component_features",
                "edges",
                "symmetry",
                "color_hist",
            ],
            banned_outputs=["admitted_actor", "admitted_goal", "solve_truth"],
            notes="CONTRA mined as fail-closed perception substrate, not solver.",
        )
        return LegalPacket(
            system_id=self.system_id,
            algorithm_id=self.algorithm_id,
            hypotheses=hyp,
            proposals=[],
            features={
                "n_components": scene["n_components"],
                "background": scene["background"],
                "n_contacts": contacts.get("contact_pair_count", 0),
            },
            receipt=receipt.to_dict(),
        )
