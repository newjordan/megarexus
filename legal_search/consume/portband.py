"""PortBand → legal edge/port geometry + edge-lock + lineup hypotheses."""

from __future__ import annotations

import numpy as np

from legal_search.algorithms.components import connected_components
from legal_search.algorithms.ports import mine_ports_full
from legal_search.consume.base import ConsumeAdapter
from legal_search.types import Hypothesis, LegalPacket, Transition


class PortBandConsume(ConsumeAdapter):
    system_id = "portband"
    algorithm_id = "geometry.object_edge_ports.v1"
    source_paths = (
        "codexlab/contra_head/portband.py",
        "synthesis_lab/engine_adapters/portband_glove_adapter.py",
        "legal_search/algorithms/ports.py",
    )
    stripped_bindings = (
        "paintflow_route_flow_lock as solve signal",
        "lineup_matrix as action authority",
        "glove=catch-play SP80 role name as truth",
        "static edge-lock as solution configuration",
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
        comps = connected_components(frame)[:12]
        h, w = np.asarray(frame).shape[:2]
        mined = mine_ports_full(
            comps, frame_shape=(h, w), transitions=transitions
        )
        locks = mined["edge_lock_ledger"]
        lineup = mined["lineup_matrix"]
        fleet = mined.get("fleet") or {}
        events = mined.get("event_surfaces") or {}
        receipt = self.receipt(
            allowed_outputs=[
                "edge_ports",
                "alignment_candidates",
                "edge_lock_hypotheses",
                "lineup_hypotheses",
                "fleet_combo_candidates",
                "event_surface_correlations",
            ],
            banned_outputs=["route_as_solution", "edge_lock_as_truth", "action_choice"],
            notes="PortBand full mine: ports + edge-lock + lineup + fleet + event surfaces as hypotheses only.",
        )
        return LegalPacket(
            system_id=self.system_id,
            algorithm_id=self.algorithm_id,
            hypotheses=[
                Hypothesis(
                    kind="object_ports",
                    payload={
                        "ports": mined["ports"],
                        "n_ports": mined["n_ports"],
                        "ports_by_side": mined["ports_by_side"],
                        "mean_port_span": mined["mean_port_span"],
                    },
                    confidence=0.55,
                    source_system=self.system_id,
                ),
                Hypothesis(
                    kind="port_alignments",
                    payload={
                        "alignments": mined["alignments"][:24],
                        "n_alignments": mined["n_alignments"],
                        "best_alignment_score": mined["best_alignment_score"],
                    },
                    confidence=0.5 if mined["n_alignments"] else 0.2,
                    source_system=self.system_id,
                ),
                Hypothesis(
                    kind="edge_lock_ledger",
                    payload=locks,
                    confidence=0.45 if locks.get("n_locks") else 0.15,
                    source_system=self.system_id,
                ),
                Hypothesis(
                    kind="lineup_matrix",
                    payload=lineup,
                    confidence=0.4 if lineup.get("n_rows") else 0.15,
                    source_system=self.system_id,
                ),
                Hypothesis(
                    kind="fleet_combos",
                    payload=fleet,
                    confidence=0.4 if fleet.get("n_candidates") else 0.15,
                    source_system=self.system_id,
                ),
                Hypothesis(
                    kind="event_surfaces",
                    payload=events,
                    confidence=0.45 if events.get("n_events") else 0.15,
                    source_system=self.system_id,
                ),
            ],
            proposals=[],
            features={
                "n_ports": mined["n_ports"],
                "n_alignments": mined["n_alignments"],
                "n_edge_locks": locks.get("n_locks", 0),
                "n_fleet": fleet.get("n_candidates", 0),
                "n_events": events.get("n_events", 0),
                "best_alignment_score": mined["best_alignment_score"],
                "best_edge_lock": locks.get("best_quality", 0.0),
                "best_lineup": lineup.get("best_lineup_score", 0.0),
                "best_fleet": fleet.get("best_rank_score", 0.0),
            },
            receipt=receipt.to_dict(),
        )
