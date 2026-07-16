"""Registry of every mined system → legal consume adapter."""

from __future__ import annotations

from typing import Any

import numpy as np

from legal_search.consume.bands import (
    ActionModBandConsume,
    DeltaBandConsume,
    EpisodeMemoryConsume,
    ExploreBandConsume,
    GeoBandConsume,
    StateConConsume,
)
from legal_search.consume.contra import ContraConsume
from legal_search.consume.dreamweaver import DreamWeaverConsume
from legal_search.consume.ghost import GhostConsume
from legal_search.consume.goose import GooseConsume
from legal_search.consume.loom_lewm import LoomLeWMConsume
from legal_search.consume.misc import (
    ActionSemanticsConsume,
    ControllableAffordancesConsume,
    CrucibleConsume,
    MatchCalibrationConsume,
    MegarexusCNNConsume,
    PufferRLConsume,
    QuarantineFenceConsume,
    RelationshipLadderConsume,
    SpecialistNoteConsume,
    SymmetryConsume,
    TabulaRasaConsume,
    VisionConsume,
)
from legal_search.consume.oracle import OracleConsume
from legal_search.consume.portband import PortBandConsume
from legal_search.types import LegalPacket, Transition


def _build_registry() -> dict[str, Any]:
    return {
        "stochastic_goose": GooseConsume(),
        "dreamweaver_sim3d": DreamWeaverConsume(),
        "contra_head": ContraConsume(),
        "portband": PortBandConsume(),
        "oracle": OracleConsume(),
        "ghost": GhostConsume(),
        "geoband": GeoBandConsume(),
        "deltaband": DeltaBandConsume(),
        "exploreband": ExploreBandConsume(),
        "actionmodband": ActionModBandConsume(),
        "statecon": StateConConsume(),
        "episode_memory": EpisodeMemoryConsume(),
        "loom_lewm_rexus": LoomLeWMConsume(),
        "symmetry": SymmetryConsume(),
        "contra_vision": VisionConsume(),
        "crucible": CrucibleConsume(),
        "megarexus_cnn": MegarexusCNNConsume(),
        "specialist_solvers": SpecialistNoteConsume(),
        "pufferlib_rl": PufferRLConsume(),
        "quarantine_fence": QuarantineFenceConsume(),
        "tabula_rasa": TabulaRasaConsume(),
        "relationship_ladder": RelationshipLadderConsume(),
        "match_calibration": MatchCalibrationConsume(),
        "controller_affordances": ControllableAffordancesConsume(),
        "action_semantics_ledger": ActionSemanticsConsume(),
    }


CONSUME_REGISTRY = _build_registry()

# Systems that may emit action proposals into the legal loop
PROPOSAL_SYSTEMS = (
    "stochastic_goose",
    "dreamweaver_sim3d",
    "exploreband",
    "ghost",
    "episode_memory",
    "loom_lewm_rexus",
)

# Systems that only inform (hypotheses)
INFORMER_SYSTEMS = tuple(
    k for k in CONSUME_REGISTRY if k not in PROPOSAL_SYSTEMS and k != "quarantine_fence"
)


def list_systems() -> list[dict[str, str]]:
    rows = []
    for key, adapter in CONSUME_REGISTRY.items():
        rows.append(
            {
                "system_id": key,
                "algorithm_id": adapter.algorithm_id,
                "role": "proposal" if key in PROPOSAL_SYSTEMS else "informer_or_guard",
            }
        )
    return rows


def consume_one(
    system_id: str,
    frame: np.ndarray,
    **kwargs: Any,
) -> LegalPacket:
    if system_id not in CONSUME_REGISTRY:
        raise KeyError(f"unknown system {system_id}; known={sorted(CONSUME_REGISTRY)}")
    return CONSUME_REGISTRY[system_id].consume(frame, **kwargs)


def consume_all(
    frame: np.ndarray,
    *,
    game_id: str = "",
    transitions: list[Transition] | None = None,
    available_actions: list[int] | None = None,
    levels_completed: int = 0,
    systems: list[str] | None = None,
) -> dict[str, Any]:
    keys = systems or list(CONSUME_REGISTRY.keys())
    packets = {}
    for key in keys:
        packets[key] = CONSUME_REGISTRY[key].consume(
            frame,
            game_id=game_id,
            transitions=transitions,
            available_actions=available_actions,
            levels_completed=levels_completed,
        )
    return {
        "schema": "legal_search.consume_all.v1",
        "game_id": game_id,
        "n_systems": len(packets),
        "systems": {k: v.to_dict() for k, v in packets.items()},
        "proposal_systems": list(PROPOSAL_SYSTEMS),
        "informer_systems": list(INFORMER_SYSTEMS),
    }
