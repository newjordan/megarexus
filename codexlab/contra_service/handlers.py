#!/usr/bin/env python3
"""Pure endpoint handlers -- the single source of truth behind every transport.

Each handler validates its request envelope and asserts the admission boundary
on its output.  The in-process client calls these directly (live-loop fast path,
no serialization); the HTTP shell is a thin adapter over the same functions, so
the two transports cannot diverge.  No subsystem admits truth here; only harness
execution does.

Dependency note: only :func:`recommendation` needs the heavy player harness, so
it is imported lazily.  Everything else depends on ``contra`` alone, keeping
``import contra_service.handlers`` cheap for the informer endpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

import contra_head
from contra_head.ghost import build_ghost, build_ghost_pass, load_ghost_context, write_ghost

from . import contracts as C
from . import frame_codec, projections

# codexlab/contra_service/handlers.py -> parents[2] == repo root (…/rexus)
ROOT = Path(__file__).resolve().parents[2]

_HARNESS = None


def _harness():
    """Lazily import + cache the player harness (heavy); only :func:`recommendation` needs it."""
    global _HARNESS
    if _HARNESS is None:
        import sp80_player_harness as _h

        _HARNESS = _h
    return _HARNESS


def _require_dict(request: Any, *, label: str) -> dict:
    if not isinstance(request, dict):
        raise ValueError(f"{label}: request must be a dict, got {type(request).__name__}")
    return request


def _resolve_frame(request: dict, *, root: Path) -> np.ndarray:
    """In-process numpy frames pass through untouched (exact live-loop parity);
    encoded / referenced frames go through the codec (canonical sha guard)."""
    rf = request.get("raw_frame")
    if isinstance(rf, np.ndarray):
        return rf
    return frame_codec.resolve_frame(request, root=root)


# --- CONTRA synthesis -------------------------------------------------------
def contra_view(request: dict, *, root: Path = ROOT) -> dict:
    _require_dict(request, label="contra_view")
    frame = _resolve_frame(request, root=root)
    response = contra_head.run_contra_view(
        raw_frame=frame,
        game_id=str(request.get("game_id")),
        level_index=int(request.get("level_index") or 0),
        calibration=request.get("calibration"),
        mode=str(request.get("mode") or "review"),
        ghost_context=request.get("ghost_context"),
    )
    C.assert_no_truth_admission(response, label="contra_view")
    return response


# --- Oracle informer --------------------------------------------------------
def oracle_pass(request: dict, *, root: Path = ROOT) -> dict:
    response = contra_view(request, root=root)
    pass_ = projections.project_oracle_pass(response)
    C.assert_no_truth_admission(pass_, label="oracle_pass")
    return pass_


# --- Ghost informer ---------------------------------------------------------
def ghost_pass(request: dict, *, root: Path = ROOT) -> dict:
    _require_dict(request, label="ghost_pass")
    pass_ = build_ghost_pass(
        game_id=str(request.get("game_id")),
        level_index=int(request.get("level_index") or 0),
        raw_frame_sha256=request.get("raw_frame_sha256"),
        ghost_context=request.get("ghost_context"),
        invalid_input=bool(request.get("invalid_input", False)),
    )
    C.assert_no_truth_admission(pass_, label="ghost_pass")
    return pass_


def ghost_context(request: dict, *, root: Path = ROOT) -> dict:
    _require_dict(request, label="ghost_context")
    run_dir = request.get("run_dir")
    if not run_dir:
        raise ValueError("ghost_context: request requires 'run_dir'")
    return load_ghost_context(
        run_dir=Path(str(run_dir)),
        root=Path(str(request.get("root") or root)),
        game_id=str(request.get("game_id")),
    )


# --- CONTRA guidance (informer to the harness) ------------------------------
def recommendation(request: dict, *, root: Path = ROOT) -> dict:
    _require_dict(request, label="recommendation")
    packet = _harness().build_contra_recommendation_packet(
        level_index=int(request.get("level_index") or 0),
        action_index=int(request.get("action_index") or 0),
        before_state=request.get("before_state") or {},
        goal_condition=request.get("goal_condition") or {},
        contra_packet=request.get("contra_packet") or {},
        action_memory=request.get("action_memory") or [],
    )
    C.assert_no_truth_admission(packet, label="recommendation")
    return packet


# --- Harness -> Ghost-memory return edge ------------------------------------
def feedback_ingest(request: dict, *, root: Path = ROOT) -> dict:
    """Persist relation memory ONLY on a verified controlled completion.

    ``build_ghost`` returns None unless ``levels_completed_delta > 0`` -- that is
    the reward-hacking gate.  ``persist`` defaults to False (dry-run): the handler
    reports whether a ghost WOULD be captured without mutating disk, so parity
    tests never write.  Stage 6 calls with ``persist=True`` at episode end.
    """
    _require_dict(request, label="feedback_ingest")
    run_dir = request.get("run_dir")
    if not run_dir:
        raise ValueError("feedback_ingest: request requires 'run_dir'")
    rdir = Path(str(run_dir))
    rroot = Path(str(request.get("root") or root))
    game_id = str(request.get("game_id"))
    result = request.get("result") or {}
    packets = request.get("packets") or []
    persist = bool(request.get("persist", False))

    ghost = build_ghost(result=result, packets=packets, run_dir=rdir, root=rroot, game_id=game_id)
    write_summary = None
    if ghost is not None and persist:
        write_summary = write_ghost(
            result=result, packets=packets, run_dir=rdir, root=rroot, game_id=game_id
        )
    return {
        "schema": C.CONTRA_SERVICE_FEEDBACK_INGEST_RESULT_SCHEMA,
        "status": "ghost_captured" if ghost is not None else "no_controlled_completion_no_ghost",
        "captured": ghost is not None,
        "persisted": write_summary is not None,
        "write_summary": write_summary,
        "ghost": ghost,
        "evidence_boundary": "ghost_memory_not_future_truth",
        "completion_gate": "levels_completed_delta_gt_0_required",
        "admission_status": "not_admitted",
        "control_truth_admitted": False,
        "goal_condition_admitted": False,
        "action_choice_admitted": False,
        "solve_condition_admitted": False,
    }


# --- Meta -------------------------------------------------------------------
def contracts(request: dict | None = None, *, root: Path = ROOT) -> dict:
    return C.contracts_document()


def health(request: dict | None = None, *, root: Path = ROOT) -> dict:
    return {"schema": C.CONTRA_SERVICE_HEALTH_SCHEMA, "status": "ok", "service": "contra_informer_dag"}


# Handler registry by contract handler-name (used by the HTTP shell + verifier).
HANDLERS = {
    "contra_view": contra_view,
    "oracle_pass": oracle_pass,
    "ghost_pass": ghost_pass,
    "ghost_context": ghost_context,
    "recommendation": recommendation,
    "feedback_ingest": feedback_ingest,
    "contracts": contracts,
    "health": health,
}
