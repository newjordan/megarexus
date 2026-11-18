"""Full CONTRA controller packet — real run_contra_view, not a thin stand-in.

Fires the entire contra_head stack once per frame and fans scores into
multiple dream_summary channels so nothing important is silently dropped.
Hypothesis-only / fail-closed admission flags preserved.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from synthesis_lab.orchestration.rdl.brain.brain_types import KnownMapView, OperatorResult, TruthFlags

_REPO = Path(__file__).resolve().parents[5]
_CODEX = _REPO / "codexlab"


def _ensure_codex_path() -> None:
    p = str(_CODEX)
    if p not in sys.path:
        sys.path.insert(0, p)


def _countish(v: Any) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (int, float, bool)):
        return float(v)
    if isinstance(v, (list, tuple, dict, set)):
        return float(len(v))
    return 0.0


def _dig_counts(obj: Any, prefix: str = "", out: dict[str, float] | None = None, depth: int = 0) -> dict[str, float]:
    """Harvest numeric / length signals from nested CONTRA packet (bounded)."""
    if out is None:
        out = {}
    if depth > 3 or len(out) > 48:
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out[key[-40:]] = float(v)
            elif isinstance(v, bool):
                out[key[-40:]] = 1.0 if v else 0.0
            elif isinstance(v, (list, tuple)):
                out[f"n_{k}"[-40:]] = float(len(v))
                if v and isinstance(v[0], dict) and depth < 2:
                    _dig_counts(v[0], f"{k}0.", out, depth + 1)
            elif isinstance(v, dict) and depth < 2:
                out[f"n_{k}"[-40:]] = float(len(v))
                _dig_counts(v, f"{k}.", out, depth + 1)
    return out


def run_full_contra_packet(
    frame: np.ndarray,
    *,
    game_id: str = "sp80-589a99af",
    level_index: int = 0,
    ghost_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call real contra_head.api.run_contra_view. Raises on hard failure."""
    _ensure_codex_path()
    from contra_head.api import run_contra_view

    return run_contra_view(
        raw_frame=np.asarray(frame),
        game_id=str(game_id),
        level_index=int(level_index),
        mode="review",
        ghost_context=ghost_context,
    )


def fanout_contra_channels(packet: dict[str, Any]) -> dict[str, OperatorResult]:
    """Split one full CONTRA response into per-layer OperatorResults."""
    channels: dict[str, OperatorResult] = {}

    def make(name: str, section: Any, extra: dict[str, float] | None = None) -> OperatorResult:
        scores: dict[str, float] = {}
        if isinstance(section, dict):
            scores["present"] = 1.0
            scores["n_keys"] = float(len(section))
            # harvest a few top-level numbers
            for k, v in list(section.items())[:20]:
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    scores[str(k)[:28]] = float(v)
                elif isinstance(v, bool):
                    scores[str(k)[:28]] = 1.0 if v else 0.0
                elif isinstance(v, (list, tuple, dict)):
                    scores[f"n_{k}"[:28]] = float(len(v))
        elif isinstance(section, list):
            scores["present"] = 1.0
            scores["n_items"] = float(len(section))
        else:
            scores["present"] = 0.0
        if extra:
            scores.update(extra)
        # fail-closed admissions from root packet
        scores["control_truth_admitted"] = 1.0 if packet.get("control_truth_admitted") else 0.0
        scores["solve_condition_admitted"] = 1.0 if packet.get("solve_condition_admitted") else 0.0
        return OperatorResult(
            name=name,
            status="ok" if section not in (None, {}, []) else "degraded",
            scores=scores,
            summary={
                "schema": packet.get("schema"),
                "admission_status": packet.get("admission_status"),
                "not_control_truth": bool(packet.get("not_control_truth", True)),
                "not_solve_truth": bool(packet.get("not_solve_truth", True)),
            },
            hypothesis_only=True,
            truth_flags=TruthFlags(),
            meta={"backend": "contra_head.api.run_contra_view", "layer": name},
            messages=[f"contra:{name}"],
        )

    # Core layers from the live controller response
    channels["contra_view"] = make(
        "contra_view",
        packet,
        {
            "status_ok": 1.0 if packet.get("status") else 0.0,
            "n_fail_closed": float(len(packet.get("fail_closed_reasons") or [])),
            "n_handoffs": _countish(packet.get("layer_handoffs")),
            "n_layer_outputs": _countish(packet.get("layer_output_packets")),
        },
    )
    channels["contra_vision"] = make("contra_vision", packet.get("raw_color_component_packet"))
    channels["contra_control"] = make("contra_control", packet.get("control_isolation_packet"))
    channels["contra_surfaces"] = make("contra_surfaces", packet.get("scene_surfaces"))
    channels["contra_symmetry"] = make("contra_symmetry", packet.get("symmetry_hypotheses"))
    channels["contra_actor"] = make("contra_actor", packet.get("actor_affordance_hypotheses"))
    channels["contra_target"] = make("contra_target", packet.get("target_affordance_hypotheses"))
    channels["contra_support"] = make("contra_support", packet.get("support_action_affordance_hypotheses"))
    channels["contra_statecon"] = make("contra_statecon", packet.get("state_condition_hypotheses"))
    channels["contra_geoband"] = make("contra_geoband", packet.get("geoband_hypotheses"))
    channels["contra_boundary"] = make("contra_boundary", packet.get("boundary_queues"))
    channels["contra_ladder"] = make("contra_ladder", packet.get("contra_relationship_ladder"))
    channels["contra_delta_gate"] = make("contra_delta_gate", packet.get("deltaband_gate"))
    channels["contra_explore"] = make("contra_explore", packet.get("exploreband_budget"))
    channels["contra_oracle"] = make("contra_oracle", packet.get("oracle_pass"))
    channels["contra_ghost"] = make("contra_ghost", packet.get("ghost_pass"))
    channels["contra_episode"] = make("contra_episode", packet.get("episode_memory"))
    channels["contra_semantics"] = make("contra_semantics", packet.get("action_semantics_ledger"))
    channels["contra_admission"] = make("contra_admission", packet.get("admission") or packet.get("admission_status"))

    # PortBand nested under geoband / contra_pass if present
    geobands = packet.get("geoband_hypotheses") or []
    port_scores: dict[str, float] = {"n_geobands": float(len(geobands)) if isinstance(geobands, list) else 0.0}
    if isinstance(geobands, list):
        n_ports = n_align = n_lineup = 0
        for g in geobands[:8]:
            if not isinstance(g, dict):
                continue
            # common nesting patterns
            for key in ("ports", "port_hypotheses", "object_ports"):
                n_ports += len(g.get(key) or []) if isinstance(g.get(key), list) else 0
            for key in ("alignments", "port_alignments"):
                n_align += len(g.get(key) or []) if isinstance(g.get(key), list) else 0
            lm = g.get("lineup_matrix") or g.get("portband_lineup_matrix") or {}
            if isinstance(lm, dict):
                rows = lm.get("rows") or lm.get("lineup_rows") or []
                n_lineup += len(rows) if isinstance(rows, list) else 0
            # dig shallow
            dig = _dig_counts(g, depth=0)
            for k, v in list(dig.items())[:6]:
                if "score" in k.lower() or "n_" in k:
                    port_scores[k[-24:]] = v
        port_scores["n_ports"] = float(n_ports)
        port_scores["n_alignments"] = float(n_align)
        port_scores["n_lineup_rows"] = float(n_lineup)
    channels["contra_portband"] = OperatorResult(
        name="contra_portband",
        status="ok",
        scores=port_scores,
        hypothesis_only=True,
        truth_flags=TruthFlags(),
        meta={"backend": "run_contra_view.geoband_portband"},
        messages=["contra_portband_from_full_view"],
    )
    return channels


class ContraFullOp:
    """Single operator that also returns fanout via .last_fanout for hemisphere."""

    name = "contra_view"

    def __init__(self) -> None:
        self.last_packet: dict[str, Any] | None = None
        self.last_fanout: dict[str, OperatorResult] = {}

    def run(self, map_view: KnownMapView, context: dict[str, Any] | None = None) -> OperatorResult:
        context = context or {}
        frame = map_view.frame if map_view.frame is not None else context.get("frame")
        if frame is None:
            return OperatorResult(
                name=self.name,
                status="degraded",
                scores={},
                messages=["no_frame"],
                hypothesis_only=True,
                truth_flags=TruthFlags(),
            )
        try:
            packet = run_full_contra_packet(
                np.asarray(frame),
                game_id=str(context.get("game_id") or "sp80-589a99af"),
                level_index=int(context.get("level_index") or 0),
                ghost_context=context.get("ghost_context"),
            )
            self.last_packet = packet
            self.last_fanout = fanout_contra_channels(packet)
            # primary channel = full view rollup
            primary = self.last_fanout.get("contra_view")
            assert primary is not None
            primary.payload = {
                "admission_status": packet.get("admission_status"),
                "fail_closed_reasons": list(packet.get("fail_closed_reasons") or [])[:8],
                "layer_names": list(self.last_fanout.keys()),
            }
            return primary
        except Exception as exc:
            self.last_packet = None
            self.last_fanout = {}
            return OperatorResult(
                name=self.name,
                status="degraded",
                scores={},
                messages=[f"run_contra_view:{type(exc).__name__}:{exc}"],
                hypothesis_only=True,
                truth_flags=TruthFlags(),
            )
