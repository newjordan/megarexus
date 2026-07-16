"""Real DeltaBand: build_deltaband_observation on before/after GeoBands from CONTRA."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from synthesis_lab.orchestration.rdl.brain.types import KnownMapView, OperatorResult, TruthFlags

_CODEX = Path(__file__).resolve().parents[5] / "codexlab"


def _first_geoband(packet: dict[str, Any] | None) -> dict[str, Any] | None:
    if not packet:
        return None
    geos = packet.get("geoband_hypotheses") or []
    if isinstance(geos, list) and geos and isinstance(geos[0], dict):
        return geos[0]
    # sometimes nested under statecons
    for sc in packet.get("state_condition_hypotheses") or []:
        if not isinstance(sc, dict):
            continue
        g = sc.get("geoband") or sc.get("geoband_hypothesis")
        if isinstance(g, dict):
            return g
    return None


class DeltaBandFullOp:
    name = "delta"

    def run(self, map_view: KnownMapView, context: dict[str, Any] | None = None) -> OperatorResult:
        context = context or {}
        prev = context.get("frame_prev")
        frame = map_view.frame if map_view.frame is not None else context.get("frame")
        before = np.asarray(prev) if prev is not None else None
        after = np.asarray(frame) if frame is not None else None
        scores: dict[str, float] = {}
        if before is not None and after is not None and before.shape == after.shape:
            scores["delta_pixels"] = float((before != after).sum())
            scores["did_change"] = 1.0 if np.any(before != after) else 0.0
            scores["delta_frac"] = float((before != after).mean())
            if scores["did_change"] > 0:
                ys, xs = np.nonzero(before != after)
                scores["change_cx"] = float(xs.mean())
                scores["change_cy"] = float(ys.mean())

        after_packet = context.get("contra_packet") or {}
        before_packet = context.get("contra_packet_prev") or {}
        after_g = _first_geoband(after_packet)
        before_g = _first_geoband(before_packet)

        if before_g is None or after_g is None:
            if scores:
                return OperatorResult(
                    name=self.name,
                    status="ok",
                    scores={**scores, "deltaband_geoband_ready": 0.0},
                    hypothesis_only=True,
                    truth_flags=TruthFlags(),
                    meta={"backend": "pixel_delta_awaiting_prev_contra"},
                    messages=["pixel_delta_need_prev_geoband"],
                )
            return OperatorResult(
                name=self.name,
                status="degraded",
                scores={},
                messages=["need_before_after_geoband"],
                hypothesis_only=True,
                truth_flags=TruthFlags(),
            )

        try:
            if str(_CODEX) not in sys.path:
                sys.path.insert(0, str(_CODEX))
            from contra_head.deltaband import build_deltaband_observation

            aid = str(context.get("last_action_id") or context.get("action_invocation_id") or "act0")
            obs = build_deltaband_observation(
                before_geoband=before_g,
                after_geoband=after_g,
                action_invocation_id=aid,
            )
            scores["deltaband_present"] = 1.0
            scores["deltaband_geoband_ready"] = 1.0
            if isinstance(obs, dict):
                for k, v in list(obs.items())[:20]:
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        scores[str(k)[:28]] = float(v)
                    elif isinstance(v, bool):
                        scores[str(k)[:28]] = 1.0 if v else 0.0
                    elif isinstance(v, (list, dict)):
                        scores[f"n_{k}"[:28]] = float(len(v))
            return OperatorResult(
                name=self.name,
                status="ok",
                scores=scores,
                summary={"action_invocation_id": aid},
                hypothesis_only=True,
                truth_flags=TruthFlags(),
                meta={"backend": "contra_head.deltaband.build_deltaband_observation"},
                messages=["deltaband_full"],
            )
        except Exception as exc:
            return OperatorResult(
                name=self.name,
                status="ok" if scores else "degraded",
                scores={**scores, "deltaband_error": 1.0},
                messages=[f"deltaband:{type(exc).__name__}:{exc}"],
                hypothesis_only=True,
                truth_flags=TruthFlags(),
                meta={"backend": "pixel_delta_after_deltaband_error"},
            )
