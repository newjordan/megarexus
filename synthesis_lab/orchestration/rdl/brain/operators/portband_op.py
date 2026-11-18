"""PortBand conceptual operator — lineup / alignment hypotheses from legal mine.

Uses legal_search.algorithms.ports (mined from contra_head/portband.py).
Hypothesis-only; never admits action/solve truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from synthesis_lab.orchestration.rdl.brain.brain_types import KnownMapView, OperatorResult, TruthFlags


class PortBandOp:
    name = "portband"

    def run(self, map_view: KnownMapView, context: dict[str, Any] | None = None) -> OperatorResult:
        context = context or {}
        components = list(map_view.components or context.get("components") or [])
        if not components:
            return OperatorResult(
                name=self.name,
                status="degraded",
                scores={},
                messages=["no_components"],
                hypothesis_only=True,
                truth_flags=TruthFlags(),
            )
        try:
            from legal_search.algorithms.ports import (
                object_ports,
                span_overlap_ratio,
                OPPOSITE,
            )
        except Exception as exc:
            return OperatorResult(
                name=self.name,
                status="degraded",
                scores={},
                messages=[f"ports_import:{type(exc).__name__}"],
                hypothesis_only=True,
                truth_flags=TruthFlags(),
            )

        all_ports: list[dict[str, Any]] = []
        for c in components:
            try:
                all_ports.extend(object_ports(c))
            except Exception:
                continue

        # Facing alignments (PortBand core): opposite sides with span overlap
        alignments: list[dict[str, Any]] = []
        by_side: dict[str, list[dict[str, Any]]] = {}
        for p in all_ports:
            by_side.setdefault(str(p.get("side")), []).append(p)

        for side, ports_a in by_side.items():
            opp = OPPOSITE.get(side)
            if not opp:
                continue
            for pa in ports_a:
                for pb in by_side.get(opp, []):
                    if pa.get("component_id") == pb.get("component_id"):
                        continue
                    ov = span_overlap_ratio(pa.get("span_px") or [0, 0], pb.get("span_px") or [0, 0])
                    # axis separation (gap)
                    gap = abs(float(pa.get("axis_coordinate_px") or 0) - float(pb.get("axis_coordinate_px") or 0))
                    # closer facing ports score higher if good overlap
                    score = float(ov) * (1.0 / (1.0 + gap / 32.0))
                    if score < 0.05:
                        continue
                    alignments.append(
                        {
                            "a": pa.get("port_id"),
                            "b": pb.get("port_id"),
                            "side": side,
                            "overlap": float(ov),
                            "gap": float(gap),
                            "score": score,
                            "admitted": False,
                        }
                    )

        alignments.sort(key=lambda r: -r["score"])
        top = alignments[:12]
        best = float(top[0]["score"]) if top else 0.0
        mean = float(np.mean([a["score"] for a in top])) if top else 0.0

        # Lineup matrix lite: per-component best alignment score
        per_comp: dict[str, float] = {}
        for a in top:
            for key in ("a", "b"):
                pid = str(a.get(key) or "")
                cid = pid.split(":")[0] if pid else ""
                if cid:
                    per_comp[cid] = max(per_comp.get(cid, 0.0), float(a["score"]))

        return OperatorResult(
            name=self.name,
            status="ok",
            scores={
                "n_ports": float(len(all_ports)),
                "n_alignments": float(len(alignments)),
                "n_top_lineup": float(len(top)),
                "best_lineup_score": best,
                "mean_top_lineup": mean,
                "n_components_ranked": float(len(per_comp)),
            },
            proposals=[
                {
                    "kind": "portband_lineup",
                    "score": a["score"],
                    "a": a["a"],
                    "b": a["b"],
                    "admitted": False,
                }
                for a in top[:8]
            ],
            summary={
                "top_alignments": top[:5],
                "per_component_best": per_comp,
                "policy": "static_portband_lineup_score_not_probability",
                "boundary": "hypothesis_only_not_action_authority",
            },
            messages=[f"portband_lineups={len(top)} best={best:.3f}"],
            hypothesis_only=True,
            truth_flags=TruthFlags(),
            meta={"backend": "legal_search.algorithms.ports"},
        )
