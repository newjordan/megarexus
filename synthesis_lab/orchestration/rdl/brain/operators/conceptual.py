"""ConceptualOp — component centroids + pairwise alignments as pseudo-ports.

Uses legal_search.algorithms.components / ports when importable; else local CC
and local facing-port alignment scores. All hypothesis flags false for truth
admission (no control/goal/action/solve truth).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from synthesis_lab.orchestration.rdl.brain.brain_types import KnownMapView, OperatorResult, TruthFlags

_REPO = Path(__file__).resolve().parents[5]

SIDES = ("top", "bottom", "left", "right")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def _local_port_row(component_id: str, color: int, bbox_xyxy: list[int], side: str) -> dict[str, Any]:
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    if side == "left":
        center = [float(x0), float(y0 + y1) / 2.0]
        span = [int(y0), int(y1)]
        coord = float(x0)
    elif side == "right":
        center = [float(x1), float(y0 + y1) / 2.0]
        span = [int(y0), int(y1)]
        coord = float(x1)
    elif side == "top":
        center = [float(x0 + x1) / 2.0, float(y0)]
        span = [int(x0), int(x1)]
        coord = float(y0)
    else:
        center = [float(x0 + x1) / 2.0, float(y1)]
        span = [int(x0), int(x1)]
        coord = float(y1)
    return {
        "port_id": f"{component_id}:{side}",
        "component_id": component_id,
        "color": int(color),
        "side": side,
        "center_xy": center,
        "axis_coordinate_px": coord,
        "span_px": span,
        "span_len": int(span[1] - span[0] + 1),
        "bbox_xyxy": [x0, y0, x1, y1],
        "admitted": False,
        "evidence_status": "static_geometry_hypothesis",
    }


def _span_overlap_ratio(span_a: list[int], span_b: list[int]) -> float:
    a0, a1 = int(span_a[0]), int(span_a[1])
    b0, b1 = int(span_b[0]), int(span_b[1])
    if a1 < a0:
        a0, a1 = a1, a0
    if b1 < b0:
        b0, b1 = b1, b0
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = max(a1, b1) - min(a0, b0) + 1
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _alignment_score(port_a: dict[str, Any], port_b: dict[str, Any], *, frame_diag: float) -> float:
    if OPPOSITE.get(port_a["side"]) != port_b["side"]:
        return 0.0
    if port_a["component_id"] == port_b["component_id"]:
        return 0.0
    overlap = _span_overlap_ratio(port_a["span_px"], port_b["span_px"])
    if overlap <= 0.0:
        return 0.0
    ax, ay = port_a["center_xy"]
    bx, by = port_b["center_xy"]
    if port_a["side"] in ("left", "right"):
        gap = abs(ax - bx)
    else:
        gap = abs(ay - by)
    gap_term = 1.0 / (1.0 + gap / max(1.0, frame_diag * 0.25))
    size_sim = min(port_a["span_len"], port_b["span_len"]) / max(
        1, max(port_a["span_len"], port_b["span_len"])
    )
    color_term = 0.15 if int(port_a.get("color") or 0) != int(port_b.get("color") or 0) else 0.05
    score = 0.45 * overlap + 0.30 * gap_term + 0.15 * size_sim + color_term
    return float(max(0.0, min(1.0, score)))


def _local_ports_and_alignments(
    components: list[dict[str, Any]],
    *,
    frame_shape: tuple[int, int],
    max_alignments: int = 24,
) -> dict[str, Any]:
    ports: list[dict[str, Any]] = []
    for c in components:
        box = c["bbox_xyxy"]
        cid = str(c["component_id"])
        color = int(c.get("color") or 0)
        ports.extend(_local_port_row(cid, color, box, side) for side in SIDES)
    h, w = int(frame_shape[0]), int(frame_shape[1])
    diag = float(np.hypot(h, w))
    alignments: list[dict[str, Any]] = []
    for i, a in enumerate(ports):
        for b in ports[i + 1 :]:
            score = _alignment_score(a, b, frame_diag=diag)
            if score < 0.25:
                continue
            alignments.append(
                {
                    "a_port": a["port_id"],
                    "b_port": b["port_id"],
                    "a_component": a["component_id"],
                    "b_component": b["component_id"],
                    "a_side": a["side"],
                    "b_side": b["side"],
                    "kind": "facing_port_pair",
                    "score": score,
                    "span_overlap": _span_overlap_ratio(a["span_px"], b["span_px"]),
                    "admitted": False,
                    "hypothesis_only": True,
                }
            )
    alignments.sort(key=lambda r: -float(r["score"]))
    alignments = alignments[:max_alignments]
    return {
        "ports": ports,
        "alignments": alignments,
        "n_ports": len(ports),
        "n_alignments": len(alignments),
        "best_alignment_score": float(alignments[0]["score"]) if alignments else 0.0,
        "source": "local_pseudo_ports",
    }


def _try_legal_ports(
    components: list[dict[str, Any]],
    frame_shape: tuple[int, int],
) -> dict[str, Any] | None:
    try:
        if str(_REPO) not in sys.path:
            sys.path.insert(0, str(_REPO))
        from legal_search.algorithms.ports import mine_ports_and_alignments  # noqa: WPS433

        packet = mine_ports_and_alignments(components, frame_shape=frame_shape)
        packet = dict(packet)
        packet["source"] = "legal_search.algorithms.ports"
        # force fail-closed admission on every alignment/port
        for p in packet.get("ports") or []:
            p["admitted"] = False
        for a in packet.get("alignments") or []:
            a["admitted"] = False
            a["hypothesis_only"] = True
        return packet
    except Exception:
        return None


def _centroid_proposals(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in components:
        cx, cy = float(c["center_xy"][0]), float(c["center_xy"][1])
        out.append(
            {
                "kind": "component_centroid",
                "component_id": c.get("component_id"),
                "color": int(c.get("color") or 0),
                "center_xy": [cx, cy],
                "area": int(c.get("area") or c.get("area_pixels") or 0),
                "bbox_xyxy": list(c.get("bbox_xyxy") or []),
                "score": float(c.get("area") or c.get("area_pixels") or 0),
                "admitted": False,
                "hypothesis_only": True,
                "evidence_status": "geometry_hypothesis",
                # truth admission — all false
                "control_admitted": False,
                "goal_admitted": False,
                "action_admitted": False,
                "solve_admitted": False,
            }
        )
    return out


def _symmetry_score(components: list[dict[str, Any]], frame_shape: tuple[int, int]) -> float:
    """Simple LR mirror-pair geometry score as a soft conceptual prior."""
    if len(components) < 2:
        return 0.0
    h, w = frame_shape
    best = 0.0
    for i, a in enumerate(components):
        ax, ay = float(a["center_xy"][0]), float(a["center_xy"][1])
        for b in components[i + 1 :]:
            bx, by = float(b["center_xy"][0]), float(b["center_xy"][1])
            expected_bx = float(w - 1) - ax
            dist = float(np.hypot(expected_bx - bx, ay - by))
            score = max(0.0, 1.0 - dist / max(1.0, float(w)))
            if int(a.get("color") or 0) == int(b.get("color") or 0):
                score *= 1.05
            best = max(best, min(1.0, score))
    return float(best)


class ConceptualOp:
    """PortBand / component conceptual proposals (hypothesis-only)."""

    name = "conceptual"

    def __init__(self, *, max_components: int = 20, max_alignments: int = 24) -> None:
        self.max_components = int(max_components)
        self.max_alignments = int(max_alignments)

    def run(self, map_view: KnownMapView | Any, context: dict | None = None) -> OperatorResult:
        ctx = context or {}
        if isinstance(map_view, KnownMapView):
            comps = list(map_view.components)[: self.max_components]
            frame = np.asarray(map_view.frame)
        else:
            comps = list(ctx.get("components") or [])[: self.max_components]
            frame = np.asarray(getattr(map_view, "frame", ctx.get("frame")))
            if frame is None or getattr(frame, "size", 0) == 0:
                frame = np.zeros((64, 64), dtype=np.int16)

        if frame.ndim > 2:
            frame = frame[..., 0]
        h, w = int(frame.shape[0]), int(frame.shape[1])

        # Ensure we have components (prefer map_view; else legal_search / local)
        if not comps:
            try:
                from synthesis_lab.orchestration.rdl.brain.map_trust import (  # noqa: WPS433
                    extract_components,
                )

                comps, _, _src = extract_components(frame)
                comps = comps[: self.max_components]
            except Exception:
                comps = []

        ports_packet = _try_legal_ports(comps, (h, w))
        if ports_packet is None:
            ports_packet = _local_ports_and_alignments(
                comps, frame_shape=(h, w), max_alignments=self.max_alignments
            )
        else:
            # cap alignments
            al = list(ports_packet.get("alignments") or [])[: self.max_alignments]
            ports_packet["alignments"] = al
            ports_packet["n_alignments"] = len(al)
            if al:
                ports_packet["best_alignment_score"] = float(al[0]["score"])

        centroid_props = _centroid_proposals(comps)
        alignment_props: list[dict[str, Any]] = []
        for a in ports_packet.get("alignments") or []:
            alignment_props.append(
                {
                    "kind": "pairwise_alignment",
                    "a_port": a.get("a_port"),
                    "b_port": a.get("b_port"),
                    "a_component": a.get("a_component"),
                    "b_component": a.get("b_component"),
                    "a_side": a.get("a_side"),
                    "b_side": a.get("b_side"),
                    "score": float(a.get("score") or 0.0),
                    "admitted": False,
                    "hypothesis_only": True,
                    "evidence_status": "port_alignment_hypothesis",
                    "control_admitted": False,
                    "goal_admitted": False,
                    "action_admitted": False,
                    "solve_admitted": False,
                }
            )

        proposals = centroid_props + alignment_props
        sym = _symmetry_score(comps, (h, w))

        scores = {
            "n_components": len(comps),
            "n_ports": int(ports_packet.get("n_ports") or 0),
            "n_alignments": int(ports_packet.get("n_alignments") or 0),
            "best_alignment_score": float(ports_packet.get("best_alignment_score") or 0.0),
            "symmetry_score": sym,
            "n_centroid_proposals": len(centroid_props),
        }

        summary = {
            "schema": "megamind.conceptual.v0",
            "ports_source": ports_packet.get("source"),
            "ports": (ports_packet.get("ports") or [])[:32],
            "alignments": (ports_packet.get("alignments") or [])[:24],
            "centroids": [
                {
                    "component_id": c.get("component_id"),
                    "center_xy": list(c.get("center_xy") or []),
                    "color": int(c.get("color") or 0),
                    "area": int(c.get("area") or c.get("area_pixels") or 0),
                }
                for c in comps
            ],
            "symmetry_score": sym,
            "admitted": False,
            "hypothesis_only": True,
            "truth_flags": TruthFlags().as_dict(),
        }

        best = float(ports_packet.get("best_alignment_score") or 0.0)
        vectors = {
            "best_alignment": [best],
            "symmetry": [sym],
            "n_components": [float(len(comps))],
        }
        return OperatorResult(
            name=self.name,
            status="ok" if comps else "degraded",
            scores=scores,
            proposals=proposals,
            summary=summary,
            payload={
                "ports_source": ports_packet.get("source"),
                "n_ports": int(ports_packet.get("n_ports") or 0),
                "n_alignments": int(ports_packet.get("n_alignments") or 0),
                "centroids": summary.get("centroids"),
            },
            vectors=vectors,
            meta={
                "schema": "megamind.conceptual.v0",
                "ports_source": ports_packet.get("source"),
                "hypothesis_only": True,
                "admitted": False,
            },
            messages=["conceptual proposals ok" if comps else "no components"],
            truth_flags=TruthFlags(),
            hypothesis_only=True,
            hypothesis=True,
        )
