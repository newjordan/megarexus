"""Port / alignment geometry — mined from codexlab/contra_head/portband.py.

KEEP: side ports, span overlap, facing alignment scores.
STRIP: route-flow lock as solve, lineup matrix as action authority,
       SP80 glove/role names as truth.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SIDES = ("top", "bottom", "left", "right")
# Opposite faces for alignment candidates
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}


def port_row(component_id: str, color: int, bbox_xyxy: list[int], side: str) -> dict[str, Any]:
    """One bbox-side port — algorithm core of PortBand `_port_row`."""
    x0, y0, x1, y1 = [int(v) for v in bbox_xyxy]
    if side == "left":
        axis = "vertical_projection"
        center = [float(x0), float(y0 + y1) / 2.0]
        coord = float(x0)
        span = [int(y0), int(y1)]
        segment = [[x0, y0], [x0, y1]]
        edge_cells = [[x0, y] for y in range(y0, y1 + 1)]
    elif side == "right":
        axis = "vertical_projection"
        center = [float(x1), float(y0 + y1) / 2.0]
        coord = float(x1)
        span = [int(y0), int(y1)]
        segment = [[x1, y0], [x1, y1]]
        edge_cells = [[x1, y] for y in range(y0, y1 + 1)]
    elif side == "top":
        axis = "horizontal_projection"
        center = [float(x0 + x1) / 2.0, float(y0)]
        coord = float(y0)
        span = [int(x0), int(x1)]
        segment = [[x0, y0], [x1, y0]]
        edge_cells = [[x, y0] for x in range(x0, x1 + 1)]
    elif side == "bottom":
        axis = "horizontal_projection"
        center = [float(x0 + x1) / 2.0, float(y1)]
        coord = float(y1)
        span = [int(x0), int(x1)]
        segment = [[x0, y1], [x1, y1]]
        edge_cells = [[x, y1] for x in range(x0, x1 + 1)]
    else:
        raise ValueError(f"unknown side {side}")
    return {
        "port_id": f"{component_id}:{side}",
        "component_id": component_id,
        "color": int(color),
        "side": side,
        "axis": axis,
        "center_xy": center,
        "axis_coordinate_px": coord,
        "span_px": span,
        "span_len": int(span[1] - span[0] + 1),
        "segment_xy": segment,
        "edge_cell_count": len(edge_cells),
        "edge_cells_xy": edge_cells[:32],  # cap for payload size
        "bbox_xyxy": [x0, y0, x1, y1],
        "evidence_status": "static_geometry_hypothesis",
        "admitted": False,
    }


def object_ports(component: dict[str, Any]) -> list[dict[str, Any]]:
    box = component["bbox_xyxy"]
    cid = str(component["component_id"])
    color = int(component.get("color") or 0)
    return [port_row(cid, color, box, side) for side in SIDES]


def span_overlap_ratio(span_a: list[int], span_b: list[int]) -> float:
    """Mined from PortBand `_span_overlap_ratio`."""
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


def alignment_static_score(
    port_a: dict[str, Any],
    port_b: dict[str, Any],
    *,
    frame_diag: float = 90.0,
) -> float:
    """Facing-port alignment score — geometry only, not route solve truth.

    Mined idea from PortBand `_alignment_static_score` / face pairing:
    opposite sides, span overlap, gap distance.
    """
    if OPPOSITE.get(port_a["side"]) != port_b["side"]:
        return 0.0
    if port_a["component_id"] == port_b["component_id"]:
        return 0.0
    overlap = span_overlap_ratio(port_a["span_px"], port_b["span_px"])
    if overlap <= 0.0:
        return 0.0
    ax, ay = port_a["center_xy"]
    bx, by = port_b["center_xy"]
    # gap along primary axis between facing ports
    if port_a["side"] in ("left", "right"):
        gap = abs(ax - bx)
    else:
        gap = abs(ay - by)
    gap_term = 1.0 / (1.0 + gap / max(1.0, frame_diag * 0.25))
    size_sim = min(port_a["span_len"], port_b["span_len"]) / max(
        1, max(port_a["span_len"], port_b["span_len"])
    )
    # color difference is weak signal only (not role truth)
    color_term = 0.15 if int(port_a.get("color") or 0) != int(port_b.get("color") or 0) else 0.05
    score = 0.45 * overlap + 0.30 * gap_term + 0.15 * size_sim + color_term
    return float(max(0.0, min(1.0, score)))


def mine_ports_and_alignments(
    components: list[dict[str, Any]],
    *,
    frame_shape: tuple[int, int] | None = None,
    max_alignments: int = 24,
) -> dict[str, Any]:
    """Full legal PortBand mine: all side ports + ranked face alignments."""
    ports: list[dict[str, Any]] = []
    for c in components:
        ports.extend(object_ports(c))
    h = w = 64
    if frame_shape:
        h, w = int(frame_shape[0]), int(frame_shape[1])
    diag = float(np.hypot(h, w))
    alignments: list[dict[str, Any]] = []
    for i, a in enumerate(ports):
        for b in ports[i + 1 :]:
            score = alignment_static_score(a, b, frame_diag=diag)
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
                    "span_overlap": span_overlap_ratio(a["span_px"], b["span_px"]),
                    "admitted": False,
                }
            )
    alignments.sort(key=lambda r: -float(r["score"]))
    alignments = alignments[:max_alignments]
    by_side = {s: sum(1 for p in ports if p["side"] == s) for s in SIDES}
    return {
        "ports": ports,
        "alignments": alignments,
        "n_ports": len(ports),
        "n_alignments": len(alignments),
        "ports_by_side": by_side,
        "best_alignment_score": float(alignments[0]["score"]) if alignments else 0.0,
        "mean_port_span": float(np.mean([p["span_len"] for p in ports])) if ports else 0.0,
    }


def shape_terms_from_boxes(boxes: list[list[int]]) -> dict[str, float]:
    """Mined from PortBand `_bbox_shape_terms` / `_shape_fit_from_terms` inputs."""
    clean = [list(map(int, b)) for b in boxes if isinstance(b, (list, tuple)) and len(b) == 4]
    if not clean:
        return {"union_width_px": 0.0, "union_height_px": 0.0, "union_area_px": 0.0}
    x0 = min(b[0] for b in clean)
    y0 = min(b[1] for b in clean)
    x1 = max(b[2] for b in clean)
    y1 = max(b[3] for b in clean)
    w = float(x1 - x0 + 1)
    h = float(y1 - y0 + 1)
    return {"union_width_px": w, "union_height_px": h, "union_area_px": w * h}


def shape_fit_score(source_terms: dict[str, float], target_terms: dict[str, float]) -> float:
    """Mined from `_shape_fit_from_terms`."""
    sw = float(source_terms.get("union_width_px") or 0.0)
    sh = float(source_terms.get("union_height_px") or 0.0)
    sa = float(source_terms.get("union_area_px") or 0.0)
    tw = float(target_terms.get("union_width_px") or 0.0)
    th = float(target_terms.get("union_height_px") or 0.0)
    ta = float(target_terms.get("union_area_px") or 0.0)
    if min(sw, sh, sa, tw, th, ta) <= 0.0:
        return 0.0
    width_fit = min(sw, tw) / max(sw, tw)
    height_fit = min(sh, th) / max(sh, th)
    area_fit = min(sa, ta) / max(sa, ta)
    return float(max(0.0, min(1.0, 0.38 * width_fit + 0.38 * height_fit + 0.24 * area_fit)))


def axis_preservation_score(axis: str, transform_xy: list[float]) -> float:
    """Mined from `_axis_preservation_score`."""
    if not transform_xy or len(transform_xy) != 2:
        return 0.0
    dx = abs(float(transform_xy[0]))
    dy = abs(float(transform_xy[1]))
    denom = max(1.0, dx + dy)
    if axis == "vertical_projection":
        return float(max(0.0, min(1.0, dx / denom)))
    if axis == "horizontal_projection":
        return float(max(0.0, min(1.0, dy / denom)))
    return 0.5


def facing_transform(port_a: dict[str, Any], port_b: dict[str, Any]) -> list[float]:
    """Translation that would bring port_a center onto port_b center."""
    ax, ay = port_a["center_xy"]
    bx, by = port_b["center_xy"]
    return [float(bx - ax), float(by - ay)]


def alignment_error_px(port_a: dict[str, Any], port_b: dict[str, Any]) -> float:
    """Axis residual after facing: for vertical ports, y-span alignment residual."""
    if OPPOSITE.get(port_a["side"]) != port_b["side"]:
        return 999.0
    if port_a["side"] in ("left", "right"):
        # vertical ports: error is residual in y-span centers
        return abs(float(port_a["center_xy"][1]) - float(port_b["center_xy"][1]))
    return abs(float(port_a["center_xy"][0]) - float(port_b["center_xy"][0]))


def build_alignment_rows(ports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Full facing alignments with transforms — mined from `_alignment_row`."""
    rows: list[dict[str, Any]] = []
    for i, a in enumerate(ports):
        for b in ports[i + 1 :]:
            if OPPOSITE.get(a["side"]) != b["side"]:
                continue
            if a["component_id"] == b["component_id"]:
                continue
            overlap = span_overlap_ratio(a["span_px"], b["span_px"])
            if overlap <= 0.0:
                continue
            err = alignment_error_px(a, b)
            transform = facing_transform(a, b)
            exact = err <= 0.5
            near = err <= 2.0
            static = max(0.0, min(1.0, 1.0 - (err / 8.0)))
            if exact:
                static = max(static, 1.0)
            elif near:
                static = max(static, 0.75)
            static = min(1.0, static + 0.05 * overlap)
            axis = a["axis"]
            shape = shape_fit_score(
                shape_terms_from_boxes([a["bbox_xyxy"]]),
                shape_terms_from_boxes([b["bbox_xyxy"]]),
            )
            rows.append(
                {
                    "alignment_id": f"{a['port_id']}->{b['port_id']}",
                    "alignment_kind": "single_port_projection",
                    "axis": axis,
                    "source_port_id": a["port_id"],
                    "target_port_id": b["port_id"],
                    "source_object_ids": [a["component_id"]],
                    "target_object_ids": [b["component_id"]],
                    "source_port_ids": [a["port_id"]],
                    "target_port_ids": [b["port_id"]],
                    "alignment_error_px": err,
                    "transform_hypothesis_xy": transform,
                    "span_overlap_ratio": overlap,
                    "exact_alignment": exact,
                    "near_alignment": near,
                    "static_score": static,
                    "shape_fit_score": shape,
                    "axis_preservation_score": axis_preservation_score(axis, transform),
                    "admitted": False,
                    "evidence_status": "static_geometry_hypothesis",
                    "not_action_authority": True,
                }
            )
    rows.sort(key=lambda r: (-float(r["static_score"]), float(r["alignment_error_px"])))
    return rows


def edge_lock_quality(
    *,
    static_score: float,
    shape_fit_score: float,
    span_overlap_score: float,
    confluence_score: float,
    boundary_strength: float,
    axis_preservation: float,
    contradiction_penalty: float = 0.0,
) -> float:
    """Mined from `_edge_lock_quality_score` (fixed legal weights, no SP80 levers)."""
    weights = (0.28, 0.18, 0.18, 0.14, 0.12, 0.10)
    signals = (
        static_score,
        shape_fit_score,
        span_overlap_score,
        confluence_score,
        boundary_strength,
        axis_preservation,
    )
    score = sum(w * max(0.0, min(1.0, float(s))) for w, s in zip(weights, signals))
    score -= max(0.0, min(1.0, float(contradiction_penalty)))
    return float(max(0.0, min(1.0, score)))


def build_edge_lock_ledger(alignments: list[dict[str, Any]]) -> dict[str, Any]:
    """Edge-lock configurations as *hypotheses* — not solve truth.

    Mined structure from `_build_edge_lock_ledger` / `_edge_lock_config_from_alignment`.
    """
    rows: list[dict[str, Any]] = []
    for al in alignments:
        confluence = min(
            1.0,
            0.22
            + 0.18 * len(al.get("source_object_ids") or [])
            + 0.18 * len(al.get("target_object_ids") or [])
            + 0.05 * len(al.get("target_port_ids") or []),
        )
        boundary = float(al.get("span_overlap_ratio") or 0.0)
        quality = edge_lock_quality(
            static_score=float(al.get("static_score") or 0.0),
            shape_fit_score=float(al.get("shape_fit_score") or 0.0),
            span_overlap_score=float(al.get("span_overlap_ratio") or 0.0),
            confluence_score=confluence,
            boundary_strength=boundary,
            axis_preservation=float(al.get("axis_preservation_score") or 0.0),
        )
        if quality < 0.2:
            continue
        transform = al.get("transform_hypothesis_xy") or [0.0, 0.0]
        rows.append(
            {
                "edge_lock_configuration_id": f"lock:{al.get('alignment_id')}",
                "lock_kind": "port_alignment_edge_lock",
                "axis": al.get("axis"),
                "source_object_ids": list(al.get("source_object_ids") or []),
                "target_object_ids": list(al.get("target_object_ids") or []),
                "target_port_ids": list(al.get("target_port_ids") or []),
                "shared_transform_or_offset_xy": transform,
                "edge_alignment_error_px": al.get("alignment_error_px"),
                "edge_lock_quality_score": quality,
                "static_score": al.get("static_score"),
                "shape_fit_score": al.get("shape_fit_score"),
                "span_overlap_score": al.get("span_overlap_ratio"),
                "confluence_score": confluence,
                "admitted": False,
                "not_solve_truth": True,
                "not_action_authority": True,
                "evidence_status": "edge_lock_hypothesis",
            }
        )
    rows.sort(key=lambda r: -float(r["edge_lock_quality_score"]))
    for rank, row in enumerate(rows, start=1):
        row["edge_lock_rank"] = rank
    best_by_source: dict[str, dict[str, Any]] = {}
    for row in rows:
        for sid in row["source_object_ids"]:
            cur = best_by_source.get(sid)
            if cur is None or float(row["edge_lock_quality_score"]) > float(cur["edge_lock_quality_score"]):
                best_by_source[sid] = row
    return {
        "rows": rows[:32],
        "n_locks": len(rows),
        "best_quality": float(rows[0]["edge_lock_quality_score"]) if rows else 0.0,
        "best_by_source": [
            {
                "source_object_id": sid,
                "edge_lock_configuration_id": row["edge_lock_configuration_id"],
                "edge_lock_quality_score": row["edge_lock_quality_score"],
                "edge_lock_rank": row["edge_lock_rank"],
            }
            for sid, row in sorted(best_by_source.items())
        ],
        "admitted": False,
        "not_solve_truth": True,
    }


def shift_bbox(box: list[int], transform_xy: list[float]) -> list[int]:
    dx, dy = float(transform_xy[0]), float(transform_xy[1])
    return [
        int(round(box[0] + dx)),
        int(round(box[1] + dy)),
        int(round(box[2] + dx)),
        int(round(box[3] + dy)),
    ]


def build_lineup_matrix(
    components: list[dict[str, Any]],
    alignments: list[dict[str, Any]],
    edge_locks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Lineup matrix as *hypothesis rows* — never action authority.

    Mined from `_build_lineup_matrix`.
    """
    by_id = {c["component_id"]: c for c in components}
    rows: list[dict[str, Any]] = []
    for al in alignments:
        transform = al.get("transform_hypothesis_xy")
        if not isinstance(transform, list) or len(transform) != 2:
            continue
        score = float(al.get("static_score") or 0.0)
        # boost from edge-lock quality if linked
        if edge_locks:
            for lock in edge_locks.get("rows") or []:
                if set(lock.get("source_object_ids") or []) & set(al.get("source_object_ids") or []):
                    score = max(score, float(lock.get("edge_lock_quality_score") or 0.0))
        for sid in al.get("source_object_ids") or []:
            src = by_id.get(sid)
            if not src:
                continue
            box = list(map(int, src["bbox_xyxy"]))
            imagined = shift_bbox(box, transform)
            rows.append(
                {
                    "row_kind": "direct_port_alignment_lineup",
                    "source_object_id": sid,
                    "target_object_ids": list(al.get("target_object_ids") or []),
                    "target_port_ids": list(al.get("target_port_ids") or []),
                    "transform_xy": transform,
                    "source_bbox_xyxy": box,
                    "imagined_bbox_xyxy": imagined,
                    "lineup_score": score,
                    "confidence_score": score,
                    "linked_alignment_id": al.get("alignment_id"),
                    "admitted": False,
                    "not_action_authority": True,
                    "evidence_status": "lineup_hypothesis",
                }
            )
    rows.sort(key=lambda r: -float(r["lineup_score"]))
    return {
        "rows": rows[:32],
        "n_rows": len(rows),
        "best_lineup_score": float(rows[0]["lineup_score"]) if rows else 0.0,
        "admitted": False,
        "not_action_authority": True,
    }


def mine_ports_full(
    components: list[dict[str, Any]],
    *,
    frame_shape: tuple[int, int] | None = None,
    max_alignments: int = 32,
) -> dict[str, Any]:
    """Complete legal PortBand mine: ports + alignments + edge locks + lineup."""
    base = mine_ports_and_alignments(components, frame_shape=frame_shape, max_alignments=max_alignments)
    ports = base["ports"]
    # replace simple alignments with full transform alignments
    alignments = build_alignment_rows(ports)[:max_alignments]
    locks = build_edge_lock_ledger(alignments)
    lineup = build_lineup_matrix(components, alignments, locks)
    return {
        **base,
        "alignments": alignments,
        "n_alignments": len(alignments),
        "best_alignment_score": float(alignments[0]["static_score"]) if alignments else 0.0,
        "edge_lock_ledger": locks,
        "lineup_matrix": lineup,
    }


def union_bbox(boxes: list[list[int]]) -> list[int]:
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def virtual_combo_edge(axis: str, side: str, union_box: list[int]) -> dict[str, Any]:
    """Mined from `_virtual_combo_edge` — synthetic port on combo union bbox."""
    x0, y0, x1, y1 = [int(v) for v in union_box]
    if side == "left":
        return {
            "side": side,
            "axis": "vertical_projection",
            "center_xy": [float(x0), float(y0 + y1) / 2.0],
            "axis_coordinate_px": float(x0),
            "span_px": [y0, y1],
            "bbox_xyxy": [x0, y0, x1, y1],
        }
    if side == "right":
        return {
            "side": side,
            "axis": "vertical_projection",
            "center_xy": [float(x1), float(y0 + y1) / 2.0],
            "axis_coordinate_px": float(x1),
            "span_px": [y0, y1],
            "bbox_xyxy": [x0, y0, x1, y1],
        }
    if side == "top":
        return {
            "side": side,
            "axis": "horizontal_projection",
            "center_xy": [float(x0 + x1) / 2.0, float(y0)],
            "axis_coordinate_px": float(y0),
            "span_px": [x0, x1],
            "bbox_xyxy": [x0, y0, x1, y1],
        }
    return {
        "side": "bottom",
        "axis": "horizontal_projection",
        "center_xy": [float(x0 + x1) / 2.0, float(y1)],
        "axis_coordinate_px": float(y1),
        "span_px": [x0, x1],
        "bbox_xyxy": [x0, y0, x1, y1],
    }


def build_fleet_combo_candidates(
    components: list[dict[str, Any]],
    alignments: list[dict[str, Any]],
    *,
    max_combo_size: int = 3,
    limit: int = 24,
) -> dict[str, Any]:
    """PortBand fleet combo shape-lock candidates — hypotheses only.

    Mined structure from `_build_portband_fleet` without route-flow solve truth.
    """
    from itertools import combinations

    comps = components[:8]
    candidates: list[dict[str, Any]] = []
    expansions = 0
    max_size = min(max_combo_size, len(comps))
    # boundary rows from alignments with target ports
    boundaries = []
    for al in alignments:
        if al.get("axis") and al.get("target_port_ids"):
            boundaries.append(
                {
                    "axis": al["axis"],
                    "target_port_ids": list(al.get("target_port_ids") or []),
                    "source_object_ids": list(al.get("source_object_ids") or []),
                    "boundary_strength": float(al.get("static_score") or 0.0),
                    "anchor_centers": [
                        # approximate from transform
                        al.get("transform_hypothesis_xy") or [0.0, 0.0]
                    ],
                    "alignment_id": al.get("alignment_id"),
                }
            )
    for size in range(2, max_size + 1):
        for combo in combinations(comps, size):
            boxes = [list(map(int, c["bbox_xyxy"])) for c in combo]
            expansions += 1
            ubox = union_bbox(boxes)
            area_sum = sum(int(c.get("area") or c.get("area_pixels") or 0) for c in combo)
            fill = float(area_sum) / float(max(1, (ubox[2] - ubox[0] + 1) * (ubox[3] - ubox[1] + 1)))
            for boundary in boundaries[:8]:
                axis = str(boundary.get("axis") or "")
                if axis == "vertical_projection":
                    sides = ("left", "right")
                elif axis == "horizontal_projection":
                    sides = ("top", "bottom")
                else:
                    continue
                src_a = virtual_combo_edge(axis, sides[0], ubox)
                src_b = virtual_combo_edge(axis, sides[1], ubox)
                source_sep = abs(float(src_b["axis_coordinate_px"]) - float(src_a["axis_coordinate_px"]))
                if source_sep <= 0:
                    continue
                # target sep from alignment transform magnitude as proxy
                txy = boundary.get("anchor_centers") or [[0.0, 0.0]]
                target_sep = max(1.0, abs(float(txy[0][0])) + abs(float(txy[0][1])))
                error = abs(source_sep - target_sep)
                fit_score = max(0.0, min(1.0, 1.0 - (error / max(1.0, target_sep))))
                span_score = 0.5  # combo virtual edges full span
                bstr = float(boundary.get("boundary_strength") or 0.0)
                rank = max(
                    0.0,
                    min(1.0, 0.58 * fit_score + 0.16 * span_score + 0.12 * fill + 0.14 * bstr),
                )
                if rank < 0.2:
                    continue
                candidates.append(
                    {
                        "candidate_id": f"fleet:{size}:{'-'.join(c['component_id'] for c in combo)}:{boundary.get('alignment_id')}",
                        "lock_kind": "fleet_combo_shape_lock",
                        "combo_size": size,
                        "source_object_ids": [c["component_id"] for c in combo],
                        "union_bbox_xyxy": ubox,
                        "axis": axis,
                        "fit_score": fit_score,
                        "fill_ratio": fill,
                        "boundary_strength": bstr,
                        "rank_score": rank,
                        "admitted": False,
                        "not_solve_truth": True,
                        "not_action_authority": True,
                    }
                )
    candidates.sort(key=lambda r: -float(r["rank_score"]))
    candidates = candidates[:limit]
    return {
        "ranked_candidates": candidates,
        "n_candidates": len(candidates),
        "total_combo_expansions": expansions,
        "best_rank_score": float(candidates[0]["rank_score"]) if candidates else 0.0,
        "admitted": False,
        "not_solve_truth": True,
    }


def event_surface_from_transitions(
    transitions: list[Any] | None,
    components: list[dict[str, Any]],
    alignments: list[dict[str, Any]],
    *,
    limit: int = 12,
) -> dict[str, Any]:
    """Correlate change regions with port alignments — event-surface hypothesis.

    Mined idea from PortBand event-surface / action-state correlation without
    treating route-flow as solve truth.
    """
    from legal_search.algorithms.components import change_regions

    trs = list(transitions or [])[-16:]
    events: list[dict[str, Any]] = []
    for tr in trs:
        if not getattr(tr, "frame_changed", False):
            continue
        regions = change_regions(tr.frame_before, tr.frame_after, max_regions=3)
        for reg in regions.get("regions") or []:
            box = reg["bbox_xyxy"]
            # overlap with component bboxes
            overlaps = []
            for c in components:
                cb = c["bbox_xyxy"]
                ix0 = max(box[0], cb[0])
                iy0 = max(box[1], cb[1])
                ix1 = min(box[2], cb[2])
                iy1 = min(box[3], cb[3])
                if ix0 <= ix1 and iy0 <= iy1:
                    inter = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
                    area = max(1, (cb[2] - cb[0] + 1) * (cb[3] - cb[1] + 1))
                    overlaps.append({"component_id": c["component_id"], "overlap_ratio": inter / area})
            overlaps.sort(key=lambda r: -r["overlap_ratio"])
            conf = min(1.0, float(reg["area"]) / 64.0)
            dynamic = max(0.0, min(1.0, 0.5 * conf + 0.3 * (overlaps[0]["overlap_ratio"] if overlaps else 0.0)))
            # alignment correlation
            matched_align = None
            for al in alignments[:12]:
                src = set(al.get("source_object_ids") or [])
                tgt = set(al.get("target_object_ids") or [])
                hit = {o["component_id"] for o in overlaps[:2]}
                if hit & (src | tgt):
                    matched_align = al.get("alignment_id")
                    dynamic = min(1.0, dynamic + 0.15)
                    break
            events.append(
                {
                    "event_id": f"evt:{getattr(tr, 'step', 0)}:{reg['region_id']}",
                    "action": tr.action.to_dict() if hasattr(tr.action, "to_dict") else {},
                    "surface": {"bbox_xyxy": box, "area": reg["area"], "center_xy": reg["center_xy"]},
                    "object_overlaps": overlaps[:4],
                    "matched_alignment_id": matched_align,
                    "dynamic_score": dynamic,
                    "confidence": conf,
                    "evidence_status": "observed_action_surface_result_delta",
                    "admitted": False,
                    "not_solve_truth": True,
                    "is_controlled_route_flow": False,  # never claim route-flow solve
                }
            )
    events.sort(key=lambda e: -float(e["dynamic_score"]))
    events = events[:limit]
    return {
        "events": events,
        "n_events": len(events),
        "best_dynamic_score": float(events[0]["dynamic_score"]) if events else 0.0,
        "n_with_alignment": sum(1 for e in events if e.get("matched_alignment_id")),
        "admitted": False,
    }


def mine_ports_full(
    components: list[dict[str, Any]],
    *,
    frame_shape: tuple[int, int] | None = None,
    max_alignments: int = 32,
    transitions: list[Any] | None = None,
) -> dict[str, Any]:
    """Complete legal PortBand mine including fleet + event surfaces."""
    base = mine_ports_and_alignments(components, frame_shape=frame_shape, max_alignments=max_alignments)
    ports = base["ports"]
    alignments = build_alignment_rows(ports)[:max_alignments]
    locks = build_edge_lock_ledger(alignments)
    lineup = build_lineup_matrix(components, alignments, locks)
    fleet = build_fleet_combo_candidates(components, alignments)
    events = event_surface_from_transitions(transitions, components, alignments)
    return {
        **base,
        "alignments": alignments,
        "n_alignments": len(alignments),
        "best_alignment_score": float(alignments[0]["static_score"]) if alignments else 0.0,
        "edge_lock_ledger": locks,
        "lineup_matrix": lineup,
        "fleet": fleet,
        "event_surfaces": events,
    }
