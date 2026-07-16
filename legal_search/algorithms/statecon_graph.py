"""StateCon branch graph + support groups — mined from statecon.py.

KEEP: macro/actor/target/support nodes and edges as hypotheses.
STRIP: engine sprite truth, admitted actor/goal.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from legal_search.algorithms.affordances import mine_affordance_candidates
from legal_search.algorithms.components import connected_components, enrich_components
from legal_search.algorithms.geometry_rel import mine_geo_relations
from legal_search.algorithms.local_symmetry import mine_symmetry


def _center(box: list[int]) -> list[float]:
    return [float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0]


def support_groups_from_components(components: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group wide/border components as support-kind hypotheses."""
    groups: list[dict[str, Any]] = []
    for i, c in enumerate(components):
        box = list(map(int, c["bbox_xyxy"]))
        bw = box[2] - box[0] + 1
        bh = box[3] - box[1] + 1
        area = int(c.get("area") or c.get("area_pixels") or 0)
        kind = "unknown_support"
        if bw >= 3 * max(1, bh):
            kind = "horizontal_band"
        elif bh >= 3 * max(1, bw):
            kind = "vertical_band"
        elif area >= 80:
            kind = "large_field_support"
        groups.append(
            {
                "support_group_id": f"sg:{c['component_id']}",
                "support_kind": kind,
                "support_ids": [f"s:{c['component_id']}"],
                "component_ids": [c["component_id"]],
                "group_bbox_xyxy": box,
                "group_center_xy": _center(box),
                "support_relation_category": "hypothesis",
                "admitted": False,
            }
        )
    return groups


def build_branch_graph(
    *,
    actor: dict[str, Any] | None,
    target: dict[str, Any] | None,
    support_groups: list[dict[str, Any]],
    group_box: list[int],
    statecon_id: str = "sc0",
) -> dict[str, Any]:
    """Mined from build_statecon_branch_graph — all nodes hypotheses."""
    nodes = [
        {
            "node_id": f"{statecon_id}:macro",
            "node_kind": "macro_statecon",
            "bbox_xyxy": group_box,
            "center_xy": _center(group_box),
            "admitted": False,
        }
    ]
    edges = []
    if actor:
        ab = list(map(int, actor.get("bbox_xyxy") or actor.get("group_bbox_xyxy") or [0, 0, 1, 1]))
        aid = f"{statecon_id}:actor"
        nodes.append(
            {
                "node_id": aid,
                "node_kind": "actor_affordance_candidate",
                "component_id": actor.get("component_id"),
                "bbox_xyxy": ab,
                "center_xy": _center(ab),
                "score": float(actor.get("score") or 0.0),
                "admitted": False,
            }
        )
        edges.append(
            {
                "edge_id": f"{statecon_id}:macro_to_actor",
                "edge_kind": "macro_to_actor",
                "from_node_id": f"{statecon_id}:macro",
                "to_node_id": aid,
            }
        )
    if target:
        tb = list(map(int, target.get("bbox_xyxy") or target.get("group_bbox_xyxy") or [0, 0, 1, 1]))
        tid = f"{statecon_id}:target"
        nodes.append(
            {
                "node_id": tid,
                "node_kind": "target_affordance_candidate",
                "component_id": target.get("component_id"),
                "bbox_xyxy": tb,
                "center_xy": _center(tb),
                "score": float(target.get("score") or 0.0),
                "admitted": False,
            }
        )
        edges.append(
            {
                "edge_id": f"{statecon_id}:macro_to_target",
                "edge_kind": "macro_to_target",
                "from_node_id": f"{statecon_id}:macro",
                "to_node_id": tid,
            }
        )
    for g in support_groups[:6]:
        sid = str(g["support_group_id"])
        nodes.append(
            {
                "node_id": sid,
                "node_kind": "statecon_support_group",
                "support_kind": g.get("support_kind"),
                "bbox_xyxy": g["group_bbox_xyxy"],
                "center_xy": g["group_center_xy"],
                "admitted": False,
            }
        )
        edges.append(
            {
                "edge_id": f"{statecon_id}:macro_to_{sid}",
                "edge_kind": "macro_to_support_group",
                "from_node_id": f"{statecon_id}:macro",
                "to_node_id": sid,
            }
        )
    return {
        "statecon_id": statecon_id,
        "nodes": nodes,
        "edges": edges,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "admitted": False,
        "not_engine_truth": True,
    }


def mine_statecon(frame: np.ndarray) -> dict[str, Any]:
    scene = enrich_components(frame)
    comps = scene["components"][:16]
    aff = mine_affordance_candidates(frame)
    geo = mine_geo_relations(comps)
    sym = mine_symmetry(frame, comps)
    supports = support_groups_from_components(comps)
    actor = None
    target = None
    if aff["actors"]:
        a = aff["actors"][0]
        match = next((c for c in comps if c["component_id"] == a["component_id"]), None)
        if match:
            actor = {**a, "bbox_xyxy": match["bbox_xyxy"], "group_bbox_xyxy": match["bbox_xyxy"]}
    if aff["targets"]:
        t = aff["targets"][0]
        match = next((c for c in comps if c["component_id"] == t["component_id"]), None)
        if match:
            target = {**t, "bbox_xyxy": match["bbox_xyxy"], "group_bbox_xyxy": match["bbox_xyxy"]}
    if comps:
        group_box = [
            min(c["bbox_xyxy"][0] for c in comps),
            min(c["bbox_xyxy"][1] for c in comps),
            max(c["bbox_xyxy"][2] for c in comps),
            max(c["bbox_xyxy"][3] for c in comps),
        ]
    else:
        group_box = [0, 0, 1, 1]
    graph = build_branch_graph(
        actor=actor,
        target=target,
        support_groups=supports,
        group_box=group_box,
    )
    return {
        "components": comps,
        "component_features": scene["component_features"][:12],
        "contacts": scene["contact_graph"],
        "spatial_nearest": geo["nearest"][:8],
        "n_components": scene["n_components"],
        "background": scene["background"],
        "support_groups": supports[:8],
        "n_support_groups": len(supports),
        "branch_graph": graph,
        "symmetry_sensors": {
            "global_h": sym["global_horizontal"],
            "global_v": sym["global_vertical"],
            "best_pair": sym["best_pair"],
        },
        "actor_candidate": actor,
        "target_candidate": target,
        "any_admitted": False,
        "not_engine_truth": True,
    }
