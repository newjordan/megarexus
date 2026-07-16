"""Full-mine encoder: every registered system → fixed slots → LoRA fuse vector.

Schema v2: mining is always-on. Each consume system owns a dedicated slice.
Frame-global cues (hist/edge/sym/comps) prepend the system block.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from legal_search.algorithms.components import connected_components, dominant_background
from legal_search.algorithms.saliency import (
    color_histogram,
    edge_map,
    symmetry_score_horizontal,
    symmetry_score_vertical,
)
from legal_search.consume.registry import CONSUME_REGISTRY, consume_all
from legal_search.types import Transition

FEATURE_SCHEMA = "legal_search.features.v2_fullmine"

# --- global frame block ---
HIST = 16
EDGE = 1
SYM = 2
COMP_N = 8
COMP_EACH = 5
COMP = COMP_N * COMP_EACH  # 40
GLOBAL_DIM = HIST + EDGE + SYM + COMP  # 59

# --- per-system slot width (dense enough to carry hypotheses + proposals) ---
SYS_SLOT = 12

# Stable order = registry insertion order (must match consume/registry.py)
SYSTEM_ORDER: tuple[str, ...] = tuple(CONSUME_REGISTRY.keys())
N_SYSTEMS = len(SYSTEM_ORDER)
SYSTEMS_DIM = N_SYSTEMS * SYS_SLOT

# availability one-hot + proposal mass + legal guards
META_DIM = N_SYSTEMS + 8  # per-system present flag + global meta

FEATURE_DIM = GLOBAL_DIM + SYSTEMS_DIM + META_DIM

FEATURE_LAYOUT = {
    "schema": FEATURE_SCHEMA,
    "global_dim": GLOBAL_DIM,
    "sys_slot": SYS_SLOT,
    "n_systems": N_SYSTEMS,
    "systems": list(SYSTEM_ORDER),
    "systems_dim": SYSTEMS_DIM,
    "meta_dim": META_DIM,
    "feature_dim": FEATURE_DIM,
}


def _safe(d: Any) -> dict[str, Any]:
    return d if isinstance(d, dict) else {}


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _hyp_payloads(packet: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for h in packet.get("hypotheses") or []:
        if isinstance(h, dict):
            out.append(_safe(h.get("payload")))
    return out


def _proposals(packet: dict[str, Any]) -> list[dict[str, Any]]:
    return [p for p in (packet.get("proposals") or []) if isinstance(p, dict)]


def _encode_global(frame: np.ndarray) -> np.ndarray:
    grid = np.asarray(frame)
    h, w = grid.shape[:2]
    vec = np.zeros(GLOBAL_DIM, dtype=np.float32)
    o = 0
    vec[o : o + HIST] = color_histogram(grid, 16).astype(np.float32)
    o += HIST
    vec[o] = float(edge_map(grid).mean())
    o += EDGE
    vec[o] = symmetry_score_horizontal(grid)
    vec[o + 1] = symmetry_score_vertical(grid)
    o += SYM
    comps = connected_components(grid)[:COMP_N]
    for i in range(COMP_N):
        base = o + i * COMP_EACH
        if i >= len(comps):
            continue
        c = comps[i]
        x0, y0, x1, y1 = c["bbox_xyxy"]
        vec[base + 0] = c["color"] / 15.0
        vec[base + 1] = min(1.0, c["area"] / float(h * w))
        vec[base + 2] = c["center_xy"][0] / max(1.0, w - 1)
        vec[base + 3] = c["center_xy"][1] / max(1.0, h - 1)
        vec[base + 4] = (x1 - x0 + 1) / max(1.0, (y1 - y0 + 1))
    return vec


def _slot_base() -> np.ndarray:
    return np.zeros(SYS_SLOT, dtype=np.float32)


def _fill_from_proposals(slot: np.ndarray, props: list[dict[str, Any]], start: int = 0) -> None:
    if not props:
        return
    scores = [float(p.get("score") or 0.0) for p in props]
    slot[start + 0] = _clip01((max(scores) + 1.0) / 2.0) if scores else 0.0
    slot[start + 1] = _clip01(float(np.mean(scores))) if scores else 0.0
    slot[start + 2] = min(1.0, len(props) / 8.0)
    # first action id mass
    aids = []
    for p in props[:6]:
        acts = p.get("actions") or []
        if acts and isinstance(acts[0], dict):
            aids.append(int(acts[0].get("action_id") or 0))
    for j, aid in enumerate(range(1, 7)):
        if start + 3 + j < SYS_SLOT:
            slot[start + 3 + j] = aids.count(aid) / max(1, len(aids)) if aids else 0.0


def encode_system_slot(system_id: str, packet: dict[str, Any], frame: np.ndarray) -> np.ndarray:
    """Mine one system's LegalPacket into SYS_SLOT floats."""
    slot = _slot_base()
    packet = _safe(packet)
    hyps = _hyp_payloads(packet)
    props = _proposals(packet)
    receipt = _safe(packet.get("receipt"))
    slot[0] = 1.0 if packet else 0.0  # present
    slot[1] = 1.0 if receipt.get("legal", True) else 0.0
    slot[2] = min(1.0, len(hyps) / 8.0)
    slot[3] = min(1.0, len(props) / 8.0)

    # system-specific mining
    if system_id == "stochastic_goose":
        for p in hyps:
            stats = _safe(p.get("action_stats"))
            rates = _safe(p.get("rates"))
            for i, aid in enumerate((1, 2, 3, 4, 5, 6)):
                if str(aid) in rates:
                    slot[4 + i] = _clip01(float(rates[str(aid)]))
                else:
                    row = stats.get(str(aid))
                    if isinstance(row, dict) and float(row.get("trials") or 0) > 0:
                        slot[4 + i] = float(row["changes"]) / float(row["trials"])
            if p.get("n_click_bins") is not None:
                slot[10] = min(1.0, float(p.get("n_click_bins") or 0) / 16.0)
            if p.get("best_click_rate") is not None:
                slot[11] = _clip01(float(p.get("best_click_rate") or 0.0))
        if props and slot[10] == 0.0 and slot[11] == 0.0:
            slot[10] = _clip01(float(props[0].get("score") or 0.5))
            slot[11] = 1.0

    elif system_id == "dreamweaver_sim3d":
        _fill_from_proposals(slot, props, start=4)
        slot[4] = min(1.0, len(props) / 4.0)
        if props:
            best = max(props, key=lambda p: float(p.get("score") or 0.0))
            slot[5] = float(np.tanh(float(best.get("score") or 0.0)))
            slot[6] = min(1.0, len(best.get("actions") or []) / 24.0)
            slot[7] = 1.0 if (best.get("kind") or "").startswith("program") else 0.5
        for p in hyps:
            if "n_proposals" in p:
                slot[8] = min(1.0, float(p.get("n_proposals") or 0) / 4.0)
            stripped = p.get("stripped") or []
            if isinstance(stripped, list):
                slot[9] = min(1.0, len(stripped) / 8.0)
            # legal fence: fixtures must stay unloaded
            slot[10] = 0.0 if p.get("fixture_priors_loaded") else 1.0
            slot[11] = 0.0 if p.get("shipped_dynamics_json_loaded") else 1.0

    elif system_id in ("contra_head", "contra_vision", "statecon"):
        comps: list = []
        n_contacts = 0
        for p in hyps:
            if isinstance(p.get("components"), list):
                comps = p["components"]
            if isinstance(p.get("n_components"), (int, float)):
                slot[8] = min(1.0, float(p["n_components"]) / 24.0)
            cg = p.get("contact_graph") or p.get("contacts")
            if isinstance(cg, dict):
                n_contacts = int(cg.get("contact_pair_count") or len(cg.get("contacts") or []))
            if "mean_edge" in p:
                slot[4] = float(p["mean_edge"])
            if "score" in p and "mean_edge" not in p and "horizontal" not in p:
                slot[5] = float(p.get("score") or 0.0)
            if "hist" in p and isinstance(p["hist"], list):
                h = np.asarray(p["hist"][:4], dtype=np.float32)
                slot[6 : 6 + min(2, len(h))] = h[:2]
            feats = p.get("component_features") or p.get("features") or []
            if isinstance(feats, list) and feats and isinstance(feats[0], dict):
                slot[6] = _clip01(float(feats[0].get("bbox_fill_ratio") or 0.0))
                slot[7] = min(1.0, float(feats[0].get("neighbor_component_count") or 0) / 6.0)
        if not slot[8]:
            slot[8] = min(1.0, len(comps) / 24.0)
        slot[9] = min(1.0, n_contacts / 12.0)
        if comps and isinstance(comps[0], dict):
            slot[10] = float(comps[0].get("area") or 0) / 4096.0
            slot[11] = float(comps[0].get("color") or 0) / 15.0

    elif system_id == "portband":
        n_ports = n_align = 0
        for p in hyps:
            if "n_ports" in p:
                n_ports = int(p.get("n_ports") or 0)
            elif "ports" in p:
                n_ports = len(p.get("ports") or [])
            if "n_alignments" in p:
                n_align = int(p.get("n_alignments") or 0)
            if "alignments" in p:
                al = p.get("alignments") or []
                n_align = max(n_align, len(al))
                if al:
                    slot[6] = float(al[0].get("static_score") or al[0].get("score") or 0.0)
            if "best_alignment_score" in p:
                slot[6] = float(p.get("best_alignment_score") or 0.0)
            if "mean_port_span" in p:
                slot[8] = min(1.0, float(p.get("mean_port_span") or 0) / 64.0)
            by_side = p.get("ports_by_side") or {}
            if isinstance(by_side, dict) and by_side:
                slot[9] = min(1.0, float(by_side.get("left", 0) + by_side.get("right", 0)) / 16.0)
                slot[10] = min(1.0, float(by_side.get("top", 0) + by_side.get("bottom", 0)) / 16.0)
            # edge-lock / lineup / fleet / events
            if "n_locks" in p or "best_quality" in p:
                slot[11] = _clip01(float(p.get("best_quality") or 0.0))
            if "best_lineup_score" in p:
                slot[7] = _clip01(float(p.get("best_lineup_score") or 0.0))
            if "best_rank_score" in p:  # fleet
                slot[11] = max(slot[11], _clip01(float(p.get("best_rank_score") or 0.0)))
            if "n_candidates" in p:
                slot[10] = max(slot[10], min(1.0, float(p.get("n_candidates") or 0) / 16.0))
            if "best_dynamic_score" in p:
                slot[9] = max(slot[9], _clip01(float(p.get("best_dynamic_score") or 0.0)))
            if "n_events" in p:
                slot[8] = max(slot[8], min(1.0, float(p.get("n_events") or 0) / 12.0))
            if "rows" in p and p.get("not_solve_truth"):
                slot[11] = max(slot[11], _clip01(float(p.get("best_quality") or 0.0)))
        slot[4] = min(1.0, n_ports / 48.0)  # 12 comps × 4 sides
        slot[5] = min(1.0, n_align / 16.0)
        if slot[7] == 0.0:
            slot[7] = float(dominant_background(frame)) / 15.0

    elif system_id == "oracle":
        for p in hyps:
            if "focus_center_xy" in p:
                cxy = p["focus_center_xy"]
                slot[4] = float(cxy[0]) / 63.0
                slot[5] = float(cxy[1]) / 63.0
                slot[6] = 1.0  # has assumption
                slot[7] = 0.3  # low confidence review-only

    elif system_id in ("ghost", "episode_memory"):
        for p in hyps:
            if "n_transitions" in p:
                slot[4] = min(1.0, float(p.get("n_transitions") or 0) / 256.0)
                slot[5] = min(1.0, float(p.get("n_successful_programs") or 0) / 32.0)
                slot[6] = min(1.0, float(p.get("n_relation_hints") or 0) / 64.0)
                slot[7] = 0.0 if p.get("cross_game_memory_admitted") else 1.0
            if "change_fraction" in p:
                slot[8] = _clip01(float(p.get("change_fraction") or 0.0))
            if "n_changed" in p:
                slot[9] = min(1.0, float(p.get("n_changed") or 0) / 64.0)
            rates = p.get("aid_change_rates") or {}
            if isinstance(rates, dict) and rates:
                vals = sorted((float(v) for v in rates.values()), reverse=True)
                slot[10] = _clip01(vals[0]) if vals else 0.0
                slot[11] = _clip01(float(np.mean(vals))) if vals else 0.0
        if props and slot[4] == 0.0:
            _fill_from_proposals(slot, props, start=4)

    elif system_id == "geoband":
        for p in hyps:
            pairs = p.get("pairs") or []
            slot[4] = min(1.0, float(p.get("n_pairs") or len(pairs)) / 12.0)
            if "min_manhattan" in p:
                slot[5] = min(1.0, float(p.get("min_manhattan") or 0) / 128.0)
            elif pairs:
                # support both old {manhattan} and new {distance_manhattan}
                m0 = pairs[0].get("distance_manhattan", pairs[0].get("manhattan") or 0)
                slot[5] = min(1.0, float(m0) / 128.0)
            if "mean_manhattan" in p:
                slot[6] = min(1.0, float(p.get("mean_manhattan") or 0) / 128.0)
            elif len(pairs) > 1:
                m1 = pairs[1].get("distance_manhattan", pairs[1].get("manhattan") or 0)
                slot[6] = min(1.0, float(m1) / 128.0)
            nearest = p.get("nearest") or []
            slot[7] = min(1.0, len(nearest) / 8.0)
            if pairs:
                bearings = [float(x.get("bearing_degrees") or 0) for x in pairs[:6]]
                if bearings:
                    slot[8] = float(np.mean(bearings)) / 360.0
                    gaps = [float(x.get("gap") or 0) for x in pairs[:6]]
                    slot[9] = min(1.0, float(np.mean(gaps)) / 64.0) if gaps else 0.0
            slot[10] = min(1.0, float(p.get("n_components") or 0) / 12.0)

    elif system_id == "deltaband":
        for p in hyps:
            rows = p.get("rows") or []
            slot[4] = min(1.0, len(rows) / 16.0)
            if rows:
                dpx = [float(r.get("delta_pixels") or 0) for r in rows]
                slot[5] = min(1.0, float(np.mean(dpx)) / 512.0)
                slot[6] = min(1.0, float(np.max(dpx)) / 1024.0)
                ld = [float(r.get("levels_delta") or 0) for r in rows]
                slot[7] = _clip01(float(np.mean(ld)) + 0.5)
            if "change_rate" in p:
                slot[8] = _clip01(float(p.get("change_rate") or 0.0))
            if "mean_delta" in p:
                slot[9] = min(1.0, float(p.get("mean_delta") or 0.0) / 512.0)
            mba = p.get("mean_delta_by_action") or {}
            if isinstance(mba, dict) and mba:
                # top two action mean deltas
                ranked = sorted((float(v) for v in mba.values()), reverse=True)
                slot[10] = min(1.0, ranked[0] / 512.0) if ranked else 0.0
                slot[11] = min(1.0, ranked[1] / 512.0) if len(ranked) > 1 else 0.0

    elif system_id == "exploreband":
        _fill_from_proposals(slot, props, start=4)
        slot[4] = min(1.0, len(props) / 8.0)
        if props:
            a0 = (props[0].get("actions") or [{}])[0]
            if isinstance(a0, dict) and a0.get("action_id") == 6:
                slot[5] = float(a0.get("x") or 0) / 63.0
                slot[6] = float(a0.get("y") or 0) / 63.0

    elif system_id == "actionmodband":
        for p in hyps:
            rates = p.get("rates") or {}
            if isinstance(rates, dict) and rates:
                vals = sorted(float(v) for v in rates.values())
                slot[4] = vals[-1] if vals else 0.0
                slot[5] = float(np.mean(vals)) if vals else 0.0
                slot[6] = min(1.0, len(rates) / 8.0)
            cells = p.get("cells") or []
            if cells:
                slot[7] = float(cells[0].get("change_rate") or 0.0)
                slot[8] = float(cells[0].get("x") or 0) / 63.0
                slot[9] = float(cells[0].get("y") or 0) / 63.0

    elif system_id == "loom_lewm_rexus":
        _fill_from_proposals(slot, props, start=4)
        by = {i: 0.0 for i in range(1, 7)}
        for p in props:
            acts = p.get("actions") or []
            if acts and isinstance(acts[0], dict):
                aid = int(acts[0].get("action_id") or 0)
                if aid in by:
                    by[aid] = float(p.get("score") or 0.0)
        for i, aid in enumerate(range(1, 7)):
            slot[4 + i] = by[aid]
        for p in hyps:
            rates = p.get("affordance_rates") or {}
            if isinstance(rates, dict):
                for i, aid in enumerate(range(1, 7)):
                    if str(aid) in rates:
                        slot[4 + i] = _clip01(float(rates[str(aid)]))
            if "multi_step_consistency" in p:
                slot[10] = _clip01(float(p.get("multi_step_consistency") or 0.0))
            cem = p.get("cem") or {}
            if isinstance(cem, dict) and cem.get("best_score") is not None:
                slot[10] = max(slot[10], min(1.0, float(cem.get("best_score") or 0.0) / 64.0))
            slot[11] = 1.0 if p.get("runtime_admitted") else 0.0
        if slot[10] == 0.0 and hyps:
            slot[10] = 1.0

    elif system_id == "symmetry":
        for p in hyps:
            slot[4] = float(p.get("horizontal") or p.get("global_horizontal") or 0.0)
            slot[5] = float(p.get("vertical") or p.get("global_vertical") or 0.0)
            if "best_self" in p:
                slot[6] = _clip01(float(p.get("best_self") or 0.0))
            if "best_pair" in p:
                slot[7] = _clip01(float(p.get("best_pair") or 0.0))
            if "n_pairs" in p:
                slot[8] = min(1.0, float(p.get("n_pairs") or 0) / 8.0)
            slot[9] = 1.0 if p.get("not_goal_key") else 0.0

    elif system_id == "crucible":
        for p in hyps:
            mov = p.get("movable") or {}
            sta = p.get("static") or {}
            if isinstance(mov, dict) and mov:
                slot[4] = max(float(v) for v in mov.values())
                slot[5] = min(1.0, len(mov) / 8.0)
            if isinstance(sta, dict) and sta:
                slot[6] = max(float(v) for v in sta.values())
                slot[7] = min(1.0, len(sta) / 8.0)
            if "n_transitions" in p:
                slot[8] = min(1.0, float(p.get("n_transitions") or 0) / 64.0)
            if "change_fraction" in p:
                slot[9] = _clip01(float(p.get("change_fraction") or 0.0))
            if p.get("top_movable") is not None:
                slot[10] = float(int(p["top_movable"])) / 15.0
            if "n_rules" in p:
                slot[11] = min(1.0, float(p.get("n_rules") or 0) / 8.0)
            elif "v0_match_fraction" in p:
                slot[11] = _clip01(float(p.get("v0_match_fraction") or 0.0))
            elif "n_move_actions" in p:
                slot[11] = min(1.0, float(p.get("n_move_actions") or 0) / 32.0)

    elif system_id == "megarexus_cnn":
        slot[4] = 1.0  # architecture available as optional FA
        slot[5] = 0.0  # not admitted as general agent
        slot[6] = 0.5  # optional online role

    elif system_id == "specialist_solvers":
        slot[4] = 0.0  # NOT included into legal loop
        slot[5] = 1.0  # research-only flag
        for p in hyps:
            slot[6] = 0.0 if p.get("included_in_legal_loop") else 1.0

    elif system_id == "pufferlib_rl":
        slot[4] = 0.0  # optional, not admitted
        slot[5] = 1.0  # available as trainer idea

    elif system_id == "quarantine_fence":
        slot[4] = 1.0  # fence active
        slot[5] = 0.0  # agent_may_import = false
        for p in hyps:
            banned = p.get("banned") or []
            slot[6] = min(1.0, len(banned) / 8.0)

    elif system_id == "tabula_rasa":
        slot[4] = 1.0  # wipe available
        for p in hyps:
            slot[5] = 1.0 if p.get("wipe_game_bindings") else 0.0

    elif system_id == "relationship_ladder":
        for p in hyps:
            rungs = p.get("rungs") or []
            slot[4] = min(1.0, len(rungs) / 8.0)
            if rungs:
                slot[5] = float(rungs[0].get("level") or 0) / 8.0
            if "completeness" in p:
                slot[6] = _clip01(float(p.get("completeness") or 0.0))
            stages = p.get("stages") or []
            for i, st in enumerate(stages[:4]):
                if 7 + i < SYS_SLOT:
                    slot[7 + i] = _clip01(float(st.get("score") or 0.0))

    elif system_id == "match_calibration":
        slot[4] = 1.0  # may reweight
        slot[5] = 0.0  # may not admit truth
        for p in hyps:
            slot[4] = 1.0 if p.get("may_reweight") else 0.0
            slot[5] = 1.0 if p.get("may_admit_truth") else 0.0
            actors = p.get("actors") or []
            targets = p.get("targets") or []
            if actors:
                slot[6] = _clip01(float(actors[0].get("score") or 0.0))
            if targets:
                slot[7] = _clip01(float(targets[0].get("score") or 0.0))
            if "prior_shift" in p:
                slot[8] = min(1.0, float(p.get("prior_shift") or 0.0))
            if "n_transitions" in p:
                slot[9] = min(1.0, float(p.get("n_transitions") or 0) / 32.0)
            s0 = p.get("stage0") or {}
            if isinstance(s0, dict) and s0.get("best_score") is not None:
                slot[6] = max(slot[6], _clip01(float(s0.get("best_score") or 0.0)))
            s1 = p.get("stage1") or {}
            if isinstance(s1, dict) and "change_rate" in s1:
                slot[7] = max(slot[7], _clip01(float(s1.get("change_rate") or 0.0)))
            slot[10] = 0.0 if p.get("admitted_actor") else 1.0
            slot[11] = 0.0 if p.get("admitted_goal") else 1.0

    elif system_id == "controller_affordances":
        for p in hyps:
            cands = p.get("candidates") or []
            slot[4] = min(1.0, len(cands) / 8.0)
            actors = p.get("actors") or []
            targets = p.get("targets") or []
            supports = p.get("supports") or []
            if actors:
                slot[5] = _clip01(float(actors[0].get("score") or 0.0))
            elif cands:
                slot[5] = 1.0
            if targets:
                slot[6] = _clip01(float(targets[0].get("score") or 0.0))
            slot[7] = 0.0  # none admitted
            if supports:
                slot[8] = _clip01(float(supports[0].get("score") or 0.0))
            slot[9] = 1.0 if p.get("may_admit_truth") is False else 0.0
            slot[10] = min(1.0, float(p.get("n_components_scored") or 0) / 12.0)

    elif system_id == "action_semantics_ledger":
        for p in hyps:
            rows = p.get("rows") or []
            slot[4] = min(1.0, len(rows) / 12.0)
            if rows:
                ch = sum(1 for r in rows if r.get("frame_changed"))
                slot[5] = ch / max(1, len(rows))
            rates = p.get("effect_rates") or {}
            if isinstance(rates, dict):
                # per-action effect rates occupy slots 6..11 (aids 1..6)
                for i, aid in enumerate((1, 2, 3, 4, 5, 6)):
                    if str(aid) in rates:
                        slot[6 + i] = _clip01(float(rates[str(aid)]))

    else:
        # generic fallback for any future system
        _fill_from_proposals(slot, props, start=4)
        slot[4] = min(1.0, len(hyps) / 4.0)

    return slot


def encode_from_frame(
    frame: np.ndarray,
    *,
    game_id: str = "",
    transitions: list[Transition] | None = None,
    available_actions: list[int] | None = None,
    levels_completed: int = 0,
    run_consume: bool = True,
) -> np.ndarray:
    """Full-mine feature vector. run_consume=True is the default training path."""
    grid = np.asarray(frame)
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    o = 0

    # global
    g = _encode_global(grid)
    vec[o : o + GLOBAL_DIM] = g
    o += GLOBAL_DIM

    bundle = None
    if run_consume:
        bundle = consume_all(
            grid,
            game_id=game_id,
            transitions=transitions,
            available_actions=available_actions,
            levels_completed=levels_completed,
            systems=list(SYSTEM_ORDER),
        )
    systems = (bundle or {}).get("systems") or {}

    # per-system slots
    present = []
    for sid in SYSTEM_ORDER:
        pkt = systems.get(sid) if systems else None
        if isinstance(pkt, dict) and "hypotheses" not in pkt and "receipt" not in pkt:
            # already to_dict form from consume_all
            slot = encode_system_slot(sid, pkt, grid)
        elif pkt is None:
            slot = _slot_base()
        else:
            # LegalPacket.to_dict()
            slot = encode_system_slot(sid, pkt if isinstance(pkt, dict) else {}, grid)
        vec[o : o + SYS_SLOT] = slot
        present.append(slot[0])
        o += SYS_SLOT

    # meta: per-system presence + guards
    for i, pflag in enumerate(present):
        vec[o + i] = pflag
    o += N_SYSTEMS
    vec[o + 0] = 1.0  # legal path
    vec[o + 1] = min(1.0, float(levels_completed) / 8.0)
    vec[o + 2] = 1.0 if game_id else 0.0
    vec[o + 3] = 1.0 if run_consume else 0.0
    vec[o + 4] = min(1.0, len(transitions or []) / 64.0)
    vec[o + 5] = float(np.mean(present)) if present else 0.0
    vec[o + 6] = 1.0  # fullmine schema marker
    vec[o + 7] = FEATURE_DIM / 512.0
    o += 8

    assert o == FEATURE_DIM, f"feature pack {o} != {FEATURE_DIM}"
    return vec


def encode_legal_features(frame: np.ndarray, **kwargs: Any) -> np.ndarray:
    kwargs.setdefault("run_consume", True)
    return encode_from_frame(frame, **kwargs)


def feature_report(vec: np.ndarray) -> dict[str, Any]:
    """Debug: per-system L2 mass + depth (nz beyond present flag)."""
    v = np.asarray(vec, dtype=np.float32)
    assert v.shape[0] == FEATURE_DIM
    masses = {}
    base = GLOBAL_DIM
    n_deep = 0
    for i, sid in enumerate(SYSTEM_ORDER):
        sl = v[base + i * SYS_SLOT : base + (i + 1) * SYS_SLOT]
        nz = int(np.count_nonzero(sl[1:]))
        # structured depth: more than legal/hyp/prop flags (indices 1-3)
        deep = nz >= 3
        if deep:
            n_deep += 1
        masses[sid] = {
            "present": float(sl[0]),
            "l2": float(np.linalg.norm(sl)),
            "mean": float(sl.mean()),
            "nz_after_present": nz,
            "depth_ok": bool(deep),
        }
    return {
        "schema": FEATURE_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "global_l2": float(np.linalg.norm(v[:GLOBAL_DIM])),
        "systems": masses,
        "n_systems_present": int(sum(1 for m in masses.values() if m["present"] > 0.5)),
        "n_systems_depth_ok": n_deep,
    }
