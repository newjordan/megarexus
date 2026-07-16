"""Crucible transition mechanics — mined from crucible/forward_v0 + grammar.

KEEP: naive translation predictions, discrepancy vs observation, rule candidates,
      local neighborhood anomalies, simple grammar induction.
STRIP: source-transcribed game rules, SP80 palette as truth domain.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from legal_search.algorithms.components import change_regions, frame_delta_count
from legal_search.algorithms.role_induce import induce_movable_colors, induce_static_colors
from legal_search.types import Transition

AID_PRIMITIVE: dict[int, str] = {
    1: "move_up",
    2: "move_down",
    3: "move_left",
    4: "move_right",
    5: "activate",
    6: "click",
}

MOVE_DELTAS: dict[str, tuple[int, int]] = {
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}

GRAMMAR_BOUNDARY = (
    "mechanic_forward_model_is_hypothesis_ledger_composition_not_control_truth"
)


def predict_v0(primitive: str) -> dict[str, Any]:
    if primitive in MOVE_DELTAS:
        return {
            "model_version": "naive_translation_v0",
            "primitive": primitive,
            "predicted_delta_xy": list(MOVE_DELTAS[primitive]),
            "predicted_blocked": False,
            "effect_kind": "forward",
        }
    if primitive in ("activate", "click"):
        return {
            "model_version": "naive_translation_v0",
            "primitive": primitive,
            "predicted_delta_xy": [0, 0],
            "predicted_blocked": False,
            "effect_kind": "static",
        }
    return {
        "model_version": "naive_translation_v0",
        "primitive": primitive,
        "predicted_delta_xy": None,
        "predicted_blocked": False,
        "effect_kind": "unknown_primitive",
    }


def observed_mass_delta(tr: Transition, color: int | None) -> tuple[int, int] | None:
    if color is None:
        return None
    before = np.asarray(tr.frame_before)
    after = np.asarray(tr.frame_after)
    if before.shape != after.shape:
        return None
    bm = before == int(color)
    am = after == int(color)
    if not np.any(bm) or not np.any(am):
        return None
    by, bx = np.nonzero(bm)
    ay, ax = np.nonzero(am)
    dx = int(round(float(ax.mean() - bx.mean())))
    dy = int(round(float(ay.mean() - by.mean())))
    return dx, dy


def discrepancy_row(tr: Transition, color: int | None) -> dict[str, Any]:
    aid = int(tr.action.action_id)
    prim = AID_PRIMITIVE.get(aid, f"aid_{aid}")
    pred = predict_v0(prim)
    obs = observed_mass_delta(tr, color)
    pred_d = pred.get("predicted_delta_xy")
    match = False
    if obs is not None and pred_d is not None:
        match = list(obs) == list(pred_d)
    blocked_hyp = (
        pred.get("effect_kind") == "forward"
        and tr.frame_changed is False
        and aid in (1, 2, 3, 4)
    )
    return {
        "primitive": prim,
        "action": tr.action.to_dict(),
        "prediction": pred,
        "observed_delta_xy": list(obs) if obs is not None else None,
        "prediction_match": match,
        "blocked_hypothesis": blocked_hyp,
        "delta_pixels": frame_delta_count(tr.frame_before, tr.frame_after),
        "frame_changed": tr.frame_changed,
        "change_regions": change_regions(tr.frame_before, tr.frame_after, max_regions=2),
        "admitted": False,
    }


def induce_rule_candidates(discrepancies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rules: list[dict[str, Any]] = []
    by_prim: dict[str, list[dict[str, Any]]] = {}
    for d in discrepancies:
        by_prim.setdefault(d["primitive"], []).append(d)
    for prim, rows in by_prim.items():
        n = len(rows)
        if n < 2:
            continue
        match_rate = sum(1 for r in rows if r["prediction_match"]) / n
        block_rate = sum(1 for r in rows if r["blocked_hypothesis"]) / n
        change_rate = sum(1 for r in rows if r["frame_changed"]) / n
        if match_rate >= 0.5:
            rules.append(
                {
                    "rule_id": f"v0_match:{prim}",
                    "selector": "per_primitive",
                    "primitive": prim,
                    "effect": "forward_matches_v0",
                    "support": n,
                    "confidence": match_rate,
                    "admitted": False,
                    "boundary": GRAMMAR_BOUNDARY,
                }
            )
        if block_rate >= 0.5:
            rules.append(
                {
                    "rule_id": f"block:{prim}",
                    "selector": "per_primitive",
                    "primitive": prim,
                    "effect": "blocked",
                    "support": n,
                    "confidence": block_rate,
                    "admitted": False,
                    "boundary": GRAMMAR_BOUNDARY,
                }
            )
        if change_rate >= 0.6 and prim.startswith("move_"):
            rules.append(
                {
                    "rule_id": f"changes:{prim}",
                    "selector": "per_primitive",
                    "primitive": prim,
                    "effect": "forward",
                    "support": n,
                    "confidence": change_rate,
                    "admitted": False,
                    "boundary": GRAMMAR_BOUNDARY,
                }
            )
    rules.sort(key=lambda r: (-float(r["confidence"]), -int(r["support"])))
    return rules[:16]


def local_neighborhood(frame, center_xy, radius: int = 2) -> list[list[int]]:
    g = np.asarray(frame)
    h, w = g.shape[:2]
    cx, cy = int(round(center_xy[0])), int(round(center_xy[1]))
    out = []
    for dy in range(-radius, radius + 1):
        row = []
        for dx in range(-radius, radius + 1):
            y, x = cy + dy, cx + dx
            if 0 <= y < h and 0 <= x < w:
                row.append(int(g[y, x]))
            else:
                row.append(-1)
        out.append(row)
    return out


def anomaly_from_discrepancy(
    d: dict[str, Any], tr: Transition, color: int | None
) -> dict[str, Any] | None:
    if d.get("prediction_match"):
        return None
    prim = d.get("primitive") or ""
    if not str(prim).startswith("move_"):
        return None
    obs = d.get("observed_delta_xy")
    pred = (d.get("prediction") or {}).get("predicted_delta_xy")
    if obs is None:
        if d.get("blocked_hypothesis"):
            anomaly_kind = "blocked"
            obs = [0, 0]
        else:
            return None
    else:
        if obs == [0, 0] and pred and pred != [0, 0]:
            anomaly_kind = "blocked"
        elif obs == pred:
            return None
        else:
            anomaly_kind = "redirect_or_partial"
    center = [32.0, 32.0]
    if color is not None:
        before = np.asarray(tr.frame_before)
        m = before == int(color)
        if np.any(m):
            ys, xs = np.nonzero(m)
            center = [float(xs.mean()), float(ys.mean())]
    return {
        "anomaly_id": f"a_{tr.step}_{prim}",
        "primitive": prim,
        "anomaly_kind": anomaly_kind,
        "observed": {"canon_delta": obs},
        "predicted_v0": pred,
        "local_context": {
            "canon_neighborhood_5x5": local_neighborhood(tr.frame_before, center),
            "actor_axis": "h" if prim in ("move_left", "move_right") else "v",
            "actor_center_xy": center,
        },
    }


def enumerate_simple_grammar(anomalies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kind_to_effect = {"blocked": "blocked", "redirect_or_partial": "forward"}
    rules: dict[str, dict[str, Any]] = {}
    for an in anomalies:
        effect = kind_to_effect.get(str(an.get("anomaly_kind")))
        if not effect:
            continue
        prim = str(an.get("primitive") or "")
        rid = f"gram:{prim}:{effect}"
        row = rules.get(rid)
        if row is None:
            row = {
                "rule_id": rid,
                "selector": {"kind": "per_primitive", "primitive": prim},
                "atoms": [{"kind": "always"}],
                "effect": effect,
                "induced_from": [],
                "admitted": False,
                "boundary": GRAMMAR_BOUNDARY,
            }
            rules[rid] = row
        aid = str(an.get("anomaly_id") or "")
        if aid and aid not in row["induced_from"]:
            row["induced_from"].append(aid)
        row["support"] = len(row["induced_from"])
        row["confidence"] = min(1.0, 0.3 + 0.1 * row["support"])
    return sorted(
        rules.values(),
        key=lambda r: (-float(r.get("confidence") or 0), -int(r.get("support") or 0)),
    )[:24]


def falsify_rules(
    rules: list[dict[str, Any]],
    discrepancies: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Falsify loop: drop or down-weight rules contradicted by discrepancies.

    Mined idea from crucible/falsify.py — hypotheses only.
    """
    out = []
    for rule in rules:
        prim = str(rule.get("primitive") or (rule.get("selector") or {}).get("primitive") or "")
        effect = str(rule.get("effect") or "")
        relevant = [d for d in discrepancies if d.get("primitive") == prim]
        if not relevant:
            row = dict(rule)
            row["falsified"] = False
            row["n_tests"] = 0
            out.append(row)
            continue
        n = len(relevant)
        if effect in ("blocked",):
            support = sum(1 for d in relevant if d.get("blocked_hypothesis"))
            contradict = sum(1 for d in relevant if d.get("frame_changed"))
        elif effect in ("forward", "forward_matches_v0", "changes"):
            support = sum(1 for d in relevant if d.get("frame_changed") or d.get("prediction_match"))
            contradict = sum(1 for d in relevant if d.get("blocked_hypothesis"))
        else:
            support = sum(1 for d in relevant if d.get("prediction_match"))
            contradict = n - support
        conf = support / max(1, n)
        falsified = contradict > support and n >= 2
        row = dict(rule)
        row["n_tests"] = n
        row["n_support"] = support
        row["n_contradict"] = contradict
        row["confidence"] = conf
        row["falsified"] = falsified
        row["admitted"] = False
        if not falsified:
            out.append(row)
    out.sort(key=lambda r: (-float(r.get("confidence") or 0), -int(r.get("n_support") or 0)))
    return out


def mine_crucible(transitions: list[Transition] | None) -> dict[str, Any]:
    trs = list(transitions or [])
    movable = induce_movable_colors(trs)
    static = induce_static_colors(trs)
    top_color = int(max(movable.items(), key=lambda kv: kv[1])[0]) if movable else None
    window = trs[-24:]
    disc = [discrepancy_row(tr, top_color) for tr in window]
    anomalies = []
    for tr, d in zip(window, disc):
        an = anomaly_from_discrepancy(d, tr, top_color)
        if an:
            anomalies.append(an)
    rules = induce_rule_candidates(disc)
    seen = {r["rule_id"] for r in rules}
    for g in enumerate_simple_grammar(anomalies):
        if g["rule_id"] not in seen:
            rules.append(g)
            seen.add(g["rule_id"])
    surviving = falsify_rules(rules, disc)
    match_frac = (
        sum(1 for d in disc if d["prediction_match"]) / max(1, len(disc)) if disc else 0.0
    )
    return {
        "movable": movable,
        "static": static,
        "top_movable": top_color,
        "discrepancies": disc[-12:],
        "n_discrepancies": len(disc),
        "anomalies": anomalies[:12],
        "n_anomalies": len(anomalies),
        "v0_match_fraction": match_frac,
        "rule_candidates": surviving[:20],
        "n_rules": len(surviving),
        "n_rules_before_falsify": len(rules),
        "n_falsified": max(0, len(rules) - len(surviving)),
        "n_transitions": len(trs),
        "n_changed": sum(1 for tr in trs if tr.frame_changed),
        "change_fraction": sum(1 for tr in trs if tr.frame_changed) / max(1, len(trs)),
        "n_move_actions": sum(1 for tr in trs if int(tr.action.action_id) in (1, 2, 3, 4)),
        "boundary": GRAMMAR_BOUNDARY,
        "source_rules_used": False,
    }
