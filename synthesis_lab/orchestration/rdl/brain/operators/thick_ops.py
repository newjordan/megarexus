"""Thick operator pack — load more legal_search / CONTRA-mined algorithms than needed.

Philosophy: seed the center with many hypothesis channels; training grows what works.
All fail-closed / hypothesis_only. Degrade to empty scores if import fails.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from synthesis_lab.orchestration.rdl.brain.types import KnownMapView, OperatorResult, TruthFlags


def _ok(name: str, scores: dict[str, float], **kwargs: Any) -> OperatorResult:
    return OperatorResult(
        name=name,
        status="ok",
        scores={k: float(v) for k, v in scores.items()},
        hypothesis_only=True,
        truth_flags=TruthFlags(),
        **kwargs,
    )


def _deg(name: str, msg: str) -> OperatorResult:
    return OperatorResult(
        name=name,
        status="degraded",
        scores={},
        messages=[msg],
        hypothesis_only=True,
        truth_flags=TruthFlags(),
    )


class SymmetryOp:
    name = "symmetry"

    def run(self, map_view: KnownMapView, context: dict[str, Any] | None = None) -> OperatorResult:
        frame = np.asarray(map_view.frame if map_view.frame is not None else context.get("frame"))
        comps = list(map_view.components or [])
        try:
            from legal_search.algorithms.local_symmetry import mine_symmetry
            from legal_search.algorithms.saliency import (
                symmetry_score_horizontal,
                symmetry_score_vertical,
            )

            pkt = mine_symmetry(frame, comps)
            scores = {
                "global_h": float(pkt.get("global_horizontal") or symmetry_score_horizontal(frame)),
                "global_v": float(pkt.get("global_vertical") or symmetry_score_vertical(frame)),
                "n_self": float(pkt.get("n_self") or 0),
                "n_pairs": float(pkt.get("n_pairs") or 0),
                "best_self": float(pkt.get("best_self") or 0),
                "best_pair": float(pkt.get("best_pair") or 0),
            }
            return _ok(self.name, scores, meta={"backend": "legal_search.local_symmetry"})
        except Exception as exc:
            return _deg(self.name, f"{type(exc).__name__}:{exc}")


class GeoOp:
    """Pair geometry + multi-variate pattern cues (not pour/SP80-bound)."""

    name = "geo"

    def run(self, map_view: KnownMapView, context: dict[str, Any] | None = None) -> OperatorResult:
        comps = list(map_view.components or [])
        context = context or {}
        try:
            from legal_search.algorithms.geometry_rel import mine_geo_relations

            pkt = mine_geo_relations(comps)
            scores = {
                "n_components": float(len(comps)),
                "n_pairs": float(pkt.get("n_pairs") or len(pkt.get("pairs") or [])),
                "n_nearest": float(len(pkt.get("nearest") or [])),
            }
            if "mean_manhattan" in pkt:
                scores["mean_manhattan"] = float(pkt["mean_manhattan"])
            if "min_manhattan" in pkt:
                scores["min_manhattan"] = float(pkt["min_manhattan"])
            # multi-variate axes (color variables, aspect spectrum)
            colors = {int(c.get("color") or 0) for c in comps}
            scores["n_color_vars"] = float(len(colors))
            scores["multivariate"] = 1.0 if len(colors) >= 2 else 0.0
            aspects = []
            for c in comps:
                box = c.get("bbox_xyxy") or c.get("bbox")
                if box and len(box) >= 4:
                    bw = max(1.0, float(box[2]) - float(box[0]) + 1)
                    bh = max(1.0, float(box[3]) - float(box[1]) + 1)
                    aspects.append(max(bw, bh) / min(bw, bh))
            if aspects:
                scores["mean_aspect"] = float(sum(aspects) / len(aspects))
                scores["max_aspect"] = float(max(aspects))
            scores["pattern_growth_geo"] = 1.0
            scores["not_sp80_bound"] = 1.0
            return _ok(
                self.name,
                scores,
                meta={"backend": "legal_search.geometry_rel+multivariate"},
            )
        except Exception as exc:
            return _deg(self.name, f"{type(exc).__name__}:{exc}")


class SaliencyOp:
    name = "saliency"

    def run(self, map_view: KnownMapView, context: dict[str, Any] | None = None) -> OperatorResult:
        frame = np.asarray(map_view.frame if map_view.frame is not None else context.get("frame"))
        try:
            from legal_search.algorithms.saliency import (
                color_histogram,
                edge_map,
                symmetry_score_horizontal,
                symmetry_score_vertical,
            )

            edges = edge_map(frame)
            hist = color_histogram(frame)
            scores = {
                "edge_density": float(edges.mean()),
                "edge_max": float(edges.max()),
                "hist_entropy": float(-np.sum((hist + 1e-9) * np.log(hist + 1e-9))),
                "n_colors": float(np.count_nonzero(hist)),
                "sym_h": float(symmetry_score_horizontal(frame)),
                "sym_v": float(symmetry_score_vertical(frame)),
                "hist_peak": float(hist.max()),
            }
            return _ok(self.name, scores, meta={"backend": "legal_search.saliency"})
        except Exception as exc:
            return _deg(self.name, f"{type(exc).__name__}:{exc}")


class AffordanceOp:
    name = "affordances"

    def run(self, map_view: KnownMapView, context: dict[str, Any] | None = None) -> OperatorResult:
        comps = list(map_view.components or [])
        try:
            from legal_search.algorithms import affordances as aff

            # call whatever public mine/build exists
            fn = getattr(aff, "mine_affordances", None) or getattr(aff, "build_affordances", None)
            if fn is None:
                # score by area rank as weak stand-in
                areas = sorted([float(c.get("area") or 0) for c in comps], reverse=True)
                scores = {
                    "n_candidates": float(len(comps)),
                    "largest_area": areas[0] if areas else 0.0,
                    "area_ratio_12": (areas[1] / areas[0]) if len(areas) > 1 and areas[0] > 0 else 0.0,
                }
                return _ok(self.name, scores, meta={"backend": "area_proxy"})
            frame = np.asarray(map_view.frame)
            pkt = fn(frame, comps) if fn.__code__.co_argcount >= 2 else fn(comps)
            scores = {"n_candidates": float(len(comps))}
            if isinstance(pkt, dict):
                for k, v in list(pkt.items())[:10]:
                    if isinstance(v, (int, float)):
                        scores[str(k)] = float(v)
                    elif isinstance(v, list):
                        scores[f"n_{k}"] = float(len(v))
            return _ok(self.name, scores, meta={"backend": "legal_search.affordances"})
        except Exception as exc:
            return _deg(self.name, f"{type(exc).__name__}:{exc}")


class StateConOp:
    name = "statecon"

    def run(self, map_view: KnownMapView, context: dict[str, Any] | None = None) -> OperatorResult:
        comps = list(map_view.components or [])
        frame = np.asarray(map_view.frame if map_view.frame is not None else context.get("frame"))
        try:
            from legal_search.algorithms.statecon_graph import mine_statecon

            pkt = mine_statecon(frame)
            scores = {"n_components": float(len(comps))}
            if isinstance(pkt, dict):
                for k, v in list(pkt.items())[:12]:
                    if isinstance(v, (int, float)):
                        scores[str(k)] = float(v)
                    elif isinstance(v, list):
                        scores[f"n_{k}"] = float(len(v))
                    elif isinstance(v, dict):
                        scores[f"n_{k}"] = float(len(v))
            return _ok(self.name, scores, meta={"backend": "legal_search.statecon_graph.mine_statecon"})
        except Exception as exc:
            return _deg(self.name, f"{type(exc).__name__}:{exc}")


class DeltaOp:
    """Change-under-action geometry if previous frame provided in context."""

    name = "delta"

    def run(self, map_view: KnownMapView, context: dict[str, Any] | None = None) -> OperatorResult:
        context = context or {}
        prev = context.get("frame_prev")
        frame = map_view.frame if map_view.frame is not None else context.get("frame")
        if prev is None or frame is None:
            return _deg(self.name, "no_frame_prev")
        try:
            from legal_search.algorithms.components import frame_delta_count

            a = np.asarray(prev)
            b = np.asarray(frame)
            if a.shape != b.shape:
                return _deg(self.name, "shape_mismatch")
            dpx = float(frame_delta_count(a, b)) if callable(frame_delta_count) else float((a != b).sum())
            mask = a != b
            scores = {
                "delta_pixels": dpx,
                "delta_frac": float(mask.mean()),
                "did_change": 1.0 if dpx > 0 else 0.0,
            }
            if mask.any():
                ys, xs = np.nonzero(mask)
                scores["change_cx"] = float(xs.mean())
                scores["change_cy"] = float(ys.mean())
                scores["change_span_x"] = float(xs.max() - xs.min() + 1)
                scores["change_span_y"] = float(ys.max() - ys.min() + 1)
            return _ok(self.name, scores, meta={"backend": "frame_delta"})
        except Exception as exc:
            # pure numpy fallback
            a = np.asarray(prev)
            b = np.asarray(frame)
            if a.shape != b.shape:
                return _deg(self.name, f"{type(exc).__name__}")
            mask = a != b
            return _ok(
                self.name,
                {
                    "delta_pixels": float(mask.sum()),
                    "delta_frac": float(mask.mean()),
                    "did_change": 1.0 if mask.any() else 0.0,
                },
                meta={"backend": "numpy_delta"},
            )


class GhostOp:
    """Episode ghost: confirmed change relations from context transitions."""

    name = "ghost"

    def run(self, map_view: KnownMapView, context: dict[str, Any] | None = None) -> OperatorResult:
        context = context or {}
        transitions = context.get("transitions") or []
        if not transitions:
            return _deg(self.name, "no_episode_transitions")
        try:
            from legal_search.algorithms.ghost_relations import build_confirmed_relation_rows
            from legal_search.types import Action, Transition

            # accept either Transition objects or plain dicts
            rows_in = []
            for i, tr in enumerate(transitions[-32:]):
                if hasattr(tr, "frame_before"):
                    rows_in.append(tr)
                elif isinstance(tr, dict) and "frame_before" in tr:
                    rows_in.append(
                        Transition(
                            game_id=str(context.get("game_id") or "megamind"),
                            step=int(tr.get("step", i)),
                            frame_before=tr["frame_before"],
                            frame_after=tr["frame_after"],
                            action=Action(
                                action_id=int(tr.get("action_id", tr.get("action", 0))),
                                x=int(tr.get("x", 0)) if tr.get("x") is not None else None,
                                y=int(tr.get("y", 0)) if tr.get("y") is not None else None,
                            ),
                            levels_before=int(tr.get("levels_before", 0)),
                            levels_after=int(tr.get("levels_after", 0)),
                            frame_changed=bool(tr.get("frame_changed", True)),
                            levels_up=bool(tr.get("levels_up", False)),
                        )
                    )
            if not rows_in:
                return _deg(self.name, "empty_transitions")
            rows = build_confirmed_relation_rows(rows_in, game_id=str(context.get("game_id") or ""))
            scores = {
                "n_confirmed": float(len(rows)),
                "n_transitions_seen": float(len(rows_in)),
            }
            if rows:
                dpx = [float((r.get("geometry") or {}).get("delta_pixels") or 0) for r in rows]
                scores["mean_confirmed_dpx"] = float(np.mean(dpx)) if dpx else 0.0
            return _ok(
                self.name,
                scores,
                proposals=[{"kind": "ghost_row", "id": r.get("relation_memory_id"), "admitted": False} for r in rows[:6]],
                meta={"backend": "legal_search.ghost_relations"},
            )
        except Exception as exc:
            return _deg(self.name, f"{type(exc).__name__}:{exc}")


class CrucibleOp:
    """Crucible static mine on frame (rules/discrepancy stubs if no corpus)."""

    name = "crucible"

    def run(self, map_view: KnownMapView, context: dict[str, Any] | None = None) -> OperatorResult:
        frame = np.asarray(map_view.frame if map_view.frame is not None else context.get("frame"))
        try:
            from legal_search.algorithms import crucible_mine as cm

            fn = getattr(cm, "mine_frame", None) or getattr(cm, "mine_static", None) or getattr(cm, "score_frame", None)
            if fn is None:
                # entropy / component stability proxy
                comps = list(map_view.components or [])
                hist = np.bincount(frame.reshape(-1), minlength=16)[:16].astype(np.float64)
                hist = hist / max(hist.sum(), 1.0)
                ent = float(-np.sum((hist + 1e-9) * np.log(hist + 1e-9)))
                return _ok(
                    self.name,
                    {
                        "n_components": float(len(comps)),
                        "entropy": ent,
                        "proxy": 1.0,
                    },
                    meta={"backend": "crucible_proxy"},
                )
            pkt = fn(frame)
            scores = {"proxy": 0.0}
            if isinstance(pkt, dict):
                for k, v in list(pkt.items())[:12]:
                    if isinstance(v, (int, float)):
                        scores[str(k)] = float(v)
                    elif isinstance(v, list):
                        scores[f"n_{k}"] = float(len(v))
            return _ok(self.name, scores, meta={"backend": "legal_search.crucible_mine"})
        except Exception as exc:
            return _deg(self.name, f"{type(exc).__name__}:{exc}")


class ExploreOp:
    """Goose-style change-by-repetition rates if transitions present."""

    name = "explore"

    def run(self, map_view: KnownMapView, context: dict[str, Any] | None = None) -> OperatorResult:
        context = context or {}
        transitions = context.get("transitions") or []
        try:
            from legal_search.algorithms import explore as ex

            fn = getattr(ex, "update_rates", None) or getattr(ex, "mine_explore_rates", None)
            if fn is None or not transitions:
                # action histogram proxy
                acts = []
                for tr in transitions[-32:]:
                    if hasattr(tr, "action"):
                        acts.append(int(tr.action.action_id))
                    elif isinstance(tr, dict):
                        acts.append(int(tr.get("action_id", tr.get("action", 0))))
                hist = np.zeros(8, dtype=np.float64)
                for a in acts:
                    if 0 <= a < 8:
                        hist[a] += 1
                if acts:
                    hist /= len(acts)
                scores = {f"rate_a{i}": float(hist[i]) for i in range(8)}
                scores["n_seen"] = float(len(acts))
                return _ok(self.name, scores, meta={"backend": "action_hist_proxy"})
            pkt = fn(transitions)
            scores = {}
            if isinstance(pkt, dict):
                for k, v in list(pkt.items())[:16]:
                    if isinstance(v, (int, float)):
                        scores[str(k)] = float(v)
            return _ok(self.name, scores, meta={"backend": "legal_search.explore"})
        except Exception as exc:
            return _deg(self.name, f"{type(exc).__name__}:{exc}")


class LegalFullmineOp:
    """Full 392-D legal consume → compact scores for the pad (+ raw stats)."""

    name = "legal392"

    def run(self, map_view: KnownMapView, context: dict[str, Any] | None = None) -> OperatorResult:
        context = context or {}
        frame = np.asarray(map_view.frame if map_view.frame is not None else context.get("frame"))
        try:
            from legal_search.train.features import FEATURE_DIM, encode_legal_features, feature_report

            vec = encode_legal_features(
                frame,
                game_id=str(context.get("game_id") or "megamind"),
                available_actions=list(context.get("available_actions") or list(range(1, 7))),
                run_consume=True,
            )
            rep = feature_report(vec)
            # compact: mass stats + how many systems present
            scores = {
                "feature_dim": float(FEATURE_DIM),
                "l2": float(np.linalg.norm(vec)),
                "mean": float(vec.mean()),
                "nz_frac": float((np.abs(vec) > 1e-6).mean()),
                "max_abs": float(np.abs(vec).max()),
            }
            if isinstance(rep, dict):
                if "n_systems_present" in rep:
                    scores["n_systems_present"] = float(rep["n_systems_present"])
                if "n_systems_depth_ok" in rep:
                    scores["n_systems_depth_ok"] = float(rep["n_systems_depth_ok"])
                if "global_l2" in rep:
                    scores["global_l2"] = float(rep["global_l2"])
                systems = rep.get("systems") or {}
                if isinstance(systems, dict) and systems:
                    masses = [float(v.get("l2") or v or 0) if isinstance(v, dict) else float(v or 0) for v in systems.values()]
                    scores["mean_system_mass"] = float(np.mean(masses)) if masses else 0.0
            # pack first 8 projection bins into scores
            bins = np.abs(vec.reshape(-1)[: 16 * 8]).reshape(8, 16).mean(axis=1) if vec.size >= 128 else np.abs(vec)[:8]
            for i, b in enumerate(np.asarray(bins).reshape(-1)[:8]):
                scores[f"bin{i}"] = float(b)
            return _ok(
                self.name,
                scores,
                summary={"feature_dim": FEATURE_DIM, "report_keys": list(rep.keys())[:12] if isinstance(rep, dict) else []},
                meta={"backend": "legal_search.features.v2_fullmine"},
                vectors={"legal_vec_head": vec[:16].astype(np.float32).tolist()},
            )
        except Exception as exc:
            return _deg(self.name, f"{type(exc).__name__}:{exc}")


class PerceptionTrustOp:
    name = "perception_trust"

    def run(self, map_view: KnownMapView, context: dict[str, Any] | None = None) -> OperatorResult:
        # Reflect map_trust outcome + optional legal perception_trust
        scores = {
            "trusted": 1.0 if map_view.trusted else 0.0,
            "n_gate_failures": float(len(map_view.gate_failures or [])),
            "n_components": float(len(map_view.components or [])),
        }
        try:
            from legal_search.algorithms import perception_trust as pt

            fn = getattr(pt, "trust_score", None) or getattr(pt, "score_frame", None)
            if fn is not None:
                frame = np.asarray(map_view.frame)
                v = fn(frame)
                if isinstance(v, (int, float)):
                    scores["legal_trust"] = float(v)
                elif isinstance(v, dict):
                    for k, x in list(v.items())[:6]:
                        if isinstance(x, (int, float)):
                            scores[str(k)] = float(x)
        except Exception:
            pass
        return _ok(self.name, scores, meta={"backend": "map_trust+legal"})


# registry used by hemisphere_i
THICK_OPS: list[type] = [
    SymmetryOp,
    GeoOp,
    SaliencyOp,
    AffordanceOp,
    StateConOp,
    DeltaOp,
    GhostOp,
    CrucibleOp,
    ExploreOp,
    LegalFullmineOp,
    PerceptionTrustOp,
]
