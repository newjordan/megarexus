"""Real crucible mine_crucible on episode transitions — no entropy proxy."""

from __future__ import annotations

from typing import Any

import numpy as np

from synthesis_lab.orchestration.rdl.brain.types import KnownMapView, OperatorResult, TruthFlags


def _as_transitions(raw: list[Any], game_id: str) -> list[Any]:
    from legal_search.types import Action, Transition

    out = []
    for i, tr in enumerate(raw[-32:]):
        if hasattr(tr, "frame_before") and hasattr(tr, "action"):
            out.append(tr)
            continue
        if not isinstance(tr, dict):
            continue
        out.append(
            Transition(
                game_id=game_id,
                step=int(tr.get("step", i)),
                frame_before=tr["frame_before"],
                frame_after=tr["frame_after"],
                action=Action(
                    action_id=int(tr.get("action_id", tr.get("action", 0))),
                    x=tr.get("x"),
                    y=tr.get("y"),
                ),
                levels_before=int(tr.get("levels_before", 0)),
                levels_after=int(tr.get("levels_after", 0)),
                frame_changed=bool(tr.get("frame_changed", True)),
                levels_up=bool(tr.get("levels_up", False)),
            )
        )
    return out


class CrucibleFullOp:
    name = "crucible"

    def run(self, map_view: KnownMapView, context: dict[str, Any] | None = None) -> OperatorResult:
        context = context or {}
        transitions = context.get("transitions") or []
        game_id = str(context.get("game_id") or "megamind")
        if len(transitions) < 1:
            return OperatorResult(
                name=self.name,
                status="degraded",
                scores={"n_transitions": 0.0},
                messages=["need_episode_transitions_for_crucible"],
                hypothesis_only=True,
                truth_flags=TruthFlags(),
            )
        try:
            from legal_search.algorithms.crucible_mine import mine_crucible

            trs = _as_transitions(transitions, game_id)
            pkt = mine_crucible(trs)
            scores = {
                "n_transitions": float(pkt.get("n_transitions") or 0),
                "n_changed": float(pkt.get("n_changed") or 0),
                "change_fraction": float(pkt.get("change_fraction") or 0),
                "n_discrepancies": float(pkt.get("n_discrepancies") or 0),
                "n_anomalies": float(pkt.get("n_anomalies") or 0),
                "n_rules": float(pkt.get("n_rules") or 0),
                "n_rules_before_falsify": float(pkt.get("n_rules_before_falsify") or 0),
                "n_falsified": float(pkt.get("n_falsified") or 0),
                "v0_match_fraction": float(pkt.get("v0_match_fraction") or 0),
                "n_move_actions": float(pkt.get("n_move_actions") or 0),
                "source_rules_used": 1.0 if pkt.get("source_rules_used") else 0.0,
                "top_movable": float(pkt.get("top_movable") or -1),
            }
            return OperatorResult(
                name=self.name,
                status="ok",
                scores=scores,
                summary={
                    "boundary": pkt.get("boundary"),
                    "n_movable_colors": len(pkt.get("movable") or {}),
                    "n_static_colors": len(pkt.get("static") or {}),
                },
                proposals=[
                    {
                        "kind": "crucible_rule",
                        "rule_id": r.get("rule_id"),
                        "admitted": False,
                    }
                    for r in (pkt.get("rule_candidates") or [])[:8]
                    if isinstance(r, dict)
                ],
                hypothesis_only=True,
                truth_flags=TruthFlags(),
                meta={"backend": "legal_search.crucible_mine.mine_crucible"},
                messages=["crucible_full_mine"],
            )
        except Exception as exc:
            return OperatorResult(
                name=self.name,
                status="degraded",
                scores={},
                messages=[f"mine_crucible:{type(exc).__name__}:{exc}"],
                hypothesis_only=True,
                truth_flags=TruthFlags(),
            )
