#!/usr/bin/env python3
"""Step-printing live debug runner for the frame-sourced SP80 backend.

This executes real SP80 primitives against the local engine, but it omits the
heavy CONTRA report packets so backend planning/control problems are visible
immediately. It is a debug/scout tool, not a gameplay report.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

import numpy as np

logging.disable(logging.CRITICAL)

from sp80_base_controller import GroundedAction, SP80BaseController
from sp80_spill_planner_backend import SpillPlannerBrain

from arcengine import GameAction


def _packet(level_index: int, action_index: int) -> dict[str, Any]:
    return {
        "schema": "codexlab.sp80_contra_recommendation_packet.v1",
        "recommendation_packet_id": f"frame_live_debug_l{level_index:02d}_a{action_index:03d}",
        "ranked_recommendations": [],
    }


def _ground(controller: SP80BaseController, game: Any, primitive: str, override: dict[str, Any] | None) -> GroundedAction:
    if primitive == "select_object" and isinstance(override, dict):
        click_xy = override.get("click_xy")
        if isinstance(click_xy, list) and len(click_xy) == 2:
            return GroundedAction(
                primitive=primitive,
                action=GameAction.ACTION6,
                data={"x": int(click_xy[0]), "y": int(click_xy[1])},
                selected_target=None,
            )
    return controller.ground_primitive(game, primitive, target=None)


def _memory_row(index: int, primitive: str, before: dict[str, Any], after: dict[str, Any], recommendation: dict[str, Any]) -> dict[str, Any]:
    before_coord = before.get("selected_coord")
    after_coord = after.get("selected_coord")
    delta = None
    if isinstance(before_coord, list) and isinstance(after_coord, list) and len(before_coord) == 2 and len(after_coord) == 2:
        delta = [int(after_coord[0]) - int(before_coord[0]), int(after_coord[1]) - int(before_coord[1])]
    return {
        "action_index": int(index),
        "primitive": primitive,
        "selected_coord_before": before_coord,
        "selected_coord_after": after_coord,
        "selected_coord_delta": delta,
        "selected_recommendation": recommendation,
        "status": "executed",
    }


def run(level_index: int, *, max_actions: int, search_time_budget: float) -> int:
    controller = SP80BaseController(level_index=level_index)
    game, obs = controller.new_game()
    brain = SpillPlannerBrain(search_time_budget=search_time_budget, search_restarts=12)
    action_memory: list[dict[str, Any]] = []
    initial_completed = int(getattr(obs, "levels_completed", 0))
    for action_index in range(max_actions):
        before_frame = np.asarray(obs.frame[-1])
        before_state = controller.state(game, obs)
        brain.bind_episode(game=game, controller=controller, target=None, before_frame=before_frame)
        packet = brain.augment_recommendation_packet(
            _packet(level_index, action_index),
            before_state=before_state,
            action_memory=action_memory,
        )
        row = (packet.get("ranked_recommendations") or [{}])[0]
        primitive = str(row.get("primitive") or "")
        if not primitive:
            print(json.dumps({"action_index": action_index, "status": "no_primitive", "packet": packet}, sort_keys=True), flush=True)
            return 2
        override = row.get("target_override") if isinstance(row.get("target_override"), dict) else None
        grounded = _ground(controller, game, primitive, override)
        obs = controller.execute_grounded_action(game, grounded)
        after_state = controller.state(game, obs)
        memory = _memory_row(action_index, primitive, before_state, after_state, row)
        action_memory.append(memory)
        meta = ((packet.get("spill_planner_overlay") or {}).get("meta") or {})
        print(
            json.dumps(
                {
                    "action_index": action_index,
                    "primitive": primitive,
                    "click_xy": None if override is None else override.get("click_xy"),
                    "before": before_state.get("selected_coord"),
                    "after": after_state.get("selected_coord"),
                    "before_source": before_state.get("selected_coord_source"),
                    "after_source": after_state.get("selected_coord_source"),
                    "before_shape": before_state.get("selected_shape"),
                    "after_shape": after_state.get("selected_shape"),
                    "delta": memory["selected_coord_delta"],
                    "defer_gate": row.get("evidence_refs", {}).get("defer_gate"),
                    "planning_state_source": row.get("evidence_refs", {}).get("planning_state_source"),
                    "privileged_state_used_for_action_choice": row.get("evidence_refs", {}).get("privileged_state_used_for_action_choice"),
                    "target_layout": row.get("evidence_refs", {}).get("planned_target_layout"),
                    "frame_track_assignment": row.get("evidence_refs", {}).get("frame_track_assignment"),
                    "moving_occupant_idx": row.get("evidence_refs", {}).get("moving_occupant_idx"),
                    "moving_track_id": row.get("evidence_refs", {}).get("moving_track_id"),
                    "occupant_target_xy": row.get("evidence_refs", {}).get("occupant_target_xy"),
                    "selected_occupant_idx": meta.get("selected_occupant_idx"),
                    "selected_occupant_idx_source": meta.get("selected_occupant_idx_source"),
                    "winning_layout_found": meta.get("winning_layout_found"),
                    "plan_strategy": meta.get("plan_strategy"),
                    "level_delta_total": int(after_state.get("levels_completed") or 0) - initial_completed,
                    "state": after_state.get("state"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if int(after_state.get("levels_completed") or 0) - initial_completed > 0:
            return 0
        if str(after_state.get("state")) == "GAME_OVER":
            return 1
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, required=True)
    parser.add_argument("--max-actions", type=int, default=40)
    parser.add_argument("--spill-search-time-budget", type=float, default=8.0)
    args = parser.parse_args()
    return run(args.level, max_actions=args.max_actions, search_time_budget=args.spill_search_time_budget)


if __name__ == "__main__":
    raise SystemExit(main())
