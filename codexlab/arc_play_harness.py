"""ARC-AGI-3 play harness: drive a game with a pluggable policy, using
CONTRA senses as the perception layer, and measure levels_completed.

Default mode is local/offline. Live ARC scorecards require ARC_ALLOW_LIVE=1.
A policy is any callable
(observation, history) -> (action_id:int, x:int|None, y:int|None). The
observation carries the raw grid plus a CONTRA perception summary so an LLM
policy reasons over senses, not blank pixels.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np

logging.disable(logging.CRITICAL)

from arc_agi import Arcade            # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402
from arcengine import GameAction      # noqa: E402

import contra_shape_fit as sf         # noqa: E402
import contra_structural_sense as st  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
KEY_PATH = Path(os.environ.get("ARC_API_KEY_PATH", REPO_ROOT / ".arc_api_key"))
HOME_ENV_PATH = Path("/home/frosty40/.env")
LOCAL_ENV_DIR = REPO_ROOT / "environment" / "environment_files"
_ACT = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
        4: GameAction.ACTION4, 5: GameAction.ACTION5, 6: GameAction.ACTION6,
        7: GameAction.ACTION7, 0: GameAction.RESET}
TRUE_VALUES = {"1", "true", "yes", "y", "on"}


def _live_allowed() -> bool:
    return os.environ.get("ARC_ALLOW_LIVE", "").strip().lower() in TRUE_VALUES


def make_env(game_id: str):
    # Drive the SDK against local environment_files unless live mode is
    # explicitly enabled. Key presence alone must not open a scorecard.
    if not _live_allowed():
        arc = Arcade(arc_api_key="anonymous", environments_dir=str(LOCAL_ENV_DIR),
                     operation_mode=OperationMode.OFFLINE)
        mode = "offline_local_environment_files"
    elif KEY_PATH.exists():
        arc = Arcade(arc_api_key=KEY_PATH.read_text().strip())
        mode = "online_key_file_explicit"
    elif os.environ.get("ARC_API_KEY"):
        arc = Arcade(arc_api_key=os.environ["ARC_API_KEY"].strip())
        mode = "online_env_explicit"
    elif _home_env_value("ARC_API_KEY"):
        arc = Arcade(arc_api_key=_home_env_value("ARC_API_KEY"))
        mode = "online_home_env_explicit"
    else:
        raise RuntimeError("ARC_ALLOW_LIVE=1 set but no ARC_API_KEY source is available")
    card = arc.open_scorecard()
    env = arc.make(game_id, scorecard_id=card)
    return arc, card, env, mode


def _home_env_value(name: str) -> str | None:
    if not HOME_ENV_PATH.exists():
        return None
    prefix = f"{name}="
    for line in HOME_ENV_PATH.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or not text.startswith(prefix):
            continue
        value = text[len(prefix):].strip().strip('"').strip("'")
        return value or None
    return None


def _json_safe(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def grid_of(fr) -> np.ndarray:
    a = np.array(fr.frame)
    return a[0] if a.ndim == 3 else a


def perception(grid: np.ndarray) -> dict:
    """Compact CONTRA perception summary for a policy to reason over."""
    objs = sf.objects_from_frame(grid)
    fits = sf.find_shape_fits(grid, min_fill=0.6)["fit_relationships"]
    structure = st.detect_structure(grid)
    # dedupe fit links by (plug_color, socket_color, kind)
    seen, links = set(), []
    for r in fits:
        k = (r["plug_color"], r["socket_color"], r["fit_kind"])
        if k in seen:
            continue
        seen.add(k)
        links.append({"plug_color": r["plug_color"], "socket_color": r["socket_color"],
                      "fit": r["fit_kind"], "socket_bbox": r["pocket_bbox"]})
    return {
        "n_objects": len(objs),
        "objects": [{"color": o["color"], "bbox": o["bbox"], "area": o["area"]} for o in objs[:14]],
        "shape_fit_links": links[:8],
        "has_floor": structure["has_floor"], "has_walls": structure["has_walls"],
    }


def run_episode(game_id: str, policy, max_actions: int = 250, auto_reset: bool = True) -> dict:
    arc, card, env, mode = make_env(game_id)
    result = None
    close_payload = None
    try:
        fr = env.reset()
        best_levels = int(getattr(fr, "levels_completed", 0) or 0)
        history = []
        policy_errors = []
        resets = 0
        for t in range(max_actions):
            grid = grid_of(fr)
            state = str(getattr(fr, "state", ""))
            if "GAME_OVER" in state and auto_reset:
                fr = env.step(_ACT[0])  # RESET to retry
                resets += 1
                continue
            obs = {"grid": grid, "levels_completed": int(getattr(fr, "levels_completed", 0) or 0),
                   "state": state, "available_actions": list(getattr(fr, "available_actions", []) or []),
                   "perception": perception(grid)}
            try:
                aid, x, y = policy(obs, history)
            except Exception as exc:
                policy_errors.append({"step": int(t), "error": repr(exc)})
                aid, x, y = 0, None, None
            act = _ACT.get(int(aid), GameAction.RESET)
            data = {"x": int(x), "y": int(y)} if int(aid) == 6 and x is not None else None
            fr = env.step(act, data=data)
            lv = int(getattr(fr, "levels_completed", 0) or 0)
            best_levels = max(best_levels, lv)
            history.append({"action": int(aid), "x": x, "y": y, "levels": lv})
            if "WIN" in str(getattr(fr, "state", "")):
                break
        result = {"game_id": game_id, "levels_completed": best_levels, "actions": len(history),
                  "resets": resets, "final_state": str(getattr(fr, "state", "")),
                  "arc_mode": mode, "arc_operation_mode": str(getattr(arc, "operation_mode", "")),
                  "scorecard_id": str(card), "policy_error_count": len(policy_errors),
                  "policy_errors": policy_errors[:8]}
        export_trace = getattr(policy, "export_trace", None)
        if callable(export_trace):
            result["policy_trace"] = export_trace()
        return result
    finally:
        try:
            close_payload = arc.close_scorecard(card)
        except Exception:
            close_payload = None
        if result is not None:
            result["scorecard_close"] = _json_safe(close_payload)


# ----------------------------------------------------------------- policies
def random_policy(seed_state=[0]):
    rng_a = [1, 2, 3, 4, 5]

    def policy(obs, history):
        # deterministic-ish pseudo-random without Math.random restrictions
        seed_state[0] = (seed_state[0] * 1103515245 + 12345) & 0x7fffffff
        r = seed_state[0]
        avail = obs["available_actions"] or [1, 2, 3, 4, 5, 6]
        a = avail[r % len(avail)]
        if a == 6:
            return 6, (r >> 3) % 64, (r >> 9) % 64
        return a, None, None
    return policy


def lfm_policy():
    """Per-step policy driven by the local LFM2.5 model over CONTRA perception."""
    import reasoner_lfm_client as lfm
    SYS = ("You play an ARC-AGI-3 grid puzzle. You are given a perception summary "
           "from a vision system (objects, and shape-fit links where a 'plug' shape "
           "fits into a 'socket' = a likely goal connection). Choose ONE next action "
           "to make progress. Actions 1-5 are discrete; action 6 is a click needing "
           "x,y (0-63). Reply STRICT JSON only.")

    def policy(obs, history):
        p = obs["perception"]
        recent = [h["action"] for h in history[-8:]]
        prompt = (
            f"levels_completed={obs['levels_completed']} available_actions={obs['available_actions']}\n"
            f"objects(color,bbox,area)={[(o['color'], o['bbox'], o['area']) for o in p['objects']]}\n"
            f"shape_fit_links(plug->socket)={p['shape_fit_links']}\n"
            f"floor={p['has_floor']} walls={p['has_walls']} recent_actions={recent}\n\n"
            'Respond JSON: {"action": <id in available>, "x": <0-63 or null>, "y": <0-63 or null>}'
        )
        try:
            parsed, _ = lfm.ask_json(prompt, system=SYS, retries=1)
            a = int(parsed.get("action"))
            return a, parsed.get("x"), parsed.get("y")
        except Exception:
            return (obs["available_actions"] or [1])[0], None, None
    return policy


def _largest_component_centroid(grid: np.ndarray) -> tuple[int, int]:
    """Pixel-only fallback click target: centroid of the largest foreground object."""
    try:
        objs = sf.objects_from_frame(grid)
    except Exception:
        objs = []
    if not objs:
        h, w = np.asarray(grid).shape[:2]
        return int(w // 2), int(h // 2)
    best = max(objs, key=lambda o: o.get("area", 0))
    x0, y0, x1, y1 = best["bbox"]
    return int((x0 + x1) // 2), int((y0 + y1) // 2)


def _resolve_click_xy(top_row: dict, grid: np.ndarray) -> tuple[int, int]:
    """Resolve a select_object click (0-63) from the chosen recommendation.

    Prefers the row's pixel-derived target_override.click_xy; otherwise falls
    back to the largest-foreground-object centroid. Never uses engine state.
    """
    tov = top_row.get("target_override") or {}
    cxy = tov.get("click_xy")
    if isinstance(cxy, (list, tuple)) and len(cxy) == 2:
        return int(round(float(cxy[0]))), int(round(float(cxy[1])))
    return _largest_component_centroid(grid)


_PRIM_TO_ACTION_ID = {
    "move_up": 1, "move_down": 2, "move_left": 3, "move_right": 4,
    "activate_flow": 5, "no_selected_motion": 1, "select_object": 6,
}


def contra_policy(game_id: str = "sp80-589a99af"):
    """Drive the SDK with the full CONTRA value-scoring controller head.

    This is the missing bridge: it runs `run_contra_view` over each SDK frame to
    build the pixel CONTRA packet (PortBand lineup / ExploreBand probe queue /
    action-semantics ledger), feeds it to `build_contra_recommendation_packet`
    (the value_scored_signal_modulation ranking layer), and maps the top-ranked
    primitive to an SDK action.

    Boundary, stated honestly: the SP80 controller normally feeds the ranker
    `before_state` and `goal_condition` from privileged engine internals
    (selected sprite, tagged bucket objects). The SDK exposes neither, so here:
      - `selected` is tracked from our OWN click actions (action memory), not
        engine ground truth;
      - `goal_condition` is left minimal, because its only consumer
        (`goal_motion_candidate`) never wins the value sort (0/1488 decisions);
      - the click target is the row's pixel-derived click_xy, else a foreground
        centroid.
    The decision-bearing signal therefore comes entirely from the pixel packet.
    """
    from contra_head import run_contra_view
    from sp80_player_harness import build_contra_recommendation_packet

    ghost_enabled = False
    mapper = None

    st_ = {"i": 0, "selected": False, "selected_xy": None,
           "mem": [], "prev_levels": 0, "last_top": None,
           "prev_grid": None, "prev_packet": None, "prev_before_state": None,
           "prev_primitive": None}

    def policy(obs, history):
        grid = np.asarray(obs["grid"])
        levels = int(obs.get("levels_completed") or 0)
        # A drop in levels_completed means the env reset after a GAME_OVER; clear
        # per-episode selection/transition state so we don't observe a spurious
        # cross-reset transition or carry a stale selection into the new attempt.
        if levels < st_["prev_levels"]:
            st_.update({"selected": False, "selected_xy": None, "last_top": None,
                        "prev_grid": None, "prev_packet": None,
                        "prev_before_state": None, "prev_primitive": None})
        levels_delta = max(0, levels - st_["prev_levels"])

        # Record the previous action's outcome into action_memory (penalty layer
        # reads levels_completed_delta + primitive history from this).
        if st_["last_top"] is not None:
            st_["mem"].append({
                "action_index": st_["i"] - 1,
                "primitive": st_["last_top"].get("primitive"),
                "case_id": f"episode_action_{st_['i'] - 1:03d}",
                "selected_recommendation": st_["last_top"],
                "levels_completed_delta": levels_delta,
                "goal_progress": {"distance_delta": None},
                "selected_coord_delta": None,
                "selected_display_coord_delta": None,
                "status": "executed",
            })

        contra_packet = run_contra_view(
            raw_frame=grid, game_id=game_id, level_index=levels,
            mode="episode_before_action", ghost_context=None,
        )

        st_["prev_levels"] = levels
        before_state = {"selected": st_["selected"],
                        "selected_display_coord": st_["selected_xy"],
                        "levels_completed": levels}
        goal_condition = {"schema": "codexlab.sp80_goal_condition.v1",
                          "condition": "pour_into_buckets", "target_label": "bucket",
                          "nearest_goal": None, "visible_goal_objects": []}
        rec = build_contra_recommendation_packet(
            level_index=levels, action_index=st_["i"],
            before_state=before_state, goal_condition=goal_condition,
            contra_packet=contra_packet, action_memory=st_["mem"],
            ghost_mechanic_map=mapper.map() if ghost_enabled and mapper is not None else None,
        )
        rows = rec.get("ranked_recommendations") or []
        top = rows[0] if rows else {"primitive": "activate_flow"}
        prim = top.get("primitive", "activate_flow")

        st_["last_top"] = top
        st_["prev_grid"] = grid
        st_["prev_packet"] = contra_packet
        st_["prev_before_state"] = before_state
        st_["prev_primitive"] = prim
        st_["i"] += 1
        if ghost_enabled and mapper is not None:
            mapper.save()  # persist the past engine across steps and runs

        aid = _PRIM_TO_ACTION_ID.get(prim, 5)
        if aid == 6:
            x, y = _resolve_click_xy(top, grid)
            st_["selected"] = True
            st_["selected_xy"] = [x, y]
            return 6, x, y
        return aid, None, None

    return policy


def _objs_by_color(grid):
    out = {}
    for o in sf.objects_from_frame(grid):
        out.setdefault(o["color"], []).append(o)
    return out


def _centroid_xy(obj):
    a = np.asarray(obj["cells"]).reshape(-1, 2)
    return int(round(a[:, 1].mean())), int(round(a[:, 0].mean()))  # (x, y)


def imagine_policy(goal_color: int = 11, source_color: int = 9, move_budget: int = 8):
    """Plan-then-act loop with fallback (Phase 4).

    Uses contra.imagine() to drive ONLY when there is a confident confluence
    opportunity (>= 2 goals + a source, plan picks a >=2-goal reroute) -- i.e. the
    L1 case the reactive head structurally misses. Everywhere else (e.g. L0's
    paint-the-platform) it defers to the reactive contra policy that already
    solves it. Re-imagines each step; selection tracked from our own clicks; state
    resets across GAME_OVER->RESET.
    """
    from contra_head import run_contra_view, imagine

    inner = contra_policy()
    st_ = {"selected": False, "prev_levels": 0, "moves": 0}

    def policy(obs, history):
        grid = np.asarray(obs["grid"])
        levels = int(obs.get("levels_completed") or 0)
        if levels < st_["prev_levels"]:
            st_.update({"selected": False, "moves": 0})
        st_["prev_levels"] = levels

        by_color = _objs_by_color(grid)
        goals = sorted(by_color.get(goal_color, []), key=lambda o: o["bbox"][0])
        src = max(by_color.get(source_color, []), key=lambda o: o["area"], default=None)
        # L0 is solved by the reactive paint-the-platform path; only take over from
        # level 1 onward, where the reactive head structurally stalls on confluence.
        if levels < 1 or len(goals) < 2 or src is None:
            return inner(obs, history)  # defer to reactive

        actor_mask = np.zeros(grid.shape, dtype=bool)
        for y, x in np.asarray(src["cells"]).reshape(-1, 2):
            actor_mask[y, x] = True
        cp = run_contra_view(raw_frame=grid, game_id="sp80-589a99af", level_index=levels,
                             mode="episode_before_action", ghost_context=None)
        goal_objs = [{"bbox": [o["bbox"][1], o["bbox"][0], o["bbox"][3], o["bbox"][2]],
                      "object_id": o["id"]} for o in goals]
        plan = imagine(grid, cp, goal_objects=goal_objs, actor_mask=actor_mask)

        if plan["max_confluence_available"] < 2 or plan["phase"] != "reroute":
            return inner(obs, history)  # not a confident confluence plan -> reactive

        covered, n_goals = plan["covered_now"], plan["goal_count"]
        if not st_["selected"]:
            x, y = _centroid_xy(src)
            st_["selected"] = True
            return 6, x, y
        if covered >= n_goals or st_["moves"] >= move_budget:
            return 5, None, None  # aligned (or budget spent) -> pour, let execution judge
        for prim in plan["route"]:
            if prim.startswith("move_"):
                st_["moves"] += 1
                return _PRIM_TO_ACTION_ID[prim], None, None
        return 5, None, None

    return policy


def tufa_action_policy(solution_root: str | None = None, game_id: str = "sp80-589a99af"):
    """Run the downloaded Tufa action learner through this local/offline harness."""
    root = Path(solution_root or os.environ.get("TUFA_ARC3_SOLUTION", "/tmp/ARC3-solution")).resolve()
    agents_root = root / "ARC-AGI-3-Agents"
    if not (root / "custom_agents" / "action.py").exists() or not agents_root.exists():
        raise RuntimeError(f"Tufa ARC3 solution not found at {root}")
    for entry in (str(root), str(agents_root)):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    from custom_agents.action import Action  # noqa: E402
    from agents.structs import FrameData, GameAction as TufaAction, GameState as TufaState  # noqa: E402

    agent = Action(
        card_id="offline-local-card",
        game_id=game_id,
        agent_name="action",
        ROOT_URL="http://127.0.0.1:9",
        record=False,
    )
    frames = []

    def to_tufa_state(value: str):
        text = str(value)
        for state in TufaState:
            if state.value in text or state.name in text:
                return state
        return TufaState.NOT_FINISHED

    def to_tufa_actions(values):
        out = []
        for value in values or []:
            try:
                out.append(TufaAction.from_id(int(getattr(value, "value", value))))
            except Exception:
                continue
        return out

    def policy(obs, history):
        grid = np.asarray(obs["grid"], dtype=int)
        if grid.shape != (64, 64):
            padded = np.zeros((64, 64), dtype=int)
            h = min(64, grid.shape[0])
            w = min(64, grid.shape[1])
            padded[:h, :w] = grid[:h, :w]
            grid = padded
        latest = FrameData(
            game_id=game_id,
            frame=[grid.tolist()],
            state=to_tufa_state(obs.get("state")),
            score=int(obs.get("levels_completed") or 0),
            available_actions=to_tufa_actions(obs.get("available_actions")),
        )
        action = agent.choose_action(frames, latest)
        frames.append(latest)
        agent.action_counter += 1
        if action is TufaAction.ACTION6:
            data = action.action_data
            return 6, int(data.x), int(data.y)
        return int(action.value), None, None

    return policy


def tufa_contra_policy(
    solution_root: str | None = None,
    game_id: str = "sp80-589a99af",
    fallback_burst_actions: int = 8,
    same_frame_threshold: int = 3,
    same_action_threshold: int = 4,
    no_progress_threshold: int = 32,
):
    """Tufa-first policy with bounded CONTRA fallback on observed stalls.

    This wrapper calls Tufa on every frame so its online learner stays
    synchronized. CONTRA may override the executed action only inside a short
    stall-triggered burst, using current-frame pixel evidence only.
    """

    tufa = tufa_action_policy(solution_root=solution_root, game_id=game_id)
    st_ = {
        "prev_frame_hash": None,
        "same_frame_hash_count": 0,
        "prev_levels": None,
        "actions_since_level_gain": 0,
        "last_tufa_candidate": None,
        "same_action_no_gain_count": 0,
        "burst_remaining": 0,
        "burst_id": 0,
        "burst_index": 0,
        "contra": None,
        "ledger": [],
    }

    def available_action_ids(values):
        ids = set()
        for value in values or []:
            try:
                ids.add(int(getattr(value, "value", value)))
            except Exception:
                continue
        return ids

    def frame_hash(grid):
        arr = np.asarray(grid, dtype=np.int16)
        return hashlib.sha256(arr.tobytes()).hexdigest()

    def candidate(aid, x, y):
        return (int(aid), None if x is None else int(x), None if y is None else int(y))

    def legal(action, legal_ids):
        aid, x, y = action
        if legal_ids and aid not in legal_ids:
            return False
        if aid == 6 and (x is None or y is None):
            return False
        return aid in _ACT

    def rejection_reason(action, legal_ids):
        if action is None:
            return "contra_exception_or_none"
        aid, x, y = action
        if aid not in _ACT:
            return "unknown_action"
        if legal_ids and aid not in legal_ids:
            return "action_not_available"
        if aid == 6 and (x is None or y is None):
            return "missing_click_xy"
        return None

    def start_burst(trigger):
        st_["burst_id"] += 1
        st_["burst_index"] = 0
        st_["burst_remaining"] = max(1, int(fallback_burst_actions))
        st_["contra"] = contra_policy(game_id=game_id)
        return trigger

    def end_burst():
        st_["burst_remaining"] = 0
        st_["burst_index"] = 0
        st_["contra"] = None

    def policy(obs, history):
        grid = np.asarray(obs["grid"])
        levels = int(obs.get("levels_completed") or 0)
        digest = frame_hash(grid)
        legal_ids = available_action_ids(obs.get("available_actions"))

        tufa_action = candidate(*tufa(obs, history))
        prev_levels = st_["prev_levels"]
        level_gain = prev_levels is not None and levels > int(prev_levels)
        if level_gain:
            st_["actions_since_level_gain"] = 0
            st_["same_action_no_gain_count"] = 0
            end_burst()
        else:
            st_["actions_since_level_gain"] += 1

        if digest == st_["prev_frame_hash"]:
            st_["same_frame_hash_count"] += 1
        else:
            st_["same_frame_hash_count"] = 1

        if tufa_action == st_["last_tufa_candidate"] and not level_gain:
            st_["same_action_no_gain_count"] += 1
        else:
            st_["same_action_no_gain_count"] = 1

        trigger = None
        if st_["burst_remaining"] <= 0:
            if st_["same_frame_hash_count"] >= int(same_frame_threshold):
                trigger = start_burst("same_frame_hash")
            elif st_["same_action_no_gain_count"] >= int(same_action_threshold):
                trigger = start_burst("same_action_no_level_gain")
            elif st_["actions_since_level_gain"] >= int(no_progress_threshold):
                trigger = start_burst("long_no_progress")

        executed = tufa_action
        source = "tufa"
        contra_action = None
        contra_error = None
        contra_reject_reason = None
        if st_["burst_remaining"] > 0 and st_["contra"] is not None:
            try:
                contra_action = candidate(*st_["contra"](obs, history))
            except Exception as exc:  # noqa: BLE001 - trace why fallback did not execute.
                contra_action = None
                contra_error = repr(exc)
            if contra_action is not None and legal(contra_action, legal_ids):
                executed = contra_action
                source = "contra_fallback"
                st_["burst_remaining"] -= 1
                st_["burst_index"] += 1
                if st_["burst_remaining"] <= 0:
                    end_burst()
            else:
                contra_reject_reason = rejection_reason(contra_action, legal_ids)
                end_burst()

        record = {
            "source": source,
            "tufa_candidate": {"action": tufa_action[0], "x": tufa_action[1], "y": tufa_action[2]},
            "contra_candidate": (
                {"action": contra_action[0], "x": contra_action[1], "y": contra_action[2]}
                if contra_action is not None else None
            ),
            "contra_error": contra_error,
            "contra_reject_reason": contra_reject_reason,
            "executed_action": {"action": executed[0], "x": executed[1], "y": executed[2]},
            "fallback_trigger": trigger,
            "fallback_burst_id": st_["burst_id"] if source == "contra_fallback" else None,
            "fallback_burst_index": st_["burst_index"] if source == "contra_fallback" else None,
            "before_frame_sha256": digest,
            "levels_before": levels,
            "same_frame_hash_count": st_["same_frame_hash_count"],
            "same_action_no_gain_count": st_["same_action_no_gain_count"],
            "actions_since_level_gain": st_["actions_since_level_gain"],
        }
        st_["ledger"].append(record)
        policy.last_action_source = source
        policy.last_action_record = record
        st_["prev_frame_hash"] = digest
        st_["prev_levels"] = levels
        st_["last_tufa_candidate"] = tufa_action
        return executed

    def export_trace():
        return {
            "policy": "tufa_contra",
            "fallback_burst_actions": int(fallback_burst_actions),
            "fallback_action_count": sum(1 for row in st_["ledger"] if row.get("source") == "contra_fallback"),
            "burst_count": int(st_["burst_id"]),
            "route_or_replay_used": False,
            "human_trace_used": False,
            "coordinate_script_used": False,
            "truth_boundary": "tufa_primary_with_bounded_contra_current_frame_fallback_not_solve_truth",
            "ledger": st_["ledger"],
        }

    policy.last_action_source = "tufa"
    policy.export_trace = export_trace
    return policy


def megarexus_cnn_policy():
    """Run the synthesis_lab Megarexus CNN checkpoint through the ARC harness."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from synthesis_lab.orchestration.megarexus_live_policy import megarexus_arc_policy

    return megarexus_arc_policy()


POLICIES = {
    "random": random_policy,
    "lfm": lfm_policy,
    "contra": contra_policy,
    "imagine": imagine_policy,
    "tufa_action": tufa_action_policy,
    "tufa_contra": tufa_contra_policy,
    "megarexus_cnn": megarexus_cnn_policy,
}

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game", nargs="?", default="sp80-589a99af")
    parser.add_argument("max_actions", nargs="?", type=int, default=250)
    parser.add_argument("policy", nargs="?", choices=sorted(POLICIES), default="random")
    parser.add_argument("--out", type=Path, default=None)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    res = run_episode(args.game, POLICIES[args.policy](), max_actions=args.max_actions)
    res["policy"] = args.policy
    text = json.dumps(res, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
