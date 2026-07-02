#!/usr/bin/env python3
"""PufferLib scout runner for SP80 paintflow catch/redirect mechanics."""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np

from arcengine import GameAction
from sp80_base_controller import ROOT, SP80_PATH, SP80BaseController, save_frame, sha256_file


CONDITION_SCHEMA = "codexlab.puffer_paintflow_condition.v1"
RESULT_SCHEMA = "codexlab.sp80_puffer_paintflow_run.v1"
MAX_SELECTABLE = 12
MOVE_ACTIONS = ["move_up", "move_down", "move_left", "move_right"]
ACTION_NAMES = [f"select_candidate_{idx}" for idx in range(MAX_SELECTABLE)] + MOVE_ACTIONS + ["activate_flow"]
POSE_BANDIT_POLICY = "level1_pose_bandit_v0"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 - provenance should fail closed but not abort mechanics run.
        return "unknown"


def verify_html_image_refs(index_path: Path) -> dict[str, Any]:
    if not index_path.exists():
        return {
            "index_file_exists": False,
            "html_image_refs_checked": False,
            "html_image_ref_count": 0,
            "html_missing_image_ref_count": None,
            "html_missing_image_refs": [],
        }
    text = index_path.read_text(encoding="utf-8")
    refs = re.findall(r"<img src='([^']+)'", text)
    missing = [ref for ref in refs if not (index_path.parent / ref).resolve().exists()]
    return {
        "index_file_exists": True,
        "html_image_refs_checked": True,
        "html_image_ref_count": len(refs),
        "html_missing_image_ref_count": len(missing),
        "html_missing_image_refs": missing[:20],
    }


def ensure_pufferlib(path: str | None) -> dict[str, Any]:
    if path:
        sys.path.insert(0, path)
    try:
        import pufferlib
        import pufferlib.pufferl

        return {
            "available": True,
            "module_file": getattr(pufferlib, "__file__", None),
            "pufferl_file": getattr(pufferlib.pufferl, "__file__", None),
            "version": getattr(pufferlib, "__version__", None),
        }
    except Exception as exc:  # noqa: BLE001 - run artifact needs exact import failure.
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def sorted_selectables(controller: SP80BaseController, game: Any) -> list[Any]:
    return sorted(controller.selectable_sprites(game), key=lambda ref: (ref.y, ref.x, ref.name, ref.index))


def sprite_packet(sprite: Any, index: int) -> dict[str, Any]:
    return {
        "index": int(index),
        "name": str(getattr(sprite, "name", "")),
        "x": int(sprite.x),
        "y": int(sprite.y),
        "width": int(sprite.width),
        "height": int(sprite.height),
        "tags": [str(tag) for tag in (getattr(sprite, "tags", []) or [])],
    }


def install_paintflow_event_hooks(game: Any) -> None:
    """Capture catch/overflow state before SP80 clears it during flow cleanup."""
    game._paintflow_events = []

    def snapshot(label: str) -> None:
        buckets = list(game.mxdlffpzkc())
        caught = set(getattr(game, "cevwbinfgl", set()) or set())
        overflow = set(getattr(game, "onoqwewztl", set()) or set())
        game._paintflow_events.append(
            {
                "label": label,
                "caught_bucket_indices": [idx for idx, bucket in enumerate(buckets) if bucket in caught],
                "caught_bucket_count": sum(1 for bucket in buckets if bucket in caught),
                "bucket_count": len(buckets),
                "overflow": bool(getattr(game, "kfdcqkodyy", False) or overflow),
                "overflow_count": len(overflow),
                "buckets": [sprite_packet(bucket, idx) for idx, bucket in enumerate(buckets)],
                "steps_remaining": int(getattr(game, "zlhbnhpcq", 0)),
                "flow_round": int(getattr(game, "lyremoheq", 0)),
            }
        )

    original_next_level = game.next_level

    def next_level_hook() -> Any:
        snapshot("before_next_level")
        return original_next_level()

    game.next_level = next_level_hook

    if hasattr(game, "lpqbikobah"):
        original_cleanup = game.lpqbikobah

        def cleanup_hook() -> Any:
            snapshot("before_flow_cleanup")
            return original_cleanup()

        game.lpqbikobah = cleanup_hook


def bucket_metrics(game: Any, target_bucket_index: int) -> dict[str, Any]:
    buckets = list(game.mxdlffpzkc())
    events = list(getattr(game, "_paintflow_events", []) or [])
    event = events[-1] if events else None
    if event is None:
        caught = set(getattr(game, "cevwbinfgl", set()) or set())
        overflow = set(getattr(game, "onoqwewztl", set()) or set())
        target = buckets[target_bucket_index] if 0 <= target_bucket_index < len(buckets) else None
        caught_indices = [idx for idx, bucket in enumerate(buckets) if bucket in caught]
        bucket_count = len(buckets)
        overflow_flag = bool(getattr(game, "kfdcqkodyy", False) or overflow)
        overflow_count = len(overflow)
    else:
        caught_indices = [int(idx) for idx in event.get("caught_bucket_indices", [])]
        bucket_count = int(event.get("bucket_count", len(buckets)))
        target = object() if 0 <= target_bucket_index < bucket_count else None
        overflow_flag = bool(event.get("overflow"))
        overflow_count = int(event.get("overflow_count", 0))
    target_caught = bool(target is not None and int(target_bucket_index) in caught_indices)
    return {
        "bucket_count": bucket_count,
        "target_bucket_index": int(target_bucket_index),
        "target_bucket_exists": target is not None,
        "caught_bucket_count": len(caught_indices),
        "caught_bucket_indices": caught_indices,
        "target_caught": target_caught,
        "any_caught": bool(caught_indices),
        "all_buckets_caught": bool(bucket_count and len(caught_indices) == bucket_count),
        "overflow_count": overflow_count,
        "overflow": overflow_flag,
        "clean_catch": bool(caught_indices and not overflow_flag),
        "target_clean_catch": bool(target_caught and not overflow_flag),
        "all_buckets_clean_catch": bool(bucket_count and len(caught_indices) == bucket_count and not overflow_flag),
        "paintflow_event_count": len(events),
        "last_paintflow_event": event,
    }


def observation_vector(obs: Any) -> np.ndarray:
    """Frame-only observation: the rendered 64x64 frame the ARC API exposes, normalized to [0,1].

    Deliberately carries NO engine-internal extras. The earlier version concatenated 10
    privileged values read from engine objects (selected sprite `vsoxmtrhqt`, source sprites
    `qbsoyazkhk()`, bucket `mxdlffpzkc()` coords, and the `zlhbnhpcq`/`lyremoheq` counters) —
    an observation leak that let the policy "see" state the held-out ARC-AGI-3 API never
    surfaces. Removed 2026-06-16: a from-pixels learner must perceive the board from the frame
    alone. See [[no-reward-hacking-cardinal-rule]] / [[sp80-spill-model-planner]].
    """
    return np.asarray(obs.frame[-1], dtype=np.float32).reshape(-1) / 15.0


class SP80PaintflowPufferEnv:
    """Single-agent PufferLib-style environment for paintflow mechanics."""

    def __init__(
        self,
        *,
        levels: list[int],
        seed: int,
        max_actions: int,
        save_frames: int,
        out_dir: Path,
        scale: int,
    ) -> None:
        self.single_observation_space = gymnasium.spaces.Box(low=0.0, high=1.0, shape=(64 * 64,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(len(ACTION_NAMES))
        self.num_agents = 1
        self.observations = np.zeros((1, 64 * 64), dtype=np.float32)
        self.rewards = np.zeros(1, dtype=np.float32)
        self.terminals = np.zeros(1, dtype=bool)
        self.truncations = np.zeros(1, dtype=bool)
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.levels = [int(level) for level in levels]
        self.rng = random.Random(int(seed))
        self.max_actions = int(max_actions)
        self.save_frames = int(save_frames)
        self.out_dir = out_dir
        self.scale = int(scale)
        self.episode_index = -1
        self.controller: SP80BaseController | None = None
        self.game: Any = None
        self.obs: Any = None
        self.target_bucket_index = 0
        self.action_index = 0
        self.frames_saved = 0
        self.episode_rows: list[dict[str, Any]] = []
        self.action_rows: list[dict[str, Any]] = []
        self.brain_rows: list[dict[str, Any]] = []
        self.planned_actions: list[int] = []
        self.active_plan: dict[str, Any] | None = None
        self.pose_q: dict[str, float] = {}
        self.pose_visits: dict[str, int] = {}

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if seed is not None:
            self.rng.seed(int(seed))
        self.episode_index += 1
        level = self.levels[self.episode_index % len(self.levels)]
        self.controller = SP80BaseController(level_index=level)
        self.game, self.obs = self.controller.new_game()
        install_paintflow_event_hooks(self.game)
        bucket_count = len(list(self.game.mxdlffpzkc()))
        self.target_bucket_index = self.episode_index % max(1, bucket_count)
        self.action_index = 0
        self.planned_actions = []
        self.active_plan = None
        self.rewards[:] = 0.0
        self.terminals[:] = False
        self.truncations[:] = False
        self.observations[0] = observation_vector(self.obs)
        self._save_frame("reset")
        return self.observations, [{"level": level, "target_bucket_index": self.target_bucket_index}]

    def step(self, action: np.ndarray | list[int] | int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        assert self.controller is not None
        action_id = int(np.asarray(action).reshape(-1)[0])
        action_name = ACTION_NAMES[action_id] if 0 <= action_id < len(ACTION_NAMES) else "invalid"
        before = bucket_metrics(self.game, self.target_bucket_index)
        before_completed = int(getattr(self.obs, "levels_completed", 0))
        before_state = self.controller.state(self.game, self.obs)
        status = "executed"
        error = None
        reward = -0.01
        try:
            self.obs = self._execute(action_id)
        except Exception as exc:  # noqa: BLE001 - per-action artifact should record failure.
            status = "error"
            error = repr(exc)
            reward -= 0.25
        after = bucket_metrics(self.game, self.target_bucket_index)
        after_completed = int(getattr(self.obs, "levels_completed", 0))
        levels_completed_delta = after_completed - before_completed
        reward += 0.5 * max(0, int(after["caught_bucket_count"]) - int(before["caught_bucket_count"]))
        if after["target_caught"] and not before["target_caught"]:
            reward += 2.0
        if after["all_buckets_caught"] and not before["all_buckets_caught"]:
            reward += 3.0
        if after["overflow"] and not before["overflow"]:
            reward -= 2.0
        if after_completed > before_completed:
            reward += 5.0
        self.action_index += 1
        terminal = bool(after_completed > before_completed or str(getattr(self.obs, "state", "")) == "GAME_OVER")
        truncation = bool(self.action_index >= self.max_actions)
        if action_name == "activate_flow":
            terminal = True
        self.rewards[0] = float(reward)
        self.terminals[0] = terminal
        self.truncations[0] = truncation
        self.observations[0] = observation_vector(self.obs)
        frame_artifact = self._save_frame(f"action_{self.action_index:03d}_{action_name}")
        row = {
            "episode_index": int(self.episode_index),
            "level_index": int(self.controller.level_index),
            "action_index": int(self.action_index),
            "action_id": int(action_id),
            "action_name": action_name,
            "status": status,
            "error": error,
            "reward": float(reward),
            "before_metrics": before,
            "after_metrics": after,
            "terminal": terminal,
            "truncation": truncation,
            "levels_completed_delta": int(levels_completed_delta),
            "frame": frame_artifact,
        }
        after_state = self.controller.state(self.game, self.obs)
        row["before_state"] = before_state
        row["after_state"] = after_state
        self.action_rows.append(row)
        if terminal or truncation:
            self._update_active_pose_plan(after)
            self.episode_rows.append(
                {
                    "episode_index": int(self.episode_index),
                    "level_index": int(self.controller.level_index),
                    "target_bucket_index": int(self.target_bucket_index),
                    "action_count": int(self.action_index),
                    "reward_sum": float(sum(r["reward"] for r in self.action_rows if r["episode_index"] == self.episode_index)),
                    "final_metrics": after,
                    "terminal": terminal,
                    "truncation": truncation,
                    "levels_completed_delta": int(levels_completed_delta),
                    "active_plan": self.active_plan,
                }
            )
        return self.observations, self.rewards, self.terminals, self.truncations, [row]

    def _execute(self, action_id: int) -> Any:
        assert self.controller is not None
        if action_id < MAX_SELECTABLE:
            candidates = sorted_selectables(self.controller, self.game)
            if action_id >= len(candidates):
                raise RuntimeError(f"no selectable candidate {action_id}")
            obs, _click = self.controller.click_sprite(self.game, candidates[action_id])
            return obs
        primitive_idx = action_id - MAX_SELECTABLE
        if primitive_idx < len(MOVE_ACTIONS):
            return self.controller.execute_grounded_action(
                self.game,
                self.controller.ground_primitive(self.game, MOVE_ACTIONS[primitive_idx]),
            )
        return self.controller.execute_grounded_action(
            self.game,
            self.controller.ground_primitive(self.game, "activate_flow"),
        )

    def _save_frame(self, label: str) -> dict[str, Any] | None:
        if self.frames_saved >= self.save_frames:
            return None
        frame = np.asarray(self.obs.frame[-1])
        path = self.out_dir / "frames" / f"episode_{self.episode_index:03d}_{label}.png"
        self.frames_saved += 1
        return save_frame(frame, path, self.scale)

    def legal_level1_pose_candidates(self) -> list[dict[str, Any]]:
        assert self.controller is not None
        selectable = sorted_selectables(self.controller, self.game)
        if not selectable:
            return []
        selected = getattr(self.game, "vsoxmtrhqt", None)
        probe = selected if selected is not None else self.game.fbrwmvzsym()[0]
        candidates: list[dict[str, Any]] = []
        grid_width, grid_height = self.game.current_level.grid_size
        for y in range(3, int(grid_height)):
            for x in range(0, int(grid_width)):
                if not self.game.husluhmboo(probe, x, y):
                    continue
                if x + int(probe.width) > int(grid_width):
                    continue
                candidates.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "width": int(probe.width),
                        "height": int(probe.height),
                        "candidate_id": f"pose_x{x:02d}_y{y:02d}",
                    }
                )
        return candidates

    def _pose_key(self, target_bucket_index: int, pose: dict[str, Any]) -> str:
        return f"target_{int(target_bucket_index)}:{pose['candidate_id']}"

    def _choose_pose_candidate(self) -> tuple[dict[str, Any], dict[str, Any]]:
        candidates = self.legal_level1_pose_candidates()
        if not candidates:
            raise RuntimeError("no legal level-1 pose candidates")
        keys = [self._pose_key(self.target_bucket_index, pose) for pose in candidates]
        untried = [(key, pose) for key, pose in zip(keys, candidates, strict=False) if self.pose_visits.get(key, 0) == 0]
        epsilon = 0.20
        positive = [(key, pose) for key, pose in zip(keys, candidates, strict=False) if self.pose_q.get(key, 0.0) > 0.0]
        if positive and self.rng.random() >= epsilon:
            best_score = max(self.pose_q.get(key, 0.0) for key, _pose in positive)
            best = [(key, pose) for key, pose in positive if self.pose_q.get(key, 0.0) == best_score]
            key, pose = self.rng.choice(best)
            reason = "positive_q_exploit"
        elif untried:
            key, pose = self.rng.choice(untried)
            reason = "untried_pose"
        elif self.rng.random() < epsilon:
            idx = self.rng.randrange(len(candidates))
            key, pose = keys[idx], candidates[idx]
            reason = "epsilon_explore"
        else:
            best_score = max(self.pose_q.get(key, 0.0) for key in keys)
            best = [(key, pose) for key, pose in zip(keys, candidates, strict=False) if self.pose_q.get(key, 0.0) == best_score]
            key, pose = self.rng.choice(best)
            reason = "q_exploit"
        return pose, {
            "policy": POSE_BANDIT_POLICY,
            "candidate_count": len(candidates),
            "chosen_pose_key": key,
            "chosen_pose": pose,
            "choice_reason": reason,
            "q_before": float(self.pose_q.get(key, 0.0)),
            "visits_before": int(self.pose_visits.get(key, 0)),
            "epsilon": epsilon,
        }

    def _plan_to_pose_actions(self, pose: dict[str, Any]) -> list[int]:
        selectable = sorted_selectables(self.controller, self.game)
        if not selectable:
            raise RuntimeError("no selectable sprite for pose plan")
        selected = getattr(self.game, "vsoxmtrhqt", None)
        current_x = int(selected.x) if selected is not None else int(selectable[0].x)
        current_y = int(selected.y) if selected is not None else int(selectable[0].y)
        target_x = int(pose["x"])
        target_y = int(pose["y"])
        actions = [0]
        if target_x > current_x:
            actions.extend([ACTION_NAMES.index("move_right")] * (target_x - current_x))
        elif target_x < current_x:
            actions.extend([ACTION_NAMES.index("move_left")] * (current_x - target_x))
        if target_y > current_y:
            actions.extend([ACTION_NAMES.index("move_down")] * (target_y - current_y))
        elif target_y < current_y:
            actions.extend([ACTION_NAMES.index("move_up")] * (current_y - target_y))
        actions.append(ACTION_NAMES.index("activate_flow"))
        return actions

    def plan_pose_bandit_action(self) -> int:
        if not self.planned_actions:
            pose, decision = self._choose_pose_candidate()
            self.planned_actions = self._plan_to_pose_actions(pose)
            self.active_plan = {
                **decision,
                "planned_action_names": [ACTION_NAMES[action] for action in self.planned_actions],
                "planned_action_count": len(self.planned_actions),
            }
            self.brain_rows.append(
                {
                    "episode_index": int(self.episode_index),
                    "level_index": int(self.controller.level_index),
                    "target_bucket_index": int(self.target_bucket_index),
                    "decision": self.active_plan,
                    "truth_boundary": "tabular pose-bandit over primitive actions; not native PufferLib trainer policy",
                }
            )
        return self.planned_actions.pop(0)

    def _update_active_pose_plan(self, final_metrics: dict[str, Any]) -> None:
        if not self.active_plan:
            return
        key = str(self.active_plan["chosen_pose_key"])
        reward = 0.0
        if final_metrics["target_clean_catch"]:
            reward += 3.0
        if final_metrics["all_buckets_clean_catch"]:
            reward += 7.0
        elif final_metrics["target_caught"]:
            reward += 0.5
        if final_metrics["overflow"]:
            reward -= 4.0
        if not final_metrics["any_caught"]:
            reward -= 0.5
        visits = self.pose_visits.get(key, 0) + 1
        old_q = self.pose_q.get(key, 0.0)
        new_q = old_q + (reward - old_q) / visits
        self.pose_visits[key] = visits
        self.pose_q[key] = new_q
        self.active_plan["bandit_reward"] = float(reward)
        self.active_plan["q_after"] = float(new_q)
        self.active_plan["visits_after"] = int(visits)


def choose_action(env: SP80PaintflowPufferEnv, policy: str) -> int:
    assert env.controller is not None
    if policy == POSE_BANDIT_POLICY:
        return env.plan_pose_bandit_action()
    if policy == "activate_first":
        return ACTION_NAMES.index("activate_flow")
    if policy == "random_valid":
        candidates = sorted_selectables(env.controller, env.game)
        valid = list(range(min(MAX_SELECTABLE, len(candidates)))) + list(range(MAX_SELECTABLE, len(ACTION_NAMES)))
        return env.rng.choice(valid)
    raise ValueError(f"unknown policy {policy!r}")


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    count = max(1, len(episodes))
    return {
        "episode_count": len(episodes),
        "target_catch_rate": sum(1 for row in episodes if row["final_metrics"]["target_caught"]) / count,
        "any_catch_rate": sum(1 for row in episodes if row["final_metrics"]["any_caught"]) / count,
        "all_bucket_catch_rate": sum(1 for row in episodes if row["final_metrics"]["all_buckets_caught"]) / count,
        "overflow_rate": sum(1 for row in episodes if row["final_metrics"]["overflow"]) / count,
        "target_clean_catch_rate": sum(1 for row in episodes if row["final_metrics"]["target_clean_catch"]) / count,
        "all_bucket_clean_catch_rate": sum(1 for row in episodes if row["final_metrics"]["all_buckets_clean_catch"]) / count,
        "level_completion_rate": sum(1 for row in episodes if int(row.get("levels_completed_delta", 0)) > 0) / count,
        "mean_episode_reward": sum(float(row["reward_sum"]) for row in episodes) / count,
    }


def write_html(path: Path, result: dict[str, Any]) -> None:
    summary = result["summary"]
    rows = "\n".join(
        f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary.items()
    )
    frame_cards = []
    for row in result["actions"][:80]:
        frame = row.get("frame")
        if not isinstance(frame, dict) or not frame.get("path"):
            continue
        rel = os.path.relpath(ROOT / frame["path"], path.parent)
        caption = (
            f"ep {row['episode_index']} action {row['action_index']} "
            f"{row['action_name']} target={row['after_metrics']['target_caught']} "
            f"clean={row['after_metrics']['target_clean_catch']} overflow={row['after_metrics']['overflow']}"
        )
        frame_cards.append(
            "<figure>"
            f"<img src='{html.escape(rel)}' alt='{html.escape(caption)}'>"
            f"<figcaption>{html.escape(caption)}</figcaption>"
            "</figure>"
        )
    frame_deck = "\n".join(frame_cards) or "<p>No frame artifacts were saved for this run.</p>"
    pose_rows = "\n".join(
        f"<tr><td>{html.escape(row['key'])}</td><td>{row['visits']}</td><td>{row['q']:.3f}</td></tr>"
        for row in result.get("pose_q_top", [])
    )
    action_rows = "\n".join(
        f"<tr><td>{row['episode_index']}</td><td>{row['action_index']}</td><td>{html.escape(row['action_name'])}</td><td>{row['reward']:.3f}</td>"
        f"<td>{row['after_metrics']['target_caught']}</td><td>{row['after_metrics']['target_clean_catch']}</td>"
        f"<td>{row['after_metrics']['all_buckets_clean_catch']}</td><td>{row['after_metrics']['overflow']}</td></tr>"
        for row in result["actions"][:120]
    )
    raw_links = "\n".join(
        f"<li><a href='{html.escape(os.path.relpath(ROOT / artifact, path.parent))}'>{html.escape(name)}</a></li>"
        for name, artifact in result.get("artifacts", {}).items()
        if isinstance(artifact, str) and artifact.endswith((".json", ".jsonl"))
    )
    path.write_text(
        "\n".join(
            [
                "<!doctype html><html><head><meta charset='utf-8'><title>SP80 Paintflow PufferLib Groundhog</title>",
                "<style>body{font-family:system-ui,sans-serif;margin:2rem;max-width:1280px}"
                "table{border-collapse:collapse;margin:1rem 0}th,td{border:1px solid #bbb;padding:.35rem .55rem;text-align:left}"
                ".note{color:#555}.deck{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:1rem}"
                "figure{margin:0}img{width:100%;image-rendering:pixelated;border:1px solid #999}figcaption{font-size:12px;color:#333}</style></head><body>",
                "<h1>SP80 Paintflow PufferLib Groundhog</h1>",
                "<p class='note'>Controlled mechanics run over primitive SP80 actions. PufferLib import is gated, but this is not a native PufferLib trainer and not full Angel/CONTRA solve evidence.</p>",
                "<h2>Frames</h2><div class='deck'>",
                frame_deck,
                "</div>",
                "<h2>Summary</h2><table>",
                rows,
                "</table><h2>Learned Pose Values</h2><table><tr><th>pose key</th><th>visits</th><th>q</th></tr>",
                pose_rows,
                "</table><h2>First Actions</h2><table><tr><th>episode</th><th>step</th><th>action</th><th>reward</th><th>target caught</th><th>target clean</th><th>all clean</th><th>overflow</th></tr>",
                action_rows,
                "</table><h2>Raw Artifacts</h2><ul>",
                raw_links,
                "</ul></body></html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", type=Path, default=ROOT / "codexlab/conditions/puffer_paintflow_groundhog_v0.json")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--pufferlib-path", default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--policy", choices=["random_valid", "activate_first", POSE_BANDIT_POLICY], default=None)
    parser.add_argument("--scale", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    condition_path = args.condition if args.condition.is_absolute() else ROOT / args.condition
    condition = load_json(condition_path)
    if condition.get("schema") != CONDITION_SCHEMA:
        raise RuntimeError(f"bad condition schema in {condition_path}")
    pufferlib_path = args.pufferlib_path or os.environ.get("PUFFERLIB_PATH") or condition.get("pufferlib_source")
    pufferlib = ensure_pufferlib(str(pufferlib_path) if pufferlib_path else None)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or ROOT / "codexlab/runs" / f"puffer_paintflow_groundhog_v0_{stamp}"
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes = int(args.episodes if args.episodes is not None else condition["episodes"])
    max_actions = int(args.max_actions if args.max_actions is not None else condition["max_actions_per_episode"])
    seed = int(args.seed if args.seed is not None else condition["seed"])
    policy = str(args.policy or condition["policy"])
    levels = [int(level) for level in condition["levels"]]
    result_base = {
        "schema": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "cannot_run" if not pufferlib["available"] else "running",
        "condition": condition,
        "condition_path": condition_path.relative_to(ROOT).as_posix(),
        "condition_sha256": sha256_file(condition_path),
        "source_script": "codexlab/sp80_puffer_paintflow_run.py",
        "source_script_sha256": sha256_file(ROOT / "codexlab/sp80_puffer_paintflow_run.py"),
        "game_source": SP80_PATH.relative_to(ROOT).as_posix(),
        "game_source_sha256": sha256_file(SP80_PATH),
        "cwd": str(ROOT),
        "host": socket.gethostname(),
        "command": " ".join(sys.argv),
        "pufferlib": pufferlib,
        "truth_boundary": condition["truth_boundary"],
    }
    if not pufferlib["available"]:
        write_json(out_dir / "result.json", {**result_base, "summary": {}, "episodes": [], "actions": []})
        write_html(out_dir / "index.html", {**result_base, "summary": {}, "episodes": [], "actions": []})
        print(json.dumps({**result_base, "summary": {}, "episodes": [], "actions": []}, indent=2, sort_keys=True))
        return 1
    rng = random.Random(seed)
    env = SP80PaintflowPufferEnv(
        levels=levels,
        seed=seed,
        max_actions=max_actions,
        save_frames=int(condition["loader_settings"]["save_review_frames"]),
        out_dir=out_dir,
        scale=max(1, int(args.scale)),
    )
    for episode_idx in range(episodes):
        env.reset(seed=seed + episode_idx)
        while True:
            _ = rng.random()
            action = choose_action(env, policy)
            _obs, _reward, terminal, truncation, _info = env.step(action)
            if bool(terminal[0]) or bool(truncation[0]):
                break
    pose_q_top = [
        {"key": key, "q": float(value), "visits": int(env.pose_visits.get(key, 0))}
        for key, value in sorted(env.pose_q.items(), key=lambda item: (-item[1], item[0]))[:20]
    ]
    result = {
        **result_base,
        "status": "completed",
        "run_label": condition["run_label"],
        "policy": policy,
        "seed": seed,
        "episodes": env.episode_rows,
        "actions": env.action_rows,
        "brain_calls": env.brain_rows,
        "pose_q_top": pose_q_top,
        "pose_q": env.pose_q,
        "pose_visits": env.pose_visits,
        "summary": summarize(env.episode_rows),
        "artifacts": {
            "result_json": (out_dir / "result.json").relative_to(ROOT).as_posix(),
            "index_html": (out_dir / "index.html").relative_to(ROOT).as_posix(),
            "packets_jsonl": (out_dir / "packets.jsonl").relative_to(ROOT).as_posix(),
            "brain_calls_jsonl": (out_dir / "brain_calls.jsonl").relative_to(ROOT).as_posix(),
            "report_manifest_json": (out_dir / "report_manifest.json").relative_to(ROOT).as_posix(),
            "report_trace_jsonl": (out_dir / "report_trace.jsonl").relative_to(ROOT).as_posix(),
            "frames_dir": (out_dir / "frames").relative_to(ROOT).as_posix(),
        },
    }
    write_json(out_dir / "result.json", result)
    write_jsonl(out_dir / "packets.jsonl", env.action_rows)
    write_jsonl(out_dir / "brain_calls.jsonl", env.brain_rows)
    trace_rows = [
        {
            "schema": "angel.report_trace_event.v1",
            "run_id": out_dir.name,
            "event_index": idx,
            "event_type": "primitive_action",
            "level_index": row["level_index"],
            "action_invocation_id": f"episode_{row['episode_index']:03d}_action_{row['action_index']:03d}",
            "function_name": "sp80_puffer_paintflow_step",
            "function_family": "paintflow_mechanics_runner",
            "output_refs": [] if row.get("frame") is None else [row["frame"]],
            "truth_boundary": "controlled_mechanics_run_not_full_angel_report",
            "status": row["status"],
        }
        for idx, row in enumerate(env.action_rows)
    ]
    write_jsonl(out_dir / "report_trace.jsonl", trace_rows)
    manifest = {
        "schema": "angel.contra_gameplay_report_manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "report_status": "controlled_run",
        "run_id": out_dir.name,
        "game_id": "sp80-589a99af",
        "working_directory": str(ROOT),
        "git_commit": git_output(["rev-parse", "HEAD"]),
        "git_dirty": bool(git_output(["status", "--porcelain"])),
        "command": " ".join(sys.argv),
        "source_paths": [
            {"path": "codexlab/sp80_puffer_paintflow_run.py", "sha256": sha256_file(ROOT / "codexlab/sp80_puffer_paintflow_run.py")},
            {"path": SP80_PATH.relative_to(ROOT).as_posix(), "sha256": sha256_file(SP80_PATH)},
            {"path": condition_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(condition_path)},
        ],
        "raw_inputs": [{"path": condition_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(condition_path)}],
        "raw_outputs": [
            {"path": result["artifacts"]["result_json"], "sha256": sha256_file(out_dir / "result.json")},
            {"path": result["artifacts"]["packets_jsonl"], "sha256": sha256_file(out_dir / "packets.jsonl")},
            {"path": result["artifacts"]["brain_calls_jsonl"], "sha256": sha256_file(out_dir / "brain_calls.jsonl")},
            {"path": result["artifacts"]["report_trace_jsonl"], "sha256": sha256_file(out_dir / "report_trace.jsonl")},
        ],
        "slide_count": 0,
        "function_slide_count": 0,
        "action_count": len(env.action_rows),
        "oracle_sidecar_count": 0,
        "verification": {
            "image_count_matches_manifest": False,
            "index_reachable": False,
            "browser_screenshot_checked": False,
            "strict_angel_visual_contract": False,
        },
        "boundaries": [
            "Controlled SP80 paintflow mechanics run over primitive actions.",
            "PufferLib is import-gated; this is not a native PufferLib trainer result.",
            "Not full CONTRA Angel gameplay report, not solve proof beyond recorded level-completion deltas.",
            "Oracle sidecars are absent: oracle_sidecar_count is 0.",
        ],
    }
    write_html(out_dir / "index.html", result)
    manifest["verification"].update(verify_html_image_refs(out_dir / "index.html"))
    write_json(out_dir / "report_manifest.json", manifest)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
