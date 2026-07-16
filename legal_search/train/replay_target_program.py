#!/usr/bin/env python3
"""Replay one recorded target program exactly under the offline environment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _frame(response: Any) -> np.ndarray:
    value = response.frame
    if isinstance(value, list):
        value = value[-1]
    array = np.asarray(value)
    if array.ndim == 3:
        array = array[-1]
    if array.ndim != 2:
        raise ValueError(f"expected 2-D frame, got {array.shape}")
    return np.asarray(array, dtype=np.int16)


def _frame_hash(frame: np.ndarray) -> str:
    array = np.ascontiguousarray(frame)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _available(response: Any) -> list[int]:
    values: list[int] = []
    for raw in getattr(response, "available_actions", []) or [1, 2, 3, 4, 5, 6]:
        try:
            aid = int(getattr(raw, "value", raw))
        except (TypeError, ValueError):
            continue
        if 1 <= aid <= 6:
            values.append(aid)
    return sorted(set(values or [1, 2, 3, 4, 5, 6]))


def load_target_program(
    manifest_path: Path,
    *,
    game_id: str,
    target_index: int,
) -> tuple[list[list[int | None]], dict[str, Any]]:
    """Load and strictly validate one exact target-program receipt."""

    path = manifest_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"target manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    try:
        raw_program = manifest["search"]["per_game"][game_id]["target_programs"][target_index]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(
            f"target program {target_index} for {game_id!r} is absent from {path}"
        ) from exc
    if not isinstance(raw_program, list) or not raw_program:
        raise ValueError("target program must be a non-empty list")

    program: list[list[int | None]] = []
    for index, raw in enumerate(raw_program):
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise ValueError(f"action {index} must be [action_id, x, y]")
        aid, x, y = raw
        try:
            aid = int(aid)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"action {index} has invalid action id") from exc
        if aid not in range(1, 7):
            raise ValueError(f"action {index} has out-of-range action id {aid}")
        if aid == 6:
            if x is None or y is None:
                raise ValueError(f"click action {index} is missing coordinates")
            try:
                x, y = int(x), int(y)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"click action {index} has invalid coordinates") from exc
            if not (0 <= x < 64 and 0 <= y < 64):
                raise ValueError(f"click action {index} is outside the 64x64 action grid")
        else:
            if x is not None or y is not None:
                raise ValueError(f"non-click action {index} must have null coordinates")
            x, y = None, None
        program.append([aid, x, y])

    return program, {
        "path": str(path),
        "sha256": _sha256(path),
        "game_id": game_id,
        "target_index": int(target_index),
        "program_sha256": _canonical_hash(program),
        "program_actions": len(program),
    }


def replay(args: argparse.Namespace) -> dict[str, Any]:
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine import GameAction

    program, source = load_target_program(
        Path(args.manifest), game_id=args.game_id, target_index=args.target_index
    )
    env_dir = Path(args.environment_dir).expanduser().resolve()
    action_map = {aid: getattr(GameAction, f"ACTION{aid}") for aid in range(1, 7)}
    episodes: list[dict[str, Any]] = []

    for episode_index in range(args.episodes):
        arcade = Arcade(
            arc_api_key="anonymous",
            environments_dir=str(env_dir),
            operation_mode=OperationMode.OFFLINE,
        )
        scorecard = arcade.open_scorecard()
        action_log: list[dict[str, Any]] = []
        stop_reason = "program_exhausted"
        error: str | None = None
        levels = 0
        state = ""
        initial_frame_hash = ""
        final_frame_hash = ""
        try:
            env = arcade.make(args.game_id, scorecard_id=scorecard)
            response = env.reset()
            frame = _frame(response)
            levels = int(getattr(response, "levels_completed", 0) or 0)
            state = str(getattr(response, "state", ""))
            initial_frame_hash = _frame_hash(frame)
            final_frame_hash = initial_frame_hash
            for action_index, (aid, x, y) in enumerate(program):
                available = _available(response)
                if int(aid) not in available:
                    stop_reason = "recorded_action_unavailable"
                    error = f"action {aid} unavailable at recorded index {action_index}"
                    break
                before_hash = _frame_hash(frame)
                levels_before = levels
                payload = {"x": int(x), "y": int(y)} if int(aid) == 6 else None
                try:
                    response = env.step(action_map[int(aid)], data=payload)
                except Exception as exc:
                    stop_reason = "step_error"
                    error = f"{type(exc).__name__}: {exc}"
                    break
                frame = _frame(response)
                levels = int(getattr(response, "levels_completed", 0) or 0)
                state = str(getattr(response, "state", ""))
                final_frame_hash = _frame_hash(frame)
                action_log.append(
                    {
                        "index": action_index,
                        "action": [int(aid), x, y],
                        "available_actions": available,
                        "levels_before": levels_before,
                        "levels_after": levels,
                        "state_after": state,
                        "frame_before_sha256": before_hash,
                        "frame_after_sha256": final_frame_hash,
                    }
                )
                if levels >= args.target_level:
                    stop_reason = "target_level_reached"
                    break
                upper = state.upper()
                if "GAME_OVER" in upper or "WIN" in upper:
                    stop_reason = "terminal_before_target"
                    break
        except Exception as exc:
            stop_reason = "environment_error"
            error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                arcade.close_scorecard(scorecard)
            except Exception:
                pass

        success = levels >= args.target_level
        episodes.append(
            {
                "episode": episode_index,
                "scorecard_id": str(scorecard),
                "success": success,
                "levels_final": levels,
                "state_final": state,
                "program_actions_available": len(program),
                "actions_executed": len(action_log),
                "stop_reason": stop_reason,
                "error": error,
                "initial_frame_sha256": initial_frame_hash,
                "final_frame_sha256": final_frame_hash,
                "action_log": action_log,
            }
        )

    successes = sum(int(row["success"]) for row in episodes)
    return {
        "schema": "legal_search.exact_target_replay.v1",
        "status": "controlled_complete",
        "condition": "Exact offline replay; no policy fallback, mutation, extra action, or generated substitute.",
        "source": source,
        "program": program,
        "game_id": args.game_id,
        "target_level": args.target_level,
        "episodes_requested": args.episodes,
        "episodes_completed": len(episodes),
        "successes": successes,
        "success_rate": successes / max(1, len(episodes)),
        "all_success": successes == len(episodes),
        "episodes": episodes,
        "provenance": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "runner": str(Path(__file__).resolve()),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "working_directory": os.getcwd(),
            "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "environment_dir": str(env_dir),
            "operation_mode": "OFFLINE",
            "python": platform.python_version(),
            "random_seed": None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--target-level", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--environment-dir", type=Path, default=ROOT / "environment_files")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.target_index < 0:
        parser.error("--target-index must be non-negative")
    if args.target_level < 1:
        parser.error("--target-level must be positive")
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    result = replay(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"[target_replay] game={args.game_id} success={result['successes']}/"
        f"{result['episodes_completed']} out={args.out}",
        flush=True,
    )
    return 0 if result["all_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
