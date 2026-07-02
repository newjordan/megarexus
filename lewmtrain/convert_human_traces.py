#!/usr/bin/env python3
"""LeWMTrain: Convert HHazard human traces into LeWM-training ARCTrajectories.

Reads the HHazard/arc-agi-3 Arrow cache and writes ARCTrajectory .pt shards
consumable by ``arc_lewm_pretrain.py``.

Key finding: ``state[0]`` is the 64×64 rendered frame (int, 0-15 ARC colors).
Actions are encoded as ``action_id`` (0=RESET, 1..7) + ``action_args.{x,y}``.

Episodes are split at ``level_id`` transitions. Each game file is one
human playthrough spanning multiple levels. We split on level changes
to produce manageable trajectory chunks for LeWM sequence windowing.

Usage:
  python lewmtrain/convert_human_traces.py --out lewmtrain/data/human_trajectories.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARC_GRID_SIZE = 64
ARC_NUM_COLORS = 16
RESET_ACTION = 0  # GameAction.RESET


@dataclass
class ARCStep:
    action_id: int
    action_x: int
    action_y: int
    action_complex: bool
    frame: np.ndarray
    state: str
    levels_completed: int


@dataclass
class ARCTrajectory:
    game_id: str
    win_levels: int
    steps: list[ARCStep]


def load_human_dataset(cache_dir: Path, game_id: str) -> list[dict]:
    """Load all rows for a game from the HHazard Arrow cache."""
    from datasets import load_dataset

    arrow_path = cache_dir / f"arc-agi-3-{game_id}.arrow"
    if not arrow_path.exists():
        return []

    ds = load_dataset("arrow", data_files=str(arrow_path), split="train")
    rows = []
    for i in range(len(ds)):
        row = ds[i]
        state_arr = np.array(row["state"], dtype=np.int64)
        rows.append({
            "game_id": row["game_id"],
            "level_id": int(row["level_id"]),
            "action_id": int(row["action_id"]),
            "action_x": int(row["action_args"]["x"]),
            "action_y": int(row["action_args"]["y"]),
            "state": state_arr,
            "level_done": bool(row["level_done"]),
            "frame_idx": int(row["frame_idx"]),
            "player": row.get("player", "unknown"),
        })
    return rows


def extract_frame(state_arr: np.ndarray) -> np.ndarray | None:
    """Extract the rendered 64x64 frame from state array.

    ``state`` is [channels, 64, 64] — ``state[0]`` is the rendered frame.
    """
    if state_arr.ndim == 3:
        frame = state_arr[0]
    elif state_arr.ndim == 2:
        frame = state_arr
    else:
        return None

    if frame.shape != (ARC_GRID_SIZE, ARC_GRID_SIZE):
        return None

    # Clip invalid values (some states have values slightly outside 0..15)
    if frame.min() < 0 or frame.max() >= ARC_NUM_COLORS:
        frame = np.clip(frame, 0, ARC_NUM_COLORS - 1)

    return frame.astype(np.int8)


def rows_to_trajectories(
    rows: list[dict], game_id: str
) -> list[ARCTrajectory]:
    """Group rows into per-level episodes.

    A new episode starts when ``level_id`` changes.
    This produces one trajectory per level the human played.
    """
    if not rows:
        return []

    # Already in frame_idx order
    rows.sort(key=lambda r: r["frame_idx"])

    trajectories: list[ARCTrajectory] = []
    current_steps: list[ARCStep] = []
    current_level_id = rows[0]["level_id"]
    completed_levels: set[int] = set()
    prev_level_id = current_level_id

    for row in rows:
        # Skip RESET actions (frame 0 initial observation)
        if row["action_id"] == RESET_ACTION:
            continue

        # Extract frame
        frame = extract_frame(row["state"])
        if frame is None:
            continue

        level_id = row["level_id"]

        # Detect level transition: commit current episode, start new one
        if level_id != current_level_id and current_steps:
            # Mark the last step of the previous level as having completed it
            if current_level_id > 0:
                completed_levels.add(current_level_id)

            trajectories.append(
                ARCTrajectory(
                    game_id=f"{game_id}-L{current_level_id}",
                    win_levels=len(completed_levels),
                    steps=current_steps,
                )
            )
            current_steps = []
            current_level_id = level_id

        action_id = row["action_id"]
        # Clamp invalid action IDs (some HHazard traces have -1)
        action_id = max(0, min(action_id, 7))
        is_complex = action_id == 6  # ACTION6

        # Determine step state
        if level_id > 0 and level_id != prev_level_id:
            # Just crossed into a new level — the prior action completed a level
            step_state = "NOT_FINISHED"  # Will be updated below
        else:
            step_state = "NOT_FINISHED"

        current_steps.append(
            ARCStep(
                action_id=action_id,
                action_x=row["action_x"] if is_complex else 0,
                action_y=row["action_y"] if is_complex else 0,
                action_complex=is_complex,
                frame=frame.copy(),
                state=step_state,
                levels_completed=len(completed_levels),
            )
        )

        prev_level_id = level_id

    # Don't forget the last episode
    if current_steps:
        if current_level_id > 0:
            completed_levels.add(current_level_id)
        trajectories.append(
            ARCTrajectory(
                game_id=f"{game_id}-L{current_level_id}",
                win_levels=len(completed_levels),
                steps=current_steps,
            )
        )

    return trajectories


def save_trajectories(trajectories: list[ARCTrajectory], path: Path) -> None:
    """Serialize trajectories to a .pt shard compatible with arc_agi_3."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "game_id": t.game_id,
            "win_levels": t.win_levels,
            "steps": [
                {
                    "action_id": s.action_id,
                    "action_x": s.action_x,
                    "action_y": s.action_y,
                    "action_complex": s.action_complex,
                    "frame": s.frame,
                    "state": s.state,
                    "levels_completed": s.levels_completed,
                }
                for s in t.steps
            ],
        }
        for t in trajectories
    ]
    torch.save(payload, str(path))
    print(
        f"Saved {len(trajectories)} trajectories "
        f"({sum(len(t.steps) for t in trajectories)} steps) to {path}"
    )


def compute_stats(trajectories: list[ARCTrajectory]) -> dict[str, Any]:
    """Compute summary statistics."""
    total_steps = sum(len(t.steps) for t in trajectories)
    action_counts: dict[int, int] = defaultdict(int)
    step_lengths = [len(t.steps) for t in trajectories]
    game_ids: set[str] = set()

    for t in trajectories:
        game_ids.add(t.game_id.split("-L")[0])
        for s in t.steps:
            action_counts[s.action_id] += 1

    return {
        "trajectory_count": len(trajectories),
        "unique_games": len(game_ids),
        "total_steps": total_steps,
        "step_lengths": {
            "min": min(step_lengths) if step_lengths else 0,
            "max": max(step_lengths) if step_lengths else 0,
            "mean": round(sum(step_lengths) / len(step_lengths), 1)
            if step_lengths
            else 0,
            "median": sorted(step_lengths)[len(step_lengths) // 2]
            if step_lengths
            else 0,
        },
        "action_distribution": dict(sorted(action_counts.items())),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="Output .pt file path")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT
        / "INTERLAB/data/human_trajectories/cache/HHazard___arc-agi-3"
        "/default/0.0.0/858b13d97a399557b0333d62954d094ed3845f20",
        help="Path to HHazard Arrow cache directory",
    )
    p.add_argument(
        "--game-ids", nargs="+", default=None, help="Game IDs (default: all)"
    )
    p.add_argument(
        "--stats-out", type=Path, default=None, help="Write stats JSON here"
    )
    p.add_argument(
        "--train-split", type=float, default=0.8, help="Train split fraction"
    )
    args = p.parse_args()

    cache_dir = args.cache_dir
    if not cache_dir.exists():
        raise SystemExit(f"Cache directory not found: {cache_dir}")

    available = sorted(
        f.stem.replace("arc-agi-3-", "")
        for f in cache_dir.glob("arc-agi-3-*.arrow")
    )

    if args.game_ids:
        game_ids = [g for g in args.game_ids if g in available]
        missing = set(args.game_ids) - set(available)
        if missing:
            print(f"Warning: games not in cache: {missing}")
    else:
        game_ids = available

    print(f"Loading {len(game_ids)} games from {cache_dir}")
    print(f"Games: {', '.join(game_ids)}")
    print()

    all_trajectories: list[ARCTrajectory] = []

    for gid in game_ids:
        rows = load_human_dataset(cache_dir, gid)
        if not rows:
            print(f"  {gid}: no rows")
            continue

        trajectories = rows_to_trajectories(rows, gid)
        stats = compute_stats(trajectories)
        all_trajectories.extend(trajectories)

        level_ids = sorted(set(r["level_id"] for r in rows))
        print(
            f"  {gid}: {stats['trajectory_count']:3d} traj, "
            f"{stats['total_steps']:5d} steps, "
            f"levels={level_ids}, "
            f"len_range={stats['step_lengths']['min']}-{stats['step_lengths']['max']}"
        )

    if not all_trajectories:
        raise SystemExit("No trajectories produced!")

    # Shuffle and split
    import random
    random.seed(42)
    random.shuffle(all_trajectories)

    split_idx = int(len(all_trajectories) * args.train_split)
    train_traj = all_trajectories[:split_idx]
    val_traj = all_trajectories[split_idx:]

    train_path = Path(str(args.out).replace(".pt", "_train.pt"))
    val_path = Path(str(args.out).replace(".pt", "_val.pt"))

    save_trajectories(train_traj, train_path)
    save_trajectories(val_traj, val_path)

    train_stats = compute_stats(train_traj)
    val_stats = compute_stats(val_traj)

    print(f"\nTrain: {train_stats['trajectory_count']} trajectories, "
          f"{train_stats['total_steps']} steps, "
          f"{train_stats['unique_games']} games")
    print(f"Val:   {val_stats['trajectory_count']} trajectories, "
          f"{val_stats['total_steps']} steps, "
          f"{val_stats['unique_games']} games")
    print(f"Action distribution: {train_stats['action_distribution']}")

    if args.stats_out:
        args.stats_out.parent.mkdir(parents=True, exist_ok=True)
        args.stats_out.write_text(
            json.dumps(
                {"train": train_stats, "val": val_stats},
                indent=2,
                default=str,
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
