#!/usr/bin/env python3
"""LeWMTrain: Collect random-policy trajectories from all 25 ARC-AGI-3 games.

Uses the existing ``arc_agi_3.collect_trajectories`` infrastructure with
random policy. These exploratory trajectories supplement the human traces
by covering diverse state transitions, which helps the world model learn
environment dynamics (per LeWM paper: "trajectories may be pseudo-expert
or exploratory").

Usage:
  python lewmtrain/collect_random.py --out lewmtrain/data/random_trajectories.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lewm.data.arc_agi_3 import (
    collect_trajectories,
    save_trajectories,
    load_trajectories,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="Output .pt file")
    p.add_argument(
        "--rollouts-per-game", type=int, default=4,
        help="Random rollouts per game (default: 4)",
    )
    p.add_argument(
        "--max-steps", type=int, default=200,
        help="Max steps per rollout (default: 200)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--game-ids", nargs="+", default=None)
    args = p.parse_args()

    print(f"Collecting {args.rollouts_per_game} rollouts × 25 games "
          f"(max {args.max_steps} steps each)...")

    trajectories = collect_trajectories(
        game_ids=args.game_ids,
        rollouts_per_game=args.rollouts_per_game,
        max_steps_per_rollout=args.max_steps,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    total_steps = sum(len(t.steps) for t in trajectories)
    games = sorted(set(t.game_id for t in trajectories))
    print(f"Collected {len(trajectories)} trajectories ({total_steps} steps) "
          f"across {len(games)} games")

    # Split into train/val
    import random
    random.seed(42)
    random.shuffle(trajectories)
    split = int(len(trajectories) * 0.8)

    train_path = Path(str(args.out).replace(".pt", "_train.pt"))
    val_path = Path(str(args.out).replace(".pt", "_val.pt"))

    save_trajectories(trajectories[:split], train_path)
    save_trajectories(trajectories[split:], val_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
