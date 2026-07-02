#!/usr/bin/env python3
"""LeWMTrain: Build staged-player dataset and convert to ARCTrajectory format.

Uses the existing ``build_staged_dataset.py`` infrastructure to generate
expert / random / perturbed episodes, then converts them to the
ARCTrajectory .pt shard format consumed by ``lewmtrain/train.py``.

This bridges the 3,043-mechanic curriculum to LeWM pretraining with zero
domain-gap: frames are 64×64 int8 (0-15), actions are ACTION1..ACTION7,
and the format matches the human-trajectory shards exactly.

Usage:
  # Generate 50k expert episodes (fast, ~10 min on CPU):
  python lewmtrain/build_staged_trajectories.py \\
      --seeds-per-game 16 --mix expert:1.0 \\
      --out lewmtrain/data/staged_trajectories.pt

  # Full mix (expert + random + perturbed, with art augmentation):
  python lewmtrain/build_staged_trajectories.py \\
      --seeds-per-game 8 --mix expert:0.3,random:0.5,perturbed:0.2 \\
      --include-art --include-randomize \\
      --out lewmtrain/data/staged_trajectories.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from loom_layers.staged_player import (
    ArcAestheticMechanic,
    PRESET_NAMES,
    RandomizedMechanic,
    STAGED,
    build_preset,
    parse_spec,
    registry,
)
from loom_layers.staged_player.engine import Mechanic

# Lightweight Episode dataclass (mirrors build_staged_dataset.py)
from dataclasses import dataclass


@dataclass
class EpisodeSpec:
    mechanic: str
    seed: int
    source: str
    art: bool = False
    randomize: bool = False
    preset_name: str | None = None


@dataclass
class Episode:
    spec: EpisodeSpec
    frames: list[np.ndarray]
    actions: list[int]
    coords: list[tuple[int, int]]
    won: bool


def _make_mechanic(
    mech_cls: type[Mechanic],
    *,
    randomize: bool,
    art: bool,
    seed: int,
    preset_name: str = "arcprize_native",
) -> Mechanic:
    """Wrap mechanic with randomization / art."""
    mech = mech_cls()
    if preset_name:
        if preset_name not in PRESET_NAMES:
            raise ValueError(
                f"unknown preset {preset_name!r}; expected one of {PRESET_NAMES}"
            )
        return build_preset(preset_name, mech, seed)

    if randomize:
        mech = RandomizedMechanic(mech, seed=seed)
    if art:
        mech = ArcAestheticMechanic(mech, seed=seed)
    return mech


def _init_state(mech: Mechanic, spec, seed: int) -> dict:
    try:
        return mech.init_state(spec, seed=seed)
    except TypeError:
        return mech.init_state(spec)


def _scripted_solve(mech: Mechanic, spec, seed: int):
    try:
        return mech.scripted_solve(spec, seed=seed)
    except TypeError:
        return mech.scripted_solve(spec)


def rollout(
    mech: Mechanic,
    spec,
    seed: int,
    source: str,
    perturb_prob: float = 0.0,
    max_actions: int = 500,
    art: bool = False,
    randomize: bool = False,
    preset_name: str = "arcprize_native",
) -> Episode | None:
    """Run one episode and return an Episode.

    Mirrors the rollout logic in build_staged_dataset.py.
    """
    rng = np.random.RandomState(seed)

    state = _init_state(mech, spec, seed)
    frames: list[np.ndarray] = [mech.render(state).copy()]
    actions: list[int] = []
    coords: list[tuple[int, int]] = []

    if source == "expert":
        solution = _scripted_solve(mech, spec, seed)
    elif source == "random":
        solution_length = rng.randint(10, max_actions)
        solution = [int(rng.randint(1, 8)) for _ in range(solution_length)]
    elif source == "perturbed":
        expert_sol = mech.scripted_solve(spec, seed=seed)
        solution = []
        for a in expert_sol:
            if rng.random() < perturb_prob:
                solution.append(int(rng.randint(1, 8)))
            solution.append(a)
        if not solution:
            solution = expert_sol
    else:
        raise ValueError(f"Unknown source: {source}")

    for act in solution:
        action_id: int
        coord: tuple | None = None

        if isinstance(act, tuple):
            action_id, y, x = act
            coord = (y, x)
        else:
            action_id = int(act)

        if not (1 <= action_id <= 7):
            continue

        state = mech.step(state, action_id, coord)
        frames.append(mech.render(state).copy())
        actions.append(action_id)
        coords.append(coord if coord else (0, 0))

        if mech.is_won(state):
            break

        if len(actions) >= max_actions:
            break

    won = mech.is_won(state)

    return Episode(
        spec=EpisodeSpec(
            mechanic=spec.title,
            seed=seed,
            source=source,
            art=art,
            randomize=randomize,
            preset_name=preset_name,
        ),
        frames=frames,
        actions=actions,
        coords=coords,
        won=won,
    )


def episode_to_trajectory(episode: Episode, mechanic_id: str) -> dict:
    """Convert an Episode to an ARCTrajectory-compatible dict."""
    steps = []
    for i, action_id in enumerate(episode.actions):
        frame = episode.frames[i]  # frame before action
        if frame.shape != (64, 64):
            continue
        # Ensure int8 and valid range
        frame = np.clip(frame, 0, 15).astype(np.int8)
        coord = episode.coords[i] if i < len(episode.coords) else (0, 0)
        steps.append({
            "action_id": action_id,
            "action_x": coord[1] if action_id == 6 else 0,
            "action_y": coord[0] if action_id == 6 else 0,
            "action_complex": action_id == 6,
            "frame": frame,
            "state": "WIN" if (i == len(episode.actions) - 1 and episode.won) else "NOT_FINISHED",
            "levels_completed": 1 if episode.won else 0,
        })

    return {
        "game_id": f"staged-{mechanic_id}-{episode.spec.source}-s{episode.spec.seed}",
        "win_levels": 1 if episode.won else 0,
        "steps": steps,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="Output .pt file")
    p.add_argument(
        "--seeds-per-game", type=int, default=4,
        help="Number of seeds per mechanic (default: 4)",
    )
    p.add_argument(
        "--mix", type=str, default="expert:0.5,random:0.3,perturbed:0.2",
        help="Source mix (e.g. expert:0.5,random:0.3,perturbed:0.2)",
    )
    p.add_argument("--include-art", action="store_true")
    p.add_argument("--include-randomize", action="store_true")
    p.add_argument("--perturb-prob", type=float, default=0.1)
    p.add_argument("--max-actions", type=int, default=500)
    p.add_argument("--preset", type=str, default="arcprize_native")
    p.add_argument(
        "--mechanics", nargs="+", default=None,
        help="Mechanic IDs to include (e.g., 'movement/test_01_single_step')",
    )
    p.add_argument("--limit-mechanics", type=int, default=None,
                   help="Limit number of mechanics (for quick testing)")
    p.add_argument("--train-split", type=float, default=0.8)
    args = p.parse_args()

    # Parse mix
    mix = {}
    for part in args.mix.split(","):
        src, frac = part.split(":")
        mix[src.strip()] = float(frac)
    total = sum(mix.values())
    mix = {k: v / total for k, v in mix.items()}

    # Get mechanics
    reg = registry()
    if args.mechanics:
        mech_keys = args.mechanics
    else:
        mech_keys = sorted(reg.keys())
    if args.limit_mechanics:
        mech_keys = mech_keys[: args.limit_mechanics]

    print(f"Building staged trajectories for {len(mech_keys)} mechanics")
    print(f"  Mix: {mix}")
    print(f"  Seeds per game: {args.seeds_per_game}")
    print(f"  Preset: {args.preset}")
    print(
        f"  Art: {args.include_art}, Randomize: {args.include_randomize} "
        "(ignored when --preset is set)"
    )

    all_trajectories: list[dict] = []
    total_episodes = 0
    total_won = 0
    t_start = time.time()

    for mech_key in mech_keys:
        mech_cls = reg[mech_key]
        spec_path = STAGED / mech_key / "SPEC.md"
        if not spec_path.exists():
            continue

        try:
            spec = parse_spec(spec_path)
        except Exception:
            continue

        for seed in range(args.seeds_per_game):
            for source, frac in mix.items():
                # Skip based on mix fraction (approximate)
                if np.random.random() > frac:
                    continue

                art = args.include_art and np.random.random() > 0.5
                randomize = args.include_randomize and np.random.random() > 0.5

                try:
                    mech = _make_mechanic(
                        mech_cls,
                        randomize=randomize,
                        art=art,
                        seed=seed,
                        preset_name=args.preset,
                    )
                    episode = rollout(
                        mech, spec, seed, source,
                        perturb_prob=args.perturb_prob,
                        max_actions=args.max_actions,
                        art=art, randomize=randomize,
                        preset_name=args.preset,
                    )
                except Exception as e:
                    if seed == 0:
                        print(f"  [warn] {mech_key} seed={seed}: {e}")
                    continue

                if episode is None or not episode.frames:
                    continue

                traj = episode_to_trajectory(episode, mech_key)
                if traj["steps"]:
                    all_trajectories.append(traj)
                    total_episodes += 1
                    if episode.won:
                        total_won += 1

        # Progress every 100 mechanics
        if len(all_trajectories) % 500 == 0 and all_trajectories:
            elapsed = time.time() - t_start
            print(f"  ... {len(all_trajectories)} episodes "
                  f"({total_won} won, {elapsed:.0f}s)")

    elapsed = time.time() - t_start
    print(f"\nGenerated {total_episodes} episodes ({total_won} won) "
          f"in {elapsed:.0f}s")

    if not all_trajectories:
        raise SystemExit("No trajectories generated!")

    # Shuffle and split
    rng = np.random.RandomState(42)
    rng.shuffle(all_trajectories)
    split = int(len(all_trajectories) * args.train_split)

    import torch
    train_path = Path(str(args.out).replace(".pt", "_train.pt"))
    val_path = Path(str(args.out).replace(".pt", "_val.pt"))

    train_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(all_trajectories[:split], str(train_path))
    torch.save(all_trajectories[split:], str(val_path))

    train_steps = sum(len(t["steps"]) for t in all_trajectories[:split])
    val_steps = sum(len(t["steps"]) for t in all_trajectories[split:])

    print(f"Train: {split} trajectories, {train_steps} steps → {train_path}")
    print(f"Val:   {len(all_trajectories) - split} trajectories, "
          f"{val_steps} steps → {val_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
