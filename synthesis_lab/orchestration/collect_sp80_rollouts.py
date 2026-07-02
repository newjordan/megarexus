"""SCOUT_EXPERIMENT: collect real SP80 transitions via local rollouts.

STOCHASTIC_GOOSE_RUNTIME drives action proposals (uniform Goose-style
sampling — its cold-start behavior); the local sp80 environment
(environment_files/sp80/589a99af, OFFLINE mode — no API, no live
scorecard) produces the ground-truth transitions. Output is a .pt shard of
trajectories in the lewmtrain convention: step i holds the frame *before*
action i, i.e. action i transitions frame_i → frame_{i+1}.

Usage:
    python synthesis_lab/orchestration/collect_sp80_rollouts.py \
        --out synthesis_lab/experiments/megarexus_cnn/data/sp80_rollouts.pt \
        --rollouts 8 --max-steps 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_LAB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _LAB_ROOT.parent
sys.path.insert(0, str(_LAB_ROOT))

from engine_adapters.goose_action_adapter import sample_goose_action  # noqa: E402

from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

GAME_ID = "sp80-589a99af"
ACTION_OF_ID = {
    1: GameAction.ACTION1,
    2: GameAction.ACTION2,
    3: GameAction.ACTION3,
    4: GameAction.ACTION4,
    5: GameAction.ACTION5,
    6: GameAction.ACTION6,
}


def all_frames(frame_data) -> list[np.ndarray]:
    """Every animation frame of the step (ACTION5 pours return ~20 frames —
    they carry the paint-fall physics, so they must not be dropped)."""
    frames = frame_data.frame
    if not isinstance(frames, list):
        frames = [frames]
    out = []
    for f in frames:
        arr = np.asarray(f, dtype=np.int8)
        if arr.shape != (64, 64):
            raise ValueError(f"unexpected frame shape {arr.shape}")
        out.append(arr)
    return out


def last_frame(frame_data) -> np.ndarray:
    return all_frames(frame_data)[-1]


def available_action_ids(frame_data) -> list[int]:
    avail = getattr(frame_data, "available_actions", None)
    ids = []
    for a in avail or []:
        name = getattr(a, "name", str(a))
        if name.startswith("ACTION") and name[6:].isdigit():
            ids.append(int(name[6:]))
    ids = [i for i in ids if i in ACTION_OF_ID]
    return ids or [1, 2, 3, 4, 5, 6]  # sp80 is keyboard_click


def flow_biased_action(
    available: list[int], rng: np.random.Generator, p_flow: float, p_click: float
) -> tuple[int, int, int]:
    """Goose-style stochastic proposal, reweighted toward flow events:
    elevated ACTION5 (activate_flow) and ACTION6 (select object) so pours —
    the catch-and-redirect mechanic — appear densely in the data."""
    r = rng.random()
    if 5 in available and r < p_flow:
        return 5, 0, 0
    if 6 in available and r < p_flow + p_click:
        return 6, int(rng.integers(64)), int(rng.integers(64))
    return sample_goose_action(available, rng=rng)


def collect(
    rollouts: int,
    max_steps: int,
    seed: int,
    p_flow: float = 0.0,
    p_click: float = 0.0,
) -> list[dict]:
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_REPO_ROOT / "environment_files"),
    )
    rng = np.random.default_rng(seed)
    trajectories = []
    policy = "goose_flowbias" if p_flow > 0 else "goose_uniform"
    for r in range(rollouts):
        env = arcade.make(GAME_ID, seed=seed + r)
        if env is None:
            raise RuntimeError(f"could not open local environment {GAME_ID}")
        fd = env.reset()
        prev = last_frame(fd)
        steps = []
        n_ticks = 0
        for _ in range(max_steps):
            if fd is None:
                break
            state = getattr(fd, "state", None)
            if state in (GameState.GAME_OVER, GameState.WIN):
                break
            aid, x, y = flow_biased_action(available_action_ids(fd), rng, p_flow, p_click)
            levels = getattr(fd, "levels_completed", 0)
            data = {"x": x, "y": y} if aid == 6 else None
            fd = env.step(ACTION_OF_ID[aid], data=data)
            if fd is None:
                break
            frames = all_frames(fd)
            # step i holds the frame BEFORE its action; animation frames after
            # the first become tick steps (action 0 = time passes)
            steps.append({
                "frame": prev, "action_id": aid, "action_x": x, "action_y": y,
                "levels_completed": levels, "state": str(state),
            })
            new_levels = getattr(fd, "levels_completed", 0)
            for mid in frames[:-1]:
                steps.append({
                    "frame": mid, "action_id": 0, "action_x": 0, "action_y": 0,
                    "levels_completed": new_levels, "state": str(getattr(fd, "state", None)),
                })
                n_ticks += 1
            prev = frames[-1]
        steps.append({
            "frame": prev, "action_id": 0, "action_x": 0, "action_y": 0,
            "levels_completed": getattr(fd, "levels_completed", 0) if fd is not None else 0,
            "state": str(getattr(fd, "state", None)) if fd is not None else "None",
        })
        trajectories.append({
            "game_id": GAME_ID,
            "rollout_id": f"{policy}_seed{seed + r}",
            "steps": steps,
        })
        print(f"[collect] rollout {r}: {len(steps)} steps ({n_ticks} animation ticks), "
              f"final levels_completed={steps[-1]['levels_completed']}")
    return trajectories


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rollouts", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--seed", type=int, default=20260701)
    p.add_argument("--p-flow", type=float, default=0.0,
                   help="extra probability mass on ACTION5 (activate_flow)")
    p.add_argument("--p-click", type=float, default=0.0,
                   help="extra probability mass on ACTION6 (select object)")
    args = p.parse_args()

    trajectories = collect(args.rollouts, args.max_steps, args.seed,
                           p_flow=args.p_flow, p_click=args.p_click)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajectories, args.out)
    n = sum(len(t["steps"]) for t in trajectories)
    print(f"[collect] wrote {len(trajectories)} rollouts / {n} steps → {args.out}")


if __name__ == "__main__":
    main()
