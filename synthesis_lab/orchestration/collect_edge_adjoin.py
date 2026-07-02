"""SCOUT_EXPERIMENT: edge-adjoin trials — PortBand edge locks made behavioral.

PortBand ranks static edge-lock configurations (port edges lining up with
wall/target edges). This collector produces the *behavioral* counterpart:
drive a controllable block (found by interaction probing) in a direction
until it stops moving, then label from pixels whether its leading port
edge now adjoins another object — a wall/scenery cell or another block —
and record the residual directional gap (the pixel analogue of PortBand's
``alignment_error_px``).

Per trial record: direction, moves issued, blocked?, adjoined?, partner
color/kind, final gap. All steps stored in the standard shard convention
so the same data also feeds the world model.

Usage:
    python synthesis_lab/orchestration/collect_edge_adjoin.py \
        --out synthesis_lab/experiments/megarexus_cnn/data/edge_adjoin_v0.pt \
        --episodes 30
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
sys.path.insert(0, str(_LAB_ROOT / "orchestration"))

from engine_adapters.portband_glove_adapter import (  # noqa: E402
    component_displacement,
    directional_gap,
    frame_components,
    glove_ports,
    masks_adjoin,
    port_edge_cells,
)
from collect_catch_play import (  # noqa: E402
    GAME_ID,
    EpisodeRecorder,
    find_gloves,
    game_live,
    step,
    track_glove,
)

from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

DIRECTION_ACTION = {"up": 1, "down": 2, "left": 3, "right": 4}
DIRECTIONS = ["right", "left", "down", "up"]
MAX_MOVES_PER_TRIAL = 14
MAX_TRIALS_PER_EPISODE = 4


def adjoin_label(grid: np.ndarray, glove: dict, direction: str) -> dict:
    """Hindsight from pixels: does the glove's leading edge adjoin something?"""
    ports = glove_ports(glove["bbox_xyxy"], glove["component_id"])
    side_port = [p for p in ports if p["side"] == direction]
    edge_mask = port_edge_cells(side_port or ports)
    others = np.zeros_like(glove["mask"])
    partner_color = None
    for comp in frame_components(grid):
        if comp["color"] == glove["color"] and comp["bbox_xyxy"] == glove["bbox_xyxy"]:
            continue
        others |= comp["mask"]
    adjoined = masks_adjoin(edge_mask | glove["mask"], others)
    gap = directional_gap(glove["bbox_xyxy"], others, direction)
    if adjoined:
        # partner = the touching component's color (first found)
        grown = glove["mask"].copy()
        grown[1:, :] |= glove["mask"][:-1, :]
        grown[:-1, :] |= glove["mask"][1:, :]
        grown[:, 1:] |= glove["mask"][:, :-1]
        grown[:, :-1] |= glove["mask"][:, 1:]
        touch = grown & others
        if touch.any():
            ys, xs = np.nonzero(touch)
            partner_color = int(np.asarray(grid)[ys[0], xs[0]])
    return {
        "adjoined": bool(adjoined),
        "gap_px": int(gap) if gap is not None else -1,
        "partner_color": partner_color,
    }


def collect(episodes: int, seed: int) -> list[dict]:
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_REPO_ROOT / "environment_files"),
    )
    rng = np.random.default_rng(seed)
    trajectories = []
    n_adjoined = n_trials = 0
    for ep in range(episodes):
        env = arcade.make(GAME_ID, seed=seed + ep)
        fd = env.reset()
        prev = np.asarray(fd.frame[-1] if isinstance(fd.frame, list) else fd.frame,
                          dtype=np.int8)
        rec = EpisodeRecorder()
        adjoins: list[dict] = []

        gloves, fd, prev = find_gloves(env, rec, fd, prev, rng, max_gloves=1)
        if gloves:
            glove = gloves[0]
            for trial_i in range(MAX_TRIALS_PER_EPISODE):
                if not game_live(fd):
                    break
                direction = DIRECTIONS[(ep + trial_i) % len(DIRECTIONS)]
                aid = DIRECTION_ACTION[direction]
                trial_start_index = len(rec.steps)
                moves = 0
                blocked = False
                for _ in range(MAX_MOVES_PER_TRIAL):
                    before = prev
                    fd, prev, _ = step(env, rec, prev, aid)
                    if not game_live(fd):
                        break
                    moves += 1
                    if component_displacement(before, prev, glove) is None:
                        blocked = True
                        break
                    tracked = track_glove(prev, glove)
                    if tracked is not None:
                        glove = tracked
                if not game_live(fd):
                    break
                tracked = track_glove(prev, glove)
                if tracked is not None:
                    glove = tracked
                label = adjoin_label(prev, glove, direction)
                n_trials += 1
                n_adjoined += int(label["adjoined"])
                adjoins.append({
                    "trial_start_index": trial_start_index,
                    "end_step_index": len(rec.steps),
                    "direction": direction,
                    "moves": moves,
                    "blocked": blocked,
                    "glove_bbox_xyxy": glove["bbox_xyxy"],
                    "glove_color": glove["color"],
                    **label,
                })

        rec.steps.append({
            "frame": prev, "action_id": 0, "action_x": 0, "action_y": 0,
            "levels_completed": getattr(fd, "levels_completed", 0) if fd else 0,
            "state": str(getattr(fd, "state", None)) if fd else "None",
        })
        trajectories.append({
            "game_id": GAME_ID,
            "rollout_id": f"edge_adjoin_seed{seed + ep}",
            "steps": rec.steps,
            "adjoins": adjoins,
        })
        marks = "".join("A" if a["adjoined"] else ("b" if a["blocked"] else ".")
                        for a in adjoins)
        print(f"[adjoin] ep {ep}: {len(rec.steps)} steps, trials={marks or '-'}")
    print(f"[adjoin] total trials={n_trials}, adjoined={n_adjoined} "
          f"({n_adjoined / max(n_trials, 1):.2f} base rate)")
    return trajectories


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=20263001)
    args = p.parse_args()
    trajectories = collect(args.episodes, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajectories, args.out)
    n = sum(len(t["steps"]) for t in trajectories)
    n_a = sum(len(t["adjoins"]) for t in trajectories)
    print(f"[adjoin] wrote {len(trajectories)} episodes / {n} steps / {n_a} trials → {args.out}")


if __name__ == "__main__":
    main()
