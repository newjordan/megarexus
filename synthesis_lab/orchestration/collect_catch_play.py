"""SCOUT_EXPERIMENT: catch-play curriculum collection in local SP80.

The game of catch (user's framing): paint is the ball, the Portband
moveable object is the glove. Per episode:

1. PROBE — find the glove by controlled interaction: click a component
   (ACTION6), issue a move (ACTION1-4), and keep the component whose
   pixels actually displaced. Evidence-based, no game-source peeking.
2. SWEEP — move the glove laterally to a curriculum offset (side to side
   across episodes, like the lateral-random-target paintflow env moved
   the goal side to side).
3. POUR — ACTION5; record every animation frame as tick transitions.
4. LABEL — hindsight from pixels: did pour cells touch the glove's
   Portband port edges / interior? (engine_adapters.portband_glove_adapter)

Output shard = standard trajectory steps (frame-before-action convention)
plus per-pour catch records: glove bbox, port edge cells, sweep offset,
caught/contact_cells. Rollouts reaching GAME_OVER just end the episode.

Usage:
    python synthesis_lab/orchestration/collect_catch_play.py \
        --out synthesis_lab/experiments/megarexus_cnn/data/catch_play_v0.pt \
        --episodes 40
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

from engine_adapters.portband_glove_adapter import (  # noqa: E402
    catch_label,
    component_displacement,
    frame_components,
    glove_ports,
)

from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402
from arcengine import GameAction, GameState  # noqa: E402

GAME_ID = "sp80-589a99af"
ACTION_OF_ID = {
    1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
    4: GameAction.ACTION4, 5: GameAction.ACTION5, 6: GameAction.ACTION6,
}
# curriculum: lateral glove offsets, swept side to side across episodes
SWEEP_OFFSETS = [-9, -6, -3, -1, 0, 1, 3, 6, 9]
MAX_POURS_PER_EPISODE = 4
PROBE_CANDIDATES = 6


class EpisodeRecorder:
    """Accumulates steps in the frame-before-action shard convention."""

    def __init__(self) -> None:
        self.steps: list[dict] = []

    def record(self, prev: np.ndarray, aid: int, x: int, y: int, fd) -> list[np.ndarray]:
        frames = fd.frame if isinstance(fd.frame, list) else [fd.frame]
        frames = [np.asarray(f, dtype=np.int8) for f in frames]
        levels = getattr(fd, "levels_completed", 0)
        self.steps.append({
            "frame": prev, "action_id": aid, "action_x": x, "action_y": y,
            "levels_completed": levels, "state": str(getattr(fd, "state", None)),
        })
        for mid in frames[:-1]:
            self.steps.append({
                "frame": mid, "action_id": 0, "action_x": 0, "action_y": 0,
                "levels_completed": levels, "state": str(getattr(fd, "state", None)),
            })
        return frames


def game_live(fd) -> bool:
    return fd is not None and getattr(fd, "state", None) not in (
        GameState.GAME_OVER, GameState.WIN
    )


def step(env, rec, prev, aid, x=0, y=0):
    data = {"x": x, "y": y} if aid == 6 else None
    fd = env.step(ACTION_OF_ID[aid], data=data)
    if fd is None:
        return None, prev, []
    frames = rec.record(prev, aid, x, y, fd)
    return fd, frames[-1], frames


def find_gloves(env, rec, fd, prev, rng, max_gloves: int = 1):
    """Click candidates and keep those whose pixels move under ACTION4/3.

    Returns (gloves, fd, prev). With max_gloves > 1 this is the
    multi-object roster: every controllable component found by interaction
    evidence, each one a candidate glove for waterfall/multi-catch play.
    """
    comps = frame_components(prev)
    rng.shuffle(comps)
    gloves = []
    for comp in comps[:PROBE_CANDIDATES]:
        if len(gloves) >= max_gloves:
            break
        cx, cy = int(comp["center_xy"][0]), int(comp["center_xy"][1])
        fd, prev, _ = step(env, rec, prev, 6, cx, cy)
        if not game_live(fd):
            return gloves, fd, prev
        before = prev
        fd, prev, _ = step(env, rec, prev, 4)  # move_right
        if not game_live(fd):
            return gloves, fd, prev
        moved = component_displacement(before, prev, comp)
        # a real controlled move under ACTION4 displaces rightward; gauge
        # drains and animations produce off-axis "motion" — reject those
        if moved is None or moved[0] < 0.4 or abs(moved[1]) > 1.0:
            continue
        fd, prev, _ = step(env, rec, prev, 3)  # move back left
        if not game_live(fd):
            return gloves, fd, prev
        gloves.append(comp)
    return gloves, fd, prev


def track_glove(grid: np.ndarray, glove: dict) -> dict | None:
    """Re-locate the glove after moves: same color, nearest to last position."""
    comps = [c for c in frame_components(grid) if c["color"] == glove["color"]]
    if not comps:
        return None
    gx, gy = glove["center_xy"]
    return min(comps, key=lambda c: (c["center_xy"][0] - gx) ** 2 + (c["center_xy"][1] - gy) ** 2)


def collect(episodes: int, seed: int, max_gloves: int = 1) -> list[dict]:
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_REPO_ROOT / "environment_files"),
    )
    rng = np.random.default_rng(seed)
    trajectories = []
    n_caught = n_pours = 0
    for ep in range(episodes):
        env = arcade.make(GAME_ID, seed=seed + ep)
        fd = env.reset()
        prev = np.asarray(fd.frame[-1] if isinstance(fd.frame, list) else fd.frame,
                          dtype=np.int8)
        rec = EpisodeRecorder()
        catches: list[dict] = []

        gloves, fd, prev = find_gloves(env, rec, fd, prev, rng, max_gloves=max_gloves)
        if gloves:
            for pour_i in range(MAX_POURS_PER_EPISODE):
                if not game_live(fd):
                    break
                # multi-object play: rotate which glove is the active catcher,
                # but re-select it first (moves apply to the selected object)
                glove = gloves[pour_i % len(gloves)]
                if len(gloves) > 1:
                    gx, gy = int(glove["center_xy"][0]), int(glove["center_xy"][1])
                    fd, prev, _ = step(env, rec, prev, 6, gx, gy)
                    if not game_live(fd):
                        break
                offset = SWEEP_OFFSETS[(ep + pour_i * 3) % len(SWEEP_OFFSETS)]
                # SWEEP: nudge the glove toward its current x + offset
                for _ in range(abs(offset)):
                    aid = 4 if offset > 0 else 3
                    before = prev
                    fd, prev, _ = step(env, rec, prev, aid)
                    if not game_live(fd):
                        break
                    if component_displacement(before, prev, glove) is None:
                        break  # blocked — placement is wherever we stopped
                    tracked = track_glove(prev, glove)
                    if tracked is not None:
                        glove = tracked
                if not game_live(fd):
                    break
                tracked = track_glove(prev, glove)
                if tracked is not None:
                    glove = tracked
                gloves[pour_i % len(gloves)] = glove
                ports = glove_ports(glove["bbox_xyxy"], glove["component_id"])
                pour_step_index = len(rec.steps)
                fd, prev, frames = step(env, rec, prev, 5)  # POUR
                if fd is None:
                    break
                label = catch_label(frames, glove["mask"], ports)
                n_pours += 1
                n_caught += int(label["caught"])
                # waterfall/multi-object: label every roster glove, not just
                # the active one — which glove did this pour actually reach?
                roster = []
                for g_i, g in enumerate(gloves):
                    g_label = label if g is glove else catch_label(
                        frames, g["mask"], glove_ports(g["bbox_xyxy"], g["component_id"])
                    )
                    roster.append({
                        "glove_index": g_i,
                        "active": g is glove,
                        "glove_bbox_xyxy": g["bbox_xyxy"],
                        "glove_color": g["color"],
                        "caught": g_label["caught"],
                        "contact_cells": g_label["contact_cells"],
                    })
                catches.append({
                    "pour_step_index": pour_step_index,
                    "sweep_offset": offset,
                    "glove_bbox_xyxy": glove["bbox_xyxy"],
                    "glove_color": glove["color"],
                    "port_count": len(ports),
                    "roster": roster,
                    **label,
                })
                if not game_live(fd):
                    break

        rec.steps.append({
            "frame": prev, "action_id": 0, "action_x": 0, "action_y": 0,
            "levels_completed": getattr(fd, "levels_completed", 0) if fd else 0,
            "state": str(getattr(fd, "state", None)) if fd else "None",
        })
        trajectories.append({
            "game_id": GAME_ID,
            "rollout_id": f"catch_play_seed{seed + ep}",
            "steps": rec.steps,
            "catches": catches,
        })
        caught_str = "".join("Y" if c["caught"] else "n" for c in catches)
        print(f"[catch] ep {ep}: {len(rec.steps)} steps, gloves={len(gloves)}, "
              f"pours={caught_str or '-'}")
    print(f"[catch] total pours={n_pours}, caught={n_caught} "
          f"({n_caught / max(n_pours, 1):.2f} base rate)")
    return trajectories


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--max-gloves", type=int, default=1,
                   help=">1 = multi-object waterfall play: rotate the active "
                        "catcher and label every roster glove per pour")
    args = p.parse_args()
    trajectories = collect(args.episodes, args.seed, max_gloves=args.max_gloves)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajectories, args.out)
    n = sum(len(t["steps"]) for t in trajectories)
    n_c = sum(len(t["catches"]) for t in trajectories)
    print(f"[catch] wrote {len(trajectories)} episodes / {n} steps / {n_c} pours → {args.out}")


if __name__ == "__main__":
    main()
