"""SCOUT_EXPERIMENT: behavioral catch probe — does the catch head play catch?

Paired comparison on identical env seeds:

- MODEL policy: scan the glove across its reachable lateral range (real
  moves, real frames), score every position with the trained catch head,
  move back to the argmax position, then pour.
- RANDOM policy: the curriculum collector's placement rule (uniform sweep
  offset), then pour.

Outcome per pour is the same hindsight pixel label used in training
(pour cells touching the Portband glove edges/interior). The comparison
metric is caught-pour rate. Local, offline, seeded; no scorecard.

Usage:
    python synthesis_lab/orchestration/behavioral_catch_probe.py \
        --run-dir synthesis_lab/experiments/megarexus_cnn/runs/catch_v3 \
        --episodes 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_LAB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _LAB_ROOT.parent
sys.path.insert(0, str(_LAB_ROOT))
sys.path.insert(0, str(_LAB_ROOT / "orchestration"))
sys.path.insert(0, str(_LAB_ROOT / "experiments" / "megarexus_cnn"))

from engine_adapters.portband_glove_adapter import (  # noqa: E402
    catch_label,
    component_displacement,
    glove_ports,
)
from collect_catch_play import (  # noqa: E402
    GAME_ID,
    SWEEP_OFFSETS,
    EpisodeRecorder,
    find_gloves,
    game_live,
    step,
    track_glove,
)
from eval_ghost import encode  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402

from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

MAX_POURS = 4
SCAN_LIMIT = 10  # max moves per direction during the scan


def glove_plane_of(bbox) -> torch.Tensor:
    plane = torch.zeros(1, 1, 16, 16)
    x0, y0, x1, y1 = (int(v) // 4 for v in bbox)
    plane[0, 0, y0:y1 + 1, x0:x1 + 1] = 1.0
    return plane


@torch.no_grad()
def catch_logit(model, device, cur, prev, bbox) -> float:
    frame = torch.cat([encode(cur), encode(prev)], dim=0).unsqueeze(0).to(device)
    return float(model.forward_catch(frame, glove_plane_of(bbox).to(device)))


def move_n(env, rec, prev_grid, glove, aid, n):
    """Issue up to n moves; returns (fd, cur, prev_of_cur, glove, moved_count)."""
    fd, cur, before = None, prev_grid, prev_grid
    moved = 0
    for _ in range(n):
        before = cur
        fd, cur, _ = step(env, rec, cur, aid)
        if not game_live(fd):
            return fd, cur, before, glove, moved
        if component_displacement(before, cur, glove) is None:
            return fd, cur, before, glove, moved  # blocked
        moved += 1
        tracked = track_glove(cur, glove)
        if tracked is not None:
            glove = tracked
    return fd, cur, before, glove, moved


@torch.no_grad()
def imagine_pour_column(model, device, cur, prev, horizon: int = 7) -> int | None:
    """Ghost-imagine ACTION5 + ticks; return the column where paint falls.

    Pure imagination (no env access): the LeWM-role dynamics head predicts
    the pour trajectory; the target column is where imagined change mass
    concentrates. Catch-logit hill-climbing fails here because the trained
    head's logits are saturated plateaus — the world model gives geometry.
    """
    grid, stack_prev = cur, prev
    paint = np.zeros((64, 64), dtype=np.int32)
    aid = 5
    for _ in range(horizon):
        frame = torch.cat([encode(grid), encode(stack_prev)], dim=0).unsqueeze(0).to(device)
        logits = model.forward_dynamics(
            frame,
            torch.tensor([aid], device=device),
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.long, device=device),
        )
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int8)
        paint += (pred != grid).astype(np.int32)
        stack_prev, grid = grid, pred
        aid = 0  # ticks after the pour
    col_mass = paint.sum(axis=0)
    if col_mass.sum() == 0:
        return None
    return int(col_mass.argmax())


def model_place(model, device, env, rec, fd, cur, prev, glove, budget: int = 10):
    """Move the glove under the imagined paint column, then let go."""
    tx = imagine_pour_column(model, device, cur, prev)
    if tx is None:
        return fd, cur, prev, glove
    while budget > 0:
        x0, _, x1, _ = glove["bbox_xyxy"]
        if x0 <= tx <= x1:
            break  # glove spans the predicted paint column
        direction = 4 if tx > x1 else 3
        fd, cur, prev, glove, m = move_n(env, rec, cur, glove, direction, 1)
        if not game_live(fd) or m == 0:
            return fd, cur, prev, glove
        budget -= 1
    return fd, cur, prev, glove


def random_place(env, rec, fd, cur, prev, glove, rng):
    offset = int(rng.choice(SWEEP_OFFSETS))
    if offset != 0:
        aid = 4 if offset > 0 else 3
        fd, cur, prev, glove, _ = move_n(env, rec, cur, glove, aid, abs(offset))
    return fd, cur, prev, glove


def run_policy(policy, model, device, episodes, seed):
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_REPO_ROOT / "environment_files"),
    )
    rng = np.random.default_rng(seed)
    pours = []
    for ep in range(episodes):
        env = arcade.make(GAME_ID, seed=seed + ep)
        fd = env.reset()
        cur = np.asarray(fd.frame[-1] if isinstance(fd.frame, list) else fd.frame,
                         dtype=np.int8)
        prev = cur
        rec = EpisodeRecorder()
        gloves, fd, cur = find_gloves(env, rec, fd, cur, rng, max_gloves=1)
        if not gloves:
            continue
        glove = gloves[0]
        for _ in range(MAX_POURS):
            if not game_live(fd):
                break
            if policy == "model":
                fd, cur, prev, glove = model_place(model, device, env, rec, fd, cur, prev, glove)
            else:
                fd, cur, prev, glove = random_place(env, rec, fd, cur, prev, glove, rng)
            if not game_live(fd):
                break
            tracked = track_glove(cur, glove)
            if tracked is not None:
                glove = tracked
            logit = catch_logit(model, device, cur, prev, glove["bbox_xyxy"])
            fd, cur, frames = step(env, rec, cur, 5)  # POUR
            if fd is None:
                break
            ports = glove_ports(glove["bbox_xyxy"], glove["component_id"])
            # strict catch: paint on the TOP edge during the early fall —
            # the eventual liquid spread touching a side is not a catch
            label = catch_label(frames, glove["mask"], ports,
                                sides=("top",), max_frames=8)
            pours.append({"episode_seed": seed + ep, "caught": label["caught"],
                          "contact_cells": label["contact_cells"],
                          "predicted_logit": logit})
            if not game_live(fd):
                break
        caught = sum(p["caught"] for p in pours if p["episode_seed"] == seed + ep)
        total = sum(1 for p in pours if p["episode_seed"] == seed + ep)
        print(f"[probe:{policy}] ep {ep}: {caught}/{total} caught")
    return pours


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=20270001)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
    stack = ckpt["config"].get("stack_frames", 2)
    model = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)).to(device).eval()
    model.load_state_dict(ckpt["model"])

    results = {}
    for policy in ("model", "random"):
        pours = run_policy(policy, model, device, args.episodes, args.seed)
        n = len(pours)
        caught = sum(p["caught"] for p in pours)
        results[policy] = {"pours": n, "caught": caught,
                           "catch_rate": caught / max(n, 1), "detail": pours}
        print(f"[probe] {policy}: {caught}/{n} = {caught / max(n, 1):.2f}")

    report = {
        "schema": "synthesis_lab.behavioral_catch_probe.v0",
        "evidence": "local offline pixels only; SCOUT_EXPERIMENT",
        "paired_seeds": [args.seed, args.seed + args.episodes - 1],
        "model": {k: v for k, v in results["model"].items() if k != "detail"},
        "random": {k: v for k, v in results["random"].items() if k != "detail"},
        "detail": {k: v["detail"] for k, v in results.items()},
    }
    out = args.run_dir / "behavioral_catch_probe.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"[probe] report → {out}")


if __name__ == "__main__":
    main()
