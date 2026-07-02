"""SCOUT_EXPERIMENT: level probe — does model-guided catch play advance SP80?

The question: "how would the system do against sp80?" The only honest
answer is environment fields from local offline episodes (levels_completed,
state). No scorecard is opened; nothing here is a benchmark claim.

Paired on identical seeds, full episodes:

- MODEL policy: ghost-imagine the pour column (world model), move the
  Portband glove under it, pour; on a level transition, re-find the glove
  in the new scene.
- RANDOM policy: curriculum sweep placement, pour.

Per episode we record pours, strict catches, levels_completed, and final
state — all read off the environment, hindsight-from-pixels for catches.

Usage:
    python synthesis_lab/orchestration/level_probe.py \
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
    frame_components,
    glove_ports,
)
from behavioral_catch_probe import (  # noqa: E402
    imagine_pour_column,
    move_n,
    random_place,
)
from collect_catch_play import (  # noqa: E402
    GAME_ID,
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

MAX_POURS = 12   # safety cap only; the gauge ends episodes first
GAUGE_ROW = 0    # empirically measured regime fact: row-0 HUD strip drains
STEP_PX = 4      # empirically measured glove displacement per move
GAUGE_MARGIN = 2  # spare gauge cells required beyond an imagined plan's cost


def background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid), return_counts=True)
    return int(vals[int(np.argmax(counts))])


def gauge_remaining(grid: np.ndarray) -> int:
    """Present-state read of the row-0 gauge strip (regime fact measured in
    the gas audit: it drains per action, GAME_OVER on exhaustion). This is
    perception of the current frame used as a controller informer — not a
    win-condition model."""
    return int((np.asarray(grid)[GAUGE_ROW] != background_color(grid)).sum())


@torch.no_grad()
def imagine_plan_drain(model, device, cur, prev, actions) -> int:
    """Dry-run an action plan in the world model and return its predicted
    gauge cost (imagined changed cells on the gauge row). Drain physics is
    learned (558/558 drained cells on val), so this is the model's own
    belief about the plan's cost — no env access."""
    grid, sp = cur, prev
    drained = 0
    for aid in actions:
        frame = torch.cat([encode(grid), encode(sp)], dim=0).unsqueeze(0).to(device)
        logits = model.forward_dynamics(
            frame,
            torch.tensor([aid], device=device),
            torch.zeros(1, dtype=torch.long, device=device),
            torch.zeros(1, dtype=torch.long, device=device),
        )
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int8)
        drained += int((pred[GAUGE_ROW] != grid[GAUGE_ROW]).sum())
        sp, grid = grid, pred
    return drained


def careful_place(model, device, env, rec, fd, cur, prev, glove, budget: int = 10):
    """Miss-gated, budget-aware placement.

    1. Imagine the pour column (world model, no env access).
    2. Imagine the intercept walk + pour; read the predicted gauge drain.
       Decline if the plan would exhaust the gauge — don't start a walk
       you can't finish.
    3. Walk for real; POUR ONLY IF the glove actually spans the imagined
       column. "You can't miss": an unassured pour is declined, never
       thrown.

    Returns (fd, cur, prev, glove, verdict) where verdict is a dict with
    decision in {"pour", "decline:<reason>"}, the imagined target column,
    and the plan's predicted gauge cost.
    """
    tx = imagine_pour_column(model, device, cur, prev)
    if tx is None:
        return fd, cur, prev, glove, {"decision": "decline:no_imagined_column"}
    x0, _, x1, _ = glove["bbox_xyxy"]
    gap = 0 if x0 <= tx <= x1 else (tx - x1 if tx > x1 else x0 - tx)
    dist = -(-gap // STEP_PX)
    direction = 4 if tx > x1 else 3
    plan = [direction] * dist + [5, 0, 0, 0]
    cost = imagine_plan_drain(model, device, cur, prev, plan)
    verdict = {"target_col": tx, "predicted_drain": cost,
               "gauge_at_decision": gauge_remaining(cur)}
    if cost + GAUGE_MARGIN > verdict["gauge_at_decision"]:
        verdict["decision"] = "decline:gauge"
        return fd, cur, prev, glove, verdict
    while budget > 0:
        x0, _, x1, _ = glove["bbox_xyxy"]
        if x0 <= tx <= x1:
            break
        direction = 4 if tx > x1 else 3
        fd, cur, prev, glove, m = move_n(env, rec, cur, glove, direction, 1)
        if not game_live(fd):
            verdict["decision"] = "decline:dead"
            return fd, cur, prev, glove, verdict
        if m == 0:
            verdict["decision"] = "decline:blocked"
            return fd, cur, prev, glove, verdict
        budget -= 1
    x0, _, x1, _ = glove["bbox_xyxy"]
    verdict["decision"] = "pour" if x0 <= tx <= x1 else "decline:unreached"
    return fd, cur, prev, glove, verdict


@torch.no_grad()
def find_gloves_learned(model, device, env, rec, fd, cur, prev, max_gloves: int = 1):
    """Glove finding via the trained controllability head: rank components
    by "would this displace under a move", click the winner to select it.
    One action (the selecting click) instead of 3 per probed candidate.
    Falls back to the interaction probe when no component scores positive.
    """
    frame = torch.cat([encode(cur), encode(prev)], dim=0).unsqueeze(0).to(device)
    scored = []
    for comp in frame_components(cur):
        plane = torch.zeros(1, 1, 16, 16, device=device)
        x0, y0, x1, y1 = (int(v) // 4 for v in comp["bbox_xyxy"])
        plane[0, 0, y0:y1 + 1, x0:x1 + 1] = 1.0
        logit = model.forward_control(frame, plane,
                                      torch.tensor([4], device=device))
        scored.append((float(logit), comp))
    scored.sort(key=lambda t: -t[0])
    gloves = []
    for logit, comp in scored[:max_gloves]:
        if logit <= 0:
            break
        cx, cy = int(comp["center_xy"][0]), int(comp["center_xy"][1])
        fd, cur, _ = step(env, rec, cur, 6, cx, cy)  # select it
        if not game_live(fd):
            return gloves, fd, cur
        gloves.append(comp)
    return gloves, fd, cur


def run_policy(policy, model, device, episodes, seed, finder="probe"):
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_REPO_ROOT / "environment_files"),
    )
    rng = np.random.default_rng(seed)
    rows = []

    def locate(fd, cur, prev):
        if finder == "learned":
            gloves, fd, cur = find_gloves_learned(model, device, env, rec, fd,
                                                  cur, prev, max_gloves=1)
            if gloves:
                return gloves, fd, cur
            # fall back to the interaction probe if the head abstains
        gloves, fd, cur = find_gloves(env, rec, fd, cur, rng, max_gloves=1)
        return gloves, fd, cur

    for ep in range(episodes):
        env = arcade.make(GAME_ID, seed=seed + ep)
        fd = env.reset()
        cur = np.asarray(fd.frame[-1] if isinstance(fd.frame, list) else fd.frame,
                         dtype=np.int8)
        prev = cur
        rec = EpisodeRecorder()
        levels = int(getattr(fd, "levels_completed", 0) or 0)
        pours = catches = 0
        pour_detail = []
        declines = []

        gloves, fd, cur = locate(fd, cur, prev)
        glove = gloves[0] if gloves else None
        for _ in range(MAX_POURS):
            if not game_live(fd) or glove is None:
                break
            verdict = {"decision": "pour"}
            if policy == "model":
                fd, cur, prev, glove, verdict = careful_place(
                    model, device, env, rec, fd, cur, prev, glove)
            else:
                fd, cur, prev, glove = random_place(env, rec, fd, cur, prev, glove, rng)
            if not game_live(fd):
                break
            if verdict["decision"] != "pour":
                # can't-miss rule: an unassured pour is never thrown
                declines.append(verdict)
                break
            tracked = track_glove(cur, glove)
            if tracked is not None:
                glove = tracked
            gauge_before = gauge_remaining(cur)
            fd, cur, frames = step(env, rec, cur, 5)  # POUR
            if fd is None:
                break
            pours += 1
            ports = glove_ports(glove["bbox_xyxy"], glove["component_id"])
            label = catch_label(frames, glove["mask"], ports,
                                sides=("top",), max_frames=8)
            # overflow-block evidence: paint contacting the glove's SIDE
            # port edges over the full animation (the orthogonal catch)
            block = catch_label(frames, glove["mask"], ports,
                                sides=("left", "right"))
            catches += int(label["caught"])
            pour_detail.append({
                "level_at_pour": levels,
                "caught": bool(label["caught"]),
                "side_block_contact": bool(block["caught"]),
                "side_contact_cells": int(block["contact_cells"]),
                "gauge_before": gauge_before,
                "gauge_after": gauge_remaining(cur),
                "predicted_plan_drain": verdict.get("predicted_drain"),
            })
            now = int(getattr(fd, "levels_completed", 0) or 0)
            if now > levels:
                levels = now
                if game_live(fd):  # new level → new scene → re-find the glove
                    prev = cur
                    gloves, fd, cur = locate(fd, cur, prev)
                    glove = gloves[0] if gloves else None
            if not game_live(fd):
                break
        rows.append({
            "episode_seed": seed + ep,
            "pours": pours,
            "strict_catches": catches,
            "levels_completed": levels,
            "final_state": str(getattr(fd, "state", None)) if fd is not None else "None",
            "gauge_final": gauge_remaining(cur),
            "declines": declines,
            "pour_detail": pour_detail,
        })
        print(f"[level:{policy}] ep {ep}: levels={levels} pours={pours} "
              f"catches={catches} declines={len(declines)} "
              f"gauge_left={rows[-1]['gauge_final']} state={rows[-1]['final_state']}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=20280001)
    p.add_argument("--finder", choices=("probe", "learned"), default="probe",
                   help="glove finding: interaction probe vs controllability head")
    p.add_argument("--out-name", default="level_probe.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
    stack = ckpt["config"].get("stack_frames", 2)
    model = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)).to(device).eval()
    model.load_state_dict(ckpt["model"])

    results = {}
    for policy in ("model", "random"):
        # the learned finder is part of the MODEL policy; random keeps the
        # interaction probe so it stays identical to the original baseline
        finder = args.finder if policy == "model" else "probe"
        rows = run_policy(policy, model, device, args.episodes, args.seed,
                          finder=finder)
        n_eps = len(rows)
        eps_with_level = sum(1 for r in rows if r["levels_completed"] > 0)
        results[policy] = {
            "episodes": n_eps,
            "episodes_with_level_completed": eps_with_level,
            "level_rate": eps_with_level / max(n_eps, 1),
            "total_levels_completed": sum(r["levels_completed"] for r in rows),
            "total_pours": sum(r["pours"] for r in rows),
            "total_strict_catches": sum(r["strict_catches"] for r in rows),
            "total_misses": sum(r["pours"] - r["strict_catches"] for r in rows),
            "total_declines": sum(len(r["declines"]) for r in rows),
            "total_side_block_contacts": sum(
                sum(1 for p in r["pour_detail"] if p["side_block_contact"])
                for r in rows),
            "detail": rows,
        }
        print(f"[level] {policy}: {eps_with_level}/{n_eps} episodes completed a level "
              f"({results[policy]['total_levels_completed']} total)")

    report = {
        "schema": "synthesis_lab.level_probe.v1",
        "evidence": "local offline environment fields only; SCOUT_EXPERIMENT",
        "paired_seeds": [args.seed, args.seed + args.episodes - 1],
        "finder": args.finder,
        "model": {k: v for k, v in results["model"].items() if k != "detail"},
        "random": {k: v for k, v in results["random"].items() if k != "detail"},
        "detail": {k: v["detail"] for k, v in results.items()},
    }
    out = args.run_dir / args.out_name
    out.write_text(json.dumps(report, indent=2))
    print(f"[level] report → {out}")


if __name__ == "__main__":
    main()
