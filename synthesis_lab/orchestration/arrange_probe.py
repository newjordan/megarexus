"""SCOUT_EXPERIMENT: arrangement planner probe — can the model SOLVE the
plumbing level through its own learned signals?

Stage-2 "caring" wired end-to-end (HANDOFF §6.3): per episode the MODEL
policy reaches the plumbing level with trained catch play, then

1. samples K candidate arrangement plans (per-movable x/y step offsets
   over the controllability-ranked roster) — the human demo plan is NOT
   injected; if the search finds a demo-like plan, that is the result;
2. IMAGINES each plan with its own dynamics head (clicks + moves rolled
   out autoregressively, components re-located inside imagination);
3. scores each imagined pre-pour frame with its OWN level-up head;
4. executes the argmax plan for real, pours LAST, and the environment
   fields judge the outcome.

The RANDOM baseline draws one plan from the same sampler on the same
seed and executes it — identical protocol, no imagination, no scoring.

Evidence: local offline environment fields only. No scorecard, no
benchmark claims, no hand-coded win model anywhere in the loop. The
probe also logs imagined-vs-real end-frame agreement on the executed
plan — an honest measure of imagination quality in this regime.

Usage:
    python synthesis_lab/orchestration/arrange_probe.py \
        --run-dir synthesis_lab/experiments/megarexus_cnn/runs/outcome_v5 \
        --episodes 10 --plans 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LAB_ROOT))
sys.path.insert(0, str(_LAB_ROOT / "orchestration"))
sys.path.insert(0, str(_LAB_ROOT / "experiments" / "megarexus_cnn"))

from collect_plumbing_play import (  # noqa: E402
    GAME_ID,
    EpisodeRecorder,
    click_and_move,
    game_live,
    rank_movables,
    reach_next_level,
    relocate,
    step,
)
from eval_ghost import encode  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402

from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

_REPO_ROOT = _LAB_ROOT.parent

N_MOVABLES = 3           # roster size for planning
MAX_STEP_X = 3           # per-movable offset range, in 4px move steps
MAX_STEP_Y = 1
AID_OF_DX = {+1: 4, -1: 3}   # right / left
AID_OF_DY = {+1: 2, -1: 1}   # down / up


def plan_actions(plan, movables):
    """Expand a plan [(movable_idx, dx, dy), ...] into the action list it
    implies; movable positions are resolved at execution/imagination time."""
    seq = []
    for m, dx, dy in plan:
        moves = []
        if dx:
            moves += [AID_OF_DX[int(np.sign(dx))]] * abs(dx)
        if dy:
            moves += [AID_OF_DY[int(np.sign(dy))]] * abs(dy)
        if moves:
            seq.append((m, moves))
    return seq


def sample_plans(rng, k):
    """K random per-movable offset plans; the null plan (pour as-is) is
    always candidate 0 so 'do nothing' competes fairly."""
    plans = [[(m, 0, 0) for m in range(N_MOVABLES)]]
    while len(plans) < k:
        plans.append([
            (m,
             int(rng.integers(-MAX_STEP_X, MAX_STEP_X + 1)),
             int(rng.integers(-MAX_STEP_Y, MAX_STEP_Y + 1)))
            for m in range(N_MOVABLES)
        ])
    return plans


def exhaustive_x_plans():
    """Every horizontal arrangement × choice of last mover.

    (dx0, dx1, dx2) ∈ {-3..3}^3 with dy = 0, and for each combo one plan
    per choice of which movable moves LAST. Both dimensions are needed
    and both were measured (2026-07-02):
    - the score landscape is a needle (demo +20, one step off → −15..−49,
      partials below null) → sampling/mutation can't find it, enumeration
      can;
    - selection recolor makes the LAST-moved component part of the scored
      state (same offsets: bar-last +19.95, platform-last −15..−41);
      first-two order is irrelevant (+19.95 either way)."""
    r = range(-MAX_STEP_X, MAX_STEP_X + 1)
    plans = [[(m, 0, 0) for m in range(N_MOVABLES)]]  # null plan
    for a in r:
        for b in r:
            for c in r:
                off = {0: a, 1: b, 2: c}
                movers = [m for m in (0, 1, 2) if off[m]]
                if not movers:
                    continue
                for last in movers:
                    order = [m for m in (0, 1, 2)
                             if m != last and off[m]] + [last]
                    plans.append([(m, off[m], 0) for m in order])
    return plans


@torch.no_grad()
def rank_movables_if_clicked(model, device, cur, prev, top_k=3):
    """Controllability is SELECTION-CONDITIONAL in this regime: only the
    selected object moves, and the control head — honestly trained on
    displacement — answers "would it move NOW", which is 'no' for any
    unselected movable (platform2 scored −84 while genuinely movable).
    The roster question is "would it move IF CLICKED": imagine the click
    with the dynamics head (selection recolor is well-supervised), then
    ask the control head on the imagined post-click frame. Two learned
    systems composed; no scene knowledge."""
    from eval_ghost import encode as _enc  # local alias for clarity
    from engine_adapters.portband_glove_adapter import frame_components as _fc

    scored = []
    for comp in _fc(cur):
        cx, cy = int(comp["center_xy"][0]), int(comp["center_xy"][1])
        clicked = imagine_frame_step(model, device, cur, prev, 6, cx, cy)
        frame = torch.cat([_enc(clicked), _enc(cur)], dim=0) \
            .unsqueeze(0).to(device)
        plane = torch.zeros(1, 1, 16, 16, device=device)
        x0, y0, x1, y1 = (int(v) // 4 for v in comp["bbox_xyxy"])
        plane[0, 0, y0:y1 + 1, x0:x1 + 1] = 1.0
        logit = max(
            float(model.forward_control(frame, plane,
                                        torch.tensor([a], device=device)))
            for a in (1, 2, 3, 4)
        )
        scored.append((logit, comp))
    scored.sort(key=lambda t: -t[0])
    return scored[:top_k]


@torch.no_grad()
def imagine_frame_step(model, device, cur, prev, aid, x=0, y=0):
    frame = torch.cat([encode(cur), encode(prev)], dim=0).unsqueeze(0).to(device)
    logits = model.forward_dynamics(
        frame,
        torch.tensor([aid], device=device),
        torch.tensor([x], device=device),
        torch.tensor([y], device=device),
    )
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int8)


@torch.no_grad()
def imagine_plan(model, device, cur, prev, movables, plan):
    """Roll the plan out in imagination; return (score, imagined_end_frame).

    score = the model's own level-up logit on the imagined pre-pour
    stack. Components are re-located inside imagination by position+area
    (selection recolors them, exactly as in the real env)."""
    g_cur, g_prev = cur, prev
    roster = list(movables)
    for m, moves in plan_actions(plan, roster):
        comp = relocate(g_cur, roster[m])
        if comp is None:
            continue
        cx, cy = int(comp["center_xy"][0]), int(comp["center_xy"][1])
        nxt = imagine_frame_step(model, device, g_cur, g_prev, 6, cx, cy)
        g_prev, g_cur = g_cur, nxt
        for aid in moves:
            nxt = imagine_frame_step(model, device, g_cur, g_prev, aid)
            g_prev, g_cur = g_cur, nxt
        tracked = relocate(g_cur, comp)
        if tracked is not None:
            roster[m] = tracked
    stack = torch.cat([encode(g_cur), encode(g_prev)], dim=0) \
        .unsqueeze(0).to(device)
    _, level_logit = model.forward_settle(stack)
    return float(level_logit.item()), g_cur


def execute_plan(env, rec, fd, cur, movables, plan):
    """Run the plan for real (clicks handled selection-recolor-aware by
    click_and_move). Returns (fd, cur, roster, moved_counts)."""
    roster = list(movables)
    moved_counts = []
    for m, moves in plan_actions(plan, roster):
        comp = relocate(cur, roster[m]) or roster[m]
        # group the move list into (aid, run-length) chunks
        i = 0
        while i < len(moves) and game_live(fd):
            aid = moves[i]
            n = 1
            while i + n < len(moves) and moves[i + n] == aid:
                n += 1
            fd, cur, _, comp, moved, _ = click_and_move(
                env, rec, fd, cur, comp, aid, n)
            moved_counts.append(moved)
            i += n
        roster[m] = comp
        if not game_live(fd):
            break
    return fd, cur, roster, moved_counts


def run_probe(model, device, policy, episodes, seed, k_plans,
              search="sample"):
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_REPO_ROOT / "environment_files"),
    )
    rng = np.random.default_rng(seed)
    rows = []
    for ep in range(episodes):
        env = arcade.make(GAME_ID, seed=seed + ep)
        fd = env.reset()
        cur = np.asarray(fd.frame[-1] if isinstance(fd.frame, list)
                         else fd.frame, dtype=np.int8)
        rec = EpisodeRecorder()
        row = {"episode_seed": seed + ep, "policy": policy,
               "reached_plumbing": False, "levels_completed": 0,
               "chosen_plan": None, "plan_score": None,
               "imagined_real_agreement": None}

        fd, cur, prev, reached, _ = reach_next_level(
            model, device, env, rec, fd, cur, cur, rng)
        row["reached_plumbing"] = bool(reached)
        if reached and game_live(fd):
            movables = [c for _, c in
                        rank_movables_if_clicked(model, device, cur, prev,
                                                 top_k=N_MOVABLES)]
            plans = sample_plans(rng, k_plans)
            imagined_end = None
            if policy == "model":
                if search == "exhaustive-x":
                    plans = exhaustive_x_plans()
                scored = []
                for plan in plans:
                    s, end = imagine_plan(model, device, cur, prev,
                                          movables, plan)
                    scored.append((s, plan, end))
                scored.sort(key=lambda t: -t[0])
                if search == "sample":
                    # one refinement round: mutate the top plans and rescore
                    for s, plan, _ in list(scored[:4]):
                        for _ in range(max(k_plans // 8, 4)):
                            mut = [(m,
                                    int(np.clip(dx + rng.integers(-1, 2),
                                                -MAX_STEP_X, MAX_STEP_X)),
                                    int(np.clip(dy + rng.integers(-1, 2),
                                                -MAX_STEP_Y, MAX_STEP_Y)))
                                   for m, dx, dy in plan]
                            s2, end2 = imagine_plan(model, device, cur, prev,
                                                    movables, mut)
                            scored.append((s2, mut, end2))
                    scored.sort(key=lambda t: -t[0])
                best_score, best_plan, imagined_end = scored[0]
                row["plan_score"] = best_score
                row["plan_score_spread"] = [round(s, 2) for s, _, _
                                            in scored[:5]]
            else:
                best_plan = plans[int(rng.integers(len(plans)))]
            row["chosen_plan"] = best_plan
            fd, cur, roster, moved = execute_plan(
                env, rec, fd, cur, movables, best_plan)
            row["moved_counts"] = moved
            if imagined_end is not None:
                row["imagined_real_agreement"] = float(
                    (imagined_end == cur).mean())
            if game_live(fd):
                fd, cur, _ = step(env, rec, cur, 5)  # POUR LAST
        row["levels_completed"] = int(
            getattr(fd, "levels_completed", 0) or 0) if fd else 0
        row["final_state"] = str(getattr(fd, "state", None)) \
            if fd is not None else "None"
        rows.append(row)
        print(f"[arrange:{policy}] ep {ep}: reached={row['reached_plumbing']} "
              f"levels={row['levels_completed']} plan={row['chosen_plan']} "
              f"score={row['plan_score']} "
              f"img_agree={row['imagined_real_agreement']}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=20280001)
    p.add_argument("--plans", type=int, default=64)
    p.add_argument("--search", choices=("sample", "exhaustive-x"),
                   default="sample")
    p.add_argument("--out-name", default="arrange_probe.json")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.run_dir / "best.pt", map_location=device,
                      weights_only=False)
    stack = ckpt["config"].get("stack_frames", 2)
    model = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)) \
        .to(device).eval()
    model.load_state_dict(ckpt["model"])

    results = {}
    for policy in ("model", "random"):
        rows = run_probe(model, device, policy, args.episodes, args.seed,
                         args.plans, search=args.search)
        done = sum(1 for r in rows if r["levels_completed"] >= 2)
        reached = sum(1 for r in rows if r["reached_plumbing"])
        results[policy] = {
            "episodes": len(rows),
            "reached_plumbing": reached,
            "completed_plumbing": done,
            "completion_rate": done / max(len(rows), 1),
            "detail": rows,
        }
        print(f"[arrange] {policy}: {done}/{len(rows)} completed the "
              f"plumbing level (reached {reached})")

    report = {
        "schema": "synthesis_lab.arrange_probe.v0",
        "evidence": "local offline environment fields only; SCOUT_EXPERIMENT",
        "paired_seeds": [args.seed, args.seed + args.episodes - 1],
        "k_plans": args.plans,
        "model": {k: v for k, v in results["model"].items() if k != "detail"},
        "random": {k: v for k, v in results["random"].items() if k != "detail"},
        "detail": {k: v["detail"] for k, v in results.items()},
    }
    out = args.run_dir / args.out_name
    out.write_text(json.dumps(report, indent=2))
    print(f"[arrange] report → {out}")


if __name__ == "__main__":
    main()
