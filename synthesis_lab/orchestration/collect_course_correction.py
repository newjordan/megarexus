"""SCOUT_EXPERIMENT: Stage 2 course-correction shard collection.

This collector is only valid after the Stage 1 perturbation gate fails. It
reaches the SP80 plumbing level with the trained catch policy, applies the
same kind of harness-authored pre-arrange perturbations used by
perturb_gate.py, then records short recovery attempts from those off-nominal
states.

The output is the standard rollout-shard format consumed by dataset.py:
steps in frame-before-action convention plus optional catch records. Labels
are still derived downstream from pixels and environment fields only. The
loaded checkpoint is a collection policy and plan scorer, never a label
source, and no new model head is introduced here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_LAB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _LAB_ROOT.parent
for _entry in (
    _LAB_ROOT,
    _LAB_ROOT / "orchestration",
    _LAB_ROOT / "experiments" / "megarexus_cnn",
):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402
from arrange_probe import (  # noqa: E402
    N_MOVABLES,
    exhaustive_x_plans,
    execute_plan,
    imagine_plan,
    plan_actions,
    rank_movables_if_clicked,
    sample_plans,
)
from collect_catch_play import EpisodeRecorder, game_live  # noqa: E402
from collect_plumbing_play import (  # noqa: E402
    GAME_ID,
    MOVE_ACTIONS,
    arrange_demo,
    arrange_random,
    click_and_move,
    pour_last,
    reach_next_level,
    relocate,
)
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402

ENV_DIR = _REPO_ROOT / "environment_files"
DEFAULT_SEED = 20290103
DEFAULT_POLICIES = ("model_replan", "demo_recovery", "random_recovery", "null_pour")


def last_grid(fd) -> np.ndarray:
    frames = fd.frame if isinstance(fd.frame, list) else [fd.frame]
    return np.asarray(frames[-1], dtype=np.int8)


def load_model(ckpt_path: Path, device: str) -> MegarexusCNN:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    stack = int(payload["config"].get("stack_frames", 2))
    model = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)).to(device).eval()
    model.load_state_dict(payload["model"])
    return model


def terminal_step(fd, cur: np.ndarray) -> dict[str, Any]:
    return {
        "frame": cur,
        "action_id": 0,
        "action_x": 0,
        "action_y": 0,
        "levels_completed": int(getattr(fd, "levels_completed", 0) or 0) if fd else 0,
        "state": str(getattr(fd, "state", None)) if fd else "None",
    }


def perturb_state(
    env,
    rec: EpisodeRecorder,
    fd,
    cur: np.ndarray,
    prev: np.ndarray,
    rng: np.random.Generator,
    min_rounds: int,
    max_rounds: int,
) -> tuple[Any, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    rounds = int(rng.integers(min_rounds, max_rounds + 1))
    log: list[dict[str, Any]] = []
    for round_index in range(rounds):
        if not game_live(fd):
            break
        comps = frame_components(cur)
        if not comps:
            break
        comp = comps[int(rng.integers(len(comps)))]
        aid = int(rng.choice(MOVE_ACTIONS))
        requested = int(rng.integers(1, 4))
        before = comp
        fd, cur, prev, comp, moved, spent = click_and_move(
            env, rec, fd, cur, relocate(cur, comp) or comp, aid, requested
        )
        log.append(
            {
                "round_index": round_index,
                "component_before_bbox": list(before["bbox_xyxy"]),
                "action_id": aid,
                "requested_steps": requested,
                "moved_steps": int(moved),
                "actions_spent": int(spent),
                "component_after_bbox": list(comp["bbox_xyxy"]),
                "levels_completed": int(getattr(fd, "levels_completed", 0) or 0) if fd else 0,
                "state": str(getattr(fd, "state", None)) if fd else "None",
            }
        )
    return fd, cur, prev, log


def candidate_plans(rng: np.random.Generator, k_plans: int, search: str) -> list:
    if search == "exhaustive-x":
        return exhaustive_x_plans()
    return sample_plans(rng, k_plans)


@torch.no_grad()
def choose_model_plan(
    model: MegarexusCNN,
    device: str,
    cur: np.ndarray,
    prev: np.ndarray,
    rng: np.random.Generator,
    k_plans: int,
    search: str,
) -> tuple[list[dict], list | None, list[float]]:
    movables = [
        c for _score, c in rank_movables_if_clicked(model, device, cur, prev, top_k=N_MOVABLES)
    ]
    if len(movables) < N_MOVABLES:
        movables = list(frame_components(cur))[:N_MOVABLES]
    if len(movables) < N_MOVABLES:
        return movables, None, []

    scored = []
    for plan in candidate_plans(rng, k_plans, search):
        score, _end = imagine_plan(model, device, cur, prev, movables, plan)
        scored.append((float(score), plan))
    scored.sort(key=lambda item: -item[0])
    return movables, scored[0][1], [round(s, 3) for s, _plan in scored[:5]]


def choose_random_plan(
    cur: np.ndarray,
    rng: np.random.Generator,
    k_plans: int,
) -> tuple[list[dict], list | None]:
    movables = list(frame_components(cur))[:N_MOVABLES]
    if len(movables) < N_MOVABLES:
        return movables, None
    plans = sample_plans(rng, k_plans)
    return movables, plans[int(rng.integers(len(plans)))]


def execute_plan_prefix(
    env,
    rec: EpisodeRecorder,
    fd,
    cur: np.ndarray,
    movables: list[dict],
    plan,
    rng: np.random.Generator,
    min_groups: int,
    max_groups: int,
) -> tuple[Any, np.ndarray, np.ndarray, list[dict], int]:
    roster = list(movables)
    groups = plan_actions(plan, roster)
    if not groups:
        return fd, cur, cur, roster, 0
    n_groups = min(len(groups), int(rng.integers(min_groups, max_groups + 1)))
    prev = cur
    executed = 0
    for m, moves in groups[:n_groups]:
        if not game_live(fd) or m >= len(roster):
            break
        comp = relocate(cur, roster[m]) or roster[m]
        i = 0
        while i < len(moves) and game_live(fd):
            aid = int(moves[i])
            n = 1
            while i + n < len(moves) and int(moves[i + n]) == aid:
                n += 1
            fd, cur, prev, comp, _moved, _spent = click_and_move(env, rec, fd, cur, comp, aid, n)
            i += n
        roster[m] = comp
        executed += 1
    return fd, cur, prev, roster, executed


def collect(
    *,
    ckpt: Path,
    episodes: int,
    seed: int,
    policies: tuple[str, ...],
    k_plans: int,
    search: str,
    perturb_min_rounds: int,
    perturb_max_rounds: int,
    prefix_min_groups: int,
    prefix_max_groups: int,
) -> list[dict[str, Any]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt, device)
    arcade = Arcade(
        arc_api_key="anonymous",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )

    trajectories: list[dict[str, Any]] = []
    reached_count = 0
    level2_count = 0
    for ep in range(episodes):
        episode_seed = seed + ep
        rng = np.random.default_rng(episode_seed)
        policy = policies[ep % len(policies)]
        env = arcade.make(GAME_ID, seed=episode_seed)
        fd = env.reset()
        cur = last_grid(fd)
        prev = cur
        rec = EpisodeRecorder()
        catches: list[dict[str, Any]] = []
        row: dict[str, Any] = {
            "episode_seed": int(episode_seed),
            "policy": policy,
            "reached_plumbing": False,
            "reach_pours": 0,
            "perturbation": [],
            "prefix_groups_executed": 0,
            "first_plan": None,
            "first_plan_top_scores": [],
            "recovery_plan": None,
            "recovery_plan_top_scores": [],
        }

        fd, cur, prev, reached, reach_pours = reach_next_level(
            model, device, env, rec, fd, cur, prev, rng
        )
        row["reached_plumbing"] = bool(reached)
        row["reach_pours"] = int(reach_pours)
        reached_count += int(reached)

        if reached and game_live(fd):
            fd, cur, prev, perturb_log = perturb_state(
                env,
                rec,
                fd,
                cur,
                prev,
                rng,
                perturb_min_rounds,
                perturb_max_rounds,
            )
            row["perturbation"] = perturb_log

        if reached and game_live(fd):
            if policy == "random_recovery":
                movables, first_plan = choose_random_plan(cur, rng, k_plans)
                row["first_plan"] = first_plan
            else:
                movables, first_plan, top_scores = choose_model_plan(
                    model, device, cur, prev, rng, k_plans, search
                )
                row["first_plan"] = first_plan
                row["first_plan_top_scores"] = top_scores

            if first_plan is not None and policy != "null_pour":
                fd, cur, prev, movables, executed = execute_plan_prefix(
                    env,
                    rec,
                    fd,
                    cur,
                    movables,
                    first_plan,
                    rng,
                    prefix_min_groups,
                    prefix_max_groups,
                )
                row["prefix_groups_executed"] = int(executed)

            if game_live(fd):
                if policy == "model_replan":
                    movables2, plan2, top_scores2 = choose_model_plan(
                        model, device, cur, prev, rng, k_plans, search
                    )
                    row["recovery_plan"] = plan2
                    row["recovery_plan_top_scores"] = top_scores2
                    if plan2 is not None:
                        fd, cur, movables2, moved = execute_plan(env, rec, fd, cur, movables2, plan2)
                        row["recovery_moved_counts"] = [int(v) for v in moved]
                        if game_live(fd):
                            fd, cur, record = pour_last(env, rec, fd, cur, movables2)
                            if record is not None:
                                catches.append(record)
                elif policy == "demo_recovery":
                    fd, cur, prev, movables2, arrangement = arrange_demo(
                        model, device, env, rec, fd, cur, prev, rng=rng
                    )
                    row["demo_arrangement"] = arrangement
                    if game_live(fd):
                        fd, cur, record = pour_last(env, rec, fd, cur, movables2)
                        if record is not None:
                            catches.append(record)
                elif policy == "random_recovery":
                    fd, cur, prev, movables2, arrangement = arrange_random(
                        model, device, env, rec, fd, cur, prev, rng
                    )
                    row["random_arrangement"] = arrangement
                    if game_live(fd):
                        fd, cur, record = pour_last(env, rec, fd, cur, movables2)
                        if record is not None:
                            catches.append(record)
                elif policy == "null_pour":
                    movables2 = list(frame_components(cur))[:6]
                    fd, cur, record = pour_last(env, rec, fd, cur, movables2)
                    if record is not None:
                        catches.append(record)
                else:
                    raise ValueError(f"unknown policy: {policy}")

        final_levels = int(getattr(fd, "levels_completed", 0) or 0) if fd else 0
        level2_count += int(final_levels >= 2)
        rec.steps.append(terminal_step(fd, cur))
        row.update(
            {
                "levels_completed": int(final_levels),
                "final_state": str(getattr(fd, "state", None)) if fd else "None",
                "steps": len(rec.steps),
                "catches": len(catches),
            }
        )
        trajectories.append(
            {
                "game_id": GAME_ID,
                "rollout_id": f"course_correction_seed{episode_seed}_{policy}",
                "policy": policy,
                "stage": "stage2_course_correction",
                "truth_boundary": "local offline pixels plus environment fields only",
                "steps": rec.steps,
                "catches": catches,
                "course_correction": row,
            }
        )
        print(
            f"[course_correction] ep={ep} seed={episode_seed} policy={policy} "
            f"reached={reached} levels={final_levels} steps={len(rec.steps)} "
            f"catches={len(catches)}",
            flush=True,
        )

    print(
        f"[course_correction] reached={reached_count}/{episodes} "
        f"levels>=2={level2_count}/{episodes}",
        flush=True,
    )
    return trajectories


def write_manifest(out: Path, trajectories: list[dict[str, Any]], args: argparse.Namespace) -> None:
    summary = {
        "schema": "synthesis_lab.course_correction_collection.v0",
        "condition": "Stage 2 narrow v10 course-correction shard after Stage 1b perturbation gate failure",
        "evidence_boundary": "local offline collection shard; not a release gate and not a live scorecard",
        "labels": "dataset.py derives labels from pixels and environment fields only",
        "checkpoint": str(args.ckpt),
        "output_shard": str(out),
        "episodes": len(trajectories),
        "seed_start": int(args.seed),
        "policies": list(args.policies),
        "k_plans": int(args.plans),
        "search": args.search,
        "perturb_rounds": [int(args.perturb_min_rounds), int(args.perturb_max_rounds)],
        "prefix_groups": [int(args.prefix_min_groups), int(args.prefix_max_groups)],
        "level2_episodes": sum(
            1 for t in trajectories if int(t["course_correction"]["levels_completed"]) >= 2
        ),
        "reached_plumbing_episodes": sum(
            1 for t in trajectories if t["course_correction"]["reached_plumbing"]
        ),
        "detail": [t["course_correction"] for t in trajectories],
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--policies", choices=DEFAULT_POLICIES, nargs="+", default=list(DEFAULT_POLICIES))
    p.add_argument("--plans", type=int, default=32)
    p.add_argument("--search", choices=("sample", "exhaustive-x"), default="sample")
    p.add_argument("--perturb-min-rounds", type=int, default=1)
    p.add_argument("--perturb-max-rounds", type=int, default=3)
    p.add_argument("--prefix-min-groups", type=int, default=1)
    p.add_argument("--prefix-max-groups", type=int, default=2)
    args = p.parse_args()

    trajectories = collect(
        ckpt=args.ckpt.resolve(),
        episodes=args.episodes,
        seed=args.seed,
        policies=tuple(args.policies),
        k_plans=args.plans,
        search=args.search,
        perturb_min_rounds=args.perturb_min_rounds,
        perturb_max_rounds=args.perturb_max_rounds,
        prefix_min_groups=args.prefix_min_groups,
        prefix_max_groups=args.prefix_max_groups,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajectories, args.out)
    write_manifest(args.out, trajectories, args)
    n_steps = sum(len(t["steps"]) for t in trajectories)
    print(
        f"[course_correction] wrote {len(trajectories)} episodes / {n_steps} steps -> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
