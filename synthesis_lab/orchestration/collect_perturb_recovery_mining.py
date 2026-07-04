"""SCOUT_EXPERIMENT: mine perturbed plumbing recovery trajectories.

This collector addresses the post-v10 data gap where course-correction
shards reached the plumbing level but produced no level-2 recoveries. It
uses local offline SP80 only:

1. reach plumbing with the trained catch policy,
2. apply seeded harness-authored pre-arrange perturbations,
3. score candidate recovery plans with the current CNN,
4. replay the exact reach+perturb action prefix into fresh envs,
5. execute candidate plans and keep successful recoveries plus hard failures.

The checkpoint is a collection policy and candidate ranker only. Labels come
from recorded pixels and environment fields through dataset.py.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
    rank_movables_if_clicked,
    sample_plans,
)
from collect_catch_play import EpisodeRecorder, game_live  # noqa: E402
from collect_course_correction import (  # noqa: E402
    DEFAULT_SEED,
    perturb_state,
    terminal_step,
)
from collect_plumbing_play import GAME_ID, pour_last, reach_next_level  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402
from perturb_gate import action_rows, last_grid, replay_actions  # noqa: E402

ENV_DIR = _REPO_ROOT / "environment_files"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(ckpt_path: Path, device: str) -> MegarexusCNN:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    stack = int(payload["config"].get("stack_frames", 2))
    model = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)).to(device).eval()
    model.load_state_dict(payload["model"])
    return model


def candidate_plan_pool(
    *,
    model: MegarexusCNN,
    device: str,
    cur: np.ndarray,
    prev: np.ndarray,
    rng: np.random.Generator,
    search: str,
    sample_count: int,
    max_candidates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    movables = [
        c for _score, c in rank_movables_if_clicked(model, device, cur, prev, top_k=N_MOVABLES)
    ]
    if len(movables) < N_MOVABLES:
        movables = list(frame_components(cur))[:N_MOVABLES]
    if len(movables) < N_MOVABLES:
        return movables, [], []

    plans = exhaustive_x_plans() if search == "exhaustive-x" else sample_plans(rng, sample_count)
    scored: list[tuple[float, Any]] = []
    for plan in plans:
        score, _end = imagine_plan(model, device, cur, prev, movables, plan)
        scored.append((float(score), plan))
    scored.sort(key=lambda item: -item[0])
    top_scores = [round(score, 4) for score, _plan in scored[:8]]
    rows = [
        {"rank": rank, "score": float(score), "plan": plan}
        for rank, (score, plan) in enumerate(scored[:max_candidates], start=1)
    ]
    return movables, rows, top_scores


def replay_and_execute_candidate(
    *,
    arcade: Arcade,
    model: MegarexusCNN,
    device: str,
    episode_seed: int,
    prefix_steps: list[dict[str, Any]],
    prefix_actions: list[dict[str, Any]],
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    env = arcade.make(GAME_ID, seed=episode_seed)
    fd, cur, prev, replay_trace = replay_actions(env, prefix_actions)
    rec = EpisodeRecorder()
    rec.steps = copy.deepcopy(prefix_steps)
    catches: list[dict[str, Any]] = []

    row: dict[str, Any] = {
        "candidate_rank": int(candidate["rank"]),
        "candidate_score": float(candidate["score"]),
        "candidate_plan": candidate["plan"],
        "replay_action_count": len(replay_trace),
        "replay_levels_completed": int(getattr(fd, "levels_completed", 0) or 0) if fd else 0,
        "status": "not_live_after_replay",
        "levels_completed": int(getattr(fd, "levels_completed", 0) or 0) if fd else 0,
        "steps": len(rec.steps),
    }
    if game_live(fd):
        movables = [
            c for _score, c in rank_movables_if_clicked(model, device, cur, prev, top_k=N_MOVABLES)
        ]
        if len(movables) < N_MOVABLES:
            movables = list(frame_components(cur))[:N_MOVABLES]
        if len(movables) >= N_MOVABLES:
            fd, cur, roster, moved = execute_plan(env, rec, fd, cur, movables, candidate["plan"])
            row["moved_counts"] = [int(v) for v in moved]
            if game_live(fd):
                fd, cur, record = pour_last(env, rec, fd, cur, roster)
                if record is not None:
                    catches.append(record)
            row["status"] = "completed"
        else:
            row["status"] = f"insufficient_roster:{len(movables)}"

    final_levels = int(getattr(fd, "levels_completed", 0) or 0) if fd else row["levels_completed"]
    rec.steps.append(terminal_step(fd, cur))
    row.update(
        {
            "levels_completed": int(final_levels),
            "success": bool(final_levels >= 2),
            "final_state": str(getattr(fd, "state", None)) if fd else "None",
            "steps": len(rec.steps),
            "catches": len(catches),
        }
    )
    traj = {
        "game_id": GAME_ID,
        "rollout_id": (
            f"perturb_recovery_seed{episode_seed}_rank{int(candidate['rank']):03d}"
        ),
        "policy": "perturb_recovery_mining",
        "stage": "stage2_positive_recovery_mining",
        "truth_boundary": "local offline pixels plus environment fields only",
        "steps": rec.steps,
        "catches": catches,
        "perturb_recovery": row,
    }
    return row, traj


def collect(
    *,
    ckpt: Path,
    episodes: int,
    seed: int,
    search: str,
    sample_count: int,
    max_candidates: int,
    keep_failures: int,
    perturb_min_rounds: int,
    perturb_max_rounds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt, device)
    arcade = Arcade(
        arc_api_key="anonymous",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )

    trajectories: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for ep in range(episodes):
        episode_seed = seed + ep
        rng = np.random.default_rng(episode_seed)
        env = arcade.make(GAME_ID, seed=episode_seed)
        fd = env.reset()
        cur = last_grid(fd)
        prev = cur
        rec = EpisodeRecorder()

        fd, cur, prev, reached, reach_pours = reach_next_level(
            model, device, env, rec, fd, cur, prev, rng
        )
        perturb_log: list[dict[str, Any]] = []
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

        prefix_steps = copy.deepcopy(rec.steps)
        prefix_actions = action_rows(rec.steps)
        seed_summary: dict[str, Any] = {
            "episode_seed": int(episode_seed),
            "reached_plumbing": bool(reached),
            "reach_pours": int(reach_pours),
            "prefix_actions": len(prefix_actions),
            "perturbation": perturb_log,
            "candidate_top_scores": [],
            "tested": [],
            "kept_successes": 0,
            "kept_failures": 0,
        }
        if not reached or not game_live(fd):
            summaries.append(seed_summary)
            print(
                f"[perturb_recovery] seed={episode_seed} reached={reached} "
                "no candidate mining",
                flush=True,
            )
            continue

        _movables, candidates, top_scores = candidate_plan_pool(
            model=model,
            device=device,
            cur=cur,
            prev=prev,
            rng=rng,
            search=search,
            sample_count=sample_count,
            max_candidates=max_candidates,
        )
        seed_summary["candidate_top_scores"] = top_scores
        for candidate in candidates:
            row, traj = replay_and_execute_candidate(
                arcade=arcade,
                model=model,
                device=device,
                episode_seed=episode_seed,
                prefix_steps=prefix_steps,
                prefix_actions=prefix_actions,
                candidate=candidate,
            )
            seed_summary["tested"].append(row)
            keep = False
            if row["success"]:
                keep = True
                seed_summary["kept_successes"] += 1
            elif seed_summary["kept_failures"] < keep_failures:
                keep = True
                seed_summary["kept_failures"] += 1
            if keep:
                trajectories.append(traj)
            print(
                f"[perturb_recovery] seed={episode_seed} rank={row['candidate_rank']} "
                f"score={row['candidate_score']:.3f} success={row['success']} "
                f"levels={row['levels_completed']} kept={keep}",
                flush=True,
            )
        summaries.append(seed_summary)
    return trajectories, summaries


def write_manifest(
    out: Path,
    trajectories: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    manifest = {
        "schema": "synthesis_lab.perturb_recovery_mining.v0",
        "condition": (
            "Mine positive and hard-negative recovery trajectories after seeded "
            "pre-arrange perturbations in local offline SP80"
        ),
        "evidence_boundary": (
            "local offline collection shard; not a release gate and not a live scorecard"
        ),
        "labels": "dataset.py derives labels from pixels and environment fields only",
        "checkpoint": str(args.ckpt),
        "checkpoint_sha256": sha256_file(args.ckpt.resolve()),
        "collector_sha256": sha256_file(Path(__file__).resolve()),
        "output_shard": str(out),
        "episodes": int(args.episodes),
        "seed_start": int(args.seed),
        "search": args.search,
        "sample_count": int(args.sample_count),
        "max_candidates": int(args.max_candidates),
        "keep_failures": int(args.keep_failures),
        "perturb_rounds": [int(args.perturb_min_rounds), int(args.perturb_max_rounds)],
        "kept_trajectories": len(trajectories),
        "kept_successes": sum(
            1 for t in trajectories if t["perturb_recovery"].get("success")
        ),
        "kept_failures": sum(
            1 for t in trajectories if not t["perturb_recovery"].get("success")
        ),
        "detail": summaries,
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED + 50_000)
    parser.add_argument("--search", choices=("sample", "exhaustive-x"), default="exhaustive-x")
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--keep-failures", type=int, default=2)
    parser.add_argument("--perturb-min-rounds", type=int, default=1)
    parser.add_argument("--perturb-max-rounds", type=int, default=3)
    args = parser.parse_args()

    trajectories, summaries = collect(
        ckpt=args.ckpt.resolve(),
        episodes=args.episodes,
        seed=args.seed,
        search=args.search,
        sample_count=args.sample_count,
        max_candidates=args.max_candidates,
        keep_failures=args.keep_failures,
        perturb_min_rounds=args.perturb_min_rounds,
        perturb_max_rounds=args.perturb_max_rounds,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajectories, args.out)
    write_manifest(args.out, trajectories, summaries, args)
    n_steps = sum(len(t["steps"]) for t in trajectories)
    print(
        f"[perturb_recovery] wrote {len(trajectories)} trajectories / {n_steps} steps "
        f"-> {args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
