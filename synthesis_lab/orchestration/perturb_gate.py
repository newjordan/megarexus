"""SCOUT_EXPERIMENT: Stage 1b perturbation arrange gate (HANDOFF §0B).

This gate reaches the SP80 plumbing level with the trained catch policy,
applies harness-authored random pre-arrange movements, then drives the REAL
`megarexus_cnn` live adapter from that perturbed state with the same obs
contract used by `codexlab.arc_suite_runner.run_game`:

    obs["frames"] = full animation frame list
    obs["grid"] = runner-compatible first frame fallback
    obs["resets"] = 0

The paired random baseline replays the exact reach+perturb actions into a
second env with the same seed, checks the perturbed frame hash matches, then
executes one sampled arrangement plan plus one pour from that same state.

Honesty boundary: perturbations are test-harness authorship, never model
capability, labels, or evidence of a solved level. Result evidence is local
offline environment fields only, not a live scorecard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

_LAB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _LAB_ROOT.parent
for _entry in (
    _REPO_ROOT / "codexlab",
    _LAB_ROOT,
    _LAB_ROOT / "orchestration",
    _LAB_ROOT / "experiments" / "megarexus_cnn",
):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402
from arc_play_harness import _ACT, grid_of  # noqa: E402
from arcengine import GameAction  # noqa: E402
from adapter_gate import judge_model_episode, max_consecutive_pours  # noqa: E402
from arrange_probe import (  # noqa: E402
    execute_plan,
    rank_movables_if_clicked,
    sample_plans,
)
from collect_catch_play import ACTION_OF_ID, EpisodeRecorder, game_live, step  # noqa: E402
from collect_plumbing_play import (  # noqa: E402
    GAME_ID,
    MOVE_ACTIONS,
    click_and_move,
    reach_next_level,
    relocate,
)
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from megarexus_live_policy import MegarexusArcPolicy, sha256_file  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402
from train import load_model_state_compatible  # noqa: E402

ENV_DIR = _REPO_ROOT / "environment_files"
DEFAULT_SEED = 20290101


def frame_hash(grid: np.ndarray) -> str:
    arr = np.asarray(grid, dtype=np.int16)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def last_grid(fd) -> np.ndarray:
    frames = fd.frame if isinstance(fd.frame, list) else [fd.frame]
    return np.asarray(frames[-1], dtype=np.int8)


def action_rows(steps: list[dict[str, Any]], start: int = 0) -> list[dict[str, Any]]:
    rows = []
    for row in steps[start:]:
        aid = int(row.get("action_id") or 0)
        if aid <= 0:
            continue
        rows.append(
            {
                "action_id": aid,
                "x": int(row.get("action_x") or 0),
                "y": int(row.get("action_y") or 0),
            }
        )
    return rows


def load_model(ckpt_path: Path, device: str) -> MegarexusCNN:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    stack = int(payload["config"].get("stack_frames", 2))
    model = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)).to(device).eval()
    load_model_state_compatible(model, payload["model"])
    return model


def replay_actions(env, actions: list[dict[str, Any]]) -> tuple[Any, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    fd = env.reset()
    cur = last_grid(fd)
    prev = cur
    trace = []
    for idx, row in enumerate(actions):
        aid = int(row["action_id"])
        data = {"x": int(row["x"]), "y": int(row["y"])} if aid == 6 else None
        before_hash = frame_hash(cur)
        fd = env.step(ACTION_OF_ID[aid], data=data)
        frames = fd.frame if isinstance(fd.frame, list) else [fd.frame]
        frames = [np.asarray(f, dtype=np.int8) for f in frames]
        prev, cur = cur, frames[-1]
        trace.append(
            {
                "index": idx,
                "action_id": aid,
                "x": int(row["x"]),
                "y": int(row["y"]),
                "before_frame_sha256": before_hash,
                "after_frame_sha256": frame_hash(cur),
                "levels_completed": int(getattr(fd, "levels_completed", 0) or 0),
                "state": str(getattr(fd, "state", None)),
            }
        )
        if not game_live(fd):
            break
    return fd, cur, prev, trace


def perturb_state(
    *,
    env,
    rec: EpisodeRecorder,
    fd,
    cur: np.ndarray,
    prev: np.ndarray,
    rng: np.random.Generator,
    min_rounds: int,
    max_rounds: int,
) -> tuple[Any, np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    start = len(rec.steps)
    rounds = int(rng.integers(min_rounds, max_rounds + 1))
    log: list[dict[str, Any]] = []
    for round_index in range(rounds):
        comps = frame_components(cur)
        if not comps or not game_live(fd):
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
    return fd, cur, prev, log, action_rows(rec.steps, start)


def obs_from_frame(fd, resets: int = 0) -> dict[str, Any]:
    raw_frames = fd.frame if isinstance(fd.frame, list) else [fd.frame]
    return {
        "grid": grid_of(fd),
        "frames": [np.asarray(f) for f in raw_frames],
        "resets": int(resets),
        "levels_completed": int(getattr(fd, "levels_completed", 0) or 0),
        "state": str(getattr(fd, "state", "")),
        "available_actions": list(getattr(fd, "available_actions", []) or []),
        "perception": {},
    }


def drive_adapter_from_state(
    *,
    env,
    fd,
    policy: MegarexusArcPolicy,
    max_actions: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "not_started",
        "levels_completed": int(getattr(fd, "levels_completed", 0) or 0),
        "actions": 0,
        "resets": 0,
        "final_state": str(getattr(fd, "state", "")),
        "policy_error_count": 0,
        "policy_errors": [],
    }
    history: list[dict[str, Any]] = []
    policy_errors: list[dict[str, Any]] = []
    best_levels = int(getattr(fd, "levels_completed", 0) or 0)
    first_hash = frame_hash(last_grid(fd))
    try:
        for step_index in range(max(0, int(max_actions))):
            state = str(getattr(fd, "state", ""))
            if "GAME_OVER" in state or "WIN" in state:
                break
            obs = obs_from_frame(fd, resets=0)
            grid = np.asarray(obs["frames"][-1], dtype=np.int8)
            grid_sha = frame_hash(grid)
            try:
                aid, x, y = policy(obs, history)
            except Exception as exc:  # noqa: BLE001 - gate records and stops cleanly.
                policy_errors.append({"step": int(step_index), "error": repr(exc)})
                aid, x, y = 0, None, None
            aid = int(aid)
            act = _ACT.get(aid, GameAction.RESET)
            data = {"x": int(x), "y": int(y)} if aid == 6 and x is not None and y is not None else None
            fd = env.step(act, data=data)
            levels = int(getattr(fd, "levels_completed", 0) or 0)
            best_levels = max(best_levels, levels)
            history.append(
                {
                    "step": int(step_index),
                    "action": aid,
                    "x": x,
                    "y": y,
                    "levels": levels,
                    "before_frame_sha256": grid_sha,
                    "source": getattr(policy, "last_action_source", "megarexus_cnn"),
                }
            )
            if "WIN" in str(getattr(fd, "state", "")):
                break
        last = last_grid(fd)
        result.update(
            {
                "status": "completed",
                "levels_completed": int(best_levels),
                "actions": len(history),
                "resets": 0,
                "final_state": str(getattr(fd, "state", "")),
                "policy_error_count": len(policy_errors),
                "policy_errors": policy_errors[:8],
                "first_frame_sha256": first_hash,
                "last_frame_sha256": frame_hash(last),
                "action_trace_sample": history[:12],
                "action_trace_tail": history[-12:] if len(history) > 12 else [],
            }
        )
        result["policy_trace"] = policy.export_trace()
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "status": "environment_error",
                "error": repr(exc),
                "policy_error_count": len(policy_errors),
                "policy_errors": policy_errors[:8],
                "actions": len(history),
                "resets": 0,
            }
        )
    return result


def run_random_baseline(
    *,
    env,
    fd,
    cur: np.ndarray,
    prev: np.ndarray,
    model: MegarexusCNN,
    device: str,
    rng: np.random.Generator,
    k_plans: int,
) -> dict[str, Any]:
    rec = EpisodeRecorder()
    row: dict[str, Any] = {
        "status": "not_started",
        "levels_completed": int(getattr(fd, "levels_completed", 0) or 0) if fd else 0,
        "actions": 0,
        "chosen_plan": None,
        "moved_counts": [],
        "final_state": str(getattr(fd, "state", None)) if fd else "None",
    }
    if not game_live(fd):
        row["status"] = "not_live_after_replay"
        return row
    movables = [c for _score, c in rank_movables_if_clicked(model, device, cur, prev, top_k=3)]
    plans = sample_plans(rng, k_plans)
    best_plan = plans[int(rng.integers(len(plans)))]
    row["chosen_plan"] = best_plan
    if len(movables) >= 3:
        fd, cur, _roster, moved = execute_plan(env, rec, fd, cur, movables, best_plan)
        row["moved_counts"] = [int(v) for v in moved]
        if game_live(fd):
            fd, cur, _frames = step(env, rec, cur, 5)
    else:
        row["status"] = f"insufficient_roster:{len(movables)}"
    row.update(
        {
            "status": "completed" if row["status"] == "not_started" else row["status"],
            "levels_completed": int(getattr(fd, "levels_completed", 0) or 0) if fd else 0,
            "actions": len(action_rows(rec.steps)),
            "final_state": str(getattr(fd, "state", None)) if fd else "None",
            "action_source": "paired_random_one_sampled_plan_plus_pour",
        }
    )
    return row


def summarise_sources(ledger: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(r.get("source")) for r in ledger))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-policy-actions", type=int, default=120)
    p.add_argument("--random-plans", type=int, default=64)
    p.add_argument("--perturb-min-rounds", type=int, default=1)
    p.add_argument("--perturb-max-rounds", type=int, default=3)
    p.add_argument("--skip-random", action="store_true")
    args = p.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt.resolve(), device)
    arcade = Arcade(
        arc_api_key="anonymous",
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(ENV_DIR),
    )

    rows = {"model": [], "random": []}
    for ep in range(args.episodes):
        episode_seed = args.seed + ep
        rng = np.random.default_rng(episode_seed)
        env_model = arcade.make(GAME_ID, seed=episode_seed)
        fd = env_model.reset()
        cur = last_grid(fd)
        prev = cur
        rec = EpisodeRecorder()
        t0 = time.time()

        fd, cur, prev, reached, reach_pours = reach_next_level(
            model, device, env_model, rec, fd, cur, prev, rng
        )
        reach_actions = action_rows(rec.steps)
        perturb_log: list[dict[str, Any]] = []
        perturb_actions: list[dict[str, Any]] = []
        if reached and game_live(fd):
            fd, cur, prev, perturb_log, perturb_actions = perturb_state(
                env=env_model,
                rec=rec,
                fd=fd,
                cur=cur,
                prev=prev,
                rng=rng,
                min_rounds=args.perturb_min_rounds,
                max_rounds=args.perturb_max_rounds,
            )

        perturbed_hash = frame_hash(cur)
        replay_match = False
        replay_levels = None
        fd_random = None
        cur_random = prev_random = None
        replay_trace: list[dict[str, Any]] = []
        if reached:
            env_random = arcade.make(GAME_ID, seed=episode_seed)
            fd_random, cur_random, prev_random, replay_trace = replay_actions(
                env_random, reach_actions + perturb_actions
            )
            replay_levels = int(getattr(fd_random, "levels_completed", 0) or 0) if fd_random else None
            replay_match = (
                fd_random is not None
                and frame_hash(cur_random) == perturbed_hash
                and replay_levels == int(getattr(fd, "levels_completed", 0) or 0)
            )
        else:
            env_random = None

        os.environ["MEGAREXUS_TRACE_DIR"] = str(out_dir / f"ep_seed{episode_seed}")
        if reached and game_live(fd):
            policy = MegarexusArcPolicy(checkpoint_path=args.ckpt)
            res = drive_adapter_from_state(
                env=env_model,
                fd=fd,
                policy=policy,
                max_actions=args.max_policy_actions,
            )
            judged = judge_model_episode(res)
        else:
            res = {
                "status": "not_reached_plumbing",
                "levels_completed": int(getattr(fd, "levels_completed", 0) or 0) if fd else 0,
                "actions": 0,
                "resets": 0,
                "policy_error_count": 0,
                "policy_trace": {},
            }
            judged = judge_model_episode(res)
            judged["passed"] = False
        os.environ.pop("MEGAREXUS_TRACE_DIR", None)

        trace = res.get("policy_trace") or {}
        ledger = trace.get("ledger") or []
        judged.update(
            {
                "episode_seed": episode_seed,
                "wall_seconds": round(time.time() - t0, 1),
                "reached_plumbing": bool(reached),
                "reach_pours": int(reach_pours),
                "perturbation_truth_boundary": "harness_authored_seeded_random_pre_arrange_moves_never_pours",
                "perturbation_log": perturb_log,
                "perturbation_actions": perturb_actions,
                "perturbed_frame_sha256": perturbed_hash,
                "replay_match": bool(replay_match),
                "replay_levels_completed": replay_levels,
                "replay_action_count": len(replay_trace),
                "final_state": res.get("final_state"),
                "status": res.get("status"),
                "error": res.get("error"),
                "source_histogram": summarise_sources(ledger),
                "max_consecutive_pours": max_consecutive_pours(ledger),
            }
        )
        if not replay_match:
            judged["passed"] = False
            judged.setdefault("checks", {})["paired_replay_match"] = False
        else:
            judged.setdefault("checks", {})["paired_replay_match"] = True
        rows["model"].append(judged)

        random_row = None
        if not args.skip_random and replay_match and fd_random is not None and cur_random is not None:
            random_row = run_random_baseline(
                env=env_random,
                fd=fd_random,
                cur=cur_random,
                prev=prev_random,
                model=model,
                device=device,
                rng=np.random.default_rng(episode_seed + 10_000_000),
                k_plans=args.random_plans,
            )
            random_row["episode_seed"] = episode_seed
            rows["random"].append(random_row)

        print(
            f"[perturb_gate:model] seed={episode_seed} passed={judged['passed']} "
            f"reached={reached} levels={judged['levels_completed']} "
            f"actions={judged['actions']} replay_match={replay_match} "
            f"checks={judged.get('checks')} wall={judged['wall_seconds']}s",
            flush=True,
        )
        if random_row is not None:
            print(
                f"[perturb_gate:random] seed={episode_seed} "
                f"levels={random_row['levels_completed']} actions={random_row['actions']} "
                f"status={random_row['status']}",
                flush=True,
            )

    model_pass = sum(1 for r in rows["model"] if r.get("passed"))
    random_ge2 = sum(1 for r in rows["random"] if int(r.get("levels_completed") or 0) >= 2)
    model_threshold = math.ceil(0.9 * max(args.episodes, 1))
    random_threshold = max(1, math.ceil(0.2 * max(len(rows["random"]), 1)))
    gate_verdict = (
        "PASS"
        if model_pass >= model_threshold
        and (args.skip_random or random_ge2 <= random_threshold)
        else "FAIL"
    )
    report = {
        "schema": "synthesis_lab.perturb_gate.v0",
        "condition": "Stage 1b local perturbation arrange gate: reach plumbing, harness-random pre-arrange perturb, drive live adapter path from perturbed state",
        "evidence": "local offline environment fields only; SCOUT_EXPERIMENT",
        "boundary": "Perturbations are harness-authored test-state generation, not model capability; not a live scorecard.",
        "source_condition_artifact": "synthesis_lab/notes/HANDOFF.md §0B Stage 1B",
        "source_hashes": {
            "perturb_gate": sha256_file(Path(__file__).resolve()),
            "adapter_policy": sha256_file(_LAB_ROOT / "orchestration" / "megarexus_live_policy.py"),
            "arc_suite_runner": sha256_file(_REPO_ROOT / "codexlab" / "arc_suite_runner.py"),
        },
        "game_id": GAME_ID,
        "checkpoint": str(args.ckpt),
        "checkpoint_sha256": sha256_file(args.ckpt.resolve()),
        "paired_seeds": [args.seed, args.seed + args.episodes - 1],
        "max_policy_actions": int(args.max_policy_actions),
        "perturb_rounds": [int(args.perturb_min_rounds), int(args.perturb_max_rounds)],
        "random_baseline": "one sampled arrangement plan plus one pour from replay-verified same perturbed state",
        "model_passed": model_pass,
        "model_episodes": len(rows["model"]),
        "model_pass_threshold": model_threshold,
        "random_levels_ge_2": random_ge2,
        "random_episodes": len(rows["random"]),
        "random_ge_2_threshold": None if args.skip_random else random_threshold,
        "gate_verdict": gate_verdict,
        "detail": rows,
    }
    out = out_dir / "perturb_gate.json"
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(
        f"[perturb_gate] verdict={gate_verdict} model {model_pass}/{len(rows['model'])} "
        f"pass (threshold {model_threshold}); random levels>=2: "
        f"{random_ge2}/{len(rows['random'])}; report -> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
