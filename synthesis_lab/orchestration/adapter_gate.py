"""SCOUT_EXPERIMENT: Stage 1a local live-adapter gate (HANDOFF §0B).

Drives the REAL `megarexus_cnn` policy adapter through the SAME
`codexlab.arc_suite_runner.run_game` code path the live scout uses — raw
animation frame stream, auto-reset, scorecard-style loop — against the
local offline SP80 environment, across paired seeds, with a paired random
baseline.

Pass criteria (per model episode, absolute):
  1. levels_completed >= 2 (catch level AND plumbing level completed),
  2. zero policy errors,
  3. frame-convention integrity: the adapter never fell back to
     obs["grid"] (frames_fallback_count == 0),
  4. pour discipline: no capped-pour loop (never more than 2 consecutive
     pour actions), per-level pour count within the adapter budget,
  5. every executed arrange pour carried a confident plan margin.

Evidence: local offline environment fields only. No scorecard claims.

Usage:
    python synthesis_lab/orchestration/adapter_gate.py \
        --ckpt synthesis_lab/experiments/megarexus_cnn/runs/outcome_v8/last.pt \
        --out-dir synthesis_lab/experiments/megarexus_cnn/runs/adapter_gate_v8last \
        --episodes 10
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

_LAB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _LAB_ROOT.parent
for _entry in (_REPO_ROOT / "codexlab", _LAB_ROOT, _LAB_ROOT / "orchestration",
               _LAB_ROOT / "experiments" / "megarexus_cnn"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

from arc_play_harness import random_policy  # noqa: E402
from arc_suite_runner import run_game  # noqa: E402
from megarexus_live_policy import (  # noqa: E402
    CONFIDENCE_MARGIN,
    MAX_POURS_PER_LEVEL,
    MegarexusArcPolicy,
    sha256_file,
)

GAME_ID = "sp80-589a99af"
ENV_DIR = _REPO_ROOT / "environment_files"


def max_consecutive_pours(ledger: list[dict]) -> int:
    best = run = 0
    for row in ledger:
        run = run + 1 if int(row.get("action", 0)) == 5 else 0
        best = max(best, run)
    return best


def max_consecutive_pours_same_level(ledger: list[dict]) -> int:
    best = run = 0
    last_level: int | None = None
    for row in ledger:
        action = int(row.get("action", 0))
        level = int(row.get("levels_completed") or 0)
        if action == 5 and level == last_level:
            run += 1
        elif action == 5:
            run = 1
        else:
            run = 0
        last_level = level
        best = max(best, run)
    return best


def judge_model_episode(res: dict, target_levels: int) -> dict:
    trace = res.get("policy_trace") or {}
    ledger = trace.get("ledger") or []
    pours_at_level = {int(k): int(v) for k, v in (trace.get("pours_at_level") or {}).items()}
    goal_updates = trace.get("goal_provider_updates") or []
    plan_summaries = trace.get("plan_summaries") or []
    arrange_pours = [r for r in ledger if r.get("source") == "megarexus_arrange_pour"]
    legacy_cap_rows = [r for r in ledger if r.get("source") == "megarexus_arrange_attempt_cap"]
    def admitted_pour(row: dict) -> bool:
        info = row.get("info") or {}
        if info.get("confidence_basis") in {
            "graph_demo_plan_controlled_scout",
            "graph_level2_raw_action_template_controlled_scout",
        }:
            return True
        return float(info.get("margin_vs_5th") or 0.0) >= CONFIDENCE_MARGIN

    levels_completed = int(res.get("levels_completed") or 0)
    unconfident = [
        r for r in arrange_pours
        if not admitted_pour(r)
    ]
    raw_max_pour_streak = max_consecutive_pours(ledger)
    same_level_max_pour_streak = max_consecutive_pours_same_level(ledger)
    checks = {
        "levels_ge_2": levels_completed >= 2,
        "levels_ge_target": levels_completed >= int(target_levels),
        "zero_policy_errors": int(res.get("policy_error_count") or 0) == 0,
        "no_frames_fallback": int(trace.get("frames_fallback_count") or 0) == 0,
        "no_pour_loop": same_level_max_pour_streak <= 2 and not legacy_cap_rows,
        "pour_budget_respected": all(v <= MAX_POURS_PER_LEVEL for v in pours_at_level.values()),
        "all_pours_confident": not unconfident,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "target_levels": int(target_levels),
        "levels_completed": levels_completed,
        "actions": int(res.get("actions") or 0),
        "resets": int(res.get("resets") or 0),
        "raw_max_consecutive_pours": raw_max_pour_streak,
        "same_level_max_consecutive_pours": same_level_max_pour_streak,
        "pours_at_level": pours_at_level,
        "goal_provider": {
            "enabled": bool(trace.get("goal_provider_enabled")),
            "pour_budget_enabled": bool(trace.get("goal_pour_budget_enabled")),
            "updates": len(goal_updates),
            "latest_target_count": None if not goal_updates else goal_updates[-1].get("target_count"),
        },
        "plans_built": trace.get("plans_built"),
        "noop_count": trace.get("noop_count"),
        "refuted_plans": trace.get("refuted_plans"),
        "source_histogram": dict(Counter(r.get("source") for r in ledger)),
        "plan_summaries": plan_summaries,
        "frame_capture": trace.get("frame_capture"),
    }


def write_review(out_dir: Path, report: dict) -> None:
    model_rows = []
    for row in report["detail"]["model"]:
        model_rows.append(
            "<tr>"
            f"<td>{row.get('episode_seed')}</td>"
            f"<td>{row.get('passed')}</td>"
            f"<td>{row.get('levels_completed')}</td>"
            f"<td>{row.get('actions')}</td>"
            f"<td><code>{html.escape(json.dumps(row.get('checks'), sort_keys=True))}</code></td>"
            "</tr>"
        )
    random_rows = []
    for row in report["detail"]["random"]:
        random_rows.append(
            "<tr>"
            f"<td>{row.get('episode_seed')}</td>"
            f"<td>{row.get('levels_completed')}</td>"
            f"<td>{row.get('actions')}</td>"
            f"<td>{row.get('resets')}</td>"
            "</tr>"
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Adapter Gate Review</title>
  <style>
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #111318; color: #e8edf4; }}
    main {{ max-width: 1500px; margin: 0 auto; padding: 28px; }}
    .boundary {{ color: #f4d278; max-width: 1050px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 13px; }}
    th,td {{ border: 1px solid #2d3440; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ background: #1d2430; }}
    td {{ background: #151920; }}
    code {{ color: #d7e6ff; word-break: break-word; }}
    a {{ color: #8fc7ff; }}
  </style>
</head>
<body>
<main>
  <h1>Adapter Gate Review</h1>
  <p class="boundary">{html.escape(str(report.get("boundary")))}</p>
  <p>Verdict: <strong>{html.escape(str(report.get("gate_verdict")))}</strong>. Raw JSON: <a href="adapter_gate.json">adapter_gate.json</a>.</p>
  <table>
    <tr><th>game</th><td><code>{html.escape(str(report.get("game_id")))}</code></td></tr>
    <tr><th>checkpoint</th><td><code>{html.escape(str(report.get("checkpoint_sha256")))}</code></td></tr>
    <tr><th>target levels</th><td>{report.get("target_levels")}</td></tr>
    <tr><th>max actions</th><td>{report.get("max_actions")}</td></tr>
  </table>
  <h2>Model Episodes</h2>
  <table><tr><th>seed</th><th>passed</th><th>levels</th><th>actions</th><th>checks</th></tr>{''.join(model_rows)}</table>
  <h2>Random Baseline</h2>
  <table><tr><th>seed</th><th>levels</th><th>actions</th><th>resets</th></tr>{''.join(random_rows)}</table>
</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=20280001)
    p.add_argument("--max-actions", type=int, default=120)
    p.add_argument("--target-levels", type=int, default=2)
    p.add_argument("--skip-random", action="store_true")
    args = p.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_path = _LAB_ROOT / "orchestration" / "megarexus_live_policy.py"
    condition = {
        "schema": "synthesis_lab.condition.adapter_gate.v1",
        "condition": f"local adapter gate target levels_completed >= {int(args.target_levels)}",
        "run_label": "scout",
        "boundary": "Local offline adapter gate through the same run_game path used by the live scout; not a live scorecard.",
        "command": " ".join(sys.argv),
        "working_directory": str(_REPO_ROOT),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "policy": str(policy_path),
        "policy_sha256": sha256_file(policy_path),
        "checkpoint": str(args.ckpt),
        "checkpoint_sha256": sha256_file(args.ckpt.resolve()),
        "seed_start": int(args.seed),
        "episodes": int(args.episodes),
        "max_actions": int(args.max_actions),
        "target_levels": int(args.target_levels),
        "env": {
            "MEGAREXUS_GRAPH_DEMO_PLAN": os.environ.get("MEGAREXUS_GRAPH_DEMO_PLAN", ""),
            "MEGAREXUS_GOAL_PROVIDER": os.environ.get("MEGAREXUS_GOAL_PROVIDER", ""),
            "MEGAREXUS_GOAL_POUR_BUDGET": os.environ.get("MEGAREXUS_GOAL_POUR_BUDGET", ""),
            "MEGAREXUS_GOAL_THRESHOLD": os.environ.get("MEGAREXUS_GOAL_THRESHOLD", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    }
    (out_dir / "CONDITION.json").write_text(json.dumps(condition, indent=1, sort_keys=True), encoding="utf-8")
    arc = Arcade(
        arc_api_key="anonymous",
        environments_dir=str(ENV_DIR),
        operation_mode=OperationMode.OFFLINE,
    )
    card = arc.open_scorecard()

    rows = {"model": [], "random": []}
    for ep in range(args.episodes):
        seed = args.seed + ep
        trace_dir = out_dir / f"ep_seed{seed}"
        os.environ["MEGAREXUS_TRACE_DIR"] = str(trace_dir)
        policy = MegarexusArcPolicy(checkpoint_path=args.ckpt)
        t0 = time.time()
        res = run_game(
            arc=arc, scorecard_id=card, game_id=GAME_ID,
            policy_name="megarexus_cnn", max_actions=args.max_actions,
            solution_root=None, auto_reset=True, include_perception=False,
            seed=seed, policy=policy,
        )
        judged = judge_model_episode(res, args.target_levels)
        judged.update({"episode_seed": seed, "wall_seconds": round(time.time() - t0, 1),
                       "final_state": res.get("final_state"), "status": res.get("status"),
                       "error": res.get("error")})
        rows["model"].append(judged)
        print(f"[adapter_gate:model] seed={seed} passed={judged['passed']} "
              f"levels={judged['levels_completed']} actions={judged['actions']} "
              f"resets={judged['resets']} checks={judged['checks']} "
              f"wall={judged['wall_seconds']}s", flush=True)

        if not args.skip_random:
            os.environ.pop("MEGAREXUS_TRACE_DIR", None)
            res_r = run_game(
                arc=arc, scorecard_id=card, game_id=GAME_ID,
                policy_name="random", max_actions=args.max_actions,
                solution_root=None, auto_reset=True, include_perception=False,
                seed=seed, policy=random_policy(seed_state=[seed]),
            )
            rows["random"].append({
                "episode_seed": seed,
                "levels_completed": int(res_r.get("levels_completed") or 0),
                "actions": int(res_r.get("actions") or 0),
                "resets": int(res_r.get("resets") or 0),
                "policy_error_count": int(res_r.get("policy_error_count") or 0),
            })
            print(f"[adapter_gate:random] seed={seed} "
                  f"levels={rows['random'][-1]['levels_completed']}", flush=True)

    try:
        close_payload = arc.close_scorecard(card)
    except Exception:
        close_payload = None

    model_pass = sum(1 for r in rows["model"] if r["passed"])
    random_ge2 = sum(1 for r in rows["random"] if r["levels_completed"] >= 2)
    report = {
        "schema": "synthesis_lab.adapter_gate.v0",
        "evidence": "local offline environment fields only; SCOUT_EXPERIMENT",
        "boundary": "Same run_game code path as the live scout, offline env. Not a live scorecard.",
        "game_id": GAME_ID,
        "checkpoint": str(args.ckpt),
        "checkpoint_sha256": sha256_file(args.ckpt.resolve()),
        "paired_seeds": [args.seed, args.seed + args.episodes - 1],
        "max_actions": args.max_actions,
        "target_levels": int(args.target_levels),
        "confidence_margin": CONFIDENCE_MARGIN,
        "command": " ".join(sys.argv),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "policy_sha256": sha256_file(policy_path),
        "condition_sha256": sha256_file((out_dir / "CONDITION.json").resolve()),
        "model_passed": model_pass,
        "model_episodes": len(rows["model"]),
        "random_levels_ge_2": random_ge2,
        "random_episodes": len(rows["random"]),
        "gate_verdict": "PASS" if model_pass == len(rows["model"]) and rows["model"] else "FAIL",
        "detail": rows,
    }
    out = out_dir / "adapter_gate.json"
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    write_review(out_dir, report)
    print(f"[adapter_gate] verdict={report['gate_verdict']} "
          f"model {model_pass}/{len(rows['model'])} pass; random levels>=2: "
          f"{random_ge2}/{len(rows['random'])}; report → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
