#!/usr/bin/env python3
"""Run a bounded native PufferLib training scout for the Rexus SP80 env."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import html
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PUFFERLIB_PATH = ROOT / ".external" / "pufferlib-4.0-newjordan"
DEFAULT_CONDITION_ID = "rexus_sp80_paintflow_native_lateral_train_v0"
DEFAULT_RUN_DIR = ROOT / "codexlab" / "runs" / "_system_run" / DEFAULT_CONDITION_ID
DEFAULT_ENV_NAME = "rexus_sp80_paintflow"
TRAIN_SCHEMA = "codexlab.pufferlib_native_train_run.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def run_command(args: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    completed = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "command": " ".join(args),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_metadata() -> dict[str, Any]:
    return {
        "commit": run_command(["git", "rev-parse", "HEAD"])["stdout"],
        "status_short": run_command(["git", "status", "--short"])["stdout"],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fresh_run_dir(path: Path) -> Path:
    if not path.exists() or not any(path.iterdir()):
        return path
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return path.with_name(f"{path.name}_{stamp}")


def source_hashes(env_name: str) -> dict[str, str | None]:
    paths = [
        ROOT / "codexlab" / "pufferlib_native_train_run.py",
        ROOT / "codexlab" / "conditions" / "rexus_sp80_paintflow_lateral_random_target_v0.json",
        PUFFERLIB_PATH / "pufferlib" / "_C.cpython-313-aarch64-linux-gnu.so",
    ]
    env_dir = ROOT / "codexlab" / "pufferlib_envs" / env_name
    paths.extend(
        [
            env_dir / f"{env_name}.h",
            env_dir / "binding.c",
            env_dir / f"{env_name}.ini",
        ]
    )
    return {rel(path) or str(path): sha256_file(path) for path in paths}


def load_args(env_name: str) -> tuple[Any, Any, dict[str, Any]]:
    sys.path.insert(0, str(PUFFERLIB_PATH))
    import pufferlib.pufferl as pufferl  # noqa: PLC0415
    import pufferlib._C as native  # noqa: PLC0415

    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        cfg = pufferl.load_config(env_name)
    finally:
        sys.argv = old_argv
    return pufferl, native, cfg


def latest_file(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime if path.exists() else 0.0)
    return matches[-1] if matches else None


def load_training_metrics(log_path: Path | None) -> dict[str, Any]:
    if log_path is None or not log_path.is_file():
        return {"available": False}
    payload = json.loads(log_path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") if isinstance(payload, dict) else None
    if not isinstance(metrics, dict):
        return {"available": False, "log_path": rel(log_path)}
    final = {}
    for key, values in metrics.items():
        if isinstance(values, list) and values:
            final[key] = values[-1]
    return {
        "available": True,
        "log_path": rel(log_path),
        "metric_keys": sorted(metrics.keys()),
        "final": final,
    }


def write_index(run_dir: Path, manifest: dict[str, Any]) -> None:
    result = manifest.get("result", {})
    final = (result.get("metrics") or {}).get("final") or {}
    metric_rows = "\n".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in sorted(final.items())
        if key in {
            "agent_steps",
            "env/target_clean_catch",
            "env/level_completion",
            "env/overflow",
            "env/lateral_target_x_min",
            "env/lateral_target_x_mean",
            "env/lateral_target_x_max",
            "loss/policy",
            "loss/value",
            "SPS",
        }
    )
    artifacts = manifest.get("artifacts", {})
    artifact_rows = "\n".join(
        f"<tr><td>{html.escape(str(key))}</td><td><a href=\"{html.escape(Path(value).name)}\">{html.escape(str(value))}</a></td></tr>"
        for key, value in artifacts.items()
        if isinstance(value, str) and Path(value).parent == run_dir
    )
    checkpoint_rows = "\n".join(
        f"<tr><td>{html.escape(str(path))}</td></tr>"
        for path in (result.get("checkpoints") or [])
    )
    text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(manifest["condition_id"])}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 28px; color: #17202a; background: #f7f8f9; }}
    main {{ max-width: 1040px; margin: 0 auto; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 12px 0 22px; }}
    th, td {{ border: 1px solid #d8dee4; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #edf2f7; }}
    code {{ background: #edf2f7; padding: 2px 4px; }}
    .boundary {{ color: #5c6773; }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(manifest["condition_id"])}</h1>
  <p>Status: <b>{html.escape(str(manifest["status"]))}</b></p>
  <p class="boundary">{html.escape(manifest["truth_boundary"])}</p>
  <table>
    <tr><th>Field</th><th>Value</th></tr>
    <tr><td>env</td><td>{html.escape(manifest["env_name"])}</td></tr>
    <tr><td>command</td><td><code>{html.escape(manifest["command"])}</code></td></tr>
    <tr><td>native extension</td><td>{html.escape(str(manifest.get("native_extension", {}).get("sha256")))}</td></tr>
  </table>
  <h2>Final Metrics</h2>
  <table><tr><th>Metric</th><th>Value</th></tr>{metric_rows}</table>
  <h2>Checkpoints</h2>
  <table><tr><th>Path</th></tr>{checkpoint_rows}</table>
  <h2>Artifacts</h2>
  <table><tr><th>Name</th><th>Path</th></tr>{artifact_rows}</table>
</main>
</body>
</html>
"""
    (run_dir / "index.html").write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-id", default=DEFAULT_CONDITION_ID)
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--total-timesteps", type=int, default=1_048_576)
    parser.add_argument("--total-agents", type=int, default=1024)
    parser.add_argument("--num-buffers", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--minibatch-size", type=int, default=4096)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=73)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = fresh_run_dir(args.run_dir.expanduser().resolve())
    run_dir.mkdir(parents=True, exist_ok=True)

    command = " ".join([sys.executable, "-B", *sys.argv])
    pufferl, native, cfg = load_args(args.env_name)
    cfg["wandb"] = False
    cfg["checkpoint_dir"] = str(run_dir / "checkpoints")
    cfg["log_dir"] = str(run_dir / "logs")
    cfg["checkpoint_interval"] = args.checkpoint_interval
    cfg["eval_episodes"] = args.eval_episodes
    cfg["seed"] = args.seed
    cfg["train"]["seed"] = args.seed
    cfg["train"]["total_timesteps"] = args.total_timesteps
    cfg["train"]["horizon"] = args.horizon
    cfg["train"]["minibatch_size"] = args.minibatch_size
    cfg["train"]["gpus"] = 1
    cfg["vec"]["total_agents"] = args.total_agents
    cfg["vec"]["num_buffers"] = args.num_buffers
    cfg["vec"]["num_threads"] = args.num_threads
    cfg["sweep"]["metric"] = "level_completion"

    condition = {
        "schema": "codexlab.pufferlib_native_train_condition.v1",
        "condition_id": args.condition_id,
        "created_at_utc": utc_now(),
        "run_label": "new_experiment",
        "env_name": args.env_name,
        "natural_language_goal": (
            "Train a native PufferLib policy scout on the randomized lateral SP80 paintflow target env "
            "to test whether the trainer can optimize catch-position control signals."
        ),
        "truth_boundary": (
            "native PufferLib training scout on the level-1 SP80 paintflow abstraction; not full Python SP80 control, "
            "not CONTRA gameplay solve evidence, and not exported policy evidence"
        ),
        "resolved_overrides": {
            "total_timesteps": args.total_timesteps,
            "total_agents": args.total_agents,
            "num_buffers": args.num_buffers,
            "num_threads": args.num_threads,
            "horizon": args.horizon,
            "minibatch_size": args.minibatch_size,
            "checkpoint_interval": args.checkpoint_interval,
            "eval_episodes": args.eval_episodes,
            "seed": args.seed,
        },
        "metric": "env/level_completion",
    }
    write_json(run_dir / "condition.json", condition)
    write_json(run_dir / "resolved_config.json", cfg)

    manifest: dict[str, Any] = {
        "schema": TRAIN_SCHEMA,
        "condition_id": args.condition_id,
        "created_at_utc": condition["created_at_utc"],
        "status": "running",
        "env_name": args.env_name,
        "command": command,
        "cwd": str(ROOT),
        "truth_boundary": condition["truth_boundary"],
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "gpu_runtime": {
            "nvidia_smi": run_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])["stdout"],
            "nvcc": run_command(["nvcc", "--version"])["stdout"],
        },
        "git": git_metadata(),
        "native_extension": {
            "env_name": str(getattr(native, "env_name", None)),
            "gpu_backend": bool(getattr(native, "gpu", False)),
            "precision_bytes": int(getattr(native, "precision_bytes", 0)),
            "path": rel(Path(getattr(native, "__file__", "")).resolve()),
            "sha256": sha256_file(Path(getattr(native, "__file__", "")).resolve()),
        },
        "source_hashes": source_hashes(args.env_name),
        "artifacts": {
            "condition": rel(run_dir / "condition.json"),
            "resolved_config": rel(run_dir / "resolved_config.json"),
            "manifest": rel(run_dir / "manifest.json"),
            "stdout": rel(run_dir / "stdout.log"),
            "stderr": rel(run_dir / "stderr.log"),
            "index": rel(run_dir / "index.html"),
        },
    }
    write_json(run_dir / "manifest.json", manifest)

    started = time.perf_counter()
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    train_error = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                pufferl.train(args.env_name, args=cfg)
    except Exception as exc:  # noqa: BLE001 - manifest records exact training failure.
        train_error = {"type": type(exc).__name__, "message": str(exc)}

    elapsed = time.perf_counter() - started
    log_root = run_dir / "logs" / args.env_name
    checkpoint_root = run_dir / "checkpoints" / args.env_name
    log_json = latest_file(log_root, "*.json")
    checkpoints = sorted(checkpoint_root.glob("**/*.bin")) if checkpoint_root.exists() else []
    metrics = load_training_metrics(log_json)
    manifest["completed_at_utc"] = utc_now()
    manifest["elapsed_seconds"] = elapsed
    manifest["status"] = "failed" if train_error else ("passed" if metrics.get("available") and checkpoints else "failed")
    if train_error:
        manifest["error"] = train_error
    manifest["result"] = {
        "metrics": metrics,
        "checkpoint_count": len(checkpoints),
        "checkpoints": [rel(path) for path in checkpoints],
        "latest_checkpoint": rel(checkpoints[-1]) if checkpoints else None,
        "puffer_log_json": rel(log_json) if log_json else None,
    }
    manifest["artifacts"]["puffer_log_json"] = rel(log_json) if log_json else None
    manifest["artifacts"]["latest_checkpoint"] = rel(checkpoints[-1]) if checkpoints else None
    write_json(run_dir / "manifest.json", manifest)
    write_index(run_dir, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
