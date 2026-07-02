#!/usr/bin/env python3
"""Probe randomized lateral target sampling in the native PufferLib SP80 env."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import html
import json
import math
import platform
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "codexlab" / "runs" / "_system_run" / "rexus_sp80_paintflow_lateral_random_target_v0"
CONDITION_PATH = ROOT / "codexlab" / "conditions" / "rexus_sp80_paintflow_lateral_random_target_v0.json"
PUFFERLIB_PATH = ROOT / ".external" / "pufferlib-4.0-newjordan"

GRID_W = 14
GRID_H = 15
TARGET_LATERAL_MIN_X = 1
TARGET_LATERAL_MAX_X = 12
TARGET_ROWS = {6, 11}
OBS_SIZE = 12


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def run_git(args: list[str]) -> str | None:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def load_puffer_args(env_name: str, total_agents: int, num_buffers: int) -> tuple[Any, Any, dict[str, Any]]:
    sys.path.insert(0, str(PUFFERLIB_PATH))
    import pufferlib.pufferl as pufferl  # noqa: PLC0415
    import pufferlib._C as native  # noqa: PLC0415

    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        cfg = pufferl.load_config(env_name)
    finally:
        sys.argv = old_argv
    cfg["vec"]["total_agents"] = total_agents
    cfg["vec"]["num_buffers"] = num_buffers
    cfg["vec"]["num_threads"] = 1
    return pufferl, native, cfg


def read_float_obs(vec: Any) -> list[float]:
    if vec.obs_elem_size != ctypes.sizeof(ctypes.c_float) or vec.obs_dtype != "FloatTensor":
        raise RuntimeError(f"unexpected observation layout: dtype={vec.obs_dtype} elem_size={vec.obs_elem_size}")
    total = int(vec.total_agents) * int(vec.obs_size)
    ptr = ctypes.cast(int(vec.obs_ptr), ctypes.POINTER(ctypes.c_float))
    return [float(ptr[i]) for i in range(total)]


def obs_to_grid(value: float, scale: int) -> int:
    return int(round(value * float(scale - 1)))


def collect_samples(native: Any, cfg: dict[str, Any], samples: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vec = native.create_vec(cfg, 0)
    rows: list[dict[str, Any]] = []
    try:
        if int(vec.obs_size) != OBS_SIZE:
            raise RuntimeError(f"unexpected obs_size: {vec.obs_size}")
        resets = int(math.ceil(samples / float(vec.total_agents)))
        for reset_index in range(resets):
            vec.reset()
            obs = read_float_obs(vec)
            for agent_index in range(int(vec.total_agents)):
                if len(rows) >= samples:
                    break
                base = agent_index * int(vec.obs_size)
                bar_x = obs_to_grid(obs[base + 0], GRID_W)
                bar_y = obs_to_grid(obs[base + 1], GRID_H)
                target_family = int(round(obs[base + 2]))
                target_x = obs_to_grid(obs[base + 3], GRID_W)
                target_y = obs_to_grid(obs[base + 4], GRID_H)
                rows.append(
                    {
                        "sample_index": len(rows),
                        "reset_index": reset_index,
                        "agent_index": agent_index,
                        "target_family": target_family,
                        "bar_x": bar_x,
                        "bar_y": bar_y,
                        "target_x": target_x,
                        "target_y": target_y,
                        "raw_obs": [round(obs[base + i], 6) for i in range(int(vec.obs_size))],
                    }
                )
        metadata = {
            "obs_size": int(vec.obs_size),
            "obs_dtype": str(vec.obs_dtype),
            "obs_elem_size": int(vec.obs_elem_size),
            "total_agents": int(vec.total_agents),
            "gpu": int(vec.gpu),
        }
    finally:
        vec.close()
    return rows, metadata


def summarize(samples: list[dict[str, Any]], min_unique_x: int) -> dict[str, Any]:
    xs = [int(row["target_x"]) for row in samples]
    ys = [int(row["target_y"]) for row in samples]
    families = [int(row["target_family"]) for row in samples]
    unique_x = sorted(set(xs))
    unique_y = sorted(set(ys))
    checks = {
        "sample_count_positive": len(samples) > 0,
        "target_x_in_lateral_bounds": all(TARGET_LATERAL_MIN_X <= x <= TARGET_LATERAL_MAX_X for x in xs),
        "target_y_on_actual_rows": all(y in TARGET_ROWS for y in ys),
        "target_family_valid": all(family in (0, 1) for family in families),
        "target_x_varies": len(unique_x) >= min_unique_x,
        "target_x_not_fixed_at_base": unique_x != [6],
        "both_target_rows_sampled": set(unique_y) == TARGET_ROWS,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "sample_count": len(samples),
        "unique_target_x": unique_x,
        "unique_target_y": unique_y,
        "target_x_histogram": dict(sorted(Counter(xs).items())),
        "target_y_histogram": dict(sorted(Counter(ys).items())),
        "target_family_histogram": dict(sorted(Counter(families).items())),
        "expected_lateral_x_min": TARGET_LATERAL_MIN_X,
        "expected_lateral_x_max": TARGET_LATERAL_MAX_X,
        "expected_target_rows": sorted(TARGET_ROWS),
        "min_unique_x_required": min_unique_x,
    }


def write_index(out_dir: Path, result: dict[str, Any], sample_preview: list[dict[str, Any]]) -> None:
    summary = result["summary"]
    status = html.escape(str(result["status"]))
    hist_rows = "\n".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in summary["target_x_histogram"].items()
    )
    check_rows = "\n".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in summary["checks"].items()
    )
    sample_rows = "\n".join(
        "<tr>"
        f"<td>{row['sample_index']}</td><td>{row['target_family']}</td>"
        f"<td>{row['target_x']}</td><td>{row['target_y']}</td>"
        f"<td>{row['bar_x']}</td><td>{row['bar_y']}</td>"
        "</tr>"
        for row in sample_preview
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Rexus SP80 PufferLib Lateral Target Probe</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 28px; color: #17202a; background: #f7f8f9; }}
    main {{ max-width: 980px; margin: 0 auto; }}
    section {{ margin: 18px 0; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ border: 1px solid #d8dee4; padding: 7px 9px; text-align: left; }}
    th {{ background: #edf2f7; }}
    code {{ background: #edf2f7; padding: 2px 4px; }}
    .status {{ font-weight: 700; }}
    .boundary {{ color: #5c6773; }}
  </style>
</head>
<body>
<main>
  <h1>Rexus SP80 PufferLib Lateral Target Probe</h1>
  <p class="status">Status: {status}</p>
  <p class="boundary">{html.escape(result["truth_boundary"])}</p>
  <section>
    <h2>Condition</h2>
    <table>
      <tr><th>Field</th><th>Value</th></tr>
      <tr><td>condition</td><td>{html.escape(result["condition_id"])}</td></tr>
      <tr><td>compiled env</td><td>{html.escape(result["compiled_env"])}</td></tr>
      <tr><td>command</td><td><code>{html.escape(result["command"])}</code></td></tr>
      <tr><td>result json</td><td><a href="result.json">result.json</a></td></tr>
      <tr><td>samples jsonl</td><td><a href="samples.jsonl">samples.jsonl</a></td></tr>
    </table>
  </section>
  <section>
    <h2>Checks</h2>
    <table><tr><th>Check</th><th>Value</th></tr>{check_rows}</table>
  </section>
  <section>
    <h2>Target X Histogram</h2>
    <table><tr><th>target_x</th><th>count</th></tr>{hist_rows}</table>
  </section>
  <section>
    <h2>Sample Preview</h2>
    <table>
      <tr><th>sample</th><th>family</th><th>target_x</th><th>target_y</th><th>bar_x</th><th>bar_y</th></tr>
      {sample_rows}
    </table>
  </section>
</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default="rexus_sp80_paintflow")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--total-agents", type=int, default=64)
    parser.add_argument("--num-buffers", type=int, default=1)
    parser.add_argument("--min-unique-x", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _, native, cfg = load_puffer_args(args.env_name, args.total_agents, args.num_buffers)
    command = " ".join([sys.executable, "-B", *sys.argv])
    samples, vec_metadata = collect_samples(native, cfg, args.samples)
    summary = summarize(samples, args.min_unique_x)
    native_path = Path(getattr(native, "__file__", "")).resolve()
    result = {
        "schema": "codexlab.pufferlib_lateral_target_probe.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "condition_id": "rexus_sp80_paintflow_lateral_random_target_v0",
        "condition_path": rel(CONDITION_PATH),
        "status": summary["status"],
        "truth_boundary": (
            "compiled native PufferLib env reset sampling probe; confirms randomized target observations only, "
            "not trained policy quality, not full SP80 gameplay control, and not CONTRA solve evidence"
        ),
        "command": command,
        "cwd": str(ROOT),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "git": {
            "commit": run_git(["rev-parse", "HEAD"]),
            "status_short": run_git(["status", "--short"]),
        },
        "compiled_env": str(getattr(native, "env_name", None)),
        "native_extension": {
            "path": rel(native_path),
            "sha256": sha256_file(native_path),
            "gpu_backend": bool(getattr(native, "gpu", False)),
            "precision_bytes": int(getattr(native, "precision_bytes", 0)),
        },
        "vec_metadata": vec_metadata,
        "source_hashes": {
            rel(ROOT / "codexlab" / "pufferlib_lateral_target_probe.py"): sha256_file(
                ROOT / "codexlab" / "pufferlib_lateral_target_probe.py"
            ),
            rel(ROOT / "codexlab" / "pufferlib_envs" / "rexus_sp80_paintflow" / "rexus_sp80_paintflow.h"): sha256_file(
                ROOT / "codexlab" / "pufferlib_envs" / "rexus_sp80_paintflow" / "rexus_sp80_paintflow.h"
            ),
            rel(ROOT / "codexlab" / "pufferlib_envs" / "rexus_sp80_paintflow" / "binding.c"): sha256_file(
                ROOT / "codexlab" / "pufferlib_envs" / "rexus_sp80_paintflow" / "binding.c"
            ),
            rel(CONDITION_PATH): sha256_file(CONDITION_PATH),
        },
        "summary": summary,
        "artifacts": {
            "result_json": rel(out_dir / "result.json"),
            "samples_jsonl": rel(out_dir / "samples.jsonl"),
            "index_html": rel(out_dir / "index.html"),
        },
    }
    with (out_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_index(out_dir, result, samples[:32])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
