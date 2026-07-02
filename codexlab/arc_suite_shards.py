#!/usr/bin/env python3
"""Print offline-local ARC suite shard commands.

This helper only discovers copied local game IDs under
environment/environment_files and prints commands for codexlab/arc_suite_runner.py.
It does not contact live ARC, create live scorecards, or execute the commands.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ENV_DIR = ROOT / "environment" / "environment_files"
RUNNER = Path("codexlab") / "arc_suite_runner.py"
SCHEMA = "codexlab.arc_suite_shards.v1"
BOUNDARY = (
    "OFFLINE-LOCAL SCOUT ONLY: generated commands target copied "
    "environment/environment_files through codexlab/arc_suite_runner.py. "
    "They are not live ARC Prize submissions, public scorecards, leaderboard "
    "proof, hidden-answer routes, replay routes, control truth, or solve claims."
)


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def discover_game_ids(env_dir: Path = ENV_DIR) -> list[str]:
    """Discover local game IDs using the same prefix-ident shape as the runner."""
    game_ids: list[str] = []
    if not env_dir.exists():
        return game_ids
    for prefix_dir in sorted(p for p in env_dir.iterdir() if p.is_dir()):
        for ident_dir in sorted(p for p in prefix_dir.iterdir() if p.is_dir()):
            has_metadata = (ident_dir / "metadata.json").exists()
            has_source = any(ident_dir.glob("*.py"))
            if has_metadata or has_source:
                game_ids.append(f"{prefix_dir.name}-{ident_dir.name}")
    return game_ids


def split_shards(items: list[str], shard_count: int) -> list[list[str]]:
    base, extra = divmod(len(items), shard_count)
    shards: list[list[str]] = []
    offset = 0
    for index in range(shard_count):
        size = base + (1 if index < extra else 0)
        shards.append(items[offset : offset + size])
        offset += size
    return shards


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def safe_slug(value: str) -> str:
    chars = [ch.lower() if ch.isalnum() else "_" for ch in value.strip()]
    slug = "".join(chars).strip("_")
    return slug or "arc_suite"


def shard_label(index: int, total: int) -> str:
    width = max(2, len(str(total)))
    return f"shard{index + 1:0{width}d}_of{total:0{width}d}"


def build_shard_records(args: argparse.Namespace, games: list[str]) -> list[dict[str, Any]]:
    shards = split_shards(games, int(args.shards))
    records: list[dict[str, Any]] = []
    default_prefix = f"arc_suite_{safe_slug(args.policy)}"
    out_root = Path(args.out_root) if args.out_root else None
    for index, shard_games in enumerate(shards):
        label = shard_label(index, int(args.shards))
        run_id = f"{args.run_id_prefix}_{label}" if args.run_id_prefix else None
        out_dir = None
        if out_root is not None:
            out_dir_name = run_id or f"{default_prefix}_{label}"
            out_dir = out_root / out_dir_name

        runner_args = [
            "python",
            "-B",
            str(RUNNER),
            "--policy",
            str(args.policy),
            "--max-actions",
            str(int(args.max_actions)),
            "--games",
            *shard_games,
        ]
        if out_dir is not None:
            runner_args.extend(["--out-dir", str(out_dir)])
        if run_id is not None:
            runner_args.extend(["--run-id", run_id])

        command_parts = [
            "ARC_ALLOW_LIVE=0",
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONPATH=codexlab",
            *runner_args,
        ]
        records.append(
            {
                "index": index + 1,
                "label": label,
                "game_count": len(shard_games),
                "games": shard_games,
                "run_id": run_id,
                "out_dir": str(out_dir) if out_dir is not None else None,
                "runner_args": runner_args,
                "shell_command": shell_join(command_parts),
            }
        )
    return records


def render_text(payload: dict[str, Any]) -> str:
    lines = [
        "# Offline-local ARC suite shard commands",
        f"# {payload['boundary']}",
        f"# cwd: {payload['cwd']}",
        f"# discovered_games: {payload['game_count']} shards: {payload['shard_count']}",
        "# These commands are printed for manual/offline scouting; this helper does not execute them.",
        "",
    ]
    for shard in payload["shards"]:
        games = " ".join(shard["games"])
        lines.extend(
            [
                f"# {shard['label']}: {shard['game_count']} games",
                f"# games: {games}",
                shard["shell_command"],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_shell(payload: dict[str, Any]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Offline-local ARC suite shard commands.",
        f"# {payload['boundary']}",
        f"# cwd: {payload['cwd']}",
        "# Run from the repository root. The commands set ARC_ALLOW_LIVE=0 explicitly.",
        "",
    ]
    for shard in payload["shards"]:
        lines.extend(
            [
                f"# {shard['label']}: {shard['game_count']} local games",
                shard["shell_command"],
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print offline-local shard commands for codexlab/arc_suite_runner.py."
    )
    parser.add_argument("--shards", type=int, default=1, help="Number of non-empty shards to print.")
    parser.add_argument("--max-actions", type=int, default=250, help="Passed through to arc_suite_runner.py.")
    parser.add_argument("--policy", default="tufa_action", help="Passed through to arc_suite_runner.py.")
    parser.add_argument("--out-root", default=None, help="Optional parent directory for per-shard --out-dir values.")
    parser.add_argument("--run-id-prefix", default=None, help="Optional prefix for per-shard --run-id values.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--shell", action="store_true", help="Print a runnable bash script.")
    group.add_argument("--json", action="store_true", help="Print a machine-readable JSON manifest.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.shards < 1:
        parser.error("--shards must be at least 1")
    if args.max_actions < 0:
        parser.error("--max-actions must be non-negative")

    games = discover_game_ids()
    if not games:
        print(f"no local games found under {repo_relative(ENV_DIR)}", file=sys.stderr)
        return 2
    if args.shards > len(games):
        parser.error(
            f"--shards {args.shards} exceeds discovered local game count {len(games)}; "
            "empty --games shards are refused because the runner would default to all local games"
        )

    payload = {
        "schema": SCHEMA,
        "boundary": BOUNDARY,
        "cwd": str(ROOT),
        "environment_dir": repo_relative(ENV_DIR),
        "runner": str(RUNNER),
        "policy": str(args.policy),
        "max_actions": int(args.max_actions),
        "shard_count": int(args.shards),
        "game_count": len(games),
        "games": games,
        "shards": build_shard_records(args, games),
    }

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.shell:
        print(render_shell(payload), end="")
    else:
        print(render_text(payload), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
