#!/usr/bin/env python3
"""Download ARC-AGI-3 game environment files for OFFLINE local play.

Recon lane tool (2026-07-06). Mirrors what arc_agi.Arcade._download_game does,
but WITHOUT constructing an Arcade (so no scorecard of any kind is opened and
nothing is fetched beyond metadata + source). Uses the public anonymous key
endpoint. Output layout matches environment_files/sp80/589a99af exactly:

    <dest>/<base_id>/<version>/metadata.json
    <dest>/<base_id>/<version>/<classname>.py

Usage:
    python download_game.py ls20-9607627b ft09-0d8bbf25 \
        --dest synthesis_lab/experiments/other_games/environment_files
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://three.arcprize.org"


def get_anon_key() -> str:
    r = requests.get(f"{BASE_URL}/api/games/anonkey", headers={"Accept": "application/json"}, timeout=15)
    r.raise_for_status()
    return str(r.json()["api_key"])


def download(game_id: str, dest: Path, api_key: str) -> Path:
    if "-" not in game_id:
        raise SystemExit(f"need full id like ls20-9607627b, got {game_id}")
    base_id, version = game_id.split("-", 1)
    headers = {"X-Api-Key": api_key, "Accept": "application/json"}

    meta_r = requests.get(f"{BASE_URL}/api/games/{game_id}", headers=headers, timeout=15)
    meta_r.raise_for_status()
    metadata = meta_r.json()

    src_r = requests.get(f"{BASE_URL}/api/games/{game_id}/source", headers=headers, timeout=30)
    src_r.raise_for_status()
    source_code = src_r.text

    env_dir = dest / base_id / version
    env_dir.mkdir(parents=True, exist_ok=True)

    class_name = metadata.get("class_name") or (base_id[0].upper() + base_id[1:])
    metadata.setdefault("game_id", game_id)
    metadata.setdefault("tags", [])
    metadata.setdefault("baseline_actions", [])
    metadata["local_dir"] = str(env_dir)
    metadata["date_downloaded"] = datetime.now(timezone.utc).isoformat()

    (env_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (env_dir / f"{class_name.lower()}.py").write_text(source_code, encoding="utf-8")
    print(f"[ok] {game_id} -> {env_dir} ({len(source_code.splitlines())} source lines, class {class_name})")
    return env_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("game_ids", nargs="+")
    ap.add_argument("--dest", type=Path,
                    default=Path(__file__).resolve().parent / "environment_files")
    args = ap.parse_args()
    key = get_anon_key()
    for gid in args.game_ids:
        download(gid, args.dest, key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
