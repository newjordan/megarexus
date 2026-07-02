#!/usr/bin/env python3
"""Download public ARC Prize scorecard recordings into a local corpus."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any


SCORECARD_URL = "https://arcprize.org/api/models/scorecard/{config_id}"
RECORDING_URL = "https://arcprize.org/api/recordings/{game_id}/{guid}"


def utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def fetch_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> tuple[int, int]:
    event_count = 0
    byte_count = 0
    with urllib.request.urlopen(url, timeout=180) as response, path.open("wb") as out:
        for raw_line in response:
            out.write(raw_line)
            byte_count += len(raw_line)
            if raw_line.strip():
                event_count += 1
    return event_count, byte_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-id", default="anthropic-opus-4-8-high")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir or f"codexlab/runs/arc_public_recordings_{args.config_id}_{utc_now_compact()}")
    raw_dir = out_dir / "raw_recordings"
    raw_dir.mkdir(parents=True, exist_ok=True)

    scorecard_url = SCORECARD_URL.format(config_id=args.config_id)
    scorecard = fetch_json(scorecard_url)
    scorecard_path = out_dir / "scorecard.json"
    scorecard_path.write_text(json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8")

    runs = []
    for env in scorecard.get("environments") or []:
        game_id = str(env["id"]).strip()
        for run in env.get("runs") or []:
            guid = run.get("guid")
            if guid:
                runs.append((game_id, run))
    if args.limit:
        runs = runs[: args.limit]

    recording_rows = []
    for index, (game_id, run) in enumerate(runs, start=1):
        guid = run["guid"]
        url = RECORDING_URL.format(game_id=game_id, guid=guid)
        path = raw_dir / f"{game_id}__{guid}.jsonl"
        print(f"[{index}/{len(runs)}] download {game_id} {guid}", file=sys.stderr)
        event_count, byte_count = download(url, path)
        recording_rows.append(
            {
                "game_id": game_id,
                "guid": guid,
                "recording_url": url,
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": byte_count,
                "events": event_count,
                "score": run.get("score"),
                "levels_completed": run.get("levels_completed"),
                "actions": run.get("actions"),
                "resets": run.get("resets"),
                "state": run.get("state"),
                "level_scores": run.get("level_scores"),
                "level_actions": run.get("level_actions"),
                "level_baseline_actions": run.get("level_baseline_actions"),
            }
        )

    manifest = {
        "schema": "codexlab.arc_public_recording_download_manifest.v1",
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "category": "historical_public_recording_download",
        "boundary": "Public ARC recording corpus for hypothesis mining only; not local CONTRA run evidence.",
        "config_id": args.config_id,
        "scorecard_url": scorecard_url,
        "scorecard_path": str(scorecard_path),
        "scorecard_sha256": sha256_file(scorecard_path),
        "recording_count": len(recording_rows),
        "total_recording_bytes": sum(row["bytes"] for row in recording_rows),
        "recordings": recording_rows,
    }
    manifest_path = out_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(str(manifest_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
