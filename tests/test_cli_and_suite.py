"""CLI reports, suite discovery, and the committed-file secret scan."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
)
ARC_KEY = re.compile(r"arc_api_key\s*=\s*[\"']([^\"']+)[\"']")
SKIP_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pt",
    ".pth",
    ".npz",
    ".npy",
    ".mp4",
    ".onnx",
}


def test_suite_lists_the_games_on_disk():
    from synthesis_lab.orchestration.rdl.brain.suite_games import (
        DEFAULT_SUITE_ENV_DIR,
        discover_game_ids,
        round_robin,
        suite_24,
    )

    found = []
    for prefix in DEFAULT_SUITE_ENV_DIR.iterdir():
        if not prefix.is_dir():
            continue
        for ident in prefix.iterdir():
            if ident.is_dir() and any(ident.glob("*.py")):
                found.append(f"{prefix.name}-{ident.name}")
    assert suite_24() == discover_game_ids() == sorted(found)
    assert "ft09-0d8bbf25" in found
    assert "ls20-9607627b" in found
    assert round_robin({"b": [1, 2], "a": [3]}) == [3, 1, 2]


def test_gate_cli_prints_a_decision_and_valid_json(tmp_path, capsys):
    from synthesis_lab.orchestration.rdl.brain.rsi_outer_gate import main

    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({"results": [], "suite": {"n_games": 2}}))
    code = main([str(candidate)])
    captured = capsys.readouterr()
    assert code == 0
    assert "REJECT" in captured.err
    assert "below_private_floor" in captured.err
    payload = json.loads(captured.out)
    assert payload["keep"] is False
    assert payload["candidate_public_loss"] is None


def test_flywheel_stop_file_exits_and_removes_the_pid(tmp_path, capsys):
    from synthesis_lab.orchestration.rdl.brain.train_flywheel import main

    (tmp_path / "STOP").write_text("stop\n")
    code = main(
        ["--root", str(tmp_path), "--max-cycles", "3", "--sleep-s", "0", "--device", "cpu"]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert not (tmp_path / "flywheel.pid").exists()
    heartbeat = json.loads((tmp_path / "HEARTBEAT.json").read_text())
    assert heartbeat["status"] == "stopped"
    assert "flywheel  device=cpu" in captured.err
    assert "flywheel stopped" in captured.err
    start = json.loads(captured.out.splitlines()[0])
    assert start["event"] == "flywheel_start"
    assert start["device"] == "cpu"


def test_no_private_keys_or_live_tokens_are_tracked():
    listed = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    paths = [item for item in listed.decode().split("\0") if item]
    hits: list[str] = []
    for rel in paths:
        name = Path(rel).name
        if name == ".env" or rel.endswith((".pem", ".key")):
            hits.append(rel)
            continue
        path = ROOT / rel
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        if path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                hits.append(f"{rel}: {pattern.pattern}")
        for match in ARC_KEY.finditer(text):
            if match.group(1) != "anonymous":
                hits.append(f"{rel}: arc_api_key={match.group(1)!r}")
    assert hits == []
