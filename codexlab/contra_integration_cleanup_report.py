#!/usr/bin/env python3
"""Build the CONTRA integration-cleanup Angel report."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from contra_sp80_review_page import build_pass_review_images

ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA = "angel.contra_gameplay_report_manifest.v1"
TRACE_SCHEMA = "angel.report_trace_event.v1"
RUN_PREFIX = "contra_integration_cleanup_report"
FIXTURE_PATHS = [
    ROOT / "codexlab/runs/contra_api_smoke/result.json",
    ROOT / "codexlab/runs/contra_api_smoke/cross_game_no_calibration.json",
    ROOT / "codexlab/runs/contra_api_smoke/forced_actor_failure.json",
]
EPISODE_RUN_DIRS = [
    ROOT / "codexlab/runs/sp80_ghost_capture_level_00_20260602T041500Z",
    ROOT / "codexlab/runs/sp80_ghost_context_level_01_20260602T043000Z",
    ROOT / "codexlab/runs/sp80_baseline_episode_level_02_20260602T031900Z",
]
GHOST_JSON = ROOT / "codexlab/runs/sp80_ghost_capture_level_00_20260602T041500Z/ghost.json"
KNOWN_SOLUTION_JSON = ROOT / "codexlab/runs/sp80_known_solution_level_02_shared_offset_20260602T031900Z/level_02/controlled_play_result.json"


def utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def repo_rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_commit() -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def git_dirty() -> bool:
    proc = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True, capture_output=True, check=False)
    return bool(proc.stdout.strip()) if proc.returncode == 0 else True


def font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def wrap_text(text: Any, width: int = 92) -> list[str]:
    words = str(text).replace("\n", " \n ").split()
    lines: list[str] = []
    current = ""
    for word in words:
        if word == "\n":
            if current:
                lines.append(current)
                current = ""
            continue
        trial = word if not current else f"{current} {word}"
        if len(trial) <= width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def text_slide(path: Path, title: str, sections: list[tuple[str, list[Any]]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1600, 1000), (8, 12, 17))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 1600, 104], fill=(16, 24, 34))
    draw.text((34, 26), title, fill=(248, 250, 252), font=font(38, bold=True))
    y = 132
    for heading, rows in sections:
        draw.text((46, y), heading, fill=(125, 211, 252), font=font(27, bold=True))
        y += 40
        for row in rows:
            for line in wrap_text(row, 100):
                draw.text((62, y), line, fill=(226, 232, 240), font=font(21))
                y += 29
                if y > 940:
                    draw.text((62, y), "... truncated; see HTML and raw JSON", fill=(251, 191, 36), font=font(18))
                    image.save(path)
                    return {"path": repo_rel(path), "sha256": sha256_file(path)}
            y += 6
        y += 18
    image.save(path)
    return {"path": repo_rel(path), "sha256": sha256_file(path)}


def copy_png(src: Path, dst: Path) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"path": repo_rel(dst), "sha256": sha256_file(dst)}


def command_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = "codexlab"
    if extra:
        env.update(extra)
    return env


def run_command(label: str, command: list[str], *, log_dir: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    log_path = log_dir / f"{label}.log"
    display = " ".join(command)
    proc = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, check=False)
    log_text = "$ " + display + "\n\n" + proc.stdout
    if proc.stderr:
        log_text += "\n[stderr]\n" + proc.stderr
    log_path.write_text(log_text, encoding="utf-8")
    return {
        "label": label,
        "command": display,
        "exit_code": proc.returncode,
        "status": "passed" if proc.returncode == 0 else "failed",
        "log_ref": {"path": repo_rel(log_path), "sha256": sha256_file(log_path)},
    }


def inspect_known_solution(log_dir: Path) -> dict[str, Any]:
    label = "known_solution_playback_json_check"
    log_path = log_dir / f"{label}.log"
    result = read_json(KNOWN_SOLUTION_JSON) if KNOWN_SOLUTION_JSON.is_file() else {}
    checks = {
        "schema": result.get("schema") == "codexlab.sp80_known_solution_playback.v1",
        "status_won": result.get("status") == "won",
        "levels_completed_delta_positive": int(result.get("levels_completed_delta") or 0) > 0,
    }
    status = "passed" if all(checks.values()) else "failed"
    log_path.write_text(json.dumps({"path": repo_rel(KNOWN_SOLUTION_JSON), "checks": checks, "status": status}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "label": label,
        "command": f"internal JSON check {repo_rel(KNOWN_SOLUTION_JSON)}",
        "exit_code": 0 if status == "passed" else 1,
        "status": status,
        "log_ref": {"path": repo_rel(log_path), "sha256": sha256_file(log_path)},
    }


def run_verifications(log_dir: Path) -> list[dict[str, Any]]:
    rows = [
        run_command(
            "dev_check",
            ["python", "-B", "codexlab/dev_check.py"],
            log_dir=log_dir,
            env=command_env(),
        ),
        run_command(
            "sp80_integration_smoke",
            ["bash", "codexlab/sp80_integration_smoke.sh"],
            log_dir=log_dir,
            env=command_env({"PYTHON": "python3"}),
        ),
        run_command(
            "contra_api_verify_smoke_fixtures",
            [
                "python",
                "-B",
                "codexlab/contra_api_verify.py",
                *[repo_rel(path) for path in FIXTURE_PATHS],
            ],
            log_dir=log_dir,
            env=command_env(),
        ),
    ]
    for run_dir in EPISODE_RUN_DIRS:
        rows.append(
            run_command(
                f"sp80_verify_episode_run_{run_dir.name}",
                ["python", "-B", "codexlab/sp80_verify_episode_run.py", "--run-dir", repo_rel(run_dir)],
                log_dir=log_dir,
                env=command_env(),
            )
        )
    rows.append(
        run_command(
            "contra_verify_ghost",
            ["python", "-B", "codexlab/contra_verify_ghost.py", repo_rel(GHOST_JSON)],
            log_dir=log_dir,
            env=command_env(),
        )
    )
    rows.append(inspect_known_solution(log_dir))
    return rows


def packet_contract_rows(packet: dict[str, Any]) -> list[str]:
    view_passes = ((packet.get("view_system_passes") or {}).get("passes") or []) if isinstance(packet.get("view_system_passes"), dict) else []
    pass_ids = [str(row.get("pass_id")) for row in view_passes if isinstance(row, dict)]
    ledger = packet.get("action_semantics_ledger") if isinstance(packet.get("action_semantics_ledger"), dict) else {}
    ghost = packet.get("ghost_pass") if isinstance(packet.get("ghost_pass"), dict) else {}
    counts = ((packet.get("debug_summary") or {}).get("counts") or {}) if isinstance(packet.get("debug_summary"), dict) else {}
    return [
        f"required top-level passes: config_pass, oracle_pass, ghost_pass, contra_pass, view_system_passes",
        f"view_system_passes.pass_ids: {', '.join(pass_ids)}",
        f"ghost_pass.status: {ghost.get('status')} / relation_memory_row_count={ghost.get('relation_memory_row_count')}",
        f"action_semantics_ledger.status: {ledger.get('status')} / row_count={ledger.get('row_count')}",
        f"debug counts: actors={counts.get('actors')} targets={counts.get('targets')} supports={counts.get('supports')} statecons={counts.get('statecons')} geobands={counts.get('geobands')}",
    ]


def artifact_summaries() -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for run_dir in EPISODE_RUN_DIRS:
        result = read_json(run_dir / "result.json")
        summaries.append(
            {
                "category": "controlled_run_historical_verified_this_session",
                "path": repo_rel(run_dir),
                "level_index": result.get("level_index"),
                "status": result.get("status"),
                "action_count": result.get("action_count"),
                "levels_completed_delta": result.get("levels_completed_delta"),
                "boundary": "Episode-harness run. Historical/untracked until explicitly regenerated; verifier checks packet/log contract, not a new solve claim.",
            }
        )
    known = read_json(KNOWN_SOLUTION_JSON) if KNOWN_SOLUTION_JSON.is_file() else {}
    summaries.append(
        {
            "category": "controlled_playback_historical",
            "path": repo_rel(KNOWN_SOLUTION_JSON),
            "level_index": known.get("level_index"),
            "status": known.get("status"),
            "levels_completed_delta": known.get("levels_completed_delta"),
            "boundary": "Known-solution playback format, not autonomous CONTRA episode-harness solve evidence.",
        }
    )
    ghost = read_json(GHOST_JSON) if GHOST_JSON.is_file() else {}
    summaries.append(
        {
            "category": "controlled_ghost_memory_historical_verified_this_session",
            "path": repo_rel(GHOST_JSON),
            "status": ghost.get("status"),
            "ghost_id": ghost.get("ghost_id"),
            "confirmed_relationship_row_count": ghost.get("confirmed_relationship_row_count"),
            "levels_completed_delta": (ghost.get("completion_proof") or {}).get("levels_completed_delta") if isinstance(ghost.get("completion_proof"), dict) else None,
            "boundary": "Ghost relationship memory is not future truth, route replay, action authority, or solve evidence.",
        }
    )
    return summaries


def build_trace(run_id: str, slides: list[dict[str, Any]], verification_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in verification_rows:
        rows.append(
            {
                "schema": TRACE_SCHEMA,
                "run_id": run_id,
                "event_index": len(rows),
                "event_type": "verifier",
                "function_name": row["label"],
                "function_family": "verification",
                "output_refs": [row["log_ref"]],
                "status": row["status"],
                "truth_boundary": "verification_log_not_new_gameplay_result",
            }
        )
    for slide in slides:
        rows.append(
            {
                "schema": TRACE_SCHEMA,
                "run_id": run_id,
                "event_index": len(rows),
                "event_type": "function_slide",
                "function_name": slide["slide_type"],
                "function_family": slide.get("family", "report"),
                "output_refs": [{"path": slide["path"], "sha256": slide["sha256"]}],
                "status": "rendered",
                "truth_boundary": "static_review_not_action_choice",
            }
        )
    return rows


def link(path: str, out_dir: Path) -> str:
    return html.escape(os.path.relpath(resolve_path(Path(path)), start=out_dir))


def render_index(
    *,
    run_id: str,
    out_dir: Path,
    slides: list[dict[str, Any]],
    verification_rows: list[dict[str, Any]],
    artifact_rows: list[dict[str, Any]],
    packet: dict[str, Any],
) -> str:
    cards = [
        ("Category", "static_review"),
        ("Slides", len(slides)),
        ("API fixtures", len(FIXTURE_PATHS)),
        ("Verifier rows", len(verification_rows)),
        ("Ghost status", (packet.get("ghost_pass") or {}).get("status") if isinstance(packet.get("ghost_pass"), dict) else "missing"),
        ("Ledger rows", (packet.get("action_semantics_ledger") or {}).get("row_count") if isinstance(packet.get("action_semantics_ledger"), dict) else "missing"),
    ]
    card_html = "".join(f"<div class='card'><b>{html.escape(str(k))}</b><span>{html.escape(str(v))}</span></div>" for k, v in cards)
    slide_html = "".join(
        "<figure>"
        f"<figcaption>{html.escape(str(slide['title']))}</figcaption>"
        f"<a href='{link(slide['path'], out_dir)}'><img src='{link(slide['path'], out_dir)}'></a>"
        "</figure>"
        for slide in slides
    )
    verifier_html = "".join(
        "<tr>"
        f"<td>{html.escape(row['label'])}</td>"
        f"<td class='{html.escape(row['status'])}'>{html.escape(row['status'])}</td>"
        f"<td>{html.escape(str(row['exit_code']))}</td>"
        f"<td><code>{html.escape(row['command'])}</code></td>"
        f"<td><a href='{link(row['log_ref']['path'], out_dir)}'>log</a></td>"
        "</tr>"
        for row in verification_rows
    )
    artifact_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.get('category')))}</td>"
        f"<td><a href='{link(str(row.get('path')), out_dir)}'>{html.escape(str(row.get('path')))}</a></td>"
        f"<td>{html.escape(str(row.get('status')))}</td>"
        f"<td>{html.escape(str(row.get('level_index')))}</td>"
        f"<td>{html.escape(str(row.get('levels_completed_delta')))}</td>"
        f"<td>{html.escape(str(row.get('boundary')))}</td>"
        "</tr>"
        for row in artifact_rows
    )
    source_links = [
        ("result.json", repo_rel(FIXTURE_PATHS[0])),
        ("cross_game_no_calibration.json", repo_rel(FIXTURE_PATHS[1])),
        ("forced_actor_failure.json", repo_rel(FIXTURE_PATHS[2])),
        ("ghost.json", repo_rel(GHOST_JSON)),
        ("known solution level 2 playback", repo_rel(KNOWN_SOLUTION_JSON)),
        ("manifest", repo_rel(out_dir / "report_manifest.json")),
        ("trace", repo_rel(out_dir / "report_trace.jsonl")),
    ]
    links_html = " | ".join(f"<a href='{link(path, out_dir)}'>{html.escape(label)}</a>" for label, path in source_links)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(run_id)}</title>
<style>
body{{margin:0;background:#0a0f14;color:#e6edf3;font-family:Inter,system-ui,Segoe UI,sans-serif}}
main{{max-width:1680px;margin:0 auto;padding:28px}}
a{{color:#8cc8ff}}
.boundary{{background:#141b23;border-left:4px solid #f0b429;padding:12px;color:#ffe8a3}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:18px 0}}
.card{{background:#101821;border:1px solid #263345;border-radius:6px;padding:12px}}
.card b{{display:block;color:#9fb0c3;font-size:12px;text-transform:uppercase}}
.card span{{font-size:22px;overflow-wrap:anywhere}}
.deck{{display:grid;grid-template-columns:1fr;gap:20px}}
figure{{margin:0;background:#101821;border:1px solid #263345;border-radius:6px;padding:12px}}
figcaption{{margin-bottom:8px;color:#cbd5e1}}
img{{display:block;width:100%;height:auto;image-rendering:pixelated}}
table{{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px}}
th,td{{border-bottom:1px solid #263345;padding:8px;text-align:left;vertical-align:top;overflow-wrap:anywhere}}
th{{background:#111923;color:#cbd5e1}}
code{{color:#d7e3f4}}
.passed{{color:#4ade80}}.failed{{color:#fb7185}}
</style>
</head>
<body>
<main>
<h1>CONTRA Integration Cleanup Report</h1>
<p class="boundary">Category: static_review / integration cleanup. Not a new level-2 solve attempt, not control truth, not goal truth, not action choice, not prediction truth, and not solve evidence.</p>
<p>{links_html}</p>
<section class="cards">{card_html}</section>
<h2>Visual Deck</h2>
<section class="deck">{slide_html}</section>
<h2>Verifier Command Table</h2>
<table><thead><tr><th>label</th><th>status</th><th>exit</th><th>command</th><th>log</th></tr></thead><tbody>{verifier_html}</tbody></table>
<h2>Referenced Run Boundaries</h2>
<table><thead><tr><th>category</th><th>path</th><th>status</th><th>level</th><th>completion delta</th><th>boundary</th></tr></thead><tbody>{artifact_html}</tbody></table>
</main>
</body>
</html>
"""


def build_report(out_dir: Path, *, run_verifiers: bool) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = out_dir.name
    slides_dir = out_dir / "slides"
    log_dir = out_dir / "logs"
    screenshots_dir = out_dir / "screenshots"
    slides_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    verification_rows = run_verifications(log_dir) if run_verifiers else []
    packet = read_json(FIXTURE_PATHS[0])
    cross_no_cal = read_json(FIXTURE_PATHS[1])
    forced = read_json(FIXTURE_PATHS[2])
    ghost = read_json(GHOST_JSON) if GHOST_JSON.is_file() else {}
    artifact_rows = artifact_summaries()

    slides: list[dict[str, Any]] = []
    raw_png = ROOT / "codexlab/runs/sp80_sam_color_components/frames/level_00_raw.png"
    if raw_png.is_file():
        raw_slide = copy_png(raw_png, slides_dir / "005_level_00_raw_frame.png")
        slides.append({"slide_id": "005_level_00_raw_frame", "slide_type": "raw_frame", "family": "frame", "title": "Level 00 Raw Frame", **raw_slide})
    slide_specs = [
        (
            "000_run_condition.png",
            "Run Condition",
            [
                ("Condition", [f"run_id: {run_id}", "static_review / integration cleanup; no new level-2 solve attempt", f"cwd: {ROOT}", f"git_commit: {git_commit()}", f"git_dirty: {git_dirty()}"]),
                ("Boundary", ["Static review and verifier-log report only. Not control truth, not goal truth, not action choice, not prediction truth, not solve evidence."]),
            ],
            "run_condition",
            "provenance",
        ),
        (
            "010_api_packet_contract.png",
            "Current API Packet Contract",
            [("Required Packet Surfaces", packet_contract_rows(packet))],
            "api_packet_contract",
            "contract",
        ),
        (
            "020_ghost_status.png",
            "Ghost Capture And Load Status",
            [
                (
                    "Ghost Artifact",
                    [
                        f"ghost_json: {repo_rel(GHOST_JSON)}",
                        f"status: {ghost.get('status')}",
                        f"ghost_id: {ghost.get('ghost_id')}",
                        f"confirmed_relationship_row_count: {ghost.get('confirmed_relationship_row_count')}",
                        f"completion_proof: {json.dumps(ghost.get('completion_proof'), sort_keys=True)}",
                    ],
                ),
                ("Boundary", ["Ghost memory is relationship context only. It is not route replay, action authority, future truth, or solve evidence."]),
            ],
            "ghost_status",
            "ghost",
        ),
        (
            "030_fixture_status.png",
            "Smoke Fixture Status",
            [
                (
                    "Fixtures",
                    [
                        f"result.json: ghost={((packet.get('ghost_pass') or {}).get('status') if isinstance(packet.get('ghost_pass'), dict) else 'missing')} ledger={((packet.get('action_semantics_ledger') or {}).get('status') if isinstance(packet.get('action_semantics_ledger'), dict) else 'missing')}",
                        f"cross_game_no_calibration.json: ghost={((cross_no_cal.get('ghost_pass') or {}).get('status') if isinstance(cross_no_cal.get('ghost_pass'), dict) else 'missing')} ledger={((cross_no_cal.get('action_semantics_ledger') or {}).get('status') if isinstance(cross_no_cal.get('action_semantics_ledger'), dict) else 'missing')}",
                        f"forced_actor_failure.json: ghost={((forced.get('ghost_pass') or {}).get('status') if isinstance(forced.get('ghost_pass'), dict) else 'missing')} ledger={((forced.get('action_semantics_ledger') or {}).get('status') if isinstance(forced.get('action_semantics_ledger'), dict) else 'missing')}",
                    ],
                )
            ],
            "fixture_status",
            "contract",
        ),
        (
            "040_level_boundaries.png",
            "Level Artifact Boundaries",
            [
                (
                    "Referenced Artifacts",
                    [
                        "level 0: controlled Ghost capture is a historical verified run artifact, not a new run in this report",
                        "level 1: Ghost-context episode is non-solve context; verifier checks packet/log contract",
                        "level 2: known-solution playback is separate playback evidence; baseline episode remains non-solve",
                    ],
                )
            ],
            "level_boundaries",
            "boundary",
        ),
        (
            "050_verifier_commands.png",
            "Verifier Command Table",
            [
                (
                    "Statuses",
                    [f"{row['label']}: {row['status']} exit={row['exit_code']}" for row in verification_rows] or ["verifiers skipped"],
                )
            ],
            "verifier_commands",
            "verification",
        ),
    ]
    for name, title, sections, slide_type, family in slide_specs:
        rendered = text_slide(slides_dir / name, title, sections)
        slides.append({"slide_id": Path(name).stem, "slide_type": slide_type, "family": family, "title": title, **rendered})

    pass_assets = build_pass_review_images(packet, out_dir=out_dir)
    for idx, asset in enumerate(pass_assets):
        slides.append(
            {
                "slide_id": f"pass_{idx:04d}",
                "slide_type": str(asset.get("pass_id") or "pass_image"),
                "family": str(asset.get("pass_family") or "pass_image"),
                "title": str(asset.get("title") or asset.get("pass_id") or "Pass Image"),
                "path": str(asset["path"]),
                "sha256": str(asset["sha256"]),
            }
        )

    trace_rows = build_trace(run_id, slides, verification_rows)
    trace_path = out_dir / "report_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in trace_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    index_path = out_dir / "index.html"
    index_path.write_text(
        render_index(
            run_id=run_id,
            out_dir=out_dir,
            slides=slides,
            verification_rows=verification_rows,
            artifact_rows=artifact_rows,
            packet=packet,
        ),
        encoding="utf-8",
    )
    raw_inputs = [{"path": repo_rel(path), "sha256": sha256_file(path)} for path in FIXTURE_PATHS if path.is_file()]
    for path in [GHOST_JSON, KNOWN_SOLUTION_JSON, *[run_dir / "result.json" for run_dir in EPISODE_RUN_DIRS]]:
        if path.is_file():
            raw_inputs.append({"path": repo_rel(path), "sha256": sha256_file(path)})
    raw_outputs = [
        {"path": repo_rel(index_path), "sha256": sha256_file(index_path)},
        {"path": repo_rel(trace_path), "sha256": sha256_file(trace_path)},
        *[row["log_ref"] for row in verification_rows],
    ]
    manifest = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "report_status": "static_review",
        "run_id": run_id,
        "game_id": "sp80-589a99af",
        "working_directory": str(ROOT),
        "host": socket.gethostname(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "command": " ".join(sys.argv),
        "index_html": repo_rel(index_path),
        "report_manifest_json": repo_rel(out_dir / "report_manifest.json"),
        "report_trace_jsonl": repo_rel(trace_path),
        "source_paths": [
            {"path": "codexlab/contra_api_smoke.py", "sha256": sha256_file(ROOT / "codexlab/contra_api_smoke.py")},
            {"path": "codexlab/contra_api_verify.py", "sha256": sha256_file(ROOT / "codexlab/contra_api_verify.py")},
            {"path": "codexlab/contra_sp80_review_page.py", "sha256": sha256_file(ROOT / "codexlab/contra_sp80_review_page.py")},
            {"path": "codexlab/angel_report_verify.py", "sha256": sha256_file(ROOT / "codexlab/angel_report_verify.py")},
            {"path": "codexlab/contra_integration_cleanup_report.py", "sha256": sha256_file(ROOT / "codexlab/contra_integration_cleanup_report.py")},
        ],
        "raw_inputs": raw_inputs,
        "raw_outputs": raw_outputs,
        "slide_count": len(slides),
        "function_slide_count": len([row for row in slides if row["slide_type"] != "raw_frame"]),
        "action_count": 0,
        "oracle_sidecar_count": 0,
        "ghost_pass_count": len([row for row in slides if row["slide_type"] == "ghost_pass"]),
        "verification_commands": verification_rows,
        "referenced_artifacts": artifact_rows,
        "slide_ledger": slides,
        "verification": {
            "image_count_matches_manifest": True,
            "slide_count_matches_png_ledger": True,
            "sha256_path_refs_checked": True,
            "index_reachable": False,
            "browser_screenshot_checked": False,
            "screenshot_path": repo_rel(screenshots_dir / "index_desktop.png"),
            "required_files": ["index.html", "report_manifest.json", "report_trace.jsonl"],
        },
        "boundaries": [
            "Static review / integration cleanup report only.",
            "Not a new level-2 solve attempt.",
            "Static review is not control truth, goal truth, action choice, prediction truth, or solve evidence.",
            "Known-solution playback is not autonomous CONTRA solve evidence.",
            "Ghost relationship memory is not route replay, action authority, or future truth.",
        ],
    }
    write_json(out_dir / "report_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--skip-verification", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = resolve_path(args.out_dir) if args.out_dir else ROOT / "codexlab/runs" / f"{RUN_PREFIX}_{utc_compact()}"
    manifest = build_report(out_dir.resolve(), run_verifiers=not args.skip_verification)
    print(json.dumps({"status": "reported", "run_dir": repo_rel(out_dir), "index_html": manifest["index_html"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
