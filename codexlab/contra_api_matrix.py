#!/usr/bin/env python3
"""Run the unified CONTRA API across paint and cross-game raw-frame samples."""

from __future__ import annotations

import argparse
import html
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODEXLAB = ROOT / "codexlab"
if str(CODEXLAB) not in sys.path:
    sys.path.insert(0, str(CODEXLAB))

import contra_head  # noqa: E402

CONTRA_API_MATRIX_MANIFEST_SCHEMA = "codexlab.contra_api_matrix_manifest.v1"


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(resolve_repo_path(path).read_text(encoding="utf-8"))


def load_sp80_calibrations(path: Path) -> dict[int, dict[str, Any]]:
    payload = read_json(path)
    rows: dict[int, dict[str, Any]] = {}
    for row in payload.get("levels", []):
        if isinstance(row, dict) and isinstance(row.get("level_index"), int):
            rows[int(row["level_index"])] = row
    return rows


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:  # noqa: BLE001 - provenance helper should not kill the controlled API run.
        return "unknown"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_api_sample(
    *,
    sample_kind: str,
    raw_grid_path: Path | None,
    raw_png_path: Path | None,
    game_id: str,
    game_short: str,
    level_index: int,
    calibration: dict[str, Any] | None,
    out_packet: Path,
    force_failing_actor_builder: bool = False,
    invalid_raw_frame: bool = False,
) -> dict[str, Any]:
    actor_builder = None
    if force_failing_actor_builder:
        def actor_builder(**_kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("forced actor builder failure for CONTRA API matrix")

    if invalid_raw_frame:
        frame: Any = [[[1, 2, 3]]]
    else:
        if raw_grid_path is None:
            raise ValueError("raw_grid_path is required unless invalid_raw_frame is true")
        frame = np.load(resolve_repo_path(raw_grid_path), allow_pickle=False)

    packet = contra_head.run_contra_view(
        raw_frame=frame,
        game_id=game_id,
        level_index=int(level_index),
        calibration=calibration,
        mode="review",
        actor_builder=actor_builder,
    )
    write_json(out_packet, packet)
    counts = packet.get("debug_summary", {}).get("counts", {})
    handoffs = packet.get("layer_handoffs", {})
    layer_outputs = packet.get("layer_output_packets", {})
    return {
        "sample_kind": sample_kind,
        "game_id": game_id,
        "game_short": game_short,
        "level_index": int(level_index),
        "raw_grid": rel(resolve_repo_path(raw_grid_path)) if raw_grid_path is not None else None,
        "raw_png": rel(resolve_repo_path(raw_png_path)) if raw_png_path is not None else None,
        "calibration_supplied": isinstance(calibration, dict),
        "contra_api_packet": rel(out_packet),
        "response_status": packet.get("status"),
        "admission_status": packet.get("admission_status"),
        "handoff_status": handoffs.get("status"),
        "hard_stop_required": handoffs.get("hard_stop_required"),
        "recovered_layer_count": handoffs.get("recovered_layer_count"),
        "fail_closed_layer_count": handoffs.get("fail_closed_layer_count"),
        "skipped_output_count": layer_outputs.get("skipped_output_count"),
        "computed_output_count": layer_outputs.get("computed_output_count"),
        "skipped_output_keys": layer_outputs.get("skipped_output_keys", []),
        "contra_ran": packet.get("debug_summary", {}).get("contra_ran"),
        "statecon_active": packet.get("debug_summary", {}).get("statecon_active"),
        "geoband_active": packet.get("debug_summary", {}).get("geoband_active"),
        "raw_colors": counts.get("raw_colors"),
        "raw_components": counts.get("raw_components"),
        "actors": counts.get("actors"),
        "targets": counts.get("targets"),
        "supports": counts.get("supports"),
        "symmetry_groups": counts.get("symmetry_groups"),
        "statecons": counts.get("statecons"),
        "geobands": counts.get("geobands"),
        "fail_closed_reasons": packet.get("fail_closed_reasons", []),
    }


def paint_samples(frames_dir: Path, calibrations: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in sorted(resolve_repo_path(frames_dir).glob("level_*_raw.npy")):
        stem = raw_path.stem
        level_index = int(stem.split("_")[1])
        png_path = raw_path.with_suffix(".png")
        rows.append(
            {
                "sample_kind": "paint_calibrated",
                "raw_grid": Path(rel(raw_path)),
                "raw_png": Path(rel(png_path)) if png_path.is_file() else None,
                "game_id": "sp80-589a99af",
                "game_short": "sp80",
                "level_index": level_index,
                "calibration": calibrations.get(level_index),
            }
        )
    return rows


def cross_game_samples(manifest_path: Path, *, limit: int) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    rows: list[dict[str, Any]] = []
    for row in manifest.get("rows", []):
        if not isinstance(row, dict):
            continue
        raw_grid = row.get("raw_frame", {}).get("raw_grid", {}) if isinstance(row.get("raw_frame"), dict) else {}
        raw_png = row.get("raw_frame", {}).get("png", {}) if isinstance(row.get("raw_frame"), dict) else {}
        if not isinstance(raw_grid, dict) or not raw_grid.get("path"):
            continue
        rows.append(
            {
                "sample_kind": "cross_game_no_calibration",
                "raw_grid": Path(str(raw_grid["path"])),
                "raw_png": Path(str(raw_png["path"])) if isinstance(raw_png, dict) and raw_png.get("path") else None,
                "game_id": str(row.get("game_id") or row.get("game_short") or "unknown-game"),
                "game_short": str(row.get("game_short") or str(row.get("game_id") or "unknown").split("-")[0]),
                "level_index": int(row.get("level_index") or 0),
                "calibration": None,
            }
        )
    return rows[:limit] if limit > 0 else rows


def cross_game_smoke_sample(calibration_path: Path | None) -> list[dict[str, Any]]:
    if calibration_path is None:
        return []
    calibration = read_json(calibration_path)
    status = str(calibration.get("calibration_status") or "")
    sample_kind = "cross_game_smoke_calibrated" if status == "user_supplied_answer" else "cross_game_smoke_pending_calibration"
    return [
        {
            "sample_kind": sample_kind,
            "raw_grid": Path(str(calibration["raw_grid"])),
            "raw_png": Path(str(calibration["raw_png"])) if calibration.get("raw_png") else None,
            "game_id": str(calibration.get("game_id") or calibration.get("game_short") or "cross-game-smoke"),
            "game_short": str(calibration.get("game_short") or str(calibration.get("game_id") or "cross").split("-")[0]),
            "level_index": int(calibration.get("level_index") or 0),
            "calibration": calibration if sample_kind == "cross_game_smoke_calibrated" else None,
        }
    ]


def render_summary_md(manifest: dict[str, Any]) -> str:
    lines = [
        "# CONTRA API v0.01 Matrix",
        "",
        "This report records API execution only. It does not claim gameplay solve, control truth, goal truth, or action-choice truth.",
        "",
        "| Kind | Game | Level | Ran | StateCon | GeoBand | Actors | Targets | Supports | Skipped Outputs | Recovered | Packet |",
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in manifest.get("rows", []):
        if not isinstance(row, dict):
            continue
        lines.append(
            "| {kind} | {game} | {level} | {ran} | {statecon} | {geoband} | {actors} | {targets} | {supports} | {skipped} | {recovered} | {packet} |".format(
                kind=row.get("sample_kind"),
                game=row.get("game_short"),
                level=row.get("level_index"),
                ran=row.get("contra_ran"),
                statecon=row.get("statecon_active"),
                geoband=row.get("geoband_active"),
                actors=row.get("actors"),
                targets=row.get("targets"),
                supports=row.get("supports"),
                skipped=row.get("skipped_output_count"),
                recovered=row.get("recovered_layer_count"),
                packet=row.get("contra_api_packet"),
            )
        )
    lines.append("")
    lines.append("v0.01 criteria frozen in this run:")
    for criterion in manifest.get("v001_criteria", []):
        lines.append(f"- {criterion}")
    lines.append("")
    return "\n".join(lines)


def href_from(base_dir: Path, path_value: Any) -> str:
    if not path_value:
        return ""
    target = resolve_repo_path(Path(str(path_value)))
    return os.path.relpath(target, start=base_dir)


def render_index_html(manifest: dict[str, Any], *, out_dir: Path) -> str:
    rows_html: list[str] = []
    for row in manifest.get("rows", []):
        if not isinstance(row, dict):
            continue
        raw_png = row.get("raw_png")
        raw_link = f'<a href="{html.escape(href_from(out_dir, raw_png))}">raw</a>' if raw_png else ""
        packet = html.escape(href_from(out_dir, row.get("contra_api_packet")))
        reasons = ", ".join(str(value) for value in row.get("fail_closed_reasons", [])[:4])
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('sample_kind')))}</td>"
            f"<td>{html.escape(str(row.get('game_short')))}</td>"
            f"<td>{html.escape(str(row.get('level_index')))}</td>"
            f"<td>{html.escape(str(row.get('contra_ran')))}</td>"
            f"<td>{html.escape(str(row.get('statecon_active')))}</td>"
            f"<td>{html.escape(str(row.get('geoband_active')))}</td>"
            f"<td>{html.escape(str(row.get('actors')))}</td>"
            f"<td>{html.escape(str(row.get('targets')))}</td>"
            f"<td>{html.escape(str(row.get('supports')))}</td>"
            f"<td>{html.escape(str(row.get('skipped_output_count')))}</td>"
            f"<td>{html.escape(str(row.get('recovered_layer_count')))}</td>"
            f"<td>{raw_link}</td>"
            f"<td><a href=\"{packet}\">packet</a></td>"
            f"<td>{html.escape(reasons)}</td>"
            "</tr>"
        )
    return """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CONTRA API v0.01 Matrix</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 24px; color: #151515; }
    table { border-collapse: collapse; width: 100%; font-size: 13px; }
    th, td { border: 1px solid #d5d5d5; padding: 6px 8px; vertical-align: top; }
    th { background: #f3f3f3; text-align: left; }
    code { background: #f4f4f4; padding: 1px 4px; }
  </style>
</head>
<body>
  <h1>CONTRA API v0.01 Matrix</h1>
  <p>Review aid only: static hypotheses remain fail-closed and not admitted as control truth, goal truth, action choice, or solve truth.</p>
  <table>
    <thead><tr><th>Kind</th><th>Game</th><th>Level</th><th>Ran</th><th>StateCon</th><th>GeoBand</th><th>Actors</th><th>Targets</th><th>Supports</th><th>Skipped Outputs</th><th>Recovered</th><th>Raw</th><th>Packet</th><th>Reasons</th></tr></thead>
    <tbody>
""" + "\n".join(rows_html) + """
    </tbody>
  </table>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paint-frames-dir", type=Path, default=Path("codexlab/runs/sp80_sam_color_components/frames"))
    parser.add_argument("--sp80-calibration", type=Path, default=Path("codexlab/sp80_surface_calibration.json"))
    parser.add_argument("--cross-game-manifest", type=Path, default=Path("codexlab/runs/cross_game_isolation_spotcheck/manifest.json"))
    parser.add_argument("--cross-game-limit", type=int, default=0, help="0 means all rows from the manifest")
    parser.add_argument("--cross-game-smoke-calibration", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("codexlab/runs/contra_api_v001_matrix"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_dir = resolve_repo_path(args.out_dir)
    packets_dir = out_dir / "packets"
    calibrations = load_sp80_calibrations(args.sp80_calibration)
    samples = [
        *paint_samples(args.paint_frames_dir, calibrations),
        *cross_game_samples(args.cross_game_manifest, limit=int(args.cross_game_limit)),
        *cross_game_smoke_sample(args.cross_game_smoke_calibration),
    ]

    rows: list[dict[str, Any]] = []
    for sample in samples:
        packet_name = f"{sample['sample_kind']}_{sample['game_short']}_level_{int(sample['level_index']):02d}.json"
        rows.append(
            run_api_sample(
                sample_kind=str(sample["sample_kind"]),
                raw_grid_path=sample["raw_grid"],
                raw_png_path=sample["raw_png"],
                game_id=str(sample["game_id"]),
                game_short=str(sample["game_short"]),
                level_index=int(sample["level_index"]),
                calibration=sample["calibration"],
                out_packet=packets_dir / packet_name,
            )
        )

    if samples:
        first = samples[0]
        rows.append(
            run_api_sample(
                sample_kind="forced_actor_failure",
                raw_grid_path=first["raw_grid"],
                raw_png_path=first["raw_png"],
                game_id=str(first["game_id"]),
                game_short=str(first["game_short"]),
                level_index=int(first["level_index"]),
                calibration=first["calibration"],
                out_packet=packets_dir / "probe_forced_actor_failure.json",
                force_failing_actor_builder=True,
            )
        )
    rows.append(
        run_api_sample(
            sample_kind="invalid_raw_frame",
            raw_grid_path=None,
            raw_png_path=None,
            game_id="invalid-frame-probe",
            game_short="invalid",
            level_index=0,
            calibration=None,
            out_packet=packets_dir / "probe_invalid_raw_frame.json",
            invalid_raw_frame=True,
        )
    )

    manifest = {
        "schema": CONTRA_API_MATRIX_MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "command": " ".join(sys.argv),
        "working_directory": str(ROOT),
        "git_commit": git_commit(),
        "condition": "same contra_head.run_contra_view API over calibrated SP80 paint frames and no-calibration cross-game raw frames, plus graceful fail-closed probes",
        "control_claim": "api_execution_and_fail_closed_layer_handoff_only",
        "not_gameplay_solve_evidence": True,
        "not_control_truth": True,
        "not_goal_truth": True,
        "not_action_choice_truth": True,
        "paint_frame_count": sum(1 for row in rows if row.get("sample_kind") == "paint_calibrated"),
        "cross_game_frame_count": sum(1 for row in rows if row.get("sample_kind") == "cross_game_no_calibration"),
        "cross_game_smoke_calibrated_count": sum(1 for row in rows if row.get("sample_kind") == "cross_game_smoke_calibrated"),
        "cross_game_smoke_pending_count": sum(1 for row in rows if row.get("sample_kind") == "cross_game_smoke_pending_calibration"),
        "probe_count": sum(1 for row in rows if str(row.get("sample_kind")).startswith(("forced_", "invalid_"))),
        "v001_criteria": [
            "one call fires the full CONTRA network",
            "normal bad input returns a fail-closed handoff packet instead of an uncaught hard stop",
            "static hypotheses never admit control truth, goal truth, action-choice truth, or solve truth",
            "inactive layers explain missing inputs or empty fail-closed outputs",
            "outputs are inspectable as JSON plus markdown/HTML debug summaries",
        ],
        "rows": rows,
    }
    write_json(out_dir / "manifest.json", manifest)
    (out_dir / "summary.md").write_text(render_summary_md(manifest), encoding="utf-8")
    (out_dir / "index.html").write_text(render_index_html(manifest, out_dir=out_dir), encoding="utf-8")
    print(
        f"wrote {rel(out_dir / 'manifest.json')} "
        f"paint={manifest['paint_frame_count']} "
        f"cross_game={manifest['cross_game_frame_count']} "
        f"probes={manifest['probe_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
