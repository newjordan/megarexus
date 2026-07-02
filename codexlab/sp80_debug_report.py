#!/usr/bin/env python3
"""Build one operator-facing debug report for the SP80 control MVP."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from players.sp80.registry import CONTROL_PRIMITIVES

ROOT = Path(__file__).resolve().parents[1]

ALLOWED_PRIMITIVES = set(CONTROL_PRIMITIVES)

MOVE_CASES = ["move_up", "move_down", "move_left", "move_right", "no_selected_motion"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def relpath(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    resolved.relative_to(ROOT)
    return resolved


def run_verifier(run_dir: Path) -> dict[str, Any]:
    cmd = [sys.executable, "codexlab/sp80_verify_run.py", relpath(run_dir)]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def check_recorded_file(meta: Any, hash_key: str, label: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(meta, dict):
        return [f"{label}: metadata is not an object"]
    path_value = meta.get("path")
    recorded_hash = meta.get(hash_key)
    if not path_value or not recorded_hash:
        return [f"{label}: missing path/{hash_key}"]
    try:
        path = resolve_repo_path(path_value)
    except Exception as exc:
        return [f"{label}: bad path {path_value!r}: {exc!r}"]
    if not path.is_file():
        return [f"{label}: missing file {path_value}"]
    actual = sha256_file(path)
    if actual != recorded_hash:
        errors.append(f"{label}: hash mismatch {path_value}")
    return errors


def count_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_level = Counter(str(row.get("level_index")) for row in rows)
    by_calibration_slice = Counter(
        str(row.get("calibration_slice_index", row.get("level_index"))) for row in rows
    )
    by_primitive = Counter((row.get("target") or {}).get("primitive") for row in rows)
    by_case = Counter(row.get("case_id") for row in rows)
    return {
        "rows": len(rows),
        "by_calibration_slice": dict(sorted(by_calibration_slice.items())),
        "by_level": dict(sorted(by_level.items())),
        "by_primitive": dict(sorted(by_primitive.items())),
        "by_case": dict(sorted(by_case.items())),
    }


def validate_dataset_row(row: dict[str, Any], source: str, line_no: int) -> list[str]:
    errors: list[str] = []
    label = f"{source}:{line_no}"
    target = row.get("target")
    if not isinstance(target, dict) or set(target) != {"primitive"}:
        errors.append(f"{label}: target is not primitive-only")
    primitive = target.get("primitive") if isinstance(target, dict) else None
    if primitive not in ALLOWED_PRIMITIVES:
        errors.append(f"{label}: invalid target primitive {primitive!r}")

    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        errors.append(f"{label}: messages is not length 3")
    else:
        assistant = messages[-1]
        try:
            assistant_payload = json.loads(assistant.get("content", ""))
        except Exception:
            assistant_payload = None
        if assistant_payload != {"primitive": primitive}:
            errors.append(f"{label}: assistant message is not primitive-only JSON")

    evidence = row.get("evidence") or {}
    for flag in ["route_or_replay_used", "human_trace_used", "coordinate_script_used"]:
        if evidence.get(flag) is not False:
            errors.append(f"{label}: evidence {flag} is not false")
    if evidence.get("controlled_runtime") is not True:
        errors.append(f"{label}: controlled_runtime is not true")

    for evidence_key in ["source_packet", "source_result", "source_brain_calls", "source_control_drift_report"]:
        path_value = evidence.get(evidence_key)
        if not path_value:
            errors.append(f"{label}: missing {evidence_key}")
            continue
        try:
            path = resolve_repo_path(path_value)
        except Exception as exc:
            errors.append(f"{label}: bad {evidence_key} {path_value!r}: {exc!r}")
            continue
        if not path.is_file():
            errors.append(f"{label}: missing {evidence_key} file {path_value}")

    for frame_name in ["before_frame", "after_frame"]:
        frame = evidence.get(frame_name) or {}
        errors.extend(check_recorded_file(frame.get("png"), "png_sha256", f"{label}: {frame_name}.png"))
        errors.extend(check_recorded_file(frame.get("raw_grid"), "npy_sha256", f"{label}: {frame_name}.raw_grid"))
    return errors


class DebugReport:
    def __init__(
        self,
        *,
        sweep_dir: Path,
        dataset_dir: Path,
        out_dir: Path,
        qwen_sweep_dir: Path | None,
        lora_result_dir: Path | None,
        lora_sweep_dir: Path | None,
    ) -> None:
        self.sweep_dir = sweep_dir.resolve()
        self.dataset_dir = dataset_dir.resolve()
        self.out_dir = out_dir.resolve()
        self.qwen_sweep_dir = None if qwen_sweep_dir is None else qwen_sweep_dir.resolve()
        self.lora_result_dir = None if lora_result_dir is None else lora_result_dir.resolve()
        self.lora_sweep_dir = None if lora_sweep_dir is None else lora_sweep_dir.resolve()
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def build(self) -> dict[str, Any]:
        summary = self.load_required_json(self.sweep_dir / "summary.json", "sweep summary")
        drift = self.load_required_json(self.sweep_dir / "control_drift_report.json", "control drift report")
        manifest = self.load_required_json(self.dataset_dir / "manifest.json", "dataset manifest")

        packets_by_level: dict[int, list[dict[str, Any]]] = {}
        verifier_by_level: dict[str, dict[str, Any]] = {}
        if summary:
            for level in summary.get("passed_levels", []):
                level_index = int(level)
                level_dir = self.sweep_dir / f"level_{level_index:02d}"
                verifier_by_level[str(level_index)] = run_verifier(level_dir)
                if verifier_by_level[str(level_index)]["returncode"] != 0:
                    self.errors.append(f"verifier failed for calibration slice {level_index}")
                packets_path = level_dir / "packets.jsonl"
                if packets_path.is_file():
                    packets_by_level[level_index] = load_jsonl(packets_path)
                else:
                    self.errors.append(f"missing packets for calibration slice {level_index}: {packets_path}")

        train_rows = self.load_required_jsonl(self.dataset_dir / "train.jsonl", "train rows")
        heldout_rows = self.load_required_jsonl(self.dataset_dir / "heldout.jsonl", "heldout rows")

        dataset_errors: list[str] = []
        for path, rows in [
            (self.dataset_dir / "train.jsonl", train_rows),
            (self.dataset_dir / "heldout.jsonl", heldout_rows),
        ]:
            for line_no, row in enumerate(rows, start=1):
                dataset_errors.extend(validate_dataset_row(row, relpath(path), line_no))
        self.errors.extend(dataset_errors)

        manifest_checks = self.check_manifest(manifest, train_rows, heldout_rows)
        movement_table = self.build_movement_table(packets_by_level)
        flow_table = self.build_flow_table(packets_by_level)
        blocked_table = self.build_blocked_table(packets_by_level)
        primitive_coverage = self.build_primitive_coverage(train_rows, heldout_rows)
        qwen_backend = self.load_qwen_backend_status()
        lora_dry_run = self.load_lora_dry_run_status()
        drift_pings = [] if not drift else drift.get("pings", [])

        if drift and drift.get("status") == "failed":
            self.errors.append("control drift report status is failed")
        elif drift and drift.get("status") == "attention":
            self.warnings.append("control drift report has attention pings")

        status = "failed" if self.errors else "attention" if self.warnings else "clean"
        return {
            "schema": "codexlab.sp80_debug_report.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": status,
            "source": {
                "sweep_dir": relpath(self.sweep_dir),
                "dataset_dir": relpath(self.dataset_dir),
                "summary_json": relpath(self.sweep_dir / "summary.json"),
                "control_drift_report_json": relpath(self.sweep_dir / "control_drift_report.json"),
                "dataset_manifest_json": relpath(self.dataset_dir / "manifest.json"),
            },
            "calibration": {
                "reporting_term": "calibration_slice",
                "reporting_note": "level_index is retained only as SP80 environment provenance; this report is not a gameplay-completion report.",
                "detected_calibration_slice_count": None
                if not summary
                else summary.get("detected_calibration_slice_count", summary.get("detected_level_count")),
                "passed_calibration_slices": []
                if not summary
                else summary.get("passed_calibration_slices", summary.get("passed_levels", [])),
                "failed_calibration_slices": []
                if not summary
                else summary.get("failed_calibration_slices", summary.get("failed_levels", [])),
                "verifier_by_calibration_slice": verifier_by_level,
            },
            "sweep": {
                "status": None if not summary else summary.get("status"),
                "detected_level_count": None if not summary else summary.get("detected_level_count"),
                "passed_levels": [] if not summary else summary.get("passed_levels", []),
                "failed_levels": [] if not summary else summary.get("failed_levels", []),
                "verifier_by_level": verifier_by_level,
            },
            "dataset": {
                "manifest_status": None if not manifest else manifest.get("status"),
                "manifest_checks": manifest_checks,
                "train": count_rows(train_rows),
                "heldout": count_rows(heldout_rows),
                "primitive_coverage": primitive_coverage,
                "target_pollution_errors": dataset_errors,
            },
            "qwen_backend": qwen_backend,
            "lora_dry_run": lora_dry_run,
            "tracking": {
                "drift_status": None if not drift else drift.get("status"),
                "drift_ping_ids": [ping.get("id") for ping in drift_pings],
                "drift_pings": drift_pings,
                "movement_display_delta_by_calibration_slice": movement_table,
                "movement_display_delta_by_level": movement_table,
                "flow_action_frame_count_by_calibration_slice": flow_table,
                "flow_action_frame_count_by_level": flow_table,
                "blocked_setup_action_count_by_calibration_slice": blocked_table,
                "blocked_setup_action_count_by_level": blocked_table,
            },
            "errors": self.errors,
            "warnings": self.warnings,
            "next_required_change": self.next_required_change(status, primitive_coverage, qwen_backend, lora_dry_run),
        }

    def load_required_json(self, path: Path, label: str) -> dict[str, Any] | None:
        if not path.is_file():
            self.errors.append(f"missing {label}: {path}")
            return None
        try:
            return load_json(path)
        except Exception as exc:
            self.errors.append(f"cannot read {label}: {exc!r}")
            return None

    def load_required_jsonl(self, path: Path, label: str) -> list[dict[str, Any]]:
        if not path.is_file():
            self.errors.append(f"missing {label}: {path}")
            return []
        try:
            return load_jsonl(path)
        except Exception as exc:
            self.errors.append(f"cannot read {label}: {exc!r}")
            return []

    def check_manifest(
        self,
        manifest: dict[str, Any] | None,
        train_rows: list[dict[str, Any]],
        heldout_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if manifest is None:
            return {"status": "missing"}
        checks: dict[str, Any] = {"status": "checked", "errors": []}
        expected_counts = {
            "row_count": len(train_rows) + len(heldout_rows),
            "train_count": len(train_rows),
            "heldout_count": len(heldout_rows),
        }
        for key, expected in expected_counts.items():
            if manifest.get(key) != expected:
                msg = f"manifest {key}={manifest.get(key)!r}, observed {expected!r}"
                checks["errors"].append(msg)
                self.errors.append(msg)
        for key, file_key, hash_key in [
            ("train", "train_jsonl", "train_jsonl_sha256"),
            ("heldout", "heldout_jsonl", "heldout_jsonl_sha256"),
        ]:
            file_value = manifest.get(file_key)
            recorded_hash = manifest.get(hash_key)
            if not file_value or not recorded_hash:
                msg = f"manifest missing {file_key}/{hash_key}"
                checks["errors"].append(msg)
                self.errors.append(msg)
                continue
            path = resolve_repo_path(file_value)
            if not path.is_file():
                msg = f"manifest {key} file missing: {file_value}"
                checks["errors"].append(msg)
                self.errors.append(msg)
                continue
            actual_hash = sha256_file(path)
            if actual_hash != recorded_hash:
                msg = f"manifest {key} hash mismatch"
                checks["errors"].append(msg)
                self.errors.append(msg)
        return checks

    def build_movement_table(self, packets_by_level: dict[int, list[dict[str, Any]]]) -> dict[str, Any]:
        table: dict[str, Any] = {}
        for level, packets in sorted(packets_by_level.items()):
            by_case = {packet.get("case_id"): packet for packet in packets}
            table[str(level)] = {
                case_id: {
                    "raw": (by_case.get(case_id) or {}).get("selected_coord_delta"),
                    "display": (by_case.get(case_id) or {}).get("selected_display_coord_delta"),
                    "rotation_quarters": ((by_case.get(case_id) or {}).get("before_state") or {}).get("rotation_quarters"),
                }
                for case_id in MOVE_CASES
            }
        return table

    def build_flow_table(self, packets_by_level: dict[int, list[dict[str, Any]]]) -> dict[str, int | None]:
        table: dict[str, int | None] = {}
        for level, packets in sorted(packets_by_level.items()):
            packet = next((p for p in packets if p.get("case_id") == "activate_flow"), None)
            table[str(level)] = None if packet is None else packet.get("action_frame_count")
        return table

    def build_blocked_table(self, packets_by_level: dict[int, list[dict[str, Any]]]) -> dict[str, int | None]:
        table: dict[str, int | None] = {}
        for level, packets in sorted(packets_by_level.items()):
            packet = next((p for p in packets if p.get("case_id") == "no_selected_motion"), None)
            table[str(level)] = None if packet is None else len(packet.get("setup_actions") or [])
        return table

    def build_primitive_coverage(
        self,
        train_rows: list[dict[str, Any]],
        heldout_rows: list[dict[str, Any]],
    ) -> dict[str, Any]:
        train = Counter((row.get("target") or {}).get("primitive") for row in train_rows)
        heldout = Counter((row.get("target") or {}).get("primitive") for row in heldout_rows)
        missing_train = sorted(ALLOWED_PRIMITIVES - set(train))
        missing_heldout = sorted(ALLOWED_PRIMITIVES - set(heldout))
        if missing_train:
            self.errors.append(f"train split missing primitives: {missing_train}")
        if missing_heldout:
            self.errors.append(f"heldout split missing primitives: {missing_heldout}")
        return {
            "train_by_primitive": dict(sorted(train.items())),
            "heldout_by_primitive": dict(sorted(heldout.items())),
            "missing_train_primitives": missing_train,
            "missing_heldout_primitives": missing_heldout,
        }

    def load_qwen_backend_status(self) -> dict[str, Any]:
        if self.qwen_sweep_dir is None:
            return {"status": "not_configured"}
        summary_path = self.qwen_sweep_dir / "summary.json"
        if not summary_path.is_file():
            return {
                "status": "missing",
                "sweep_dir": relpath(self.qwen_sweep_dir),
                "needed_change": "Run the qwen_http calibration-slice sweep before treating Qwen as attached.",
            }
        try:
            summary = load_json(summary_path)
        except Exception as exc:
            return {"status": "unreadable", "sweep_dir": relpath(self.qwen_sweep_dir), "error": repr(exc)}
        passed = summary.get("passed_calibration_slices", summary.get("passed_levels", []))
        failed = summary.get("failed_calibration_slices", summary.get("failed_levels", []))
        return {
            "status": "passed" if summary.get("status") == "passed" and not failed else "failed",
            "sweep_dir": relpath(self.qwen_sweep_dir),
            "summary_json": relpath(summary_path),
            "backend": summary.get("backend"),
            "passed_calibration_slices": passed,
            "failed_calibration_slices": failed,
            "reporting_note": "Qwen status here means primitive-boundary calibration only, not gameplay completion.",
        }

    def load_lora_dry_run_status(self) -> dict[str, Any]:
        if self.lora_result_dir is None:
            return {"status": "not_configured"}
        result_path = self.lora_result_dir / "result.json"
        if not result_path.is_file():
            return {
                "status": "missing",
                "result_dir": relpath(self.lora_result_dir),
                "needed_change": "Run the offline LoRA dry-run before treating adapter mechanics as attached.",
            }
        try:
            result = load_json(result_path)
        except Exception as exc:
            return {"status": "unreadable", "result_dir": relpath(self.lora_result_dir), "error": repr(exc)}
        sweep_status: dict[str, Any] = {"status": "not_configured"}
        if self.lora_sweep_dir is not None:
            summary_path = self.lora_sweep_dir / "summary.json"
            if summary_path.is_file():
                try:
                    summary = load_json(summary_path)
                    failed = summary.get("failed_calibration_slices", summary.get("failed_levels", []))
                    sweep_status = {
                        "status": "passed" if summary.get("status") == "passed" and not failed else "failed",
                        "sweep_dir": relpath(self.lora_sweep_dir),
                        "summary_json": relpath(summary_path),
                        "passed_calibration_slices": summary.get("passed_calibration_slices", summary.get("passed_levels", [])),
                        "failed_calibration_slices": failed,
                    }
                except Exception as exc:
                    sweep_status = {"status": "unreadable", "sweep_dir": relpath(self.lora_sweep_dir), "error": repr(exc)}
            else:
                sweep_status = {"status": "missing", "sweep_dir": relpath(self.lora_sweep_dir)}
        return {
            "status": "passed" if result.get("status") == "passed" else "failed",
            "result_dir": relpath(self.lora_result_dir),
            "result_json": relpath(result_path),
            "adapter_pt": result.get("adapter_pt"),
            "train_accuracy": result.get("train_accuracy"),
            "heldout_accuracy": result.get("heldout_accuracy"),
            "qwen_weights_used": result.get("qwen_weights_used"),
            "qwen_lora_claim": result.get("qwen_lora_claim"),
            "contra": result.get("contra"),
            "adapter_sweep": sweep_status,
            "reporting_note": "Offline LoRA dry-run means adapter mechanics only, not Qwen training and not gameplay completion.",
        }

    def next_required_change(
        self,
        status: str,
        primitive_coverage: dict[str, Any],
        qwen_backend: dict[str, Any],
        lora_dry_run: dict[str, Any],
    ) -> str:
        if status == "failed":
            return "Fix verifier/evidence/dataset errors before training or Qwen smoke tests."
        if primitive_coverage.get("missing_train_primitives") or primitive_coverage.get("missing_heldout_primitives"):
            return "Regenerate the dataset so each split covers every primitive."
        if lora_dry_run.get("status") == "passed" and (lora_dry_run.get("adapter_sweep") or {}).get("status") == "passed":
            return "Lock a real Qwen-family LoRA condition, or add the next game player with the same primitive/controller/verifier contract."
        if qwen_backend.get("status") == "passed":
            return "Prepare the first LoRA dry-run against the primitive-only SFT rows; keep the motor/controller unchanged."
        return "Smoke-test the qwen_http backend against this same primitive-only interface before LoRA training."

    def write(self, report: dict[str, Any]) -> tuple[Path, Path]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.out_dir / "sp80_debug_report.json"
        md_path = self.out_dir / "sp80_debug_report.md"
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        md_path.write_text(render_markdown(report, json_path), encoding="utf-8")
        return json_path, md_path


def render_markdown(report: dict[str, Any], json_path: Path) -> str:
    lines = [
        "# SP80 Debug Report",
        "",
        f"- Status: `{report['status']}`",
        f"- JSON: `{relpath(json_path)}`",
        "- Reporting term: `calibration slice`",
        "- Provenance note: `level_index` is kept only because the SP80 API uses it; slice status is not gameplay completion.",
        f"- Next required change: {report['next_required_change']}",
        "",
        "## Dataset",
        "",
        f"- Train rows: `{report['dataset']['train']['rows']}`",
        f"- Heldout rows: `{report['dataset']['heldout']['rows']}`",
        f"- Train by primitive: `{json.dumps(report['dataset']['train']['by_primitive'], sort_keys=True)}`",
        f"- Heldout by primitive: `{json.dumps(report['dataset']['heldout']['by_primitive'], sort_keys=True)}`",
        "",
        "## Qwen Backend",
        "",
        f"- Status: `{report['qwen_backend']['status']}`",
        f"- Passed calibration slices: `{report['qwen_backend'].get('passed_calibration_slices', [])}`",
        f"- Failed calibration slices: `{report['qwen_backend'].get('failed_calibration_slices', [])}`",
        "",
        "## LoRA Dry Run",
        "",
        f"- Status: `{report['lora_dry_run']['status']}`",
        f"- Heldout accuracy: `{report['lora_dry_run'].get('heldout_accuracy')}`",
        f"- Adapter sweep status: `{(report['lora_dry_run'].get('adapter_sweep') or {}).get('status')}`",
        f"- Adapter sweep passed calibration slices: `{(report['lora_dry_run'].get('adapter_sweep') or {}).get('passed_calibration_slices', [])}`",
        f"- Contra backend: `{(report['lora_dry_run'].get('contra') or {}).get('backend')}`",
        "",
        "## Sweep",
        "",
        f"- Sweep status: `{report['sweep']['status']}`",
        f"- Passed calibration slices: `{report['calibration']['passed_calibration_slices']}`",
        f"- Failed calibration slices: `{report['calibration']['failed_calibration_slices']}`",
        "",
        "## Drift Pings",
        "",
    ]
    for ping_id in report["tracking"]["drift_ping_ids"]:
        lines.append(f"- `{ping_id}`")
    lines.extend(["", "## Movement Display Deltas", ""])
    for level, cases in report["tracking"]["movement_display_delta_by_calibration_slice"].items():
        deltas = {case: info["display"] for case, info in cases.items()}
        lines.append(f"- Calibration slice `{level}`: `{json.dumps(deltas, sort_keys=True)}`")
    lines.extend(["", "## Flow Frames", ""])
    lines.append(f"`{json.dumps(report['tracking']['flow_action_frame_count_by_calibration_slice'], sort_keys=True)}`")
    lines.extend(["", "## Blocked Setup Counts", ""])
    lines.append(f"`{json.dumps(report['tracking']['blocked_setup_action_count_by_calibration_slice'], sort_keys=True)}`")
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=Path("codexlab/runs/sp80_player_harness_tiny_mlp_levels"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("codexlab/datasets/sp80_control_sft_v0"))
    parser.add_argument("--out-dir", type=Path, default=Path("codexlab/runs/sp80_debug"))
    parser.add_argument("--qwen-sweep-dir", type=Path, default=None)
    parser.add_argument("--lora-result-dir", type=Path, default=Path("codexlab/runs/sp80_lora_dry_run_v0"))
    parser.add_argument("--lora-sweep-dir", type=Path, default=Path("codexlab/runs/sp80_player_harness_lora_adapter_levels"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    builder = DebugReport(
        sweep_dir=resolve_repo_path(args.sweep_dir),
        dataset_dir=resolve_repo_path(args.dataset_dir),
        out_dir=resolve_repo_path(args.out_dir),
        qwen_sweep_dir=resolve_repo_path(args.qwen_sweep_dir) if args.qwen_sweep_dir else None,
        lora_result_dir=resolve_repo_path(args.lora_result_dir) if args.lora_result_dir else None,
        lora_sweep_dir=resolve_repo_path(args.lora_sweep_dir) if args.lora_sweep_dir else None,
    )
    report = builder.build()
    json_path, md_path = builder.write(report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "json_report": relpath(json_path),
                "markdown_report": relpath(md_path),
                "errors": len(report["errors"]),
                "warnings": len(report["warnings"]),
                "next_required_change": report["next_required_change"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
