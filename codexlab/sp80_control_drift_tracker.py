#!/usr/bin/env python3
"""Diagnose SP80 control-space drift from calibration-slice packets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sp80_base_controller import ROOT, sha256_file


MOVE_DISPLAY_EXPECTED = {
    "move_up": [0, -1],
    "move_down": [0, 1],
    "move_left": [-1, 0],
    "move_right": [1, 0],
}

CONTROL_CASES = sorted([*MOVE_DISPLAY_EXPECTED, "select_object", "activate_flow", "no_selected_motion"])


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


def severity_rank(severity: str) -> int:
    return {"info": 0, "notice": 1, "warn": 2, "fail": 3}.get(severity, 3)


class DriftTracker:
    def __init__(self, sweep_dir: Path) -> None:
        self.sweep_dir = sweep_dir.resolve()
        self.errors: list[str] = []

    def run(self, *, out_path: Path | None) -> int:
        report = self.build_report()
        output_path = out_path.resolve() if out_path is not None else self.sweep_dir / "control_drift_report.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary_for_stdout(report, output_path), indent=2, sort_keys=True))
        return 0 if report["status"] != "failed" else 1

    def build_report(self) -> dict[str, Any]:
        summary_path = self.sweep_dir / "summary.json"
        if not summary_path.is_file():
            return self.failed_report([f"missing sweep summary: {summary_path}"])
        summary = load_json(summary_path)
        if summary.get("schema") != "codexlab.sp80_level_sweep.v1":
            return self.failed_report(["bad or missing sweep schema"])

        level_rows = summary.get("rows") or []
        packets_by_level: dict[int, dict[str, dict[str, Any]]] = {}
        packet_sources: dict[int, str] = {}
        for row in level_rows:
            level_index = int(row["level_index"])
            run_dir = ROOT / row["run_dir"]
            packets_path = run_dir / "packets.jsonl"
            if not packets_path.is_file():
                self.errors.append(f"missing packets for level {level_index}: {packets_path}")
                continue
            packets = load_jsonl(packets_path)
            packets_by_level[level_index] = {str(packet.get("case_id")): packet for packet in packets}
            packet_sources[level_index] = relpath(packets_path)

        if self.errors:
            return self.failed_report(self.errors)

        pings = []
        pings.extend(self.detect_raw_display_transform(packets_by_level))
        pings.extend(self.detect_display_stability(packets_by_level))
        pings.extend(self.detect_rotation_lever(packets_by_level))
        pings.extend(self.detect_flow_timing_lever(packets_by_level))
        pings.extend(self.detect_blocked_motion_setup_lever(packets_by_level))
        pings.extend(self.detect_select_auto_focus_lever(packets_by_level))

        max_severity = max((severity_rank(ping["severity"]) for ping in pings), default=0)
        status = "failed" if max_severity >= severity_rank("fail") else "attention" if max_severity >= severity_rank("notice") else "clean"

        return {
            "schema": "codexlab.sp80_control_drift_report.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": status,
            "source_sweep_dir": relpath(self.sweep_dir),
            "source_summary": relpath(summary_path),
            "source_summary_sha256": sha256_file(summary_path),
            "tracker_source": "codexlab/sp80_control_drift_tracker.py",
            "tracker_source_sha256": sha256_file(ROOT / "codexlab" / "sp80_control_drift_tracker.py"),
            "reporting_term": "calibration_slice",
            "reporting_note": "level_index is retained only as SP80 environment provenance; slice status is not gameplay completion.",
            "detected_calibration_slice_count": summary.get(
                "detected_calibration_slice_count",
                summary.get("detected_level_count"),
            ),
            "passed_calibration_slices": summary.get("passed_calibration_slices", summary.get("passed_levels", [])),
            "failed_calibration_slices": summary.get("failed_calibration_slices", summary.get("failed_levels", [])),
            "detected_level_count": summary.get("detected_level_count"),
            "passed_levels": summary.get("passed_levels", []),
            "failed_levels": summary.get("failed_levels", []),
            "packet_sources": packet_sources,
            "pings": pings,
            "recommended_training_space": "selected_display_coord_delta",
            "controller_boundary": "brain emits primitive only; SP80 motor handles coordinate/action transforms",
        }

    def failed_report(self, errors: list[str]) -> dict[str, Any]:
        return {
            "schema": "codexlab.sp80_control_drift_report.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "status": "failed",
            "source_sweep_dir": relpath(self.sweep_dir),
            "errors": errors,
            "pings": [
                {
                    "id": "tracker_input_invalid",
                    "severity": "fail",
                    "message": "Tracker input is missing or malformed.",
                    "needed_change": "Run sp80_sweep_levels.py first, then rerun this tracker on its output directory.",
                    "evidence": errors,
                }
            ],
        }

    def detect_raw_display_transform(
        self,
        packets_by_level: dict[int, dict[str, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        mismatches = []
        per_case: dict[str, list[int]] = defaultdict(list)
        for level, packets in packets_by_level.items():
            for case_id in MOVE_DISPLAY_EXPECTED:
                packet = packets.get(case_id)
                if not packet:
                    continue
                raw_delta = packet.get("selected_coord_delta")
                display_delta = packet.get("selected_display_coord_delta")
                if raw_delta != display_delta:
                    per_case[case_id].append(level)
                    mismatches.append(
                        {
                            "level_index": level,
                            "case_id": case_id,
                            "rotation_quarters": (packet.get("before_state") or {}).get("rotation_quarters"),
                            "raw_delta": raw_delta,
                            "display_delta": display_delta,
                        }
                    )
        if not mismatches:
            return [
                {
                    "id": "raw_display_transform_absent",
                    "severity": "info",
                    "message": "No raw/display movement transform was observed in the sweep.",
                    "needed_change": "None for this sweep.",
                    "evidence": [],
                }
            ]
        return [
            {
                "id": "raw_display_transform_detected",
                "severity": "notice",
                "message": "Some calibration slices rotate or transform raw grid movement relative to visible player controls.",
                "needed_change": "Train and verify movement intent in display-space; leave raw grid transforms inside the SP80 motor/controller.",
                "calibration_slices": sorted({item["level_index"] for item in mismatches}),
                "levels": sorted({item["level_index"] for item in mismatches}),
                "cases_by_calibration_slice": mismatches,
                "cases_by_level": mismatches,
                "cases_by_primitive": {case: levels for case, levels in sorted(per_case.items())},
            }
        ]

    def detect_display_stability(
        self,
        packets_by_level: dict[int, dict[str, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        failures = []
        for level, packets in packets_by_level.items():
            for case_id, expected in MOVE_DISPLAY_EXPECTED.items():
                observed = (packets.get(case_id) or {}).get("selected_display_coord_delta")
                if observed != expected:
                    failures.append({"level_index": level, "case_id": case_id, "expected": expected, "observed": observed})
        if failures:
            return [
                {
                    "id": "display_control_drift",
                    "severity": "fail",
                    "message": "Visible directional control changed across calibration slices.",
                    "needed_change": "Do not train Qwen/LoRA on this sweep until the controller or verifier explains the drift.",
                    "failures": failures,
                }
            ]
        return [
            {
                "id": "display_controls_stable",
                "severity": "info",
                "message": "Visible directional movement stayed stable for all swept calibration slices.",
                "needed_change": "Use display-space deltas as the movement target for primitive training.",
                "expected_display_deltas": MOVE_DISPLAY_EXPECTED,
                "calibration_slices": sorted(packets_by_level),
                "levels": sorted(packets_by_level),
            }
        ]

    def detect_rotation_lever(
        self,
        packets_by_level: dict[int, dict[str, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        rotations: dict[int, list[int]] = defaultdict(list)
        for level, packets in packets_by_level.items():
            packet = packets.get("move_up") or next(iter(packets.values()))
            rotation = (packet.get("before_state") or {}).get("rotation_quarters")
            rotations[int(rotation or 0)].append(level)
        if len(rotations) <= 1:
            return []
        return [
            {
                "id": "calibration_slice_rotation_lever",
                "severity": "notice",
                "message": "SP80 calibration slices expose different rotation settings while preserving visible controls.",
                "needed_change": "Include rotation/display transform fields in LoRA training records and keep motor grounding deterministic.",
                "calibration_slices_by_rotation_quarters": {str(k): v for k, v in sorted(rotations.items())},
                "levels_by_rotation_quarters": {str(k): v for k, v in sorted(rotations.items())},
            }
        ]

    def detect_flow_timing_lever(
        self,
        packets_by_level: dict[int, dict[str, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        frame_counts = {}
        for level, packets in packets_by_level.items():
            packet = packets.get("activate_flow")
            if packet:
                frame_counts[level] = packet.get("action_frame_count")
        unique_counts = sorted({count for count in frame_counts.values() if count is not None})
        if len(unique_counts) <= 1:
            return []
        return [
            {
                "id": "flow_timing_lever",
                "severity": "notice",
                "message": "Activate-flow emits different action-frame counts across calibration slices.",
                "needed_change": "Treat flow as a temporal mechanic; do not train flow understanding from a single final frame only.",
                "action_frame_count_by_calibration_slice": {str(k): v for k, v in sorted(frame_counts.items())},
                "action_frame_count_by_level": {str(k): v for k, v in sorted(frame_counts.items())},
            }
        ]

    def detect_blocked_motion_setup_lever(
        self,
        packets_by_level: dict[int, dict[str, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        setup_lengths = {}
        for level, packets in packets_by_level.items():
            packet = packets.get("no_selected_motion")
            if packet:
                setup_lengths[level] = len(packet.get("setup_actions") or [])
        unique_lengths = sorted(set(setup_lengths.values()))
        if len(unique_lengths) <= 1:
            return []
        return [
            {
                "id": "blocked_motion_setup_lever",
                "severity": "notice",
                "message": "The number of setup moves needed to prove blocked/no-motion differs by calibration slice.",
                "needed_change": "Generate no-motion examples from observed blocked states, not from a fixed one-step setup.",
                "setup_action_count_by_calibration_slice": {str(k): v for k, v in sorted(setup_lengths.items())},
                "setup_action_count_by_level": {str(k): v for k, v in sorted(setup_lengths.items())},
            }
        ]

    def detect_select_auto_focus_lever(
        self,
        packets_by_level: dict[int, dict[str, dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        moved_on_select = []
        for level, packets in packets_by_level.items():
            packet = packets.get("select_object")
            if not packet:
                continue
            raw_delta = packet.get("selected_coord_delta")
            display_delta = packet.get("selected_display_coord_delta")
            if raw_delta not in (None, [0, 0]) or display_delta not in (None, [0, 0]):
                moved_on_select.append(
                    {
                        "level_index": level,
                        "rotation_quarters": (packet.get("before_state") or {}).get("rotation_quarters"),
                        "raw_delta": raw_delta,
                        "display_delta": display_delta,
                    }
                )
        if not moved_on_select:
            return []
        return [
            {
                "id": "select_auto_focus_lever",
                "severity": "notice",
                "message": "Selecting can change the selected object because SP80 may auto-focus one object before click selection.",
                "needed_change": "Score select_object by selected target identity/coord after click, not by zero movement delta.",
                "calibration_slices": [item["level_index"] for item in moved_on_select],
                "levels": [item["level_index"] for item in moved_on_select],
                "cases_by_calibration_slice": moved_on_select,
                "cases_by_level": moved_on_select,
            }
        ]


def summary_for_stdout(report: dict[str, Any], output_path: Path) -> dict[str, Any]:
    return {
        "status": report["status"],
        "report": relpath(output_path),
        "ping_ids": [ping["id"] for ping in report.get("pings", [])],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_dir", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return DriftTracker(ROOT / args.sweep_dir if not args.sweep_dir.is_absolute() else args.sweep_dir).run(
        out_path=ROOT / args.out if args.out is not None and not args.out.is_absolute() else args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
