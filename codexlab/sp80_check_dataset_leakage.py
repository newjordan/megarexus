#!/usr/bin/env python3
"""Check an exported SP80 control SFT dataset for train/heldout leakage.

The exporter (codexlab/sp80_export_control_dataset.py) assigns every calibration
slice (level_index) entirely to one split, so a clean dataset has disjoint
levels, no shared row_ids, and split labels that match the file each row lives
in. This is the post-hoc guard that re-checks those invariants against the files
on disk, plus a content check for byte-identical training inputs appearing in
the heldout split (memorization leakage).

Hard failures (exit 1):
  - a row_id present in both splits, or duplicated within a split
  - a level_index / calibration_slice_index present in both splits
  - a row whose `split` field disagrees with the file it was read from
  - manifest train/heldout level lists that overlap or disagree with the rows

Input overlap (identical `input` JSON across splits) is a failure by default;
pass --allow-input-overlap to downgrade it to a warning (some structured cases
can legitimately produce identical observations across slices).

Usage:
  python3 codexlab/sp80_check_dataset_leakage.py codexlab/datasets/sp80_control_sft_v0
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ROW_SCHEMA = "codexlab.sp80_control_sft_row.v1"


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    resolved.relative_to(ROOT)  # raises if outside the repo
    return resolved


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}")
    return rows


def input_key(row: dict[str, Any]) -> str:
    return json.dumps(row.get("input"), sort_keys=True)


def check_leakage(dataset_dir: Path, *, allow_input_overlap: bool) -> dict[str, Any]:
    train_path = dataset_dir / "train.jsonl"
    heldout_path = dataset_dir / "heldout.jsonl"
    manifest_path = dataset_dir / "manifest.json"
    for path in (train_path, heldout_path):
        if not path.is_file():
            raise SystemExit(f"missing dataset split: {path}")

    train_rows = load_jsonl(train_path)
    heldout_rows = load_jsonl(heldout_path)

    failures: list[str] = []
    warnings: list[str] = []

    # 1. split field must match the file the row came from.
    for split_name, rows in (("train", train_rows), ("heldout", heldout_rows)):
        for row in rows:
            if row.get("split") != split_name:
                failures.append(
                    f"{row.get('row_id')}: in {split_name}.jsonl but split={row.get('split')!r}"
                )

    # 2. row_id duplicates (within and across splits).
    id_counts = Counter(row.get("row_id") for row in (train_rows + heldout_rows))
    for row_id, count in sorted(id_counts.items()):
        if count > 1:
            failures.append(f"duplicate row_id across dataset: {row_id!r} appears {count}x")

    # 3. level_index disjointness between splits.
    def levels(rows: list[dict[str, Any]]) -> set[int]:
        out: set[int] = set()
        for row in rows:
            lvl = row.get("level_index")
            slc = row.get("calibration_slice_index")
            if lvl != slc:
                failures.append(
                    f"{row.get('row_id')}: level_index {lvl} != calibration_slice_index {slc}"
                )
            if lvl is not None:
                out.add(int(lvl))
        return out

    train_levels = levels(train_rows)
    heldout_levels = levels(heldout_rows)
    shared_levels = sorted(train_levels & heldout_levels)
    if shared_levels:
        failures.append(f"levels present in BOTH splits (leakage): {shared_levels}")

    # 4. content overlap: identical input JSON across splits.
    train_inputs = {input_key(row): row.get("row_id") for row in train_rows}
    overlap_rows = [
        (row.get("row_id"), train_inputs[input_key(row)])
        for row in heldout_rows
        if input_key(row) in train_inputs
    ]
    if overlap_rows:
        detail = ", ".join(f"{h}~{t}" for h, t in overlap_rows)
        msg = f"{len(overlap_rows)} heldout rows have an input identical to a train row: {detail}"
        (warnings if allow_input_overlap else failures).append(msg)

    # 5. cross-check the manifest, if present.
    manifest_check: dict[str, Any] = {"present": manifest_path.is_file()}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        m_train = set(int(x) for x in manifest.get("train_levels", []))
        m_heldout = set(int(x) for x in manifest.get("heldout_levels", []))
        manifest_check.update(
            {
                "manifest_train_levels": sorted(m_train),
                "manifest_heldout_levels": sorted(m_heldout),
            }
        )
        if m_train & m_heldout:
            failures.append(
                f"manifest train_levels and heldout_levels overlap: {sorted(m_train & m_heldout)}"
            )
        if m_train != train_levels:
            failures.append(
                f"manifest train_levels {sorted(m_train)} != levels in train.jsonl {sorted(train_levels)}"
            )
        if m_heldout != heldout_levels:
            failures.append(
                f"manifest heldout_levels {sorted(m_heldout)} != levels in heldout.jsonl {sorted(heldout_levels)}"
            )

    return {
        "schema": "codexlab.sp80_dataset_leakage_report.v1",
        "dataset_dir": dataset_dir.relative_to(ROOT).as_posix(),
        "status": "passed" if not failures else "failed",
        "train_count": len(train_rows),
        "heldout_count": len(heldout_rows),
        "train_levels": sorted(train_levels),
        "heldout_levels": sorted(heldout_levels),
        "shared_levels": shared_levels,
        "input_overlap_count": len(overlap_rows),
        "allow_input_overlap": allow_input_overlap,
        "manifest": manifest_check,
        "failures": failures,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "dataset_dir",
        type=Path,
        nargs="?",
        default=Path("codexlab/datasets/sp80_control_sft_v0"),
    )
    parser.add_argument("--allow-input-overlap", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = resolve_repo_path(str(args.dataset_dir))
    report = check_leakage(dataset_dir, allow_input_overlap=args.allow_input_overlap)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
