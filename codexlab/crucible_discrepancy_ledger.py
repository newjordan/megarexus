#!/usr/bin/env python3
"""Build a Crucible discrepancy ledger from the transition corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contra_head.crucible.corpus import DEFAULT_CORPUS_DIR
from contra_head.crucible.ledger import DEFAULT_DISCREPANCY_DIR, build_discrepancy_ledger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DISCREPANCY_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ledger = build_discrepancy_ledger(args.corpus_dir, args.out_dir)
    keys = [
        "schema",
        "ledger_json",
        "anomalies_jsonl",
        "transition_count",
        "anomaly_count",
        "status_counts",
        "anomaly_kind_by_primitive",
        "forward_cell_color_histogram",
    ]
    print(json.dumps({key: ledger.get(key) for key in keys}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

