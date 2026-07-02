#!/usr/bin/env python3
"""Build the offline source-edge activation outcome model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contra_head.crucible.activation import DEFAULT_ACTIVATION_DIR
from contra_head.crucible.source_edge_outcome import (
    DEFAULT_SOURCE_EDGE_OUTCOME_DIR,
    build_source_edge_outcome_model,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-corpus", type=Path, default=DEFAULT_ACTIVATION_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SOURCE_EDGE_OUTCOME_DIR)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = build_source_edge_outcome_model(args.activation_corpus, args.out_dir)
    print(
        json.dumps(
            {
                "schema": manifest.get("schema"),
                "manifest_json": manifest.get("manifest_json"),
                "ledger_json": manifest.get("ledger_json"),
                "entry_count": manifest.get("entry_count"),
                "tier_histogram": manifest.get("tier_histogram"),
                "c0008_contradiction_audit": manifest.get("c0008_contradiction_audit"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
