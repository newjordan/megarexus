#!/usr/bin/env python3
"""Mine the Crucible SP80 transition corpus from controlled run packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contra_head.crucible.corpus import DEFAULT_ADMITTED_MANIFEST, DEFAULT_CORPUS_DIR, mine_transition_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("codexlab/runs"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--admitted-manifest", type=Path, default=DEFAULT_ADMITTED_MANIFEST)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = mine_transition_corpus(args.runs_root, args.out_dir, args.admitted_manifest)
    keys = [
        "schema",
        "manifest_json",
        "corpus_jsonl",
        "row_count",
        "dedup_duplicate_count",
        "scanned_packets_jsonl_count",
        "admitted_manifest",
        "not_admitted_skipped_count",
        "skip_reasons",
        "baseline_mismatch_summary",
    ]
    print(json.dumps({key: manifest.get(key) for key in keys}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
