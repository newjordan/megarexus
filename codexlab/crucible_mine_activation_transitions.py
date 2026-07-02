#!/usr/bin/env python3
"""Mine the Crucible SP80 activation-transition corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from contra_head.crucible.activation import DEFAULT_ACTIVATION_DIR, mine_activation_corpus
from contra_head.crucible.corpus import DEFAULT_ADMITTED_MANIFEST


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("codexlab/runs"))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_ACTIVATION_DIR)
    parser.add_argument("--admitted-manifest", type=Path, default=DEFAULT_ADMITTED_MANIFEST)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = mine_activation_corpus(args.runs_root, args.out_dir, args.admitted_manifest)
    keys = [
        "schema",
        "manifest_json",
        "corpus_jsonl",
        "row_count",
        "field_presence_histogram",
        "outcome_class_histogram",
        "not_admitted_skipped_count",
        "skip_reasons",
    ]
    print(json.dumps({key: manifest.get(key) for key in keys}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
