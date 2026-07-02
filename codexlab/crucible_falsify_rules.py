#!/usr/bin/env python3
"""Stage D: replay-falsify Crucible candidate rules and build the mechanic ledger.

Loads the Stage C rules artifact and the transition corpus, replays every rule
against every MOVE transition, promotes only zero-contradiction, well-supported,
multi-run rules, prunes redundant rules, and records the composed-model residual
mismatch versus the v0 baseline.

Durable outputs:
  codexlab/datasets/sp80_crucible_mechanics_v0/mechanic_ledger.json
  codexlab/datasets/sp80_crucible_mechanics_v0/manifest.json
  codexlab/runs/_crucible/crucible_falsify_<UTCstamp>/  (run snapshot)

Boundary: the promoted set is a hypothesis-ledger composition, not control truth.
All artifacts carry all-false truth flags and the boundary literal
`mechanic_forward_model_is_hypothesis_ledger_composition_not_control_truth`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from contra_head.crucible.corpus import ROOT, rel, resolve_repo_path, sha256_file, source_hashes
from contra_head.crucible.falsify import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_MECHANICS_DIR,
    DEFAULT_RULES_DIR,
    MECHANIC_MANIFEST_SCHEMA,
    build_mechanic_ledger,
    build_move_samples,
    corpus_paths,
    read_jsonl,
)
from contra_head.crucible.grammar import truth_boundary
from contra_head.crucible.ledger import DEFAULT_DISCREPANCY_DIR


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def falsify_rules(
    rules_dir: Path = DEFAULT_RULES_DIR,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    out_dir: Path = DEFAULT_MECHANICS_DIR,
    discrepancy_dir: Path = DEFAULT_DISCREPANCY_DIR,
) -> dict[str, Any]:
    rules_root = rules_dir if rules_dir.is_absolute() else ROOT / rules_dir
    rules_manifest_path = rules_root / "manifest.json"
    rules_manifest = json.loads(rules_manifest_path.read_text(encoding="utf-8"))
    rules_path = resolve_repo_path(str(rules_manifest.get("rules_jsonl") or rules_root / "rules.jsonl"))
    rules = read_jsonl(rules_path)

    corpus_manifest_path, corpus_jsonl_path = corpus_paths(corpus_dir)
    corpus_rows = read_jsonl(corpus_jsonl_path)
    samples = build_move_samples(corpus_rows)

    discrepancy_root = discrepancy_dir if discrepancy_dir.is_absolute() else ROOT / discrepancy_dir
    discrepancy_ref = rel(discrepancy_root / "ledger.json") if (discrepancy_root / "ledger.json").is_file() else None

    ledger = build_mechanic_ledger(
        rules,
        samples,
        rules_manifest_path=rules_manifest_path,
        corpus_manifest_path=corpus_manifest_path,
        corpus_jsonl_path=corpus_jsonl_path,
        discrepancy_ledger_ref=discrepancy_ref,
    )

    resolved_out = out_dir if out_dir.is_absolute() else ROOT / out_dir
    resolved_out.mkdir(parents=True, exist_ok=True)
    ledger_path = resolved_out / "mechanic_ledger.json"
    ledger["mechanic_ledger_json"] = rel(ledger_path)
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_path = resolved_out / "manifest.json"
    manifest = {
        "schema": MECHANIC_MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest_json": rel(manifest_path),
        "mechanic_ledger_json": rel(ledger_path),
        "ledger_sha256": ledger["ledger_sha256"],
        "mechanic_ledger_file_sha256": sha256_file(ledger_path),
        "grammar_version": ledger["grammar_version"],
        "candidate_rule_count": ledger["candidate_rule_count"],
        "promoted_rule_count": ledger["promoted_rule_count"],
        "minimal_promoted_rule_count": ledger["minimal_promoted_rule_count"],
        "promoted_effect_histogram": ledger["promoted_effect_histogram"],
        "composed_model": ledger["composed_model"],
        "source_hashes": source_hashes(
            [
                "codexlab/contra_head/crucible/grammar.py",
                "codexlab/contra_head/crucible/falsify.py",
                "codexlab/crucible_falsify_rules.py",
            ]
        ),
        "truth_boundary": truth_boundary(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_dir = ROOT / "codexlab" / "runs" / "_crucible" / f"crucible_falsify_{_utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stage_summary.json").write_text(
        json.dumps(
            {
                "stage": "D_falsification",
                "grammar_version": ledger["grammar_version"],
                "candidate_rule_count": ledger["candidate_rule_count"],
                "promoted_rule_count": ledger["promoted_rule_count"],
                "minimal_promoted_rule_count": ledger["minimal_promoted_rule_count"],
                "promoted_effect_histogram": ledger["promoted_effect_histogram"],
                "composed_model": ledger["composed_model"],
                "truth_boundary": truth_boundary(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest["run_dir"] = rel(run_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rules-dir", type=Path, default=DEFAULT_RULES_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_MECHANICS_DIR)
    parser.add_argument("--discrepancy-dir", type=Path, default=DEFAULT_DISCREPANCY_DIR)
    args = parser.parse_args()
    manifest = falsify_rules(args.rules_dir, args.corpus_dir, args.out_dir, args.discrepancy_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
