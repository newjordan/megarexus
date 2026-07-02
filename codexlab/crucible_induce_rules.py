#!/usr/bin/env python3
"""Stage C: induce Crucible candidate rules from the discrepancy anomalies.

Loads the transition corpus and the discrepancy anomalies, enumerates the
`crucible_grammar.v0` candidate rule set (exhaustive single-atom rules plus
data-driven two-atom conjunctions instantiated only from explaining anomaly
contexts), and writes a durable rules artifact.

Durable outputs:
  codexlab/datasets/sp80_crucible_rules_v0/rules.jsonl
  codexlab/datasets/sp80_crucible_rules_v0/manifest.json
  codexlab/runs/_crucible/crucible_induce_<UTCstamp>/  (run snapshot)

Boundary: candidate rules are falsifiable hypotheses, not control truth. The
artifact carries all-false truth flags and the boundary literal
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
from contra_head.crucible.grammar import (
    GRAMMAR_VERSION,
    RULES_MANIFEST_SCHEMA,
    enumerate_rules,
    truth_boundary,
)
from contra_head.crucible.ledger import DEFAULT_DISCREPANCY_DIR


DEFAULT_RULES_DIR = ROOT / "codexlab" / "datasets" / "sp80_crucible_rules_v0"
DEFAULT_CORPUS_DIR = ROOT / "codexlab" / "datasets" / "sp80_crucible_transitions_v0"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def induce_rules(
    discrepancy_dir: Path = DEFAULT_DISCREPANCY_DIR,
    out_dir: Path = DEFAULT_RULES_DIR,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
) -> dict[str, Any]:
    discrepancy_root = discrepancy_dir if discrepancy_dir.is_absolute() else ROOT / discrepancy_dir
    ledger_path = discrepancy_root / "ledger.json"
    discrepancy_ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    anomalies_path = resolve_repo_path(str(discrepancy_ledger.get("anomalies_jsonl") or discrepancy_root / "anomalies.jsonl"))
    anomalies = _read_jsonl(anomalies_path)

    corpus_root = corpus_dir if corpus_dir.is_absolute() else ROOT / corpus_dir
    corpus_manifest_path = corpus_root / "manifest.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    corpus_jsonl_path = resolve_repo_path(str(corpus_manifest.get("corpus_jsonl") or corpus_root / "corpus.jsonl"))

    rules = enumerate_rules(anomalies)

    resolved_out = out_dir if out_dir.is_absolute() else ROOT / out_dir
    resolved_out.mkdir(parents=True, exist_ok=True)
    rules_path = resolved_out / "rules.jsonl"
    with rules_path.open("w", encoding="utf-8") as handle:
        for rule in rules:
            handle.write(json.dumps(rule, sort_keys=True) + "\n")

    selector_hist = Counter(
        (rule.get("selector") or {}).get("kind", "?")
        if (rule.get("selector") or {}).get("kind") != "per_primitive"
        else f"per_primitive:{(rule.get('selector') or {}).get('primitive')}"
        for rule in rules
    )
    effect_hist = Counter(str(rule.get("effect")) for rule in rules)
    arity_hist = Counter(int(rule.get("arity") or 0) for rule in rules)
    induced_rule_count = sum(1 for rule in rules if rule.get("induced_from"))

    manifest = {
        "schema": RULES_MANIFEST_SCHEMA,
        "grammar_version": GRAMMAR_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "rules_jsonl": rel(rules_path),
        "manifest_json": rel(resolved_out / "manifest.json"),
        "discrepancy_ledger": rel(ledger_path),
        "discrepancy_ledger_sha256": sha256_file(ledger_path),
        "anomalies_jsonl": rel(anomalies_path),
        "anomalies_jsonl_sha256": sha256_file(anomalies_path),
        "anomaly_count": len(anomalies),
        "corpus_manifest": rel(corpus_manifest_path),
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "corpus_jsonl": rel(corpus_jsonl_path),
        "rule_count": len(rules),
        "induced_rule_count": induced_rule_count,
        "selector_histogram": dict(sorted(selector_hist.items())),
        "effect_histogram": dict(sorted(effect_hist.items())),
        "arity_histogram": {str(key): value for key, value in sorted(arity_hist.items())},
        "source_hashes": source_hashes(
            [
                "codexlab/contra_head/crucible/grammar.py",
                "codexlab/crucible_induce_rules.py",
            ]
        ),
        "truth_boundary": truth_boundary(),
    }
    manifest_path = resolved_out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    run_dir = ROOT / "codexlab" / "runs" / "_crucible" / f"crucible_induce_{_utc_stamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stage_summary.json").write_text(
        json.dumps(
            {
                "stage": "C_rule_induction",
                "grammar_version": GRAMMAR_VERSION,
                "rules_jsonl": rel(rules_path),
                "rule_count": len(rules),
                "induced_rule_count": induced_rule_count,
                "effect_histogram": dict(sorted(effect_hist.items())),
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
    parser.add_argument("--discrepancy-dir", type=Path, default=DEFAULT_DISCREPANCY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RULES_DIR)
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    args = parser.parse_args()
    manifest = induce_rules(args.discrepancy_dir, args.out_dir, args.corpus_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
