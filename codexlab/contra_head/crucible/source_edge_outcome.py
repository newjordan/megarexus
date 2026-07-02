"""Build offline source-edge activation outcome evidence from activation rows."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .activation import DEFAULT_ACTIVATION_DIR
from .corpus import ROOT, read_jsonl, rel, resolve_repo_path, sha256_file, source_hashes, stable_json


DEFAULT_SOURCE_EDGE_OUTCOME_DIR = ROOT / "codexlab" / "datasets" / "sp80_source_edge_outcome_v0"
SOURCE_EDGE_OUTCOME_SCHEMA = "codexlab.source_edge_outcome_ledger.v1"
SOURCE_EDGE_OUTCOME_MANIFEST_SCHEMA = "codexlab.source_edge_outcome_manifest.v1"
SOURCE_EDGE_OUTCOME_BOUNDARY_LITERAL = "source_edge_outcome_model_is_activation_hypothesis_evidence_not_control_truth"


def truth_boundary() -> dict[str, Any]:
    return {
        "control_truth_admitted": False,
        "goal_truth_admitted": False,
        "goal_condition_admitted": False,
        "solve_truth_admitted": False,
        "solve_condition_admitted": False,
        "action_choice_admitted": False,
        "route_or_replay_used": False,
        "coordinate_copy_used": False,
        "boundary": SOURCE_EDGE_OUTCOME_BOUNDARY_LITERAL,
    }


def resolve_activation_corpus(path: Path = DEFAULT_ACTIVATION_DIR) -> tuple[Path, dict[str, Any]]:
    base = path if path.is_absolute() else ROOT / path
    manifest_path = base / "manifest.json" if base.is_dir() else base
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_path = resolve_repo_path(str(manifest.get("corpus_jsonl") or ""))
    return corpus_path, manifest


def selected_identity(selection: dict[str, Any]) -> str:
    name = selection.get("selected_name")
    if isinstance(name, str) and name:
        return name
    shape = selection.get("selected_shape")
    digest = hashlib.sha256(stable_json(shape).encode("utf-8")).hexdigest()[:16]
    return f"shape_sha256:{digest}"


def key_for_row(row: dict[str, Any]) -> dict[str, Any]:
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    context = row.get("source_context") if isinstance(row.get("source_context"), dict) else {}
    click = row.get("click") if isinstance(row.get("click"), dict) else {}
    selection = row.get("selection_before") if isinstance(row.get("selection_before"), dict) else {}
    return {
        "primitive": action.get("primitive") or "activate_flow",
        "active_portband_source_key": context.get("active_portband_source_key"),
        "focus_edge": context.get("focus_edge"),
        "click_display_coord": click.get("display_coord"),
        "before_selected_identity": selected_identity(selection),
    }


def _example(row: dict[str, Any]) -> dict[str, Any]:
    context = row.get("source_context") if isinstance(row.get("source_context"), dict) else {}
    progress = row.get("goal_progress") if isinstance(row.get("goal_progress"), dict) else {}
    provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
    selection_before = row.get("selection_before") if isinstance(row.get("selection_before"), dict) else {}
    selection_after = row.get("selection_after") if isinstance(row.get("selection_after"), dict) else {}
    return {
        "activation_id": row.get("activation_id"),
        "key": key_for_row(row),
        "outcome_class": row.get("outcome_class"),
        "distance_delta": progress.get("distance_delta"),
        "levels_completed_delta": row.get("levels_completed_delta"),
        "target_distance": context.get("target_distance"),
        "candidate_keys": context.get("candidate_keys"),
        "before_selected_identity": selected_identity(selection_before),
        "after_selected_identity": selected_identity(selection_after),
        "run_dir": provenance.get("run_dir"),
        "packets_jsonl": provenance.get("packets_jsonl"),
        "line_number": provenance.get("line_number"),
        "action_index": provenance.get("action_index"),
    }


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    phat = successes / total
    denom = 1 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
    return max(0.0, (centre - margin) / denom)


def _tier(contradictions: int, support: int, wilson: float, spans: int) -> str:
    if contradictions == 0 and support >= 8 and wilson >= 0.75 and spans >= 2:
        return "promoted"
    if contradictions == 0 and support >= 2 and spans >= 2:
        return "candidate"
    return "observed"


def _entry(group_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    examples = [_example(row) for row in rows]
    outcomes = Counter(str(example.get("outcome_class")) for example in examples)
    support = len(examples)
    favorable = outcomes.get("level_completed", 0) + outcomes.get("distance_positive", 0)
    contradictions = 0 if len(outcomes) <= 1 else support - max(outcomes.values())
    spans = len({str(example.get("run_dir")) for example in examples})
    wilson = round(wilson_lower_bound(favorable, support), 6)
    tier = _tier(contradictions, support, wilson, spans)
    return {
        "entry_id": hashlib.sha256(group_key.encode("utf-8")).hexdigest(),
        "key": examples[0]["key"],
        "outcome_histogram": dict(sorted(outcomes.items())),
        "support": support,
        "contradictions": contradictions,
        "wilson_lower_bound": wilson,
        "spans_run_dirs": spans,
        "observed_examples": examples,
        "tier": tier,
        "truth_boundary": truth_boundary(),
    }


def c0008_contradiction_audit(entries: list[dict[str, Any]]) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for entry in entries:
        for example in entry.get("observed_examples") or []:
            key = example.get("key") if isinstance(example.get("key"), dict) else {}
            if (
                key.get("active_portband_source_key") == "source_component:c0008"
                and key.get("focus_edge") == 0
                and key.get("click_display_coord") == [62, 52]
                and example.get("outcome_class") in {"distance_positive", "distance_negative"}
            ):
                matches.append(example)
    positives = [item for item in matches if item.get("outcome_class") == "distance_positive"]
    negatives = [item for item in matches if item.get("outcome_class") == "distance_negative"]
    if not positives or not negatives:
        return {
            "separated": False,
            "reason": "c0008_rows_not_found",
            "matching_example_count": len(matches),
            "separating_fields": [],
        }
    positive = positives[0]
    negative = negatives[0]
    key_a = positive.get("key")
    key_b = negative.get("key")
    separated = key_a != key_b
    return {
        "separated": bool(separated),
        "reason": None if separated else "available_fields_cannot_separate",
        "separating_fields": ["before_selected_identity"] if separated else [],
        "key_a": key_a,
        "key_b": key_b,
        "positive_example": positive,
        "negative_example": negative,
    }


def build_source_edge_outcome_model(
    activation_corpus: Path = DEFAULT_ACTIVATION_DIR,
    out_dir: Path = DEFAULT_SOURCE_EDGE_OUTCOME_DIR,
) -> dict[str, Any]:
    corpus_path, activation_manifest = resolve_activation_corpus(activation_corpus)
    rows = read_jsonl(corpus_path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = key_for_row(row)
        groups[stable_json(key)].append(row)
    entries = [_entry(key, group_rows) for key, group_rows in sorted(groups.items())]
    audit = c0008_contradiction_audit(entries)
    resolved_out = out_dir if out_dir.is_absolute() else ROOT / out_dir
    resolved_out.mkdir(parents=True, exist_ok=True)
    ledger_path = resolved_out / "ledger.json"
    manifest_path = resolved_out / "manifest.json"
    tier_histogram = Counter(str(entry.get("tier")) for entry in entries)
    ledger = {
        "schema": SOURCE_EDGE_OUTCOME_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "ledger_json": rel(ledger_path),
        "activation_corpus_jsonl": rel(corpus_path),
        "activation_corpus_sha256": sha256_file(corpus_path),
        "activation_manifest": activation_manifest.get("manifest_json"),
        "activation_manifest_sha256": sha256_file(resolve_repo_path(str(activation_manifest.get("manifest_json") or ""))),
        "boundary": SOURCE_EDGE_OUTCOME_BOUNDARY_LITERAL,
        "entry_count": len(entries),
        "entries": entries,
        "c0008_contradiction_audit": audit,
        "promotion_gates": {
            "promoted": {
                "contradictions": 0,
                "support_min": 8,
                "wilson_lower_bound_min": 0.75,
                "spans_run_dirs_min": 2,
            },
            "candidate": {
                "contradictions": 0,
                "support_min": 2,
                "spans_run_dirs_min": 2,
                "non_consumable": True,
            },
        },
        "truth_boundary": truth_boundary(),
    }
    ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": SOURCE_EDGE_OUTCOME_MANIFEST_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest_json": rel(manifest_path),
        "ledger_json": rel(ledger_path),
        "ledger_sha256": sha256_file(ledger_path),
        "entry_count": len(entries),
        "tier_histogram": dict(sorted(tier_histogram.items())),
        "c0008_contradiction_audit": audit,
        "source_hashes": source_hashes(
            [
                "codexlab/contra_head/crucible/source_edge_outcome.py",
                "codexlab/crucible_build_source_edge_outcome_model.py",
            ]
        ),
        "truth_boundary": truth_boundary(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
