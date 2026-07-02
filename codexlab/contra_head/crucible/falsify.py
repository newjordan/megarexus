"""Stage D replay falsification and mechanic-ledger composition for Crucible.

Every candidate rule from Stage C is replayed against every corpus MOVE
transition (selection_changed rows excluded). When a rule's condition fires we
compare its predicted actor delta against the observed delta and record support
or a contradiction. Only zero-contradiction, well-supported, multi-run rules are
promoted into the forward model.

Authority boundary: the promoted set is a hypothesis-ledger composition, not
control truth. All artifacts carry all-false truth flags and the boundary
literal `mechanic_forward_model_is_hypothesis_ledger_composition_not_control_truth`.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .corpus import (
    ROOT,
    rel,
    resolve_repo_path,
    sha256_file,
    source_hashes,
    stable_id,
    stable_json,
)
from .forward_v0 import MOVE_DELTAS, PREDICT_V0_VERSION
from .grammar import (
    GRAMMAR_VERSION,
    EFFECTS,
    effect_delta,
    eval_condition,
    selector_applies,
    selector_at_least_as_general,
    truth_boundary,
)
from .ledger import build_local_context
from .model import _PredictionShim

# REUSE the Wilson lower-bound reliability math. Its default z is 1.96; the
# Stage D contract pins z=1.0, so we always pass it explicitly. This matches the
# attribution lineage from git b77d0a43:codexlab/contra/ghost_mechanics.py.
from .source_edge_outcome import wilson_lower_bound


MECHANIC_LEDGER_SCHEMA = "codexlab.crucible_mechanic_ledger.v1"
MECHANIC_MANIFEST_SCHEMA = "codexlab.crucible_mechanic_ledger_manifest.v1"
DEFAULT_MECHANICS_DIR = ROOT / "codexlab" / "datasets" / "sp80_crucible_mechanics_v0"
DEFAULT_RULES_DIR = ROOT / "codexlab" / "datasets" / "sp80_crucible_rules_v0"
DEFAULT_CORPUS_DIR = ROOT / "codexlab" / "datasets" / "sp80_crucible_transitions_v0"

PROMOTION_SUPPORT_MIN = 8
PROMOTION_RELIABILITY_MIN = 0.75
PROMOTION_SPANS_MIN = 2
PROMOTION_Z = 1.0
CONTRADICTION_ID_CAP = 20

# Map grammar effects to the effect_kind / blocked flags that the unchanged
# CrucibleModel.predict() understands.
_EFFECT_KIND = {
    "blocked": "blocked",
    "forward": "forward",
    "redirect_left": "redirect_left",
    "redirect_right": "redirect_right",
    "reverse": "reverse",
    "slide_until_blocked": "slide",
}


def _tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    a, b = value
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    return int(a), int(b)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def corpus_paths(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> tuple[Path, Path]:
    root = corpus_dir if corpus_dir.is_absolute() else ROOT / corpus_dir
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_path = resolve_repo_path(str(manifest.get("corpus_jsonl") or root / "corpus.jsonl"))
    return manifest_path, corpus_path


def build_move_samples(corpus_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Precompute a compact falsification sample for every MOVE transition.

    Each sample caches the 5x5 canonical neighborhood, the actor axis, the
    observed delta, and the effect->delta table for the issued direction so the
    falsification scan is pure dictionary work.
    """
    samples: list[dict[str, Any]] = []
    for row in corpus_rows:
        primitive = str(((row.get("action") or {}).get("primitive") or ""))
        actor = row.get("actor") if isinstance(row.get("actor"), dict) else {}
        if primitive not in MOVE_DELTAS or actor.get("selection_changed") is True:
            continue
        observed = _tuple((row.get("observed") or {}).get("canon_delta"))
        if observed is None:
            continue
        try:
            context = build_local_context(row, _PredictionShim(MOVE_DELTAS[primitive]))
        except Exception:
            continue
        neighborhood = context.get("canon_neighborhood_5x5")
        if not isinstance(neighborhood, list) or len(neighborhood) != 5:
            continue
        direction = MOVE_DELTAS[primitive]
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        samples.append(
            {
                "primitive": primitive,
                "direction": direction,
                "neighborhood": [[int(cell) for cell in line] for line in neighborhood],
                "actor_axis": context.get("actor_axis"),
                "observed": observed,
                "effect_deltas": {effect: effect_delta(effect, direction, neighborhood) for effect in EFFECTS},
                "run_dir": str(provenance.get("run_dir") or ""),
                "transition_id": str(row.get("transition_id") or ""),
            }
        )
    return samples


def evaluate_rule(rule: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Replay one rule over the move samples and return falsification stats."""
    selector = rule.get("selector") if isinstance(rule.get("selector"), dict) else {}
    atoms = ((rule.get("condition") or {}).get("atoms")) or []
    effect = str(rule.get("effect") or "")
    support = 0
    contradictions = 0
    support_run_dirs: set[str] = set()
    contradicting_ids: list[str] = []
    for sample in samples:
        if not selector_applies(selector, sample["primitive"]):
            continue
        if not eval_condition(atoms, sample["neighborhood"], sample["actor_axis"], sample["direction"]):
            continue
        predicted = sample["effect_deltas"].get(effect, (0, 0))
        if predicted == sample["observed"]:
            support += 1
            support_run_dirs.add(sample["run_dir"])
        else:
            contradictions += 1
            if len(contradicting_ids) < CONTRADICTION_ID_CAP:
                contradicting_ids.append(sample["transition_id"])
    total = support + contradictions
    reliability = round(wilson_lower_bound(support, total, z=PROMOTION_Z), 6) if total else 0.0
    spans = len(support_run_dirs)
    accuracy = round(support / total, 6) if total else 0.0
    promoted = (
        contradictions == 0
        and support >= PROMOTION_SUPPORT_MIN
        and reliability >= PROMOTION_RELIABILITY_MIN
        and spans >= PROMOTION_SPANS_MIN
    )
    return {
        "support": support,
        "contradictions": contradictions,
        "reliability": reliability,
        "accuracy": accuracy,
        "spans_run_dirs": spans,
        "support_run_dirs": sorted(support_run_dirs),
        "contradicting_transition_ids": contradicting_ids,
        "promoted": promoted,
    }


def _atom_key_set(rule: dict[str, Any]) -> frozenset[str]:
    atoms = ((rule.get("condition") or {}).get("atoms")) or []
    return frozenset(stable_json(atom) for atom in atoms)


def prune_redundant(promoted: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mark rules redundant when a strictly-more-general promoted rule exists.

    A rule B is redundant if some other promoted rule A has the same effect, a
    selector at least as general, and an atom set that is a subset of B's, with
    A strictly more general (proper atom subset or strictly broader selector).
    Exact-condition conflicts (same selector+atoms, different effect) are kept
    and flagged on both sides.
    """
    minimal: list[dict[str, Any]] = []
    for rule in promoted:
        rule_atoms = _atom_key_set(rule)
        rule_selector = rule.get("selector") or {}
        subsumed_by = None
        conflict_with: list[str] = []
        for other in promoted:
            if other is rule:
                continue
            other_atoms = _atom_key_set(other)
            other_selector = other.get("selector") or {}
            if other_atoms == rule_atoms and other_selector == rule_selector and other.get("effect") != rule.get("effect"):
                conflict_with.append(str(other.get("rule_id")))
                continue
            if other.get("effect") != rule.get("effect"):
                continue
            if not selector_at_least_as_general(other_selector, rule_selector):
                continue
            if not other_atoms.issubset(rule_atoms):
                continue
            strictly_more_general = (
                other_atoms < rule_atoms
                or (
                    other_atoms == rule_atoms
                    and selector_at_least_as_general(other_selector, rule_selector)
                    and not selector_at_least_as_general(rule_selector, other_selector)
                )
            )
            if strictly_more_general:
                subsumed_by = str(other.get("rule_id"))
        rule["pruned_redundant"] = subsumed_by is not None
        rule["subsumed_by"] = subsumed_by
        rule["exact_conflict_with"] = sorted(set(conflict_with))
        if subsumed_by is None:
            minimal.append(rule)
    pruned = [rule for rule in promoted if rule.get("pruned_redundant")]
    return minimal, pruned


def _predict_for_sample(promoted: list[dict[str, Any]], sample: dict[str, Any]) -> tuple[int, int]:
    """Composed-model prediction: first firing promoted rule wins, else forward."""
    for rule in promoted:
        selector = rule.get("selector") or {}
        if not selector_applies(selector, sample["primitive"]):
            continue
        atoms = ((rule.get("condition") or {}).get("atoms")) or []
        if eval_condition(atoms, sample["neighborhood"], sample["actor_axis"], sample["direction"]):
            return sample["effect_deltas"].get(str(rule.get("effect")), (0, 0))
    return sample["direction"]


def composed_residual(promoted: list[dict[str, Any]], samples: list[dict[str, Any]]) -> dict[str, Any]:
    # Deterministic priority: more specific (higher arity) first, then rule_id.
    ordered = sorted(
        promoted,
        key=lambda rule: (-int(rule.get("arity") or 0), str(rule.get("rule_id"))),
    )
    total = len(samples)
    baseline_mismatch = 0
    composed_mismatch = 0
    for sample in samples:
        observed = sample["observed"]
        if sample["direction"] != observed:
            baseline_mismatch += 1
        if _predict_for_sample(ordered, sample) != observed:
            composed_mismatch += 1
    return {
        "move_transition_count": total,
        "baseline_model_version": PREDICT_V0_VERSION,
        "baseline_mismatch_count": baseline_mismatch,
        "baseline_mismatch_rate": round(baseline_mismatch / total, 6) if total else 0.0,
        "residual_mismatch_count": composed_mismatch,
        "residual_mismatch_rate": round(composed_mismatch / total, 6) if total else 0.0,
        "improvement_count": baseline_mismatch - composed_mismatch,
        "residual_no_worse_than_baseline": composed_mismatch <= baseline_mismatch,
    }


def _back_compat_fields(rule: dict[str, Any], stats: dict[str, Any]) -> dict[str, Any]:
    """Fields that let the unchanged CrucibleModel.predict() consume a rule.

    Only per_primitive single cell_color(ahead1, C) rules reduce to predict()'s
    forward_cell_color form; richer rules carry no forward_cell_color and are
    intentionally invisible to predict() (a documented v0 limitation).
    """
    selector = rule.get("selector") or {}
    atoms = ((rule.get("condition") or {}).get("atoms")) or []
    effect = str(rule.get("effect") or "")
    primitive = selector.get("primitive") if selector.get("kind") == "per_primitive" else None
    fields: dict[str, Any] = {
        "primitive": primitive,
        "effect_kind": _EFFECT_KIND.get(effect, effect),
        "confidence": stats["reliability"],
    }
    if primitive is not None and effect != "slide_until_blocked":
        delta = effect_delta(effect, MOVE_DELTAS[primitive])
        fields["predicted_canon_delta"] = [delta[0], delta[1]]
    else:
        fields["predicted_canon_delta"] = None
    # Single ahead1 cell_color atom -> forward_cell_color condition for predict().
    if len(atoms) == 1 and atoms[0].get("atom") == "cell_color" and atoms[0].get("offset") == "ahead1":
        color = atoms[0].get("color")
        fields["forward_cell_color"] = -1 if color == "OOB" else int(color)
    return fields


def build_mechanic_ledger(
    rules: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    *,
    rules_manifest_path: Path,
    corpus_manifest_path: Path,
    corpus_jsonl_path: Path,
    discrepancy_ledger_ref: str | None,
) -> dict[str, Any]:
    candidate_count = len(rules)
    promoted: list[dict[str, Any]] = []
    for rule in rules:
        stats = evaluate_rule(rule, samples)
        if not stats["promoted"]:
            continue
        entry = dict(rule)
        entry.pop("truth_boundary", None)
        condition = dict(entry.get("condition") or {})
        back_compat = _back_compat_fields(rule, stats)
        forward_cell_color = back_compat.pop("forward_cell_color", None)
        if forward_cell_color is not None:
            condition["forward_cell_color"] = forward_cell_color
        entry["condition"] = condition
        entry.update(back_compat)
        entry["support"] = stats["support"]
        entry["contradictions"] = stats["contradictions"]
        entry["reliability"] = stats["reliability"]
        entry["accuracy"] = stats["accuracy"]
        entry["spans_run_dirs"] = stats["spans_run_dirs"]
        entry["support_run_dirs"] = stats["support_run_dirs"]
        entry["contradicting_transition_ids"] = stats["contradicting_transition_ids"]
        entry["promotion_status"] = "stage_d_promoted"
        entry["truth_boundary"] = truth_boundary()
        promoted.append(entry)

    minimal, pruned = prune_redundant(promoted)
    residual = composed_residual(promoted, samples)
    promoted_effect_hist = Counter(str(rule.get("effect")) for rule in promoted)
    minimal_effect_hist = Counter(str(rule.get("effect")) for rule in minimal)

    corpus_manifest = json.loads(corpus_manifest_path.read_text(encoding="utf-8"))
    baseline_summary = corpus_manifest.get("baseline_mismatch_summary") if isinstance(corpus_manifest.get("baseline_mismatch_summary"), dict) else {}

    ledger = {
        "schema": MECHANIC_LEDGER_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "grammar_version": GRAMMAR_VERSION,
        "predictor_baseline_version": PREDICT_V0_VERSION,
        "rules_manifest": rel(rules_manifest_path),
        "rules_manifest_sha256": sha256_file(rules_manifest_path),
        "corpus_manifest": rel(corpus_manifest_path),
        "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
        "corpus_jsonl": rel(corpus_jsonl_path),
        "corpus_jsonl_sha256": sha256_file(corpus_jsonl_path),
        "discrepancy_ledger": discrepancy_ledger_ref,
        "candidate_rule_count": candidate_count,
        "promoted_rule_count": len(promoted),
        "minimal_promoted_rule_count": len(minimal),
        "promoted_effect_histogram": dict(sorted(promoted_effect_hist.items())),
        "minimal_effect_histogram": dict(sorted(minimal_effect_hist.items())),
        "promotion_gates": {
            "contradictions": 0,
            "support_min": PROMOTION_SUPPORT_MIN,
            "reliability_min": PROMOTION_RELIABILITY_MIN,
            "reliability_z": PROMOTION_Z,
            "spans_run_dirs_min": PROMOTION_SPANS_MIN,
        },
        "baseline_mismatch_summary": baseline_summary,
        "composed_model": residual,
        "redundancy": {
            "minimal_rule_ids": sorted(str(rule.get("rule_id")) for rule in minimal),
            "pruned_rule_ids": sorted(str(rule.get("rule_id")) for rule in pruned),
        },
        "promoted_rules": promoted,
        "source_hashes": source_hashes(
            [
                "codexlab/contra_head/crucible/grammar.py",
                "codexlab/contra_head/crucible/falsify.py",
                "codexlab/contra_head/crucible/model.py",
                "codexlab/crucible_induce_rules.py",
                "codexlab/crucible_falsify_rules.py",
                "codexlab/contra_verify_crucible.py",
            ]
        ),
        "truth_boundary": truth_boundary(),
    }
    ledger["ledger_sha256"] = stable_id(
        "crucible_mechanic_ledger",
        ledger["grammar_version"],
        ledger["corpus_jsonl_sha256"],
        sorted(str(rule.get("rule_id")) for rule in promoted),
        ledger["composed_model"]["residual_mismatch_rate"],
    )
    return ledger
