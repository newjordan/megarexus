"""Public Crucible forward-model interface."""

from __future__ import annotations

import json
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .canon import canonicalize_coord, canonicalize_grid
from .corpus import resolve_repo_path, stable_id
from .forward_v0 import MOVE_DELTAS
from .grammar import EFFECTS, effect_delta, eval_condition, selector_applies
from .ledger import _component_stats, build_local_context


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEDGER_PATH = ROOT / "codexlab" / "datasets" / "sp80_crucible_mechanics_v0" / "mechanic_ledger.json"
DEFAULT_DISCREPANCY_LEDGER_PATH = ROOT / "codexlab" / "datasets" / "sp80_crucible_discrepancy_v0" / "ledger.json"
BOUNDARY_LITERAL = "mechanic_forward_model_is_hypothesis_ledger_composition_not_control_truth"
MIN_RULE_SUPPORT = 8


@dataclass(frozen=True)
class CruciblePrediction:
    predicted_canon_delta: tuple[int, int] | None
    predicted_blocked: bool
    effect_kind: str
    matched_rule_id: str | None
    confidence: float
    opinion: Literal["prediction", "no_opinion"]
    control_truth_admitted: bool = False
    goal_truth_admitted: bool = False
    solve_truth_admitted: bool = False
    action_choice_admitted: bool = False
    boundary: str = BOUNDARY_LITERAL

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.predicted_canon_delta is not None:
            payload["predicted_canon_delta"] = list(self.predicted_canon_delta)
        return payload


class CrucibleModel:
    def __init__(self, ledger: dict[str, Any], ledger_path: Path) -> None:
        self.ledger = ledger
        self.ledger_path = ledger_path
        self.ledger_sha256 = ledger.get("ledger_sha256")
        self.rules = self._rules_from_ledger(ledger)

    @classmethod
    def load(cls, ledger_path: str | Path = DEFAULT_LEDGER_PATH) -> "CrucibleModel | None":
        path = Path(ledger_path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            return None
        try:
            ledger = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(ledger, dict):
            return None
        return cls(ledger, path)

    @staticmethod
    def _tuple_delta(value: Any) -> tuple[int, int] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        a, b = value
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            return None
        return int(a), int(b)

    @classmethod
    def _rules_from_ledger(cls, ledger: dict[str, Any]) -> list[dict[str, Any]]:
        schema = ledger.get("schema")
        if schema == "codexlab.crucible_discrepancy_ledger.v1":
            return cls._rules_from_discrepancy_ledger(ledger)
        rules = ledger.get("promoted_rules")
        return list(rules) if isinstance(rules, list) else []

    @classmethod
    def _rules_from_discrepancy_ledger(cls, ledger: dict[str, Any]) -> list[dict[str, Any]]:
        """Build PRE-GRAMMAR BRIDGE rules from discrepancy-derived exact scans.

        These rules are not Stage C/D-falsified and are not the promoted
        mechanic ledger. They are an offline bridge for existing consumers that
        already load the discrepancy ledger fail-open.
        """
        corpus_path = resolve_repo_path(str(ledger.get("corpus_jsonl") or ""))
        if not corpus_path.is_file():
            return []
        support: dict[tuple[str, int], int] = {}
        contradictions: dict[tuple[str, int], int] = {}
        local_delta_counts: dict[tuple[str, tuple[tuple[int, ...], ...]], dict[tuple[int, int], int]] = {}
        with corpus_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    continue
                primitive = str(((row.get("action") or {}).get("primitive") or ""))
                actor = row.get("actor") if isinstance(row.get("actor"), dict) else {}
                if primitive not in MOVE_DELTAS or actor.get("selection_changed") is True:
                    continue
                try:
                    context = build_local_context(row, _PredictionShim(MOVE_DELTAS[primitive]))
                except Exception:
                    continue
                observed = cls._tuple_delta((row.get("observed") or {}).get("canon_delta"))
                if observed is None:
                    continue
                neighborhood = context.get("canon_neighborhood_5x5")
                if isinstance(neighborhood, list) and len(neighborhood) == 5:
                    try:
                        neighborhood_key = tuple(tuple(int(cell) for cell in line) for line in neighborhood)
                    except (TypeError, ValueError):
                        neighborhood_key = ()
                    if len(neighborhood_key) == 5 and all(len(line) == 5 for line in neighborhood_key):
                        key = (primitive, neighborhood_key)
                        bucket = local_delta_counts.setdefault(key, {})
                        bucket[observed] = bucket.get(observed, 0) + 1
                ray = context.get("forward_ray_3")
                if not isinstance(ray, list) or not ray:
                    continue
                key = (primitive, int(ray[0]))
                if observed == (0, 0):
                    support[key] = support.get(key, 0) + 1
                else:
                    contradictions[key] = contradictions.get(key, 0) + 1
        rules: list[dict[str, Any]] = []
        for (primitive, neighborhood), deltas in sorted(local_delta_counts.items()):
            total = sum(deltas.values())
            if total < MIN_RULE_SUPPORT or len(deltas) != 1:
                continue
            predicted_delta = next(iter(deltas))
            effect_kind = "blocked" if predicted_delta == (0, 0) else "local_delta"
            rule_digest = hashlib.sha256(
                json.dumps([primitive, neighborhood], sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()[:12]
            rules.append(
                {
                    "rule_id": f"crucible_discrepancy_local_delta.{primitive}.{total}.{rule_digest}.v0",
                    "primitive": primitive,
                    "condition": {"canon_neighborhood_5x5": [list(line) for line in neighborhood]},
                    "effect_kind": effect_kind,
                    "predicted_canon_delta": [predicted_delta[0], predicted_delta[1]],
                    "confidence": 1.0,
                    "support": total,
                    "contradictions": 0,
                    "promotion_status": "pre_grammar_bridge_not_stage_cd_promoted",
                    "source": "zero_contradiction_exact_local_context_scan",
                }
            )
        for (primitive, forward_color), count in sorted(support.items()):
            contradiction_count = contradictions.get((primitive, forward_color), 0)
            if count < MIN_RULE_SUPPORT or contradiction_count != 0:
                continue
            rules.append(
                {
                    "rule_id": f"crucible_discrepancy_blocked.{primitive}.forward_color_{forward_color}.v0",
                    "primitive": primitive,
                    "condition": {"forward_cell_color": forward_color},
                    "effect_kind": "blocked",
                    "predicted_canon_delta": [0, 0],
                    "confidence": 1.0,
                    "support": count,
                    "contradictions": contradiction_count,
                    "promotion_status": "pre_grammar_bridge_not_stage_cd_promoted",
                    "source": "zero_contradiction_discrepancy_scan",
                }
            )
        return rules

    def predict(
        self,
        before_grid: Any,
        action_primitive: str,
        actor_coord: Any,
        rotation_quarters: int = 0,
    ) -> CruciblePrediction:
        primitive = str(action_primitive or "")
        if primitive not in MOVE_DELTAS:
            return CruciblePrediction(
                predicted_canon_delta=None,
                predicted_blocked=False,
                effect_kind="no_rule",
                matched_rule_id=None,
                confidence=0.0,
                opinion="no_opinion",
            )
        try:
            grid = np.asarray(before_grid)
            canon_grid = canonicalize_grid(grid, int(rotation_quarters))
            canon_coord = canonicalize_coord(actor_coord, grid.shape, int(rotation_quarters))
        except Exception:
            canon_coord = None
            canon_grid = None
        if canon_coord is None or canon_grid is None:
            return CruciblePrediction(
                predicted_canon_delta=None,
                predicted_blocked=False,
                effect_kind="no_rule",
                matched_rule_id=None,
                confidence=0.0,
                opinion="no_opinion",
            )
        delta = MOVE_DELTAS[primitive]
        row = canon_coord[0] + delta[0]
        col = canon_coord[1] + delta[1]
        forward_color = -1
        if 0 <= row < canon_grid.shape[0] and 0 <= col < canon_grid.shape[1]:
            forward_color = int(canon_grid[row, col])
        neighborhood = _neighborhood_5x5_for_prediction(canon_grid, canon_coord)
        for rule in self.rules:
            condition = rule.get("condition") if isinstance(rule.get("condition"), dict) else {}
            if rule.get("primitive") != primitive:
                continue
            if "canon_neighborhood_5x5" in condition:
                if condition.get("canon_neighborhood_5x5") != neighborhood:
                    continue
                predicted_delta = self._tuple_delta(rule.get("predicted_canon_delta")) or (0, 0)
                return CruciblePrediction(
                    predicted_canon_delta=predicted_delta,
                    predicted_blocked=rule.get("effect_kind") == "blocked",
                    effect_kind=str(rule.get("effect_kind") or "local_delta"),
                    matched_rule_id=str(rule.get("rule_id") or ""),
                    confidence=float(rule.get("confidence") or 0.0),
                    opinion="prediction",
                )
        for rule in self.rules:
            condition = rule.get("condition") if isinstance(rule.get("condition"), dict) else {}
            if rule.get("primitive") != primitive:
                continue
            if "forward_cell_color" not in condition:
                continue
            if int(condition.get("forward_cell_color")) != int(forward_color):
                continue
            predicted_delta = self._tuple_delta(rule.get("predicted_canon_delta")) or (0, 0)
            return CruciblePrediction(
                predicted_canon_delta=predicted_delta,
                predicted_blocked=rule.get("effect_kind") == "blocked",
                effect_kind=str(rule.get("effect_kind") or "blocked"),
                matched_rule_id=str(rule.get("rule_id") or ""),
                confidence=float(rule.get("confidence") or 0.0),
                opinion="prediction",
            )
        return CruciblePrediction(
            predicted_canon_delta=None,
            predicted_blocked=False,
            effect_kind="no_rule",
            matched_rule_id=None,
            confidence=0.0,
            opinion="no_opinion",
        )

    def disagreements(
        self,
        before_grid: Any,
        actor_coord: Any,
        rotation_quarters: int,
        reachable_actions: list[str],
    ) -> list[dict[str, Any]]:
        """Per-action rule disagreements over promoted grammar rules.

        For each reachable move action, every promoted grammar rule whose
        condition fires contributes a predicted actor delta. When at least two
        firing rules predict DIFFERENT deltas, the action is a discrimination
        candidate. Fails open (empty list) when no grammar rules are loaded or
        the grid/coordinate cannot be canonicalized.

        This is hypothesis material only; rows carry the model boundary literal.
        """
        grammar_rules = [
            rule
            for rule in self.rules
            if isinstance(rule, dict)
            and isinstance(rule.get("selector"), dict)
            and isinstance(rule.get("condition"), dict)
            and isinstance(rule["condition"].get("atoms"), list)
            and rule.get("effect") in EFFECTS
        ]
        if not grammar_rules:
            return []
        try:
            grid = np.asarray(before_grid)
            canon_grid = canonicalize_grid(grid, int(rotation_quarters))
            canon_coord = canonicalize_coord(actor_coord, grid.shape, int(rotation_quarters))
        except Exception:
            return []
        if canon_coord is None or canon_grid is None:
            return []
        neighborhood = _neighborhood_5x5_for_prediction(canon_grid, canon_coord)
        try:
            actor_color = int(canon_grid[canon_coord[0], canon_coord[1]])
            actor_axis = _component_stats(canon_grid, canon_coord, actor_color).get("axis")
        except Exception:
            actor_axis = None

        rows: list[dict[str, Any]] = []
        for primitive in reachable_actions or []:
            if primitive not in MOVE_DELTAS:
                continue
            direction = MOVE_DELTAS[primitive]
            firing: list[tuple[str, list[int], float]] = []
            for rule in grammar_rules:
                selector = rule.get("selector") or {}
                if not selector_applies(selector, primitive):
                    continue
                atoms = rule["condition"]["atoms"]
                if not eval_condition(atoms, neighborhood, actor_axis, direction):
                    continue
                predicted = effect_delta(str(rule.get("effect")), direction, neighborhood)
                reliability = float(rule.get("reliability") or rule.get("confidence") or 0.0)
                firing.append((str(rule.get("rule_id") or ""), [predicted[0], predicted[1]], reliability))
            distinct = {tuple(pred) for _, pred, _ in firing}
            if len(firing) < 2 or len(distinct) < 2:
                continue
            weights: dict[tuple[int, int], float] = {}
            for _, pred, reliability in firing:
                key = (pred[0], pred[1])
                weights[key] = weights.get(key, 0.0) + max(reliability, 1e-9)
            total = sum(weights.values())
            entropy = 0.0
            if total > 0:
                for weight in weights.values():
                    prob = weight / total
                    if prob > 0:
                        entropy -= prob * math.log(prob)
            rule_ids = sorted(rid for rid, _, _ in firing)
            rule_predictions = {rid: pred for rid, pred, _ in firing}
            rows.append(
                {
                    "disagreement_id": stable_id(
                        "crucible_disagreement",
                        primitive,
                        rule_ids,
                        [[int(cell) for cell in line] for line in neighborhood],
                    ),
                    "action_primitive": primitive,
                    "disagreeing_rule_ids": rule_ids,
                    "rule_predictions": rule_predictions,
                    "posterior_entropy": round(entropy, 6),
                    "boundary": BOUNDARY_LITERAL,
                }
            )
        return rows


@dataclass(frozen=True)
class _PredictionShim:
    predicted_canon_delta: tuple[int, int]


def _neighborhood_5x5_for_prediction(grid: np.ndarray, coord: tuple[int, int]) -> list[list[int]]:
    rows: list[list[int]] = []
    for dr in range(-2, 3):
        out: list[int] = []
        for dc in range(-2, 3):
            row = coord[0] + dr
            col = coord[1] + dc
            if row < 0 or col < 0 or row >= grid.shape[0] or col >= grid.shape[1]:
                out.append(-1)
            else:
                out.append(int(grid[row, col]))
        rows.append(out)
    return rows
