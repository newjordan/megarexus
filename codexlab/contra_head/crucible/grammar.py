"""Stage C rule grammar for Crucible mechanic induction.

This module defines `crucible_grammar.v0`: a small, machine-checkable grammar of
local transition rules and the enumeration that turns discrepancy anomalies into
candidate rules. The grammar predicts ACTOR DELTA ONLY; paint/flow grid effects
are out of scope.

Authority boundary: every rule produced here is a falsifiable hypothesis, not
control truth, goal truth, or solve truth. Rules and their downstream ledgers
carry all-false truth flags and the boundary literal
`mechanic_forward_model_is_hypothesis_ledger_composition_not_control_truth`.

Coordinate convention (shared with canon.py / forward_v0.py): canonical frame is
display orientation, coordinates are `[x, y]`, and the move deltas are
`move_up=(0,-1)`, `move_down=(0,1)`, `move_left=(-1,0)`, `move_right=(1,0)`.
Direction-relative offsets are resolved against the issued move's delta.
"""

from __future__ import annotations

import itertools
from typing import Any, Iterable

from .corpus import stable_id, stable_json
from .forward_v0 import MOVE_DELTAS


GRAMMAR_VERSION = "crucible_grammar.v0"
RULE_SCHEMA = "codexlab.crucible_rule.v1"
RULES_MANIFEST_SCHEMA = "codexlab.crucible_rule_corpus.v1"
GRAMMAR_BOUNDARY_LITERAL = "mechanic_forward_model_is_hypothesis_ledger_composition_not_control_truth"

# Palette per the pinned contract plus the out-of-bounds sentinel.
PALETTE: tuple[int, ...] = (1, 4, 6, 8, 9, 11, 12, 14)
OOB_TOKEN = "OOB"
OOB_CELL = -1
COLOR_DOMAIN: tuple[Any, ...] = (*PALETTE, OOB_TOKEN)

OFFSETS: tuple[str, ...] = (
    "ahead1",
    "ahead2",
    "left1",
    "right1",
    "behind1",
    "ahead_left1",
    "ahead_right1",
)
EFFECTS: tuple[str, ...] = (
    "blocked",
    "forward",
    "redirect_left",
    "redirect_right",
    "reverse",
    "slide_until_blocked",
)
AXES: tuple[str, ...] = ("h", "v")
MOVE_PRIMITIVES: tuple[str, ...] = tuple(sorted(MOVE_DELTAS))
SELECTOR_KINDS: tuple[str, ...] = ("per_primitive", "direction_generic")

# Cap on how many sample anomaly ids each rule records under induced_from.
INDUCED_FROM_SAMPLE_CAP = 16


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
        "boundary": GRAMMAR_BOUNDARY_LITERAL,
    }


# ---------------------------------------------------------------------------
# Direction algebra (mirrors codexlab/contra_head/crucible/ledger.py helpers)
# ---------------------------------------------------------------------------

def left_of(delta: tuple[int, int]) -> tuple[int, int]:
    dx, dy = delta
    return dy, -dx


def right_of(delta: tuple[int, int]) -> tuple[int, int]:
    dx, dy = delta
    return -dy, dx


def reverse_of(delta: tuple[int, int]) -> tuple[int, int]:
    dx, dy = delta
    return -dx, -dy


def offset_delta(offset: str, direction: tuple[int, int]) -> tuple[int, int]:
    """Resolve a direction-relative offset name into an absolute cell delta."""
    dx, dy = direction
    lx, ly = left_of(direction)
    rx, ry = right_of(direction)
    table = {
        "ahead1": (dx, dy),
        "ahead2": (dx * 2, dy * 2),
        "left1": (lx, ly),
        "right1": (rx, ry),
        "behind1": (-dx, -dy),
        "ahead_left1": (dx + lx, dy + ly),
        "ahead_right1": (dx + rx, dy + ry),
    }
    return table[offset]


# ---------------------------------------------------------------------------
# Atom / condition / selector helpers
# ---------------------------------------------------------------------------

def cell_color_atom(offset: str, color: Any) -> dict[str, Any]:
    return {"atom": "cell_color", "offset": offset, "color": color}


def actor_axis_atom(axis: str) -> dict[str, Any]:
    return {"atom": "actor_axis", "axis": axis}


def atom_is_valid(atom: Any) -> bool:
    if not isinstance(atom, dict):
        return False
    kind = atom.get("atom")
    if kind == "cell_color":
        return atom.get("offset") in OFFSETS and atom.get("color") in COLOR_DOMAIN
    if kind == "actor_axis":
        return atom.get("axis") in AXES
    return False


def canonical_condition(atoms: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic ordering of atoms for ids and equality."""
    return sorted((dict(atom) for atom in atoms), key=stable_json)


def canonical_selector(selector: dict[str, Any]) -> list[Any]:
    kind = selector.get("kind")
    if kind == "per_primitive":
        return ["per_primitive", selector.get("primitive")]
    return ["direction_generic"]


def selector_is_valid(selector: Any) -> bool:
    if not isinstance(selector, dict):
        return False
    kind = selector.get("kind")
    if kind == "per_primitive":
        return selector.get("primitive") in MOVE_PRIMITIVES
    return kind == "direction_generic"


def selector_applies(selector: dict[str, Any], primitive: str) -> bool:
    if primitive not in MOVE_DELTAS:
        return False
    if selector.get("kind") == "per_primitive":
        return selector.get("primitive") == primitive
    return selector.get("kind") == "direction_generic"


def selector_at_least_as_general(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True if selector ``a`` covers every primitive selector ``b`` covers."""
    if a.get("kind") == "direction_generic":
        return True
    return (
        a.get("kind") == "per_primitive"
        and b.get("kind") == "per_primitive"
        and a.get("primitive") == b.get("primitive")
    )


def rule_id(selector: dict[str, Any], atoms: Iterable[dict[str, Any]], effect: str) -> str:
    return stable_id(
        "crucible_rule",
        canonical_selector(selector),
        canonical_condition(atoms),
        effect,
    )


# ---------------------------------------------------------------------------
# Evaluation against a 5x5 canonical neighborhood (OOB cells are -1)
# ---------------------------------------------------------------------------

def neighborhood_cell(neighborhood: list[list[int]], delta: tuple[int, int]) -> int:
    dr, dc = delta
    if -2 <= dr <= 2 and -2 <= dc <= 2:
        return int(neighborhood[dr + 2][dc + 2])
    return OOB_CELL


def eval_atom(
    atom: dict[str, Any],
    neighborhood: list[list[int]],
    actor_axis: Any,
    direction: tuple[int, int],
) -> bool:
    kind = atom.get("atom")
    if kind == "actor_axis":
        return actor_axis == atom.get("axis")
    if kind == "cell_color":
        value = neighborhood_cell(neighborhood, offset_delta(str(atom.get("offset")), direction))
        color = atom.get("color")
        if color == OOB_TOKEN:
            return value == OOB_CELL
        return value == int(color)
    return False


def eval_condition(
    atoms: Iterable[dict[str, Any]],
    neighborhood: list[list[int]],
    actor_axis: Any,
    direction: tuple[int, int],
) -> bool:
    return all(eval_atom(atom, neighborhood, actor_axis, direction) for atom in atoms)


def effect_delta(
    effect: str,
    direction: tuple[int, int],
    neighborhood: list[list[int]] | None = None,
) -> tuple[int, int]:
    """Map an effect + issued direction to a predicted ACTOR delta."""
    if effect == "blocked":
        return (0, 0)
    if effect == "forward":
        return direction
    if effect == "redirect_left":
        return left_of(direction)
    if effect == "redirect_right":
        return right_of(direction)
    if effect == "reverse":
        return reverse_of(direction)
    if effect == "slide_until_blocked":
        return _slide_delta(direction, neighborhood)
    return (0, 0)


def _slide_delta(direction: tuple[int, int], neighborhood: list[list[int]] | None) -> tuple[int, int]:
    """Deterministic corridor slide bounded by the 5x5 neighborhood radius.

    The actor advances along ``direction`` while the next cell shares the color
    of the first forward cell. Bounded by radius 2 (the available context).
    This is a hypothesis forward model; it has no positive support in the v0
    corpus and is never expected to promote.
    """
    if neighborhood is None:
        return direction
    first = neighborhood_cell(neighborhood, offset_delta("ahead1", direction))
    if first == OOB_CELL:
        return (0, 0)
    steps = 1
    while steps < 2:
        nxt = neighborhood_cell(neighborhood, (direction[0] * (steps + 1), direction[1] * (steps + 1)))
        if nxt == first and nxt != OOB_CELL:
            steps += 1
        else:
            break
    return (direction[0] * steps, direction[1] * steps)


ANOMALY_KIND_TO_EFFECT: dict[str, str] = {
    "blocked": "blocked",
    "redirect_left": "redirect_left",
    "redirect_right": "redirect_right",
    "reverse": "reverse",
    "slide": "slide_until_blocked",
}


# ---------------------------------------------------------------------------
# Rule construction
# ---------------------------------------------------------------------------

def make_rule(
    selector: dict[str, Any],
    atoms: list[dict[str, Any]],
    effect: str,
    induced_from: list[str] | None = None,
) -> dict[str, Any]:
    canonical = canonical_condition(atoms)
    return {
        "schema": RULE_SCHEMA,
        "grammar_version": GRAMMAR_VERSION,
        "rule_id": rule_id(selector, canonical, effect),
        "selector": dict(selector),
        "condition": {"atoms": canonical},
        "effect": effect,
        "arity": len(canonical),
        "induced_from": list(induced_from or []),
        "truth_boundary": truth_boundary(),
    }


def all_single_atoms() -> list[dict[str, Any]]:
    atoms: list[dict[str, Any]] = []
    for offset in OFFSETS:
        for color in COLOR_DOMAIN:
            atoms.append(cell_color_atom(offset, color))
    for axis in AXES:
        atoms.append(actor_axis_atom(axis))
    return atoms


def all_selectors() -> list[dict[str, Any]]:
    selectors = [{"kind": "per_primitive", "primitive": primitive} for primitive in MOVE_PRIMITIVES]
    selectors.append({"kind": "direction_generic"})
    return selectors


def context_atoms_for_anomaly(local_context: dict[str, Any], direction: tuple[int, int]) -> list[dict[str, Any]]:
    """Representable atoms that hold in this anomaly's local context."""
    neighborhood = local_context.get("canon_neighborhood_5x5")
    axis = local_context.get("actor_axis")
    atoms: list[dict[str, Any]] = []
    if isinstance(neighborhood, list) and len(neighborhood) == 5:
        for offset in OFFSETS:
            value = neighborhood_cell(neighborhood, offset_delta(offset, direction))
            if value == OOB_CELL:
                atoms.append(cell_color_atom(offset, OOB_TOKEN))
            elif value in PALETTE:
                atoms.append(cell_color_atom(offset, int(value)))
    if axis in AXES:
        atoms.append(actor_axis_atom(axis))
    return atoms


def enumerate_rules(anomalies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exhaustive single-atom rules plus data-driven two-atom conjunctions.

    Single-atom rules are emitted for every (selector, atom, effect). Two-atom
    conjunctions are instantiated ONLY from anomaly contexts that explain them.
    induced_from is populated only with anomalies whose context satisfies the
    condition and whose observed delta equals the effect's predicted delta.
    """
    selectors = all_selectors()
    rules: dict[str, dict[str, Any]] = {}
    induced: dict[str, list[str]] = {}

    # 1) Exhaustive single-atom rules.
    for selector in selectors:
        for atom in all_single_atoms():
            for effect in EFFECTS:
                rule = make_rule(selector, [atom], effect)
                rules[rule["rule_id"]] = rule
                induced.setdefault(rule["rule_id"], [])

    # 2) Walk anomalies; record induced_from on matching single-atom rules and
    #    instantiate data-driven two-atom conjunctions.
    for anomaly in anomalies:
        primitive = str(anomaly.get("primitive") or "")
        if primitive not in MOVE_DELTAS:
            continue
        effect = ANOMALY_KIND_TO_EFFECT.get(str(anomaly.get("anomaly_kind")))
        if effect is None:
            continue
        direction = MOVE_DELTAS[primitive]
        local_context = anomaly.get("local_context") if isinstance(anomaly.get("local_context"), dict) else {}
        observed = _tuple(anomaly.get("observed", {}).get("canon_delta") if isinstance(anomaly.get("observed"), dict) else None)
        if observed is None:
            continue
        neighborhood = local_context.get("canon_neighborhood_5x5")
        axis = local_context.get("actor_axis")
        predicted = effect_delta(effect, direction, neighborhood if isinstance(neighborhood, list) else None)
        if predicted != observed:
            # Anomaly does not actually exemplify this effect; skip linking.
            continue
        anomaly_id = str(anomaly.get("anomaly_id") or "")
        context_atoms = context_atoms_for_anomaly(local_context, direction)
        applicable_selectors = [{"kind": "per_primitive", "primitive": primitive}, {"kind": "direction_generic"}]

        # Single-atom links (rules already exist from the exhaustive pass).
        for atom in context_atoms:
            for selector in applicable_selectors:
                rid = rule_id(selector, [atom], effect)
                if rid in rules:
                    _append_induced(induced, rid, anomaly_id)

        # Two-atom conjunctions, instantiated from this explaining context.
        for atom_a, atom_b in itertools.combinations(context_atoms, 2):
            for selector in applicable_selectors:
                rule = make_rule(selector, [atom_a, atom_b], effect)
                rid = rule["rule_id"]
                if rid not in rules:
                    rules[rid] = rule
                    induced.setdefault(rid, [])
                _append_induced(induced, rid, anomaly_id)

    out: list[dict[str, Any]] = []
    for rid, rule in sorted(rules.items()):
        rule["induced_from"] = sorted(set(induced.get(rid, [])))[:INDUCED_FROM_SAMPLE_CAP]
        out.append(rule)
    return out


def _append_induced(induced: dict[str, list[str]], rid: str, anomaly_id: str) -> None:
    bucket = induced.setdefault(rid, [])
    if anomaly_id and anomaly_id not in bucket:
        bucket.append(anomaly_id)


def _tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    a, b = value
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return None
    return int(a), int(b)
