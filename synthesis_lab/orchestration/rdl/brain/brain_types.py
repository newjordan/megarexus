"""Shared MEGAMIND types — map views, operator results, brain state.

Coordinates WP-A (physics), WP-B (map/conceptual), WP-C (hemispheres).
All truth admission flags default False (fail-closed / hypothesis-only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_DROP = object()


def _json_value(value: Any) -> Any:
    """Copy a snapshot value into something ``json.dumps`` can emit.

    Private keys (prefix ``_``) and tensors stay out of the snapshot. The
    facade stashes dream tensors under those keys for ``judge_plan``.
    """
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    shape = getattr(value, "shape", None)
    if hasattr(value, "item") and shape is not None and len(shape) == 0:
        return _json_value(value.item())
    if shape is not None:
        return _DROP
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).startswith("_"):
                continue
            packed = _json_value(item)
            if packed is _DROP:
                continue
            out[str(key)] = packed
        return out
    if isinstance(value, (list, tuple)):
        packed_items = []
        for item in value:
            packed = _json_value(item)
            if packed is _DROP:
                continue
            packed_items.append(packed)
        return packed_items
    return _DROP


@dataclass
class TruthFlags:
    """Control / goal / action / solve admission — always False at construction."""

    control: bool = False
    goal: bool = False
    action: bool = False
    solve: bool = False
    # aliases used by some call sites
    control_admitted: bool = False
    goal_admitted: bool = False
    action_admitted: bool = False
    solve_admitted: bool = False

    def __post_init__(self) -> None:
        # Hard fail-closed: never allow True via construction defaults misuse.
        self.control = False
        self.goal = False
        self.action = False
        self.solve = False
        self.control_admitted = False
        self.goal_admitted = False
        self.action_admitted = False
        self.solve_admitted = False

    def any_true(self) -> bool:
        # Construction always False; kept for API completeness.
        return False

    def as_dict(self) -> dict[str, bool]:
        return {
            "control": False,
            "goal": False,
            "action": False,
            "solve": False,
            "control_admitted": False,
            "goal_admitted": False,
            "action_admitted": False,
            "solve_admitted": False,
        }

    @property
    def any_admitted(self) -> bool:
        return False


@dataclass
class OperatorResult:
    """Structured result from a physics / map / conceptual operator.

    status: "ok" | "degraded" | "failed"
    scores: JSON-serializable scalar metrics for fusion / dream_summary
    payload: richer structure (WP-A body/ray)
    proposals / summary: conceptual / field packets (WP-B)
    vectors / meta: hemisphere pack slots (WP-C)
    """

    name: str
    status: str = "ok"
    scores: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    messages: list[str] = field(default_factory=list)
    proposals: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    vectors: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    hypothesis: bool = True
    hypothesis_only: bool = True
    truth_flags: TruthFlags = field(default_factory=TruthFlags)
    admitted: bool = False

    def __post_init__(self) -> None:
        self.hypothesis = True
        self.hypothesis_only = True
        self.admitted = False
        if self.summary and not self.meta:
            # convenience: summary feeds meta for pack_operator_feats consumers
            self.meta = {k: v for k, v in self.summary.items() if not isinstance(v, (list, dict))}
        if self.payload and not self.vectors:
            # optional vector extracts from common keys
            for key in ("delta_xy", "suggested_move_delta", "origin_xy"):
                src = self.payload if key in self.payload else self.summary
                if key in src and isinstance(src[key], (list, tuple)):
                    self.vectors[key] = list(src[key])

    def scalar_list(self, n: int = 16) -> list[float]:
        """Flatten scores (+ short vectors) into a fixed-length float list."""
        slot: list[float] = []
        for v in self.scores.values():
            if isinstance(v, bool):
                slot.append(1.0 if v else 0.0)
            elif isinstance(v, (int, float)):
                slot.append(float(v))
            elif isinstance(v, (list, tuple)) and v and isinstance(v[0], (int, float)):
                slot.extend(float(x) for x in v[:4])
        for v in self.vectors.values():
            if isinstance(v, (list, tuple)):
                slot.extend(float(x) for x in v[:8] if isinstance(x, (int, float)))
            elif isinstance(v, (int, float)):
                slot.append(float(v))
        if len(slot) < n:
            slot = slot + [0.0] * (n - len(slot))
        return slot[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "scores": dict(self.scores),
            "payload": dict(self.payload),
            "messages": list(self.messages),
            "proposals": list(self.proposals),
            "summary": dict(self.summary),
            "vectors": dict(self.vectors),
            "meta": dict(self.meta),
            "hypothesis": True,
            "hypothesis_only": True,
            "admitted": False,
            "truth_flags": self.truth_flags.as_dict(),
        }


@dataclass
class KnownMapView:
    """Frame-derived map for operators.

    WP-A fields: solid / spouts / mouths / floor_rows / frame
    WP-B fields: components / background / gate_failures / trusted
    Trusted only if perception gate passed (fail closed).
    """

    # physics lattice (WP-A)
    solid: Any = None  # np.ndarray [H,W] bool
    spouts: tuple = ()
    mouths: tuple = ()
    floor_rows: tuple = ()
    glyph_cells: tuple = ()
    capacities: tuple = ()
    emit_volume: float = 0.0
    frame: Any = None
    trusted: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    # mapping (WP-B)
    gate_failures: list = field(default_factory=list)
    components: list = field(default_factory=list)
    background: int = 0
    status: str = "degraded"  # "ok" | "degraded"
    provenance: dict[str, Any] = field(default_factory=dict)
    scene_summary: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, Any] = field(default_factory=dict)
    solid_mask: Any = None  # alias used by some hemispheres

    def __post_init__(self) -> None:
        if self.solid_mask is None and self.solid is not None:
            self.solid_mask = self.solid
        elif self.solid is None and self.solid_mask is not None:
            self.solid = self.solid_mask
        if not self.provenance and self.meta:
            self.provenance = dict(self.meta)
        if not self.meta and self.provenance:
            self.meta = dict(self.provenance)
        # fail closed unless explicitly trusted
        if not self.trusted:
            self.trusted = False
            if self.status == "ok" and self.gate_failures:
                self.status = "degraded"

    def get_solid(self) -> Any:
        """Prefer lattice solid; fall back to solid_mask."""
        if self.solid is not None:
            return self.solid
        return self.solid_mask

    def to_dict(self) -> dict[str, Any]:
        shape = None
        if self.frame is not None:
            try:
                shape = list(self.frame.shape)
            except Exception:
                shape = None
        return {
            "trusted": bool(self.trusted),
            "gate_failures": list(self.gate_failures),
            "n_components": len(self.components),
            "background": int(self.background),
            "status": self.status,
            "provenance": dict(self.provenance),
            "scene_summary": dict(self.scene_summary),
            "frame_shape": shape,
            "n_spouts": len(self.spouts or ()),
            "n_mouths": len(self.mouths or ()),
            "emit_volume": float(self.emit_volume or 0.0),
        }


@dataclass
class BrainState:
    """Facade snapshot — fields match MEGAMIND_BUILD_SPEC."""

    frame: Any = None
    trusted: bool = False
    gate_failures: list = field(default_factory=list)
    h_I: Any = None
    h_E: Any = None
    dream_summary: dict = field(default_factory=dict)
    world_summary: dict = field(default_factory=dict)
    disagreement: float = 0.0
    judge_I: float = 0.0
    action_logits: Any = None
    propose_logits: Any = None
    trust: Any = None  # [α_I, α_E]
    mode_logits: Any = None
    truth_flags: Any = field(default_factory=TruthFlags)
    extra_feats: Any = None
    status: str = "ok"
    map_view: KnownMapView | None = None

    def __post_init__(self) -> None:
        # force fail-closed truth flags (accept TruthFlags or dict)
        self.truth_flags = TruthFlags()

    def truth_flags_dict(self) -> dict[str, bool]:
        tf = self.truth_flags
        if isinstance(tf, TruthFlags):
            return {
                "control": False,
                "goal": False,
                "action": False,
                "solve": False,
            }
        if isinstance(tf, dict):
            return {
                "control": bool(tf.get("control", False)),
                "goal": bool(tf.get("goal", False)),
                "action": bool(tf.get("action", False)),
                "solve": bool(tf.get("solve", False)),
            }
        return {"control": False, "goal": False, "action": False, "solve": False}

    def all_truth_flags_false(self) -> bool:
        return not any(self.truth_flags_dict().values())

    def to_dict(self) -> dict[str, Any]:
        disagreement = float(self.disagreement)
        judge = float(self.judge_I)
        return {
            "trusted": bool(self.trusted),
            "gate_failures": _json_value(list(self.gate_failures)),
            "dream_summary": _json_value(self.dream_summary) or {},
            "world_summary": _json_value(self.world_summary) or {},
            "disagreement": disagreement if math.isfinite(disagreement) else None,
            "judge_I": judge if math.isfinite(judge) else None,
            "truth_flags": self.truth_flags_dict(),
            "status": self.status,
            "map": None if self.map_view is None else self.map_view.to_dict(),
        }
