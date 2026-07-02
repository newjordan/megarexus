"""Naive movement baseline for Crucible discrepancy mining."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


PREDICT_V0_VERSION = "naive_translation_v0"

MOVE_DELTAS: dict[str, tuple[int, int]] = {
    # Mirrors the current Stage A corpus convention: SP80 selected_coord is
    # emitted as [x, y] and is indexable into saved grids in that order.
    "move_up": (0, -1),
    "move_down": (0, 1),
    "move_left": (-1, 0),
    "move_right": (1, 0),
}

STATIC_DELTAS: dict[str, tuple[int, int]] = {
    "select_object": (0, 0),
    "activate_flow": (0, 0),
}


@dataclass(frozen=True)
class V0Prediction:
    model_version: str
    primitive: str
    predicted_canon_delta: tuple[int, int] | None
    predicted_blocked: bool
    effect_kind: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.predicted_canon_delta is not None:
            payload["predicted_canon_delta"] = list(self.predicted_canon_delta)
        return payload


def predict_v0(primitive: str) -> V0Prediction:
    """Predict actor delta using only the issued primitive."""
    if primitive in MOVE_DELTAS:
        return V0Prediction(
            model_version=PREDICT_V0_VERSION,
            primitive=primitive,
            predicted_canon_delta=MOVE_DELTAS[primitive],
            predicted_blocked=False,
            effect_kind="forward",
        )
    if primitive in STATIC_DELTAS:
        return V0Prediction(
            model_version=PREDICT_V0_VERSION,
            primitive=primitive,
            predicted_canon_delta=STATIC_DELTAS[primitive],
            predicted_blocked=False,
            effect_kind="static",
        )
    return V0Prediction(
        model_version=PREDICT_V0_VERSION,
        primitive=primitive,
        predicted_canon_delta=None,
        predicted_blocked=False,
        effect_kind="unknown_primitive",
    )

