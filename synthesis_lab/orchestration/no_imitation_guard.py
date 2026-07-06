"""Fail-closed checks for no-imitation CNN training.

The project goal is to train a CNN to learn game mechanics from interaction.
Any human/demo/route/replay/eval-answer lineage in a training input destroys
that claim, so active trainers and dataset builders call this before work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


BANNED_LINEAGE_MARKERS = (
    "human",
    "demo",
    "route",
    "replay",
    "answer",
    "graphdemo",
    "graph_demo_plan",
    "arc_live",
    "MEGAREXUS_GRAPH_DEMO_PLAN",
    "MEGAREXUS_ROUTE_",
)

KNOWN_CONTAMINATED_MARKERS = (
    "stage25_goal_label_dataset_20260704",
    "outcome_v14_goal_head_v12parent_20260704",
    "stage33_paint_output_label_dataset_20260704",
    "outcome_v15_paint_output_goal_head_v12parent_20260704",
)

# Reward-hack artifacts: hardcoded win conditions / privileged engine-state
# solvers. Quarantined 2026-07-04; never a policy, planner, or label source.
REWARD_HACK_MARKERS = (
    "sp80_spill_model",
    "spill_planner",
    "spillplannerbrain",
    "sp80_frame_spill_perception",
)


@dataclass(frozen=True)
class LineageViolation:
    context: str
    marker: str
    value: str


def _iter_strings(value: Any, context: str) -> Iterable[tuple[str, str]]:
    if value is None:
        return
    if isinstance(value, Path):
        yield context, str(value)
        return
    if isinstance(value, str):
        yield context, value
        return
    if isinstance(value, dict):
        for key, child in value.items():
            child_context = f"{context}.{key}" if context else str(key)
            if isinstance(key, str):
                yield child_context, key
            yield from _iter_strings(child, child_context)
        return
    if isinstance(value, (list, tuple, set)):
        for idx, child in enumerate(value):
            yield from _iter_strings(child, f"{context}[{idx}]")


def find_banned_lineage(*values: Any, context: str = "input") -> list[LineageViolation]:
    markers = BANNED_LINEAGE_MARKERS + KNOWN_CONTAMINATED_MARKERS + REWARD_HACK_MARKERS
    violations: list[LineageViolation] = []
    for idx, value in enumerate(values):
        root = f"{context}[{idx}]" if len(values) > 1 else context
        for item_context, text in _iter_strings(value, root):
            lowered = text.lower()
            for marker in markers:
                if marker.lower() in lowered:
                    violations.append(LineageViolation(item_context, marker, text))
                    break
    return violations


def format_violations(violations: Iterable[LineageViolation]) -> str:
    lines = []
    for row in violations:
        lines.append(f"{row.context}: marker={row.marker!r} value={row.value!r}")
    return "\n".join(lines)


def assert_no_imitation_lineage(*values: Any, purpose: str = "CNN training") -> None:
    violations = find_banned_lineage(*values, context=purpose)
    if violations:
        raise ValueError(
            f"no-imitation guard refused {purpose}; banned answer/imitation "
            "lineage found:\n" + format_violations(violations)
        )
