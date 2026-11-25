"""Pairwise symmetry stays exact on small boards and finishes on noise."""

from __future__ import annotations

import time

import numpy as np

from codexlab.contra_head.symmetry import (
    _PAIRWISE_COMPONENT_LIMIT,
    _component_pair_symmetry,
    _pairwise_components,
    build_symmetry_hypotheses,
)
from codexlab.contra_head.vision import analyze_scene_surface_hypotheses


def _component(component_id: str, area: int, center: tuple[float, float], color: int = 1) -> dict:
    x, y = center
    return {
        "component_id": component_id,
        "area_pixels": area,
        "center_xy": [x, y],
        "bbox_xyxy": [int(x) - 1, int(y) - 1, int(x) + 1, int(y) + 1],
        "raw_pixel_value": color,
        "shape_mask_rows": ["111", "111", "111"],
    }


def test_pairwise_cap_keeps_every_component_until_the_limit():
    rows = [_component(f"c{i}", i + 1, (float(i), 1.0)) for i in range(4)]
    assert _pairwise_components(rows) == rows


def test_pairwise_cap_keeps_the_largest_components():
    rows = [_component(f"c{i}", i + 1, (float(i), 1.0)) for i in range(_PAIRWISE_COMPONENT_LIMIT + 5)]
    kept = _pairwise_components(rows)
    assert len(kept) == _PAIRWISE_COMPONENT_LIMIT
    assert kept[0]["component_id"] == rows[-1]["component_id"]
    assert all(int(row["area_pixels"]) > 5 for row in kept)


def test_mirrored_pair_is_still_found():
    left = _component("left", 12, (10.0, 20.0), color=4)
    right = _component("right", 12, (53.0, 20.0), color=4)
    found = _component_pair_symmetry(
        left,
        right,
        axis="vertical",
        frame_shape=(64, 64),
    )
    assert found is not None
    assert found["member_component_ids"] == ["left", "right"]
    assert found["admission_status"] == "not_admitted"


def test_noisy_board_symmetry_finishes():
    frame = np.random.default_rng(0).integers(0, 16, size=(64, 64))
    started = time.perf_counter()
    surfaces, label_map, lookup = analyze_scene_surface_hypotheses(frame)
    packet = build_symmetry_hypotheses(
        frame=frame,
        scene_surfaces=surfaces,
        label_map=label_map,
        component_lookup=lookup,
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 8.0
    assert packet["admission_status"] == "not_admitted"
    assert "symmetry_relationship_group_count" in packet
