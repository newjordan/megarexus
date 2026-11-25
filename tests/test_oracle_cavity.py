"""Cavity fitting stays exact for a small key and finishes on a noisy board."""

from __future__ import annotations

import time

import numpy as np

from codexlab.contra_head.vision import analyze_scene_surface_hypotheses
from codexlab.oracle import _best_open_cavity_fit, actor_goal_relation_from_contra_scene


def test_single_pixel_key_fits_the_open_cell():
    fit = _best_open_cavity_fit(
        {
            "component_id": "lock",
            "raw_pixel_value": 8,
            "shape_mask_rows": ["11", "10"],
        },
        {
            "component_id": "key",
            "raw_pixel_value": 3,
            "shape_mask_rows": ["1"],
        },
    )
    assert fit["lock_key_shape_fit"] is True
    assert fit["offset_xy"] == [1, 1]
    assert fit["fit_score"] == 1.0
    assert fit["collision_ratio"] == 0.0


def test_noisy_board_actor_goal_relation_finishes():
    frame = np.random.default_rng(0).integers(0, 16, size=(64, 64))
    started = time.perf_counter()
    surfaces, _label_map, _lookup = analyze_scene_surface_hypotheses(frame)
    relation = actor_goal_relation_from_contra_scene(surfaces)
    elapsed = time.perf_counter() - started
    assert elapsed < 8.0
    assert relation is None or relation.get("admission_status") == "not_admitted"
