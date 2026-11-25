"""Spout placement stays inside the lattice."""

from __future__ import annotations

import numpy as np
import torch

from synthesis_lab.orchestration.sim3d.scene import extract_scene


def test_bottom_reservoir_does_not_index_past_the_lattice():
    torch.manual_seed(0)
    frame = torch.randint(0, 16, (64, 64)).numpy()
    scene = extract_scene(np.asarray(frame, dtype=np.int8))
    height, width = scene.solid.shape
    for y, x in scene.spouts:
        assert 0 <= y < height
        assert 0 <= x < width
