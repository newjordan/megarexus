"""CONTRA_THINKING_ENGINE adapter: frame → perception feature planes.

Turns Contra's raw color/component packet (codexlab/contra_head) into a
fixed stack of ``[N_PLANES, 64, 64]`` float feature planes that the
MegarexusCNN consumes as extra input channels alongside the 16-channel
one-hot frame.

Contra remains an informer here: the planes are deterministic functions of
the frame produced by Contra's own perception code
(``build_raw_color_component_packet`` + ``raw_pixel_components_with_labels``).
No action choice, no truth admission — see codexlab's informer boundary.

Planes (all in [0, 1]):

0. background   — pixel belongs to Contra's dominant background color
1. objectness   — pixel belongs to a non-background component
2. area         — log-normalized area of the pixel's component
3. centroid_off — normalized distance from pixel to its component centroid
4. edge         — pixel has a 4-neighbor of a different color
5. color_freq   — global frequency of the pixel's color (pixel_count / 4096)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CODEXLAB = Path(__file__).resolve().parents[2] / "codexlab"
if str(_CODEXLAB) not in sys.path:
    sys.path.insert(0, str(_CODEXLAB))

from contra_head.controller_head import build_raw_color_component_packet  # noqa: E402
from contra_head.vision import raw_pixel_components_with_labels  # noqa: E402

N_PLANES = 6
GRID = 64


def contra_planes(frame: np.ndarray) -> np.ndarray:
    """``[64, 64]`` int frame (values 0-15) → ``[6, 64, 64]`` float32 planes."""
    grid = np.asarray(frame, dtype=np.int16)
    if grid.shape != (GRID, GRID):
        raise ValueError(f"expected (64, 64) frame, got {grid.shape}")

    packet = build_raw_color_component_packet(grid)
    components, label_map = raw_pixel_components_with_labels(grid)

    planes = np.zeros((N_PLANES, GRID, GRID), dtype=np.float32)

    bg = packet["background_raw_pixel_value"]
    planes[0] = (grid == bg).astype(np.float32)
    planes[1] = 1.0 - planes[0]

    # per-component area and centroid offset via Contra's label map
    for comp_idx, comp in enumerate(components):
        mask = label_map == comp_idx
        if not mask.any():
            continue
        area = float(mask.sum())
        planes[2][mask] = np.log1p(area) / np.log1p(GRID * GRID)
        ys, xs = np.nonzero(mask)
        cy, cx = ys.mean(), xs.mean()
        dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
        planes[3][mask] = dist / (GRID * np.sqrt(2.0))

    edge = np.zeros((GRID, GRID), dtype=bool)
    edge[:-1, :] |= grid[:-1, :] != grid[1:, :]
    edge[1:, :] |= grid[1:, :] != grid[:-1, :]
    edge[:, :-1] |= grid[:, :-1] != grid[:, 1:]
    edge[:, 1:] |= grid[:, 1:] != grid[:, :-1]
    planes[4] = edge.astype(np.float32)

    freq = np.bincount(grid.reshape(-1).clip(0, 15), minlength=16) / float(GRID * GRID)
    planes[5] = freq[grid.clip(0, 15)].astype(np.float32)

    return planes
