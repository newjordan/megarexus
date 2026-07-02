"""Presence-aware structural sense -- floor / walls without force-picking.

CONTRA's existing floor/wall detectors always emit a pick, so they are right only
when the layout happens to match (2-3 of 6 SP80 levels). This sense instead reads
the boundary structure straight from geometry and emits a floor ONLY when a
wide low band exists, walls ONLY when vertical sides / a wide top band exist, and
abstains otherwise. Pure geometry, no labels, no forced pick.

Classification per non-background component:
  - wide (>=50% width) and low (centre in bottom ~half), not full-height -> FLOOR
  - tall (>=50% height), not full-width                                  -> WALL (side)
  - wide and high (centre in top ~half), not full-height                 -> WALL (top)
  - wide AND tall (a frame outline touching all edges)                   -> WALL (border)
"""

from __future__ import annotations

import numpy as np

NEI4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    H, W = mask.shape
    out = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in NEI4:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append(np.array(cells))
    return out


def _background(frame):
    v, c = np.unique(frame, return_counts=True)
    return int(v[c.argmax()])


def detect_structure(frame, wide_frac: float = 0.5, tall_frac: float = 0.5, min_cells: int = 24) -> dict:
    frame = np.asarray(frame)
    if frame.ndim == 3:
        frame = frame[0]
    H, W = frame.shape
    bg = _background(frame)
    floors, walls = [], []
    band = 5  # thickness of the edge band we test
    for color in np.unique(frame):
        if int(color) == bg:
            continue
        for comp in connected_components(frame == color):
            if len(comp) < min_cells:
                continue
            ys, xs = comp[:, 0], comp[:, 1]
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            # must touch the frame edge to be structural (interior objects don't)
            if not (y1 <= 2 or y2 >= H - 3 or x1 <= 2 or x2 >= W - 3):
                continue
            bbox = [x1, y1, x2, y2]

            def _span(sel_axis, edge_cells_axis):
                return (edge_cells_axis.max() - edge_cells_axis.min() + 1) if len(edge_cells_axis) else 0

            # a structural component is decomposed into its edge BANDS, so a unified
            # frame outline contributes BOTH floor (bottom band) and walls (sides).
            bx = xs[ys >= H - band]          # bottom band -> floor if it spans wide
            if _span("x", bx) >= wide_frac * W:
                floors.append({"color": int(color), "bbox": bbox, "kind": "bottom_band"})
            tx = xs[ys <= band - 1]          # top band -> top wall
            if _span("x", tx) >= wide_frac * W:
                walls.append({"color": int(color), "bbox": bbox, "kind": "top_band"})
            ly = ys[xs <= band - 1]          # left band -> side wall
            if _span("y", ly) >= tall_frac * H:
                walls.append({"color": int(color), "bbox": bbox, "kind": "left_side"})
            ry = ys[xs >= W - band]          # right band -> side wall
            if _span("y", ry) >= tall_frac * H:
                walls.append({"color": int(color), "bbox": bbox, "kind": "right_side"})
    return {
        "schema": "codexlab.contra_structural_sense.v1",
        "background": bg,
        "has_floor": bool(floors),
        "has_walls": bool(walls),
        "floors": floors,
        "walls": walls,
    }
