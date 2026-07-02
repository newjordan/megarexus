"""General shape-fit relationship finder -- pure geometry, no labels, no gating.

Principle (relationship-finding primitive): ANY shape that nests perfectly into
another shape's negative space is a strong signal that the two "separate" objects
share a designed relationship. A perfect complementary fit almost never happens
by accident. So for every ordered pair of objects (A, B) in a frame we ask: does
A fit into a concavity (pocket) of B? If yes, we record a LINK -- B is the
'socket' (it has the pocket), A is the 'plug' (it fills it). Defining what that
relationship *does* is a later, dynamic step; this layer only FINDS it.

Unlike CONTRA's existing lock/key sensor, this is NOT gated behind polar/reverse
symmetry -- it works for any pair of shapes, in any of 8 orientations, on any
game. It reads pixels, not an answer key.

Pipeline:
  1. objects = connected components of every non-background color (background =
     most common color; a geometric fact, not a label)
  2. for each object B, find its concavity pockets = convex_hull(B) minus B,
     split into connected pockets
  3. for each other object A, test if A (in any of 8 orientations) is congruent
     to a pocket (perfect fit) or fits inside it (snug/partial fit)
  4. emit fit relationships with scores
"""

from __future__ import annotations

import numpy as np

NEI4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


# --------------------------------------------------------------- object extraction
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


def background_color(frame: np.ndarray) -> int:
    v, c = np.unique(frame, return_counts=True)
    return int(v[c.argmax()])


def objects_from_frame(frame: np.ndarray, min_area: int = 3) -> list[dict]:
    bg = background_color(frame)
    objs = []
    for color in np.unique(frame):
        if int(color) == bg:
            continue
        for cells in connected_components(frame == color):
            if len(cells) < min_area:
                continue
            ys, xs = cells[:, 0], cells[:, 1]
            objs.append({
                "id": f"o_{int(color)}_{int(ys.min())}_{int(xs.min())}",
                "color": int(color),
                "cells": cells,
                "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "area": int(len(cells)),
            })
    return objs


# ------------------------------------------------------------------- convex hull
def _convex_hull(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Monotone-chain convex hull (CCW) of integer points (x, y)."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _point_in_hull(px: float, py: float, hull: list[tuple[int, int]]) -> bool:
    n = len(hull)
    if n < 3:
        return False
    for i in range(n):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % n]
        # CCW hull: interior is to the left of each directed edge (cross >= 0)
        if (bx - ax) * (py - ay) - (by - ay) * (px - ax) < -1e-9:
            return False
    return True


def concavity_pockets(cells: np.ndarray) -> list[np.ndarray]:
    """convex_hull(cells) minus cells, split into connected pockets."""
    ys, xs = cells[:, 0], cells[:, 1]
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    if (y1 - y0) < 1 or (x1 - x0) < 1:
        return []
    # corner points of every filled cell -> hull
    corners = []
    for y, x in cells:
        corners += [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
    hull = _convex_hull(corners)
    cellset = {(int(y), int(x)) for y, x in cells}
    H = y1 - y0 + 1
    W = x1 - x0 + 1
    pocket_mask = np.zeros((H, W), dtype=bool)
    for r in range(y0, y1 + 1):
        for c in range(x0, x1 + 1):
            if (r, c) in cellset:
                continue
            if _point_in_hull(c + 0.5, r + 0.5, hull):
                pocket_mask[r - y0, c - x0] = True
    pockets = []
    for comp in connected_components(pocket_mask):
        pockets.append(comp + np.array([y0, x0]))  # back to frame coords
    return pockets


# ------------------------------------------------------------- shape congruence
def _normalize(cells: np.ndarray) -> frozenset:
    ys, xs = cells[:, 0], cells[:, 1]
    return frozenset(zip((ys - ys.min()).tolist(), (xs - xs.min()).tolist()))


def _orientations(cells: np.ndarray) -> list[frozenset]:
    """8 orientations (4 rotations x optional reflection), as normalized sets."""
    ys, xs = cells[:, 0], cells[:, 1]
    base = np.stack([ys, xs], axis=1)
    variants = []
    pts = base.copy()
    for refl in (False, True):
        p = pts.copy()
        if refl:
            p[:, 1] = -p[:, 1]
        for _ in range(4):
            variants.append(_normalize(p))
            # rotate 90: (r, c) -> (c, -r)
            p = np.stack([p[:, 1], -p[:, 0]], axis=1)
    return list(set(variants))


def fit_test(a_cells: np.ndarray, pocket: np.ndarray) -> dict | None:
    """Best fit of object A into a pocket. Returns kind/fill_ratio or None."""
    pocket_norm = _normalize(pocket)
    a_orients = _orientations(a_cells)
    # perfect: A congruent to the whole pocket
    if pocket_norm in a_orients:
        return {"kind": "perfect", "fill_ratio": 1.0, "pocket_area": int(len(pocket)),
                "plug_area": int(len(a_cells))}
    # snug/partial: some orientation of A fits inside the pocket by translation
    pocket_set = set(_normalize(pocket))
    pys = [p[0] for p in pocket_set]; pxs = [p[1] for p in pocket_set]
    ph, pw = max(pys) + 1, max(pxs) + 1
    best = None
    for orient in a_orients:
        oys = [o[0] for o in orient]; oxs = [o[1] for o in orient]
        oh, ow = max(oys) + 1, max(oxs) + 1
        if oh > ph or ow > pw or len(orient) > len(pocket_set):
            continue
        for dy in range(ph - oh + 1):
            for dx in range(pw - ow + 1):
                placed = {(y + dy, x + dx) for y, x in orient}
                if placed <= pocket_set:
                    fr = len(placed) / len(pocket_set)
                    if best is None or fr > best["fill_ratio"]:
                        best = {"kind": "snug" if fr >= 0.8 else "partial",
                                "fill_ratio": round(fr, 3),
                                "pocket_area": len(pocket_set), "plug_area": len(orient)}
    return best


# ------------------------------------------------------------- top-level finder
def find_shape_fits(frame: np.ndarray, min_fill: float = 0.6) -> dict:
    objs = objects_from_frame(frame)
    pockets_by_obj = {o["id"]: concavity_pockets(o["cells"]) for o in objs}
    relationships = []
    for socket in objs:
        for pi, pocket in enumerate(pockets_by_obj[socket["id"]]):
            if len(pocket) < 2:
                continue
            pys, pxs = pocket[:, 0], pocket[:, 1]
            pbb = [int(pxs.min()), int(pys.min()), int(pxs.max()), int(pys.max())]
            for plug in objs:
                if plug["id"] == socket["id"]:
                    continue
                fit = fit_test(plug["cells"], pocket)
                if fit and fit["fill_ratio"] >= min_fill:
                    relationships.append({
                        "plug_id": plug["id"], "plug_color": plug["color"],
                        "socket_id": socket["id"], "socket_color": socket["color"],
                        "pocket_index": pi, "pocket_bbox": pbb,
                        "fit_kind": fit["kind"], "fill_ratio": fit["fill_ratio"],
                        "pocket_area": fit["pocket_area"], "plug_area": fit["plug_area"],
                    })
    relationships.sort(key=lambda r: (-{"perfect": 2, "snug": 1, "partial": 0}[r["fit_kind"]], -r["fill_ratio"]))
    return {
        "schema": "codexlab.contra_shape_fit.v1",
        "n_objects": len(objs),
        "objects": [{k: o[k] for k in ("id", "color", "bbox", "area")} for o in objs],
        "n_pockets": sum(len(p) for p in pockets_by_obj.values()),
        "fit_relationships": relationships,
    }
