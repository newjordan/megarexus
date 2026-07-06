"""Faithful droplet-CA pour (the user's "ball that splits into two orthogonal directions and
falls off the side of a wall" — 2026-07-06). A direct port of the GAME's droplet stepping
(sp80.py L749-820), NOT the hydraulic pooling model. The key mechanic the hydraulic lane
missed: paint is DROPLETS that split around blocks and continue down, and any droplet that
runs off the LEFT/RIGHT board edge is DISPOSED (the fail sensor `waoewejnqzc` is only the
BOTTOM wall — the sides are open), so excess paint is shed off the sides, not flooded.

On levels with NO redirect (L2), every droplet moves straight DOWN and a block forks it to
x-1/x+1 (both still down). The pour outcome (which mouths fill, whether the bottom wall is
touched) is therefore a DETERMINISTIC reachability graph — volume is irrelevant (one droplet
fills a mouth enclosed; one droplet touching the wall fails). We compute it exactly with a BFS
over (row,col) down-moving droplet states. Microseconds, exact.

Contract mirrors the game: fill a target when a droplet ENTERS its enclosed mouth cell (glyph
below + both perpendicular sides, L775); FAIL when a droplet's next cell is the bottom wall
(L815); dispose a droplet that leaves the x-edges. WIN = all targets filled AND no wall touch.

    from dropletsim import droplet_pour
    out = droplet_pour(frame)   # {'filled', 'fail', 'win', 'n_targets', 'margin'}
"""
from __future__ import annotations

import numpy as np

import _paths  # noqa: F401
from scene import extract_scene, M, MOVABLE, SELECTED, TARGET, WALL


def _typed(scene):
    """Per-cell type masks on the canonical macro grid (flow = down)."""
    g = scene.g
    block = np.isin(g, (MOVABLE, SELECTED))            # movable split blocks (8/9)
    target = (g == TARGET)                             # U-glyph (11)
    wall = (g == WALL)                                 # wall/flash (14)
    for r in scene.floor_rows:                         # the bottom lava/wall strip
        wall[r, :] = True
    tid = -np.ones((M, M), dtype=int)
    for i, t in enumerate(scene.targets):
        for (yy, xx) in zip(*np.nonzero(t.glyph)):
            tid[yy, xx] = i
    return block, target, wall, tid


def pour_core(block, target, wall, tid, spouts, n_targets):
    """Exact droplet-CA outcome on typed grids (no-redirect: all motion DOWN). Grids are [H,W]
    bool of ANY size (frame-macro 16x16 OR native sprite coords); tid = [H,W] int target id per
    glyph cell (-1 elsewhere); spouts = [(y,x)]. Off the L/R x-edges = disposed. sp80 mechanic:
    a bucket is 'painted' the tick a droplet enters it enclosed; touching the floor fails;
    LEFTOVER paint is irrelevant (win = every bucket painted AND floor never touched)."""
    H, W = block.shape

    def kind(y, x):
        if x < 0 or x >= W or y < 0 or y >= H:
            return None
        if block[y, x]:
            return "block"
        if target[y, x]:
            return "target"
        if wall[y, x]:
            return "wall"
        return None

    filled, wall_hits = set(), set()
    stack = [(int(sy), int(sx)) for (sy, sx) in spouts]
    seen = set()
    while stack:
        y, x = stack.pop()
        if (y, x) in seen:
            continue
        seen.add((y, x))
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        nxt = kind(y + 1, x)
        if nxt is None:
            stack.append((y + 1, x))
        elif nxt == "wall":
            wall_hits.add((y + 1, x))
        elif nxt == "block":
            for nx in (x - 1, x + 1):
                if kind(y, nx) is None:
                    stack.append((y, nx))
        elif nxt == "target":
            le, ri = kind(y, x - 1), kind(y, x + 1)
            if le == "target" and ri == "target":
                filled.add(int(tid[y + 1, x]))
            else:
                for nx in (x - 1, x + 1):
                    if kind(y, nx) is None:
                        stack.append((y, nx))
    filled.discard(-1)
    fail = len(wall_hits) > 0
    win = (not fail) and n_targets > 0 and len(filled) >= n_targets
    margin = (1.0 if win else (len(filled) / max(n_targets, 1)) - min(1.0, 0.1 * len(wall_hits)))
    return {"filled": sorted(filled), "n_filled": len(filled), "fail": bool(fail),
            "n_wall": len(wall_hits), "win": bool(win), "n_targets": n_targets,
            "margin": float(margin)}


def pour_core_v(block, target, wall, redir, tid, spouts, n_targets):
    """Exact droplet CA WITH velocity + redirect (game L749-820, incl. tuvkdkhdokr 90-deflect
    L787-803). Droplets carry velocity (dx,dy); state = (y,x,dx,dy). A block/target forks to the
    two perpendicular cells keeping velocity; a redirect rotates velocity 90 deg toward its body;
    a wall fails; off any edge disposes. Reduces to pour_core when there are no redirects (spouts
    all start moving down). Grids [H,W] bool."""
    H, W = block.shape

    def kind(y, x):
        if x < 0 or x >= W or y < 0 or y >= H:
            return None
        if block[y, x]:
            return "block"
        if target[y, x]:
            return "target"
        if wall[y, x]:
            return "wall"
        if redir[y, x]:
            return "redir"
        return None

    def perp(dx, dy):
        return [(-1, 0), (1, 0)] if dy != 0 else [(0, -1), (0, 1)]

    filled, wall_hits = set(), set()
    stack = [(int(sy), int(sx), 0, 1) for (sy, sx) in spouts]
    seen = set()
    while stack:
        y, x, dx, dy = stack.pop()
        if (y, x, dx, dy) in seen:
            continue
        seen.add((y, x, dx, dy))
        if x < 0 or x >= W or y < 0 or y >= H:
            continue
        nx, ny = x + dx, y + dy
        k = kind(ny, nx)
        p = perp(dx, dy)
        if k is None:
            stack.append((ny, nx, dx, dy))                       # move
        elif k == "wall":
            wall_hits.add((ny, nx))
        elif k == "block":
            for (ox, oy) in p:
                if kind(y + oy, x + ox) is None:
                    stack.append((y + oy, x + ox, dx, dy))       # fork, keep velocity
        elif k == "target":
            (ax, ay), (bx, by) = p
            a, b = kind(y + ay, x + ax), kind(y + by, x + bx)
            if a == "target" and b == "target":
                filled.add(int(tid[ny, nx]))
            else:
                for (ox, oy) in p:
                    if kind(y + oy, x + ox) is None:
                        stack.append((y + oy, x + ox, dx, dy))
        elif k == "redir":
            # Faithful to game L787-811: TWO separate ifs, and the `else` (split) attaches to the
            # SECOND if — so side-A deflect has NO continue and FALLS THROUGH to a split unless
            # side-B also deflects. Net: a clean 90-deflect happens only when an ADJOINING piece
            # walls off the opposite side (user's rule); otherwise the redirect deflects AND splits
            # (spreading paint that then reaches the floor). This is the L4 false-positive fix.
            (ax, ay), (bx, by) = p                               # side A (owyqnbmyij), side B (faksfmaezd)
            a, b = kind(y + ay, x + ax), kind(y + by, x + bx)
            did_a = (a == "redir" and b is None)                 # A: side-A is the redirect, B empty
            did_b = (b == "redir" and a is None)                 # B: side-B is the redirect, A empty
            if did_a:                                            # deflect toward A (L790, no continue)
                stack.append((y - dx, x + dy, dy, -dx))
            if did_b:                                            # deflect toward B (L797)
                stack.append((y + dx, x - dy, -dy, dx))
            else:                                                # else binds to the did_b if (L804)
                for (ox, oy) in p:
                    if kind(y + oy, x + ox) is None:
                        stack.append((y + oy, x + ox, dx, dy))
    filled.discard(-1)
    fail = len(wall_hits) > 0
    win = (not fail) and n_targets > 0 and len(filled) >= n_targets
    margin = (1.0 if win else (len(filled) / max(n_targets, 1)) - min(1.0, 0.1 * len(wall_hits)))
    return {"filled": sorted(filled), "n_filled": len(filled), "fail": bool(fail),
            "n_wall": len(wall_hits), "win": bool(win), "n_targets": n_targets, "margin": float(margin)}


def droplet_pour(frame, scene=None):
    """Exact droplet-CA outcome for one pour (no-redirect levels: all motion is DOWN)."""
    sc = scene if scene is not None else extract_scene(frame)
    block, target, wall, tid = _typed(sc)
    return pour_core(block, target, wall, tid, list(sc.spouts or [sc.spout]), len(sc.targets))
