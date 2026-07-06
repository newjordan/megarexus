"""Build a pour scene from the GAME's native sprite grid (exact positions/pixels), not the
64px frame macro. The frame-macro extractor (16x16, 4px) mis-sizes levels whose native cell
!= 4px (L3's buckets render 3px-wide but are 1-cell U's natively). Working in native coords =
the same grid the game's droplet logic (L749-820) runs on, so the droplet CA is exact.

Sidewalls seen in the frame (L3 cols 0-1/62-63 color-1) are HUD border, NOT game sprites — the
only fail sensor is the `waoewejnqzc` wall. Sides are open (disposal), same as L2.
"""
from __future__ import annotations

import numpy as np

SRC_TAG = "sowlljgtjvn"
TGT_TAG = "repwkzbkhxl"
WALL_TAG = "waoewejnqzc"
BLOCK_TAG = "plzwjbfyfli"
REDIR_TAG = "tuvkdkhdokr"


def sprite_cells(s):
    """Occupied native cells of a sprite = its non-negative pixels at (x+px, y+py)."""
    px = np.asarray(s.pixels)
    out = []
    for yy in range(px.shape[0]):
        for xx in range(px.shape[1]):
            if int(px[yy, xx]) >= 0:
                out.append((s.y + yy, s.x + xx))    # (row, col) native
    return out


def native_grid(game):
    """Typed native-coord grids from the live level's sprites. Returns dict with block/target/
    wall bool grids, tid (target id per glyph cell), spouts (droplet spawn cells), n_targets, dims."""
    cl = game.current_level
    def by(tag):
        return list(cl.get_sprites_by_tag(tag))
    srcs, tgts, walls = by(SRC_TAG), by(TGT_TAG), by(WALL_TAG)
    blocks, redirs = by(BLOCK_TAG), by(REDIR_TAG)

    allcells = []
    for grp in (srcs, tgts, walls, blocks, redirs):
        for s in grp:
            allcells += sprite_cells(s)
    H = max(r for r, _ in allcells) + 2
    W = max(c for _, c in allcells) + 1
    block = np.zeros((H, W), bool); target = np.zeros((H, W), bool)
    wall = np.zeros((H, W), bool); redir = np.zeros((H, W), bool)
    tid = -np.ones((H, W), int)
    for s in blocks:
        for (r, c) in sprite_cells(s):
            block[r, c] = True
    for s in redirs:
        for (r, c) in sprite_cells(s):
            redir[r, c] = True
    for i, s in enumerate(tgts):
        for (r, c) in sprite_cells(s):
            target[r, c] = True; tid[r, c] = i
    for s in walls:
        for (r, c) in sprite_cells(s):
            wall[r, c] = True
    # spouts: a droplet spawns just BELOW each color-4 SOURCE pixel that is unobstructed
    # (game L644-650). A "source" sprite is often a wide bar whose pixels are mostly block
    # (color 8) with a SINGLE color-4 emitter embedded (e.g. L3: [8 8 8 4 8 8 8] -> emits only
    # its col-4 pixel); the block-colored cells are solid, not emitters. Filter to color 4.
    spouts = []
    occ = block | target | wall | redir
    for s in srcs:
        px = np.asarray(s.pixels)
        for yy in range(px.shape[0]):
            for xx in range(px.shape[1]):
                if int(px[yy, xx]) == 4:
                    r, c = s.y + yy, s.x + xx
                    br = r + 1
                    if br < H and not occ[br, c]:
                        spouts.append((br, c))
        # the wide source bar's non-emitter cells are solid (a movable block co-located with it)
        for (r, c) in sprite_cells(s):
            if not target[r, c] and not wall[r, c]:
                block[r, c] = True
    spouts = sorted(set(spouts))
    return {"block": block, "target": target, "wall": wall, "redir": redir, "tid": tid,
            "spouts": spouts, "n_targets": len(tgts), "H": H, "W": W}
