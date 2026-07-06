"""V031-style perception trust gate (Dreamweaver port) — DW plan Phase 1.

Round-trip trust for the extractor: before any sim/imagination lane consumes a
Scene, prove the Scene's semantic claims against the OBSERVED frame itself.
Ported concepts from dreamweaver `chronometric_map_perception.py` (V031):
strict thresholds -> `trusted` bool + `gate_failures` list; rays as a geometry
trust signal ("prove the internal drawing matches the environment before the
planner treats ray output as evidence").

All checks compare Scene claims to FRAME PIXELS only — no engine internals.
Checks run on the scene's NATIVE lattice (Scene.cell/pad, induced from glyph
pixel runs): L0-L2 = 4px origin-0 (the historical fixed grid), L3-L5 = 3px
origin-2. The padded canonical frame is rebuilt here INDEPENDENTLY from the
raw frame, so a corrupted Scene cannot vouch for itself.

Gate checks:
  targets    #4-connected color-11 pixel components == len(scene.targets)
             (the L3 failure class: macro components merged 4 targets into 2)
  glyphs     union of target.cells_px == (frame==11) mask exactly, no overlap
  spouts     #color-4 housing pixel components == len(scene.spouts); each spout
             open and column-aligned under a housing
             (the L2 failure class: 3 sources collapsed to 1 spout)
  mouths     every mouth cell open; the mouth boundary has EXACTLY ONE open
             side, which matches the target's claimed `opening` direction
             (generalized from up-only U-enclosure: L4/L5 mouths face sideways)
  mouth_rays a ray entering each mouth line along its opening direction
             first-contacts the mouth's far wall, at lattice AND pixel level
  floor      floor_rows re-derived from pixels == scene.floor_rows
  coverage   every solid-color pixel lies inside a solid lattice cell

Non-gating diagnostics: unexplained solid cells, spout down-ray macro/pixel
divergence, and UNMODELED COLORS present in frame (e.g. color 15 redirect
blocks on L4/L5 — geometry the Scene does not yet claim physics for).
"""
from __future__ import annotations

import numpy as np

from scene import (GRID, OPEN, SOURCE, HUD, TARGET, SOLID_COLORS, PALETTE,
                   _pixel_components)

# material ENTERS a mouth moving opposite to the side the mouth opens toward
ENTRY = {"up": (1, 0), "down": (-1, 0), "left": (0, 1), "right": (0, -1)}
KNOWN_COLORS = set(PALETTE.values()) | {0}     # 0 = score digits / black chrome


def _cell_ray(solid, y0, x0, d, skip=()):
    """First solid lattice cell from (y0,x0) along d (skipping `skip`), or None."""
    gh, gw = solid.shape
    y, x = y0, x0
    while True:
        y += d[0]; x += d[1]
        if not (0 <= y < gh and 0 <= x < gw):
            return None
        if (y, x) in skip:
            continue
        if solid[y, x]:
            return (y, x)


def _pixel_ray(psolid, py, px, d):
    """First solid pixel from (py,px) along d, or None."""
    h, w = psolid.shape
    y, x = py, px
    while True:
        y += d[0]; x += d[1]
        if not (0 <= y < h and 0 <= x < w):
            return None
        if psolid[y, x]:
            return (y, x)


def gate_scene(frame, scene):
    """Scene-vs-frame trust gate. Returns V031-shaped verdict dict."""
    pre = np.asarray(frame, dtype=np.int8)
    u = int(getattr(scene, "cell", 4))
    pad_y, pad_x = (int(p) for p in getattr(scene, "pad", (0, 0)))
    gh, gw = scene.solid.shape
    solid = scene.solid

    # Rebuild the padded canonical frame from raw pixels (independent of Scene).
    f0 = np.flipud(pre) if scene.flipped else pre
    f = np.full((gh * u, gw * u), OPEN, dtype=np.int8)
    f[pad_y:pad_y + GRID, pad_x:pad_x + GRID] = f0
    W = gw * u
    checks = {}

    # Pixel-level solid mask (padded canonical), incl. lava strip in floor rows.
    psolid = np.isin(f, np.asarray(SOLID_COLORS, dtype=f.dtype))
    for r in scene.floor_rows:
        rows = slice(r * u, (r + 1) * u)
        psolid[rows, :] |= (f[rows, :] == HUD)

    # -- targets: pixel component count vs extracted target count (L3 class)
    tgt_comps = _pixel_components(pre == TARGET)          # original coords, like cells_px
    checks["targets"] = {
        "pass": len(tgt_comps) == len(scene.targets),
        "pixel_components": len(tgt_comps),
        "scene_targets": len(scene.targets),
    }

    # -- glyphs: exact pixel reconstruction, no overlap
    union = np.zeros((GRID, GRID), dtype=bool)
    overlap = 0
    for t in scene.targets:
        overlap += int((union & t.cells_px).sum())
        union |= t.cells_px
    exact = bool((union == (pre == TARGET)).all())
    checks["glyphs"] = {
        "pass": exact and overlap == 0,
        "exact_reconstruction": exact,
        "overlap_pixels": overlap,
        "missed_pixels": int(((pre == TARGET) & ~union).sum()),
        "phantom_pixels": int((union & (pre != TARGET)).sum()),
    }

    # -- spouts: one per color-4 housing, open, column-aligned (L2 class)
    housing = _pixel_components(f == SOURCE)
    aligned, open_ok = 0, 0
    for (sy, sx) in scene.spouts:
        if 0 <= sy < gh and not solid[sy, sx]:
            open_ok += 1
        if any(int(xs.min()) // u <= sx <= int(xs.max()) // u for _, xs in housing):
            aligned += 1
    checks["spouts"] = {
        "pass": (len(housing) == len(scene.spouts)
                 and aligned == len(scene.spouts) and open_ok == len(scene.spouts)),
        "housing_components": len(housing),
        "scene_spouts": len(scene.spouts),
        "aligned": aligned,
        "open": open_ok,
    }

    # -- mouths: open cells + single-opening enclosure matching the claim
    mouth_ok, mouth_detail = True, []
    for i, t in enumerate(scene.targets):
        d = {"target": i, "mouth_cells": len(t.mouth)}
        if not t.mouth:
            d["fail"] = "empty_mouth"; mouth_ok = False; mouth_detail.append(d); continue
        mset = set(t.mouth)
        bad = [c for c in t.mouth if solid[c]]
        open_dirs = set()
        for (my, mx) in t.mouth:
            for name, (dy, dx) in (("up", (-1, 0)), ("down", (1, 0)),
                                   ("left", (0, -1)), ("right", (0, 1))):
                ny, nx = my + dy, mx + dx
                if (ny, nx) in mset:
                    continue
                if 0 <= ny < gh and 0 <= nx < gw and not solid[ny, nx]:
                    open_dirs.add(name)
        d.update(solid_mouth_cells=len(bad),
                 open_sides=sorted(open_dirs),
                 claimed_opening=t.opening)
        if bad or len(open_dirs) != 1 or t.opening not in open_dirs:
            d["fail"] = "enclosure"; mouth_ok = False
        mouth_detail.append(d)
    checks["mouths"] = {"pass": mouth_ok, "targets": mouth_detail}

    # -- mouth_rays: entry ray along the opening direction first-contacts the
    #    mouth's far wall, at lattice AND pixel level
    ray_total, ray_exact, ray_detail = 0, 0, []
    for i, t in enumerate(scene.targets):
        if not t.mouth or t.opening not in ENTRY:
            continue
        dy, dx = ENTRY[t.opening]
        mset = set(t.mouth)
        # group mouth cells into entry lines (same column for vertical entry,
        # same row for horizontal entry)
        key = (lambda c: c[1]) if dy else (lambda c: c[0])
        along = (lambda c: c[0]) if dy else (lambda c: c[1])
        sgn = dy + dx                                     # +1 or -1 along the axis
        for line in sorted({key(c) for c in mset}):
            cells = sorted((c for c in mset if key(c) == line), key=along)
            first = cells[0] if sgn > 0 else cells[-1]    # entry-most mouth cell
            last = cells[-1] if sgn > 0 else cells[0]
            expect = (last[0] + dy, last[1] + dx)         # far wall behind the mouth
            m_hit = _cell_ray(solid, first[0] - dy, first[1] - dx, (dy, dx), skip=mset)
            pc = (first[0] * u + u // 2, first[1] * u + u // 2)
            p_hit = _pixel_ray(psolid, pc[0] - dy * u, pc[1] - dx * u, (dy, dx))
            p_cell = (p_hit[0] // u, p_hit[1] // u) if p_hit else None
            ray_total += 1
            ok = (m_hit == expect and p_cell == expect)
            ray_exact += int(ok)
            if not ok:
                ray_detail.append({"target": i, "line": int(line),
                                   "expect": list(expect),
                                   "macro_hit": list(m_hit) if m_hit else None,
                                   "pixel_hit_cell": list(p_cell) if p_cell else None})
    checks["mouth_rays"] = {
        "pass": ray_total > 0 and ray_exact == ray_total,
        "total": ray_total, "exact": ray_exact, "mismatches": ray_detail,
    }

    # -- floor: strip rows re-derived from pixels (same induction as extractor)
    tgt_rows_c = np.nonzero((f == TARGET).any(axis=1))[0]
    if len(tgt_rows_c):
        strip = [r for r in np.nonzero((f == HUD).all(axis=1)
                                       | ((f == HUD).sum(axis=1) > W // 2))[0]
                 if r > tgt_rows_c.max()]
        floor_px = sorted({r // u for r in strip})
    else:
        floor_px = []
    checks["floor"] = {
        "pass": floor_px == sorted(scene.floor_rows),
        "pixel_derived": floor_px, "scene_floor_rows": sorted(scene.floor_rows),
    }

    # -- coverage: every solid-color pixel sits inside a solid lattice cell
    pix_any = np.isin(f, np.asarray(SOLID_COLORS, dtype=f.dtype))
    blocks = pix_any.reshape(gh, u, gw, u).any(axis=(1, 3))
    uncovered = int((blocks & ~solid).sum())
    checks["coverage"] = {"pass": uncovered == 0, "uncovered_macro_cells": uncovered}

    # -- diagnostics (reported, not gated)
    explained = np.zeros((gh, gw), dtype=bool)
    for t in scene.targets:
        explained |= t.glyph
    for r in scene.floor_rows:
        explained[r, :] = True
    diag_unexplained = int((solid & ~explained & ~blocks).sum())
    spout_rays = []
    for (sy, sx) in scene.spouts:
        m_hit = _cell_ray(solid, sy, sx, (1, 0))
        p_hit = _pixel_ray(psolid, (sy + 1) * u - 1, sx * u + u // 2, (1, 0))
        m_row = m_hit[0] if m_hit else gh
        p_row = (p_hit[0] // u) if p_hit else gh
        spout_rays.append({"spout": [int(sy), int(sx)],
                           "macro_hit": int(m_row), "pixel_hit": int(p_row),
                           "agree": bool(m_row == p_row)})
    unmodeled = sorted(int(c) for c in np.unique(pre) if int(c) not in KNOWN_COLORS)

    failures = [name for name, c in checks.items() if not c["pass"]]
    return {
        "trusted": not failures,
        "gate_failures": failures,
        "checks": checks,
        "diagnostics": {
            "phantom_solid_cells": diag_unexplained,
            "spout_down_rays": spout_rays,
            "unmodeled_colors": unmodeled,
        },
    }
