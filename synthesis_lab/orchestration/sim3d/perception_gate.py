"""V031-style perception trust gate (Dreamweaver port) — DW plan Phase 1.

Round-trip trust for the extractor: before any sim/imagination lane consumes a
Scene, prove the Scene's semantic claims against the OBSERVED frame itself.
Ported concepts from dreamweaver `chronometric_map_perception.py` (V031):
strict thresholds -> `trusted` bool + `gate_failures` list; rays as a geometry
trust signal ("prove the internal drawing matches the environment before the
planner treats ray output as evidence").

All checks compare Scene claims to FRAME PIXELS only — no engine internals.

Gate checks:
  targets    #4-connected color-11 pixel components == len(scene.targets)
             (the L3 failure class: extractor found 2 of 4 targets)
  glyphs     union of target.cells_px == (frame==11) mask exactly, no overlap
             (mis-sized glyphs from macro-vote at wrong native cell scale)
  spouts     #color-4 housing pixel components == len(scene.spouts); each spout
             open and column-aligned under a housing
             (the L2 failure class: 3 sources collapsed to 1 spout)
  mouths     every mouth cell open; enclosed left/right/below (U-shape)
  mouth_rays down-ray entering each mouth column first-contacts the mouth floor,
             and macro-grid ray contact == pixel-level ray contact
  floor      floor_rows re-derived from pixels == scene.floor_rows
  coverage   every solid-color pixel lies inside a solid macro cell

Non-gating diagnostics: unexplained solid cells, spout down-ray macro/pixel
divergence (partial-solid cells such as 62px wall-bar ends can legitimately
diverge on a center-line pixel ray; reported, not gated).
"""
from __future__ import annotations

import numpy as np

from scene import (GRID, M, CELL, SOURCE, PAINT, HUD, TARGET, SOLID_COLORS)

DOWN = (1, 0)


def _pixel_components(mask):
    """4-connected components of a bool [64,64] pixel mask -> list of (ys, xs)."""
    mask = np.asarray(mask, dtype=bool)
    lab = np.full(mask.shape, -1, dtype=np.int32)
    comps = []
    for y, x in zip(*np.nonzero(mask)):
        if lab[y, x] >= 0:
            continue
        stack, cells = [(y, x)], []
        lab[y, x] = len(comps)
        while stack:
            cy, cx = stack.pop()
            cells.append((cy, cx))
            for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                if (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]
                        and mask[ny, nx] and lab[ny, nx] < 0):
                    lab[ny, nx] = len(comps)
                    stack.append((ny, nx))
        ys = np.asarray([c[0] for c in cells]); xs = np.asarray([c[1] for c in cells])
        comps.append((ys, xs))
    return comps


def _macro_ray_down(solid, y0, x):
    """First solid macro row at/below y0 in column x, or M."""
    y = max(y0, 0)
    while y < M and not solid[y, x]:
        y += 1
    return y


def _pixel_ray_down(psolid, py0, px):
    """First solid pixel row at/below py0 in pixel column px, or GRID."""
    y = max(py0, 0)
    while y < GRID and not psolid[y, px]:
        y += 1
    return y


def gate_scene(frame, scene):
    """Scene-vs-frame trust gate. Returns V031-shaped verdict dict."""
    pre = np.asarray(frame, dtype=np.int8)
    f = np.flipud(pre) if scene.flipped else pre          # canonical (flow down)
    solid = scene.solid
    checks = {}

    # Pixel-level solid mask (canonical), incl. the lava strip inside floor rows.
    psolid = np.isin(f, np.asarray(SOLID_COLORS, dtype=f.dtype))
    for r in scene.floor_rows:
        rows = slice(r * CELL, (r + 1) * CELL)
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
        if 0 <= sy < M and not solid[sy, sx]:
            open_ok += 1
        if any(int(xs.min()) // CELL <= sx <= int(xs.max()) // CELL for _, xs in housing):
            aligned += 1
    checks["spouts"] = {
        "pass": (len(housing) == len(scene.spouts)
                 and aligned == len(scene.spouts) and open_ok == len(scene.spouts)),
        "housing_components": len(housing),
        "scene_spouts": len(scene.spouts),
        "aligned": aligned,
        "open": open_ok,
    }

    # -- mouths: open + U-enclosure (left/right walls per row, floor per column)
    mouth_ok, mouth_detail = True, []
    for i, t in enumerate(scene.targets):
        d = {"target": i, "mouth_cells": len(t.mouth)}
        if not t.mouth:
            d["fail"] = "empty_mouth"; mouth_ok = False; mouth_detail.append(d); continue
        rows = sorted({y for y, _ in t.mouth}); cols = sorted({x for _, x in t.mouth})
        bad = [c for c in t.mouth if solid[c]]
        side = all(
            (x0 - 1 < 0 or solid[r, x0 - 1]) and (x1 + 1 >= M or solid[r, x1 + 1])
            for r in rows
            for x0, x1 in [(min(x for y, x in t.mouth if y == r),
                            max(x for y, x in t.mouth if y == r))]
        )
        floor_ok = all(
            (my + 1 >= M) or solid[my + 1, c]
            for c in cols
            for my in [max(y for y, x in t.mouth if x == c)]
        )
        d.update(solid_mouth_cells=len(bad), side_walls=side, floor_below=floor_ok)
        if bad or not side or not floor_ok:
            mouth_ok = False
        mouth_detail.append(d)
    checks["mouths"] = {"pass": mouth_ok, "targets": mouth_detail}

    # -- mouth_rays: macro down-ray first-contacts the mouth floor; pixel ray agrees
    ray_total, ray_exact, ray_detail = 0, 0, []
    for i, t in enumerate(scene.targets):
        if not t.mouth:
            continue
        top = min(y for y, _ in t.mouth)
        for c in sorted({x for _, x in t.mouth}):
            col_rows = [y for y, x in t.mouth if x == c]
            expect = max(col_rows) + 1                     # mouth floor (glyph bottom)
            m_hit = _macro_ray_down(solid, top, c)
            p_hit = _pixel_ray_down(psolid, top * CELL, c * CELL + CELL // 2) // CELL
            ray_total += 1
            ok = (m_hit == expect and p_hit == expect)
            ray_exact += int(ok)
            if not ok:
                ray_detail.append({"target": i, "col": c, "expect": expect,
                                   "macro_hit": m_hit, "pixel_hit": p_hit})
    checks["mouth_rays"] = {
        "pass": ray_total > 0 and ray_exact == ray_total,
        "total": ray_total, "exact": ray_exact, "mismatches": ray_detail,
    }

    # -- floor: strip rows re-derived from pixels
    tgt_rows_c = np.nonzero((f == TARGET).any(axis=1))[0]
    if len(tgt_rows_c):
        strip = [r for r in np.nonzero((f == HUD).all(axis=1)
                                       | ((f == HUD).sum(axis=1) > GRID // 2))[0]
                 if r > tgt_rows_c.max()]
        floor_px = sorted({r // CELL for r in strip})
    else:
        floor_px = []
    checks["floor"] = {
        "pass": floor_px == sorted(scene.floor_rows),
        "pixel_derived": floor_px, "scene_floor_rows": sorted(scene.floor_rows),
    }

    # -- coverage: every solid-color pixel sits inside a solid macro cell
    pix_any = np.isin(f, np.asarray(SOLID_COLORS, dtype=f.dtype))
    blocks = pix_any.reshape(M, CELL, M, CELL).any(axis=(1, 3))
    uncovered = int((blocks & ~solid).sum())
    checks["coverage"] = {"pass": uncovered == 0, "uncovered_macro_cells": uncovered}

    # -- diagnostics (reported, not gated)
    explained = np.zeros((M, M), dtype=bool)
    for t in scene.targets:
        explained |= t.glyph
    for r in scene.floor_rows:
        explained[r, :] = True
    diag_unexplained = int((solid & ~explained & ~blocks).sum())
    spout_rays = []
    for (sy, sx) in scene.spouts:
        m_hit = _macro_ray_down(solid, sy + 1, sx)
        p_hit = _pixel_ray_down(psolid, (sy + 1) * CELL, sx * CELL + CELL // 2) // CELL
        spout_rays.append({"spout": [int(sy), int(sx)],
                           "macro_hit": int(m_hit), "pixel_hit": int(p_hit),
                           "agree": bool(m_hit == p_hit)})

    failures = [name for name, c in checks.items() if not c["pass"]]
    return {
        "trusted": not failures,
        "gate_failures": failures,
        "checks": checks,
        "diagnostics": {
            "phantom_solid_cells": diag_unexplained,
            "spout_down_rays": spout_rays,
        },
    }
