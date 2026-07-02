"""PORTBAND glove adapter: the present-state controllable object = the glove.

Portband's role (per the user's framing and codexlab docs): it is the
"present" — collection, grouping, and each object finding its own edges —
and it holds the moveable objects. In the catch-play regime the paint is
the ball and the Portband-identified moveable object is the glove.

This adapter wraps Portband's REAL port extraction
(codexlab/contra_head/portband.py::_object_ports → per-side edge cell
sets, schema codexlab.geoband_object_port_hypothesis.v1) around an
empirically identified glove:

- glove identification is by controlled interaction evidence (click a
  component, issue a move, observe displacement) — the same
  observed-action-result principle as Contra's action-state tags
  (CONTRA_ACTION_STATE_TAG_POLICY), never by reading game source;
- ports/edges come from Portband's own `_object_ports`;
- hindsight catch labels come from pixels only: did pour-animation cells
  touch the glove's port edge cells or interior?

Truth boundary preserved: gloves and catch labels are training data and
evidence rows, not solve/goal/control truth claims.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CODEXLAB = Path(__file__).resolve().parents[2] / "codexlab"
if str(_CODEXLAB) not in sys.path:
    sys.path.insert(0, str(_CODEXLAB))

from contra_head.portband import _object_ports  # noqa: E402
from contra_head.vision import raw_pixel_components_with_labels  # noqa: E402

GRID = 64


def frame_components(grid: np.ndarray, background: int | None = None) -> list[dict]:
    """Contra components with numpy masks, background excluded."""
    arr = np.asarray(grid, dtype=np.int16)
    if background is None:
        vals, counts = np.unique(arr, return_counts=True)
        background = int(vals[int(np.argmax(counts))])
    components, label_map = raw_pixel_components_with_labels(arr)
    out = []
    for idx, comp in enumerate(components):
        if int(comp["raw_pixel_value"]) == background:
            continue
        mask = label_map == idx
        area = int(mask.sum())
        if area < 3 or area > 800:
            continue  # noise or scenery-scale regions can't be gloves
        ys, xs = np.nonzero(mask)
        # HUD strips / borders (full-width or full-height bands) are not
        # gloves — same suppression heuristic as lewm.perception.sprites
        if xs.max() - xs.min() >= 60 or ys.max() - ys.min() >= 60:
            continue
        out.append({
            "component_id": comp.get("component_id", f"c{idx}"),
            "color": int(comp["raw_pixel_value"]),
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "center_xy": [float(xs.mean()), float(ys.mean())],
            "area": area,
            "mask": mask,
        })
    out.sort(key=lambda c: -c["area"])
    return out


def glove_ports(bbox_xyxy: list[int], component_id: str = "glove") -> list[dict]:
    """Portband port rows (per-side edge cell sets) for the glove bbox."""
    return _object_ports(
        statecon_id="synthesis_lab_catch_play",
        object_id=f"glove_{component_id}",
        object_role="actor",
        component_ids=[component_id],
        box=[int(v) for v in bbox_xyxy],
    )


def port_edge_cells(ports: list[dict]) -> np.ndarray:
    """All port edge cells as a [64, 64] bool mask (clipped to the grid)."""
    mask = np.zeros((GRID, GRID), dtype=bool)
    for port in ports:
        for x, y in port.get("edge_cells_xy") or []:
            if 0 <= x < GRID and 0 <= y < GRID:
                mask[y, x] = True
    return mask


def component_displacement(
    before: np.ndarray, after: np.ndarray, comp: dict
) -> tuple[float, float] | None:
    """Where did this component's color-mass move between frames?

    Returns (dx, dy) of the component-color centroid restricted to the
    neighborhood of the original bbox, or None if it did not move.
    """
    x0, y0, x1, y1 = comp["bbox_xyxy"]
    pad = 4
    ys = slice(max(0, y0 - pad), min(GRID, y1 + 1 + pad))
    xs = slice(max(0, x0 - pad), min(GRID, x1 + 1 + pad))
    color = comp["color"]
    b = np.asarray(before)[ys, xs] == color
    a = np.asarray(after)[ys, xs] == color
    if not b.any() or not a.any():
        return None
    by, bx = np.nonzero(b)
    ay, ax = np.nonzero(a)
    dx, dy = float(ax.mean() - bx.mean()), float(ay.mean() - by.mean())
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return None
    return dx, dy


def masks_adjoin(a: np.ndarray, b: np.ndarray) -> bool:
    """True when any cell of a is 4-adjacent to (or overlapping) a cell of b."""
    grown = a.copy()
    grown[1:, :] |= a[:-1, :]
    grown[:-1, :] |= a[1:, :]
    grown[:, 1:] |= a[:, :-1]
    grown[:, :-1] |= a[:, 1:]
    return bool((grown & b).any())


def directional_gap(
    glove_bbox: list[int], partner_mask: np.ndarray, direction: str
) -> int | None:
    """Empty cells between the glove's leading port edge and the partner
    along `direction` (up/down/left/right), within the glove's span band.
    0 = edges adjoin (PortBand alignment_error_px analogue, from pixels).
    None = no partner cells in the band."""
    x0, y0, x1, y1 = (int(v) for v in glove_bbox)
    if direction in ("left", "right"):
        band = partner_mask[y0:y1 + 1, :]
        cols = np.nonzero(band.any(axis=0))[0]
        if direction == "right":
            ahead = cols[cols > x1]
            return int(ahead.min() - x1 - 1) if len(ahead) else None
        ahead = cols[cols < x0]
        return int(x0 - ahead.max() - 1) if len(ahead) else None
    band = partner_mask[:, x0:x1 + 1]
    rows = np.nonzero(band.any(axis=1))[0]
    if direction == "down":
        ahead = rows[rows > y1]
        return int(ahead.min() - y1 - 1) if len(ahead) else None
    ahead = rows[rows < y0]
    return int(y0 - ahead.max() - 1) if len(ahead) else None


def catch_label(
    pour_frames: list[np.ndarray],
    glove_mask: np.ndarray,
    ports: list[dict],
    sides: tuple[str, ...] | None = None,
    max_frames: int | None = None,
) -> dict:
    """Hindsight label from pixels: did the pour touch the glove?

    contact = a cell that changed during the pour animation and lies on a
    Portband port edge cell, inside the glove, or 4-adjacent to it.

    Strict mode (behavioral probe): pass ``sides=("top",)`` and a
    ``max_frames`` limit so only paint landing on the glove's top edge
    during the early fall counts — the eventual liquid spread touching a
    side does not count as a catch.
    """
    if sides is not None:
        ports = [p for p in ports if p.get("side") in sides]
        glove_mask = np.zeros_like(glove_mask)
    if max_frames is not None:
        pour_frames = pour_frames[: max_frames + 1]
    touch_zone = glove_mask | port_edge_cells(ports)
    grown = touch_zone.copy()
    grown[1:, :] |= touch_zone[:-1, :]
    grown[:-1, :] |= touch_zone[1:, :]
    grown[:, 1:] |= touch_zone[:, :-1]
    grown[:, :-1] |= touch_zone[:, 1:]

    contact_cells = 0
    changed_total = 0
    for prev, nxt in zip(pour_frames, pour_frames[1:]):
        changed = np.asarray(prev) != np.asarray(nxt)
        changed_total += int(changed.sum())
        contact_cells += int((changed & grown).sum())
    return {
        "caught": bool(contact_cells > 0),
        "contact_cells": contact_cells,
        "pour_changed_cells": changed_total,
    }
