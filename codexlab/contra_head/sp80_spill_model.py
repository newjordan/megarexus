"""SP80 spill forward-model — a faithful, FREE reimplementation of the engine's
ACTION5 spill, so imaginationland can predict an outcome without spending fuel.

Mechanic (verified against environment/environment_files/sp80/589a99af/sp80.py,
class Sp80, the "spill" branch of step()): ACTION5 releases liquid (color-6
`liolfvkveqg` source drops + color-4 `sowlljgtjvn` platform/marker holes) which
falls under gravity, spreads across platforms (`plzwjbfyfli`) and bucket lips
(`repwkzbkhxl`), is redirected by diverters (`tuvkdkhdokr`), and is caught by
buckets. A level is WON iff after the spill EVERY bucket is filled AND no liquid
ever reached a boundary wall (`waoewejnqzc`, the overflow marker).

This module reads the engine's *sprite-level* state (positions/tags/shapes, not
pixels — which sidesteps the per-level board rotation `fahhoimkk`) and replays the
spill cellular automaton tick-for-tick, including the engine's exact
get_sprite_at semantics (layer-desc, stable insertion order, PIXEL_PERFECT so the
cup's center-top gap is passable). See [[sp80-engine-verified-mechanic]].

Boundary: this is a prediction, validated against the live engine in
codexlab/sp80_spill_model_validate.py. It is not a solve.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SPILL_MODEL_SCHEMA = "codexlab.sp80_spill_forward.v1"

# Sprite-tag -> occupant kind. Order of checks mirrors the engine's spill branch.
_BUCKET = "repwkzbkhxl"
_PLATFORM = "plzwjbfyfli"
_WALL = "waoewejnqzc"
_DIVERTER = "tuvkdkhdokr"
_SOURCE = "liolfvkveqg"
_HOLE = "sowlljgtjvn"


def _classify(tags: list[str]) -> str:
    if _BUCKET in tags:
        return "bucket"
    if _PLATFORM in tags:
        return "platform"
    if _WALL in tags:
        return "wall"
    if _DIVERTER in tags:
        return "diverter"
    if _SOURCE in tags:
        return "source"
    if _HOLE in tags:
        return "marker"
    return "border"


def extract_state(game: Any) -> dict[str, Any]:
    """Read sprite-level spill state from a live Sp80 game (e.g. env._game).

    Returns an ordered occupant list preserving the engine's _sprites order (which
    is its get_sprite_at tie-break). Each occupant carries its absolute occupied
    cells and any color-4 hole cells, computed from sprite.render() (rotation-aware
    at the sprite level)."""
    level = game.current_level
    gw, gh = level.grid_size
    occupants = []
    for idx, s in enumerate(level.get_sprites()):
        px = np.asarray(s.render())
        if px.ndim != 2:
            px = px.reshape(px.shape[-2], px.shape[-1])
        h, w = px.shape
        occ, holes = set(), set()
        for r in range(h):
            for c in range(w):
                v = int(px[r, c])
                if v == -1:
                    continue
                occ.add((s.x + c, s.y + r))
                if v == 4:
                    holes.add((s.x + c, s.y + r))
        occupants.append({
            "idx": idx,
            "kind": _classify(list(s.tags)),
            "name": s.name,
            "x": int(s.x), "y": int(s.y),
            "occ": occ,
            "holes": holes,
            "movable": (_PLATFORM in s.tags) or (_DIVERTER in s.tags),
        })
    return {
        "schema": SPILL_MODEL_SCHEMA,
        "grid": (int(gw), int(gh)),
        "rotation_k": int(getattr(game, "fahhoimkk", 0) or 0),
        "occupants": occupants,
    }


def _shift_occupant(o: dict[str, Any], nx: int, ny: int) -> dict[str, Any]:
    dx, dy = nx - o["x"], ny - o["y"]
    return {**o,
            "x": nx, "y": ny,
            "occ": {(x + dx, y + dy) for (x, y) in o["occ"]},
            "holes": {(x + dx, y + dy) for (x, y) in o["holes"]}}


def simulate_spill(state: dict[str, Any],
                   platform_positions: dict[int, tuple[int, int]] | None = None,
                   *, max_ticks: int = 4096) -> dict[str, Any]:
    """Replay the spill. Returns predicted filled bucket idxs, overflow, win.

    platform_positions: optional {occupant_idx: (x, y)} overriding movable-sprite
    positions for a hypothetical layout. Omitted occupants keep their state position.
    """
    occupants = []
    for o in state["occupants"]:
        if platform_positions and o["idx"] in platform_positions:
            nx, ny = platform_positions[o["idx"]]
            occupants.append(_shift_occupant(o, int(nx), int(ny)))
        else:
            occupants.append(o)

    bucket_idxs = {o["idx"] for o in occupants if o["kind"] == "bucket"}
    # Static occupancy: cell -> occupant (earliest in order wins, matching the
    # engine's stable layer-0 sort).
    static_at: dict[tuple[int, int], dict[str, Any]] = {}
    for o in occupants:
        for cell in o["occ"]:
            static_at.setdefault(cell, o)

    liquid: set[tuple[int, int]] = set()  # dynamically-added drops (all layer 0)

    def sprite_at(x: int, y: int):
        """Mirror level.get_sprite_at: statics (insertion order) then liquid."""
        o = static_at.get((x, y))
        if o is not None:
            return ("occ", o)
        if (x, y) in liquid:
            return ("liquid", None)
        return (None, None)

    def perp(dx: int, dy: int):
        if dy != 0:
            obz, unv = -1, 1
            vww = kgh = 0
        else:
            obz = unv = 0
            vww, kgh = -1, 1
        return obz, unv, vww, kgh, [(obz, vww), (unv, kgh)]

    # --- initial fronts (vdwhttyyfq) ---
    fronts: list[tuple[int, int, int, int]] = []  # (x, y, dx, dy)
    for o in occupants:
        if o["kind"] == "source":
            for (x, y) in o["occ"]:
                fronts.append((x, y, 0, 1))
    for o in occupants:
        for (hx, hy) in o["holes"]:
            if sprite_at(hx, hy + 1)[0] is None:
                liquid.add((hx, hy + 1))
                fronts.append((hx, hy + 1, 0, 1))

    filled: set[int] = set()
    overflow = False

    def add_liquid(x, y):
        liquid.add((x, y))

    ticks = 0
    while fronts and ticks < max_ticks:
        ticks += 1
        liquid_before = len(liquid)
        filled_before = len(filled)
        overflow_before = overflow
        nxt: list[tuple[int, int, int, int]] = []
        seen_next: set[tuple[int, int, int, int]] = set()

        def push(x, y, dx, dy):
            key = (x, y, dx, dy)
            if key not in seen_next:
                seen_next.add(key)
                nxt.append(key)

        for (tx, ty, dx, dy) in fronts:
            obz, unv, vww, kgh, spread = perp(dx, dy)
            ax, ay = tx + dx, ty + dy
            kind, ahead = sprite_at(ax, ay)

            if kind is None:
                add_liquid(ax, ay)
                push(ax, ay, dx, dy)
                continue
            if kind == "liquid":
                push(ax, ay, dx, dy)
                continue
            # kind == "occ": dispatch on the occupant's category
            cat = ahead["kind"]
            if cat == "platform":
                for ox, oy in spread:
                    if sprite_at(tx + ox, ty + oy)[0] is None:
                        add_liquid(tx + ox, ty + oy)
                        push(tx + ox, ty + oy, dx, dy)
                continue
            if cat == "bucket":
                lk, lo = sprite_at(tx + obz, ty + vww)
                rk, ro = sprite_at(tx + unv, ty + kgh)
                if lk == "occ" and rk == "occ" and lo is ahead and ro is ahead:
                    filled.add(ahead["idx"])
                    continue
                for ox, oy in spread:
                    if sprite_at(tx + ox, ty + oy)[0] is None:
                        add_liquid(tx + ox, ty + oy)
                        push(tx + ox, ty + oy, dx, dy)
                continue
            if cat == "diverter":
                lk, lo = sprite_at(tx + obz, ty + vww)
                rk, ro = sprite_at(tx + unv, ty + kgh)
                redirected = False
                if lk == "occ" and lo is ahead and rk is None:
                    ndx, ndy = dy, -dx
                    add_liquid(tx + ndx, ty + ndy)
                    push(tx + ndx, ty + ndy, ndx, ndy)
                    redirected = True
                if rk == "occ" and ro is ahead and lk is None:
                    ndx, ndy = -dy, dx
                    add_liquid(tx + ndx, ty + ndy)
                    push(tx + ndx, ty + ndy, ndx, ndy)
                    redirected = True
                if not redirected:
                    for ox, oy in spread:
                        if sprite_at(tx + ox, ty + oy)[0] is None:
                            add_liquid(tx + ox, ty + oy)
                            push(tx + ox, ty + oy, dx, dy)
                continue
            if cat == "wall":
                overflow = True
                continue
            # border / marker / source: liquid dies silently (no engine branch)
            continue

        fronts = nxt
        # Fixpoint guard: a tick that changes nothing terminal cannot make progress.
        if (len(liquid) == liquid_before and len(filled) == filled_before
                and overflow == overflow_before):
            break

    win = bool(bucket_idxs) and filled >= bucket_idxs and not overflow
    return {
        "schema": SPILL_MODEL_SCHEMA,
        "filled_bucket_idxs": sorted(filled),
        "bucket_idxs": sorted(bucket_idxs),
        "overflow": overflow,
        "win": win,
        "ticks": ticks,
        "terminated": not fronts,
    }


__all__ = ["extract_state", "simulate_spill", "SPILL_MODEL_SCHEMA"]
