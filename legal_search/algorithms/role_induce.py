"""Induce color roles from transitions — replaces hardcoded SP80 palettes."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from legal_search.types import Transition


def induce_movable_colors(transitions: Iterable[Transition], *, min_trials: int = 2) -> dict[int, float]:
    """Colors whose mass translates under move actions (1-4) score as movable."""
    move_ids = {1, 2, 3, 4}
    scores: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for tr in transitions:
        if tr.action.action_id not in move_ids:
            continue
        before = np.asarray(tr.frame_before)
        after = np.asarray(tr.frame_after)
        if before.shape != after.shape:
            continue
        # colors that lost cells in one place and gained elsewhere
        for color in np.unique(before):
            c = int(color)
            bmask = before == c
            amask = after == c
            lost = int(np.count_nonzero(bmask & ~amask))
            gained = int(np.count_nonzero(amask & ~bmask))
            scores[c][1] += 1.0
            if lost > 0 and gained > 0:
                scores[c][0] += 1.0
    out: dict[int, float] = {}
    for c, (hits, trials) in scores.items():
        if trials >= min_trials:
            out[c] = hits / trials
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def induce_static_colors(transitions: Iterable[Transition], *, min_trials: int = 3) -> dict[int, float]:
    """Colors that almost never change under any action → static scenery prior."""
    scores: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for tr in transitions:
        before = np.asarray(tr.frame_before)
        after = np.asarray(tr.frame_after)
        if before.shape != after.shape:
            continue
        for color in np.unique(before):
            c = int(color)
            bmask = before == c
            changed = int(np.count_nonzero((before != after) & bmask))
            scores[c][1] += 1.0
            if changed == 0:
                scores[c][0] += 1.0
    out: dict[int, float] = {}
    for c, (stable, trials) in scores.items():
        if trials >= min_trials:
            out[c] = stable / trials
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def induce_click_sensitive_cells(
    transitions: Iterable[Transition],
    *,
    grid: int = 64,
) -> list[dict]:
    """Cells where ACTION6 often caused a frame change — generic saliency."""
    bins: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])
    for tr in transitions:
        if tr.action.action_id != 6 or tr.action.x is None or tr.action.y is None:
            continue
        key = (int(tr.action.x), int(tr.action.y))
        bins[key][1] += 1.0
        bins[key][0] += 1.0 if tr.frame_changed else 0.0
    rows = []
    for (x, y), (ch, n) in bins.items():
        if n <= 0:
            continue
        rows.append({"x": x, "y": y, "change_rate": ch / n, "trials": n})
    rows.sort(key=lambda r: (-r["change_rate"], -r["trials"]))
    return rows[:32]
