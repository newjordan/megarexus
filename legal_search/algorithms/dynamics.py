"""Imagination / dynamics hypothesis interface — mined from sim3d imagination_kernel.

KEEP: plug-in DynamicsHypothesis, roll-forward, score by observed transitions.
STRIP: shipped SP80 r4/r5 JSON, fixture win banks, hardcoded paint/lava/mouth roles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from legal_search.algorithms.components import frame_delta_count, frame_changed
from legal_search.types import Action, Proposal, Transition


# ARC action → grid delta (x, y) convention used across legal_search
MOVE_DELTAS: dict[int, tuple[int, int]] = {
    1: (0, -1),  # up
    2: (0, 1),  # down
    3: (-1, 0),  # left
    4: (1, 0),  # right
}


@dataclass
class GridState:
    """Legal SimState stand-in: raw grid only, no role maps."""

    grid: np.ndarray
    step: int = 0

    def clone(self) -> "GridState":
        return GridState(grid=np.asarray(self.grid).copy(), step=self.step)

    def signature(self) -> tuple:
        g = np.asarray(self.grid)
        return (int(self.step), int(g.sum()), int(g.max()) if g.size else 0, g.shape)


class DynamicsHypothesis(Protocol):
    name: str

    def apply(self, state: GridState, action: Action) -> GridState: ...


@dataclass
class TranslateColorBlob:
    """If action is a move, shift the most-recently-moved color by one cell.

    Induced color from transitions when available; else largest non-bg component.
    """

    name: str = "translate_color_blob"
    color: int | None = None

    def apply(self, state: GridState, action: Action) -> GridState:
        out = state.clone()
        out.step += 1
        delta = MOVE_DELTAS.get(int(action.action_id))
        if delta is None or self.color is None:
            return out
        g = out.grid
        h, w = g.shape[:2]
        dx, dy = delta
        mask = g == int(self.color)
        if not np.any(mask):
            return out
        ys, xs = np.nonzero(mask)
        # clear then paint shifted (clip bounds — cells that leave are lost)
        g[mask] = 0  # temporary; restore bg via mode of non-color later
        bg = 12
        # estimate bg from non-mask cells if available
        others = g[~mask]
        if others.size:
            vals, counts = np.unique(others, return_counts=True)
            bg = int(vals[int(np.argmax(counts))])
        g[mask] = bg
        for y, x in zip(ys, xs):
            ny, nx = int(y) + dy, int(x) + dx
            if 0 <= ny < h and 0 <= nx < w:
                g[ny, nx] = int(self.color)
        return out


@dataclass
class NoOpDynamics:
    name: str = "noop"

    def apply(self, state: GridState, action: Action) -> GridState:
        out = state.clone()
        out.step += 1
        return out


@dataclass
class ClickPaintProbe:
    """ACTION6: if click lands on background, paint a 1-cell mark of max(color)+1.

    Pure exploratory dynamics hypothesis — not a game answer.
    """

    name: str = "click_paint_probe"

    def apply(self, state: GridState, action: Action) -> GridState:
        out = state.clone()
        out.step += 1
        if int(action.action_id) != 6 or action.x is None or action.y is None:
            return out
        g = out.grid
        h, w = g.shape[:2]
        x, y = int(action.x), int(action.y)
        if not (0 <= y < h and 0 <= x < w):
            return out
        mark = int(min(15, int(g.max()) + 1)) if g.size else 1
        g[y, x] = mark
        return out


def induce_blob_color(transitions: list[Transition]) -> int | None:
    from legal_search.algorithms.role_induce import induce_movable_colors

    rates = induce_movable_colors(transitions, min_trials=1)
    if not rates:
        return None
    return int(max(rates.items(), key=lambda kv: kv[1])[0])


def score_hypothesis_on_transitions(
    hyp: DynamicsHypothesis,
    transitions: list[Transition],
) -> dict[str, Any]:
    """How well does hyp.predict match observed next frames?"""
    if not transitions:
        return {"name": getattr(hyp, "name", "?"), "n": 0, "mean_overlap": 0.0, "exact_frac": 0.0}
    overlaps = []
    exact = 0
    for tr in transitions[-24:]:
        st = GridState(grid=np.asarray(tr.frame_before).copy())
        pred = hyp.apply(st, tr.action)
        after = np.asarray(tr.frame_after)
        pred_g = np.asarray(pred.grid)
        if pred_g.shape != after.shape:
            overlaps.append(0.0)
            continue
        same = float(np.mean(pred_g == after))
        overlaps.append(same)
        if np.array_equal(pred_g, after):
            exact += 1
    return {
        "name": getattr(hyp, "name", type(hyp).__name__),
        "n": len(overlaps),
        "mean_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
        "exact_frac": exact / max(1, len(overlaps)),
    }


def roll_forward(
    frame: np.ndarray,
    program: list[Action],
    hyp: DynamicsHypothesis,
) -> list[np.ndarray]:
    st = GridState(grid=np.asarray(frame).copy())
    frames = [st.grid.copy()]
    for a in program:
        st = hyp.apply(st, a)
        frames.append(st.grid.copy())
    return frames


def mine_dynamics(
    frame: np.ndarray,
    transitions: list[Transition] | None = None,
    *,
    available_actions: list[int] | None = None,
) -> dict[str, Any]:
    """Rank legal dynamics hypotheses; propose programs under best hyp.

    Never loads fixture win banks or shipped SP80 JSON.
    """
    trs = list(transitions or [])
    color = induce_blob_color(trs)
    hyps: list[DynamicsHypothesis] = [
        NoOpDynamics(),
        TranslateColorBlob(color=color),
        ClickPaintProbe(),
    ]
    scores = [score_hypothesis_on_transitions(h, trs) for h in hyps]
    scores.sort(key=lambda r: (-r["mean_overlap"], -r["exact_frac"]))
    best_name = scores[0]["name"] if scores else "noop"
    best_hyp = next((h for h in hyps if getattr(h, "name", "") == best_name), hyps[0])

    # short programs: single-step of high-change actions under best hyp
    avail = available_actions or [1, 2, 3, 4, 5, 6]
    proposals: list[Proposal] = []
    base = GridState(grid=np.asarray(frame).copy())
    for aid in avail:
        if aid == 6:
            # probe a few centers
            h, w = np.asarray(frame).shape[:2]
            for x, y in ((w // 2, h // 2), (w // 4, h // 4), (3 * w // 4, 3 * h // 4)):
                a = Action(6, x=x, y=y, source="dynamics.imagine")
                nxt = best_hyp.apply(base.clone(), a)
                d = frame_delta_count(base.grid, nxt.grid)
                proposals.append(
                    Proposal(
                        actions=[a],
                        score=float(d) / float(max(1, base.grid.size)),
                        kind="imagine_program",
                        evidence={"hyp": best_name, "delta_pixels": d},
                        algorithm_id="search.imagine_then_rank.v1",
                    )
                )
        else:
            a = Action(aid, source="dynamics.imagine")
            nxt = best_hyp.apply(base.clone(), a)
            d = frame_delta_count(base.grid, nxt.grid)
            proposals.append(
                Proposal(
                    actions=[a],
                    score=float(d) / float(max(1, base.grid.size)),
                    kind="imagine_program",
                    evidence={"hyp": best_name, "delta_pixels": d},
                    algorithm_id="search.imagine_then_rank.v1",
                )
            )
    proposals.sort(key=lambda p: -p.score)
    return {
        "hypotheses": scores,
        "best_hypothesis": best_name,
        "induced_blob_color": color,
        "proposals": proposals[:8],
        "fixture_priors_loaded": False,
        "shipped_dynamics_json_loaded": False,
        "n_transitions_scored": len(trs),
        "interface": "DynamicsHypothesis.apply(GridState, Action) -> GridState",
    }
