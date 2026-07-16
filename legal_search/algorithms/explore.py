"""Online change-explorer — algorithm mined from Stochastic Goose.

Legal core: predict / prefer actions that change the current frame.
No game win knowledge. Memory is online counts only.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from legal_search.types import Action


class ChangeExplorer:
    """Lightweight tabular stand-in for Goose's action-change learner.

    Full Goose CNN remains available via consume/goose.py when the external
    runtime is present. This class is the *algorithm contract* with zero deps.
    """

    algorithm_id = "explore.change_by_repetition.v1"

    def __init__(self, *, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng(0)
        # (action_id) -> [changes, trials]
        self.stats: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
        # click cell coarse bins 8x8 -> stats
        self.click_stats: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0])

    def observe(self, action: Action, changed: bool) -> None:
        aid = int(action.action_id)
        self.stats[aid][1] += 1.0
        self.stats[aid][0] += 1.0 if changed else 0.0
        if aid == 6 and action.x is not None and action.y is not None:
            key = (int(action.x) // 8, int(action.y) // 8)
            self.click_stats[key][1] += 1.0
            self.click_stats[key][0] += 1.0 if changed else 0.0

    def _rate(self, pair: list[float], prior: float = 0.5) -> float:
        c, n = pair
        # Beta(1,1) smoothed
        return (c + prior) / (n + 1.0)

    def propose(
        self,
        available: Iterable[int],
        *,
        grid: int = 64,
        temperature: float = 1.0,
    ) -> Action:
        avail = [int(a) for a in available]
        if not avail:
            return Action(1, source=self.algorithm_id)
        weights = []
        for aid in avail:
            if aid == 6:
                # average best click bin rate with action-6 rate
                base = self._rate(self.stats[6])
                if self.click_stats:
                    best = max(self._rate(v) for v in self.click_stats.values())
                    weights.append(0.5 * base + 0.5 * best)
                else:
                    weights.append(base)
            else:
                weights.append(self._rate(self.stats[aid]))
        w = np.asarray(weights, dtype=np.float64)
        w = np.power(np.clip(w, 1e-6, None), 1.0 / max(temperature, 1e-3))
        w /= w.sum()
        aid = int(self.rng.choice(avail, p=w))
        if aid != 6:
            return Action(aid, source=self.algorithm_id)
        # sample click: prefer high-rate bins, else uniform
        if self.click_stats:
            bins = list(self.click_stats.keys())
            rates = np.array([self._rate(self.click_stats[b]) for b in bins], dtype=np.float64)
            rates = np.power(np.clip(rates, 1e-6, None), 1.0 / max(temperature, 1e-3))
            rates /= rates.sum()
            bx, by = bins[int(self.rng.choice(len(bins), p=rates))]
            x = int(self.rng.integers(bx * 8, min(grid, bx * 8 + 8)))
            y = int(self.rng.integers(by * 8, min(grid, by * 8 + 8)))
        else:
            x = int(self.rng.integers(0, grid))
            y = int(self.rng.integers(0, grid))
        return Action(6, x=x, y=y, source=self.algorithm_id)

    def snapshot(self) -> dict:
        return {
            "algorithm_id": self.algorithm_id,
            "action_stats": {str(k): {"changes": v[0], "trials": v[1]} for k, v in self.stats.items()},
            "click_bins": {
                f"{k[0]}_{k[1]}": {"changes": v[0], "trials": v[1]} for k, v in self.click_stats.items()
            },
        }
