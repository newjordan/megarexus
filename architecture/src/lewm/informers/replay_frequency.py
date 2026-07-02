"""ReplayFrequencyInformer — bias toward actions seen in past successful sequences.

Connection: the user's ``scripts/arcprize_replay_sequence_bank.py`` outputs
per-game "best-known sequence banks" — actions that historically led to
``WIN`` or ``level++`` on that game. This informer counts action occurrences
in those banks (per game_id) and emits a logit proportional to the count.

When no bank entry exists for the current game_id, the informer returns
zero logits (uniform passthrough). Cost: O(|bank entries for this game|)
on first call per game_id; cached thereafter for the lifetime of the
informer instance.

Bank format expected (flexible):
  - ``dict[str, list[list[int]]]``: game_id → list of action_id sequences
  - ``dict[str, list[int]]``: game_id → flat list of action_ids
"""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Optional, Union

import torch

from lewm.imagination import Informer, InformerContext


BankType = Union[
    Mapping[str, list[list[int]]],
    Mapping[str, list[int]],
]


def _flatten_bank_for_game(bank: BankType, game_id: str) -> list[int]:
    entry = bank.get(game_id) if bank is not None else None
    if entry is None:
        return []
    if not entry:
        return []
    first = entry[0]
    if isinstance(first, (list, tuple)):
        # list[list[int]] form: concat all sequences
        out: list[int] = []
        for seq in entry:
            out.extend(int(x) for x in seq)
        return out
    # list[int] form
    return [int(x) for x in entry]


def ReplayFrequencyInformer(
    *,
    bank: Optional[BankType] = None,
    log_scale: bool = True,
    weight: float = 1.0,
    enabled: bool = True,
) -> Informer:
    """Factory: returns an Informer that scores actions by their past frequency.

    ``log_scale=True`` (default): logit = log(1 + count). Avoids extreme
    dominance when one action is wildly over-represented. ``False`` uses
    raw count.
    """

    cache: dict[str, torch.Tensor] = {}

    def _per_game_counts(game_id: Optional[str], avail: list[int]) -> torch.Tensor:
        if game_id is None or bank is None:
            return torch.zeros(len(avail), dtype=torch.float32)
        ck = f"{game_id}::{len(avail)}::{','.join(map(str, avail))}"
        cached = cache.get(ck)
        if cached is not None:
            return cached
        flat = _flatten_bank_for_game(bank, game_id)
        if not flat:
            out = torch.zeros(len(avail), dtype=torch.float32)
            cache[ck] = out
            return out
        counts = Counter(flat)
        if log_scale:
            scores = [float(torch.log1p(torch.tensor(float(counts.get(a, 0))))) for a in avail]
        else:
            scores = [float(counts.get(a, 0)) for a in avail]
        out = torch.tensor(scores, dtype=torch.float32)
        # Center to zero mean so logits aren't biased high overall.
        out = out - out.mean()
        cache[ck] = out
        return out

    def fn(ctx: InformerContext, avail: list[int]) -> torch.Tensor:
        return _per_game_counts(ctx.game_id, avail)

    return Informer(name="replay_frequency", fn=fn, weight=weight, enabled=enabled)


__all__ = ["ReplayFrequencyInformer"]
