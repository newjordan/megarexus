"""Base consume adapter protocol."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from legal_search.contract import ConsumeReceipt, TRUTH_BOUNDARY
from legal_search.types import LegalPacket, Transition


class ConsumeAdapter(ABC):
    system_id: str
    algorithm_id: str
    source_paths: tuple[str, ...] = ()
    stripped_bindings: tuple[str, ...] = ()

    def receipt(self, **kwargs: Any) -> ConsumeReceipt:
        return ConsumeReceipt(
            system_id=self.system_id,
            algorithm_id=self.algorithm_id,
            truth_boundary=dict(TRUTH_BOUNDARY),
            stripped_bindings=list(self.stripped_bindings),
            allowed_outputs=list(kwargs.get("allowed_outputs") or []),
            banned_outputs=list(kwargs.get("banned_outputs") or []),
            notes=str(kwargs.get("notes") or ""),
            legal=bool(kwargs.get("legal", True)),
            violations=list(kwargs.get("violations") or []),
        )

    @abstractmethod
    def consume(
        self,
        frame: np.ndarray,
        *,
        game_id: str = "",
        transitions: list[Transition] | None = None,
        available_actions: list[int] | None = None,
        levels_completed: int = 0,
    ) -> LegalPacket:
        ...
