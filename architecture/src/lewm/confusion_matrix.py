"""Sparse confusion-matrix builder for the mechanic head.

Used to mine hard-negative pairs: mechanics that the trained model
confuses with each other (top-1 prediction lands on a different mechanic
in the same family) are exactly the pairs we want to push apart with a
contrastive loss.

The matrix is sparse — 3043 × 3043 ≈ 9.2M cells would be dense, but
the typical model confuses each true mechanic with a handful of others.
We accumulate (true_idx, pred_idx) → count tuples and serialise to JSON.

Streamed in chunks; no full ``(N, N)`` tensor materialised.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


@dataclass
class ConfusionMatrix:
    """Sparse confusion matrix keyed by (true, pred) mechanic indices.

    ``counts[(true_idx, pred_idx)]`` is the number of times the model
    predicted ``pred_idx`` when the true label was ``true_idx``.
    ``mechanic_names`` provides the index → registered-name mapping; the
    indices come from ``GoalPropertyDataset``'s ``game_idx`` field.
    """

    counts: Counter = field(default_factory=Counter)
    mechanic_names: list[str] = field(default_factory=list)
    total_examples: int = 0

    def add(self, true_idx: int, pred_idx: int, weight: int = 1) -> None:
        self.counts[(int(true_idx), int(pred_idx))] += int(weight)
        self.total_examples += int(weight)

    def add_batch(self, true_idxs: torch.Tensor, pred_idxs: torch.Tensor) -> None:
        ti = true_idxs.detach().cpu().long().view(-1).tolist()
        pi = pred_idxs.detach().cpu().long().view(-1).tolist()
        for t, p in zip(ti, pi, strict=True):
            self.counts[(t, p)] += 1
            self.total_examples += 1

    # ----- queries -----

    def row(self, true_idx: int) -> dict[int, int]:
        """Predictions distribution for a given true mechanic index."""
        return {
            pred: c for (t, pred), c in self.counts.items() if t == true_idx
        }

    def row_total(self, true_idx: int) -> int:
        return sum(c for (t, _), c in self.counts.items() if t == true_idx)

    def offdiagonal_pairs(
        self, *, min_count: int = 1,
    ) -> list[tuple[int, int, int]]:
        """All (true, pred, count) tuples where true != pred and count >= min_count.

        Sorted by count desc — the most-confused pairs first.
        """
        rows = [
            (t, p, c)
            for (t, p), c in self.counts.items()
            if t != p and c >= min_count
        ]
        rows.sort(key=lambda r: -r[2])
        return rows

    # ----- I/O -----

    def to_jsonable(self) -> dict:
        return {
            "schema": "lewm.confusion_matrix.v001",
            "total_examples": self.total_examples,
            "num_mechanics": len(self.mechanic_names),
            "mechanic_names": list(self.mechanic_names),
            "counts": [
                {"true": int(t), "pred": int(p), "count": int(c)}
                for (t, p), c in sorted(self.counts.items())
            ],
        }

    @classmethod
    def from_jsonable(cls, payload: dict) -> "ConfusionMatrix":
        cm = cls(mechanic_names=list(payload.get("mechanic_names", [])))
        for row in payload.get("counts", []):
            cm.counts[(int(row["true"]), int(row["pred"]))] = int(row["count"])
            cm.total_examples += int(row["count"])
        return cm

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_jsonable(), indent=2, sort_keys=True))
        return p

    @classmethod
    def load(cls, path: str | Path) -> "ConfusionMatrix":
        return cls.from_jsonable(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Eval-time builder
# ---------------------------------------------------------------------------


@torch.no_grad()
def build_confusion_matrix(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    mechanic_names: Iterable[str] | None = None,
    amp: str = "none",
    non_blocking: bool = False,
    active_mechanic_indices: torch.Tensor | None = None,
) -> ConfusionMatrix:
    """Walk a dataloader, predict mechanics, accumulate the confusion matrix.

    Requires the model to emit ``mechanic_logits`` in its forward output —
    GoalPropertyModel does so when ``num_mechanics > 0``.

    ``active_mechanic_indices``: when set, restrict prediction to the
    sub-vocab of active mechanic IDs (matches the existing
    ``_masked_mechanic_predictions`` flow in goal_property.py).
    """
    model.eval()
    amp_enabled = amp != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    cm = ConfusionMatrix(mechanic_names=list(mechanic_names) if mechanic_names else [])

    for batch in loader:
        cf = batch["context_frames"].to(device, non_blocking=non_blocking)
        ca = batch["context_actions"].to(device, non_blocking=non_blocking)
        true_ids = batch["mechanic"].to(device, non_blocking=non_blocking)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(cf, ca)
        logits = outputs.get("mechanic_logits")
        if logits is None:
            raise RuntimeError(
                "build_confusion_matrix requires a model that emits mechanic_logits"
            )
        if active_mechanic_indices is not None:
            idxs = active_mechanic_indices.to(logits.device)
            sub_logits = logits.index_select(1, idxs)
            sub_pred = sub_logits.argmax(dim=1)
            preds = idxs[sub_pred]
        else:
            preds = logits.argmax(dim=1)
        cm.add_batch(true_ids, preds)

    return cm
