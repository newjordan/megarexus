"""Hard-negative pair mining from a confusion matrix + capability table.

Pairs of mechanics that the trained model confuses (off-diagonal cells
with high count in the confusion matrix) are exactly the pairs whose
trunk latents should be pushed apart by a contrastive hinge — provided
the two mechanics actually differ on capability/role/topology axes that
the model could learn from.

The margin for each pair is scaled by capability_hamming so that
mechanics that look identical in study_tags (small hamming) don't get
pushed apart on noise.

Output is a JSON list of ``ContrastivePair`` records, serialisable and
replayable across training runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from lewm.confusion_matrix import ConfusionMatrix
from lewm.mechanic_capability_table import (
    MechanicProfile,
    capability_hamming,
    hamming_margin,
)


SOURCE_HARD_NEGATIVE = "hard_negative"
SOURCE_FAMILY_CONTRASTIVE = "family_contrastive"


@dataclass
class ContrastivePair:
    """A pair of mechanics that should be pulled apart in trunk latent space.

    ``margin`` is the hinge margin to use in the contrastive loss.
    Larger margin = pushed apart harder. ``confusion_count`` is the
    raw confusion-matrix cell value (0 for pairs sourced from
    family-contrastive mining instead of confusion).
    """

    mechanic_a: str
    mechanic_b: str
    margin: float
    confusion_count: int = 0
    source: str = SOURCE_HARD_NEGATIVE
    capability_hamming: float = 0.0

    def to_jsonable(self) -> dict:
        return {
            "mechanic_a": self.mechanic_a,
            "mechanic_b": self.mechanic_b,
            "margin": float(self.margin),
            "confusion_count": int(self.confusion_count),
            "source": self.source,
            "capability_hamming": float(self.capability_hamming),
        }

    @classmethod
    def from_jsonable(cls, row: dict) -> "ContrastivePair":
        return cls(
            mechanic_a=str(row["mechanic_a"]),
            mechanic_b=str(row["mechanic_b"]),
            margin=float(row["margin"]),
            confusion_count=int(row.get("confusion_count", 0)),
            source=str(row.get("source", SOURCE_HARD_NEGATIVE)),
            capability_hamming=float(row.get("capability_hamming", 0.0)),
        )


def mine_hard_negative_pairs(
    confusion: ConfusionMatrix,
    capability_table: dict[str, MechanicProfile],
    *,
    min_confusion_count: int = 5,
    base_margin: float = 1.0,
    require_same_family: bool = True,
    require_nonzero_hamming: bool = True,
    top_k_per_mechanic: int | None = None,
) -> list[ContrastivePair]:
    """Convert confusion-matrix off-diagonal cells into ContrastivePair records.

    Filters:
      - ``min_confusion_count``: drop pairs with fewer raw confusions
      - ``require_same_family``: only emit when both mechanics share goal_family
      - ``require_nonzero_hamming``: only emit when capability_hamming > 0
        (no point pushing apart mechanics with identical capabilities)
      - ``top_k_per_mechanic``: cap how many partners each mechanic
        contributes (keeps the registry from being dominated by
        a single ambiguous mechanic)
    """
    names = confusion.mechanic_names
    if not names:
        raise ValueError("ConfusionMatrix has no mechanic_names; mining requires the table")

    out: list[ContrastivePair] = []
    by_a: dict[str, list[ContrastivePair]] = {}
    for t_idx, p_idx, count in confusion.offdiagonal_pairs(min_count=min_confusion_count):
        if t_idx >= len(names) or p_idx >= len(names):
            continue
        a, b = names[t_idx], names[p_idx]
        prof_a = capability_table.get(a)
        prof_b = capability_table.get(b)
        if prof_a is None or prof_b is None:
            continue
        if require_same_family and prof_a.goal_family != prof_b.goal_family:
            continue
        ham = capability_hamming(prof_a, prof_b)
        if require_nonzero_hamming and ham <= 0.0:
            continue
        margin = hamming_margin(prof_a, prof_b, base_margin=base_margin)
        pair = ContrastivePair(
            mechanic_a=a, mechanic_b=b, margin=margin,
            confusion_count=count, source=SOURCE_HARD_NEGATIVE,
            capability_hamming=ham,
        )
        by_a.setdefault(a, []).append(pair)

    for a, partners in by_a.items():
        partners.sort(key=lambda p: -p.confusion_count)
        if top_k_per_mechanic is not None:
            partners = partners[:top_k_per_mechanic]
        out.extend(partners)

    return out


def save_pairs(pairs: list[ContrastivePair], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(
        {
            "schema": "lewm.contrastive_pairs.v001",
            "pairs": [pair.to_jsonable() for pair in pairs],
        },
        indent=2, sort_keys=True,
    ))
    return p


def load_pairs(path: str | Path) -> list[ContrastivePair]:
    payload = json.loads(Path(path).read_text())
    return [ContrastivePair.from_jsonable(row) for row in payload.get("pairs", [])]
