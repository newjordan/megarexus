"""Family-contrastive pair mining from the capability table.

Different from hard-negative mining (which is data-driven, from a
confusion matrix), family-contrastive mining is *structure-driven*: we
look at every pair of mechanics that share a goal_family but differ on
at least one capability flag. The hypothesis is that within-family
discrimination is the structural gap a contrastive loss should close.

The two miners produce the same ``ContrastivePair`` shape, so a single
``ContrastivePairRegistry`` (later) can consume both.
"""

from __future__ import annotations

import json
from pathlib import Path

from lewm.hard_negative_mining import (
    SOURCE_FAMILY_CONTRASTIVE,
    ContrastivePair,
    save_pairs,
)
from lewm.mechanic_capability_table import (
    MechanicProfile,
    capability_hamming,
    hamming_margin,
    group_by_goal_family,
)


def mine_family_contrastive_pairs(
    capability_table: dict[str, MechanicProfile],
    *,
    base_margin: float = 1.0,
    min_hamming: float = 1.0,
    max_pairs_per_family: int | None = None,
    include_unknown_family: bool = False,
) -> list[ContrastivePair]:
    """Emit ContrastivePair records for every within-family mechanic pair
    whose capability_hamming >= ``min_hamming``.

    ``max_pairs_per_family`` caps the explosion when a family is large
    (e.g., 200+ tier2 mechanics in ``reach`` would produce 19,900 pairs).
    When set, pairs are sorted by capability_hamming descending and
    truncated.
    """
    out: list[ContrastivePair] = []
    buckets = group_by_goal_family(capability_table)
    for family, profs in buckets.items():
        if family == "__unknown__" and not include_unknown_family:
            continue
        family_pairs: list[ContrastivePair] = []
        for i in range(len(profs)):
            for j in range(i + 1, len(profs)):
                ham = capability_hamming(profs[i], profs[j])
                if ham < min_hamming:
                    continue
                margin = hamming_margin(profs[i], profs[j], base_margin=base_margin)
                family_pairs.append(ContrastivePair(
                    mechanic_a=profs[i].mechanic,
                    mechanic_b=profs[j].mechanic,
                    margin=margin,
                    confusion_count=0,
                    source=SOURCE_FAMILY_CONTRASTIVE,
                    capability_hamming=ham,
                ))
        family_pairs.sort(key=lambda p: -p.capability_hamming)
        if max_pairs_per_family is not None:
            family_pairs = family_pairs[:max_pairs_per_family]
        out.extend(family_pairs)
    return out


def merge_pair_lists(
    *pair_lists: list[ContrastivePair],
) -> list[ContrastivePair]:
    """Deduplicate by unordered ``(mechanic_a, mechanic_b)``.

    When the same pair shows up in multiple sources (hard-negative AND
    family-contrastive), the version with the *larger* margin and the
    hard-negative source wins — hard-negatives are evidence-backed.
    """
    by_key: dict[tuple[str, str], ContrastivePair] = {}
    for src in pair_lists:
        for pair in src:
            key = tuple(sorted((pair.mechanic_a, pair.mechanic_b)))
            cur = by_key.get(key)
            if cur is None:
                by_key[key] = pair
                continue
            # Prefer hard-negative source over family-contrastive on tie.
            if pair.source == "hard_negative" and cur.source != "hard_negative":
                by_key[key] = pair
            elif pair.source == cur.source and pair.margin > cur.margin:
                by_key[key] = pair
    return list(by_key.values())
