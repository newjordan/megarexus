"""Trunk-latent contrastive loss + batch-augmentation registry.

Consumed by ``PropertyTrainer.train_step`` when ``--contrastive-pairs``
is supplied to the training CLI. Two sources feed the same loss:

  - hard-negative pairs mined from a confusion matrix
  - family-contrastive pairs mined from the capability table

Both produce ``ContrastivePair`` records. This module is the runtime
view: load the pair list, group pair instances by mechanic, and let
training inject auxiliary episodes drawn from the partner mechanics so
the trunk latents can be pulled-apart with a hinge loss.

The loss itself is the standard contrastive hinge:

    L = mean( max(0, margin - ||z_a - z_b||_2 )^2 )

where ``z_a, z_b`` are trunk latents of the two mechanics' episodes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from lewm.hard_negative_mining import ContrastivePair, load_pairs


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def contrastive_trunk_loss(
    latents_a: torch.Tensor,
    latents_b: torch.Tensor,
    margin: torch.Tensor | float,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Hinge loss on trunk latents.

    ``latents_a``, ``latents_b`` are ``[N, D]`` tensors.
    ``margin`` is either a scalar or ``[N]`` per-pair margin.
    Returns a scalar (mean / sum) or ``[N]`` per-pair (none).
    """
    dist = (latents_a - latents_b).norm(dim=-1)
    if isinstance(margin, (int, float)):
        margin = torch.tensor(float(margin), device=dist.device, dtype=dist.dtype)
    else:
        margin = margin.to(device=dist.device, dtype=dist.dtype)
    elem = F.relu(margin - dist).pow(2)
    if reduction == "mean":
        return elem.mean()
    if reduction == "sum":
        return elem.sum()
    return elem


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass
class ContrastivePairRegistry:
    """In-memory index over a list of ContrastivePair records.

    Provides per-mechanic partner lookup, partner sampling, and a single
    ``sample_pair`` API used by the batch augmenter.
    """

    pairs: list[ContrastivePair] = field(default_factory=list)
    _index: dict[str, list[ContrastivePair]] = field(
        default_factory=dict, init=False, repr=False,
    )

    def __post_init__(self) -> None:
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index.clear()
        for pair in self.pairs:
            self._index.setdefault(pair.mechanic_a, []).append(pair)
            # Pairs are unordered for sampling — register both directions.
            if pair.mechanic_b != pair.mechanic_a:
                self._index.setdefault(pair.mechanic_b, []).append(pair)

    @classmethod
    def from_file(cls, path: str | Path) -> "ContrastivePairRegistry":
        return cls(pairs=load_pairs(path))

    @classmethod
    def from_paths(cls, paths: Iterable[str | Path]) -> "ContrastivePairRegistry":
        from lewm.family_contrastive import merge_pair_lists
        all_pairs: list[list[ContrastivePair]] = []
        for p in paths:
            all_pairs.append(load_pairs(p))
        return cls(pairs=merge_pair_lists(*all_pairs))

    def partners_for(self, mechanic: str) -> list[ContrastivePair]:
        return self._index.get(mechanic, [])

    def has(self, mechanic: str) -> bool:
        return mechanic in self._index

    def sample_partner(
        self, mechanic: str, rng: random.Random,
    ) -> ContrastivePair | None:
        partners = self._index.get(mechanic)
        if not partners:
            return None
        return rng.choice(partners)

    def partner_mechanic(self, pair: ContrastivePair, anchor: str) -> str:
        if pair.mechanic_a == anchor:
            return pair.mechanic_b
        return pair.mechanic_a

    def __len__(self) -> int:
        return len(self.pairs)


# ---------------------------------------------------------------------------
# Batch augmenter
# ---------------------------------------------------------------------------


@dataclass
class ContrastiveBatchAugmenter:
    """Given a base batch, build aligned ``(anchor, partner)`` pair tensors.

    For each anchor episode whose mechanic is in the registry, sample
    one partner mechanic from the registry, then look up an episode of
    that partner mechanic in ``episode_pool`` (a dataset that supports
    ``items_by_mechanic[name] -> list[item_dict]``). The augmenter
    returns:

      - anchor_indices: ``[N]`` long, indices into the base batch
      - partner_frames: ``[N, K, H, W]`` long (one-hot encoded later)
      - partner_actions: ``[N, K-1]`` long
      - margins: ``[N]`` float, per-pair margins from the registry

    Anchors without a registered partner are simply omitted from the
    auxiliary pair tensors, so the contrastive term scales with the
    fraction of confusable mechanics in the batch.
    """

    registry: ContrastivePairRegistry
    episode_pool: "EpisodePoolByMechanic"
    seed: int = 0
    sample_rate: float = 1.0

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def augment(
        self, batch: dict, mechanic_names: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Returns a dict with ``partner_context_frames``, ``partner_context_actions``,
        ``anchor_indices``, ``margins``; or ``None`` if no anchors had partners.

        ``mechanic_names`` is the index → name table used to look up
        anchors via batch['mechanic'] integer IDs. Required.
        """
        if mechanic_names is None:
            raise ValueError(
                "ContrastiveBatchAugmenter.augment requires mechanic_names lookup"
            )

        true_ids = batch["mechanic"]
        n_batch = int(true_ids.shape[0])
        anchor_indices: list[int] = []
        partner_frame_list: list[torch.Tensor] = []
        partner_action_list: list[torch.Tensor] = []
        margins: list[float] = []

        for i in range(n_batch):
            if self._rng.random() > self.sample_rate:
                continue
            mid = int(true_ids[i].item())
            if mid < 0 or mid >= len(mechanic_names):
                continue
            anchor_name = mechanic_names[mid]
            pair = self.registry.sample_partner(anchor_name, self._rng)
            if pair is None:
                continue
            partner_name = self.registry.partner_mechanic(pair, anchor_name)
            partner_item = self.episode_pool.sample_item(partner_name, self._rng)
            if partner_item is None:
                continue
            anchor_indices.append(i)
            partner_frame_list.append(partner_item["context_frames"])
            partner_action_list.append(partner_item["context_actions"])
            margins.append(float(pair.margin))

        if not anchor_indices:
            return None
        return {
            "anchor_indices": torch.tensor(anchor_indices, dtype=torch.long),
            "partner_context_frames": torch.stack(partner_frame_list),
            "partner_context_actions": torch.stack(partner_action_list),
            "margins": torch.tensor(margins, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Episode pool — lookup by mechanic name for partner sampling
# ---------------------------------------------------------------------------


class EpisodePoolByMechanic:
    """Index a base dataset by mechanic name for contrastive partner sampling.

    Walks the dataset once at construction time, applies the optional
    ``name_resolver`` to each item to derive the mechanic name string,
    and stores the integer index in a per-name bucket. Storage is
    shallow (list[int] per mechanic) so it stays cheap even on
    200K-episode datasets.

    The ``name_resolver`` is the bridge between dataset items (which
    carry ``game_idx`` integers from the goal metadata) and the registry
    (which is keyed by registered mechanic name strings). Typical use:

        pool = EpisodePoolByMechanic(
            train_dataset,
            name_resolver=lambda item: concept_names[item["game_idx"]],
        )

    Default resolver looks up ``item["mechanic"]``.
    """

    def __init__(
        self,
        base_dataset: object,
        *,
        name_resolver=None,
    ) -> None:
        self.base_dataset = base_dataset
        if name_resolver is None:
            name_resolver = lambda item: item.get("mechanic") if isinstance(item, dict) else None
        self._name_resolver = name_resolver
        self._indices_by_mechanic: dict[str, list[int]] = {}
        n = len(base_dataset)  # type: ignore[arg-type]
        for i in range(n):
            try:
                item = base_dataset[i]
            except Exception:  # noqa: BLE001
                continue
            name = name_resolver(item)
            if not name:
                continue
            self._indices_by_mechanic.setdefault(str(name), []).append(i)

    def has(self, mechanic_name: str) -> bool:
        return mechanic_name in self._indices_by_mechanic

    def sample_item(
        self, mechanic_name: str, rng: random.Random,
    ) -> dict | None:
        idxs = self._indices_by_mechanic.get(mechanic_name)
        if not idxs:
            return None
        i = rng.choice(idxs)
        return self.base_dataset[i]

    def __len__(self) -> int:
        return sum(len(v) for v in self._indices_by_mechanic.values())

    @property
    def mechanic_names(self) -> list[str]:
        return list(self._indices_by_mechanic)
