"""study_tags.v001 supervision adapter for GoalPropertyModel training.

Joins ``study_tags.jsonl`` (sidecar emitted by
``scripts/build_lewmsight_stage_tensors.py``) with the goal-property
dataset by ``game_id``. Yields the four supervision tensors expected by
``goal_property.study_tags_loss``:

  goal_family         [B] long          STUDY_TAGS_GOAL_FAMILIES index
  role_buckets        [B, 16, 9] float  binary (color × role) cells
  role_buckets_mask   [B, 16, 9] bool   True = "supervised cell"
  capability_flags    [B, 7] float      0/1
  capability_mask     [B, 7] bool       True = "mechanic declared this flag"
  topology            [B] long          STUDY_TAGS_TOPOLOGY index
  topology_mask       [B] bool          True = "mechanic declared topology"

The role bucket / capability / topology fields are sparse — most
mechanics declare a subset. Masks let the loss ignore undeclared cells.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from lewm.goal_property import (
    ARC_NUM_COLORS,
    GOAL_FAMILY_TO_IDX,
    NUM_STUDY_TAGS_CAPABILITIES,
    NUM_STUDY_TAGS_ROLES,
    NUM_STUDY_TAGS_TOPOLOGY,
    ROLE_FIELD_TO_IDX,
    STUDY_TAGS_CAPABILITY_FIELDS,
    TOPOLOGY_TO_IDX,
)


# ---------------------------------------------------------------------------
# Sidecar parser
# ---------------------------------------------------------------------------


def load_study_tags_sidecar(path: str | Path) -> dict[str, dict]:
    """Read ``study_tags.jsonl`` and index rows by ``audit.game_id``."""
    rows: dict[str, dict] = {}
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            gid = row.get("audit", {}).get("game_id")
            if gid is None:
                continue
            rows[str(gid)] = row
    return rows


# ---------------------------------------------------------------------------
# Per-row → supervision tensors
# ---------------------------------------------------------------------------


def goal_family_id_from_row(row: dict) -> int:
    """Map a sidecar row's goal_family value to its slot index.

    Falls back to ``unknown`` (idx 13) if the field is missing or
    unrecognised. Random_NN slots resolve to indices 14..45.
    """
    value = row.get("goal", {}).get("goal_family")
    if value is None:
        return GOAL_FAMILY_TO_IDX["unknown"]
    return GOAL_FAMILY_TO_IDX.get(str(value), GOAL_FAMILY_TO_IDX["unknown"])


def role_buckets_from_row(row: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(values [16, 9], mask [16, 9])``.

    A cell ``(c, r)`` is marked supervised (mask=True) when the
    mechanic's hook declared role ``r`` (movable_colors / hazard_colors /
    etc.) — i.e. the field is present and not None. Within that role,
    a color is True if it appears in the declared color list, False
    otherwise. Roles the mechanic did not declare leave the entire row
    masked False.
    """
    objects = row.get("objects", {}) or {}
    values = torch.zeros(ARC_NUM_COLORS, NUM_STUDY_TAGS_ROLES, dtype=torch.float32)
    mask = torch.zeros(ARC_NUM_COLORS, NUM_STUDY_TAGS_ROLES, dtype=torch.bool)
    for field, idx in ROLE_FIELD_TO_IDX.items():
        if field not in objects:
            continue
        # Field present → mark the whole column as supervised.
        mask[:, idx] = True
        colors = objects.get(field) or []
        for c in colors:
            try:
                ci = int(c)
            except (TypeError, ValueError):
                continue
            if 0 <= ci < ARC_NUM_COLORS:
                values[ci, idx] = 1.0
    # Mechanic-level player_color also implies role 0 (movable) of that color
    # would be FALSE info, NOT TRUE — we only know which colors play movable,
    # not which don't. So we do NOT auto-fill player_color here. The hooks
    # supply movable_colors explicitly when they want it.
    return values, mask


def capability_flags_from_row(row: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(values [7], mask [7])`` from the mechanic.requires_* fields."""
    mech = row.get("mechanic", {}) or {}
    values = torch.zeros(NUM_STUDY_TAGS_CAPABILITIES, dtype=torch.float32)
    mask = torch.zeros(NUM_STUDY_TAGS_CAPABILITIES, dtype=torch.bool)
    for i, field in enumerate(STUDY_TAGS_CAPABILITY_FIELDS):
        if field in mech:
            mask[i] = True
            values[i] = 1.0 if bool(mech[field]) else 0.0
    return values, mask


def topology_id_from_row(row: dict) -> tuple[int, bool]:
    """Return ``(slot_idx, supervised)`` for player_topology_signature."""
    sig = row.get("player", {}).get("player_topology_signature")
    if sig is None:
        return TOPOLOGY_TO_IDX["unknown"], False
    return TOPOLOGY_TO_IDX.get(str(sig), TOPOLOGY_TO_IDX["unknown"]), True


def supervision_from_row(row: dict) -> dict[str, torch.Tensor]:
    """One-row build of every supervision tensor.

    Batching is left to the caller — typically by wrapping a base
    dataset that yields ``(frames, actions)`` and merging the
    supervision tensors into the same batch dict.
    """
    rb_values, rb_mask = role_buckets_from_row(row)
    cap_values, cap_mask = capability_flags_from_row(row)
    topo_idx, topo_present = topology_id_from_row(row)
    return {
        "goal_family": torch.tensor(goal_family_id_from_row(row), dtype=torch.long),
        "role_buckets": rb_values,
        "role_buckets_mask": rb_mask,
        "capability_flags": cap_values,
        "capability_mask": cap_mask,
        "topology": torch.tensor(topo_idx, dtype=torch.long),
        "topology_mask": torch.tensor(topo_present, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# Wrapper Dataset that augments a frames+actions dataset with study_tags labels
# ---------------------------------------------------------------------------


class StudyTagsSupervisionWrapper(Dataset):
    """Wrap a base ``GoalPropertyDataset``-style dataset, joining each item to
    its ``study_tags.jsonl`` row by ``game_id``.

    The base dataset must expose ``game_id`` on each item dict (the
    ``manifest.json`` written by ``build_lewmsight_stage_tensors.py``
    already records it; downstream loaders just need to plumb it
    through). Items missing a sidecar row are filtered out at init —
    no silent label-of-zero supervision.
    """

    def __init__(
        self,
        base: Dataset,
        sidecar_path: str | Path,
        *,
        game_id_field: str = "game_id",
    ) -> None:
        super().__init__()
        self.base = base
        self.game_id_field = game_id_field
        self.sidecar = load_study_tags_sidecar(sidecar_path)
        # Filter base indices to those with a sidecar row.
        self._indices: list[int] = []
        n = len(base)  # type: ignore[arg-type]
        for i in range(n):
            item = base[i]
            gid = str(item.get(game_id_field, "")) if isinstance(item, dict) else ""
            if gid and gid in self.sidecar:
                self._indices.append(i)
        if not self._indices:
            raise RuntimeError(
                f"no base-dataset item joined to {sidecar_path} via {game_id_field}"
            )

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        base_item = self.base[self._indices[idx]]
        gid = str(base_item[self.game_id_field])
        supervision = supervision_from_row(self.sidecar[gid])
        # Merge — base item takes priority on overlapping keys (shouldn't overlap).
        return {**supervision, **base_item}
