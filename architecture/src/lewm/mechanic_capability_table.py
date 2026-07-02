"""Per-mechanic capability + role + topology summary, built from study_tags.jsonl.

Foundational utility for the supervision-shape upgrades (hard-negative
mining, family-contrastive sampling, capability-conditional mechanic
head). Loads the sidecar emitted by
``scripts/build_lewmsight_stage_tensors.py``, deduplicates by
mechanic name, and surfaces:

  - 7-bit ``requires_*`` capability vector per mechanic
  - goal_family per mechanic
  - player_topology_signature per mechanic
  - declared role-color sets per mechanic
  - capability_hamming(a, b) → int in 0..7
  - hamming_margin(a, b, base) → float, base * hamming / 7

Mechanics that fail to declare a capability flag are treated as
``None`` (unknown) — they contribute ``0.5`` to the hamming term when
paired with a mechanic that DID declare the flag, biasing the margin
toward zero so we don't push pairs apart on missing data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


# Order must match study_tags.STUDY_TAGS_CAPABILITY_FIELDS — slot indices
# in the capability vector correspond 1:1 to the codepoint slots in
# Block C of reserved_tokens.json.
CAPABILITY_FIELDS: tuple[str, ...] = (
    "requires_memory", "requires_counting", "requires_contact",
    "requires_color_binding", "requires_topology",
    "requires_sequence_order", "requires_nonlocal_causality",
)
NUM_CAPABILITIES = len(CAPABILITY_FIELDS)

# Default margin contribution from a missing capability field, used by
# the hamming computation when one side declared the field and the other
# didn't. 0.5 is the indifference value.
UNDECLARED_HAMMING_CONTRIBUTION = 0.5


@dataclass
class MechanicProfile:
    """Capability + role profile for a single mechanic name."""

    mechanic: str
    capabilities: tuple[float | None, ...] = field(
        default_factory=lambda: tuple(None for _ in range(NUM_CAPABILITIES)),
    )
    goal_family: str | None = None
    topology: str | None = None
    role_colors: dict[str, frozenset[int]] = field(default_factory=dict)
    episode_count: int = 0

    def capability_bits(self) -> tuple[int, ...]:
        """Capabilities cast to 0/1 — unknown → 0. Use ``capability_hamming``
        for the version that respects unknowns."""
        return tuple(int(bool(c)) if c is not None else 0 for c in self.capabilities)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _coerce_capability_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(bool(value))
    return None


def _merge_profiles(existing: MechanicProfile, row: dict) -> None:
    """Fold one sidecar row into an existing per-mechanic profile.

    Capability values from successive rows must AGREE — the
    ``study_tags_hook`` returns the same profile for every episode of a
    given mechanic, so any disagreement signals a bug. The first row
    wins on tie; subsequent contradictions are recorded as the value
    becoming ``None`` (i.e. unknown / disputed).
    """
    mech_block = row.get("mechanic", {}) or {}
    new_caps = [_coerce_capability_value(mech_block.get(field)) for field in CAPABILITY_FIELDS]
    merged: list[float | None] = list(existing.capabilities)
    for i in range(NUM_CAPABILITIES):
        a, b = merged[i], new_caps[i]
        if a is None:
            merged[i] = b
        elif b is None:
            pass
        elif a != b:
            merged[i] = None  # disputed → mark unknown
    existing.capabilities = tuple(merged)

    goal_block = row.get("goal", {}) or {}
    gf = goal_block.get("goal_family")
    if existing.goal_family is None:
        existing.goal_family = gf
    elif gf is not None and existing.goal_family != gf:
        # Same mechanic shouldn't shift goal_family across episodes.
        # Mark as None on conflict to be safe.
        existing.goal_family = None

    player_block = row.get("player", {}) or {}
    topo = player_block.get("player_topology_signature")
    if existing.topology is None:
        existing.topology = topo
    elif topo is not None and existing.topology != topo:
        existing.topology = None

    objects = row.get("objects", {}) or {}
    for field_name, colors in objects.items():
        if not isinstance(colors, list):
            continue
        try:
            colset = frozenset(int(c) for c in colors)
        except (TypeError, ValueError):
            continue
        prev = existing.role_colors.get(field_name)
        if prev is None:
            existing.role_colors[field_name] = colset
        else:
            existing.role_colors[field_name] = prev | colset

    existing.episode_count += 1


def build_capability_table(
    sidecar_path: str | Path,
) -> dict[str, MechanicProfile]:
    """Stream ``study_tags.jsonl`` and emit ``{mechanic_name: MechanicProfile}``.

    Falls back to an empty table if the sidecar doesn't exist.
    """
    profiles: dict[str, MechanicProfile] = {}
    p = Path(sidecar_path)
    if not p.exists():
        return profiles
    with p.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            mech_block = row.get("mechanic", {}) or {}
            name = mech_block.get("mechanic_id")
            if not name:
                continue
            name = str(name)
            if name not in profiles:
                profiles[name] = MechanicProfile(mechanic=name)
            _merge_profiles(profiles[name], row)
    return profiles


# ---------------------------------------------------------------------------
# Hamming + margin
# ---------------------------------------------------------------------------


def capability_hamming(a: MechanicProfile, b: MechanicProfile) -> float:
    """Hamming-like distance over capability vectors, treating unknowns at
    ``UNDECLARED_HAMMING_CONTRIBUTION``.

    Returns a float in ``[0.0, NUM_CAPABILITIES]``. Pure-bit case
    (both sides declared every flag) returns an integer in 0..7.
    """
    total = 0.0
    for ca, cb in zip(a.capabilities, b.capabilities, strict=True):
        if ca is None and cb is None:
            continue  # both unknown — contribute nothing
        if ca is None or cb is None:
            total += UNDECLARED_HAMMING_CONTRIBUTION
            continue
        if ca != cb:
            total += 1.0
    return total


def hamming_margin(
    a: MechanicProfile,
    b: MechanicProfile,
    *,
    base_margin: float = 1.0,
) -> float:
    """Margin to use in a contrastive hinge loss between ``a`` and ``b``.

    Linear in capability_hamming. base_margin is the value when every
    capability flag differs; mechanics with identical capabilities get
    margin 0 (the hinge will not push them apart).
    """
    if NUM_CAPABILITIES == 0:
        return 0.0
    return base_margin * (capability_hamming(a, b) / NUM_CAPABILITIES)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def to_jsonable(table: dict[str, MechanicProfile]) -> dict:
    return {
        "schema": "lewm.mechanic_capability_table.v001",
        "num_capabilities": NUM_CAPABILITIES,
        "capability_fields": list(CAPABILITY_FIELDS),
        "mechanics": [
            {
                "mechanic": p.mechanic,
                "capabilities": list(p.capabilities),
                "goal_family": p.goal_family,
                "topology": p.topology,
                "role_colors": {k: sorted(v) for k, v in p.role_colors.items()},
                "episode_count": p.episode_count,
            }
            for p in sorted(table.values(), key=lambda x: x.mechanic)
        ],
    }


def save_table(table: dict[str, MechanicProfile], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(table), indent=2, sort_keys=True))
    return p


def load_table(path: str | Path) -> dict[str, MechanicProfile]:
    p = Path(path)
    payload = json.loads(p.read_text())
    out: dict[str, MechanicProfile] = {}
    for row in payload.get("mechanics", []):
        name = str(row["mechanic"])
        caps_raw = row.get("capabilities") or [None] * NUM_CAPABILITIES
        caps: list[float | None] = []
        for c in caps_raw:
            if c is None:
                caps.append(None)
            else:
                caps.append(float(c))
        role_colors = {
            str(k): frozenset(int(c) for c in v)
            for k, v in (row.get("role_colors") or {}).items()
        }
        out[name] = MechanicProfile(
            mechanic=name,
            capabilities=tuple(caps),
            goal_family=row.get("goal_family"),
            topology=row.get("topology"),
            role_colors=role_colors,
            episode_count=int(row.get("episode_count", 0)),
        )
    return out


# ---------------------------------------------------------------------------
# Grouping helpers (used by family-contrastive miner)
# ---------------------------------------------------------------------------


def group_by_goal_family(
    table: dict[str, MechanicProfile],
) -> dict[str, list[MechanicProfile]]:
    """Bucket mechanics by goal_family. Mechanics with no goal_family go to None."""
    buckets: dict[str, list[MechanicProfile]] = {}
    for prof in table.values():
        key = prof.goal_family if prof.goal_family is not None else "__unknown__"
        buckets.setdefault(key, []).append(prof)
    return buckets


def pairs_within_family(
    table: dict[str, MechanicProfile],
    *,
    include_unknown_family: bool = False,
    base_margin: float = 1.0,
    min_margin: float = 0.0,
) -> list[tuple[str, str, float]]:
    """Cartesian pairs of mechanics that share a goal_family.

    Pairs with capability-hamming-margin below ``min_margin`` are
    filtered out — they would not contribute to a hinge loss anyway.
    """
    out: list[tuple[str, str, float]] = []
    for fam, profs in group_by_goal_family(table).items():
        if fam == "__unknown__" and not include_unknown_family:
            continue
        for i in range(len(profs)):
            for j in range(i + 1, len(profs)):
                margin = hamming_margin(profs[i], profs[j], base_margin=base_margin)
                if margin <= min_margin:
                    continue
                out.append((profs[i].mechanic, profs[j].mechanic, margin))
    return out
