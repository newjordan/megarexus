#!/usr/bin/env python3
"""Collect horizontal experience with an environment-scored prefix archive.

The archive's genotype is deliberately game-agnostic.  A gene selects either
an available action by rank, a point relative to the current frame, or a point
on a current-frame connected component by rank.  Genes are resolved again on
every fresh episode; exact coordinates and action ids are observations written
to the executed-program receipt, never mutation-time game routes.

Fitness and archive descriptors use only values returned by the environment:
``levels_completed``, state, and before/after frame deltas.  Existing Level-0
success memory is replayed before archived prefixes are evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_search.algorithms.components import (  # noqa: E402
    connected_components,
    frame_changed,
    frame_delta_count,
)
from legal_search.train.collect import (  # noqa: E402
    _discover_game_ids,
    annotate_level_weights,
    rows_to_tensors,
)
from legal_search.train.features import (  # noqa: E402
    FEATURE_DIM,
    FEATURE_LAYOUT,
    FEATURE_SCHEMA,
    encode_legal_features,
)
from legal_search.types import Action, Transition  # noqa: E402


GENE_SCHEMA = "legal_search.relative_component_gene.v1"
ARCHIVE_SCHEMA = "legal_search.environment_prefix_archive.v1"
GENE_KINDS = ("component_rank", "frame_relative", "available_rank")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_artifact_path(raw: str | Path, *, memory_path: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    root_relative = ROOT / path
    if root_relative.is_file():
        return root_relative
    return memory_path.parent / path


def _normalize_program(raw: Any, *, game_id: str, source: str) -> list[list[int | None]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"bootstrap program for {game_id!r} from {source} is empty or not a list")
    program: list[list[int | None]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"bootstrap {game_id!r} action {index} must be [action_id, x, y]")
        aid, x, y = item
        aid = int(aid)
        if aid not in range(1, 7):
            raise ValueError(f"bootstrap {game_id!r} action {index} has invalid action_id={aid}")
        if aid == 6:
            if x is None or y is None:
                raise ValueError(f"bootstrap {game_id!r} click {index} is missing coordinates")
            x, y = int(x), int(y)
        else:
            x, y = None, None
        program.append([aid, x, y])
    return program


@dataclass(frozen=True)
class BootstrapBundle:
    programs: dict[str, list[list[int | None]]]
    provenance: dict[str, Any]


def load_bootstrap_programs(path: Path) -> BootstrapBundle:
    """Load direct or referenced success programs with source hashes.

    Referenced programs use the same ``program_sources`` contract as
    :mod:`legal_search.train.policy` and ``program_collect.py``.
    """

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"bootstrap memory not found: {path}")
    memory = json.loads(path.read_text(encoding="utf-8"))
    programs: dict[str, list[list[int | None]]] = {}
    sources: dict[str, dict[str, Any]] = {}

    direct = memory.get("programs") or {}
    if not isinstance(direct, dict):
        raise ValueError("bootstrap memory 'programs' must be an object")
    for game_id, raw in direct.items():
        game_id = str(game_id)
        programs[game_id] = _normalize_program(raw, game_id=game_id, source=str(path))
        sources[game_id] = {"kind": "direct", "memory": str(path)}

    referenced = memory.get("program_sources") or {}
    if not isinstance(referenced, dict):
        raise ValueError("bootstrap memory 'program_sources' must be an object")
    for game_id, ref in referenced.items():
        game_id = str(game_id)
        if not isinstance(ref, dict) or "artifact" not in ref:
            raise ValueError(f"bootstrap source for {game_id!r} must contain artifact")
        artifact_path = _resolve_artifact_path(ref["artifact"], memory_path=path).resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"bootstrap artifact for {game_id!r} not found: {artifact_path}")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        try:
            winners = artifact["search"]["per_game"][game_id]["winning_programs"]
            winner_index = int(ref.get("winner_index", 0))
            raw = winners[winner_index]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid winning-program reference for {game_id!r} in {artifact_path}"
            ) from exc
        programs[game_id] = _normalize_program(
            raw,
            game_id=game_id,
            source=f"{artifact_path}:winner[{winner_index}]",
        )
        sources[game_id] = {
            "kind": "winning_program_reference",
            "artifact": str(artifact_path),
            "artifact_sha256": _sha256(artifact_path),
            "winner_index": winner_index,
        }

    if not programs:
        raise ValueError(f"bootstrap memory contains no programs: {path}")
    return BootstrapBundle(
        programs=programs,
        provenance={
            "path": str(path),
            "sha256": _sha256(path),
            "schema": memory.get("schema"),
            "truth_boundary": memory.get("truth_boundary"),
            "sources": sources,
        },
    )


def _grid(response: Any) -> np.ndarray:
    frame = response.frame
    if isinstance(frame, list):
        frame = frame[-1]
    array = np.asarray(frame)
    if array.ndim == 3:
        array = array[-1]
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D environment frame, got shape={array.shape}")
    return np.asarray(array, dtype=np.int16)


def _available(response: Any) -> list[int]:
    result: list[int] = []
    for raw in getattr(response, "available_actions", []) or [1, 2, 3, 4, 5, 6]:
        try:
            aid = int(getattr(raw, "value", raw))
        except (TypeError, ValueError):
            continue
        if aid in range(1, 7):
            result.append(aid)
    return sorted(set(result or [1, 2, 3, 4, 5, 6]))


@dataclass(frozen=True)
class ActionGene:
    """Game-agnostic, frame-relative action representation."""

    kind: str
    rank: int
    u: float
    v: float

    def __post_init__(self) -> None:
        if self.kind not in GENE_KINDS:
            raise ValueError(f"unsupported gene kind: {self.kind}")
        if not (0.0 <= float(self.u) <= 1.0 and 0.0 <= float(self.v) <= 1.0):
            raise ValueError("gene relative coordinates must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GENE_SCHEMA,
            "kind": self.kind,
            "rank": int(self.rank),
            "u": float(self.u),
            "v": float(self.v),
        }


def random_gene(rng: np.random.Generator) -> ActionGene:
    # Component and relative clicks get most of the capacity while ranked
    # discrete actions remain represented.  These are representation priors,
    # not environment or game rules.
    draw = float(rng.random())
    kind = "component_rank" if draw < 0.45 else "frame_relative" if draw < 0.75 else "available_rank"
    return ActionGene(
        kind=kind,
        rank=int(rng.integers(0, 32)),
        u=float(rng.random()),
        v=float(rng.random()),
    )


def _relative_xy(frame: np.ndarray, u: float, v: float) -> tuple[int, int]:
    height, width = np.asarray(frame).shape[:2]
    x = int(round(float(u) * max(0, width - 1)))
    y = int(round(float(v) * max(0, height - 1)))
    return min(max(x, 0), width - 1), min(max(y, 0), height - 1)


def _component_xy(frame: np.ndarray, gene: ActionGene) -> tuple[int, int] | None:
    array = np.asarray(frame)
    components = connected_components(
        array,
        min_area=1,
        max_area=int(array.size),
        exclude_full_span=True,
    )
    if not components:
        return None
    component = components[int(gene.rank) % len(components)]
    x0, y0, x1, y1 = [int(value) for value in component["bbox_xyxy"]]
    target_x = x0 + float(gene.u) * max(0, x1 - x0)
    target_y = y0 + float(gene.v) * max(0, y1 - y0)
    color = int(component["color"])
    coordinates = np.argwhere(array[y0 : y1 + 1, x0 : x1 + 1] == color)
    if not len(coordinates):
        return None
    best_y, best_x = min(
        ((y0 + int(y), x0 + int(x)) for y, x in coordinates),
        key=lambda yx: ((yx[1] - target_x) ** 2 + (yx[0] - target_y) ** 2, yx[0], yx[1]),
    )
    return best_x, best_y


def resolve_gene(gene: ActionGene, frame: np.ndarray, available: Sequence[int]) -> Action:
    """Resolve a generic gene using only the current frame/action availability."""

    avail = sorted(set(int(aid) for aid in available if int(aid) in range(1, 7)))
    if not avail:
        raise ValueError("environment exposed no valid actions")
    discrete = [aid for aid in avail if aid != 6]
    if gene.kind == "available_rank" and discrete:
        aid = discrete[int(gene.rank) % len(discrete)]
        return Action(aid, source="prefix_archive.available_rank")
    if 6 in avail:
        xy = _component_xy(frame, gene) if gene.kind == "component_rank" else None
        if xy is None:
            xy = _relative_xy(frame, gene.u, gene.v)
            source = "prefix_archive.frame_relative"
        else:
            source = "prefix_archive.component_rank"
        return Action(6, x=xy[0], y=xy[1], source=source)
    aid = discrete[int(gene.rank) % len(discrete)]
    return Action(aid, source="prefix_archive.available_rank_fallback")


def mutate_prefix(
    parent: Sequence[ActionGene],
    *,
    rng: np.random.Generator,
    max_length: int,
    initial_length: int,
    mutation_span: int,
) -> tuple[ActionGene, ...]:
    """Branch or edit a prefix without ever introducing resolved game actions."""

    if max_length < 1:
        return ()
    genes = list(parent[:max_length])
    span = max(1, min(int(mutation_span), max_length))
    if not genes:
        return tuple(random_gene(rng) for _ in range(min(max_length, max(1, initial_length))))

    operation = int(rng.integers(0, 4))
    if operation == 0 and len(genes) < max_length:
        # Archive growth: extend a measured prefix.
        for _ in range(min(span, max_length - len(genes))):
            genes.append(random_gene(rng))
    elif operation == 1:
        # Point/span mutation preserves most of an observed prefix.
        start = int(rng.integers(0, len(genes)))
        for index in range(start, min(len(genes), start + span)):
            genes[index] = random_gene(rng)
    elif operation == 2:
        # Branch from an earlier measured prefix and continue generically.
        keep = int(rng.integers(0, len(genes) + 1))
        genes = genes[:keep]
        for _ in range(min(span, max_length - len(genes))):
            genes.append(random_gene(rng))
    else:
        # Insertion exposes new sequence degrees of freedom.
        start = int(rng.integers(0, len(genes) + 1))
        insertion = [random_gene(rng) for _ in range(min(span, max_length - len(genes)))]
        genes[start:start] = insertion
        genes = genes[:max_length]
    if not genes:
        genes.append(random_gene(rng))
    return tuple(genes)


def complete_candidate_suffix(
    genes: Sequence[ActionGene],
    *,
    rng: np.random.Generator,
    candidate_suffix_length: int,
    max_length: int,
) -> tuple[ActionGene, ...]:
    """Optionally complete a mutated prefix to an explicit generic horizon.

    ``candidate_suffix_length == 0`` is the v1 compatibility path: the input
    prefix is returned unchanged and the RNG is not consumed.  A positive
    setting produces exactly ``min(candidate_suffix_length, max_length)`` genes
    by retaining the mutated prefix up to that horizon and appending only fresh
    :class:`ActionGene` values.  No resolved action or game observation enters
    completion.
    """

    requested = int(candidate_suffix_length)
    if requested < 0:
        raise ValueError("candidate_suffix_length must be non-negative")
    if requested == 0:
        return tuple(genes)
    target = min(requested, max(0, int(max_length)))
    completed = list(genes[:target])
    while len(completed) < target:
        completed.append(random_gene(rng))
    return tuple(completed)


def state_bucket(state: str) -> str:
    upper = str(state).upper()
    if "WIN" in upper:
        return "win"
    if "GAME_OVER" in upper:
        return "game_over"
    return "not_finished"


def _frame_fingerprint(frame: np.ndarray) -> str:
    array = np.ascontiguousarray(frame)
    digest = hashlib.blake2b(digest_size=12)
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


@dataclass
class PrefixSignals:
    """Cumulative environment observations for one post-bootstrap prefix."""

    start_level: int
    levels: int
    state: str
    changed_steps: int = 0
    delta_pixels: int = 0
    frame_fingerprints: set[str] = field(default_factory=set)

    def observe(self, before: np.ndarray, after: np.ndarray, *, levels: int, state: str) -> None:
        delta = frame_delta_count(before, after)
        self.changed_steps += int(delta > 0)
        self.delta_pixels += int(delta)
        self.levels = max(int(self.levels), int(levels))
        self.state = str(state)
        self.frame_fingerprints.add(_frame_fingerprint(after))

    def descriptor(self, prefix_length: int) -> tuple[Any, ...]:
        delta_bin = min(15, int(math.log2(1 + max(0, self.delta_pixels))))
        return (
            int(self.levels),
            state_bucket(self.state),
            min(7, int(self.changed_steps)),
            delta_bin,
            min(7, len(self.frame_fingerprints)),
            min(31, max(0, int(prefix_length) - 1) // 4),
        )

    def score(self, prefix_length: int) -> tuple[int, ...]:
        terminal_rank = {"game_over": 0, "not_finished": 1, "win": 2}[state_bucket(self.state)]
        return (
            int(self.levels),
            terminal_rank,
            len(self.frame_fingerprints),
            int(self.changed_steps),
            int(self.delta_pixels),
            -int(prefix_length),
        )


@dataclass(frozen=True)
class ArchiveEntry:
    game_id: str
    genes: tuple[ActionGene, ...]
    descriptor: tuple[Any, ...]
    score: tuple[int, ...]
    observations: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "genes": [gene.to_dict() for gene in self.genes],
            "descriptor": list(self.descriptor),
            "score": list(self.score),
            "observations": dict(self.observations),
        }


class PrefixArchive:
    """Behavior-cell archive whose admission is entirely observation-scored."""

    def __init__(self, capacity: int, *, retain_games: bool) -> None:
        if capacity < 1:
            raise ValueError("archive capacity must be positive")
        self.capacity = int(capacity)
        self.retain_games = bool(retain_games)
        self.entries: dict[tuple[Any, ...], ArchiveEntry] = {}
        self.considered = 0
        self.admitted = 0
        self.replaced = 0
        self.evicted = 0

    def __len__(self) -> int:
        return len(self.entries)

    def _key(self, entry: ArchiveEntry) -> tuple[Any, ...]:
        if self.retain_games:
            return (entry.game_id, *entry.descriptor)
        return tuple(entry.descriptor)

    def add(self, entry: ArchiveEntry) -> bool:
        self.considered += 1
        key = self._key(entry)
        previous = self.entries.get(key)
        if previous is not None:
            if entry.score <= previous.score:
                return False
            self.entries[key] = entry
            self.admitted += 1
            self.replaced += 1
            return True
        if len(self.entries) >= self.capacity:
            worst_key, worst = min(self.entries.items(), key=lambda item: (item[1].score, item[0]))
            if entry.score <= worst.score:
                return False
            del self.entries[worst_key]
            self.evicted += 1
        self.entries[key] = entry
        self.admitted += 1
        return True

    def select(self, rng: np.random.Generator, tournament: int = 4) -> ArchiveEntry | None:
        values = list(self.entries.values())
        if not values:
            return None
        count = min(max(1, int(tournament)), len(values))
        indices = rng.choice(len(values), size=count, replace=False)
        return max((values[int(index)] for index in indices), key=lambda entry: entry.score)

    def summary(self, *, include_entries: int = 32) -> dict[str, Any]:
        best = sorted(self.entries.values(), key=lambda entry: entry.score, reverse=True)
        return {
            "schema": ARCHIVE_SCHEMA,
            "capacity": self.capacity,
            "cells": len(self.entries),
            "considered": self.considered,
            "admitted": self.admitted,
            "replaced": self.replaced,
            "evicted": self.evicted,
            "top_entries": [entry.to_dict() for entry in best[: max(0, include_entries)]],
        }


def archive_entry(
    *,
    game_id: str,
    genes: Sequence[ActionGene],
    signals: PrefixSignals,
) -> ArchiveEntry:
    prefix = tuple(genes)
    return ArchiveEntry(
        game_id=str(game_id),
        genes=prefix,
        descriptor=signals.descriptor(len(prefix)),
        score=signals.score(len(prefix)),
        observations={
            "start_level": int(signals.start_level),
            "max_level": int(signals.levels),
            "state_bucket": state_bucket(signals.state),
            "changed_steps": int(signals.changed_steps),
            "delta_pixels": int(signals.delta_pixels),
            "unique_observed_frames": len(signals.frame_fingerprints),
        },
    )


def _action_from_bootstrap(item: Sequence[int | None]) -> Action:
    aid, x, y = item
    return Action(
        int(aid),
        x=None if x is None else int(x),
        y=None if y is None else int(y),
        source="prefix_archive.bootstrap_memory",
    )


def _program_json(actions: Iterable[Action]) -> list[list[int | None]]:
    return [
        [int(action.action_id), None if action.x is None else int(action.x), None if action.y is None else int(action.y)]
        for action in actions
    ]


def _is_terminal(state: str) -> bool:
    return state_bucket(state) in {"win", "game_over"}


def collect(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one horizontal archive round per game before advancing the round."""

    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine import GameAction

    env_dir = Path(args.environment_dir).expanduser().resolve()
    games = _discover_game_ids(env_dir)
    if not games:
        raise SystemExit(f"no installed games discovered under {env_dir}")
    bootstrap = load_bootstrap_programs(Path(args.bootstrap_programs))
    missing = sorted(set(games).difference(bootstrap.programs))
    extra = sorted(set(bootstrap.programs).difference(games))
    if missing:
        raise SystemExit(f"bootstrap memory is missing installed games: {missing}")
    if args.max_actions <= max(len(bootstrap.programs[game_id]) for game_id in games):
        raise SystemExit("max-actions must exceed every installed game's Level-0 bootstrap length")

    rng = np.random.default_rng(args.seed)
    per_game_archives = {
        game_id: PrefixArchive(args.archive_capacity, retain_games=False) for game_id in games
    }
    shared_archive = PrefixArchive(args.shared_archive_capacity, retain_games=True)
    action_map = {index: getattr(GameAction, f"ACTION{index}") for index in range(1, 7)}
    rows: list[dict[str, Any]] = []
    receipts: dict[str, dict[str, Any]] = {
        game_id: {
            "episodes": 0,
            "rows": 0,
            "level_ups": 0,
            "best_level": 0,
            "states": {},
            "bootstrap_failures": 0,
            "step_errors": 0,
            "full_programs": [],
            "winning_programs": [],
            "target_programs": [],
            "candidate_origins": {"fresh": 0, "per_game": 0, "shared_transfer": 0},
            "candidate_suffix_mode": (
                "v1_variable_prefix"
                if args.candidate_suffix_length == 0
                else "v2_horizon_complete"
            ),
            "candidate_suffix_length_requested": int(args.candidate_suffix_length),
            "candidate_suffix_length_effective": (
                0
                if args.candidate_suffix_length == 0
                else min(
                    int(args.candidate_suffix_length),
                    args.max_actions - len(bootstrap.programs[game_id]),
                )
            ),
            "candidate_suffix_lengths": [],
            "executed_suffix_lengths": [],
        }
        for game_id in games
    }

    for round_index in range(args.episodes_per_game):
        for game_id in games:
            own_archive = per_game_archives[game_id]
            parent: ArchiveEntry | None = None
            origin = "fresh"
            if len(shared_archive) and float(rng.random()) < float(args.transfer_rate):
                parent = shared_archive.select(rng)
                origin = "shared_transfer"
            elif len(own_archive):
                parent = own_archive.select(rng)
                origin = "per_game"
            elif len(shared_archive):
                parent = shared_archive.select(rng)
                origin = "shared_transfer"
            max_suffix = args.max_actions - len(bootstrap.programs[game_id])
            genes = mutate_prefix(
                () if parent is None else parent.genes,
                rng=rng,
                max_length=max_suffix,
                initial_length=args.initial_prefix_length,
                mutation_span=args.mutation_span,
            )
            genes = complete_candidate_suffix(
                genes,
                rng=rng,
                candidate_suffix_length=args.candidate_suffix_length,
                max_length=max_suffix,
            )
            receipt = receipts[game_id]
            receipt["candidate_origins"][origin] += 1
            receipt["candidate_suffix_lengths"].append(len(genes))

            arcade = Arcade(
                arc_api_key="anonymous",
                environments_dir=str(env_dir),
                operation_mode=OperationMode.OFFLINE,
            )
            scorecard = arcade.open_scorecard()
            episode_rows: list[dict[str, Any]] = []
            transitions: list[Transition] = []
            executed: list[Action] = []
            levels = 0
            state = ""
            response: Any = None
            frame: np.ndarray | None = None
            step_error = False

            def take_action(action: Action) -> tuple[bool, int]:
                nonlocal response, frame, levels, state, step_error
                assert frame is not None
                available = _available(response)
                if int(action.action_id) not in available:
                    return False, 0
                features = encode_legal_features(
                    frame,
                    game_id=game_id,
                    transitions=transitions[-16:],
                    available_actions=available,
                    levels_completed=levels,
                    run_consume=True,
                )
                before = frame.copy()
                payload = (
                    {"x": int(action.x or 0), "y": int(action.y or 0)}
                    if int(action.action_id) == 6
                    else None
                )
                try:
                    response = env.step(action_map[int(action.action_id)], data=payload)
                except Exception:
                    step_error = True
                    receipt["step_errors"] += 1
                    return False, 0
                after = _grid(response)
                levels_after = int(getattr(response, "levels_completed", 0) or 0)
                state = str(getattr(response, "state", ""))
                changed = frame_changed(before, after)
                transition = Transition(
                    game_id=game_id,
                    step=len(transitions),
                    frame_before=before,
                    action=action,
                    frame_after=after,
                    levels_before=levels,
                    levels_after=levels_after,
                    state_after=state,
                    frame_changed=changed,
                    levels_up=levels_after > levels,
                )
                transitions.append(transition)
                executed.append(action)
                episode_rows.append(
                    {
                        "frame": before,
                        "features": features.astype(np.float32),
                        "action_id": int(action.action_id),
                        "x": int(action.x or 0),
                        "y": int(action.y or 0),
                        "did_change": bool(changed),
                        "levels_up": bool(transition.levels_up),
                        "levels_after": levels_after,
                        "game_id": game_id,
                        "source": action.source,
                    }
                )
                delta = frame_delta_count(before, after)
                frame, levels = after, levels_after
                return True, delta

            try:
                env = arcade.make(game_id, scorecard_id=scorecard)
                response = env.reset()
                frame = _grid(response)
                levels = int(getattr(response, "levels_completed", 0) or 0)
                state = str(getattr(response, "state", ""))

                for item in bootstrap.programs[game_id]:
                    if len(executed) >= args.max_actions or levels > 0 or _is_terminal(state):
                        break
                    action = _action_from_bootstrap(item)
                    ok, _ = take_action(action)
                    if not ok:
                        break

                bootstrap_steps = len(executed)
                bootstrap_ok = levels > 0
                if not bootstrap_ok:
                    receipt["bootstrap_failures"] += 1

                signals = PrefixSignals(start_level=levels, levels=levels, state=state)
                if bootstrap_ok and levels < args.target_level and not _is_terminal(state):
                    for gene_index, gene in enumerate(genes):
                        if len(executed) >= args.max_actions:
                            break
                        action = resolve_gene(gene, frame, _available(response))
                        before = frame.copy()
                        ok, _ = take_action(action)
                        if not ok:
                            break
                        signals.observe(before, frame, levels=levels, state=state)
                        entry = archive_entry(
                            game_id=game_id,
                            genes=genes[: gene_index + 1],
                            signals=signals,
                        )
                        own_archive.add(entry)
                        shared_archive.add(entry)
                        if levels >= args.target_level or _is_terminal(state):
                            break
            finally:
                try:
                    arcade.close_scorecard(scorecard)
                except Exception:
                    pass

            annotate_level_weights(episode_rows)
            rows.extend(episode_rows)
            receipt["episodes"] += 1
            receipt["rows"] += len(episode_rows)
            receipt["level_ups"] += sum(int(row["levels_up"]) for row in episode_rows)
            receipt["best_level"] = max(int(receipt["best_level"]), int(levels))
            receipt["states"][state] = int(receipt["states"].get(state, 0)) + 1
            receipt["executed_suffix_lengths"].append(max(0, len(executed) - bootstrap_steps))
            full_program = _program_json(executed)
            receipt["full_programs"].append(full_program)
            if any(bool(row["levels_up"]) for row in episode_rows):
                receipt["winning_programs"].append(full_program)
            if levels >= args.target_level:
                receipt["target_programs"].append(full_program)
            print(
                f"[prefix_archive] round={round_index + 1}/{args.episodes_per_game} "
                f"game={game_id} rows={len(episode_rows)} level={levels} "
                f"cells={len(own_archive)} target={len(receipt['target_programs'])} "
                f"step_error={int(step_error)}",
                flush=True,
            )

    for game_id, archive in per_game_archives.items():
        candidate_counts: dict[str, int] = {}
        for length in receipts[game_id]["candidate_suffix_lengths"]:
            key = str(int(length))
            candidate_counts[key] = candidate_counts.get(key, 0) + 1
        executed_counts: dict[str, int] = {}
        for length in receipts[game_id]["executed_suffix_lengths"]:
            key = str(int(length))
            executed_counts[key] = executed_counts.get(key, 0) + 1
        receipts[game_id]["candidate_suffix_length_distribution"] = candidate_counts
        receipts[game_id]["executed_suffix_length_distribution"] = executed_counts
        receipts[game_id]["prefix_archive"] = archive.summary(include_entries=args.manifest_archive_entries)
    search = {
        "schema": ARCHIVE_SCHEMA,
        "games": games,
        "all_installed_games": True,
        "candidate_suffix_mode": (
            "v1_variable_prefix"
            if args.candidate_suffix_length == 0
            else "v2_horizon_complete"
        ),
        "candidate_suffix_length": int(args.candidate_suffix_length),
        "environment_dir": str(env_dir),
        "bootstrap": bootstrap.provenance,
        "per_game": receipts,
        "shared_prefix_archive": shared_archive.summary(
            include_entries=args.manifest_archive_entries
        ),
        "extra_bootstrap_games_not_installed": extra,
    }
    return rows, search


def _balance_rows(
    rows: Sequence[dict[str, Any]], *, rng: np.random.Generator
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_game[str(row["game_id"])].append(row)
    if not by_game or any(not game_rows for game_rows in by_game.values()):
        raise ValueError("each installed game must contribute rows")
    target = min(len(game_rows) for game_rows in by_game.values())
    balanced: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for game_id, game_rows in sorted(by_game.items()):
        priority = [row for row in game_rows if row.get("levels_up") or row.get("success_window")]
        priority_ids = {id(row) for row in priority}
        ordinary = [row for row in game_rows if id(row) not in priority_ids]
        keep = priority[:target]
        remaining = target - len(keep)
        if remaining:
            order = rng.permutation(len(ordinary))[:remaining]
            keep.extend(ordinary[int(index)] for index in order)
        balanced.extend(keep)
        counts[game_id] = len(keep)
    return balanced, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-programs", type=Path, required=True)
    parser.add_argument("--environment-dir", type=Path, default=ROOT / "environment_files")
    parser.add_argument("--episodes-per-game", type=int, default=64)
    parser.add_argument("--max-actions", type=int, default=220)
    parser.add_argument("--target-level", type=int, default=2)
    parser.add_argument("--seed", type=int, default=71111)
    parser.add_argument("--archive-capacity", type=int, default=512)
    parser.add_argument("--shared-archive-capacity", type=int, default=1024)
    parser.add_argument("--initial-prefix-length", type=int, default=12)
    parser.add_argument("--mutation-span", type=int, default=6)
    parser.add_argument(
        "--candidate-suffix-length",
        type=int,
        default=0,
        help=(
            "0 preserves v1 variable-prefix behavior; positive N completes every mutated "
            "candidate with generic genes to min(N, the per-game suffix budget)"
        ),
    )
    parser.add_argument("--transfer-rate", type=float, default=0.25)
    parser.add_argument("--manifest-archive-entries", type=int, default=32)
    args = parser.parse_args()

    if args.episodes_per_game < 1:
        parser.error("--episodes-per-game must be positive")
    if args.max_actions < 2:
        parser.error("--max-actions must be at least 2")
    if args.target_level < 1:
        parser.error("--target-level must be positive")
    if args.initial_prefix_length < 1 or args.mutation_span < 1:
        parser.error("prefix lengths must be positive")
    if args.candidate_suffix_length < 0:
        parser.error("--candidate-suffix-length must be non-negative")
    if not 0.0 <= args.transfer_rate <= 1.0:
        parser.error("--transfer-rate must be in [0, 1]")

    from legal_search.train.mining_gate import assert_mining_complete

    assert_mining_complete()
    rows, search = collect(args)
    if len(rows) < 2:
        raise SystemExit("fewer than two environment-observed rows collected")
    observed_games = {str(row["game_id"]) for row in rows}
    missing_row_games = sorted(set(search["games"]).difference(observed_games))
    if missing_row_games:
        raise SystemExit(
            f"installed games produced no environment-observed rows: {missing_row_games}"
        )

    rng = np.random.default_rng(args.seed)
    balanced, balanced_counts = _balance_rows(rows, rng=rng)
    search["balanced_rows_per_game"] = balanced_counts
    order = rng.permutation(len(balanced))
    balanced = [balanced[int(index)] for index in order]
    requested_val = max(32, len(balanced) // 10)
    n_val = min(len(balanced) - 1, requested_val)
    train_rows, val_rows = balanced[n_val:], balanced[:n_val]

    payload = {
        "schema": "legal_search.blend_dataset.v2_fullmine",
        "feature_schema": FEATURE_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "feature_layout": FEATURE_LAYOUT,
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "train": rows_to_tensors(train_rows),
        "val": rows_to_tensors(val_rows),
        "meta": {
            "collector": (
                "environment_scored_program_prefix_archive_v1"
                if args.candidate_suffix_length == 0
                else "environment_scored_program_prefix_archive_v2_horizon_complete"
            ),
            "archive_schema": ARCHIVE_SCHEMA,
            "gene_schema": GENE_SCHEMA,
            "legal": True,
            "truth_boundary": (
                "Archive admission uses only environment-returned levels_completed, state, "
                "and measured before/after frame deltas. Gene mutation contains no resolved "
                "game route or target label."
            ),
            "environment_signals": ["levels_completed", "state", "frame_delta"],
            "all_installed_games": True,
            "episodes_per_game": args.episodes_per_game,
            "max_actions": args.max_actions,
            "target_level": args.target_level,
            "seed": args.seed,
            "archive_capacity": args.archive_capacity,
            "shared_archive_capacity": args.shared_archive_capacity,
            "initial_prefix_length": args.initial_prefix_length,
            "mutation_span": args.mutation_span,
            "candidate_suffix_mode": (
                "v1_variable_prefix"
                if args.candidate_suffix_length == 0
                else "v2_horizon_complete"
            ),
            "candidate_suffix_length": args.candidate_suffix_length,
            "transfer_rate": args.transfer_rate,
            "bootstrap_programs": str(Path(args.bootstrap_programs).expanduser().resolve()),
            "search": search,
            "n_levels_up": sum(int(row["levels_up"]) for row in balanced),
            "banned_lineage": [
                "sp80_spill_model",
                "dropletsim",
                "human_mined_routes",
                "fixture_win_banks_as_labels",
                "game_specific_route_genes",
            ],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    manifest = args.out.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(payload["meta"], indent=2) + "\n", encoding="utf-8")
    print(
        f"[prefix_archive] wrote {args.out} train={len(train_rows)} val={len(val_rows)} "
        f"games={len(search['games'])}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
