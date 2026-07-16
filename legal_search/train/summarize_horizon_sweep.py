#!/usr/bin/env python3
"""Fail-closed summary for a recorded horizon-complete ultra sweep.

The condition is the authority for run membership and resolved settings.  A
run is rankable only when its manifest exists, parses, covers exactly the
declared games, and agrees with every checked condition field.  Missing or
mismatched runs make the sweep summary incomplete; they are never filled from
logs, neighboring runs, or historical artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
CONDITION_SCHEMA = "synthesis_lab.ultra_sweep_condition.v1"
SUMMARY_SCHEMA = "legal_search.horizon_ultra_sweep_summary.v1"
EXPECTED_LOADER = "legal_search.environment_scored_program_prefix_archive_v2_horizon_complete"
EXPECTED_COLLECTOR = "environment_scored_program_prefix_archive_v2_horizon_complete"
EXPECTED_SUFFIX_MODE = "v2_horizon_complete"


class ConditionError(ValueError):
    """The sweep condition is not sufficiently explicit to adjudicate."""


class ManifestError(ValueError):
    """A declared run manifest is malformed or internally inconsistent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _plain_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}, got {value!r}")
    return value


def _string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ConditionError(f"{field} must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in value):
        raise ConditionError(f"{field} entries must be non-empty strings")
    if len(set(value)) != len(value):
        raise ConditionError(f"{field} contains duplicates")
    return list(value)


def _resolve_workspace_path(raw: Any, *, field: str, workspace_root: Path) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ConditionError(f"{field} must be a non-empty path string")
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (workspace_root / path).resolve()


def _state_bucket(raw: str) -> str:
    upper = str(raw).upper()
    if "WIN" in upper:
        return "win"
    if "GAME_OVER" in upper:
        return "game_over"
    if "NOT_FINISHED" in upper:
        return "not_finished"
    raise ManifestError(f"unrecognized environment state: {raw!r}")


def _count_distribution(value: Any, *, field: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ManifestError(f"{field} must be an object")
    out: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        key = str(raw_key)
        try:
            length = int(key)
        except ValueError as exc:
            raise ManifestError(f"{field} key is not an integer: {key!r}") from exc
        if length < 0 or str(length) != key:
            raise ManifestError(f"{field} key is not a canonical non-negative integer: {key!r}")
        try:
            count = _plain_int(raw_count, field=f"{field}.{key}")
        except ValueError as exc:
            raise ManifestError(str(exc)) from exc
        out[key] = count
    return dict(sorted(out.items(), key=lambda item: int(item[0])))


def _state_counts(value: Any, *, field: str) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise ManifestError(f"{field} must be a non-empty object")
    out: dict[str, int] = {}
    for raw_state, raw_count in value.items():
        if not isinstance(raw_state, str):
            raise ManifestError(f"{field} keys must be strings")
        try:
            out[raw_state] = _plain_int(raw_count, field=f"{field}.{raw_state}")
        except ValueError as exc:
            raise ManifestError(str(exc)) from exc
    return dict(sorted(out.items()))


def _check_equal(
    errors: list[str], *, field: str, expected: Any, actual: Any
) -> None:
    if actual != expected:
        errors.append(f"{field}: expected {expected!r}, got {actual!r}")


def _validate_condition(condition: Mapping[str, Any]) -> dict[str, Any]:
    if condition.get("schema") != CONDITION_SCHEMA:
        raise ConditionError(
            f"schema must be {CONDITION_SCHEMA!r}, got {condition.get('schema')!r}"
        )
    sweep_id = condition.get("sweep_id")
    if not isinstance(sweep_id, str) or not sweep_id:
        raise ConditionError("sweep_id must be a non-empty string")
    games = _string_list(condition.get("games"), field="games")

    dataset = condition.get("dataset")
    if not isinstance(dataset, dict):
        raise ConditionError("dataset must be an object")
    loader = dataset.get("loader")
    if loader != EXPECTED_LOADER:
        raise ConditionError(
            f"dataset.loader must be the v2 loader {EXPECTED_LOADER!r}, got {loader!r}"
        )
    try:
        episodes = _plain_int(
            dataset.get("episodes_per_game"), field="dataset.episodes_per_game", minimum=1
        )
        max_actions = _plain_int(
            dataset.get("max_actions"), field="dataset.max_actions", minimum=2
        )
        target_level = _plain_int(
            dataset.get("target_level"), field="dataset.target_level", minimum=1
        )
    except ValueError as exc:
        raise ConditionError(str(exc)) from exc

    factor = condition.get("sweep_factor")
    if not isinstance(factor, dict) or factor.get("name") != "candidate_suffix_length":
        raise ConditionError("sweep_factor.name must be 'candidate_suffix_length'")
    try:
        horizons = [
            _plain_int(value, field="sweep_factor.values[]", minimum=1)
            for value in factor.get("values", [])
        ]
        seeds = [
            _plain_int(value, field="sweep_factor.paired_seeds[]", minimum=0)
            for value in factor.get("paired_seeds", [])
        ]
    except (TypeError, ValueError) as exc:
        raise ConditionError(str(exc)) from exc
    if (
        not horizons
        or not seeds
        or len(set(horizons)) != len(horizons)
        or len(set(seeds)) != len(seeds)
    ):
        raise ConditionError("sweep horizons and paired seeds must be non-empty and unique")

    runs = condition.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ConditionError("runs must be a non-empty list")
    run_ids: set[str] = set()
    run_cells: set[tuple[int, int]] = set()
    normalized_runs: list[dict[str, Any]] = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ConditionError(f"runs[{index}] must be an object")
        run_id = run.get("id")
        if not isinstance(run_id, str) or not run_id:
            raise ConditionError(f"runs[{index}].id must be a non-empty string")
        if run_id in run_ids:
            raise ConditionError(f"duplicate run id: {run_id}")
        run_ids.add(run_id)
        try:
            seed = _plain_int(run.get("seed"), field=f"runs[{index}].seed")
            suffix = _plain_int(
                run.get("candidate_suffix_length"),
                field=f"runs[{index}].candidate_suffix_length",
                minimum=1,
            )
        except ValueError as exc:
            raise ConditionError(str(exc)) from exc
        if seed not in seeds or suffix not in horizons:
            raise ConditionError(
                f"run {run_id!r} cell ({suffix}, {seed}) is outside sweep_factor"
            )
        cell = (suffix, seed)
        if cell in run_cells:
            raise ConditionError(f"duplicate sweep cell candidate_suffix_length/seed={cell}")
        run_cells.add(cell)
        normalized_runs.append(
            {
                "id": run_id,
                "manifest": run.get("manifest"),
                "seed": seed,
                "candidate_suffix_length": suffix,
                "episodes_per_game": episodes,
                "max_actions": max_actions,
                "target_level": target_level,
                "collector": EXPECTED_COLLECTOR,
            }
        )

    expected_cells = {(suffix, seed) for suffix in horizons for seed in seeds}
    if run_cells != expected_cells:
        missing = sorted(expected_cells.difference(run_cells))
        extra = sorted(run_cells.difference(expected_cells))
        raise ConditionError(
            f"runs do not cover the declared sweep grid; missing={missing}, extra={extra}"
        )

    return {
        "sweep_id": sweep_id,
        "games": games,
        "dataset": {
            "loader": loader,
            "collector": EXPECTED_COLLECTOR,
            "episodes_per_game": episodes,
            "max_actions": max_actions,
            "target_level": target_level,
        },
        "runs": normalized_runs,
    }


def _summarize_game(
    receipt: Any,
    *,
    game_id: str,
    episodes_per_game: int,
    candidate_suffix_length: int,
    target_level: int,
) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise ManifestError(f"search.per_game.{game_id} must be an object")
    try:
        episodes = _plain_int(receipt.get("episodes"), field=f"{game_id}.episodes", minimum=1)
        best_level = _plain_int(receipt.get("best_level"), field=f"{game_id}.best_level")
        bootstrap_failures = _plain_int(
            receipt.get("bootstrap_failures"), field=f"{game_id}.bootstrap_failures"
        )
        step_errors = _plain_int(receipt.get("step_errors"), field=f"{game_id}.step_errors")
        requested = _plain_int(
            receipt.get("candidate_suffix_length_requested"),
            field=f"{game_id}.candidate_suffix_length_requested",
            minimum=1,
        )
        effective = _plain_int(
            receipt.get("candidate_suffix_length_effective"),
            field=f"{game_id}.candidate_suffix_length_effective",
            minimum=1,
        )
    except ValueError as exc:
        raise ManifestError(str(exc)) from exc
    if episodes != episodes_per_game:
        raise ManifestError(
            f"{game_id}.episodes: expected {episodes_per_game!r}, got {episodes!r}"
        )
    if requested != candidate_suffix_length:
        raise ManifestError(
            f"{game_id}.candidate_suffix_length_requested: expected "
            f"{candidate_suffix_length!r}, got {requested!r}"
        )
    if effective > requested:
        raise ManifestError(
            f"{game_id}.candidate_suffix_length_effective exceeds requested horizon"
        )
    if receipt.get("candidate_suffix_mode") != EXPECTED_SUFFIX_MODE:
        raise ManifestError(
            f"{game_id}.candidate_suffix_mode is not {EXPECTED_SUFFIX_MODE!r}"
        )

    states = _state_counts(receipt.get("states"), field=f"{game_id}.states")
    executed = _count_distribution(
        receipt.get("executed_suffix_length_distribution"),
        field=f"{game_id}.executed_suffix_length_distribution",
    )
    proposed = _count_distribution(
        receipt.get("candidate_suffix_length_distribution"),
        field=f"{game_id}.candidate_suffix_length_distribution",
    )
    if sum(states.values()) != episodes:
        raise ManifestError(
            f"{game_id}.states counts sum to {sum(states.values())}, expected {episodes}"
        )
    if sum(executed.values()) != episodes:
        raise ManifestError(
            f"{game_id}.executed_suffix_length_distribution sums to "
            f"{sum(executed.values())}, expected {episodes}"
        )
    if sum(proposed.values()) != episodes or set(proposed) != {str(effective)}:
        raise ManifestError(
            f"{game_id}.candidate_suffix_length_distribution must contain exactly "
            f"{episodes} proposals at effective horizon {effective}"
        )
    if bootstrap_failures > episodes or step_errors > episodes:
        raise ManifestError(f"{game_id} failure counts exceed episode count")

    target_programs = receipt.get("target_programs")
    if not isinstance(target_programs, list):
        raise ManifestError(f"{game_id}.target_programs must be a list")
    if len(target_programs) > episodes:
        raise ManifestError(f"{game_id}.target_programs exceeds episode count")
    if bool(target_programs) != (best_level >= target_level):
        raise ManifestError(
            f"{game_id}.target_programs/best_level disagree at target_level={target_level}"
        )
    if executed and max(int(length) for length in executed) > effective:
        raise ManifestError(
            f"{game_id}.executed_suffix_length_distribution exceeds effective horizon"
        )

    try:
        nonterminal = sum(
            count
            for state, count in states.items()
            if _state_bucket(state) == "not_finished"
        )
    except ManifestError as exc:
        raise ManifestError(f"{game_id}.states: {exc}") from exc
    terminal = episodes - nonterminal
    return {
        "episodes": episodes,
        "target_programs": len(target_programs),
        "best_level": best_level,
        "bootstrap_failures": bootstrap_failures,
        "step_errors": step_errors,
        "states": states,
        "nonterminal_episodes": nonterminal,
        "terminal_episodes": terminal,
        "nonterminal_rate": nonterminal / episodes,
        "candidate_suffix_length_requested": requested,
        "candidate_suffix_length_effective": effective,
        "executed_suffix_length_distribution": executed,
    }


def _summarize_manifest(
    manifest: Mapping[str, Any], *, expected: Mapping[str, Any], games: Sequence[str]
) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    _check_equal(
        errors,
        field="collector",
        expected=expected["collector"],
        actual=manifest.get("collector"),
    )
    _check_equal(errors, field="seed", expected=expected["seed"], actual=manifest.get("seed"))
    _check_equal(
        errors,
        field="candidate_suffix_length",
        expected=expected["candidate_suffix_length"],
        actual=manifest.get("candidate_suffix_length"),
    )
    _check_equal(
        errors,
        field="episodes_per_game",
        expected=expected["episodes_per_game"],
        actual=manifest.get("episodes_per_game"),
    )
    _check_equal(
        errors,
        field="max_actions",
        expected=expected["max_actions"],
        actual=manifest.get("max_actions"),
    )
    _check_equal(
        errors,
        field="target_level",
        expected=expected["target_level"],
        actual=manifest.get("target_level"),
    )
    _check_equal(
        errors,
        field="candidate_suffix_mode",
        expected=EXPECTED_SUFFIX_MODE,
        actual=manifest.get("candidate_suffix_mode"),
    )
    _check_equal(
        errors,
        field="all_installed_games",
        expected=True,
        actual=manifest.get("all_installed_games"),
    )

    search = manifest.get("search")
    if not isinstance(search, dict):
        errors.append("search: expected an object")
        return None, errors
    _check_equal(errors, field="search.games", expected=list(games), actual=search.get("games"))
    _check_equal(
        errors,
        field="search.all_installed_games",
        expected=True,
        actual=search.get("all_installed_games"),
    )
    _check_equal(
        errors,
        field="search.candidate_suffix_mode",
        expected=EXPECTED_SUFFIX_MODE,
        actual=search.get("candidate_suffix_mode"),
    )
    _check_equal(
        errors,
        field="search.candidate_suffix_length",
        expected=expected["candidate_suffix_length"],
        actual=search.get("candidate_suffix_length"),
    )
    per_game = search.get("per_game")
    if not isinstance(per_game, dict):
        errors.append("search.per_game: expected an object")
        return None, errors
    _check_equal(
        errors,
        field="search.per_game keys",
        expected=sorted(games),
        actual=sorted(per_game),
    )
    if errors:
        return None, errors

    game_summaries: dict[str, dict[str, Any]] = {}
    for game_id in games:
        try:
            game_summaries[game_id] = _summarize_game(
                per_game.get(game_id),
                game_id=game_id,
                episodes_per_game=int(expected["episodes_per_game"]),
                candidate_suffix_length=int(expected["candidate_suffix_length"]),
                target_level=int(expected["target_level"]),
            )
        except ManifestError as exc:
            errors.append(str(exc))
    if errors:
        return None, errors

    target_counts = [game_summaries[game]["target_programs"] for game in games]
    nonterminal_rates = [game_summaries[game]["nonterminal_rate"] for game in games]
    total_episodes = sum(game_summaries[game]["episodes"] for game in games)
    total_nonterminal = sum(game_summaries[game]["nonterminal_episodes"] for game in games)
    aggregate = {
        "min_per_game_targets": min(target_counts),
        "total_targets": sum(target_counts),
        "min_per_game_nonterminal_rate": min(nonterminal_rates),
        "total_nonterminal_rate": total_nonterminal / total_episodes,
        "total_bootstrap_failures": sum(
            game_summaries[game]["bootstrap_failures"] for game in games
        ),
        "total_step_errors": sum(game_summaries[game]["step_errors"] for game in games),
    }
    aggregate["zero_failure_gate"] = (
        aggregate["total_bootstrap_failures"] == 0
        and aggregate["total_step_errors"] == 0
    )
    return {"per_game": game_summaries, "aggregate": aggregate}, []


def summarize_sweep(
    condition: Mapping[str, Any], *, condition_path: Path, workspace_root: Path = ROOT
) -> dict[str, Any]:
    """Summarize manifests named by ``condition`` without using external state."""

    resolved = _validate_condition(condition)
    run_summaries: list[dict[str, Any]] = []
    rankable: list[dict[str, Any]] = []
    for declaration in resolved["runs"]:
        run_id = declaration["id"]
        manifest_path = _resolve_workspace_path(
            declaration["manifest"],
            field=f"runs[{run_id}].manifest",
            workspace_root=workspace_root,
        )
        record: dict[str, Any] = {
            "id": run_id,
            "manifest": str(manifest_path),
            "expected": {
                key: declaration[key]
                for key in (
                    "seed",
                    "candidate_suffix_length",
                    "episodes_per_game",
                    "max_actions",
                    "target_level",
                    "collector",
                )
            },
        }
        if not manifest_path.is_file():
            record.update(
                status="missing_manifest",
                rankable=False,
                errors=[f"declared manifest does not exist: {manifest_path}"],
            )
            run_summaries.append(record)
            continue
        record["manifest_sha256"] = _sha256(manifest_path)
        try:
            manifest = _read_object(manifest_path, label=f"run {run_id} manifest")
        except ValueError as exc:
            record.update(status="invalid_manifest", rankable=False, errors=[str(exc)])
            run_summaries.append(record)
            continue
        summary, errors = _summarize_manifest(
            manifest, expected=declaration, games=resolved["games"]
        )
        if errors or summary is None:
            record.update(status="mismatched", rankable=False, errors=errors)
            run_summaries.append(record)
            continue
        record.update(status="complete", rankable=True, errors=[], **summary)
        run_summaries.append(record)
        rankable.append(record)
        del manifest

    ranked = sorted(
        rankable,
        key=lambda run: (
            -run["aggregate"]["min_per_game_targets"],
            -run["aggregate"]["total_targets"],
            -run["aggregate"]["min_per_game_nonterminal_rate"],
            -run["aggregate"]["total_nonterminal_rate"],
            run["id"],
        ),
    )
    ranking = [
        {
            "rank": index,
            "id": run["id"],
            **run["aggregate"],
        }
        for index, run in enumerate(ranked, start=1)
    ]
    incomplete_ids = [run["id"] for run in run_summaries if not run["rankable"]]
    complete = not incomplete_ids and len(run_summaries) == len(resolved["runs"])
    return {
        "schema": SUMMARY_SCHEMA,
        "sweep_id": resolved["sweep_id"],
        "status": "complete" if complete else "incomplete",
        "complete": complete,
        "truth_boundary": (
            "Only manifests declared by the sweep condition are read. Missing, malformed, "
            "or setting-mismatched runs are excluded from ranking and make this summary incomplete."
        ),
        "condition": str(condition_path.resolve()),
        "games": resolved["games"],
        "resolved_dataset": resolved["dataset"],
        "declared_runs": len(resolved["runs"]),
        "complete_runs": len(rankable),
        "incomplete_run_ids": incomplete_ids,
        "ranking_rule": [
            "min_per_game_targets descending",
            "total_targets descending",
            "min_per_game_nonterminal_rate descending",
            "total_nonterminal_rate descending",
            "run id ascending deterministic tie break",
        ],
        "ranking_admits_complete_matching_manifests_only": True,
        "ranking": ranking,
        "runs": run_summaries,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    condition_path = args.condition.expanduser().resolve()
    output_path = args.out.expanduser().resolve()

    try:
        condition = _read_object(condition_path, label="sweep condition")
        summary = summarize_sweep(
            condition, condition_path=condition_path, workspace_root=ROOT
        )
        summary["provenance"] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "condition_sha256": _sha256(condition_path),
            "summarizer": str(Path(__file__).resolve()),
            "summarizer_sha256": _sha256(Path(__file__).resolve()),
            "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "working_directory": str(Path.cwd()),
        }
        exit_code = 0 if summary["complete"] else 1
    except (ConditionError, ValueError) as exc:
        summary = {
            "schema": SUMMARY_SCHEMA,
            "status": "incomplete",
            "complete": False,
            "truth_boundary": "Condition invalid; no run is admitted or ranked.",
            "condition": str(condition_path),
            "condition_errors": [str(exc)],
            "ranking": [],
            "runs": [],
            "provenance": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "summarizer": str(Path(__file__).resolve()),
                "summarizer_sha256": _sha256(Path(__file__).resolve()),
                "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                "working_directory": str(Path.cwd()),
            },
        }
        if condition_path.is_file():
            summary["provenance"]["condition_sha256"] = _sha256(condition_path)
        exit_code = 2
    _write_json(output_path, summary)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
