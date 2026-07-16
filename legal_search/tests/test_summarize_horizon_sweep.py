"""Pure tests for the fail-closed horizon sweep summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_search.train.summarize_horizon_sweep import (
    CONDITION_SCHEMA,
    EXPECTED_COLLECTOR,
    EXPECTED_LOADER,
    ConditionError,
    summarize_sweep,
)


GAMES = ["ft09-build", "sp80-build"]


def _game_receipt(
    *, targets: int, nonterminal: int, horizon: int, episodes: int = 4
) -> dict:
    terminal = episodes - nonterminal
    states = {}
    if nonterminal:
        states["GameState.NOT_FINISHED"] = nonterminal
    if terminal:
        states["GameState.GAME_OVER"] = terminal
    return {
        "episodes": episodes,
        "best_level": 2 if targets else 1,
        "bootstrap_failures": 0,
        "step_errors": 0,
        "states": states,
        "target_programs": [[[6, 1, 1]]] * targets,
        "candidate_suffix_mode": "v2_horizon_complete",
        "candidate_suffix_length_requested": horizon,
        "candidate_suffix_length_effective": horizon,
        "candidate_suffix_length_distribution": {str(horizon): episodes},
        "executed_suffix_length_distribution": {str(horizon): nonterminal, "2": terminal},
    }


def _manifest(
    *,
    seed: int,
    horizon: int,
    target_counts: tuple[int, int] = (0, 0),
    nonterminal_counts: tuple[int, int] = (4, 4),
) -> dict:
    return {
        "collector": EXPECTED_COLLECTOR,
        "seed": seed,
        "candidate_suffix_length": horizon,
        "episodes_per_game": 4,
        "max_actions": 220,
        "target_level": 2,
        "candidate_suffix_mode": "v2_horizon_complete",
        "all_installed_games": True,
        "search": {
            "games": GAMES,
            "all_installed_games": True,
            "candidate_suffix_mode": "v2_horizon_complete",
            "candidate_suffix_length": horizon,
            "per_game": {
                game: _game_receipt(
                    targets=target_counts[index],
                    nonterminal=nonterminal_counts[index],
                    horizon=horizon,
                )
                for index, game in enumerate(GAMES)
            },
        },
    }


def _condition(runs: list[dict], *, horizons: list[int], seeds: list[int]) -> dict:
    return {
        "schema": CONDITION_SCHEMA,
        "sweep_id": "pure-test-sweep",
        "games": GAMES,
        "dataset": {
            "loader": EXPECTED_LOADER,
            "episodes_per_game": 4,
            "max_actions": 220,
            "target_level": 2,
        },
        "sweep_factor": {
            "name": "candidate_suffix_length",
            "values": horizons,
            "paired_seeds": seeds,
        },
        "runs": runs,
    }


def _run(run_id: str, path: Path, *, seed: int, horizon: int) -> dict:
    return {
        "id": run_id,
        "manifest": path.name,
        "seed": seed,
        "candidate_suffix_length": horizon,
    }


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_complete_sweep_ranks_horizontal_targets_then_survival(tmp_path: Path) -> None:
    # c and a both solve both games; c wins their tie on horizontal survival.
    # b has the same total target count but collapses one game and ranks last.
    paths = {name: tmp_path / f"{name}.json" for name in ("a", "b", "c")}
    _write(
        paths["a"],
        _manifest(seed=1, horizon=64, target_counts=(1, 1), nonterminal_counts=(2, 2)),
    )
    _write(
        paths["b"],
        _manifest(seed=2, horizon=64, target_counts=(2, 0), nonterminal_counts=(4, 4)),
    )
    _write(
        paths["c"],
        _manifest(seed=3, horizon=64, target_counts=(1, 1), nonterminal_counts=(4, 4)),
    )
    condition = _condition(
        [
            _run("a", paths["a"], seed=1, horizon=64),
            _run("b", paths["b"], seed=2, horizon=64),
            _run("c", paths["c"], seed=3, horizon=64),
        ],
        horizons=[64],
        seeds=[1, 2, 3],
    )

    summary = summarize_sweep(
        condition, condition_path=tmp_path / "condition.json", workspace_root=tmp_path
    )

    assert summary["status"] == "complete"
    assert summary["complete_runs"] == 3
    assert [item["id"] for item in summary["ranking"]] == ["c", "a", "b"]
    assert summary["ranking"][0]["min_per_game_targets"] == 1
    assert summary["runs"][0]["per_game"][GAMES[0]]["target_programs"] == 1
    distribution = summary["runs"][0]["per_game"][GAMES[0]][
        "executed_suffix_length_distribution"
    ]
    assert distribution == {"2": 2, "64": 2}


def test_missing_manifest_marks_whole_sweep_incomplete_and_is_not_ranked(tmp_path: Path) -> None:
    present = tmp_path / "present.json"
    missing = tmp_path / "missing.json"
    _write(present, _manifest(seed=1, horizon=64))
    condition = _condition(
        [
            _run("present", present, seed=1, horizon=64),
            _run("missing", missing, seed=2, horizon=64),
        ],
        horizons=[64],
        seeds=[1, 2],
    )

    summary = summarize_sweep(
        condition, condition_path=tmp_path / "condition.json", workspace_root=tmp_path
    )

    assert summary["status"] == "incomplete"
    assert summary["incomplete_run_ids"] == ["missing"]
    assert [item["id"] for item in summary["ranking"]] == ["present"]
    assert summary["runs"][1]["status"] == "missing_manifest"


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("seed", 999),
        ("candidate_suffix_length", 96),
        ("episodes_per_game", 3),
        ("max_actions", 219),
        ("collector", "environment_scored_program_prefix_archive_v1"),
    ],
)
def test_key_resolved_setting_mismatch_is_fail_closed(
    tmp_path: Path, field: str, bad_value: object
) -> None:
    path = tmp_path / "run.json"
    manifest = _manifest(seed=1, horizon=64)
    manifest[field] = bad_value
    _write(path, manifest)
    condition = _condition(
        [_run("run", path, seed=1, horizon=64)], horizons=[64], seeds=[1]
    )

    summary = summarize_sweep(
        condition, condition_path=tmp_path / "condition.json", workspace_root=tmp_path
    )

    assert summary["status"] == "incomplete"
    assert summary["ranking"] == []
    assert summary["runs"][0]["status"] == "mismatched"
    assert any(field in error for error in summary["runs"][0]["errors"])


def test_internally_inconsistent_receipt_is_not_admitted(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    manifest = _manifest(seed=1, horizon=64)
    manifest["search"]["per_game"][GAMES[0]]["states"] = {"GameState.NOT_FINISHED": 3}
    _write(path, manifest)
    condition = _condition(
        [_run("run", path, seed=1, horizon=64)], horizons=[64], seeds=[1]
    )

    summary = summarize_sweep(
        condition, condition_path=tmp_path / "condition.json", workspace_root=tmp_path
    )

    assert summary["status"] == "incomplete"
    assert summary["runs"][0]["status"] == "mismatched"
    assert "states counts sum" in summary["runs"][0]["errors"][0]


def test_unknown_state_is_not_credited_as_nonterminal(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    manifest = _manifest(seed=1, horizon=64)
    manifest["search"]["per_game"][GAMES[0]]["states"] = {"MYSTERY": 4}
    _write(path, manifest)
    condition = _condition(
        [_run("run", path, seed=1, horizon=64)], horizons=[64], seeds=[1]
    )

    summary = summarize_sweep(
        condition, condition_path=tmp_path / "condition.json", workspace_root=tmp_path
    )

    assert summary["status"] == "incomplete"
    assert "unrecognized environment state" in summary["runs"][0]["errors"][0]


def test_condition_requires_complete_declared_factor_grid(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    condition = _condition(
        [_run("only", path, seed=1, horizon=64)], horizons=[64, 96], seeds=[1]
    )

    with pytest.raises(ConditionError, match="do not cover"):
        summarize_sweep(
            condition, condition_path=tmp_path / "condition.json", workspace_root=tmp_path
        )
