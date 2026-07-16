"""Pure tests for exact target-program loading and provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_search.train.replay_target_program import load_target_program


def _manifest(tmp_path: Path, program: list) -> Path:
    path = tmp_path / "data.manifest.json"
    path.write_text(
        json.dumps(
            {
                "search": {
                    "per_game": {
                        "game-build": {"target_programs": [program]}
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_loads_exact_program_and_stable_hash(tmp_path: Path) -> None:
    path = _manifest(tmp_path, [[1, None, None], [6, 3, 4]])
    first, source = load_target_program(path, game_id="game-build", target_index=0)
    second, source2 = load_target_program(path, game_id="game-build", target_index=0)
    assert first == [[1, None, None], [6, 3, 4]]
    assert first == second
    assert source["program_sha256"] == source2["program_sha256"]
    assert source["program_actions"] == 2
    assert len(source["sha256"]) == 64


@pytest.mark.parametrize(
    "program",
    [
        [],
        [[7, None, None]],
        [[6, None, 2]],
        [[6, 64, 2]],
        [[1, 2, None]],
        [[1, None]],
    ],
)
def test_rejects_invalid_programs(tmp_path: Path, program: list) -> None:
    path = _manifest(tmp_path, program)
    with pytest.raises(ValueError):
        load_target_program(path, game_id="game-build", target_index=0)


def test_rejects_missing_game_or_index(tmp_path: Path) -> None:
    path = _manifest(tmp_path, [[1, None, None]])
    with pytest.raises(ValueError):
        load_target_program(path, game_id="other", target_index=0)
    with pytest.raises(ValueError):
        load_target_program(path, game_id="game-build", target_index=1)
