"""Focused structural tests for the environment-scored prefix archive."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from legal_search.train.program_prefix_archive import (
    ActionGene,
    ArchiveEntry,
    PrefixArchive,
    PrefixSignals,
    archive_entry,
    complete_candidate_suffix,
    load_bootstrap_programs,
    mutate_prefix,
    resolve_gene,
)


def test_component_and_relative_genes_resolve_from_current_frame() -> None:
    frame = np.zeros((12, 16), dtype=np.int16)
    frame[3:7, 8:13] = 4

    component = ActionGene("component_rank", rank=0, u=0.5, v=0.5)
    action = resolve_gene(component, frame, [6])
    assert action.action_id == 6
    assert frame[action.y, action.x] == 4
    assert action.source == "prefix_archive.component_rank"

    relative = ActionGene("frame_relative", rank=0, u=1.0, v=0.0)
    action = resolve_gene(relative, frame, [6])
    assert action.as_tuple() == (6, 15, 0)
    assert action.source == "prefix_archive.frame_relative"


def test_available_rank_gene_never_embeds_a_game_action_route() -> None:
    frame = np.zeros((8, 8), dtype=np.int16)
    gene = ActionGene("available_rank", rank=5, u=0.25, v=0.75)
    action = resolve_gene(gene, frame, [1, 4, 6])
    assert action.as_tuple() == (4, None, None)

    # A component gene falls back generically when clicking is unavailable.
    action = resolve_gene(ActionGene("component_rank", 3, 0.5, 0.5), frame, [2, 5])
    assert action.action_id in {2, 5}
    assert action.source == "prefix_archive.available_rank_fallback"


def test_prefix_mutation_stays_in_generic_gene_space() -> None:
    rng = np.random.default_rng(17)
    prefix = mutate_prefix(
        (), rng=rng, max_length=12, initial_length=5, mutation_span=3
    )
    assert len(prefix) == 5
    for _ in range(20):
        prefix = mutate_prefix(
            prefix, rng=rng, max_length=12, initial_length=5, mutation_span=3
        )
        assert 1 <= len(prefix) <= 12
        assert all(isinstance(gene, ActionGene) for gene in prefix)
        assert all(0.0 <= gene.u <= 1.0 and 0.0 <= gene.v <= 1.0 for gene in prefix)


def test_horizon_completion_is_deterministic_and_preserves_mutated_prefix() -> None:
    prefix = (
        ActionGene("component_rank", 3, 0.2, 0.4),
        ActionGene("available_rank", 7, 0.6, 0.8),
    )
    first = complete_candidate_suffix(
        prefix,
        rng=np.random.default_rng(123),
        candidate_suffix_length=7,
        max_length=10,
    )
    second = complete_candidate_suffix(
        prefix,
        rng=np.random.default_rng(123),
        candidate_suffix_length=7,
        max_length=10,
    )
    assert first == second
    assert first[: len(prefix)] == prefix
    assert len(first) == 7
    assert all(isinstance(gene, ActionGene) for gene in first)


def test_horizon_completion_clips_to_per_game_maximum() -> None:
    prefix = (ActionGene("frame_relative", 1, 0.1, 0.9),)
    completed = complete_candidate_suffix(
        prefix,
        rng=np.random.default_rng(9),
        candidate_suffix_length=40,
        max_length=5,
    )
    assert len(completed) == 5
    assert completed[0] == prefix[0]


def test_zero_horizon_preserves_v1_prefix_and_rng_state() -> None:
    prefix = (
        ActionGene("component_rank", 2, 0.25, 0.75),
        ActionGene("frame_relative", 4, 0.5, 0.5),
    )
    used = np.random.default_rng(88)
    control = np.random.default_rng(88)
    completed = complete_candidate_suffix(
        prefix,
        rng=used,
        candidate_suffix_length=0,
        max_length=1,
    )
    assert completed == prefix
    assert used.random() == control.random()


def test_archive_admission_is_observation_scored() -> None:
    before = np.zeros((8, 8), dtype=np.int16)
    after = before.copy()
    after[2:4, 2:4] = 3
    signals = PrefixSignals(start_level=1, levels=1, state="NOT_FINISHED")
    signals.observe(before, after, levels=1, state="NOT_FINISHED")
    genes = (ActionGene("frame_relative", 0, 0.25, 0.25),)
    observed = archive_entry(game_id="arbitrary-installed-game", genes=genes, signals=signals)

    archive = PrefixArchive(4, retain_games=False)
    assert archive.add(observed)
    assert not archive.add(observed)

    improved = ArchiveEntry(
        game_id=observed.game_id,
        genes=observed.genes,
        descriptor=observed.descriptor,
        score=(2, *observed.score[1:]),
        observations={**observed.observations, "max_level": 2},
    )
    assert archive.add(improved)
    assert archive.replaced == 1
    assert next(iter(archive.entries.values())).observations["max_level"] == 2


def test_bootstrap_loader_resolves_winning_program_and_hashes_source(tmp_path: Path) -> None:
    game_id = "game-family-build"
    artifact = tmp_path / "data.manifest.json"
    artifact.write_text(
        json.dumps(
            {
                "search": {
                    "per_game": {
                        game_id: {
                            "winning_programs": [
                                [[1, 9, 9], [6, 4, 5]],
                            ]
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    memory = tmp_path / "success_programs.json"
    memory.write_text(
        json.dumps(
            {
                "schema": "legal_search.success_programs.v1",
                "truth_boundary": "controlled environment level counter advanced",
                "program_sources": {
                    game_id: {"artifact": str(artifact), "winner_index": 0}
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = load_bootstrap_programs(memory)
    # Coordinates on non-click actions are normalized away; click coordinates
    # remain exact replay data.
    assert bundle.programs[game_id] == [[1, None, None], [6, 4, 5]]
    assert len(bundle.provenance["sha256"]) == 64
    source = bundle.provenance["sources"][game_id]
    assert source["winner_index"] == 0
    assert len(source["artifact_sha256"]) == 64
