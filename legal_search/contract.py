"""Legal-consume contract for mined algorithms.

Human-coded *methods* are allowed. Pre-loaded *answers for a specific game*
are not. Every consume adapter must pass this contract before its outputs
may enter the legal agent loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

SCHEMA = "legal_search.contract.v1"

# Paths / symbols that must never feed the legal agent path.
BANNED_SOURCE_MARKERS: tuple[str, ...] = (
    "sp80_spill_model",
    "dropletsim",
    "native_scene",
    "environment_files/",  # game source as knowledge, not as env runner
    "human_mined",
    "answer_key",
    "route_replay",
    "known_win_route",
)

# Permanent fixture win banks for a public game (research ok; legal agent no).
BANNED_PRIOR_PATTERNS: tuple[str, ...] = (
    "dw_solve_r4_L",
    "dw_solve_r5_L",
    "env_win_genome_bank",
    "fixture_genome_prior",  # as permanent shipped prior
    "fixture_direct_prior",
)

TRUTH_BOUNDARY = {
    "control_truth_admitted": False,
    "goal_condition_admitted": False,
    "action_choice_admitted": False,
    "solve_condition_admitted": False,
    "game_specific_binding_admitted": False,
    "cross_game_memory_admitted": False,
}


@dataclass(frozen=True)
class LegalRule:
    """One enforceable legality rule."""

    rule_id: str
    statement: str
    severity: str = "hard"  # hard | soft


LEGAL_RULES: tuple[LegalRule, ...] = (
    LegalRule(
        "R1_method_not_answer",
        "Ship algorithms and interfaces, not game win conditions or source-derived rules.",
    ),
    LegalRule(
        "R2_env_is_judge",
        "Only the environment may confirm levels_completed / WIN / GAME_OVER.",
    ),
    LegalRule(
        "R3_episodic_memory",
        "Success memory is per game/scorecard; wipe on new game_id.",
    ),
    LegalRule(
        "R4_no_banned_lineage",
        "No spill_model / dropletsim / human route-replay / source transcription in the path.",
    ),
    LegalRule(
        "R5_bindings_induced",
        "Color/role/movable bindings are induced from interaction or left unknown — not hardcoded per game.",
        "soft",
    ),
    LegalRule(
        "R6_hypothesis_not_truth",
        "Perception, ports, oracle, ghost, bands emit hypotheses only; fail-closed admission.",
    ),
    LegalRule(
        "R7_private_game_test",
        "Same binary must be runnable on a private never-seen game without human edit.",
        "soft",
    ),
)


@dataclass
class ConsumeReceipt:
    """Receipt that a system was consumed under the legal contract."""

    system_id: str
    algorithm_id: str
    schema: str = SCHEMA
    truth_boundary: dict[str, bool] = field(default_factory=lambda: dict(TRUTH_BOUNDARY))
    stripped_bindings: list[str] = field(default_factory=list)
    allowed_outputs: list[str] = field(default_factory=list)
    banned_outputs: list[str] = field(default_factory=list)
    notes: str = ""
    legal: bool = True
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "system_id": self.system_id,
            "algorithm_id": self.algorithm_id,
            "legal": self.legal,
            "truth_boundary": dict(self.truth_boundary),
            "stripped_bindings": list(self.stripped_bindings),
            "allowed_outputs": list(self.allowed_outputs),
            "banned_outputs": list(self.banned_outputs),
            "notes": self.notes,
            "violations": list(self.violations),
        }


def scan_text_for_bans(text: str, *, extra: Iterable[str] = ()) -> list[str]:
    """Return ban markers found in free text (provenance scan)."""
    lowered = text.lower()
    hits: list[str] = []
    for marker in (*BANNED_SOURCE_MARKERS, *BANNED_PRIOR_PATTERNS, *extra):
        if marker.lower() in lowered:
            hits.append(marker)
    return hits


def assert_legal_receipt(receipt: ConsumeReceipt) -> None:
    if not receipt.legal or receipt.violations:
        raise RuntimeError(
            f"illegal consume of {receipt.system_id}: {receipt.violations}"
        )
    for key, value in TRUTH_BOUNDARY.items():
        if receipt.truth_boundary.get(key, False) is not False and value is False:
            # any True admission flag is a hard fail
            if receipt.truth_boundary.get(key) is True:
                raise RuntimeError(
                    f"illegal truth admission {key} on {receipt.system_id}"
                )
