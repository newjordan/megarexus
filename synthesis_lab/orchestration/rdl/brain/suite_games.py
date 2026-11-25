"""Public multi-game suite (24) for level-group training.

Source of truth for game IDs under other_games/environment_files.
Not SP80-centric. Used so MegaMind trains robust patterns across ALL games
at once within a level band (e.g. L0–L1 on every game in one cohort).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

_REPO = Path(__file__).resolve().parents[4]
DEFAULT_SUITE_ENV_DIR = (
    _REPO / "synthesis_lab" / "experiments" / "other_games" / "environment_files"
)
# Optional extra roots (never required for the 24)
EXTRA_ENV_DIRS: tuple[Path, ...] = (
    _REPO / "environment_files",
)


def discover_game_ids(env_dir: Path | None = None) -> list[str]:
    """Return sorted full game ids ``prefix-ident`` under env_dir."""
    root = Path(env_dir) if env_dir is not None else DEFAULT_SUITE_ENV_DIR
    if not root.is_dir():
        return []
    ids: list[str] = []
    for prefix in sorted(p for p in root.iterdir() if p.is_dir()):
        for ident in sorted(p for p in prefix.iterdir() if p.is_dir()):
            if any(ident.glob("*.py")):
                ids.append(f"{prefix.name}-{ident.name}")
    return ids


def suite_24(env_dir: Path | None = None) -> list[str]:
    """Every game directory under ``env_dir``.

    The name is historical. Training notes aimed at 24 games; the returned
    list is whatever ``prefix/ident`` folders contain a ``.py`` file.
    """
    return discover_game_ids(env_dir)


def prefix_of(game_id: str) -> str:
    return game_id.split("-", 1)[0] if game_id else ""


def round_robin(groups: dict[str, list], *, max_per: int | None = None) -> list:
    """Interleave lists by key so all games appear in one stream (robust batching)."""
    keys = sorted(groups.keys())
    pointers = {k: 0 for k in keys}
    out: list = []
    if max_per is not None:
        caps = {k: min(len(groups[k]), max_per) for k in keys}
    else:
        caps = {k: len(groups[k]) for k in keys}
    while True:
        progressed = False
        for k in keys:
            i = pointers[k]
            if i < caps[k]:
                out.append(groups[k][i])
                pointers[k] = i + 1
                progressed = True
        if not progressed:
            break
    return out


def level_groups_default() -> list[tuple[int, int]]:
    """Default training cohorts: adjacent level pairs.

    Group = inclusive band [lo, hi]. First cohort is L0+L1 on all games.
    """
    return [(0, 1), (2, 3), (4, 5), (6, 7)]


def describe_suite(game_ids: Iterable[str] | None = None) -> dict:
    ids = list(game_ids) if game_ids is not None else suite_24()
    return {
        "n_games": len(ids),
        "game_ids": ids,
        "prefixes": [prefix_of(g) for g in ids],
        "env_dir": str(DEFAULT_SUITE_ENV_DIR),
        "doctrine": "level_groups_across_all_games_for_robust_patterns",
        "not_sp80_focus": True,
    }
