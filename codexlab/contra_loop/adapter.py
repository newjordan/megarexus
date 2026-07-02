#!/usr/bin/env python3
"""GameAdapter protocol + SP80Adapter reference impl + GAME_TABLE dispatch.

Replaces the cockpit's hardcoded GAME_ID / CALIBRATION / HARNESS_SRC constants
(contra.py:36-39) with a per-game dispatch table that the loop, gate, and
LoopOrchestrator can thread through every call site.

Design intent (from CONTRA_LOOP.md / adaptation_interface spec):
  - run_contra_view (controller_head.py:315) is already fully game-generic.
    Its SP80 coupling lives only in callers. This file makes that seam explicit.
  - A new game implements GameAdapter (4 deterministic onboarding steps, no
    contra_head edits required), registers itself in GAME_TABLE, and the whole
    loop/gate/registry machinery works without change.
  - The loop threads adapter.game_id / calibration_for_level(level_index) /
    episode_verifier() through every episode call.

Honest limit (structural, not proven):
  - 13+ literal 'sp80-589a99af' strings in sp80_player_harness.py and
    SP80_PATH at sp80_base_controller.py:32 must be threaded from
    adapter.game_id before a second game truly runs the full episode loop.
  - Until a second players/<game>/ directory exists, the multi-game claim is
    structural. GAME_TABLE centralises the dispatch; the harness threading is a
    separate, tracked TODO.

Usage (day-one sanity check, no episode or solve required):
    python codexlab/contra_loop/adapter.py --dry-run
    python codexlab/contra_loop/adapter.py --list-games
    python codexlab/contra_loop/adapter.py --list-adapters
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

# ---------------------------------------------------------------------------
# Repo paths (mirrors ROOT/CODEXLAB pattern used throughout codexlab).
# ---------------------------------------------------------------------------
import sys as _sys

ROOT = Path(__file__).resolve().parents[2]
CODEXLAB = ROOT / "codexlab"
ENV_ROOT = ROOT / "environment" / "environment_files"

if str(CODEXLAB) not in _sys.path:
    _sys.path.insert(0, str(CODEXLAB))

# Safe top-level imports: no heavy deps (torch / PIL / arcengine).
from players.base import GamePlayerDescriptor, PrimitiveRegistry  # noqa: E402
from players.sp80.registry import PRIMITIVE_REGISTRY as _SP80_REGISTRY  # noqa: E402

# ---------------------------------------------------------------------------
# Schema tag embedded in dry-run JSON output.
# ---------------------------------------------------------------------------
SCHEMA = "codexlab.contra_loop.adapter.v1"

# ---------------------------------------------------------------------------
# GroundTruthSource -- literal type for GameAdapter.ground_truth() return value.
# Ordered by preference: win_bank > engine_source > raw_recordings > none.
# ---------------------------------------------------------------------------
GroundTruthSource = Literal["win_bank", "engine_source", "raw_recordings", "none"]


# ---------------------------------------------------------------------------
# GameObs: minimal observation returned by new_game() / perform_action().
# The ONLY solve signal the loop uses is levels_completed (cardinal rule).
# ---------------------------------------------------------------------------
@dataclass
class GameObs:
    """Minimal observation returned by GameAdapter.new_game / perform_action.

    The loop reads ONLY levels_completed as solve evidence (cardinal rule:
    never trust status strings; levels_completed_delta > 0 is the only proof).
    frame is a 2D int16 np.ndarray of raw pixel values.
    """

    frame: Any  # np.ndarray dtype=int16, shape=(H, W)
    levels_completed: int
    state: str = "unknown"  # game state label, informational only
    steps_remaining: int | None = None


# ---------------------------------------------------------------------------
# GameAdapter: Protocol.  Every method has a corresponding SP80 implementation
# below as the reference.  New games subclass or implement this protocol.
# ---------------------------------------------------------------------------
class GameAdapter:
    """Protocol for per-game onboarding seam.

    All members are backed by concrete seams in the existing codebase:
      game_id          -- threaded into run_contra_view (controller_head.py:315)
      calibration      -- full calibration payload or None (uncalibrated mode)
      primitive_registry() -- players/base.py:86 PrimitiveRegistry
      new_game()       -- returns reset frame + levels_completed=0
      perform_action() -- returns next frame + current levels_completed
      episode_verifier()  -- returns verify_run callable (sp80_verify_episode_run.py:52)
      ground_truth()   -- 'win_bank' | 'engine_source' | 'raw_recordings' | 'none'

    These are called by the loop; the loop never calls the episode runner
    (sp80_run_contra_levels.py) directly -- that goes through contra.py 'run'.
    """

    # Concrete subclasses must set these as class or instance attributes.
    game_id: str
    _calibration_payload: dict[str, Any] | None

    @property
    def calibration(self) -> dict[str, Any] | None:
        """Full calibration payload.  None => uncalibrated mode (valid)."""
        return self._calibration_payload

    def calibration_for_level(self, level_index: int) -> dict[str, Any] | None:
        """Extract the per-level calibration row.

        Mirrors contra.py _load_calibration(level_index) logic.
        Returns None if no matching row exists (uncalibrated for this level).
        """
        payload = self.calibration
        if payload is None:
            return None
        for row in payload.get("levels", []):
            if isinstance(row, dict) and int(row.get("level_index", -1)) == level_index:
                return row
        return None

    def primitive_registry(self) -> PrimitiveRegistry:
        raise NotImplementedError

    def new_game(self, level_index: int) -> GameObs:
        raise NotImplementedError

    def perform_action(
        self, primitive: str, *, data: dict[str, Any] | None = None
    ) -> GameObs:
        raise NotImplementedError

    def episode_verifier(self) -> Callable[[Path], dict[str, Any]]:
        raise NotImplementedError

    def ground_truth(self) -> GroundTruthSource:
        raise NotImplementedError

    # Concrete helpers used by the LoopOrchestrator / gate.py subprocess calls.
    def runner_script_rel(self) -> str:
        """Relative (from repo ROOT) path to the episode runner script."""
        raise NotImplementedError

    def verifier_script_rel(self) -> str:
        """Relative (from repo ROOT) path to the episode verifier script."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# SP80Adapter: reference implementation for SP80 / Liquid Pour (sp80-589a99af).
# ---------------------------------------------------------------------------

#: SP80 game identifier (matches the string used throughout the codebase and in
#: summary.json rows: sp80_run_contra_levels.py:158).
_SP80_GAME_ID = "sp80-589a99af"

#: Path to the calibration file on disk.
_SP80_CAL_PATH = CODEXLAB / "sp80_surface_calibration.json"

#: Win-bank file for SP80 ground-truth replay.
_SP80_WIN_BANK_PATH = (
    ROOT / "human" / "win_bank_2026-05-16_human_cheat_map_v1" / "sp80-589a99af_wins.pkl"
)


class SP80Adapter(GameAdapter):
    """Reference GameAdapter for SP80 / Liquid Pour (sp80-589a99af).

    Wraps SP80BaseController (sp80_base_controller.py) which is the sole
    owner of the SP80 engine load path (SP80_PATH) and the sprite API.

    CALIBRATION WARNING: the committed sp80_surface_calibration.json seeds
    the wrong actor (magenta liquid via user_supplied_answer; the real actor
    is the BLUE platform).  Any non-dry iteration that loads this calibration
    will ground the forward-model incorrectly (cov=0/N trap).  The
    StallDetector in stalls.py includes an actor-identity check to catch this.
    """

    game_id: str = _SP80_GAME_ID

    def __init__(self) -> None:
        # Lazy state: populated by new_game() before perform_action() is valid.
        self._game: Any = None
        self._controller: Any = None
        self._level_index: int = 0

        # Load calibration eagerly (tiny JSON read, no heavy deps).
        self._calibration_payload: dict[str, Any] | None = None
        if _SP80_CAL_PATH.is_file():
            try:
                self._calibration_payload = json.loads(
                    _SP80_CAL_PATH.read_text(encoding="utf-8")
                )
            except Exception:
                self._calibration_payload = None

    # -- PrimitiveRegistry ---------------------------------------------------

    def primitive_registry(self) -> PrimitiveRegistry:
        """Return SP80's PrimitiveRegistry (players/sp80/registry.py:91)."""
        return _SP80_REGISTRY

    # -- Game lifecycle ------------------------------------------------------

    def new_game(self, level_index: int) -> GameObs:
        """Create a fresh SP80 game at level_index and return the reset frame.

        Heavy deps (torch, arcengine, PIL) are lazy-imported here so that
        importing this module never forces those deps if only the table or
        protocol is needed.
        """
        import numpy as np  # noqa: PLC0415
        from sp80_base_controller import SP80BaseController  # noqa: PLC0415

        self._level_index = level_index
        self._controller = SP80BaseController(level_index=level_index)
        game, obs = self._controller.new_game()
        self._game = game
        frame = np.asarray(obs.frame[-1], dtype=np.int16)
        levels_completed = int(getattr(obs, "levels_completed", 0))
        state = (
            str(obs.state.value if hasattr(obs.state, "value") else obs.state)
            if hasattr(obs, "state")
            else "unknown"
        )
        steps = int(getattr(game, "zlhbnhpcq", 0)) or None
        return GameObs(
            frame=frame,
            levels_completed=levels_completed,
            state=state,
            steps_remaining=steps,
        )

    def perform_action(
        self, primitive: str, *, data: dict[str, Any] | None = None
    ) -> GameObs:
        """Apply a control primitive to the current SP80 game state.

        For select_object without explicit click data, the first available
        selectable sprite is used (GroundTruthOracle diffing pattern).
        The caller MUST call new_game() first.

        Returns GameObs; obs.levels_completed is the ONLY solve signal.
        """
        if self._game is None or self._controller is None:
            raise RuntimeError(
                "SP80Adapter.perform_action() called before new_game(); "
                "call new_game(level_index) first."
            )
        import numpy as np  # noqa: PLC0415

        # For select_object, resolve the click target if not supplied.
        target = None
        if primitive == "select_object" and not data:
            sprites = self._controller.selectable_sprites(self._game)
            if sprites:
                target = sprites[0]

        # ground_primitive handles the full primitive->GameAction mapping
        # (sp80_base_controller.py:686-725).
        grounded = self._controller.ground_primitive(
            self._game,
            primitive,
            target=target,
        )
        if data and not grounded.data:
            # Caller supplied explicit action data (e.g., a specific click).
            from arcengine import ActionInput  # noqa: PLC0415

            game_action = grounded.action
            obs = self._game.perform_action(
                ActionInput(id=game_action, data=data), raw=True
            )
        else:
            obs = self._controller.execute_grounded_action(self._game, grounded)

        if obs is None or not obs.frame:
            raise RuntimeError(
                f"SP80Adapter.perform_action({primitive!r}) produced no frame"
            )
        frame = np.asarray(obs.frame[-1], dtype=np.int16)
        levels_completed = int(getattr(obs, "levels_completed", 0))
        state = (
            str(obs.state.value if hasattr(obs.state, "value") else obs.state)
            if hasattr(obs, "state")
            else "unknown"
        )
        steps = int(getattr(self._game, "zlhbnhpcq", 0)) or None
        return GameObs(
            frame=frame,
            levels_completed=levels_completed,
            state=state,
            steps_remaining=steps,
        )

    # -- Verification --------------------------------------------------------

    def episode_verifier(self) -> Callable[[Path], dict[str, Any]]:
        """Return a callable wrapping sp80_verify_episode_run.verify_run.

        The callable is suitable for direct in-process use by gate.py and
        registry.py (which call it as verifier(run_dir) -> dict).
        sp80_verify_episode_run.verify_run is lazy-imported to avoid
        pulling in the verifier at module import time.
        """

        def _verify(run_dir: Path) -> dict[str, Any]:
            from sp80_verify_episode_run import verify_run  # noqa: PLC0415

            return verify_run(Path(run_dir))

        return _verify

    # -- Ground-truth source -------------------------------------------------

    def ground_truth(self) -> GroundTruthSource:
        """Return the preferred ground-truth source for this game.

        SP80 prefers the human win-bank (deterministic replay, 0 mismatches,
        established in transcripts as win#5 / 6 levels / 151 steps).
        Falls back to engine_source (sp80.py step() CA) if the win-bank pkl
        is missing.

        NOTE: win-bank replay API (get_level_frames / replay_actions) is a
        TODO extension point in ground_truth.py; until that API exists,
        keystones must declare ground_truth_source='engine_source' explicitly
        so the registry records honest provenance.
        """
        if _SP80_WIN_BANK_PATH.is_file():
            return "win_bank"
        return "engine_source"

    # -- Subprocess runner / verifier paths ----------------------------------

    def runner_script_rel(self) -> str:
        """Relative path (from ROOT) to the SP80 episode runner.

        Matches loop_driver.py ADAPTERS['sp80']['runner'].
        Used by LoopOrchestrator when shelling to 'contra run'.
        """
        return "codexlab/sp80_run_contra_levels.py"

    def verifier_script_rel(self) -> str:
        """Relative path (from ROOT) to the SP80 episode verifier.

        Matches loop_driver.py ADAPTERS['sp80']['verifier'].
        Used by gate.py for Layer 4 (per-level verify).
        """
        return "codexlab/sp80_verify_episode_run.py"

    # -- Player descriptor ---------------------------------------------------

    def player_descriptor(self) -> GamePlayerDescriptor:
        """Return the SP80_PLAYER GamePlayerDescriptor (players/sp80/player.py:8).

        Provided so consumers can inspect schemas without importing the player
        package directly.
        """
        from players.sp80.player import SP80_PLAYER  # noqa: PLC0415

        return SP80_PLAYER


# ---------------------------------------------------------------------------
# GAME_TABLE: authoritative per-game dispatch table.
#
# Keys are game short names ("sp80" in "sp80:1"), NOT the full game_id.
# Values are adapter instances.
#
# Replaces the triple of hardcoded constants in contra.py:36-39:
#   GAME_ID     = "sp80-589a99af"
#   CALIBRATION = CODEXLAB / "sp80_surface_calibration.json"
#   HARNESS_SRC = CODEXLAB / "sp80_player_harness.py"
#
# To register a new game:
#   1. Drop it under environment/environment_files/<short>/<hash>/<short>.py
#      with a metadata.json; discover_games() finds it automatically.
#   2. Subclass GameAdapter and implement all abstract methods.
#      SP80Adapter is the reference implementation.
#   3. Add it here: GAME_TABLE["<short>"] = MyGameAdapter()
#   4. Run `python codexlab/contra_loop/adapter.py --dry-run` to verify.
# ---------------------------------------------------------------------------
GAME_TABLE: dict[str, Any] = {
    "sp80": SP80Adapter(),
    # TODO: register additional game adapters here as they are onboarded.
    # Each adapter must implement GameAdapter (game_id, calibration,
    # primitive_registry, new_game, perform_action, episode_verifier,
    # ground_truth, runner_script_rel, verifier_script_rel).
}


# ---------------------------------------------------------------------------
# Lookup helpers used by the LoopOrchestrator and gate.
# ---------------------------------------------------------------------------

def lookup_adapter(game_short: str) -> Any:
    """Return the adapter for the given game short name.

    Raises KeyError (not SystemExit) so callers can catch and handle.
    game_short is the part before ':' in 'sp80:1'.
    """
    try:
        return GAME_TABLE[game_short]
    except KeyError:
        known = sorted(GAME_TABLE)
        raise KeyError(
            f"no adapter registered for game_short={game_short!r}; "
            f"known adapters: {known}"
        ) from None


def lookup_adapter_for_target(target: str) -> tuple[Any, int]:
    """Parse 'game:level' (e.g. 'sp80:1') and return (adapter, level_index).

    Raises KeyError if the game short name is not in GAME_TABLE.
    Raises ValueError if level_index is not an integer.
    """
    game_short, _, level_str = target.partition(":")
    level_index = int(level_str) if level_str else 0
    return lookup_adapter(game_short), level_index


# ---------------------------------------------------------------------------
# discover_games: replicated from cross_game_isolation_viewer.py:158-190.
#
# We replicate rather than import the viewer to avoid a fragile script-to-
# module dependency (the viewer imports PIL, numpy, torch at top level).
# Logic is identical: glob ENV_ROOT for metadata.json + matching .py files.
# ---------------------------------------------------------------------------

def discover_games() -> list[dict[str, Any]]:
    """Discover all registered ARC games under environment/environment_files/.

    Returns a list of dicts with:
      game_id, title, game_short, game_hash,
      metadata_path (relative), source_path (relative),
      has_adapter (True if registered in GAME_TABLE).

    Does NOT load game classes (no arcengine import needed).
    Replicates cross_game_isolation_viewer.discover_games (lines 158-190).
    """
    games: list[dict[str, Any]] = []
    for meta_path in sorted(ENV_ROOT.glob("*/*/metadata.json")):
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        game_short = meta_path.parents[1].name
        game_hash = meta_path.parent.name
        py_path = meta_path.parent / f"{game_short}.py"
        if not py_path.is_file():
            continue
        games.append(
            {
                "game_id": str(
                    metadata.get("game_id") or f"{game_short}-{game_hash}"
                ),
                "title": str(metadata.get("title") or game_short),
                "game_short": game_short,
                "game_hash": game_hash,
                "metadata_path": meta_path.relative_to(ROOT).as_posix(),
                "source_path": py_path.relative_to(ROOT).as_posix(),
                "has_adapter": game_short in GAME_TABLE,
            }
        )
    return games


# ---------------------------------------------------------------------------
# _adapter_summary: machine-readable summary of one adapter (for --dry-run).
# Heavy deps are NOT imported here; we use what's available without new_game().
# ---------------------------------------------------------------------------

def _adapter_summary(name: str, adapter: Any) -> dict[str, Any]:
    cal = adapter.calibration
    if cal is not None:
        cal_desc = (
            f"loaded ({len(cal.get('levels', []))} level rows)"
            if isinstance(cal, dict)
            else "loaded"
        )
    else:
        cal_desc = "none (uncalibrated mode)"

    gt: str
    try:
        gt = adapter.ground_truth()
    except Exception as exc:
        gt = f"error: {exc}"

    runner: str
    try:
        runner = adapter.runner_script_rel()
    except Exception:
        runner = "n/a"

    verifier: str
    try:
        verifier = adapter.verifier_script_rel()
    except Exception:
        verifier = "n/a"

    primitive_names: list[str]
    try:
        primitive_names = adapter.primitive_registry().names()
    except Exception as exc:
        primitive_names = [f"error: {exc}"]

    return {
        "game_short": name,
        "game_id": adapter.game_id,
        "calibration": cal_desc,
        "ground_truth_source": gt,
        "primitive_count": len(primitive_names),
        "primitives": primitive_names,
        "runner_script": runner,
        "verifier_script": verifier,
    }


# ---------------------------------------------------------------------------
# __main__: day-one sanity check; runs without a solve, episode, or network.
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        prog="contra_loop.adapter",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print GAME_TABLE and discovered games as JSON; exit 0",
    )
    parser.add_argument(
        "--list-adapters",
        action="store_true",
        help="print registered adapters in GAME_TABLE (same as --dry-run)",
    )
    parser.add_argument(
        "--list-games",
        action="store_true",
        help="print all discovered games under environment/environment_files/",
    )
    parser.add_argument(
        "--target",
        default=None,
        help="lookup a specific target (e.g. sp80:1) and print its adapter",
    )
    args = parser.parse_args()

    if args.list_games:
        games = discover_games()
        print(json.dumps(
            {
                "schema": SCHEMA,
                "game_count": len(games),
                "games": games,
            },
            indent=2,
        ))
        return 0

    if args.target:
        try:
            adapter, level_index = lookup_adapter_for_target(args.target)
        except (KeyError, ValueError) as exc:
            print(f"ERROR: {exc}", file=_sys.stderr)
            return 1
        print(json.dumps(
            {
                "schema": SCHEMA,
                "target": args.target,
                "level_index": level_index,
                "adapter": _adapter_summary(
                    args.target.split(":", 1)[0], adapter
                ),
                "calibration_for_level": (
                    "available"
                    if adapter.calibration_for_level(level_index) is not None
                    else "none (uncalibrated for this level)"
                ),
            },
            indent=2,
        ))
        return 0

    # Default: --dry-run or --list-adapters or bare invocation -> print table.
    adapters_summary = {
        name: _adapter_summary(name, adapter)
        for name, adapter in GAME_TABLE.items()
    }
    discovered = discover_games()
    not_adapted = [g for g in discovered if not g["has_adapter"]]

    print(json.dumps(
        {
            "schema": SCHEMA,
            "adapter_count": len(GAME_TABLE),
            "registered_adapters": adapters_summary,
            "discovered_game_count": len(discovered),
            "discovered_games_without_adapter": [
                g["game_short"] for g in not_adapted
            ],
            "honest_limit": (
                "13+ literal 'sp80-589a99af' strings in sp80_player_harness.py "
                "and SP80_PATH at sp80_base_controller.py:32 must be threaded "
                "from adapter.game_id before a second game truly runs the full "
                "episode loop (adapter.py centralises dispatch; the harness "
                "threading is a tracked TODO)."
            ),
            "anti_reward_hacking_note": (
                "GameAdapter.perform_action() and new_game() are for "
                "GroundTruthOracle frame diffing (ground_truth.py). "
                "Actual episode RUNNING goes through 'contra run' "
                "(sp80_run_contra_levels.py) which produces the only "
                "valid provenance artifact (summary.json + verifier rc=0 + "
                "levels_completed_delta>0). The adapter never fabricates "
                "a solve signal."
            ),
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
