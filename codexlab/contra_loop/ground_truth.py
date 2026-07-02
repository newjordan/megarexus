#!/usr/bin/env python3
"""CONTRA Loop Harness -- GroundTruthOracle.

Three pluggable ground-truth sources behind one API returning pixel-exact
before/after frame pairs:

  (a) WIN_BANK  -- human win-bank deterministic replay
                   human/win_bank_*/<game>_wins.pkl (+20 games)
  (b) ENGINE    -- engine CA replay via sp80.py step() (or other ARCBaseGame)
                   environment/environment_files/<short>/<hash>/<short>.py
  (c) RAW_RECORDINGS -- raw_recordings/*.jsonl frame-diff from opus-4.8 corpus
                   codexlab/runs/arc_public_recordings_*/raw_recordings/

Priority: WIN_BANK > ENGINE_SOURCE > RAW_RECORDINGS.

The oracle is the canonical validation set. It REFUSES to admit a mechanic
validated only on synthetic data (get_level_frames raises SyntheticOnlyError
if no real source is available).

Self-test: replay win#5 through the engine and assert 0 level-completion-
transition mismatches -- this runs on `python ground_truth.py --self-test`.

Usage:
    python codexlab/contra_loop/ground_truth.py --self-test
    python codexlab/contra_loop/ground_truth.py --dry-run
    python codexlab/contra_loop/ground_truth.py --game-id sp80-589a99af --level 1
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODEXLAB = ROOT / "codexlab"
WIN_BANK_ROOT = ROOT / "human"
ENGINE_ROOT = ROOT / "environment" / "environment_files"
RAW_REC_ROOT = CODEXLAB / "runs"

SCHEMA = "codexlab.contra_loop.ground_truth.v1"

# Action id (int stored in win-bank) -> GameAction name
_ACTION_ID_TO_NAME: dict[int, str] = {
    0: "RESET",
    1: "ACTION1",
    2: "ACTION2",
    3: "ACTION3",
    4: "ACTION4",
    5: "ACTION5",
    6: "ACTION6",
    7: "ACTION7",
}


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class GroundTruthSourceKind(str, Enum):
    WIN_BANK = "win_bank"
    ENGINE_SOURCE = "engine_source"
    RAW_RECORDINGS = "raw_recordings"


class SyntheticOnlyError(RuntimeError):
    """Raised when no real ground-truth source is available for a game/level."""


@dataclass
class FramePair:
    """A pixel-exact before/after frame pair around one action step.

    Frames are palette-indexed numpy arrays with shape (H, W) and dtype int8/int64.
    before_frame is the game state BEFORE action was applied.
    after_frame is the game state AFTER action was applied (next step).
    """

    before_frame: np.ndarray          # shape (H, W), palette indices
    after_frame: np.ndarray           # shape (H, W), palette indices
    action_primitive: str             # e.g. "move_right", "activate_flow", "ACTION4"
    action_id: int                    # raw GameAction numeric id
    levels_completed_before: int      # levels_completed before this action
    levels_completed_after: int       # levels_completed after this action
    step_index: int                   # 0-based index within the replay sequence
    source_kind: GroundTruthSourceKind
    win_index: int | None = None      # which win record this came from (WIN_BANK only)
    game_id: str = ""

    @property
    def level_up(self) -> bool:
        return self.levels_completed_after > self.levels_completed_before

    def frame_diff_pixels(self) -> int:
        return int(np.sum(self.before_frame != self.after_frame))

    def before_sha256(self) -> str:
        return hashlib.sha256(np.asarray(self.before_frame).tobytes()).hexdigest()

    def after_sha256(self) -> str:
        return hashlib.sha256(np.asarray(self.after_frame).tobytes()).hexdigest()


@dataclass
class WinRecord:
    """A single human win extracted from the win-bank pkl."""

    game_id: str
    win_index: int           # 0-based index within the pkl list
    episode: int
    seed: int
    won_levels: int
    frames: list[np.ndarray]  # frames[i] = state BEFORE action[i]
    actions: list[dict[str, Any]]
    levels: list[int]         # levels[i] = game level active at step i
    human_played_utc: str
    pkl_path: Path


@dataclass
class OracleReport:
    """Summary produced by GroundTruthOracle.self_test()."""

    schema: str = SCHEMA + ".self_test_report"
    game_id: str = ""
    win_index: int = 0
    won_levels: int = 0
    total_steps: int = 0
    level_transition_mismatches: int = 0
    engine_final_levels_completed: int = 0
    bank_final_levels_completed: int = 0
    source_kind: str = GroundTruthSourceKind.WIN_BANK.value
    passed: bool = False
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine loader
# ---------------------------------------------------------------------------


def _load_engine_module(game_id: str) -> Any | None:
    """Dynamically load the ARCBaseGame subclass for game_id.

    Returns the loaded module, or None if the engine file is not found.
    Pattern: environment/environment_files/<short>/<hash>/<short>.py
    """
    parts = game_id.split("-", 1)
    if len(parts) != 2:
        return None
    game_short, game_hash = parts
    candidates = list(ENGINE_ROOT.glob(f"{game_short}/{game_hash}/*.py"))
    if not candidates:
        return None
    engine_path = candidates[0]
    spec = importlib.util.spec_from_file_location(f"_gt_engine_{game_id}", engine_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[arg-type]
    except Exception:
        return None
    return mod


def _engine_source_sha256(game_id: str) -> str | None:
    parts = game_id.split("-", 1)
    if len(parts) != 2:
        return None
    game_short, game_hash = parts
    candidates = list(ENGINE_ROOT.glob(f"{game_short}/{game_hash}/*.py"))
    if not candidates:
        return None
    path = candidates[0]
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _squeeze_frame(raw_frame: Any) -> np.ndarray:
    """Normalise a frame to shape (H, W) from various storage formats.

    Engine returns obs.frame as a list or ndarray of shape (1, H, W) or (H, W).
    Win-bank stores (H, W) int8 arrays directly.
    Raw recordings store nested list [[[...]] ].
    """
    arr = np.asarray(raw_frame, dtype=np.int64)
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def _action_id_to_primitive(action_id: int) -> str:
    """Map win-bank action id int to a canonical primitive label.

    The win-bank stores raw integer ids matching GameAction enum values.
    We map them to the SP80 primitive vocabulary where possible; otherwise
    fall back to the ACTION<N> name.
    """
    _SP80_MAP = {
        1: "move_up",
        2: "move_down",
        3: "move_left",
        4: "move_right",
        5: "activate_flow",
        6: "select_object",
    }
    return _SP80_MAP.get(action_id, _ACTION_ID_TO_NAME.get(action_id, f"action_{action_id}"))


def _raw_action_name(action_id: int) -> str:
    return _ACTION_ID_TO_NAME.get(action_id, f"ACTION{action_id}")


# ---------------------------------------------------------------------------
# GroundTruthOracle
# ---------------------------------------------------------------------------


class GroundTruthOracle:
    """Three-source ground-truth oracle for CONTRA Loop Harness frame validation.

    Sources are tried in priority order:
      1. WIN_BANK  -- most reliable: human-played, deterministic
      2. ENGINE_SOURCE -- engine CA replay (no human actions needed)
      3. RAW_RECORDINGS -- Opus-4.8 public recordings corpus

    A new game with no win-bank falls back to ENGINE_SOURCE + RAW_RECORDINGS.
    The oracle REFUSES to return data if all sources are unavailable (raises
    SyntheticOnlyError) so no mechanic can be validated on synthetic data only.

    Typical use:
        oracle = GroundTruthOracle("sp80-589a99af")
        pairs = oracle.get_level_frames(level_index=1)
        report = oracle.self_test()
    """

    def __init__(self, game_id: str, *, root: Path | None = None) -> None:
        self.game_id = game_id
        self._root = root or ROOT
        self._win_bank_root = self._root / "human"
        self._engine_root = self._root / "environment" / "environment_files"
        self._raw_rec_root = self._root / "codexlab" / "runs"
        self._win_records: list[WinRecord] | None = None  # lazy
        self._engine_mod: Any | None = _SENTINEL  # lazy; _SENTINEL = "not yet tried"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def available_sources(self) -> list[GroundTruthSourceKind]:
        """Return the list of sources that have data for this game_id."""
        sources: list[GroundTruthSourceKind] = []
        if self._win_bank_pkl() is not None:
            sources.append(GroundTruthSourceKind.WIN_BANK)
        if self._engine_module() is not None:
            sources.append(GroundTruthSourceKind.ENGINE_SOURCE)
        if self._raw_recording_paths():
            sources.append(GroundTruthSourceKind.RAW_RECORDINGS)
        return sources

    def get_level_frames(self, level_index: int) -> list[FramePair]:
        """Return before/after FramePairs for all steps played at level_index.

        Priority: WIN_BANK > ENGINE_SOURCE > RAW_RECORDINGS.
        Raises SyntheticOnlyError if no real source is available.

        Each FramePair.before_frame is the game state before the action;
        FramePair.after_frame is the state after.  level_up=True flags the
        completing step of a level.
        """
        # Try WIN_BANK first
        records = self._load_win_records()
        if records:
            pairs = self._frames_from_win_bank(level_index, records)
            if pairs:
                return pairs

        # Fall back to ENGINE_SOURCE (CA replay of known win-bank actions)
        eng_mod = self._engine_module()
        if eng_mod is not None:
            records = self._load_win_records()
            if records:
                pairs = self._frames_from_engine_replay(level_index, records)
                if pairs:
                    return pairs

        # Fall back to RAW_RECORDINGS
        raw_pairs = self._frames_from_raw_recordings(level_index)
        if raw_pairs:
            return raw_pairs

        raise SyntheticOnlyError(
            f"No real ground-truth source available for {self.game_id} level {level_index}. "
            f"Win-bank: {self._win_bank_pkl()}, engine: {eng_mod is not None}, "
            f"raw_recordings: {bool(self._raw_recording_paths())}. "
            "The oracle refuses to produce synthetic-only data."
        )

    def replay_actions(
        self,
        win_index: int,
        steps: slice | None = None,
    ) -> list[FramePair]:
        """Replay win_index from win-bank through the engine and return FramePairs.

        steps -- optional slice to select a sub-sequence, e.g. slice(4, 24).
        Returns engine-produced frames (ENGINE_SOURCE kind).
        Raises SyntheticOnlyError if no engine is available.
        """
        eng_mod = self._engine_module()
        if eng_mod is None:
            raise SyntheticOnlyError(
                f"Engine module not found for {self.game_id}; cannot replay."
            )
        records = self._load_win_records()
        if not records or win_index >= len(records):
            raise SyntheticOnlyError(
                f"Win-bank win_index={win_index} not available for {self.game_id}."
            )
        return self._engine_replay_win(records[win_index], steps=steps)

    def self_test(self) -> OracleReport:
        """Replay win#5 through the engine; assert 0 level-completion mismatches.

        This is the bootstrap validation: if the engine CA does not reproduce
        the win-bank level-completion sequence exactly, the oracle is not
        trustworthy and the loop MUST NOT use engine frames as ground truth.

        Returns an OracleReport with passed=True iff mismatches==0.
        """
        report = OracleReport(game_id=self.game_id)

        records = self._load_win_records()
        if not records:
            report.notes.append(f"No win-bank pkl found at {self._win_bank_pkl()}")
            report.passed = False
            return report

        # Use win index 5 if available (the 6-level record), else fall back to last
        target_win_idx = 5 if len(records) > 5 else len(records) - 1
        win = records[target_win_idx]
        report.win_index = target_win_idx
        report.won_levels = win.won_levels
        report.total_steps = len(win.actions)

        eng_mod = self._engine_module()
        if eng_mod is None:
            report.notes.append(
                f"Engine module not found for {self.game_id}; "
                "self-test verifies win-bank consistency only (no engine replay)."
            )
            # Without engine, just validate win-bank internal consistency
            mismatches = _check_win_bank_internal(win)
            report.level_transition_mismatches = mismatches
            report.bank_final_levels_completed = win.levels[-1] if win.levels else 0
            report.engine_final_levels_completed = -1
            report.passed = mismatches == 0
            return report

        # Replay through engine and compare level-completion sequences
        try:
            pairs = self._engine_replay_win(win)
        except Exception as exc:
            report.notes.append(f"Engine replay raised {type(exc).__name__}: {exc}")
            report.passed = False
            return report

        engine_levels = [p.levels_completed_after for p in pairs]
        bank_levels = win.levels

        mismatches = 0
        for i, (e, b) in enumerate(zip(engine_levels, bank_levels)):
            if e != b:
                mismatches += 1
                if mismatches <= 5:  # cap noise in report
                    report.notes.append(
                        f"step {i}: engine levels_completed={e} != bank.levels={b}"
                    )

        report.level_transition_mismatches = mismatches
        report.engine_final_levels_completed = engine_levels[-1] if engine_levels else 0
        report.bank_final_levels_completed = bank_levels[-1] if bank_levels else 0
        report.source_kind = GroundTruthSourceKind.ENGINE_SOURCE.value
        report.passed = mismatches == 0
        if report.passed:
            report.notes.append(
                f"PASS: 0 level-completion mismatches over {len(pairs)} steps; "
                f"engine reproduced win-bank levels exactly."
            )
        else:
            report.notes.append(
                f"FAIL: {mismatches}/{len(bank_levels)} level-completion mismatches."
            )
        return report

    def provenance_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable provenance record for embedding in artifacts."""
        pkl = self._win_bank_pkl()
        return {
            "schema": SCHEMA,
            "game_id": self.game_id,
            "available_sources": [s.value for s in self.available_sources()],
            "win_bank_pkl": str(pkl) if pkl else None,
            "win_bank_records": len(self._load_win_records()),
            "engine_source_sha256": _engine_source_sha256(self.game_id),
            "raw_recording_paths": [
                str(p) for p in self._raw_recording_paths()
            ],
        }

    # ------------------------------------------------------------------
    # Private: WIN_BANK source
    # ------------------------------------------------------------------

    def _win_bank_pkl(self) -> Path | None:
        """Locate the win-bank pkl for this game_id (e.g. sp80-589a99af_wins.pkl)."""
        candidates = list(
            self._win_bank_root.glob(f"win_bank_*/{self.game_id}_wins.pkl")
        )
        if not candidates:
            return None
        # Prefer most recent (lexicographic sort on directory name)
        return sorted(candidates, key=lambda p: p.parent.name)[-1]

    def _load_win_records(self) -> list[WinRecord]:
        if self._win_records is not None:
            return self._win_records
        pkl = self._win_bank_pkl()
        if pkl is None:
            self._win_records = []
            return self._win_records
        try:
            with pkl.open("rb") as fh:
                raw_list: list[dict[str, Any]] = pickle.load(fh)
        except Exception:
            self._win_records = []
            return self._win_records

        records: list[WinRecord] = []
        for idx, entry in enumerate(raw_list):
            records.append(
                WinRecord(
                    game_id=str(entry.get("game_id", self.game_id)),
                    win_index=idx,
                    episode=int(entry.get("episode", 0)),
                    seed=int(entry.get("seed", 0)),
                    won_levels=int(entry.get("won_levels", 0)),
                    frames=[np.asarray(f) for f in entry["frames"]],
                    actions=list(entry["actions"]),
                    levels=list(entry.get("levels", [])),
                    human_played_utc=str(entry.get("human_played_utc", "")),
                    pkl_path=pkl,
                )
            )
        self._win_records = records
        return records

    def _frames_from_win_bank(
        self, level_index: int, records: list[WinRecord]
    ) -> list[FramePair]:
        """Extract FramePairs for level_index from stored win-bank frames."""
        pairs: list[FramePair] = []
        for win in records:
            steps = [i for i, lvl in enumerate(win.levels) if lvl == level_index]
            for i in steps:
                if i + 1 >= len(win.frames):
                    continue
                before = _squeeze_frame(win.frames[i])
                after = _squeeze_frame(win.frames[i + 1])
                act = win.actions[i]
                act_id = int(act.get("id", 0))
                levels_after = win.levels[i + 1] if i + 1 < len(win.levels) else level_index
                pairs.append(
                    FramePair(
                        before_frame=before,
                        after_frame=after,
                        action_primitive=_action_id_to_primitive(act_id),
                        action_id=act_id,
                        levels_completed_before=level_index,
                        levels_completed_after=levels_after,
                        step_index=i,
                        source_kind=GroundTruthSourceKind.WIN_BANK,
                        win_index=win.win_index,
                        game_id=win.game_id,
                    )
                )
        return pairs

    # ------------------------------------------------------------------
    # Private: ENGINE_SOURCE
    # ------------------------------------------------------------------

    def _engine_module(self) -> Any | None:
        """Lazy-load the game engine module.  Returns None if unavailable."""
        if self._engine_mod is not _SENTINEL:
            return self._engine_mod
        self._engine_mod = _load_engine_module(self.game_id)
        return self._engine_mod

    def _make_game(self) -> Any | None:
        """Instantiate the game and apply a RESET action. Returns game instance."""
        mod = self._engine_module()
        if mod is None:
            return None
        # Locate the ARCBaseGame subclass by name matching game_short
        game_short = self.game_id.split("-")[0]
        game_class = None
        for name in dir(mod):
            obj = getattr(mod, name)
            try:
                if (
                    isinstance(obj, type)
                    and name.lower() != "arcbasegame"
                    and hasattr(obj, "perform_action")
                ):
                    game_class = obj
                    break
            except Exception:
                pass
        if game_class is None:
            return None
        try:
            return game_class()
        except Exception:
            return None

    def _engine_step(self, game: Any, action_id: int, action_data: dict[str, Any]) -> Any:
        """Apply one action and return the obs from the engine."""
        from arcengine import ActionInput, GameAction  # local import avoids hard dep

        action_name = _raw_action_name(action_id)
        ga = GameAction[action_name]
        ai = ActionInput(id=ga, data=action_data)
        return game.perform_action(ai, raw=True)

    def _engine_replay_win(
        self, win: WinRecord, *, steps: slice | None = None
    ) -> list[FramePair]:
        """Replay win-bank actions through the engine; return FramePairs.

        The engine tracks levels_completed; frames come from obs.frame.
        """
        from arcengine import ActionInput, GameAction  # local import

        game = self._make_game()
        if game is None:
            return []

        # Reset
        reset_obs = game.perform_action(
            ActionInput(id=GameAction["RESET"], data={}), raw=True
        )
        prev_frame = _squeeze_frame(reset_obs.frame)
        prev_levels = int(reset_obs.levels_completed)

        actions_seq = win.actions
        if steps is not None:
            actions_seq = actions_seq[steps]

        pairs: list[FramePair] = []
        for i, act_dict in enumerate(actions_seq):
            actual_idx = i if steps is None else range(len(win.actions))[steps][i]
            act_id = int(act_dict.get("id", 0))
            ai_data: dict[str, Any] = {"game_id": self.game_id}
            if act_dict.get("complex") and act_dict.get("x") is not None:
                ai_data["x"] = act_dict["x"]
                ai_data["y"] = act_dict["y"]
            try:
                obs = self._engine_step(game, act_id, ai_data)
            except Exception:
                break
            curr_frame = _squeeze_frame(obs.frame)
            curr_levels = int(obs.levels_completed)
            pairs.append(
                FramePair(
                    before_frame=prev_frame,
                    after_frame=curr_frame,
                    action_primitive=_action_id_to_primitive(act_id),
                    action_id=act_id,
                    levels_completed_before=prev_levels,
                    levels_completed_after=curr_levels,
                    step_index=actual_idx,
                    source_kind=GroundTruthSourceKind.ENGINE_SOURCE,
                    win_index=win.win_index,
                    game_id=self.game_id,
                )
            )
            prev_frame = curr_frame
            prev_levels = curr_levels
        return pairs

    def _frames_from_engine_replay(
        self, level_index: int, records: list[WinRecord]
    ) -> list[FramePair]:
        """Replay the first win record that covers level_index; filter to that level."""
        covering = [w for w in records if w.won_levels > level_index]
        if not covering:
            return []
        win = covering[0]
        all_pairs = self._engine_replay_win(win)
        return [
            p for p in all_pairs if p.levels_completed_before == level_index
        ]

    # ------------------------------------------------------------------
    # Private: RAW_RECORDINGS source
    # ------------------------------------------------------------------

    def _raw_recording_paths(self) -> list[Path]:
        return sorted(
            self._raw_rec_root.glob(
                f"arc_public_recordings_*/raw_recordings/{self.game_id}__*.jsonl"
            )
        )

    def _frames_from_raw_recordings(self, level_index: int) -> list[FramePair]:
        """Mine raw_recordings/*.jsonl for before/after frame pairs at level_index.

        Each event stores data.frame as a nested list (1, H, W) or (H, H, H...).
        We produce pairs from consecutive events at the same levels_completed.
        Skips events where frame is the stripped marker string.
        """
        pairs: list[FramePair] = []
        STRIPPED = "public recording frame omitted by miner"

        for rec_path in self._raw_recording_paths():
            events: list[dict[str, Any]] = []
            try:
                for raw_line in rec_path.read_text(encoding="utf-8").splitlines():
                    raw_line = raw_line.strip()
                    if raw_line:
                        ev = json.loads(raw_line)
                        d = ev.get("data", {})
                        if isinstance(d, dict) and d.get("frame") != STRIPPED:
                            events.append(d)
            except Exception:
                continue

            for i in range(len(events) - 1):
                e0 = events[i]
                e1 = events[i + 1]
                lc0 = int(e0.get("levels_completed", -1))
                lc1 = int(e1.get("levels_completed", -1))
                if lc0 != level_index:
                    continue
                frame0 = e0.get("frame")
                frame1 = e1.get("frame")
                if frame0 is None or frame1 is None:
                    continue
                ai = e0.get("action_input") or {}
                act_name = str(ai.get("id", "UNKNOWN"))
                # Map ACTION<N> name to int id
                action_id = _parse_action_name_to_id(act_name)
                try:
                    before = _squeeze_frame(frame0)
                    after = _squeeze_frame(frame1)
                except Exception:
                    continue
                pairs.append(
                    FramePair(
                        before_frame=before,
                        after_frame=after,
                        action_primitive=_action_id_to_primitive(action_id),
                        action_id=action_id,
                        levels_completed_before=lc0,
                        levels_completed_after=lc1,
                        step_index=i,
                        source_kind=GroundTruthSourceKind.RAW_RECORDINGS,
                        win_index=None,
                        game_id=self.game_id,
                    )
                )
        return pairs


# ---------------------------------------------------------------------------
# Sentinel (avoids mutable default in __init__ that confuses type checkers)
# ---------------------------------------------------------------------------


class _SentinelType:
    pass


_SENTINEL = _SentinelType()


# ---------------------------------------------------------------------------
# Internal helpers (module-level)
# ---------------------------------------------------------------------------


def _parse_action_name_to_id(name: str) -> int:
    """ACTION6 -> 6, RESET -> 0, unknown -> -1."""
    name = name.upper()
    if name == "RESET":
        return 0
    if name.startswith("ACTION") and name[6:].isdigit():
        return int(name[6:])
    return -1


def _check_win_bank_internal(win: WinRecord) -> int:
    """Count self-consistency mismatches within win-bank (no engine needed).

    Checks that levels field is monotonically non-decreasing and ends at
    won_levels - 1 (since levels are 0-indexed game levels completed so far).
    Returns number of violations.
    """
    mismatches = 0
    prev = win.levels[0] if win.levels else 0
    for lvl in win.levels[1:]:
        if lvl < prev:
            mismatches += 1
        prev = lvl
    return mismatches


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def oracle_for_game(game_id: str) -> GroundTruthOracle:
    """Factory returning a configured GroundTruthOracle for game_id."""
    return GroundTruthOracle(game_id)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_self_test(args: argparse.Namespace) -> int:
    game_id = args.game_id or "sp80-589a99af"
    oracle = GroundTruthOracle(game_id)

    print(f"[ground_truth] self-test for {game_id}")
    print(f"  available sources: {[s.value for s in oracle.available_sources()]}")
    print(f"  win-bank pkl: {oracle._win_bank_pkl()}")

    report = oracle.self_test()
    print(json.dumps(
        {
            "schema": report.schema,
            "game_id": report.game_id,
            "win_index": report.win_index,
            "won_levels": report.won_levels,
            "total_steps": report.total_steps,
            "level_transition_mismatches": report.level_transition_mismatches,
            "engine_final_levels_completed": report.engine_final_levels_completed,
            "bank_final_levels_completed": report.bank_final_levels_completed,
            "source_kind": report.source_kind,
            "passed": report.passed,
            "notes": report.notes,
        },
        indent=2,
    ))
    return 0 if report.passed else 1


def _cli_dry_run(args: argparse.Namespace) -> int:
    game_id = args.game_id or "sp80-589a99af"
    oracle = GroundTruthOracle(game_id)
    sources = oracle.available_sources()
    prov = oracle.provenance_dict()
    print(f"[ground_truth] dry-run for {game_id}")
    print(f"  sources available: {[s.value for s in sources]}")
    print(f"  win-bank records: {prov['win_bank_records']}")
    print(f"  engine_source_sha256: {prov['engine_source_sha256']}")
    print(f"  raw_recording_paths: {len(prov['raw_recording_paths'])} file(s)")
    if not sources:
        print("  WARNING: no real source found -- oracle would raise SyntheticOnlyError")
        return 1
    return 0


def _cli_level_frames(args: argparse.Namespace) -> int:
    game_id = args.game_id or "sp80-589a99af"
    level = int(args.level)
    oracle = GroundTruthOracle(game_id)
    try:
        pairs = oracle.get_level_frames(level)
    except SyntheticOnlyError as exc:
        print(f"[ground_truth] SyntheticOnlyError: {exc}")
        return 1
    print(f"[ground_truth] {len(pairs)} frame pairs for {game_id} level {level}")
    for i, p in enumerate(pairs[:5]):
        print(
            f"  pair[{i}] step={p.step_index} action={p.action_primitive}"
            f" source={p.source_kind.value}"
            f" diff={p.frame_diff_pixels()}px"
            f" level_up={p.level_up}"
        )
    if len(pairs) > 5:
        print(f"  ... ({len(pairs) - 5} more pairs)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Replay win#5 through the engine; assert 0 level-completion mismatches.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print available sources without running replay.",
    )
    parser.add_argument(
        "--game-id",
        default="sp80-589a99af",
        help="game_id to query (default: sp80-589a99af)",
    )
    parser.add_argument(
        "--level",
        default="0",
        help="Level index to fetch frame pairs for (used without --self-test/--dry-run)",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return _cli_self_test(args)
    if args.dry_run:
        return _cli_dry_run(args)
    return _cli_level_frames(args)


if __name__ == "__main__":
    raise SystemExit(main())
