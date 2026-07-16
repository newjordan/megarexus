#!/usr/bin/env python3
"""Collect balanced multi-game experience with persistent action-program search.

The search state persists across fresh episodes per game, but contains only
actions tried and environment-observed frame deltas. No game rules or routes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_search.algorithms.components import (
    change_regions,
    connected_components,
    dominant_background,
    frame_changed,
)
from legal_search.train.collect import (
    _discover_game_ids,
    annotate_level_weights,
    rows_to_tensors,
)
from legal_search.train.features import FEATURE_DIM, FEATURE_LAYOUT, FEATURE_SCHEMA, encode_legal_features
from legal_search.types import Action, Transition


def _load_bootstrap_programs(path: Path | None) -> dict[str, list[list]]:
    if path is None:
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    programs = {str(game): list(program) for game, program in manifest.get("programs", {}).items()}
    for game_id, source in manifest.get("program_sources", {}).items():
        artifact_path = Path(source["artifact"])
        if not artifact_path.is_absolute():
            artifact_path = ROOT / artifact_path
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        winners = artifact["search"]["per_game"][game_id]["winning_programs"]
        programs[str(game_id)] = list(winners[int(source.get("winner_index", 0))])
    return programs


def _grid(response) -> np.ndarray:
    frame = response.frame
    if isinstance(frame, list):
        frame = frame[-1]
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[-1]
    return np.asarray(arr, dtype=np.int16)


def _available(response) -> list[int]:
    out = []
    for value in getattr(response, "available_actions", []) or [1, 2, 3, 4, 5, 6]:
        try:
            out.append(int(getattr(value, "value", value)))
        except Exception:
            pass
    return sorted(set(out or [1, 2, 3, 4, 5, 6]))


def _click_candidates(frame: np.ndarray) -> list[tuple[int, int]]:
    """All non-background pixels, rare colours first, with coarse grid fallback."""
    bg = dominant_background(frame)
    values, counts = np.unique(frame[frame != bg], return_counts=True)
    rarity = {int(v): int(c) for v, c in zip(values, counts)}
    coords = np.argwhere(frame != bg)
    points = sorted(
        ((int(x), int(y)) for y, x in coords),
        key=lambda xy: (rarity.get(int(frame[xy[1], xy[0]]), frame.size), xy[1], xy[0]),
    )
    points.extend((x, y) for y in range(4, 64, 8) for x in range(4, 64, 8))
    return list(dict.fromkeys(points))


def _component_points(frame: np.ndarray, limit: int = 8) -> list[tuple[int, int]]:
    """One guaranteed on-component pixel per frame-derived object."""
    points: list[tuple[int, int]] = []
    for comp in connected_components(frame, min_area=1, max_area=1600)[:limit]:
        x0, y0, x1, y1 = [int(v) for v in comp["bbox_xyxy"]]
        color = int(comp["color"])
        coords = np.argwhere(frame[y0 : y1 + 1, x0 : x1 + 1] == color)
        if not len(coords):
            continue
        cx, cy = comp["center_xy"]
        best = min(
            ((x0 + int(x), y0 + int(y)) for y, x in coords),
            key=lambda xy: (xy[0] - float(cx)) ** 2 + (xy[1] - float(cy)) ** 2,
        )
        points.append(best)
    return list(dict.fromkeys(points))


def _bucket(value: int, *, cap: int = 15) -> int:
    """Log-bucket a non-negative observable without preserving game-scale detail."""
    return min(cap, int(math.log2(max(1, int(value)))))


def _geometry_signature(frame: np.ndarray, *, limit: int = 8) -> tuple:
    """Coarse, component-derived state key with no semantic or game identifiers."""
    grid = np.asarray(frame)
    h, w = grid.shape[:2]
    comps = connected_components(
        grid,
        min_area=1,
        max_area=max(1, h * w),
        exclude_full_span=False,
    )[:limit]
    descriptors = []
    for comp in comps:
        x0, y0, x1, y1 = [int(v) for v in comp["bbox_xyxy"]]
        cx, cy = [float(v) for v in comp["center_xy"]]
        descriptors.append(
            (
                int(comp["color"]),
                _bucket(int(comp["area"])),
                min(7, int(8.0 * cx / max(1, w))),
                min(7, int(8.0 * cy / max(1, h))),
                _bucket(x1 - x0 + 1, cap=7),
                _bucket(y1 - y0 + 1, cap=7),
            )
        )
    return (
        int(h),
        int(w),
        int(dominant_background(grid)),
        min(15, len(comps)),
        tuple(descriptors),
    )


def _delta_signature(before: np.ndarray, after: np.ndarray) -> tuple:
    """Describe observed changed regions, not inferred mechanics or goals."""
    aa = np.asarray(before)
    bb = np.asarray(after)
    mined = change_regions(aa, bb, max_regions=4)
    h, w = aa.shape[:2]
    regions = []
    for region in mined.get("regions", []):
        cx, cy = [float(v) for v in region["center_xy"]]
        regions.append(
            (
                _bucket(int(region["area"])),
                min(7, int(8.0 * cx / max(1, w))),
                min(7, int(8.0 * cy / max(1, h))),
                tuple(int(v) for v in region.get("colors_before", [])[:4]),
                tuple(int(v) for v in region.get("colors_after", [])[:4]),
            )
        )
    return (
        _bucket(int(mined.get("n_changed", 0))),
        min(8, int(mined.get("n_regions", 0))),
        tuple(regions),
    )


def _component_action_candidates(frame: np.ndarray, *, limit: int = 12) -> list[tuple[Action, tuple]]:
    """Click candidates paired with coarse observable component identities."""
    grid = np.asarray(frame)
    h, w = grid.shape[:2]
    out: list[tuple[Action, tuple]] = []
    for comp in connected_components(
        grid,
        min_area=1,
        max_area=max(1, h * w),
        exclude_full_span=False,
    )[:limit]:
        x0, y0, x1, y1 = [int(v) for v in comp["bbox_xyxy"]]
        color = int(comp["color"])
        coords = np.argwhere(grid[y0 : y1 + 1, x0 : x1 + 1] == color)
        if not len(coords):
            continue
        cx, cy = [float(v) for v in comp["center_xy"]]
        x, y = min(
            ((x0 + int(px), y0 + int(py)) for py, px in coords),
            key=lambda xy: (xy[0] - cx) ** 2 + (xy[1] - cy) ** 2,
        )
        descriptor = (
            "component",
            color,
            _bucket(int(comp["area"])),
            min(7, int(8.0 * cx / max(1, w))),
            min(7, int(8.0 * cy / max(1, h))),
            _bucket(x1 - x0 + 1, cap=7),
            _bucket(y1 - y0 + 1, cap=7),
        )
        out.append(
            (
                Action(6, x=x, y=y, source="program_collect.adaptive_delta"),
                descriptor,
            )
        )
    return out


def _debruijn(k: int, n: int) -> list[int]:
    """De Bruijn sequence B(k,n): every length-n node sequence occurs once."""
    if k <= 1:
        return [0]
    a = [0] * (k * n)
    sequence: list[int] = []

    def db(t: int, p: int) -> None:
        if t > n:
            if n % p == 0:
                sequence.extend(a[1 : p + 1])
            return
        a[t] = a[t - p]
        db(t + 1, p)
        for j in range(a[t - p] + 1, k):
            a[t] = j
            db(t + 1, t)

    db(1, 1)
    return sequence


class ProgramMemory:
    def __init__(self, seed: int, representation: str = "flat") -> None:
        self.rng = np.random.default_rng(seed)
        self.click_cursor = 0
        self.action_trials: Counter[int] = Counter()
        self.action_changes: Counter[int] = Counter()
        self.programs: set[tuple] = set()
        self.representation = representation
        self.graph_cursor = 0
        self.macro_cursor = 0
        self.grid_order = self.rng.permutation(64 * 64).tolist()
        self.grid_cursor = 0
        # adaptive_delta keeps only observable state/action/outcome statistics.
        # Values are [reward_sum, trials, changes, level_ups, game_overs].
        self.adaptive_stats: dict[tuple, list[float]] = defaultdict(
            lambda: [0.0, 0.0, 0.0, 0.0, 0.0]
        )
        self.adaptive_context_visits: Counter[tuple] = Counter()
        self.adaptive_action_trials: Counter[tuple] = Counter()
        self.adaptive_state_signatures: Counter[tuple] = Counter()
        self.adaptive_delta_signatures: Counter[tuple] = Counter()
        self.adaptive_decisions = 0
        self.adaptive_observations = 0
        self.adaptive_reward_sum = 0.0
        self._pending_adaptive: tuple[tuple, tuple, tuple] | None = None

    def begin_episode(self) -> None:
        """Drop an unobserved pending decision after reset/step failure."""
        self._pending_adaptive = None

    def _adaptive_action(
        self,
        frame: np.ndarray,
        available: list[int],
        *,
        levels_completed: int,
    ) -> Action:
        context = (int(levels_completed), _geometry_signature(frame))
        candidates: list[tuple[Action, tuple]] = []
        for aid in available:
            if aid != 6:
                candidates.append(
                    (
                        Action(int(aid), source="program_collect.adaptive_delta"),
                        ("action", int(aid)),
                    )
                )
        if 6 in available:
            candidates.extend(_component_action_candidates(frame))
            # Keep a deterministic whole-grid exploration floor. Offering only
            # every fourth decision lets learned component/action outcomes win
            # most choices instead of degenerating into another grid scan.
            if self.adaptive_decisions % 4 == 0 or not candidates:
                idx = int(self.grid_order[self.grid_cursor % len(self.grid_order)])
                candidates.append(
                    (
                        Action(
                            6,
                            x=idx % 64,
                            y=idx // 64,
                            source="program_collect.adaptive_delta",
                        ),
                        ("grid_probe", idx % 64, idx // 64),
                    )
                )
        if not candidates:
            aid = int(available[0]) if available else 1
            candidates = [
                (
                    Action(aid, source="program_collect.adaptive_delta"),
                    ("action", aid),
                )
            ]

        # Deduplicate identical executable actions while retaining the first
        # geometry-derived key for persistent credit assignment.
        deduped: list[tuple[Action, tuple]] = []
        seen: set[tuple] = set()
        for action, action_key in candidates:
            signature = action.as_tuple()
            if signature in seen:
                continue
            seen.add(signature)
            deduped.append((action, action_key))
        candidates = deduped

        untried = [
            i
            for i, (_action, action_key) in enumerate(candidates)
            if self.adaptive_stats.get(
                (context, action_key),
                (0.0, 0.0, 0.0, 0.0, 0.0),
            )[1]
            == 0
        ]
        if untried:
            # Prefer globally less-tested action abstractions, randomly breaking
            # exact ties with the run seed.
            least = min(self.adaptive_action_trials[candidates[i][1]] for i in untried)
            choices = [
                i
                for i in untried
                if self.adaptive_action_trials[candidates[i][1]] == least
            ]
            chosen = int(self.rng.choice(choices))
        else:
            total = max(1, int(self.adaptive_context_visits[context]))
            scores = []
            for _action, action_key in candidates:
                reward_sum, trials, _changes, _level_ups, _game_overs = self.adaptive_stats.get(
                    (context, action_key),
                    (0.0, 0.0, 0.0, 0.0, 0.0),
                )
                mean = reward_sum / max(1.0, trials)
                explore = math.sqrt(2.0 * math.log(total + 1.0) / max(1.0, trials))
                global_n = self.adaptive_action_trials[action_key]
                scores.append(mean + explore + 0.05 / math.sqrt(global_n + 1.0))
            best = max(scores)
            choices = [i for i, score in enumerate(scores) if abs(score - best) < 1e-12]
            chosen = int(self.rng.choice(choices))

        action, action_key = candidates[chosen]
        if action_key and action_key[0] == "grid_probe":
            self.grid_cursor += 1
        self.adaptive_context_visits[context] += 1
        self.adaptive_decisions += 1
        self._pending_adaptive = (context, action_key, action.as_tuple())
        return action

    def action(
        self,
        frame: np.ndarray,
        available: list[int],
        step: int,
        *,
        levels_completed: int = 0,
    ) -> Action:
        # Cover each discrete action before sampling; click coordinates never repeat
        # until the current frame-derived candidate list is exhausted.
        if self.representation == "adaptive_delta":
            return self._adaptive_action(
                frame,
                available,
                levels_completed=levels_completed,
            )
        if self.representation == "grid_scan" and available == [6]:
            idx = int(self.grid_order[self.grid_cursor % len(self.grid_order)])
            self.grid_cursor += 1
            return Action(
                6,
                x=idx % 64,
                y=idx // 64,
                source="program_collect.grid_scan",
            )
        if self.representation in ("component_graph", "grid_scan"):
            points = _component_points(frame)
            if 6 in available and points:
                if available == [6]:
                    order = _debruijn(len(points), 3)
                    node = order[self.graph_cursor % len(order)]
                    self.graph_cursor += 1
                    x, y = points[node]
                    return Action(6, x=x, y=y, source="program_collect.component_graph")
                # Generic object -> transform(s) -> activate macro. The transform
                # and node indices rotate across episodes without game knowledge.
                transforms = [a for a in available if a not in (5, 6)] or [a for a in available if a != 6]
                macro_len = 4
                phase = self.macro_cursor % macro_len
                macro_id = self.macro_cursor // macro_len
                self.macro_cursor += 1
                if phase == 0:
                    x, y = points[macro_id % len(points)]
                    return Action(6, x=x, y=y, source="program_collect.component_graph")
                if phase == macro_len - 1 and 5 in available:
                    return Action(5, source="program_collect.component_graph")
                aid = transforms[(macro_id + phase - 1) % len(transforms)]
                return Action(aid, source="program_collect.component_graph")
        untried = [aid for aid in available if self.action_trials[aid] == 0]
        aid = untried[0] if untried else int(self.rng.choice(available))
        if 6 not in available and aid == 6:
            aid = available[0]
        if aid != 6:
            return Action(aid, source="program_collect")
        points = _click_candidates(frame)
        x, y = points[self.click_cursor % len(points)]
        self.click_cursor += 1
        return Action(6, x=x, y=y, source="program_collect")

    def observe(
        self,
        action: Action,
        changed: bool,
        *,
        frame_before: np.ndarray | None = None,
        frame_after: np.ndarray | None = None,
        levels_before: int = 0,
        levels_after: int = 0,
        state_after: str = "",
    ) -> None:
        aid = int(action.action_id)
        self.action_trials[aid] += 1
        self.action_changes[aid] += int(bool(changed))
        if self.representation != "adaptive_delta":
            return
        if frame_before is not None and frame_after is not None:
            after_signature = _geometry_signature(frame_after)
            delta_signature = _delta_signature(frame_before, frame_after)
            new_state = self.adaptive_state_signatures[after_signature] == 0
            new_delta = bool(changed) and self.adaptive_delta_signatures[delta_signature] == 0
            self.adaptive_state_signatures[after_signature] += 1
            self.adaptive_delta_signatures[delta_signature] += 1
        else:
            new_state = False
            new_delta = False

        # Bootstrap replay is observed for novelty, but only adaptive choices
        # receive action credit. This prevents stale pending credit after resets.
        if action.source != "program_collect.adaptive_delta" or self._pending_adaptive is None:
            self._pending_adaptive = None
            return
        context, action_key, expected_action = self._pending_adaptive
        self._pending_adaptive = None
        if action.as_tuple() != expected_action:
            return
        gained = max(0, int(levels_after) - int(levels_before))
        terminal = str(state_after).upper()
        game_over = "GAME_OVER" in terminal
        win = "WIN" in terminal
        reward = 100.0 * gained
        reward += 50.0 if win else 0.0
        reward -= 25.0 if game_over else 0.0
        reward += 1.0 if changed else -0.05
        reward += 1.0 if new_delta else 0.0
        reward += 0.25 if new_state else 0.0
        row = self.adaptive_stats[(context, action_key)]
        row[0] += reward
        row[1] += 1.0
        row[2] += float(bool(changed))
        row[3] += float(gained)
        row[4] += float(game_over)
        self.adaptive_action_trials[action_key] += 1
        self.adaptive_observations += 1
        self.adaptive_reward_sum += reward

    def adaptive_snapshot(self) -> dict:
        return {
            "decisions": int(self.adaptive_decisions),
            "observations": int(self.adaptive_observations),
            "contexts": len(self.adaptive_context_visits),
            "context_actions": len(self.adaptive_stats),
            "state_signatures": len(self.adaptive_state_signatures),
            "delta_signatures": len(self.adaptive_delta_signatures),
            "game_overs": int(sum(row[4] for row in self.adaptive_stats.values())),
            "level_ups": int(sum(row[3] for row in self.adaptive_stats.values())),
            "reward_sum": float(self.adaptive_reward_sum),
        }


def collect(args) -> tuple[list[dict], dict]:
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine import GameAction

    env_dir = ROOT / "environment_files"
    bootstrap_programs = _load_bootstrap_programs(args.bootstrap_programs)
    games = _discover_game_ids(env_dir)
    if args.games:
        requested = set(args.games)
        games = [game for game in games if game in requested]
        missing = sorted(requested.difference(games))
        if missing:
            raise SystemExit(f"requested games not installed: {missing}")
    memories = {
        g: ProgramMemory(args.seed + i * 100003, representation=args.representation)
        for i, g in enumerate(games)
    }
    act_map = {i: getattr(GameAction, f"ACTION{i}") for i in range(1, 7)}
    rows: list[dict] = []
    receipts: dict[str, dict] = {
        g: {
            "episodes": 0,
            "rows": 0,
            "level_ups": 0,
            "best_level": 0,
            "states": {},
            "winning_programs": [],
            "target_programs": [],
        }
        for g in games
    }
    for round_idx in range(args.episodes_per_game):
        for game_id in games:
            memory = memories[game_id]
            memory.begin_episode()
            arcade = Arcade(
                arc_api_key="anonymous",
                environments_dir=str(env_dir),
                operation_mode=OperationMode.OFFLINE,
            )
            card = arcade.open_scorecard()
            env = arcade.make(game_id, scorecard_id=card)
            response = env.reset()
            frame = _grid(response)
            levels = int(getattr(response, "levels_completed", 0) or 0)
            transitions: list[Transition] = []
            episode_rows: list[dict] = []
            signature: list[tuple] = []
            for step in range(args.max_actions):
                available = _available(response)
                features = encode_legal_features(
                    frame,
                    game_id=game_id,
                    transitions=transitions[-16:],
                    available_actions=available,
                    levels_completed=levels,
                    run_consume=True,
                )
                bootstrap = bootstrap_programs.get(game_id, [])
                if levels == 0 and step < len(bootstrap):
                    aid, x, y = bootstrap[step]
                    action = Action(
                        int(aid),
                        x=None if x is None else int(x),
                        y=None if y is None else int(y),
                        source="program_collect.bootstrap_memory",
                    )
                else:
                    action = memory.action(
                        frame,
                        available,
                        step,
                        levels_completed=levels,
                    )
                signature.append(action.as_tuple())
                before = frame.copy()
                data = {"x": int(action.x or 0), "y": int(action.y or 0)} if action.action_id == 6 else None
                try:
                    response = env.step(act_map[action.action_id], data=data)
                except Exception:
                    break
                frame2 = _grid(response)
                levels2 = int(getattr(response, "levels_completed", 0) or 0)
                changed = frame_changed(before, frame2)
                state_after = str(getattr(response, "state", ""))
                memory.observe(
                    action,
                    changed,
                    frame_before=before,
                    frame_after=frame2,
                    levels_before=levels,
                    levels_after=levels2,
                    state_after=state_after,
                )
                tr = Transition(
                    game_id=game_id,
                    step=step,
                    frame_before=before,
                    action=action,
                    frame_after=frame2,
                    levels_before=levels,
                    levels_after=levels2,
                    state_after=state_after,
                    frame_changed=changed,
                    levels_up=levels2 > levels,
                )
                transitions.append(tr)
                episode_rows.append({
                    "frame": before,
                    "features": features.astype(np.float32),
                    "action_id": int(action.action_id),
                    "x": int(action.x or 0),
                    "y": int(action.y or 0),
                    "did_change": bool(changed),
                    "levels_up": bool(tr.levels_up),
                    "levels_after": levels2,
                    "game_id": game_id,
                    "source": action.source,
                })
                frame, levels = frame2, levels2
                state = str(getattr(response, "state", ""))
                if levels2 >= args.target_level or "GAME_OVER" in state.upper() or "WIN" in state.upper():
                    break
            memory.programs.add(tuple(signature))
            annotate_level_weights(episode_rows)
            rows.extend(episode_rows)
            receipt = receipts[game_id]
            receipt["episodes"] += 1
            receipt["rows"] += len(episode_rows)
            receipt["level_ups"] += sum(int(r["levels_up"]) for r in episode_rows)
            receipt["best_level"] = max(int(receipt["best_level"]), levels)
            if any(r["levels_up"] for r in episode_rows):
                receipt["winning_programs"].append(
                    [[int(a), None if x is None else int(x), None if y is None else int(y)] for a, x, y in signature]
                )
            if levels >= args.target_level:
                receipt["target_programs"].append(
                    [[int(a), None if x is None else int(x), None if y is None else int(y)] for a, x, y in signature]
                )
            state = str(getattr(response, "state", ""))
            receipt["states"][state] = int(receipt["states"].get(state, 0)) + 1
            arcade.close_scorecard(card)
            print(
                f"[program_collect] round={round_idx+1}/{args.episodes_per_game} game={game_id} "
                f"rows={len(episode_rows)} level={levels} total={len(rows)}",
                flush=True,
            )
    for game_id, memory in memories.items():
        receipts[game_id]["action_trials"] = dict(memory.action_trials)
        receipts[game_id]["action_changes"] = dict(memory.action_changes)
        receipts[game_id]["unique_programs"] = len(memory.programs)
        if memory.representation == "adaptive_delta":
            receipts[game_id]["adaptive_delta"] = memory.adaptive_snapshot()
    return rows, {"games": games, "per_game": receipts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes-per-game", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=150)
    parser.add_argument("--seed", type=int, default=71003)
    parser.add_argument(
        "--representation",
        choices=["flat", "component_graph", "grid_scan", "adaptive_delta"],
        default="flat",
    )
    parser.add_argument("--games", nargs="*", default=None, help="Optional installed game-id subset for targeted signal acquisition")
    parser.add_argument("--bootstrap-programs", type=Path, default=None)
    parser.add_argument("--target-level", type=int, default=1)
    args = parser.parse_args()
    from legal_search.train.mining_gate import assert_mining_complete
    assert_mining_complete()
    rows, search = collect(args)
    if not rows:
        raise SystemExit("no program rows collected")
    rng = np.random.default_rng(args.seed)
    by_game: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_game[str(row["game_id"])].append(row)
    target = min(len(game_rows) for game_rows in by_game.values())
    balanced: list[dict] = []
    balanced_counts: dict[str, int] = {}
    for game_id, game_rows in sorted(by_game.items()):
        priority = [r for r in game_rows if r.get("levels_up") or r.get("success_window")]
        priority_ids = {id(r) for r in priority}
        ordinary = [r for r in game_rows if id(r) not in priority_ids]
        keep = priority[:target]
        remaining = target - len(keep)
        if remaining > 0:
            idx = rng.permutation(len(ordinary))[:remaining]
            keep.extend(ordinary[int(i)] for i in idx)
        balanced.extend(keep)
        balanced_counts[game_id] = len(keep)
    rows = balanced
    search["balanced_rows_per_game"] = balanced_counts
    order = rng.permutation(len(rows))
    rows = [rows[int(i)] for i in order]
    n_val = max(32, len(rows) // 10)
    train_rows, val_rows = rows[n_val:], rows[:n_val]
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
            "collector": "persistent_env_graded_program_search_v1",
            "legal": True,
            "episodes_per_game": args.episodes_per_game,
            "max_actions": args.max_actions,
            "seed": args.seed,
            "representation": args.representation,
            "bootstrap_programs": None if args.bootstrap_programs is None else str(args.bootstrap_programs),
            "target_level": args.target_level,
            "search": search,
            "n_levels_up": sum(int(r["levels_up"]) for r in rows),
            "banned_lineage": ["sp80_spill_model", "dropletsim", "human_mined_routes", "fixture_win_banks_as_labels"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    manifest = args.out.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(payload["meta"], indent=2) + "\n", encoding="utf-8")
    print(f"[program_collect] wrote {args.out} train={len(train_rows)} val={len(val_rows)}")
    print(json.dumps(search, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
