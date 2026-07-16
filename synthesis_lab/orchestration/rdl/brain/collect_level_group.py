"""Collect transitions for a level band across ALL suite games at once.

Doctrine
--------
Train **level groups**, not one game forever:
  e.g. level_lo=0, level_hi=1 → L0 and L1 experience from every game
  in one interleaved cohort so the model learns **robust general patterns**.

Honest progression only — no on_set_level injection.
If Arcade unavailable for a game, emit multi-variate pattern growth synthetics
tagged with that game_id so the cohort stays multi-game.

Legal: env frames only. No spill/win-bank labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .suite_games import (
    DEFAULT_SUITE_ENV_DIR,
    round_robin,
    suite_24,
)

DEFAULT_STEPS_PER_GAME = 16
DEFAULT_MAX_ACTIONS = 80


def _synthetic_game_band(
    game_id: str,
    n: int,
    seed: int,
    level_lo: int,
    level_hi: int,
) -> list[dict[str, Any]]:
    """Per-game multi-variate growth stub when Arcade fails (still multi-game)."""
    g = torch.Generator().manual_seed(seed + sum(ord(c) for c in game_id) * 17)
    h = w = 64
    n_colors = 4 + (sum(ord(c) for c in game_id) % 5)
    board = torch.zeros((h, w), dtype=torch.long)
    for _ in range(3 + (seed % 4)):
        y = int(torch.randint(3, h - 3, (1,), generator=g).item())
        x = int(torch.randint(3, w - 3, (1,), generator=g).item())
        c = 1 + int(torch.randint(0, n_colors, (1,), generator=g).item())
        board[y - 1 : y + 2, x - 1 : x + 2] = c
    rows: list[dict[str, Any]] = []
    cur = board.clone()
    # fake levels: spend half of steps at lo, half at hi (band presence)
    for t in range(n):
        lvl = level_lo if t < max(1, n // 2) else min(level_hi, level_lo + 1)
        action = 1 + int(torch.randint(0, 6, (1,), generator=g).item())
        nxt = cur.clone()
        ys, xs = torch.nonzero(cur > 0, as_tuple=True)
        if ys.numel():
            idx = torch.randperm(ys.numel(), generator=g)[: min(200, ys.numel())]
            for i in idx.tolist():
                y, x = int(ys[i]), int(xs[i])
                c = int(cur[y, x])
                k = int(torch.randint(0, 4, (1,), generator=g).item())
                dy, dx = ((0, 1), (0, -1), (1, 0), (-1, 0))[k]
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and int(nxt[ny, nx]) == 0:
                    nxt[ny, nx] = c
        rows.append(
            {
                "frame": cur.clone(),
                "action": action,
                "frame_next": nxt.clone(),
                "game_id": game_id,
                "levels_completed": int(lvl),
                "level_lo": int(level_lo),
                "level_hi": int(level_hi),
                "source": "pattern_growth_synthetic",
                "in_band": True,
            }
        )
        cur = nxt
    return rows


def _last_frame(fd) -> np.ndarray:
    frames = fd.frame
    if not isinstance(frames, list):
        frames = [frames]
    arr = np.asarray(frames[-1], dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"bad frame shape {arr.shape}")
    return arr


def _available_ids(fd, action_of_id: dict) -> list[int]:
    avail = getattr(fd, "available_actions", None)
    ids: list[int] = []
    for a in avail or []:
        name = getattr(a, "name", str(a))
        if name.startswith("ACTION") and name[6:].isdigit():
            i = int(name[6:])
            if i in action_of_id:
                ids.append(i)
    return ids or list(action_of_id.keys())


def collect_game_level_band(
    game_id: str,
    *,
    level_lo: int = 0,
    level_hi: int = 1,
    env_dir: Path | None = None,
    seed: int = 0,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    max_in_band: int = DEFAULT_STEPS_PER_GAME,
    policy_fn: Any | None = None,
    epsilon: float = 1.0,
) -> dict[str, Any]:
    """Play one game from reset; keep transitions while levels in [lo, hi].

    Stops when levels leave the band, game ends, or max_actions hit.
    No level teleport.

    policy_fn(frame_np, available_ids, rng) -> action_id (optional).
    epsilon: prob of random explore vs policy (1.0 = pure random).
    """
    env_dir = Path(env_dir) if env_dir is not None else DEFAULT_SUITE_ENV_DIR
    try:
        from arc_agi import Arcade
        from arc_agi.base import OperationMode
        from arcengine import GameAction, GameState
    except Exception as exc:
        rows = _synthetic_game_band(game_id, max_in_band, seed, level_lo, level_hi)
        return {
            "game_id": game_id,
            "rows": rows,
            "status": "degraded",
            "note": f"import_fail:{type(exc).__name__}",
            "level_lo": level_lo,
            "level_hi": level_hi,
            "max_levels_seen": max((r["levels_completed"] for r in rows), default=0),
            "n_level_ups": 0,
            "mean_delta_frac": 0.0,
            "policy": "synthetic",
        }

    action_of_id = {
        1: GameAction.ACTION1,
        2: GameAction.ACTION2,
        3: GameAction.ACTION3,
        4: GameAction.ACTION4,
        5: GameAction.ACTION5,
        6: GameAction.ACTION6,
    }

    try:
        arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(env_dir),
        )
        env = arcade.make(game_id, seed=seed)
        if env is None:
            rows = _synthetic_game_band(game_id, max_in_band, seed, level_lo, level_hi)
            return {
                "game_id": game_id,
                "rows": rows,
                "status": "degraded",
                "note": "make_none",
                "level_lo": level_lo,
                "level_hi": level_hi,
                "max_levels_seen": level_lo,
                "n_level_ups": 0,
                "mean_delta_frac": 0.0,
                "policy": "synthetic",
            }

        rng = np.random.default_rng(seed + hash(game_id) % 10_000)
        fd = env.reset()
        if fd is None:
            rows = _synthetic_game_band(game_id, max_in_band, seed, level_lo, level_hi)
            return {
                "game_id": game_id,
                "rows": rows,
                "status": "degraded",
                "note": "reset_none",
                "level_lo": level_lo,
                "level_hi": level_hi,
                "max_levels_seen": 0,
                "n_level_ups": 0,
                "mean_delta_frac": 0.0,
                "policy": "synthetic",
            }

        prev = _last_frame(fd)
        levels = int(getattr(fd, "levels_completed", 0) or 0)
        max_seen = levels
        rows: list[dict[str, Any]] = []
        left_band = False
        n_level_ups = 0
        delta_fracs: list[float] = []
        n_policy = 0
        n_random = 0
        eps = float(max(0.0, min(1.0, epsilon)))

        for _step in range(max_actions):
            state = getattr(fd, "state", None)
            if state in (GameState.GAME_OVER, GameState.WIN):
                break
            levels = int(getattr(fd, "levels_completed", 0) or 0)
            max_seen = max(max_seen, levels)
            if levels > level_hi:
                left_band = True
                break
            # only store transitions while currently in band
            in_band = level_lo <= levels <= level_hi

            ids = _available_ids(fd, action_of_id)
            use_policy = (
                policy_fn is not None
                and eps < 1.0
                and float(rng.random()) >= eps
            )
            if use_policy:
                try:
                    aid = int(policy_fn(prev, ids, rng))
                    if aid not in ids:
                        aid = int(rng.choice(ids))
                        n_random += 1
                    else:
                        n_policy += 1
                except Exception:
                    aid = int(rng.choice(ids))
                    n_random += 1
            else:
                aid = int(rng.choice(ids))
                n_random += 1
            data = None
            if aid == 6:
                data = {
                    "x": int(rng.integers(0, max(1, prev.shape[1]))),
                    "y": int(rng.integers(0, max(1, prev.shape[0]))),
                }
            fd = env.step(action_of_id[aid], data=data)
            if fd is None:
                break
            nxt = _last_frame(fd)
            levels_after = int(getattr(fd, "levels_completed", 0) or 0)
            max_seen = max(max_seen, levels_after)
            if levels_after > levels:
                n_level_ups += 1
            dfrac = float(np.mean(prev != nxt)) if prev.shape == nxt.shape else 0.0
            delta_fracs.append(dfrac)
            if in_band and len(rows) < max_in_band:
                rows.append(
                    {
                        "frame": torch.as_tensor(prev, dtype=torch.long),
                        "action": aid,
                        "frame_next": torch.as_tensor(nxt, dtype=torch.long),
                        "game_id": game_id,
                        "levels_completed": int(levels),
                        "levels_after": int(levels_after),
                        "level_lo": int(level_lo),
                        "level_hi": int(level_hi),
                        "source": "arcade_offline",
                        "in_band": True,
                        "delta_frac": dfrac,
                        "level_up": bool(levels_after > levels),
                    }
                )
            prev = nxt
            if levels_after > level_hi:
                left_band = True
                break
            if len(rows) >= max_in_band and levels_after >= level_lo:
                # enough in-band data for this game
                break

        mean_delta = float(sum(delta_fracs) / len(delta_fracs)) if delta_fracs else 0.0
        pol_name = "epsilon_greedy" if policy_fn is not None else "random"
        if not rows:
            rows = _synthetic_game_band(game_id, max_in_band, seed, level_lo, level_hi)
            return {
                "game_id": game_id,
                "rows": rows,
                "status": "degraded",
                "note": "no_in_band_rows",
                "level_lo": level_lo,
                "level_hi": level_hi,
                "max_levels_seen": max_seen,
                "n_level_ups": n_level_ups,
                "mean_delta_frac": mean_delta,
                "policy": pol_name,
            }

        return {
            "game_id": game_id,
            "rows": rows,
            "status": "ok",
            "note": (
                f"in_band={len(rows)} left_band={left_band} max_lvl={max_seen} "
                f"ups={n_level_ups} pol={n_policy} rnd={n_random} eps={eps:.2f}"
            ),
            "level_lo": level_lo,
            "level_hi": level_hi,
            "max_levels_seen": max_seen,
            "n_in_band": len(rows),
            "n_level_ups": n_level_ups,
            "mean_delta_frac": mean_delta,
            "n_policy_actions": n_policy,
            "n_random_actions": n_random,
            "epsilon": eps,
            "policy": pol_name,
        }
    except Exception as exc:
        rows = _synthetic_game_band(game_id, max_in_band, seed, level_lo, level_hi)
        return {
            "game_id": game_id,
            "rows": rows,
            "status": "degraded",
            "note": f"runtime:{type(exc).__name__}:{exc}",
            "level_lo": level_lo,
            "level_hi": level_hi,
            "max_levels_seen": 0,
            "n_level_ups": 0,
            "mean_delta_frac": 0.0,
            "policy": "synthetic",
        }


def collect_level_group_cohort(
    *,
    level_lo: int = 0,
    level_hi: int = 1,
    game_ids: list[str] | None = None,
    env_dir: Path | None = None,
    seed: int = 0,
    max_actions: int = DEFAULT_MAX_ACTIONS,
    max_in_band_per_game: int = DEFAULT_STEPS_PER_GAME,
    policy_fn: Any | None = None,
    epsilon: float = 1.0,
) -> dict[str, Any]:
    """Collect L[lo]–L[hi] transitions from **all** games, round-robin interleaved.

    This is the unit of multi-game training: one level group, every game at once.
    Optional epsilon-greedy policy (MegaMind propose) for plateau breakout.
    """
    env_dir = Path(env_dir) if env_dir is not None else DEFAULT_SUITE_ENV_DIR
    games = list(game_ids) if game_ids is not None else suite_24(env_dir)
    by_game: dict[str, list[dict[str, Any]]] = {}
    per_game_meta: list[dict[str, Any]] = []
    n_ok = n_deg = 0

    for i, gid in enumerate(games):
        pkt = collect_game_level_band(
            gid,
            level_lo=level_lo,
            level_hi=level_hi,
            env_dir=env_dir,
            seed=seed + i * 13,
            max_actions=max_actions,
            max_in_band=max_in_band_per_game,
            policy_fn=policy_fn,
            epsilon=epsilon,
        )
        by_game[gid] = list(pkt["rows"])
        per_game_meta.append(
            {
                "game_id": gid,
                "status": pkt["status"],
                "n_rows": len(pkt["rows"]),
                "note": pkt.get("note"),
                "max_levels_seen": pkt.get("max_levels_seen"),
                "n_level_ups": pkt.get("n_level_ups") or 0,
                "mean_delta_frac": pkt.get("mean_delta_frac") or 0.0,
                "source": (pkt["rows"][0].get("source") if pkt["rows"] else None),
                "policy": pkt.get("policy"),
                "epsilon": pkt.get("epsilon"),
            }
        )
        if pkt["status"] == "ok":
            n_ok += 1
        else:
            n_deg += 1

    interleaved = round_robin(by_game)
    # tensors for train_transition mini-batches
    if interleaved:
        frames = torch.stack([r["frame"] for r in interleaved], dim=0)
        actions = torch.tensor([int(r["action"]) for r in interleaved], dtype=torch.long)
        frames_next = torch.stack([r["frame_next"] for r in interleaved], dim=0)
        levels = torch.tensor(
            [int(r["levels_completed"]) for r in interleaved], dtype=torch.long
        )
        game_index = {g: i for i, g in enumerate(games)}
        game_ids_t = torch.tensor(
            [game_index[str(r["game_id"])] for r in interleaved], dtype=torch.long
        )
    else:
        frames = torch.zeros((0, 64, 64), dtype=torch.long)
        actions = torch.zeros((0,), dtype=torch.long)
        frames_next = torch.zeros((0, 64, 64), dtype=torch.long)
        levels = torch.zeros((0,), dtype=torch.long)
        game_ids_t = torch.zeros((0,), dtype=torch.long)

    max_lvls = [float(p.get("max_levels_seen") or 0) for p in per_game_meta]
    level_ups = [float(p.get("n_level_ups") or 0) for p in per_game_meta]
    deltas = [float(p.get("mean_delta_frac") or 0) for p in per_game_meta]
    return {
        "schema": "megamind.level_group_cohort.v2",
        "level_lo": int(level_lo),
        "level_hi": int(level_hi),
        "level_group": f"L{level_lo}-L{level_hi}",
        "n_games": len(games),
        "game_ids": games,
        "n_transitions": int(frames.shape[0]),
        "n_games_ok": n_ok,
        "n_games_degraded": n_deg,
        "per_game": per_game_meta,
        "frames": frames,
        "actions": actions,
        "frames_next": frames_next,
        "levels": levels,
        "game_ids_tensor": game_ids_t,
        "rows": interleaved,
        "interleaved": True,
        "doctrine": "all_games_same_level_band_for_robust_patterns",
        "env_dir": str(env_dir),
        "epsilon": float(epsilon),
        "policy": "epsilon_greedy" if policy_fn is not None else "random",
        "mean_max_levels": float(sum(max_lvls) / len(max_lvls)) if max_lvls else 0.0,
        "max_levels_any": float(max(max_lvls)) if max_lvls else 0.0,
        "total_level_ups": float(sum(level_ups)),
        "games_with_level_up": float(sum(1 for x in level_ups if x > 0)),
        "mean_delta_frac": float(sum(deltas) / len(deltas)) if deltas else 0.0,
    }
