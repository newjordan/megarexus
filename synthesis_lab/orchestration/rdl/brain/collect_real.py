"""Transition collector for MegaMind — multi-variate pattern growth first.

Default: synthetic **dynamic multi-variate pattern** transitions (growing
blobs, multi-color fields, accrete/erode). Optional offline Arcade fetch is
available but **not** the garden focus (geometry is game-agnostic).

Legal: frame tensors only — no spill_model, no fixture labels, no human routes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

GAME_ID = "pattern-growth-mv"
# Optional legacy offline env id (not garden focus)
_OPTIONAL_ARCADE_GAME_ID = "sp80-589a99af"
DEFAULT_STEPS = 8


def _repo_root() -> Path:
    # brain/ → rdl/ → orchestration/ → synthesis_lab/ → repo
    return Path(__file__).resolve().parents[4]


def _synthetic(n: int = DEFAULT_STEPS, seed: int = 0) -> dict[str, Any]:
    """Multi-variate growing patterns (not SP80-shaped noise).

    Starts with a few colored seeds and expands / competes over steps so
    G5/G7/G23 delta geometry has real signal.
    """
    g = torch.Generator().manual_seed(seed)
    h = w = 64
    n_colors = 6
    board = torch.zeros((h, w), dtype=torch.long)
    # seed multiple color variables at random sites
    n_seeds = 5 + int(torch.randint(0, 4, (1,), generator=g).item())
    for _ in range(n_seeds):
        y = int(torch.randint(4, h - 4, (1,), generator=g).item())
        x = int(torch.randint(4, w - 4, (1,), generator=g).item())
        c = 1 + int(torch.randint(0, n_colors, (1,), generator=g).item())
        board[y - 1 : y + 2, x - 1 : x + 2] = c

    frames_list: list[torch.Tensor] = []
    actions_list: list[int] = []
    next_list: list[torch.Tensor] = []
    cur = board.clone()
    for t in range(n):
        frames_list.append(cur.clone())
        action = 1 + int(torch.randint(0, 6, (1,), generator=g).item())
        actions_list.append(action)
        nxt = cur.clone()
        # growth: expand each non-zero into a random 4-neighbor (multi-var)
        ys, xs = torch.nonzero(cur > 0, as_tuple=True)
        if ys.numel() > 0:
            # sample subset for speed
            idx = torch.randperm(ys.numel(), generator=g)[: min(400, ys.numel())]
            for i in idx.tolist():
                y, x = int(ys[i]), int(xs[i])
                c = int(cur[y, x])
                dyx = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                k = int(torch.randint(0, 4, (1,), generator=g).item())
                ny, nx = y + dyx[k][0], x + dyx[k][1]
                if 0 <= ny < h and 0 <= nx < w and int(nxt[ny, nx]) == 0:
                    # slight action bias: odd actions prefer grow, even may erode
                    if action % 2 == 1 or int(torch.randint(0, 3, (1,), generator=g).item()) > 0:
                        nxt[ny, nx] = c
                elif 0 <= ny < h and 0 <= nx < w and action % 2 == 0:
                    if int(torch.randint(0, 5, (1,), generator=g).item()) == 0:
                        nxt[y, x] = 0  # rare erosion
        # inject a new seed occasionally (new variable growth)
        if t % 3 == 2:
            y = int(torch.randint(2, h - 2, (1,), generator=g).item())
            x = int(torch.randint(2, w - 2, (1,), generator=g).item())
            c = 1 + int(torch.randint(0, n_colors, (1,), generator=g).item())
            if int(nxt[y, x]) == 0:
                nxt[y, x] = c
        next_list.append(nxt.clone())
        cur = nxt

    frames = torch.stack(frames_list, dim=0)
    frames_next = torch.stack(next_list, dim=0)
    actions = torch.tensor(actions_list, dtype=torch.long)
    return {
        "frames": frames,
        "actions": actions,
        "frames_next": frames_next,
        "n": n,
        "source": "pattern_growth_synthetic",
        "status": "ok",
        "game_id": GAME_ID,
        "note": "dynamic multi-variate pattern growth transitions (game-agnostic)",
    }


def collect_sp80_few(
    n_steps: int = DEFAULT_STEPS,
    seed: int = 0,
) -> dict[str, Any]:
    """Optional legacy Arcade offline collector (not garden focus).

    Prefer ``collect_transitions`` / multi-variate pattern synthetics.
    """
    try:
        from arc_agi import Arcade
        from arc_agi.base import OperationMode
        from arcengine import GameAction, GameState
    except Exception as exc:
        out = _synthetic(n_steps, seed=seed)
        out["note"] = f"import fail: {type(exc).__name__}:{exc}"
        return out

    action_of_id = {
        1: GameAction.ACTION1,
        2: GameAction.ACTION2,
        3: GameAction.ACTION3,
        4: GameAction.ACTION4,
        5: GameAction.ACTION5,
        6: GameAction.ACTION6,
    }

    def last_frame(frame_data):
        frames = frame_data.frame
        if not isinstance(frames, list):
            frames = [frames]
        import numpy as np

        arr = np.asarray(frames[-1], dtype="int64")
        if arr.shape != (64, 64):
            raise ValueError(f"unexpected frame shape {arr.shape}")
        return arr

    def available_ids(frame_data) -> list[int]:
        avail = getattr(frame_data, "available_actions", None)
        ids: list[int] = []
        for a in avail or []:
            name = getattr(a, "name", str(a))
            if name.startswith("ACTION") and name[6:].isdigit():
                i = int(name[6:])
                if i in action_of_id:
                    ids.append(i)
        return ids or [1, 2, 3, 4, 5, 6]

    try:
        root = _repo_root()
        env_dir = root / "environment_files"
        arcade = Arcade(
            operation_mode=OperationMode.OFFLINE,
            environments_dir=str(env_dir),
        )
        env = arcade.make(_OPTIONAL_ARCADE_GAME_ID, seed=seed)
        if env is None:
            out = _synthetic(n_steps, seed=seed)
            out["note"] = f"arcade.make({_OPTIONAL_ARCADE_GAME_ID}) returned None"
            return out

        import numpy as np

        rng = np.random.default_rng(seed)
        fd = env.reset()
        prev = last_frame(fd)
        rows_s: list = []
        rows_a: list[int] = []
        rows_sp: list = []

        for _ in range(n_steps):
            if fd is None:
                break
            state = getattr(fd, "state", None)
            if state in (GameState.GAME_OVER, GameState.WIN):
                break
            ids = available_ids(fd)
            aid = int(rng.choice(ids))
            data = None
            if aid == 6:
                data = {"x": int(rng.integers(64)), "y": int(rng.integers(64))}
            fd = env.step(action_of_id[aid], data=data)
            if fd is None:
                break
            nxt = last_frame(fd)
            rows_s.append(prev)
            rows_a.append(aid)
            rows_sp.append(nxt)
            prev = nxt

        if not rows_s:
            out = _synthetic(n_steps, seed=seed)
            out["note"] = "no transitions from optional arcade env"
            return out

        return {
            "frames": torch.as_tensor(np.stack(rows_s), dtype=torch.long),
            "actions": torch.as_tensor(rows_a, dtype=torch.long),
            "frames_next": torch.as_tensor(np.stack(rows_sp), dtype=torch.long),
            "n": len(rows_a),
            "source": "optional_arcade",
            "status": "ok",
            "game_id": _OPTIONAL_ARCADE_GAME_ID,
            "note": f"optional arcade steps={len(rows_a)} (not pattern-growth default)",
        }
    except Exception as exc:
        out = _synthetic(n_steps, seed=seed)
        out["note"] = f"runtime fail: {type(exc).__name__}:{exc}"
        return out


def collect_transitions(
    n_steps: int = DEFAULT_STEPS,
    seed: int = 0,
    prefer_real: bool = False,
) -> dict[str, Any]:
    """Public entry: multi-variate pattern growth by default.

    Set prefer_real=True only for optional Arcade offline (legacy).
    """
    if prefer_real:
        return collect_sp80_few(n_steps=n_steps, seed=seed)
    return _synthetic(n_steps, seed=seed)


if __name__ == "__main__":
    batch = collect_transitions(n_steps=4, seed=0)
    print(
        f"status={batch['status']} source={batch['source']} n={batch['n']} "
        f"frames={tuple(batch['frames'].shape)} note={batch['note']}"
    )
