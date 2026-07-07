#!/usr/bin/env python3
"""Phase-A observation harness for any locally-downloaded ARC-AGI-3 game.

Recon lane tool (2026-07-06). OBSERVATION + LEGAL ENV INTERACTION ONLY:
random/scripted probing actions through arc_agi OFFLINE mode; no training,
no route replay, no live scorecards.

What it does per game:
  1. RESET and dump the initial frame (ASCII + color histogram + available actions).
  2. Single-action probes: fresh reset, one action, pixel-diff read (what moved, where).
  3. Random rollouts: E episodes x S steps of random available actions, harvesting a
     frame corpus (npz) with states / levels_completed / actions per step.

Usage:
    python observe_game.py --game ls20 --episodes 8 --steps 60 --seed 7 \
        --out-dir synthesis_lab/experiments/other_games/corpora
"""
from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

from arc_agi import Arcade                    # noqa: E402
from arc_agi.base import OperationMode        # noqa: E402
from arcengine import GameAction              # noqa: E402

HERE = Path(__file__).resolve().parent
ENV_DIR = HERE / "environment_files"
HEX = "0123456789ABCDEF"

SIMPLE = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
          4: GameAction.ACTION4, 5: GameAction.ACTION5, 7: GameAction.ACTION7}
CLICK_ID = 6


def ascii_frame(g: np.ndarray, every: int = 1) -> str:
    """Hex-digit ASCII dump ('.' for color 0). every=2 halves resolution."""
    rows = []
    for y in range(0, g.shape[0], every):
        rows.append("".join(("." if v == 0 else HEX[int(v) % 16])
                            for v in g[y, ::every]))
    return "\n".join(rows)


def last_grid(fd) -> np.ndarray:
    f = fd.frame
    g = f[-1] if isinstance(f, list) else f
    return np.asarray(g, dtype=np.int16)


def all_grids(fd) -> list[np.ndarray]:
    f = fd.frame
    gs = f if isinstance(f, list) else [f]
    return [np.asarray(g, dtype=np.int16) for g in gs]


def hist(g: np.ndarray) -> dict[int, int]:
    v, c = np.unique(g, return_counts=True)
    return {int(a): int(b) for a, b in zip(v, c)}


def diff_read(pre: np.ndarray, post: np.ndarray) -> dict:
    """Frames-only read of what one action changed."""
    if pre.shape != post.shape:
        return {"changed": -1, "note": f"shape {pre.shape}->{post.shape}"}
    m = pre != post
    n = int(m.sum())
    out: dict = {"changed": n}
    if n:
        ys, xs = np.nonzero(m)
        out["bbox_yxyx"] = [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]
        gained = Counter(post[m].tolist())
        lost = Counter(pre[m].tolist())
        out["colors_gained"] = {int(k): int(v) for k, v in gained.most_common(5)}
        out["colors_lost"] = {int(k): int(v) for k, v in lost.most_common(5)}
        # centroid shift of each color that both lost and gained pixels (movement signal)
        moves = {}
        for col in set(gained) & set(lost):
            src = np.argwhere((pre == col) & m)
            dst = np.argwhere((post == col) & m)
            if len(src) and len(dst):
                d = dst.mean(axis=0) - src.mean(axis=0)
                moves[int(col)] = [round(float(d[0]), 1), round(float(d[1]), 1)]  # (dy, dx)
        if moves:
            out["color_centroid_shift_dydx"] = moves
    return out


def make_env(game: str, seed: int):
    arc = Arcade(arc_api_key="anonymous", environments_dir=str(ENV_DIR),
                 operation_mode=OperationMode.OFFLINE)
    env = arc.make(game, seed=seed)
    if env is None:
        raise SystemExit(f"could not make {game} from {ENV_DIR}")
    return env


def probe_actions(game: str, seed: int, avail: list[int], log) -> None:
    """One fresh env per action; single step; report the frame diff."""
    log(f"\n== single-action probes (seed {seed}) ==")
    for aid in avail:
        if aid == 0:
            continue
        env = make_env(game, seed)
        fd0 = env.reset()
        pre = last_grid(fd0)
        if aid == CLICK_ID:
            for (cx, cy) in [(16, 16), (32, 32), (48, 48)]:
                env2 = make_env(game, seed)
                env2.reset()
                fd = env2.step(GameAction.ACTION6, data={"x": cx, "y": cy})
                if fd is None:
                    log(f"  ACTION6 click({cx},{cy}): step returned None")
                    continue
                d = diff_read(pre, last_grid(fd))
                log(f"  ACTION6 click({cx},{cy}): state={fd.state.value} "
                    f"anim_frames={len(all_grids(fd))} diff={json.dumps(d)}")
        elif aid in SIMPLE:
            fd = env.step(SIMPLE[aid])
            if fd is None:
                log(f"  ACTION{aid}: step returned None")
                continue
            d = diff_read(pre, last_grid(fd))
            log(f"  ACTION{aid}: state={fd.state.value} "
                f"anim_frames={len(all_grids(fd))} diff={json.dumps(d)}")


def random_rollouts(game: str, seed: int, episodes: int, steps: int,
                    out_dir: Path, log) -> Path:
    rng = np.random.default_rng(seed)
    frames, acts, states, levels, ep_ids, clicks = [], [], [], [], [], []
    log(f"\n== random rollouts: {episodes} episodes x <= {steps} steps ==")
    for ep in range(episodes):
        env = make_env(game, seed + ep)
        fd = env.reset()
        frames.append(last_grid(fd).astype(np.int8)); acts.append(0)
        states.append(fd.state.value); levels.append(int(fd.levels_completed))
        ep_ids.append(ep); clicks.append((-1, -1))
        wins = 0
        for t in range(steps):
            avail = [a for a in (fd.available_actions or [1, 2, 3, 4, 5, 6]) if a != 0]
            if not avail:
                break
            aid = int(rng.choice(avail))
            if aid == CLICK_ID:
                x, y = int(rng.integers(64)), int(rng.integers(64))
                fd2 = env.step(GameAction.ACTION6, data={"x": x, "y": y})
                click = (x, y)
            elif aid in SIMPLE:
                fd2 = env.step(SIMPLE[aid]); click = (-1, -1)
            else:
                continue
            if fd2 is None:
                break
            fd = fd2
            frames.append(last_grid(fd).astype(np.int8)); acts.append(aid)
            states.append(fd.state.value); levels.append(int(fd.levels_completed))
            ep_ids.append(ep); clicks.append(click)
            if fd.state.value == "GAME_OVER":
                log(f"  ep{ep}: GAME_OVER at step {t+1} (levels={fd.levels_completed})")
                break
            if fd.state.value == "WIN":
                log(f"  ep{ep}: WIN at step {t+1}")
                break
            if int(fd.levels_completed) > wins:
                wins = int(fd.levels_completed)
                log(f"  ep{ep}: levels_completed -> {wins} at step {t+1} (random!)")
        else:
            log(f"  ep{ep}: budget exhausted, state={fd.state.value} "
                f"levels={fd.levels_completed}")
        if fd.state.value in ("GAME_OVER", "WIN"):
            continue
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"obs_{game}_seed{seed}_e{episodes}s{steps}.npz"
    np.savez_compressed(
        out, frames=np.stack(frames), actions=np.array(acts, dtype=np.int8),
        states=np.array(states), levels=np.array(levels, dtype=np.int16),
        episode=np.array(ep_ids, dtype=np.int16),
        clicks=np.array(clicks, dtype=np.int16))
    log(f"  corpus saved: {out} ({len(frames)} frames)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--out-dir", type=Path, default=HERE / "corpora")
    ap.add_argument("--log", type=Path, default=None)
    a = ap.parse_args()

    lines: list[str] = []

    def log(s: str) -> None:
        print(s)
        lines.append(s)

    env = make_env(a.game, a.seed)
    fd = env.reset()
    g0 = last_grid(fd)
    log(f"== {a.game} initial frame (seed {a.seed}) ==")
    log(f"grid shape: {g0.shape}  state: {fd.state.value}  "
        f"levels_completed: {fd.levels_completed}/{fd.win_levels}")
    log(f"available_actions: {fd.available_actions}")
    log(f"color histogram: {json.dumps(hist(g0))}")
    log("initial frame (2x downsampled hex):")
    log(ascii_frame(g0, every=2))

    probe_actions(a.game, a.seed, fd.available_actions or [1, 2, 3, 4, 5, 6], log)
    random_rollouts(a.game, a.seed, a.episodes, a.steps, a.out_dir, log)

    if a.log:
        a.log.parent.mkdir(parents=True, exist_ok=True)
        a.log.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[log written to {a.log}]")
    return 0


if __name__ == "__main__":
    return_code = main()
    raise SystemExit(return_code)
