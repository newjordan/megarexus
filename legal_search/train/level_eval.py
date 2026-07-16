#!/usr/bin/env python3
"""Path B eval: levels_completed / synthetic level-up only (not change_acc)."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_search.algorithms.components import frame_changed  # noqa: E402
from legal_search.train.policy import BlendLoRAPolicy  # noqa: E402
from legal_search.types import Action, Transition  # noqa: E402


def _synth_level_probe(
    pol: BlendLoRAPolicy,
    *,
    n_eps: int = 20,
    max_steps: int = 80,
    seed: int = 0,
) -> dict:
    """Same micro-env as collect synthetic — env-judged levels only."""
    rng = np.random.default_rng(seed)
    wins = 0
    steps_list = []
    for ep in range(n_eps):
        grid_n = 64
        frame = np.full((grid_n, grid_n), 12, dtype=np.int16)
        frame[0, :] = 14
        ty, tx = int(rng.integers(40, 55)), int(rng.integers(40, 55))
        frame[ty : ty + 4, tx : tx + 4] = 11
        my, mx = int(rng.integers(8, 20)), int(rng.integers(8, 30))
        h, w = 3, 6
        frame[my : my + h, mx : mx + w] = 8
        selected = False
        levels = 0
        transitions: list[Transition] = []

        def paint(f, y, x, col=8):
            f = f.copy()
            f[f == 8] = 12
            f[f == 9] = 12
            f[y : y + h, x : x + w] = col
            f[ty : ty + 4, tx : tx + 4] = 11
            f[0, :] = 14
            return f

        won = False
        for t in range(max_steps):
            action = pol.decide(
                frame,
                game_id="synthetic_blob",
                transitions=transitions,
                available_actions=[1, 2, 3, 4, 5, 6],
                levels_completed=levels,
            )
            before = frame.copy()
            ny, nx = my, mx
            if action.action_id == 6 and action.x is not None and action.y is not None:
                if mx <= action.x <= mx + w and my <= action.y <= my + h:
                    selected = True
            elif selected and action.action_id in (1, 2, 3, 4):
                if action.action_id == 1:
                    ny = max(1, my - 1)
                elif action.action_id == 2:
                    ny = min(grid_n - h - 1, my + 1)
                elif action.action_id == 3:
                    nx = max(0, mx - 1)
                elif action.action_id == 4:
                    nx = min(grid_n - w - 1, mx + 1)
                my, mx = ny, nx
            elif action.action_id == 5 and selected:
                if not (mx + w < tx or mx > tx + 4 or my + h < ty or my > ty + 4):
                    levels = 1
            frame = paint(frame, my, mx, col=9 if selected else 8)
            ch = frame_changed(before, frame)
            tr = Transition(
                game_id="synthetic_blob",
                step=t,
                frame_before=before,
                action=action,
                frame_after=frame,
                levels_before=0 if t == 0 else int(levels == 1 and not won),
                levels_after=int(levels),
                frame_changed=ch,
                levels_up=levels == 1 and not won,
            )
            if levels == 1 and not won:
                won = True
                wins += 1
                steps_list.append(t + 1)
                break
            transitions.append(tr)
        if not won:
            steps_list.append(max_steps)
    return {
        "env": "synthetic_blob",
        "n_eps": n_eps,
        "n_level_up": wins,
        "level_up_rate": wins / max(1, n_eps),
        "mean_steps": float(np.mean(steps_list)) if steps_list else 0.0,
    }


def _arcade_probe(
    pol: BlendLoRAPolicy,
    *,
    games: list[str],
    n_eps: int = 2,
    max_steps: int = 150,
) -> list[dict]:
    try:
        from arc_agi import Arcade
        from arc_agi.base import OperationMode
        from arcengine import GameAction
    except Exception as exc:
        return [{"error": f"arcade unavailable: {exc}"}]

    act_map = {
        1: GameAction.ACTION1,
        2: GameAction.ACTION2,
        3: GameAction.ACTION3,
        4: GameAction.ACTION4,
        5: GameAction.ACTION5,
        6: GameAction.ACTION6,
    }
    env_dir = ROOT / "environment_files"
    out = []
    for game_id in games:
        for ep in range(n_eps):
            arc = Arcade(
                arc_api_key="anonymous",
                environments_dir=str(env_dir),
                operation_mode=OperationMode.OFFLINE,
            )
            card = arc.open_scorecard()
            try:
                env = arc.make(game_id, scorecard_id=card)
                fr = env.reset()
            except Exception as e:
                out.append({"game_id": game_id, "ep": ep, "error": str(e)})
                continue
            grid = np.array(fr.frame)
            if grid.ndim == 3:
                grid = grid[0]
            grid = np.asarray(grid, dtype=np.int16)
            levels0 = int(getattr(fr, "levels_completed", 0) or 0)
            levels = levels0
            transitions: list[Transition] = []
            max_levels = levels
            n_change = 0
            action_hist: Counter[int] = Counter()
            source_hist: Counter[str] = Counter()
            last_state = str(getattr(fr, "state", ""))
            t = 0
            for t in range(max_steps):
                avail = []
                for v in getattr(fr, "available_actions", []) or [1, 2, 3, 4, 5, 6]:
                    try:
                        avail.append(int(getattr(v, "value", v)))
                    except Exception:
                        pass
                if not avail:
                    avail = [1, 2, 3, 4, 5, 6]
                action = pol.decide(
                    grid,
                    game_id=game_id,
                    transitions=transitions,
                    available_actions=avail,
                    levels_completed=levels,
                )
                action_hist[int(action.action_id)] += 1
                source_hist[str(action.source)] += 1
                before = grid.copy()
                ga = act_map.get(action.action_id, GameAction.ACTION1)
                data = (
                    {"x": int(action.x or 0), "y": int(action.y or 0)}
                    if action.action_id == 6
                    else None
                )
                try:
                    fr = env.step(ga, data=data)
                except Exception as e:
                    last_state = f"step_err:{e}"
                    break
                grid2 = np.array(fr.frame)
                if grid2.ndim == 3:
                    grid2 = grid2[0]
                grid2 = np.asarray(grid2, dtype=np.int16)
                levels2 = int(getattr(fr, "levels_completed", 0) or 0)
                ch = frame_changed(before, grid2)
                if ch:
                    n_change += 1
                tr = Transition(
                    game_id=game_id,
                    step=t,
                    frame_before=before,
                    action=action,
                    frame_after=grid2,
                    levels_before=levels,
                    levels_after=levels2,
                    state_after=str(getattr(fr, "state", "")),
                    frame_changed=ch,
                    levels_up=levels2 > levels,
                )
                transitions.append(tr)
                grid, levels = grid2, levels2
                max_levels = max(max_levels, levels)
                last_state = str(getattr(fr, "state", ""))
                if levels > levels0:
                    break
                if "GAME_OVER" in last_state.upper():
                    break
            try:
                arc.close_scorecard(card)
            except Exception:
                pass
            out.append(
                {
                    "game_id": game_id,
                    "ep": ep,
                    "steps": t + 1,
                    "levels0": levels0,
                    "levels_final": levels,
                    "max_levels": max_levels,
                    "level_up": max_levels > levels0,
                    "n_change": n_change,
                    "change_rate": n_change / max(1, t + 1),
                    "state": last_state[:80],
                    "action_hist": {str(k): int(v) for k, v in sorted(action_hist.items())},
                    "source_hist": dict(sorted(source_hist.items())),
                }
            )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpt",
        type=Path,
        default=ROOT / "legal_search" / "runs" / "blend_lora_level_v1" / "best.pt",
    )
    ap.add_argument("--synth-eps", type=int, default=30)
    ap.add_argument("--arcade-eps", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--success-programs", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if not args.ckpt.is_file():
        print(f"missing ckpt {args.ckpt}", file=sys.stderr)
        return 2

    print(f"[level_eval] ckpt={args.ckpt} device={args.device}")
    pol = BlendLoRAPolicy(args.ckpt, device=args.device, success_programs=args.success_programs)
    synth = _synth_level_probe(pol, n_eps=args.synth_eps, seed=7)
    print(f"[level_eval] synthetic: {json.dumps(synth)}")
    arcade = _arcade_probe(
        pol,
        games=["sp80-589a99af", "ft09-0d8bbf25"],
        n_eps=args.arcade_eps,
    )
    for row in arcade:
        print(f"[level_eval] arcade: {json.dumps(row)}")
    any_arcade_up = any(r.get("level_up") for r in arcade if isinstance(r, dict))
    summary = {
        "ckpt": str(args.ckpt),
        "synthetic": synth,
        "arcade": arcade,
        "any_synth_level": synth["n_level_up"] > 0,
        "any_arcade_level": any_arcade_up,
        "floor_bar": synth["n_level_up"] > 0 or any_arcade_up,
    }
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    out = args.out or (args.ckpt.parent / "level_eval.json")
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    # exit 0 if synthetic level rate > 0 (micro-env floor) OR arcade level
    return 0 if summary["floor_bar"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
