#!/usr/bin/env python3
"""Collect legal_search training transitions (Arcade if available, else synthetic)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_search.algorithms.components import frame_changed  # noqa: E402
from legal_search.loop import LegalAgent  # noqa: E402
from legal_search.train.features import (  # noqa: E402
    FEATURE_DIM,
    FEATURE_LAYOUT,
    FEATURE_SCHEMA,
    encode_legal_features,
    feature_report,
)
from legal_search.train.model import game_bucket  # noqa: E402
from legal_search.types import Action, Transition  # noqa: E402


def annotate_level_weights(
    rows: list[dict],
    *,
    success_window: int = 12,
    thrash_change_rate: float = 0.85,
    thrash_weight: float = 0.12,
) -> list[dict]:
    """Post-process one episode: success-window boost + thrash downweight.

    Legal: only uses env-measured levels_up / did_change — no game answer keys.
    """
    if not rows:
        return rows
    for r in rows:
        r.setdefault("success_window", False)
        r.setdefault("thrash", False)
        r.setdefault("sample_weight", 1.0)
        r.setdefault("change_weight", 1.0)

    ups = [i for i, r in enumerate(rows) if r.get("levels_up")]
    for u in ups:
        for j in range(max(0, u - success_window), u + 1):
            rows[j]["success_window"] = True
            # later steps in the window (closer to level-up) get more weight
            dist = u - j
            boost = 12.0 if dist == 0 else max(3.0, 8.0 - 0.4 * dist)
            rows[j]["sample_weight"] = max(float(rows[j]["sample_weight"]), boost)
            # useful change: keep change loss in success windows
            rows[j]["change_weight"] = max(float(rows[j]["change_weight"]), 1.5)

    # thrash: no level-ups, mostly changing frames → late steps are noise
    if not ups and len(rows) >= 12:
        ch = sum(1 for r in rows if r.get("did_change")) / float(len(rows))
        if ch >= thrash_change_rate:
            cut = max(4, len(rows) // 3)
            for j in range(cut, len(rows)):
                rows[j]["thrash"] = True
                rows[j]["sample_weight"] = min(float(rows[j]["sample_weight"]), thrash_weight)
                # don't train "any change is good" on suicide thrash
                rows[j]["change_weight"] = min(float(rows[j]["change_weight"]), 0.25)
    return rows


def synthetic_episode(
    rng: np.random.Generator,
    max_steps: int = 40,
    *,
    teacher_frac: float = 0.67,
    level_shaped: bool = False,
) -> list[dict]:
    """Movable blob micro-env: legal labels only, no SP80 answers.

    level_shaped=True → more teacher toward level-up, then success-window weights.
    """
    grid = 64
    frame = np.full((grid, grid), 12, dtype=np.int16)
    frame[0, :] = 14
    # target region — level_shaped: closer starts so success windows are common
    if level_shaped:
        ty, tx = int(rng.integers(28, 48)), int(rng.integers(28, 48))
        my, mx = int(rng.integers(12, 28)), int(rng.integers(12, 28))
    else:
        ty, tx = int(rng.integers(40, 55)), int(rng.integers(40, 55))
        my, mx = int(rng.integers(8, 20)), int(rng.integers(8, 30))
    frame[ty : ty + 4, tx : tx + 4] = 11
    h, w = 3, 6
    frame[my : my + h, mx : mx + w] = 8
    selected = False
    rows = []
    agent = LegalAgent(game_id="synthetic_blob")
    agent.reset_for_game("synthetic_blob")
    levels = 0
    transitions: list[Transition] = []
    tf = 0.95 if level_shaped else float(teacher_frac)
    # allow longer episodes when hunting levels
    if level_shaped and max_steps < 96:
        max_steps = 96

    def paint_movable(f, y, x, col=8):
        f = f.copy()
        f[f == 8] = 12
        f[f == 9] = 12
        f[y : y + h, x : x + w] = col
        f[ty : ty + 4, tx : tx + 4] = 11
        f[0, :] = 14
        return f

    for t in range(max_steps):
        feats = encode_legal_features(
            frame,
            game_id="synthetic_blob",
            transitions=transitions[-16:],
            available_actions=[1, 2, 3, 4, 5, 6],
            levels_completed=levels,
            run_consume=True,  # full mine every step → LoRA
        )
        # mix: legal agent explore vs weak teacher toward target (not answer-key dump)
        if rng.random() > tf:
            action = agent.decide(frame, levels_completed=levels, available_actions=[1, 2, 3, 4, 5, 6])
        else:
            if not selected:
                action = Action(6, x=mx + w // 2, y=my + h // 2, source="synth_teacher")
            else:
                if my + h < ty:
                    action = Action(2, source="synth_teacher")  # down
                elif my > ty + 4:
                    action = Action(1, source="synth_teacher")
                elif mx + w < tx:
                    action = Action(4, source="synth_teacher")
                elif mx > tx + 4:
                    action = Action(3, source="synth_teacher")
                else:
                    action = Action(5, source="synth_teacher")  # "activate"

        before = frame.copy()
        ny, nx = my, mx
        if action.action_id == 6 and action.x is not None and action.y is not None:
            if mx <= action.x <= mx + w and my <= action.y <= my + h:
                selected = True
        elif selected and action.action_id in (1, 2, 3, 4):
            # level_shaped teacher: bigger strides when far from target
            stride = 2 if level_shaped else 1
            if action.action_id == 1:
                ny = max(1, my - stride)
            elif action.action_id == 2:
                ny = min(grid - h - 1, my + stride)
            elif action.action_id == 3:
                nx = max(0, mx - stride)
            elif action.action_id == 4:
                nx = min(grid - w - 1, mx + stride)
            my, mx = ny, nx
        elif action.action_id == 5 and selected:
            # level-up if overlapping target
            if not (mx + w < tx or mx > tx + 4 or my + h < ty or my > ty + 4):
                levels = 1

        frame = paint_movable(frame, my, mx, col=9 if selected else 8)
        changed = frame_changed(before, frame)
        levels_before = int(rows[-1]["levels_after"]) if rows else 0
        tr = Transition(
            game_id="synthetic_blob",
            step=t,
            frame_before=before,
            action=action,
            frame_after=frame,
            levels_before=levels_before,
            levels_after=int(levels),
            frame_changed=changed,
            levels_up=int(levels) > levels_before,
        )
        agent.observe_transition(tr)
        transitions.append(tr)
        rows.append(
            {
                "frame": before.astype(np.int16),
                "features": feats.astype(np.float32),
                "action_id": int(action.action_id),
                "x": int(action.x) if action.x is not None else 0,
                "y": int(action.y) if action.y is not None else 0,
                "did_change": bool(changed),
                "levels_up": bool(tr.levels_up),
                "levels_after": int(levels),
                "game_id": "synthetic_blob",
                "source": action.source,
            }
        )
        if levels >= 1:
            break
    return annotate_level_weights(rows)


def _discover_game_ids(env_dir: Path) -> list[str]:
    """Offline game ids like sp80-589a99af, ft09-0d8bbf25 if present."""
    out: list[str] = []
    for family in sorted(p for p in env_dir.iterdir() if p.is_dir()):
        for gid in sorted(p for p in family.iterdir() if p.is_dir()):
            # need a .py game module inside
            if any(gid.glob("*.py")):
                out.append(f"{family.name}-{gid.name}")
    # prefer known stables first
    preferred = [g for g in out if g.startswith("sp80-") or g.startswith("ft09-")]
    rest = [g for g in out if g not in preferred]
    return preferred + rest


def try_arcade_collect(n_episodes: int, max_actions: int, seed: int) -> list[dict] | None:
    try:
        from arc_agi import Arcade
        from arc_agi.base import OperationMode
        from arcengine import GameAction
    except Exception as exc:
        print(f"[collect] arcade unavailable: {exc}")
        return None

    env_dir = ROOT / "environment_files"
    if not env_dir.exists():
        return None
    game_ids = _discover_game_ids(env_dir)
    if not game_ids:
        game_ids = ["sp80-589a99af"]
    print(f"[collect] arcade games available={game_ids[:8]}{'...' if len(game_ids)>8 else ''}")
    act_map = {
        1: GameAction.ACTION1,
        2: GameAction.ACTION2,
        3: GameAction.ACTION3,
        4: GameAction.ACTION4,
        5: GameAction.ACTION5,
        6: GameAction.ACTION6,
        0: GameAction.RESET,
    }
    rows: list[dict] = []
    rng = np.random.default_rng(seed)
    for ep in range(n_episodes):
        game_id = game_ids[int(ep) % len(game_ids)]
        try:
            arc = Arcade(arc_api_key="anonymous", environments_dir=str(env_dir), operation_mode=OperationMode.OFFLINE)
            card = arc.open_scorecard()
            env = arc.make(game_id, scorecard_id=card)
            fr = env.reset()
        except Exception as exc:
            print(f"[collect] arcade episode failed game={game_id}: {exc}")
            # try next game once
            if len(game_ids) > 1:
                game_id = game_ids[(int(ep) + 1) % len(game_ids)]
                try:
                    arc = Arcade(
                        arc_api_key="anonymous",
                        environments_dir=str(env_dir),
                        operation_mode=OperationMode.OFFLINE,
                    )
                    card = arc.open_scorecard()
                    env = arc.make(game_id, scorecard_id=card)
                    fr = env.reset()
                except Exception as exc2:
                    print(f"[collect] arcade fallback failed: {exc2}")
                    if rows:
                        continue
                    return rows or None
            else:
                return rows or None
        agent = LegalAgent(game_id=game_id)
        agent.reset_for_game(game_id)
        transitions: list[Transition] = []
        ep_rows: list[dict] = []
        grid = np.array(fr.frame)
        if grid.ndim == 3:
            grid = grid[0]
        grid = np.asarray(grid, dtype=np.int16)
        levels = int(getattr(fr, "levels_completed", 0) or 0)
        for t in range(max_actions):
            feats = encode_legal_features(
                grid,
                game_id=game_id,
                transitions=transitions[-12:],
                available_actions=list(getattr(fr, "available_actions", []) or [1, 2, 3, 4, 5, 6]),
                levels_completed=levels,
                run_consume=True,  # full mine every step → LoRA
            )
            avail = []
            for v in getattr(fr, "available_actions", []) or [1, 2, 3, 4, 5, 6]:
                try:
                    avail.append(int(getattr(v, "value", v)))
                except Exception:
                    pass
            if not avail:
                avail = [1, 2, 3, 4, 5, 6]
            # epsilon mix random
            if rng.random() < 0.35:
                aid = int(rng.choice(avail))
                if aid == 6:
                    action = Action(6, x=int(rng.integers(0, 64)), y=int(rng.integers(0, 64)), source="epsilon")
                else:
                    action = Action(aid, source="epsilon")
            else:
                action = agent.decide(grid, levels_completed=levels, available_actions=avail)

            before = grid.copy()
            ga = act_map.get(action.action_id, GameAction.ACTION1)
            data = None
            if action.action_id == 6:
                data = {"x": int(action.x or 0), "y": int(action.y or 0)}
            try:
                fr = env.step(ga, data=data)
            except Exception:
                break
            grid2 = np.array(fr.frame)
            if grid2.ndim == 3:
                grid2 = grid2[0]
            grid2 = np.asarray(grid2, dtype=np.int16)
            levels2 = int(getattr(fr, "levels_completed", 0) or 0)
            changed = frame_changed(before, grid2)
            tr = Transition(
                game_id=game_id,
                step=t,
                frame_before=before,
                action=action,
                frame_after=grid2,
                levels_before=levels,
                levels_after=levels2,
                state_after=str(getattr(fr, "state", "")),
                frame_changed=changed,
                levels_up=levels2 > levels,
            )
            agent.observe_transition(tr)
            transitions.append(tr)
            ep_rows.append(
                {
                    "frame": before.astype(np.int16),
                    "features": feats.astype(np.float32),
                    "action_id": int(action.action_id),
                    "x": int(action.x) if action.x is not None else 0,
                    "y": int(action.y) if action.y is not None else 0,
                    "did_change": bool(changed),
                    "levels_up": bool(tr.levels_up),
                    "levels_after": int(levels2),
                    "game_id": game_id,
                    "source": action.source,
                }
            )
            grid, levels = grid2, levels2
            st = str(getattr(fr, "state", ""))
            if "WIN" in st:
                break
            if "GAME_OVER" in st:
                # end episode — thrash annotate will downweight if no level_up
                break
        annotate_level_weights(ep_rows)
        n_up = sum(1 for r in ep_rows if r.get("levels_up"))
        rows.extend(ep_rows)
        try:
            arc.close_scorecard(card)
        except Exception:
            pass
        print(
            f"[collect] arcade ep {ep+1}/{n_episodes} game={game_id} "
            f"ep_rows={len(ep_rows)} levels_up={n_up} total={len(rows)}",
            flush=True,
        )
    return rows


def rows_to_tensors(rows: list[dict]) -> dict[str, torch.Tensor]:
    frames = torch.stack([torch.from_numpy(np.asarray(r["frame"], dtype=np.int16)).long() for r in rows])
    features = torch.stack([torch.from_numpy(np.asarray(r["features"], dtype=np.float32)) for r in rows])
    action_id = torch.tensor([int(r["action_id"]) for r in rows], dtype=torch.long)
    x = torch.tensor([int(r["x"]) for r in rows], dtype=torch.long)
    y = torch.tensor([int(r["y"]) for r in rows], dtype=torch.long)
    did_change = torch.tensor([1.0 if r["did_change"] else 0.0 for r in rows], dtype=torch.float32)
    levels_up = torch.tensor([1.0 if r["levels_up"] else 0.0 for r in rows], dtype=torch.float32)
    sample_weight = torch.tensor(
        [float(r.get("sample_weight", 1.0)) for r in rows], dtype=torch.float32
    )
    change_weight = torch.tensor(
        [float(r.get("change_weight", 1.0)) for r in rows], dtype=torch.float32
    )
    success_window = torch.tensor(
        [1.0 if r.get("success_window") else 0.0 for r in rows], dtype=torch.float32
    )
    game_idx = torch.tensor(
        [game_bucket(str(r.get("game_id", ""))) for r in rows], dtype=torch.long
    )
    return {
        "frame": frames,
        "features": features,
        "action_id": action_id,
        "x": x,
        "y": y,
        "did_change": did_change,
        "levels_up": levels_up,
        "sample_weight": sample_weight,
        "change_weight": change_weight,
        "success_window": success_window,
        "game_idx": game_idx,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "legal_search" / "runs" / "blend_lora_gated_v1" / "data.pt")
    p.add_argument("--episodes", type=int, default=24)
    p.add_argument("--max-actions", type=int, default=48)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--synthetic-only", action="store_true")
    p.add_argument("--arcade-episodes", type=int, default=4)
    p.add_argument(
        "--allow-ungated-scout",
        action="store_true",
        help="DANGER: collect for scout only; forces path under runs/_ungated_scout/",
    )
    p.add_argument(
        "--mine-probe-only",
        action="store_true",
        help="Allow collect during mining (JOB 1 probes). Does not unlock training.",
    )
    p.add_argument(
        "--level-shaped",
        action="store_true",
        help="Path B: teacher-heavy synth + success-window weights + thrash downweight",
    )
    args = p.parse_args()

    from legal_search.train.mining_gate import MiningIncompleteError, assert_mining_complete

    # Training-oriented collect is gated. Mine probes may pass --mine-probe-only.
    if not args.mine_probe_only:
        try:
            gate = assert_mining_complete(allow_ungated_scout=bool(args.allow_ungated_scout))
        except MiningIncompleteError as exc:
            print(str(exc), file=sys.stderr)
            print(
                "Hint: use --mine-probe-only for mining-job data probes "
                "(does not unlock train). See legal_search/MINING_THEN_TRAIN.md",
                file=sys.stderr,
            )
            return 2
        if gate.get("scout"):
            # force scout tree
            name = args.out.name
            args.out = ROOT / "legal_search" / "runs" / "_ungated_scout" / "data" / name
            print(f"[collect] scout out forced → {args.out}")
    else:
        print("[collect] mine-probe-only: gate not required (not training data)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    if not args.synthetic_only:
        arcade_rows = try_arcade_collect(args.arcade_episodes, args.max_actions, args.seed)
        if arcade_rows:
            rows.extend(arcade_rows)
            print(f"[collect] arcade rows={len(arcade_rows)}")
    for ep in range(args.episodes):
        rows.extend(
            synthetic_episode(
                np.random.default_rng(args.seed + ep + 1),
                max_steps=args.max_actions,
                level_shaped=bool(args.level_shaped),
            )
        )
        if (ep + 1) % 4 == 0:
            n_up = sum(1 for r in rows if r.get("levels_up"))
            n_sw = sum(1 for r in rows if r.get("success_window"))
            print(
                f"[collect] synthetic {ep+1}/{args.episodes} total_rows={len(rows)} "
                f"levels_up={n_up} success_window={n_sw}",
                flush=True,
            )
    if not rows:
        raise SystemExit("no rows collected")
    # shuffle split
    idx = rng.permutation(len(rows))
    rows = [rows[i] for i in idx]
    n_val = max(32, len(rows) // 10)
    train_rows, val_rows = rows[n_val:], rows[:n_val]
    # mining coverage report on first train row
    mine_cov = feature_report(train_rows[0]["features"]) if train_rows else {}
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
            "episodes_synthetic": args.episodes,
            "arcade_episodes": 0 if args.synthetic_only else args.arcade_episodes,
            "full_mine_every_step": True,
            "level_shaped": bool(args.level_shaped),
            "n_levels_up": sum(1 for r in rows if r.get("levels_up")),
            "n_success_window": sum(1 for r in rows if r.get("success_window")),
            "n_thrash": sum(1 for r in rows if r.get("thrash")),
            "n_systems": FEATURE_LAYOUT["n_systems"],
            "mining_coverage_sample": mine_cov,
            "banned_lineage": [
                "sp80_spill_model",
                "dropletsim",
                "human_mined_routes",
                "fixture_win_banks_as_labels",
            ],
            "legal": True,
        },
    }
    torch.save(payload, args.out)
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "out": str(args.out),
                "n_train": payload["n_train"],
                "n_val": payload["n_val"],
                "feature_dim": FEATURE_DIM,
                "legal": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[collect] wrote {args.out} train={payload['n_train']} val={payload['n_val']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
