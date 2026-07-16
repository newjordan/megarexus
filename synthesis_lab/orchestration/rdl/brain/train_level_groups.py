#!/usr/bin/env python3
"""Train MegaMind on level groups across ALL suite games at once.

Doctrine
--------
- Levels trained **in groups** (e.g. L0+L1 together), not one game forever.
- Every cohort hits **all 24 games** interleaved → general solving, robust patterns.
- Multi-variate pattern growth pad stays live; geometry is game-agnostic.
- No SP80 focus. No on_set_level injection. Env grades only.

Example
-------
  # Full L0–L1 cohort on all 24 games (short smoke)
  python -B synthesis_lab/orchestration/rdl/brain/train_level_groups.py \\
      --groups 0-1 --steps-per-game 8 --epochs 2 --device cpu

  # Deeper raise: L0-1 then L2-3, more steps
  python -B synthesis_lab/orchestration/rdl/brain/train_level_groups.py \\
      --groups 0-1,2-3 --steps-per-game 24 --epochs 5 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_groups(s: str) -> list[tuple[int, int]]:
    """'0-1,2-3' → [(0,1),(2,3)]"""
    out: list[tuple[int, int]] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            lo = hi = int(part)
        else:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
        if hi < lo:
            lo, hi = hi, lo
        out.append((lo, hi))
    return out or [(0, 1)]


def _batch_iter(n: int, batch_size: int):
    for i in range(0, n, batch_size):
        yield i, min(i + batch_size, n)


def _mean_phase_loss(logs: list[dict[str, Any]]) -> float:
    losses = []
    for row in logs:
        if not isinstance(row, dict):
            continue
        for k, v in row.items():
            if "loss" in str(k).lower() and isinstance(v, (int, float)):
                losses.append(float(v))
        if "mean_loss" in row and isinstance(row["mean_loss"], (int, float)):
            losses.append(float(row["mean_loss"]))
    return float(sum(losses) / len(losses)) if losses else 0.0


def train_cohort(
    mm,
    cohort: dict[str, Any],
    *,
    epochs: int,
    batch_size: int,
    steps_per_phase: int,
) -> dict[str, Any]:
    """Train one shared MegaMind on an interleaved multi-game level-group batch."""
    frames = cohort["frames"]
    actions = cohort["actions"]
    frames_next = cohort["frames_next"]
    levels = cohort["levels"]
    game_ids_t = cohort["game_ids_tensor"]
    game_ids = list(cohort["game_ids"])
    n = int(frames.shape[0])
    if n == 0:
        return {"status": "empty", "epochs": 0, "batches": 0}

    epoch_rows: list[dict[str, Any]] = []
    total_batches = 0
    # Track per-game how often it appears in train steps (robustness audit)
    game_hits = {g: 0 for g in game_ids}

    for ep in range(epochs):
        # shuffle indices but keep multi-game mixture via random perm of interleaved stream
        perm = torch.randperm(n)
        ep_loss = 0.0
        ep_batches = 0
        for a, b in _batch_iter(n, batch_size):
            idx = perm[a:b]
            fb = frames[idx]
            ab = actions[idx]
            fn = frames_next[idx]
            # episode memory: remember each transition for Ghost/Delta seeds
            for j in range(fb.shape[0]):
                gi = int(game_ids_t[idx[j]].item())
                gid = game_ids[gi] if 0 <= gi < len(game_ids) else "unknown"
                game_hits[gid] = game_hits.get(gid, 0) + 1
                mm.reset_episode(game_id=gid)
                mm.I.remember_transition(
                    fb[j].numpy(),
                    int(ab[j].item()),
                    fn[j].numpy(),
                    levels_before=int(levels[idx[j]].item()),
                )
            # shared weights see mixed games in one train_transition call
            logs = mm.train_transition(
                fb,
                ab,
                fn,
                game_id=f"{cohort['level_group']}:mixed:{ep}:{a}",
                steps_per_phase=steps_per_phase,
            )
            loss = _mean_phase_loss(logs if isinstance(logs, list) else [logs])
            ep_loss += loss
            ep_batches += 1
            total_batches += 1
        epoch_rows.append(
            {
                "epoch": ep,
                "mean_loss": ep_loss / max(1, ep_batches),
                "batches": ep_batches,
                "n": n,
            }
        )

    n_games_hit = sum(1 for g, c in game_hits.items() if c > 0)
    return {
        "status": "ok",
        "epochs": epochs,
        "batches": total_batches,
        "n_transitions": n,
        "epoch_curve": epoch_rows,
        "final_mean_loss": epoch_rows[-1]["mean_loss"] if epoch_rows else 0.0,
        "games_hit": n_games_hit,
        "n_games": len(game_ids),
        "game_hits": game_hits,
        "robust_coverage": float(n_games_hit) / float(len(game_ids) or 1),
        "level_group": cohort.get("level_group"),
    }


def evaluate_pattern_robustness(mm, cohort: dict[str, Any], *, max_games: int = 24) -> dict[str, Any]:
    """Observe one frame per game; check pattern-growth pad fires multi-game."""
    game_ids = list(cohort["game_ids"])[:max_games]
    rows = cohort.get("rows") or []
    # first row per game
    first: dict[str, Any] = {}
    for r in rows:
        gid = str(r.get("game_id"))
        if gid not in first:
            first[gid] = r
    per: list[dict[str, Any]] = []
    g_ok = 0
    for gid in game_ids:
        r = first.get(gid)
        if r is None:
            per.append({"game_id": gid, "status": "missing"})
            continue
        mm.reset_episode(game_id=gid)
        st = mm.observe(r["frame"].numpy())
        ds = st.dream_summary or {}
        # count G* channels ok
        g_ids = [k for k in ds if str(k).startswith("G") and str(k)[1:2].isdigit()]
        ok = sum(1 for k in g_ids if (ds.get(k) or {}).get("status") == "ok")
        multi = float(((ds.get("G1_multichannel_variables") or {}).get("scores") or {}).get("is_multivariate") or 0)
        not_sp80 = float(((ds.get("G38_not_sp80_bound") or {}).get("scores") or {}).get("not_sp80_bound") or 0)
        if ok >= 30:
            g_ok += 1
        per.append(
            {
                "game_id": gid,
                "status": "ok",
                "pattern_growth_ok": ok,
                "multivariate": multi,
                "not_sp80_bound": not_sp80,
                "extra_dim": None if st.extra_feats is None else int(st.extra_feats.shape[-1]),
            }
        )
    return {
        "n_games": len(game_ids),
        "n_games_pattern_ok": g_ok,
        "pattern_robust_frac": float(g_ok) / float(len(game_ids) or 1),
        "per_game": per,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--groups",
        default="0-1",
        help="Comma level groups e.g. 0-1,2-3 (default: 0-1 on all games)",
    )
    ap.add_argument("--steps-per-game", type=int, default=12)
    ap.add_argument("--max-actions", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--steps-per-phase", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--env-dir",
        default="",
        help="Override suite env dir (default: other_games/environment_files)",
    )
    ap.add_argument("--max-games", type=int, default=0, help="0 = all suite games")
    ap.add_argument(
        "--out-dir",
        default="",
        help="Write metrics JSON (default under synthesis_lab/experiments/...)",
    )
    ap.add_argument("--smoke", action="store_true", help="Tiny run: 4 games, 4 steps, 1 epoch")
    args = ap.parse_args(argv)

    from synthesis_lab.orchestration.rdl.brain.collect_level_group import (
        collect_level_group_cohort,
    )
    from synthesis_lab.orchestration.rdl.brain.megamind import MegaMind
    from synthesis_lab.orchestration.rdl.brain.suite_games import (
        DEFAULT_SUITE_ENV_DIR,
        describe_suite,
        suite_24,
    )

    env_dir = Path(args.env_dir) if args.env_dir else DEFAULT_SUITE_ENV_DIR
    games = suite_24(env_dir)
    if args.smoke:
        games = games[:4]
        args.steps_per_game = min(args.steps_per_game, 4)
        args.epochs = 1
        args.max_actions = 20
    elif args.max_games and args.max_games > 0:
        games = games[: args.max_games]

    groups = _parse_groups(args.groups)
    suite_meta = describe_suite(games)
    print(
        json.dumps(
            {
                "event": "suite",
                **{k: suite_meta[k] for k in ("n_games", "prefixes", "doctrine")},
                "groups": [f"L{a}-L{b}" for a, b in groups],
                "env_dir": str(env_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARN: cuda requested but unavailable; falling back to cpu", flush=True)
        device = "cpu"

    mm = MegaMind(device=device)
    t0 = time.time()
    group_results: list[dict[str, Any]] = []

    for lo, hi in groups:
        print(
            f"=== COHORT L{lo}-L{hi} on {len(games)} games "
            f"(steps/game={args.steps_per_game}) ===",
            flush=True,
        )
        cohort = collect_level_group_cohort(
            level_lo=lo,
            level_hi=hi,
            game_ids=games,
            env_dir=env_dir,
            seed=args.seed + lo * 100,
            max_actions=args.max_actions,
            max_in_band_per_game=args.steps_per_game,
        )
        print(
            json.dumps(
                {
                    "event": "cohort_collected",
                    "level_group": cohort["level_group"],
                    "n_games": cohort["n_games"],
                    "n_transitions": cohort["n_transitions"],
                    "n_games_ok": cohort["n_games_ok"],
                    "n_games_degraded": cohort["n_games_degraded"],
                    "interleaved": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        train_stats = train_cohort(
            mm,
            cohort,
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_phase=args.steps_per_phase,
        )
        robust = evaluate_pattern_robustness(mm, cohort, max_games=len(games))
        group_results.append(
            {
                "level_group": cohort["level_group"],
                "level_lo": lo,
                "level_hi": hi,
                "collect": {
                    "n_transitions": cohort["n_transitions"],
                    "n_games_ok": cohort["n_games_ok"],
                    "n_games_degraded": cohort["n_games_degraded"],
                    "per_game": cohort["per_game"],
                },
                "train": train_stats,
                "pattern_robustness": {
                    "n_games_pattern_ok": robust["n_games_pattern_ok"],
                    "pattern_robust_frac": robust["pattern_robust_frac"],
                },
            }
        )
        print(
            json.dumps(
                {
                    "event": "cohort_trained",
                    "level_group": cohort["level_group"],
                    "final_mean_loss": train_stats.get("final_mean_loss"),
                    "robust_coverage": train_stats.get("robust_coverage"),
                    "pattern_robust_frac": robust["pattern_robust_frac"],
                    "games_hit": train_stats.get("games_hit"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    elapsed = time.time() - t0
    # checkpoint
    out_dir = Path(args.out_dir) if args.out_dir else (
        _ROOT
        / "synthesis_lab"
        / "experiments"
        / "megamind_level_groups"
        / f"run_{_utc()}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "megamind_level_groups.pt"
    torch.save(
        {
            "central": mm.central.state_dict(),
            "net_I": mm._net_I.state_dict(),
            "net_E": mm._net_E.state_dict(),
            "schema": "megamind.level_group_train.v1",
            "groups": [f"L{a}-L{b}" for a, b in groups],
            "n_games": len(games),
            "game_ids": games,
        },
        ckpt_path,
    )
    metrics = {
        "schema": "megamind.level_group_train_metrics.v1",
        "created_utc": _utc(),
        "elapsed_s": elapsed,
        "device": device,
        "suite": suite_meta,
        "groups": [f"L{a}-L{b}" for a, b in groups],
        "epochs": args.epochs,
        "steps_per_game": args.steps_per_game,
        "batch_size": args.batch_size,
        "results": group_results,
        "checkpoint": str(ckpt_path.relative_to(_ROOT)) if ckpt_path.is_relative_to(_ROOT) else str(ckpt_path),
        "doctrine": {
            "level_groups": True,
            "all_games_at_once": True,
            "interleaved": True,
            "robust_patterns": True,
            "not_sp80_focus": True,
            "no_level_injection": True,
        },
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "event": "done",
                "elapsed_s": round(elapsed, 2),
                "out_dir": str(out_dir),
                "checkpoint": str(ckpt_path),
                "n_groups": len(group_results),
                "mean_pattern_robust": float(
                    sum(g["pattern_robustness"]["pattern_robust_frac"] for g in group_results)
                    / max(1, len(group_results))
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    # fail closed if no game was hit in train
    if not group_results:
        return 1
    if any(g["train"].get("games_hit", 0) < max(1, len(games) // 2) for g in group_results):
        print("WARN: low multi-game train coverage", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
