#!/usr/bin/env python3
"""MegaMind training flywheel — keep spinning.

Bi-level (Weco AIDE²-aligned):
  OUTER: schedule level groups × all games; keep checkpoint only if private better
  INNER: collect L-band transitions → train_transition → pattern robustness

Keeps running until --max-cycles (0 = forever) or STOP file present.

  python -B synthesis_lab/orchestration/rdl/brain/train_flywheel.py \\
      --device cuda --root synthesis_lab/experiments/megamind_flywheel

Stop:  touch $ROOT/STOP
Status: $ROOT/HEARTBEAT.json  $ROOT/journal.jsonl  $ROOT/best/
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Cycle schedule: (groups_str, steps_per_game, epochs, label)
# Doctrine: WIDE · SLOW · STEADY — all 24 games, progressive L0–L1 depth first,
# gentle joint bands; high levels only as light scouts (no thrash).
DEFAULT_SCHEDULE: list[tuple[str, int, int, str]] = [
    ("0-1", 12, 2, "L0L1_core"),
    ("0-1", 14, 2, "L0L1_steady"),
    ("0-1", 16, 3, "L0L1_deeper"),
    ("0-1,2-3", 12, 2, "L0L3_joint"),
    ("0-1", 18, 3, "L0L1_heavy"),
    ("0-1,2-3,4-5", 10, 2, "broad_low"),
    ("0-1", 20, 3, "L0L1_wider"),
    ("2-3", 10, 2, "L2L3_scout"),
    ("0-1", 16, 2, "L0L1_return"),
    ("0-1,2-3", 12, 2, "L0L3_joint2"),
    ("4-5", 8, 2, "L4L5_scout"),
    ("0-1", 22, 3, "L0L1_deepest"),
]


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _append_journal(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _load_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_mm(mm, path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "central": mm.central.state_dict(),
            "net_I": mm._net_I.state_dict(),
            "net_E": mm._net_E.state_dict(),
            "schema": "megamind.flywheel_ckpt.v1",
            "meta": meta,
            "saved_utc": _utc(),
        },
        path,
    )


def _compatible_load(module: torch.nn.Module, state: dict, name: str) -> int:
    """Load only tensors whose names and shapes match (pad growth safe)."""
    if not state:
        return 0
    cur = module.state_dict()
    filtered = {}
    skipped = 0
    for k, v in state.items():
        if k not in cur:
            skipped += 1
            continue
        if tuple(cur[k].shape) != tuple(v.shape):
            skipped += 1
            continue
        filtered[k] = v
    if filtered:
        module.load_state_dict(filtered, strict=False)
    print(
        json.dumps(
            {
                "event": "warm_load",
                "module": name,
                "loaded": len(filtered),
                "skipped": skipped,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return len(filtered)


def _load_mm(mm, path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        blob = torch.load(path, map_location=mm.device, weights_only=False)
        n = 0
        if "central" in blob:
            n += _compatible_load(mm.central, blob["central"], "central")
        if "net_I" in blob:
            n += _compatible_load(mm._net_I, blob["net_I"], "net_I")
        if "net_E" in blob:
            n += _compatible_load(mm._net_E, blob["net_E"], "net_E")
        return n > 0
    except Exception as exc:
        print(f"WARN: load_ckpt failed {path}: {exc}", flush=True)
        return False


def _enriched_private(metrics: dict[str, Any]) -> dict[str, float]:
    """Private score v2 axes (continuous efficiency + multi-game coverage)."""
    from synthesis_lab.orchestration.rdl.brain.rsi_outer_gate import private_axes

    return private_axes(metrics)


def keep_flywheel(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    *,
    min_private: float = 0.35,
) -> dict[str, Any]:
    """Keep if composite private v2 improves (AIDE² keep-only-if-better)."""
    from synthesis_lab.orchestration.rdl.brain.rsi_outer_gate import keep_only_if_better

    return keep_only_if_better(candidate, incumbent, min_private=min_private)


def run_one_cycle(
    *,
    cycle: int,
    groups: str,
    steps_per_game: int,
    epochs: int,
    label: str,
    device: str,
    seed: int,
    batch_size: int,
    max_actions: int,
    work_dir: Path,
    best_ckpt: Path,
    best_metrics_path: Path,
) -> dict[str, Any]:
    from synthesis_lab.orchestration.rdl.brain.collect_level_group import (
        collect_level_group_cohort,
    )
    from synthesis_lab.orchestration.rdl.brain.megamind import MegaMind
    from synthesis_lab.orchestration.rdl.brain.suite_games import (
        DEFAULT_SUITE_ENV_DIR,
        describe_suite,
        suite_24,
    )
    from synthesis_lab.orchestration.rdl.brain.train_level_groups import (
        _parse_groups,
        evaluate_pattern_robustness,
        train_cohort,
    )

    cycle_dir = work_dir / f"cycle_{cycle:04d}_{label}_{_utc()}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    games = suite_24(DEFAULT_SUITE_ENV_DIR)
    suite_meta = describe_suite(games)
    group_pairs = _parse_groups(groups)

    mm = MegaMind(device=device)
    warm = _load_mm(mm, best_ckpt)
    t0 = time.time()
    group_results: list[dict[str, Any]] = []

    # Plateau breakout: epsilon-greedy propose policy (not pure random).
    # Decay explore over cycles but keep floor so we don't freeze.
    epsilon = float(max(0.25, 0.85 - 0.01 * max(0, cycle - 90)))

    def _policy_fn(frame_np, available_ids, rng):
        """Fast Internal propose_logits — no full operator garden (too slow)."""
        import torch
        import torch.nn.functional as F

        ids = list(available_ids) or [1, 2, 3, 4, 5, 6]
        try:
            mm._net_I.eval()
            with torch.no_grad():
                t = torch.as_tensor(frame_np, device=mm.device)
                if t.dim() == 2:
                    t = t.unsqueeze(0)
                logits = mm._net_I.propose_logits(t.long()).reshape(-1)
                # mask to available
                mask = torch.full_like(logits, -1e9)
                for a in ids:
                    if 0 <= int(a) < logits.numel():
                        mask[int(a)] = logits[int(a)]
                # also allow if action space larger than available — pick best avail
                if torch.isfinite(mask).any() and (mask > -1e8).any():
                    # softmax sample for exploration within policy
                    probs = F.softmax(mask, dim=-1)
                    aid = int(torch.multinomial(probs, 1).item())
                    if aid in ids:
                        return aid
                return int(ids[int(rng.integers(0, len(ids)))])
        except Exception:
            return int(ids[int(rng.integers(0, len(ids)))])

    for lo, hi in group_pairs:
        cohort = collect_level_group_cohort(
            level_lo=lo,
            level_hi=hi,
            game_ids=games,
            env_dir=DEFAULT_SUITE_ENV_DIR,
            seed=seed + cycle * 97 + lo * 13,
            max_actions=max_actions,
            max_in_band_per_game=steps_per_game,
            policy_fn=_policy_fn,
            epsilon=epsilon,
        )
        train_stats = train_cohort(
            mm,
            cohort,
            epochs=epochs,
            batch_size=batch_size,
            steps_per_phase=1,
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
                    "per_game": cohort.get("per_game") or [],
                    "mean_max_levels": cohort.get("mean_max_levels"),
                    "max_levels_any": cohort.get("max_levels_any"),
                    "total_level_ups": cohort.get("total_level_ups"),
                    "games_with_level_up": cohort.get("games_with_level_up"),
                    "mean_delta_frac": cohort.get("mean_delta_frac"),
                    "epsilon": cohort.get("epsilon"),
                    "policy": cohort.get("policy"),
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
                    "event": "cycle_group_done",
                    "cycle": cycle,
                    "level_group": cohort["level_group"],
                    "n_transitions": cohort["n_transitions"],
                    "n_games_ok": cohort["n_games_ok"],
                    "pattern_robust_frac": robust["pattern_robust_frac"],
                    "games_hit": train_stats.get("games_hit"),
                    "final_mean_loss": train_stats.get("final_mean_loss"),
                    "epsilon": epsilon,
                    "mean_max_levels": cohort.get("mean_max_levels"),
                    "total_level_ups": cohort.get("total_level_ups"),
                    "max_levels_any": cohort.get("max_levels_any"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    elapsed = time.time() - t0
    cand_ckpt = cycle_dir / "megamind.pt"
    metrics = {
        "schema": "megamind.flywheel_cycle_metrics.v1",
        "created_utc": _utc(),
        "cycle": cycle,
        "label": label,
        "groups": groups,
        "steps_per_game": steps_per_game,
        "epochs": epochs,
        "elapsed_s": elapsed,
        "device": device,
        "warm_started": warm,
        "suite": suite_meta,
        "results": group_results,
        "checkpoint": str(cand_ckpt),
        "doctrine": {
            "flywheel": True,
            "level_groups": True,
            "all_games_at_once": True,
            "keep_only_if_better": True,
            "not_sp80_focus": True,
            "weco_rsi_aligned": True,
            "epsilon_greedy_policy": True,
            "private_score_v3_env_progress": True,
            "plateau_algo_adjust": True,
        },
        "epsilon": epsilon,
        "algo_note": "policy-guided collect + env level-ups in private v3",
    }
    _save_mm(
        mm,
        cand_ckpt,
        {"cycle": cycle, "label": label, "groups": groups},
    )
    metrics_path = cycle_dir / "metrics.json"
    _write_json(metrics_path, metrics)

    incumbent = _load_metrics(best_metrics_path if best_metrics_path.is_file() else None)
    decision = keep_flywheel(metrics, incumbent)
    decision["cycle"] = cycle
    decision["label"] = label
    decision["cycle_dir"] = str(cycle_dir)

    if decision.get("keep"):
        best_dir = work_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        # promote
        import shutil

        shutil.copy2(cand_ckpt, best_ckpt)
        _write_json(best_metrics_path, metrics)
        _write_json(best_dir / "decision.json", decision)
        print(
            json.dumps(
                {
                    "event": "PROMOTED",
                    "cycle": cycle,
                    "label": label,
                    "private": decision.get("candidate_private"),
                    "reason": decision.get("reason"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        print(
            json.dumps(
                {
                    "event": "REJECTED",
                    "cycle": cycle,
                    "label": label,
                    "reason": decision.get("reason"),
                    "candidate_private": decision.get("candidate_private"),
                    "incumbent_private": decision.get("incumbent_private"),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    _write_json(cycle_dir / "decision.json", decision)
    return {
        "cycle": cycle,
        "label": label,
        "decision": decision,
        "metrics_path": str(metrics_path),
        "elapsed_s": elapsed,
        "kept": bool(decision.get("keep")),
        "private": decision.get("candidate_private"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default=str(_ROOT / "synthesis_lab" / "experiments" / "megamind_flywheel"),
        help="Flywheel root (journal, best, cycles)",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-actions", type=int, default=70)
    ap.add_argument("--sleep-s", type=float, default=5.0, help="Pause between cycles")
    ap.add_argument("--max-cycles", type=int, default=0, help="0 = run forever")
    ap.add_argument("--start-cycle", type=int, default=0)
    ap.add_argument(
        "--seed-best-from",
        default="",
        help="Optional metrics.json+pt dir to seed best/ (e.g. prior L0L1 raise)",
    )
    args = ap.parse_args(argv)

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    cycles_root = root / "cycles"
    cycles_root.mkdir(exist_ok=True)
    best_dir = root / "best"
    best_dir.mkdir(exist_ok=True)
    best_ckpt = best_dir / "megamind_best.pt"
    best_metrics = best_dir / "metrics.json"
    journal = root / "journal.jsonl"
    heartbeat = root / "HEARTBEAT.json"
    stop_file = root / "STOP"
    pid_file = root / "flywheel.pid"

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("WARN: cuda unavailable → cpu", flush=True)
        device = "cpu"

    # seed best from prior raise if empty
    if args.seed_best_from and not best_ckpt.is_file():
        src = Path(args.seed_best_from)
        pt = src / "megamind_level_groups.pt"
        met = src / "metrics.json"
        if pt.is_file() and met.is_file():
            import shutil

            shutil.copy2(pt, best_ckpt)
            shutil.copy2(met, best_metrics)
            print(f"seeded best from {src}", flush=True)

    pid_file.write_text(str(os.getpid()) + "\n")
    _write_json(
        root / "FLYWHEEL.json",
        {
            "schema": "megamind.flywheel.v1",
            "started_utc": _utc(),
            "pid": os.getpid(),
            "device": device,
            "schedule": [
                {"groups": g, "steps": s, "epochs": e, "label": lab}
                for g, s, e, lab in DEFAULT_SCHEDULE
            ],
            "stop_file": str(stop_file),
            "doctrine": "keep_spinning_level_groups_all24_private_gate",
        },
    )

    print(
        json.dumps(
            {
                "event": "flywheel_start",
                "root": str(root),
                "pid": os.getpid(),
                "device": device,
                "max_cycles": args.max_cycles,
                "best_exists": best_ckpt.is_file(),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    cycle = int(args.start_cycle)
    consecutive_fail = 0
    running = True

    def _sig_handler(signum, frame):
        nonlocal running
        print(json.dumps({"event": "signal", "signum": signum}), flush=True)
        running = False

    signal.signal(signal.SIGTERM, _sig_handler)
    signal.signal(signal.SIGINT, _sig_handler)

    while running:
        if stop_file.is_file():
            print(json.dumps({"event": "stop_file", "path": str(stop_file)}), flush=True)
            break
        if args.max_cycles > 0 and cycle >= args.max_cycles:
            print(json.dumps({"event": "max_cycles", "n": args.max_cycles}), flush=True)
            break

        sched = DEFAULT_SCHEDULE[cycle % len(DEFAULT_SCHEDULE)]
        groups, steps, epochs, label = sched
        # progressive depth: every full schedule pass, bump steps slightly
        pass_n = cycle // len(DEFAULT_SCHEDULE)
        steps_eff = steps + min(12, pass_n * 2)
        epochs_eff = epochs + (1 if pass_n >= 2 and cycle % 2 == 0 else 0)

        hb = {
            "utc": _iso(),
            "cycle": cycle,
            "label": label,
            "groups": groups,
            "steps_per_game": steps_eff,
            "epochs": epochs_eff,
            "status": "running",
            "pid": os.getpid(),
            "device": device,
            "best_private": None,
        }
        inc = _load_metrics(best_metrics)
        if inc is not None:
            hb["best_private"] = _enriched_private(inc).get("private_composite")
        _write_json(heartbeat, hb)

        print(
            json.dumps(
                {
                    "event": "cycle_start",
                    "cycle": cycle,
                    "label": label,
                    "groups": groups,
                    "steps_per_game": steps_eff,
                    "epochs": epochs_eff,
                },
                sort_keys=True,
            ),
            flush=True,
        )

        try:
            row = run_one_cycle(
                cycle=cycle,
                groups=groups,
                steps_per_game=steps_eff,
                epochs=epochs_eff,
                label=label,
                device=device,
                seed=args.seed,
                batch_size=args.batch_size,
                max_actions=args.max_actions,
                work_dir=cycles_root,
                best_ckpt=best_ckpt,
                best_metrics_path=best_metrics,
            )
            consecutive_fail = 0
            _append_journal(
                journal,
                {
                    "utc": _iso(),
                    "event": "cycle",
                    **{k: row[k] for k in ("cycle", "label", "kept", "private", "elapsed_s")},
                    "reason": (row.get("decision") or {}).get("reason"),
                    "metrics_path": row.get("metrics_path"),
                },
            )
            hb["status"] = "idle_between"
            hb["last_kept"] = row.get("kept")
            hb["last_private"] = row.get("private")
            hb["utc"] = _iso()
            _write_json(heartbeat, hb)
        except Exception as exc:
            consecutive_fail += 1
            err = {
                "utc": _iso(),
                "event": "cycle_error",
                "cycle": cycle,
                "label": label,
                "error": f"{type(exc).__name__}:{exc}",
                "traceback": traceback.format_exc()[-2000:],
                "consecutive_fail": consecutive_fail,
            }
            _append_journal(journal, err)
            print(json.dumps(err, sort_keys=True), flush=True)
            if consecutive_fail >= 8:
                print(json.dumps({"event": "abort_too_many_failures"}), flush=True)
                return 1
            time.sleep(min(60.0, 5.0 * consecutive_fail))

        cycle += 1
        if args.sleep_s > 0 and running:
            time.sleep(float(args.sleep_s))

    _write_json(
        heartbeat,
        {
            "utc": _iso(),
            "status": "stopped",
            "cycle": cycle,
            "pid": os.getpid(),
        },
    )
    print(json.dumps({"event": "flywheel_stop", "cycle": cycle}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
