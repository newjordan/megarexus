"""Outer-loop gate: keep checkpoint only if better on private multi-game metrics.

Synthesizes Weco AIDE² “keep only if better” + public/private split for
MegaMind level-group raises. Does **not** claim RSI Level 1.

Private score v2 (richer)
-------------------------
Old v1 saturated at ~0.85 once all 24 games hit + pattern-ok (binary axes).
v2 adds **continuous** axes so deeper L0–L1 work can still promote:

  coverage   — multi-game robust / hit / arcade-ok (floor, not ceiling alone)
  efficiency — log-mapped train loss under fixed budget (AIDE² efficiency)
  improve    — epoch loss drop (learning happened this cycle)
  volume     — log transitions (more honest data, soft saturate)
  balance    — min/max game_hits (no single-game dominance)
  levels     — mean max_levels_seen when any (env progress)

Public loss alone never decides keep (only private composite + optional
public as weak tie-break when private is nearly equal).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

PRIVATE_SCORE_VERSION = 3


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def public_loss(metrics: dict[str, Any]) -> float:
    """Lower is better; secondary signal only."""
    results = metrics.get("results") or []
    losses = []
    for r in results:
        v = (r.get("train") or {}).get("final_mean_loss")
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            losses.append(float(v))
    return float(sum(losses) / len(losses)) if losses else float("inf")


def _loss_efficiency(loss: float) -> float:
    """Map loss → (0,1], higher is better. Continuous — never fully saturates early."""
    if not math.isfinite(loss) or loss < 0:
        return 0.0
    # log1p(loss)/log1p(ref): ref=800 → loss 0→1, 800→0.5-ish scale
    # efficiency = 1 / (1 + log1p(loss)/c)
    return float(1.0 / (1.0 + math.log1p(loss) / math.log1p(800.0)))


def _epoch_improve(train: dict[str, Any]) -> float:
    curve = train.get("epoch_curve") or []
    losses = []
    for row in curve:
        if isinstance(row, dict) and isinstance(row.get("mean_loss"), (int, float)):
            losses.append(float(row["mean_loss"]))
    if len(losses) < 2:
        return 0.0
    first, last = losses[0], losses[-1]
    if first <= 0:
        return 0.0
    drop = (first - last) / (abs(first) + 1e-9)
    return _clamp(drop, 0.0, 1.0)


def private_axes(metrics: dict[str, Any]) -> dict[str, float]:
    """Full private feature vector + composite (higher better)."""
    results = metrics.get("results") or []
    if not results:
        return {
            "schema": float(PRIVATE_SCORE_VERSION),
            "private_composite": 0.0,
            "private_base_v1": 0.0,
        }

    robust: list[float] = []
    hit: list[float] = []
    ok_nums: list[float] = []
    ok_dens: list[float] = []
    losses: list[float] = []
    improves: list[float] = []
    n_trans = 0.0
    max_lvls: list[float] = []
    n_arcade = 0.0
    n_pg = 0.0
    hit_mins: list[float] = []
    hit_maxs: list[float] = []
    level_ups: list[float] = []
    delta_fracs: list[float] = []
    games_with_up = 0.0

    for r in results:
        col = r.get("collect") or {}
        tr = r.get("train") or {}
        pr = r.get("pattern_robustness") or {}

        if isinstance(pr.get("pattern_robust_frac"), (int, float)):
            robust.append(float(pr["pattern_robust_frac"]))
        if isinstance(tr.get("robust_coverage"), (int, float)):
            hit.append(float(tr["robust_coverage"]))
        elif isinstance(tr.get("games_hit"), (int, float)) and isinstance(
            tr.get("n_games"), (int, float)
        ):
            ng = float(tr["n_games"]) or 1.0
            hit.append(float(tr["games_hit"]) / ng)

        n_ok = float(col.get("n_games_ok") or 0)
        n_g = float(tr.get("n_games") or 0)
        if n_g <= 0:
            # fall back to suite or per_game length
            n_g = float(len(col.get("per_game") or []) or (metrics.get("suite") or {}).get("n_games") or 0)
        if n_g > 0:
            ok_nums.append(n_ok)
            ok_dens.append(n_g)

        if isinstance(tr.get("final_mean_loss"), (int, float)):
            losses.append(float(tr["final_mean_loss"]))
        improves.append(_epoch_improve(tr))

        n_trans += float(col.get("n_transitions") or tr.get("n_transitions") or 0)

        gh = tr.get("game_hits") or {}
        if isinstance(gh, dict) and gh:
            vals = [float(v) for v in gh.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if vals:
                hit_mins.append(min(vals))
                hit_maxs.append(max(vals))

        # Per-game rows already roll up into the cohort totals. Adding both
        # doubles level-up counts (a sum) and pulls means toward the aggregate.
        pg_levels: list[float] = []
        pg_ups: list[float] = []
        pg_deltas: list[float] = []
        for pg in col.get("per_game") or []:
            if not isinstance(pg, dict):
                continue
            n_pg += 1.0
            if pg.get("source") == "arcade_offline" or pg.get("status") == "ok":
                n_arcade += 1.0
            if isinstance(pg.get("max_levels_seen"), (int, float)) and not isinstance(pg.get("max_levels_seen"), bool):
                pg_levels.append(float(pg["max_levels_seen"]))
            if isinstance(pg.get("n_level_ups"), (int, float)) and not isinstance(pg.get("n_level_ups"), bool):
                pg_ups.append(float(pg["n_level_ups"]))
            if isinstance(pg.get("mean_delta_frac"), (int, float)) and not isinstance(pg.get("mean_delta_frac"), bool):
                pg_deltas.append(float(pg["mean_delta_frac"]))

        if pg_levels:
            max_lvls.extend(pg_levels)
        elif isinstance(col.get("mean_max_levels"), (int, float)) and not isinstance(col.get("mean_max_levels"), bool):
            max_lvls.append(float(col["mean_max_levels"]))
        if pg_ups:
            level_ups.extend(pg_ups)
        elif isinstance(col.get("total_level_ups"), (int, float)) and not isinstance(col.get("total_level_ups"), bool):
            level_ups.append(float(col["total_level_ups"]))
        if pg_deltas:
            delta_fracs.extend(pg_deltas)
        elif isinstance(col.get("mean_delta_frac"), (int, float)) and not isinstance(col.get("mean_delta_frac"), bool):
            delta_fracs.append(float(col["mean_delta_frac"]))
        if isinstance(col.get("games_with_level_up"), (int, float)) and not isinstance(col.get("games_with_level_up"), bool):
            games_with_up += float(col["games_with_level_up"])

    robust_frac = _mean(robust)
    hit_frac = _mean(hit)
    ok_frac = (sum(ok_nums) / sum(ok_dens)) if ok_dens and sum(ok_dens) > 0 else 0.0
    mean_loss = _mean(losses) if losses else float("inf")
    loss_eff = _loss_efficiency(mean_loss if math.isfinite(mean_loss) else 1e9)
    improve = _mean(improves)
    # volume: log scale, soft saturate around ~1000 transitions
    volume = _clamp(math.log1p(n_trans) / math.log1p(1000.0))
    # balance across games
    if hit_mins and hit_maxs and max(hit_maxs) > 0:
        balance = _clamp(min(hit_mins) / max(hit_maxs))
    else:
        balance = 1.0 if hit_frac >= 0.99 else 0.5
    arcade_frac = (n_arcade / n_pg) if n_pg > 0 else ok_frac
    mean_lvl = _mean(max_lvls)
    level_axis = _clamp(mean_lvl / 2.0)  # L1 clear → strong signal
    # level-ups across suite (env truth — private)
    total_ups = sum(level_ups)
    up_axis = _clamp(math.log1p(total_ups) / math.log1p(12.0))
    suite_n = float((metrics.get("suite") or {}).get("n_games") or 0)
    if suite_n <= 0:
        suite_n = float(
            sum(float((r.get("train") or {}).get("n_games") or 0) for r in results)
        )
    games_denom = n_pg if n_pg > 0 else suite_n
    games_up_frac = _clamp(games_with_up / games_denom) if games_denom > 0 else 0.0
    # controllable dynamics: frame actually changes under actions
    delta_axis = _clamp(_mean(delta_fracs) * 4.0)  # ~0.25 mean → 1.0

    # v1 base (for comparison / logging)
    base_v1 = 0.6 * robust_frac + 0.4 * hit_frac

    # v3 composite — env progress can break train-loss plateau (weights sum 1.0)
    composite = (
        0.12 * robust_frac
        + 0.10 * hit_frac
        + 0.08 * ok_frac
        + 0.08 * arcade_frac
        + 0.16 * loss_eff
        + 0.10 * improve
        + 0.06 * volume
        + 0.04 * balance
        + 0.12 * level_axis  # env levels_completed
        + 0.08 * up_axis  # level-up events
        + 0.04 * games_up_frac
        + 0.02 * delta_axis  # frame change (control signal)
    )

    return {
        "schema": float(PRIVATE_SCORE_VERSION),
        "private_composite": float(composite),
        "private_base_v1": float(base_v1),
        "robust_frac": float(robust_frac),
        "hit_frac": float(hit_frac),
        "ok_frac": float(ok_frac),
        "arcade_frac": float(arcade_frac),
        "mean_loss": float(mean_loss) if math.isfinite(mean_loss) else -1.0,
        "loss_eff": float(loss_eff),
        "epoch_improve": float(improve),
        "n_transitions": float(n_trans),
        "volume": float(volume),
        "hit_balance": float(balance),
        "mean_max_levels": float(mean_lvl),
        "level_axis": float(level_axis),
        "total_level_ups": float(total_ups),
        "up_axis": float(up_axis),
        "games_up_frac": float(games_up_frac),
        "delta_axis": float(delta_axis),
    }


def private_score(metrics: dict[str, Any]) -> float:
    """Higher is better. Private composite v2."""
    return float(private_axes(metrics).get("private_composite") or 0.0)


def keep_only_if_better(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    *,
    min_private: float = 0.35,
    require_all_games: bool = False,
    private_eps: float = 1e-5,
    public_tie_ratio: float = 0.92,
) -> dict[str, Any]:
    """AIDE²-style gate: keep candidate only if private score improves."""
    cand_ax = private_axes(candidate)
    cand_p = float(cand_ax["private_composite"])
    cand_pub = public_loss(candidate)
    n_games = int((candidate.get("suite") or {}).get("n_games") or 0)
    if require_all_games and n_games < 24:
        return {
            "keep": False,
            "reason": "require_all_games",
            "candidate_private": cand_p,
            "candidate_public_loss": cand_pub,
            "axes": cand_ax,
            "private_score_version": PRIVATE_SCORE_VERSION,
        }
    if cand_p < min_private:
        return {
            "keep": False,
            "reason": "below_private_floor",
            "candidate_private": cand_p,
            "min_private": min_private,
            "candidate_public_loss": cand_pub,
            "axes": cand_ax,
            "private_score_version": PRIVATE_SCORE_VERSION,
        }
    if incumbent is None:
        return {
            "keep": True,
            "reason": "no_incumbent",
            "candidate_private": cand_p,
            "candidate_public_loss": cand_pub,
            "axes": cand_ax,
            "private_score_version": PRIVATE_SCORE_VERSION,
        }
    inc_ax = private_axes(incumbent)
    inc_p = float(inc_ax["private_composite"])
    inc_pub = public_loss(incumbent)
    if cand_p > inc_p + private_eps:
        return {
            "keep": True,
            "reason": "private_improved",
            "candidate_private": cand_p,
            "incumbent_private": inc_p,
            "delta_private": cand_p - inc_p,
            "candidate_public_loss": cand_pub,
            "incumbent_public_loss": inc_pub,
            "axes": cand_ax,
            "incumbent_axes": inc_ax,
            "private_score_version": PRIVATE_SCORE_VERSION,
        }
    # near-tie on private → allow public efficiency win (still multi-game covered)
    near = abs(cand_p - inc_p) <= max(private_eps * 10, 5e-4)
    covered = float(cand_ax.get("hit_frac") or 0) >= 0.99 and float(cand_ax.get("robust_frac") or 0) >= 0.99
    if near and covered and math.isfinite(cand_pub) and math.isfinite(inc_pub):
        if cand_pub < inc_pub * public_tie_ratio:
            return {
                "keep": True,
                "reason": "private_tie_public_improved",
                "candidate_private": cand_p,
                "incumbent_private": inc_p,
                "candidate_public_loss": cand_pub,
                "incumbent_public_loss": inc_pub,
                "axes": cand_ax,
                "incumbent_axes": inc_ax,
                "private_score_version": PRIVATE_SCORE_VERSION,
            }
    return {
        "keep": False,
        "reason": "not_better_on_private",
        "candidate_private": cand_p,
        "incumbent_private": inc_p,
        "delta_private": cand_p - inc_p,
        "candidate_public_loss": cand_pub,
        "incumbent_public_loss": inc_pub,
        "axes": cand_ax,
        "incumbent_axes": inc_ax,
        "private_score_version": PRIVATE_SCORE_VERSION,
        "note": "public_loss_not_sole_selector",
    }


def gate_metrics_file(
    candidate_path: Path,
    incumbent_path: Path | None = None,
    *,
    min_private: float = 0.35,
) -> dict[str, Any]:
    cand = json.loads(Path(candidate_path).read_text())
    inc = json.loads(Path(incumbent_path).read_text()) if incumbent_path else None
    decision = keep_only_if_better(cand, inc, min_private=min_private)
    decision["candidate_path"] = str(candidate_path)
    decision["incumbent_path"] = str(incumbent_path) if incumbent_path else None
    decision["schema"] = "megamind.rsi_outer_gate.v2"
    decision["weco_aligned"] = True
    return decision


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats so the CLI report is valid JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    """Print a one-line decision on stderr and the JSON report on stdout."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Decide whether a MegaMind checkpoint beats the incumbent "
            "on the private multi-game score."
        )
    )
    parser.add_argument("candidate", type=Path, help="Candidate metrics JSON")
    parser.add_argument(
        "incumbent",
        nargs="?",
        type=Path,
        help="Incumbent metrics JSON. Omit when there is no checkpoint yet.",
    )
    parser.add_argument(
        "--min-private",
        type=float,
        default=0.35,
        help="Reject a candidate whose private composite is below this floor (default: 0.35).",
    )
    args = parser.parse_args(argv)
    if not args.candidate.is_file():
        print(f"candidate metrics not found: {args.candidate}", file=sys.stderr)
        return 1
    if args.incumbent is not None and not args.incumbent.is_file():
        print(f"incumbent metrics not found: {args.incumbent}", file=sys.stderr)
        return 1
    decision = gate_metrics_file(
        args.candidate,
        args.incumbent,
        min_private=args.min_private,
    )
    kept = "KEEP" if decision.get("keep") else "REJECT"
    cand_p = decision.get("candidate_private")
    inc_p = decision.get("incumbent_private")

    def _fmt(number: Any) -> str:
        return f"{float(number):.4f}" if isinstance(number, (int, float)) else str(number)

    if inc_p is None:
        score = f"private={_fmt(cand_p)}"
    else:
        score = f"private {_fmt(inc_p)} -> {_fmt(cand_p)}"
    print(f"{kept}  reason={decision.get('reason')}  {score}", file=sys.stderr)
    print(json.dumps(_json_safe(decision), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
