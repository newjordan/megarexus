"""SIM3D stage S2 — generalization sweep (THE overfit antidote).

MANDATORY before any further claim (execution plan §5.S2, honesty rule §4.2): the
tonight micro-rules were iterated until S0 passed, so S0 alone can be accidental
overfit. This measures whether the SAME simulator (S0-gate config, emit_volume=37)
generalizes to boards it never touched during rule-fitting.

Protocol (honest, no fitting here — this only MEASURES):
  * Harvest fresh `sp80_variants` boards, multi-base (base 0 AND 1), never seen.
  * Per board, SAMPLE arrangements uniformly from the proposer distribution
    (`_candidates`) — NEVER selected by sim score (that would leak). Pour each once
    for real; the env grades win + animation.
  * Run the sim on each pre-pour frame; read its margin with the SAME
    `pertarget_labels` scorer used for the real animation.

Reports:
  win-sign agreement  = mean( (sim_margin>0) == env_win )
  margin AUC          = P(sim_margin(win) > sim_margin(loss))  (tie-aware)
  winner_ok agreement = mean( sim_winner_ok == env_win )
  per-board argmax    = of boards with >=1 real win among sampled arrangements,
                        does the sim's top-margin arrangement actually env-win?
                        (the v52-killer: AUC-good but argmax-imprecise, §5 failure map)

GATE (execution plan §5.S2): win-sign >= 95% AND margin AUC >= 0.98 AND S0 PASS.
If it fails, the failing boards are the S1 tick-match corpus — do NOT touch the t0 five.

    python3 s2_generalize.py --variants 200 --arr-per 6 --bases 0,1 \
        --out fixtures/s2_variants_20260705.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _paths  # noqa: F401
from _paths import LAB, PKG, FIXTURES

from imagine_search_gate import _candidates, _apply_real, _grid, ACTION_OF_ID, game_live  # noqa: E402
from sp80_variants import make_env_variant  # noqa: E402
from pertarget_labels import pertarget_labels  # noqa: E402
from pour3d import imagine_pour  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402


def auc_score(scores, labels):
    """Tie-aware ROC-AUC = Mann-Whitney U / (n_pos*n_neg). NaN if a class is empty."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    n_pos = int(labels.sum()); n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(s_sorted):                       # average-rank ties
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    sum_pos = ranks[labels].sum()
    return float((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def real_pour(env2, pre_frame, base_wall, ntarg, r):
    """One real ACTION5; env-graded win + per-target margin from pertarget_labels.
    The env is freshly built per pour (no prior completions), so win = completed>0."""
    fd = env2.step(ACTION_OF_ID[5])
    if fd is None:
        return None
    won = int(getattr(fd, "levels_completed", 0) or 0) > 0
    f = fd.frame
    frames = ([np.asarray(x, dtype=np.int8) for x in f] if isinstance(f, list)
              else [np.asarray(f, dtype=np.int8)])
    lab = pertarget_labels(pre_frame, frames, r=r)
    margins = [t["fill"] - t["lavan"] for t in lab["targets"]] or [0.0]
    return {"env_win": bool(won), "real_margin": float(min(margins)),
            "n_targets_real": int(lab["n_targets"]),
            "sus": float(lab["global"]["sus"]), "lava": int(lab["global"]["lava"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=200)
    ap.add_argument("--arr-per", type=int, default=6)
    ap.add_argument("--bases", default="0,1")
    ap.add_argument("--base-seed", type=int, default=30000)   # disjoint from harvest_variants (20000)
    ap.add_argument("--lane", default="stream")
    ap.add_argument("--emit-volume", type=float, default=None,  # None = S0 gate default (37, measured)
                    help="sim emission cap; default matches the decisive-test config")
    ap.add_argument("--r", type=int, default=3)
    ap.add_argument("--ledge-spill-floor", action="store_true",
                    help="S1 iter2: HydroParams.ledge_spill_floor (floor-gated bar-lip cascade)")
    ap.add_argument("--n-sources", type=int, default=None,
                    help="P1 v2 MULTI-SOURCE family: >=2 spaced sources (default None = single-source)")
    ap.add_argument("--with-redirect", action="store_true",
                    help="P1 v2 TRUE-REDIRECT family: place a tuvkdkhdokr block (use --bases 4,5)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # v2 families draw from rng only when engaged, so single-source runs stay byte-identical.
    vkw = {}
    if a.n_sources is not None:
        vkw["n_sources"] = a.n_sources
    if a.with_redirect:
        vkw["with_redirect"] = True

    from raycast import HydroParams
    sim_params = (HydroParams(ledge_spill_floor=True)
                  if a.ledge_spill_floor and a.lane == "stream" else None)
    bases = [int(x) for x in a.bases.split(",") if x != ""]
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)

    recs = []
    nvar = 0
    for v in range(a.variants):
        base = bases[v % len(bases)]
        seed = a.base_seed + v
        env, g = make_env_variant(arc, seed=seed,
                                  rng=np.random.default_rng(seed), base_level=base, **vkw)
        if env is None:
            continue
        nvar += 1
        ntarg = int((g == 11).sum()); base_wall = int((g == 14).sum())
        crng = np.random.default_rng(seed)
        cands = _candidates(g, 6, (1, 2, 3), a.arr_per, crng)
        # SAMPLE uniformly from the proposer distribution (never by sim score).
        idx = crng.choice(len(cands), size=min(a.arr_per, len(cands)), replace=False)
        for ai in idx:
            arrangement = cands[int(ai)]
            try:
                env2, g2 = make_env_variant(arc, seed=seed,
                                            rng=np.random.default_rng(seed), base_level=base, **vkw)
                if env2 is None:
                    break
                fd = env2.step(ACTION_OF_ID[1]); cur = _grid(fd)
                fd, cur, _ = _apply_real(env2, fd, cur, arrangement)
                if not game_live(fd):
                    continue
                pre = cur.astype(np.int8)
                real = real_pour(env2, pre, base_wall, ntarg, a.r)
                if real is None:
                    continue
                sim = imagine_pour(pre, lane=a.lane, params=sim_params,
                                   emit_volume=a.emit_volume)
            except (ValueError, AttributeError, RuntimeError, IndexError):
                continue
            recs.append({
                "variant": v, "base": base, "n_targets": ntarg,
                "pre": pre,
                "sim_margin": float(sim["margin"]),
                "sim_ok": bool(sim["winner_ok"]),
                "sim_native_fill": float(np.mean(sim["native"]["mouth_fill"]))
                if sim["native"]["mouth_fill"] else 0.0,
                "sim_native_lava": bool(sim["native"]["lava"]),
                "n_targets_sim": int(sim["labels"]["n_targets"]),
                "env_win": real["env_win"], "real_margin": real["real_margin"],
                "n_targets_real": real["n_targets_real"],
                "real_sus": real["sus"], "real_lava": real["lava"],
            })
        if (v + 1) % 25 == 0:
            nw = sum(r["env_win"] for r in recs)
            print(f"[s2] {v+1}/{a.variants} variants ({nvar} valid): "
                  f"{len(recs)} pours, {nw} env-wins", flush=True)

    # ---- metrics ----
    sim_m = np.array([r["sim_margin"] for r in recs], dtype=float)
    sim_ok = np.array([r["sim_ok"] for r in recs], dtype=bool)
    win = np.array([r["env_win"] for r in recs], dtype=bool)
    n = len(recs); nw = int(win.sum())

    win_sign = float(np.mean((sim_m > 0) == win)) if n else float("nan")
    ok_agree = float(np.mean(sim_ok == win)) if n else float("nan")
    auc = auc_score(sim_m, win)

    # per-board argmax (the v52-killer): boards with >=1 real win among sampled arrs
    by_board = {}
    for r in recs:
        by_board.setdefault(r["variant"], []).append(r)
    boards_with_win = [b for b in by_board.values() if any(x["env_win"] for x in b)]
    argmax_hits = 0
    for b in boards_with_win:
        top = max(b, key=lambda x: x["sim_margin"])
        argmax_hits += int(top["env_win"])
    n_boards_win = len(boards_with_win)
    argmax_prec = argmax_hits / n_boards_win if n_boards_win else float("nan")

    # base breakdown (base 0 = src-top geometry, base 1 = rotated-inherited)
    base_rows = {}
    for base in sorted(set(r["base"] for r in recs)):
        sub = [r for r in recs if r["base"] == base]
        sm = np.array([r["sim_margin"] for r in sub]); wn = np.array([r["env_win"] for r in sub])
        base_rows[base] = {"n": len(sub), "wins": int(wn.sum()),
                           "win_sign": float(np.mean((sm > 0) == wn)),
                           "auc": auc_score(sm, wn)}

    emit = 37.0 if a.emit_volume is None else a.emit_volume
    gate = (win_sign >= 0.95) and (auc >= 0.98)
    report = {
        "stage": "S2", "lane": a.lane, "emit_volume": emit,
        "variants_requested": a.variants, "valid_variants": nvar,
        "n_pours": n, "n_env_wins": nw, "win_rate": nw / max(n, 1),
        "win_sign_agreement": win_sign, "winner_ok_agreement": ok_agree,
        "margin_auc": auc,
        "boards_with_win": n_boards_win, "argmax_precision_at1": argmax_prec,
        "by_base": base_rows,
        "gate_win_sign_>=0.95": bool(win_sign >= 0.95),
        "gate_auc_>=0.98": bool(auc >= 0.98),
        "GATE": "PASS" if gate else "FAIL",
    }
    print("\n=== SIM3D STAGE S2 — generalization sweep ===")
    print(json.dumps(report, indent=1))

    if a.out:
        from pathlib import Path
        out = Path(a.out)
        if not out.is_absolute():
            out = FIXTURES / out.name
        np.savez_compressed(
            out,
            sim_margin=sim_m, sim_ok=sim_ok, env_win=win,
            real_margin=np.array([r["real_margin"] for r in recs], dtype=float),
            base=np.array([r["base"] for r in recs], dtype=np.int16),
            variant=np.array([r["variant"] for r in recs], dtype=np.int32),
            n_targets=np.array([r["n_targets"] for r in recs], dtype=np.int16),
            n_targets_real=np.array([r["n_targets_real"] for r in recs], dtype=np.int16),
            n_targets_sim=np.array([r["n_targets_sim"] for r in recs], dtype=np.int16),
            real_lava=np.array([r["real_lava"] for r in recs], dtype=np.int32),
            pre=np.stack([r["pre"] for r in recs]) if recs else np.zeros((0, 64, 64), np.int8),
            report=json.dumps(report),
        )
        print(f"[s2] wrote {out}")
    return report


if __name__ == "__main__":
    main()
