"""SIM3D stage S1 — fidelity closure via tick-matching (execution plan §5.S1).

Reproduces a variant's S2-sampled arrangements EXACTLY (same per-variant rng as
s2_generalize), pours each for real, and lines the real animation up against the sim
rollout so the first divergence is visible. Focus: the S2 overshoot-FPs — arrangements
the sim scores +1 (fills, no lava) but the env floods the floor (real_lava~254).

Legal: variant boards (not the t0 five) are a fitting corpus (§4.1). Diagnoses / measures;
does not silently tune emit_volume against one board — use `--measure-emit` for the honest
volume, `--emit-sweep` only to CONFIRM the lever.

    python3 s1_tickmatch.py --seed 30336 --base 0            # reproduce + classify board 336
    python3 s1_tickmatch.py --seed 30336 --base 0 --emit-sweep 37,60,90,120,160
"""
from __future__ import annotations

import argparse

import numpy as np

import _paths  # noqa: F401

from imagine_search_gate import _candidates, _apply_real, _grid, ACTION_OF_ID, game_live  # noqa: E402
from sp80_variants import make_env_variant  # noqa: E402
from pertarget_labels import pertarget_labels  # noqa: E402
from pour3d import imagine_pour  # noqa: E402
from scene import PAINT, WALL  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402
from _paths import LAB  # noqa: E402


def real_pour_frames(env2):
    fd = env2.step(ACTION_OF_ID[5])
    won = int(getattr(fd, "levels_completed", 0) or 0) > 0
    f = fd.frame
    frames = ([np.asarray(x, dtype=np.int8) for x in f] if isinstance(f, list)
              else [np.asarray(f, dtype=np.int8)])
    return frames, won


def classify(real_lab, won):
    fills = [t["fill"] for t in real_lab["targets"]]
    lava = real_lab["global"]["lava"]
    if won:
        return "WIN"
    if fills and min(fills) >= 0.85 and lava:
        return "FILL+FLOOD"          # the overshoot-FP class: all filled but floor flooded
    if lava:
        return "FLOOD"
    return "LOSS"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=30336)
    ap.add_argument("--base", type=int, default=None, help="default: seed%%1000 even->0 odd->1 (the sweep rule)")
    ap.add_argument("--arr-per", type=int, default=12)
    ap.add_argument("--emit-sweep", default=None, help="comma emit_volumes to score each arr at")
    ap.add_argument("--lane", default="stream")
    a = ap.parse_args()
    base = a.base if a.base is not None else (0 if (a.seed % 1000) % 2 == 0 else 1)
    emits = [float(x) for x in a.emit_sweep.split(",")] if a.emit_sweep else [None]

    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    env, g = make_env_variant(arc, seed=a.seed, rng=np.random.default_rng(a.seed), base_level=base)
    if env is None:
        print("degenerate variant"); return
    ntarg = int((g == 11).sum())
    crng = np.random.default_rng(a.seed)
    cands = _candidates(g, 6, (1, 2, 3), a.arr_per, crng)
    idx = crng.choice(len(cands), size=min(a.arr_per, len(cands)), replace=False)

    hdr = f"seed={a.seed} base={base} ntarg~{ntarg//16} macro-targets | emit sweep {emits}"
    print(hdr)
    print(f"{'arr':>4} {'real':>10} {'r_peak6':>7} {'r_n14':>6} {'r_frm':>5} | sim margin @emit")
    rows = []
    for k, ai in enumerate(idx):
        arrangement = cands[int(ai)]
        # real pour (fresh env each time — deterministic rebuild)
        env2, _ = make_env_variant(arc, seed=a.seed, rng=np.random.default_rng(a.seed), base_level=base)
        fd = env2.step(ACTION_OF_ID[1]); cur = _grid(fd)
        fd, cur, _ = _apply_real(env2, fd, cur, arrangement)
        if not game_live(fd):
            continue
        pre = cur.astype(np.int8)
        base_wall = int((pre == WALL).sum())
        frames, won = real_pour_frames(env2)
        real_lab = pertarget_labels(pre, frames, r=3)
        cls = classify(real_lab, won)
        peak6 = max(int((x == PAINT).sum()) for x in frames) / 16.0     # macro cells of paint
        n14 = max(int((x == WALL).sum()) for x in frames) - base_wall
        # sim at each emit_volume
        sim_ms = []
        for e in emits:
            out = imagine_pour(pre, lane=a.lane, emit_volume=e)
            sim_ms.append(out["margin"])
        rows.append((k, cls, won, peak6, n14, len(frames), sim_ms))
        ms = " ".join(f"{m:+.0f}" for m in sim_ms)
        print(f"{k:>4} {cls:>10} {peak6:>7.1f} {n14:>6} {len(frames):>5} | {ms}")

    # summary: does raising emit split WIN from FILL+FLOOD?
    wins = [r for r in rows if r[1] == "WIN"]
    floods = [r for r in rows if r[1] == "FILL+FLOOD"]
    print(f"\n{len(wins)} WIN, {len(floods)} FILL+FLOOD (overshoot-FP), {len(rows)} total")
    if emits != [None] and wins and floods:
        print("emit_volume :  WIN sim-margin(min)  |  FILL+FLOOD sim-margin(max)  |  separates?")
        for j, e in enumerate(emits):
            wmin = min(r[6][j] for r in wins)
            fmax = max(r[6][j] for r in floods)
            sep = "YES" if wmin > 0 and fmax < 0 else ("partial" if wmin > fmax else "no")
            print(f"  {str(e):>10} :  {wmin:+.2f}              |  {fmax:+.2f}                 |  {sep}")
    if wins:
        print(f"real emitted paint (macro cells): WIN peak6 = "
              f"{[round(r[3],1) for r in wins]}  FILL+FLOOD peak6 = {[round(r[3],1) for r in floods]}")


if __name__ == "__main__":
    main()
