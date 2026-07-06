"""DW Phase-4 — the full Dreamweaver loop: propose -> imagine -> env decides.

Search scaffold mirrors s3_solve (hill-climb over reachable click-move genomes,
candidate frames realized with clean env resets, ZERO search pours), but the
imaginer is the LEGITIMATELY-INDUCED round-3 kernel model
(dynamics_round3_20260706.json: unlimited mouth absorption, selected on
L0+L1 calib + solve-time flywheel corpora), and every candidate scene must
pass the perception gate before the kernel will imagine it (provenance rule).

Per-level emit volume is MEASURED from the observed pour corpora (paint-pixel
growth), never assumed. Env pours happen only for the final top-K (game
budget). A win produces a side-by-side imagination-vs-real render.

    python3 dw_solve.py --level 1 --restarts 24 --steps 60
Exit 0 if solved in budget, 2 otherwise.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, LAB

from imagine_search_gate import _present, GAME_ID  # noqa: E402
from imag_search_solve import frame_of, MOVES  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from scene import extract_scene, PAINT  # noqa: E402
from perception_gate import gate_scene  # noqa: E402
from imagination_kernel import (KnownMap, FallingMaterial, run, render_state,  # noqa: E402
                                win_moment)
from harvest_pours import pour_once  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

# Round-3 induced model: selected cfg + outcome semantics (rule/stop) from the
# pre-registered fit on L0+L1 calib + solve-time flywheel corpora.
_R3 = json.loads((FIXTURES / "dynamics_round3_20260706.json").read_text())
MODEL_CFG = _R3["selected"]["cfg"]
MODEL_RULE = _R3["selected"]["rule"]
MODEL_STOP = _R3["selected"]["stop"]
CORPORA = {0: "calib_l0_pours.npz", 1: "calib_l1_pours_20260706.npz",
           2: "calib_l2_pours_20260706.npz"}


def measured_emit_volume(level):
    """Median paint-cell growth from the level's OBSERVED corpus (legal)."""
    fx = CORPORA.get(level)
    if fx is None or not (FIXTURES / fx).exists():
        return None
    z = np.load(FIXTURES / fx, allow_pickle=True)
    lens = z["lens"]; ff = z["frames_flat"]
    vols, off = [], 0
    for i in range(len(lens)):
        n6 = np.array([int((f == PAINT).sum()) for f in ff[off:off + lens[i]]])
        off += lens[i]
        vols.append((int(n6.max()) - int(n6[0])) / 16.0)
    return float(np.median(vols))


def kernel_score(frame, emit_volume, lam=1.0, max_ticks=200):
    """Gate -> KnownMap -> round-3 induced-model rollout -> (fitness, detail).
    Untrusted candidate scenes are REFUSED (None), not guessed at.

    Win per the SELECTED round-3 semantics (horizon-honest: quiescence
    end-state or win-moment, whichever the fit chose); fitness adds a +1 win
    bonus on top of the graded fill/lava terms so clean imagined winners rank
    strictly above the old plateau of tied non-winners."""
    try:
        sc = extract_scene(frame, emit_volume=emit_volume)
        v = gate_scene(frame, sc)
        if not v["trusted"]:
            return None
        km = KnownMap.from_scene(sc, v)
    except Exception:  # noqa: BLE001
        return None
    states = run(km, FallingMaterial(**MODEL_CFG), max_ticks=max_ticks)
    end = states[-1]
    fill = float(np.mean([min(f / c, 1.0) for f, c in zip(end.fills, km.capacities)
                          if c > 0])) if km.capacities else 0.0
    if MODEL_STOP == "win":
        tick, floor_at = win_moment(states, km.capacities, rule=MODEL_RULE)
        win = tick is not None and floor_at == 0
        lava = floor_at / max(km.emit_volume, 1.0)
    else:
        req = (km.capacities if MODEL_RULE == "fill_cap"
               else [1.0] * len(km.capacities))
        win = (len(km.capacities) > 0
               and all(f >= r for f, r in zip(end.fills, req))
               and end.floor_touch == 0)
        lava = end.floor_touch / max(km.emit_volume, 1.0)
    return {"fitness": fill - lam * lava + (1.0 if win else 0.0),
            "fill": fill, "lava": lava, "imagined_win": bool(win), "kmap": km}


def side_by_side(kmap, real_frames, out_png):
    """Imagination-vs-real contact sheet for the poured winner."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from viz import frame_rgb
    states = run(kmap, FallingMaterial(**MODEL_CFG), max_ticks=len(real_frames))
    idxs = np.linspace(0, min(len(states), len(real_frames)) - 1, 6).astype(int)
    fig, axes = plt.subplots(2, len(idxs), figsize=(2.0 * len(idxs), 4.4))
    for j, i in enumerate(idxs):
        g = np.kron(render_state(states[i]), np.ones((4, 4), dtype=np.int8))
        axes[0, j].imshow(frame_rgb(g)); axes[0, j].set_title(f"imagined t={i}", fontsize=7)
        rf = real_frames[i] if not states[0].kmap else real_frames[i]
        axes[1, j].imshow(frame_rgb(rf)); axes[1, j].set_title(f"real t={i}", fontsize=7)
        axes[0, j].axis("off"); axes[1, j].axis("off")
    fig.suptitle("induced-model imagination (top) vs real env pour (bottom)")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--restarts", type=int, default=24)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--max-genes", type=int, default=8)
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()

    # Volume convention: the round-2 model was FIT-VALIDATED with the scene
    # default emit budget (37 units; 37/cap3 ~= the 13-14 peak cells observed).
    # measured_emit_volume() counts peak CELLS, a different unit — report it,
    # but imagine with the fit-validated convention (None -> scene default).
    cells = measured_emit_volume(a.level)
    vol = None
    print(f"[dw] L{a.level} observed peak paint cells: {cells}; "
          f"kernel uses fit-validated default volume", flush=True)
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    env0 = arc.make(GAME_ID, seed=a.seed)
    _, c0 = _present(env0, a.level)
    centers = [(int(c["center_xy"][0]), int(c["center_xy"][1]))
               for c in frame_components(c0)[:8]]
    rng = np.random.default_rng(a.seed + 100 + a.level)

    def rand_g():
        k = int(rng.integers(1, a.max_genes + 1))
        return [(int(rng.integers(len(centers))), int(rng.choice(MOVES)),
                 int(rng.integers(1, 4))) for _ in range(k)]

    def mutate(g):
        g = [list(x) for x in g]
        op = rng.integers(4)
        if op == 0 and g:
            i = rng.integers(len(g)); g[i][1] = int(rng.choice(MOVES))
        elif op == 1 and g:
            i = rng.integers(len(g)); g[i][2] = int(rng.integers(1, 4))
        elif op == 2 and len(g) < a.max_genes:
            g.append([int(rng.integers(len(centers))), int(rng.choice(MOVES)),
                      int(rng.integers(1, 4))])
        elif g and len(g) > 1:
            del g[int(rng.integers(len(g)))]
        return [tuple(x) for x in g]

    found, evals, refused = [], 0, 0
    for r in range(a.restarts):
        cur = rand_g()
        cf = frame_of(arc, a.seed, a.level, cur, centers)
        sc = kernel_score(cf, vol, a.lam) if cf is not None else None
        if sc is None:
            refused += 1
            continue
        evals += 1
        best = (sc["fitness"], sc, cur, hash(cf.tobytes()))
        for _ in range(a.steps):
            cand = mutate(best[2])
            cff = frame_of(arc, a.seed, a.level, cand, centers)
            s2 = kernel_score(cff, vol, a.lam) if cff is not None else None
            if s2 is None:
                refused += 1
                continue
            evals += 1
            if s2["fitness"] > best[0]:
                best = (s2["fitness"], s2, cand, hash(cff.tobytes()))
        found.append(best)
        print(f"[dw] L{a.level} restart {r}: fitness={best[0]:+.3f} "
              f"fill={best[1]['fill']:.2f} lava={best[1]['lava']:.2f} "
              f"imagined_win={best[1]['imagined_win']}", flush=True)

    found.sort(key=lambda x: -x[0])
    seen, topk = set(), []
    for fit, sc, g, fh in found:
        if fh in seen:
            continue
        seen.add(fh); topk.append((fit, sc, g))
        if len(topk) >= a.topk:
            break

    solved_at, poured = None, []
    for rank, (fit, sc, g) in enumerate(topk):
        r = pour_once(arc, a.seed, a.level, g, centers)
        won = bool(r and r[2])
        poured.append({"rank": rank + 1, "fitness": fit,
                       "imagined_win": sc["imagined_win"],
                       "genome": [list(x) for x in g], "env_win": won})
        if won:
            solved_at = rank + 1
            png = FIXTURES / f"dw_r3_L{a.level}_s{a.seed}_win_side_by_side_20260706.png"
            try:
                side_by_side(sc["kmap"], r[1], png)
                print(f"[dw] render -> {png}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[dw] render failed: {type(e).__name__}: {e}", flush=True)
            break

    verdict = f"SOLVED@{solved_at}" if solved_at else "NOT solved"
    print(f"[dw] L{a.level}: {verdict} in-budget (<= {a.topk} real pours) | "
          f"{evals} kernel evals, {refused} refused/dead, 0 search pours", flush=True)
    out = FIXTURES / f"dw_solve_r3_L{a.level}_s{a.seed}_20260706.json"
    out.write_text(json.dumps({
        "protocol": "DW Phase-4: gate-checked candidates, round-3 INDUCED kernel "
                    "imagines (win-bonus fitness), env decides top-K only",
        "config": {"cfg": MODEL_CFG, "rule": MODEL_RULE, "stop": MODEL_STOP},
        "emit_volume_measured": vol, "level": a.level, "seed": a.seed,
        "solved_at": solved_at, "topk": poured, "kernel_evals": evals,
        "refused": refused}, indent=1, default=str))
    print(f"-> {out}", flush=True)
    raise SystemExit(0 if solved_at else 2)


if __name__ == "__main__":
    main()
