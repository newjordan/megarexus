"""DW Phase-3 data harvest — observed pour corpora via legal env interaction.

Random legal arrangements (click-move genomes, reach_l2 convention) on a REAL
level, then one ACTION5 pour with the full animation captured. Output schema
matches calib_l0_pours.npz (pre, frames_flat, lens, wins) so dynamics_fit
consumes it directly. EVAL/observation-only: env interaction data, no engine
internals, no training.

Extra genomes (e.g. the S3 legally-discovered, env-verified L1 winner) can be
appended with --extra-genomes to give the corpus win-side observations; their
provenance is recorded in the fixture.

    python3 harvest_pours.py --level 1 --n 60 \
        --extra-genomes '[[[3,4,2],[0,1,2],[4,1,2],[2,2,3],[5,1,3],[7,3,2]]]' \
        --out calib_l1_pours_20260706.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, LAB

from imagine_search_gate import _present, GAME_ID, ACTION_OF_ID, game_live  # noqa: E402
from imag_search_solve import actions, _apply_real, MOVES  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402


def pour_once(arc, seed, level, genome, centers):
    """Arrange via genome, pour, capture the animation. None if arrange dies."""
    env = arc.make(GAME_ID, seed=seed)
    fd, cur = _present(env, level)
    base = int(getattr(fd, "levels_completed", 0) or 0)
    if genome:
        fd, cur, _ = _apply_real(env, fd, cur, actions(genome, centers))
        if not game_live(fd):
            return None
    pre = np.asarray(cur, dtype=np.int8)
    try:
        fd = env.step(ACTION_OF_ID[5])
    except Exception:  # noqa: BLE001
        return None
    if fd is None:
        return None
    f = fd.frame
    frames = ([np.asarray(x, dtype=np.int8) for x in f] if isinstance(f, list)
              else [np.asarray(f, dtype=np.int8)])
    won = int(getattr(fd, "levels_completed", 0) or 0) > base
    return pre, frames, bool(won)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max-clicks", type=int, default=8)
    ap.add_argument("--extra-genomes", default=None,
                    help="JSON list of genomes to append (recorded in provenance)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    env0 = arc.make(GAME_ID, seed=a.seed)
    _, c0 = _present(env0, a.level)
    centers = [(int(c["center_xy"][0]), int(c["center_xy"][1]))
               for c in frame_components(c0)[:8]]
    rng = np.random.default_rng(a.seed)

    genomes = [[]]                                     # default board pour first
    for _ in range(a.n - 1):
        k = int(rng.integers(0, a.max_clicks + 1))
        genomes.append([(int(rng.integers(len(centers))), int(rng.choice(MOVES)),
                         int(rng.integers(1, 4))) for _ in range(k)])
    extra = json.loads(a.extra_genomes) if a.extra_genomes else []
    genomes += [[tuple(g) for g in gen] for gen in extra]

    pres, frames_flat, lens, wins = [], [], [], []
    dead = 0
    for i, g in enumerate(genomes):
        r = pour_once(arc, a.seed, a.level, g, centers)
        if r is None:
            dead += 1
            continue
        pre, frames, won = r
        pres.append(pre); frames_flat.extend(frames)
        lens.append(len(frames)); wins.append(won)
        if (i + 1) % 20 == 0:
            print(f"[harvest] L{a.level} {i+1}/{len(genomes)}: "
                  f"{len(pres)} pours, {sum(wins)} wins, {dead} dead", flush=True)

    out = FIXTURES / a.out
    np.savez_compressed(
        out, pre=np.stack(pres), frames_flat=np.stack(frames_flat),
        lens=np.asarray(lens, dtype=np.int32), wins=np.asarray(wins, dtype=bool))
    meta = {"level": a.level, "seed": a.seed, "n_requested": len(genomes),
            "n_pours": len(pres), "n_wins": int(sum(wins)), "dead": dead,
            "extra_genomes": extra,
            "provenance": "legal env interaction (random click-move arrangements "
                          "+ replay of legally-discovered genomes); observation "
                          "data for dynamics validation only"}
    (FIXTURES / (out.stem + ".provenance.json")).write_text(json.dumps(meta, indent=1))
    print(f"[harvest] wrote {out}: {len(pres)} pours, {sum(wins)} wins, {dead} dead")


if __name__ == "__main__":
    main()
