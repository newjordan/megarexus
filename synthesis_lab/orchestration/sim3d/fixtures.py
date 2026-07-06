"""Dump REAL env fixtures for offline sim3d development (run once, commit the npz).

Everything here is a pure forward-physics READ of the env (present a level, apply the
already-recorded canonical genomes, pour) — no answer injection, no route lineage.
The five canonical L1 arrangements (verified winner + v52's 4 false positives) come
from runs/t0_pertarget_20260705.json; they are the DECISIVE TEST SET and are dumped
here so the simulator can be scored against their recorded real labels WITHOUT
touching the env again. HONESTY: those five are EVAL-ONLY — never calibrate on them.

Outputs (sim3d/fixtures/):
  present_l0_s1234.npz   frame [64,64] int8              L0 as presented
  present_l1_s1234.npz   frame                           L1 as presented (rotated level)
  t0_arrangements.npz    pre[5,64,64], meta json          winner + 4 FPs, pre-pour frames
  calib_l0_pours.npz     pre[N,64,64], frames (ragged)    L0 default + structured pours,
                                                          WITH full pour animations —
                                                          the CALIBRATION corpus seed
"""
from __future__ import annotations

import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, LAB

from collect_catch_play import GAME_ID, ACTION_OF_ID, game_live  # noqa: E402
from imagine_search_gate import _present, _grid, _apply_real, _candidates  # noqa: E402
from imag_search_solve import actions  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

SEED = 1234
T0_JSON = LAB / "experiments" / "megarexus_cnn" / "runs" / "t0_pertarget_20260705.json"


def _arc():
    return Arcade(arc_api_key="anonymous",
                  environments_dir=str((LAB.parent / "environment_files").resolve()),
                  operation_mode=OperationMode.OFFLINE)


def _pour_frames(env):
    """One ACTION5 press; returns the whole animation as [T,64,64] int8."""
    fd = env.step(ACTION_OF_ID[5])
    f = fd.frame
    frames = f if isinstance(f, list) else [f]
    return np.stack([np.asarray(x, dtype=np.int8) for x in frames]), fd


def dump_presents(arc):
    for level in (0, 1):
        env = arc.make(GAME_ID, seed=SEED)
        fd, cur = _present(env, level)
        np.savez_compressed(FIXTURES / f"present_l{level}_s{SEED}.npz", frame=cur)
        print(f"present L{level}: colors {dict(zip(*map(list, np.unique(cur, return_counts=True))))}")


def dump_t0_arrangements(arc):
    t0 = json.loads(T0_JSON.read_text())
    genomes = [t0["winner"]["genome"]] + [fp["genome"] for fp in t0["false_positives"]]
    names = ["winner"] + [f"fp{i}" for i in range(len(genomes) - 1)]
    pres = []
    for name, g in zip(names, genomes):
        env = arc.make(GAME_ID, seed=t0["seed"])
        fd, cur = _present(env, t0["level"])
        centers = [tuple(map(int, c["center_xy"])) for c in frame_components(cur)[:8]]
        fd, cur, _ = _apply_real(env, fd, cur, actions([tuple(x) for x in g], centers))
        assert game_live(fd), f"{name}: arrangement died"
        pres.append(cur.astype(np.int8))
        print(f"t0 {name}: arranged ok")
    np.savez_compressed(FIXTURES / "t0_arrangements.npz",
                        pre=np.stack(pres), names=np.array(names))


def dump_calib_l0(arc, n_random=24):
    """L0 pours WITH animations = the seed calibration corpus (extend with
    sp80_variants via calibrate.py --harvest for real coverage)."""
    rng = np.random.default_rng(0)
    env = arc.make(GAME_ID, seed=SEED)
    fd, cur = _present(env, 0)
    cands = _candidates(cur, max_comps=6, per_comp_moves=(1, 2, 3), n_random=n_random, rng=rng)
    pres, anims, lens = [], [], []
    for i, arrangement in enumerate(cands):
        env = arc.make(GAME_ID, seed=SEED)
        fd, cur = _present(env, 0)
        base = int(getattr(fd, "levels_completed", 0) or 0)
        fd, cur, _ = _apply_real(env, fd, cur, arrangement)
        if not game_live(fd):
            continue
        frames, fd = _pour_frames(env)
        won = int(getattr(fd, "levels_completed", 0) or 0) > base
        pres.append(cur.astype(np.int8))
        anims.append(frames)
        lens.append((len(frames), bool(won)))
    flat = np.concatenate(anims, axis=0)
    np.savez_compressed(FIXTURES / "calib_l0_pours.npz",
                        pre=np.stack(pres), frames_flat=flat,
                        lens=np.array([l for l, _ in lens], dtype=np.int32),
                        wins=np.array([w for _, w in lens], dtype=bool))
    print(f"calib L0: {len(pres)} pours, wins={sum(w for _, w in lens)}")


def load_pour(npz, i):
    """Unpack pour i from calib npz -> (pre[64,64], frames[T,64,64], win)."""
    lens = npz["lens"]; start = int(lens[:i].sum())
    return npz["pre"][i], npz["frames_flat"][start:start + int(lens[i])], bool(npz["wins"][i])


if __name__ == "__main__":
    FIXTURES.mkdir(exist_ok=True)
    arc = _arc()
    dump_presents(arc)
    dump_t0_arrangements(arc)
    dump_calib_l0(arc)
    print("fixtures written to", FIXTURES)
