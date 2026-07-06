"""DW Phase-2 acceptance — imagination kernel smoke test.

Proves the four Phase-2 acceptance properties on real observed frames:
  1. trusted path: gate-passing L0 scene -> KnownMap -> deterministic rollout
  2. provenance refusal: gate-failing L3 scene -> UntrustedSceneError
  3. determinism: two rollouts produce identical state sequences
  4. renderable record: per-tick macro grids -> contact-sheet PNG + npy

    python3 kernel_smoke.py [--seed 1234] [--ticks 120]
Exit 0 all pass / 2 otherwise. Fixture: fixtures/kernel_smoke_20260706.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
from _paths import LAB, FIXTURES

from imagine_search_gate import _present, GAME_ID  # noqa: E402
from scene import extract_scene  # noqa: E402
from perception_gate import gate_scene  # noqa: E402
from imagination_kernel import (KnownMap, UntrustedSceneError, GravityFall,
                                run, cast_rays, render_state)  # noqa: E402
from viz import frame_rgb  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ticks", type=int, default=120)
    a = ap.parse_args()
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    results = {}

    # 1. trusted path (L0)
    env = arc.make(GAME_ID, seed=a.seed)
    _, cur0 = _present(env, 0)
    sc0 = extract_scene(cur0)
    v0 = gate_scene(cur0, sc0)
    kmap = KnownMap.from_scene(sc0, v0, level=0, seed=a.seed)
    states = run(kmap, GravityFall(), max_ticks=a.ticks)
    results["trusted_path"] = {
        "pass": bool(v0["trusted"]) and states[-1].terminal == "quiescent",
        "gate_trusted": v0["trusted"], "ticks": states[-1].tick,
        "terminal": states[-1].terminal, "emitted": states[-1].emitted,
        "fills": states[-1].fills, "floor_touch": states[-1].floor_touch,
        "disposed": states[-1].disposed,
        "provenance": kmap.provenance,
    }

    # 2. provenance refusal: a TAMPERED frame (one target's pixels split by an
    #    erased stripe -> pixel components no longer match the scene's claim)
    #    must be gate-failing and REFUSED by the kernel. (L3 was the original
    #    specimen; the lattice-aware extractor made real L3 legitimately
    #    trusted, so the smoke now manufactures its untrusted scene.)
    cur_t = np.asarray(cur0, dtype=np.int8).copy()
    sc_t = extract_scene(cur0)                                  # claims from CLEAN frame
    tys, txs = np.nonzero(sc_t.targets[0].cells_px)             # ONE glyph's pixels
    cur_t[tys.min():tys.max() + 1, int(np.median(txs))] = 12   # split that glyph
    v_t = gate_scene(cur_t, sc_t)                               # checked vs TAMPERED frame
    try:
        KnownMap.from_scene(sc_t, v_t, level=0, seed=a.seed, tampered=True)
        refused = False
    except UntrustedSceneError:
        refused = True
    results["provenance_refusal"] = {"pass": refused and not v_t["trusted"],
                                     "tampered_gate_failures": v_t["gate_failures"]}

    # 3. determinism
    states_b = run(kmap, GravityFall(), max_ticks=a.ticks)
    same = (len(states) == len(states_b)
            and all(s1.signature() == s2.signature() for s1, s2 in zip(states, states_b)))
    results["determinism"] = {"pass": bool(same), "n_states": len(states)}

    # 4. renderable record + ray bundle
    grids = np.stack([render_state(s) for s in states])
    np.save(FIXTURES / "kernel_smoke_grids_20260706.npy", grids)
    rays = cast_rays(states[0])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        idxs = np.linspace(0, len(states) - 1, min(8, len(states))).astype(int)
        fig, axes = plt.subplots(1, len(idxs), figsize=(2.1 * len(idxs), 2.4))
        for ax, i in zip(np.atleast_1d(axes), idxs):
            ax.imshow(frame_rgb(np.kron(grids[i], np.ones((4, 4), dtype=np.int8))))
            ax.set_title(f"t={states[i].tick}", fontsize=8); ax.axis("off")
        fig.suptitle("imagination kernel rollout (GravityFall reference machinery, L0)")
        fig.tight_layout()
        png = FIXTURES / "kernel_smoke_rollout_20260706.png"
        fig.savefig(png, dpi=110); plt.close(fig)
        render_ok, png_path = True, str(png)
    except Exception as e:  # noqa: BLE001
        render_ok, png_path = False, f"{type(e).__name__}: {e}"
    results["renderable"] = {"pass": render_ok, "png": png_path,
                             "grids_npy": "kernel_smoke_grids_20260706.npy",
                             "ray_count": len(rays),
                             "ray_networks": sorted({r["network"] for r in rays})}

    ok = all(r["pass"] for r in results.values())
    payload = {"seed": a.seed, "hypothesis": "gravity_fall (reference machinery only)",
               "results": results, "all_pass": ok}
    out = FIXTURES / "kernel_smoke_20260706.json"
    out.write_text(json.dumps(payload, indent=1, default=str))
    for k, r in results.items():
        print(f"{'PASS' if r['pass'] else 'FAIL'} {k}: "
              f"{ {kk: vv for kk, vv in r.items() if kk != 'pass'} }")
    print(f"-> {out}")
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
