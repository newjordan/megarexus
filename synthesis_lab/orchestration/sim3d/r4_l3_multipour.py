"""Round-4 L3 multi-pour probe — the user's two-spout hint, env-decided.

Hypothesis (user hint 2026-07-06: "two pouring spouts from here on out" +
the 2026-07-03 pour-count-calibration memory): the L3 solve is a POUR
SEQUENCE — clean partial pours filling mouth subsets, with fills PERSISTING
across clean pours (the earlier "fills reset" observation covered only
FLASHING pours). Each step: drive bars to a kernel-found clean-partial
placement (legal clicks, closed loop), kernel-check, pour, observe px13
persistence + flash + levels_completed. Env decides everything.

    python3 r4_l3_multipour.py --plan r4_l3_feas_t01_20260706.json,r4_l3_feas_t23_20260706.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, LAB

from imagine_search_gate import _present, GAME_ID, ACTION_OF_ID  # noqa: E402
from scene import extract_scene, WALL, FILLED  # noqa: E402
from imagination_kernel import KnownMap, FlowTrail, run  # noqa: E402
from perception_gate import gate_scene  # noqa: E402
from r4_l3_realize import (frame_np, bars_of, selected_of, ensure_selected,  # noqa: E402
                           centroid_px, MOVE_IDS, R4_CFG)
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402


def drive(env, cur, bars0, offsets, dir_map, counter):
    """Move each bar from wherever it is to bars0+offset (closed loop)."""
    targets = [sorted((y + dy, x + dx) for (y, x) in bars0[i])
               for i, (dy, dx) in enumerate(offsets)]
    order = sorted(range(len(bars0)),
                   key=lambda i: -abs(offsets[i][0]) - abs(offsets[i][1]))
    landed_all = True
    for i in order:
        for attempt in range(4):
            sc, bars = bars_of(cur)
            cands = [b for b in bars if len(b) == len(bars0[i])]
            if not cands:
                # comps fused (adjacent same-color bars): the driven bar stays
                # SELECTED (color 9) and remains trackable through that mask
                sel, _ = selected_of(cur)
                if sel and len(sel) == len(bars0[i]):
                    cands = [sel]
                else:
                    break
            cells = min(cands, key=lambda b: abs(b[0][0] - targets[i][0][0])
                        + abs(b[0][1] - targets[i][0][1]))
            dy = targets[i][0][0] - cells[0][0]
            dx = targets[i][0][1] - cells[0][1]
            if (dy, dx) == (0, 0):
                break
            cur, counter = ensure_selected(env, cur, cells, sc, counter)
            axes = ([("y", dy), ("x", dx)] if attempt % 2 == 0
                    else [("x", dx), ("y", dy)])
            for axis, delta in axes:
                key = ((int(np.sign(delta)), 0) if axis == "y"
                       else (0, int(np.sign(delta))))
                for _ in range(abs(delta)):
                    fd = env.step(ACTION_OF_ID[dir_map[key]])
                    counter += 1
                    cur = frame_np(fd)
        _, bars = bars_of(cur)
        sel, _ = selected_of(cur)
        ok = (any(sorted(b) == targets[i] for b in bars)
              or (sel is not None and sel == targets[i]))
        landed_all &= ok or offsets[i] == (0, 0)
    return cur, counter, landed_all


def kernel_check(frame, targets):
    """Best-effort rollout on the current frame; partial-clean verdict."""
    try:
        sc = extract_scene(frame)
        v = gate_scene(frame, sc)
        km = KnownMap.from_scene(sc, dict(v, trusted=True))
        end = run(km, FlowTrail(**R4_CFG), max_ticks=200)[-1]
        return {"fills": list(map(int, end.fills)), "floor": int(end.floor_touch),
                "gate_trusted": bool(v["trusted"]), "n_targets": len(km.capacities)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--plan", required=True,
                    help="comma-separated feasibility fixture names, pour order")
    a = ap.parse_args()

    steps = []
    for fx in a.plan.split(","):
        d = json.loads((FIXTURES / fx).read_text())
        entry = d["win_examples"][0] if d.get("win_examples") else d["best"]
        steps.append({"fixture": fx, "targets": d.get("targets"),
                      "offsets": [tuple(o) for o in entry["offsets"]],
                      "kernel_expect": {k: entry[k] for k in ("fills", "floor")}})

    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    env = arc.make(GAME_ID, seed=a.seed)
    fd, cur = _present(env, 3)
    base_levels = int(getattr(fd, "levels_completed", 0) or 0)
    cur = np.asarray(cur, dtype=np.int8)
    sc0, bars0 = bars_of(cur)
    counter = 0

    # direction map (empirical, on the first bar the plan moves)
    probe_i = next(i for i, o in enumerate(steps[0]["offsets"]) if o != (0, 0))
    sc, bars = bars_of(cur)
    cur, counter = ensure_selected(env, cur, bars[probe_i], sc, counter)
    dir_map = {}
    for mid in MOVE_IDS:
        before, _ = selected_of(cur)
        fd = env.step(ACTION_OF_ID[mid]); counter += 1
        cur = frame_np(fd)
        after, _ = selected_of(cur)
        if before and after and len(after) == len(before):
            dy, dx = after[0][0] - before[0][0], after[0][1] - before[0][1]
            if (dy, dx) != (0, 0):
                dir_map[(int(np.sign(dy)), int(np.sign(dx)))] = mid
    print(f"[mp] dir_map {dir_map}", flush=True)

    log = {"protocol": "multi-pour probe (user two-spout hint): clean partial "
                       "pours in sequence; persistence/flash/level observed; "
                       "env decides", "seed": a.seed, "steps": []}
    won = False
    for si, st in enumerate(steps):
        cur, counter, landed = drive(env, cur, bars0, st["offsets"], dir_map, counter)
        chk = kernel_check(cur, st["targets"])
        pre13 = int((cur == FILLED).sum())
        pre_rows = (cur == WALL).mean(axis=1) > 0.5
        print(f"[mp] step {si} ({st['fixture']}): landed={landed} kernel={chk} "
              f"pre13={pre13}", flush=True)
        if not landed:
            log["steps"].append({"step": si, **st, "landed": False,
                                 "kernel_on_frame": chk, "aborted": "not landed",
                                 "offsets": [list(o) for o in st["offsets"]]})
            print(f"[mp] step {si} ABORTED (arrangement not realized) — "
                  "no pour, budget preserved", flush=True)
            break
        fd = env.step(ACTION_OF_ID[5]); counter += 1
        frames = fd.frame if isinstance(fd.frame, list) else [fd.frame]
        last = np.asarray(frames[-1], dtype=np.int8)
        n13 = [int((np.asarray(f) == FILLED).sum()) for f in frames]
        flash = any(int(((np.asarray(f) == WALL).mean(axis=1) > 0.5)
                        [~pre_rows].sum()) >= 2 for f in frames)
        levels = int(getattr(fd, "levels_completed", 0) or 0)
        won = levels > base_levels
        rec = {"step": si, **st, "landed": landed, "kernel_on_frame": chk,
               "pre13": pre13, "n13_first_max_last": [n13[0], max(n13), n13[-1]],
               "flash": bool(flash), "levels_completed": levels,
               "state": str(fd.state), "env_win": bool(won)}
        rec["offsets"] = [list(o) for o in rec["offsets"]]
        log["steps"].append(rec)
        print(f"[mp] step {si} POUR: n13 {n13[0]}/{max(n13)}/{n13[-1]} "
              f"flash={flash} levels={levels} state={fd.state}", flush=True)
        cur = last
        if won or "GAME_OVER" in str(fd.state):
            break

    log["env_win"] = bool(won)
    out = FIXTURES / f"r4_l3_multipour_s{a.seed}_20260706.json"
    out.write_text(json.dumps(log, indent=1, default=str))
    print(f"[mp] {'LEVEL WON' if won else 'not won'} -> {out}", flush=True)
    raise SystemExit(0 if won else 2)


if __name__ == "__main__":
    main()
