"""Round-4 L3 realization — drive bars to a kernel-certified winning placement.

Takes the clean-win placements from r4_l3_feasibility (kernel space) and
REALIZES one in the env with legal actions only: click a bar (ACTION6 at its
current pixel center), apply move actions (direction map measured empirically
in-episode, never assumed), verify each landing from the FRAME (closed loop),
then gate + kernel-certify the final arrangement and pour (ACTION5) only if
the kernel imagines a clean win. Env decides. Zero search pours.

    python3 r4_l3_realize.py [--seed 1234] [--win-index 0]
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, LAB

from imagine_search_gate import _present, GAME_ID, ACTION_OF_ID, game_live  # noqa: E402
from scene import extract_scene, _pixel_components, MOVABLE, SELECTED, GRID  # noqa: E402
from perception_gate import gate_scene  # noqa: E402
from imagination_kernel import KnownMap, FlowTrail, run  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

_R4 = json.loads((FIXTURES / "dynamics_round4_20260706.json").read_text())
R4_CFG = {**_R4["selected"]["cfg"]}
R4_CFG["volume"] = None if R4_CFG.get("volume") != "measured" else \
    _R4["measured_volumes"].get("l3_calib")
MOVE_IDS = (1, 2, 3, 4)


def frame_np(fd):
    f = fd.frame
    return np.asarray(f[-1] if isinstance(f, list) else f, dtype=np.int8)


def bars_of(frame):
    """Movable bars as sorted lattice-cell lists + the scene (canonical coords)."""
    sc = extract_scene(frame)
    g = np.asarray(sc.g)
    comps = _pixel_components((g == MOVABLE) | (g == SELECTED))
    bars = sorted(([(int(y), int(x)) for y, x in zip(*c)] for c in comps),
                  key=lambda cells: min(cells))
    return sc, bars


def selected_of(frame):
    """Cells of the currently SELECTED bar (color 9), or None."""
    sc = extract_scene(frame)
    comps = _pixel_components(np.asarray(sc.g) == SELECTED)
    if len(comps) != 1:
        return None, sc
    ys, xs = comps[0]
    return sorted((int(y), int(x)) for y, x in zip(ys, xs)), sc


def ensure_selected(env, cur, cells, sc, counter):
    """Click the bar at `cells` unless it is already the selection."""
    sel, _ = selected_of(cur)
    if sel == sorted(cells):
        return cur, counter
    cx, cy = centroid_px(cells, sc)
    fd = env.step(ACTION_OF_ID[6], data={"x": int(cx), "y": int(cy)})
    counter += 1
    cur = frame_np(fd)
    sel, _ = selected_of(cur)
    if sel != sorted(cells):
        raise SystemExit(f"selection failed: wanted {cells[:2]}.., got {sel and sel[:2]}")
    return cur, counter


def centroid_px(cells, sc):
    """Lattice cells -> ORIGINAL-frame pixel center (L3 is unflipped)."""
    u, (py, px) = sc.cell, sc.pad
    ys = [y for y, _ in cells]; xs = [x for _, x in cells]
    cy = int(round(np.mean(ys) * u - py + u / 2))
    cx = int(round(np.mean(xs) * u - px + u / 2))
    return cx, cy


def kernel_certify(frame):
    sc = extract_scene(frame)
    v = gate_scene(frame, sc)
    if not v["trusted"]:
        return None, None
    km = KnownMap.from_scene(sc, v, stage="l3_realize")
    states = run(km, FlowTrail(**R4_CFG), max_ticks=200)
    end = states[-1]
    win = (all(f >= c for f, c in zip(end.fills, km.capacities))
           and end.floor_touch == 0)
    return {"fills": list(map(int, end.fills)), "floor": int(end.floor_touch),
            "win": bool(win)}, km


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--win-index", type=int, default=0,
                    help="-1 = use the search's BEST (non-win) placement")
    ap.add_argument("--diagnostic-pour", action="store_true",
                    help="pour even without kernel certification (pre-registered "
                         "diagnostic: tests the kernel's floor prediction on a "
                         "near-clean arrangement; the animation is flywheel data)")
    a = ap.parse_args()

    feas = json.loads((FIXTURES / "r4_l3_feasibility_20260706.json").read_text())
    entry = (feas["best"] if a.win_index < 0
             else feas["win_examples"][a.win_index])
    target_offsets = [tuple(o) for o in entry["offsets"]]
    print(f"[realize] target offsets {target_offsets} "
          f"(kernel expectation: {entry.get('fills')} floor={entry.get('floor')})",
          flush=True)

    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    env = arc.make(GAME_ID, seed=a.seed)
    fd, cur = _present(env, 3)
    base_levels = int(getattr(fd, "levels_completed", 0) or 0)
    cur = np.asarray(cur, dtype=np.int8)
    sc0, bars0 = bars_of(cur)
    print(f"[realize] {len(bars0)} bars, sizes {[len(b) for b in bars0]}", flush=True)
    if len(bars0) != len(target_offsets):
        raise SystemExit(f"bar count mismatch: {len(bars0)} vs {len(target_offsets)}")

    # --- measure the move-direction map on a bar that must move anyway ---
    dir_map, actions_used = {}, 0
    probe_i = next(i for i, o in enumerate(target_offsets) if o != (0, 0))
    sc, bars = bars_of(cur)
    cur, actions_used = ensure_selected(env, cur, bars[probe_i], sc, actions_used)
    for mid in MOVE_IDS:
        before, _ = selected_of(cur)
        fd = env.step(ACTION_OF_ID[mid]); actions_used += 1
        cur = frame_np(fd)
        after, _ = selected_of(cur)
        if before and after and len(after) == len(before):
            dy, dx = after[0][0] - before[0][0], after[0][1] - before[0][1]
            if (dy, dx) != (0, 0):
                dir_map[(int(np.sign(dy)), int(np.sign(dx)))] = mid
    print(f"[realize] dir_map {dir_map} ({actions_used} actions)", flush=True)
    if len(dir_map) < 4:
        raise SystemExit(f"direction map incomplete: {dir_map}")

    # --- drive each bar to its target (closed loop, one bar at a time) ---
    order = sorted(range(len(bars0)), key=lambda i: -abs(target_offsets[i][0])
                   - abs(target_offsets[i][1]))
    targets = [sorted((y + dy, x + dx) for (y, x) in bars0[i])
               for i, (dy, dx) in enumerate(target_offsets)]
    claimed = []                                    # targets already landed

    def find_bar(i, bars):
        """Current incarnation of original bar i: same size, nearest its target,
        not a bar already parked on another claimed target."""
        cands = [b for b in bars if len(b) == len(bars0[i])
                 and sorted(b) not in claimed]
        return min(cands, key=lambda b: abs(b[0][0] - targets[i][0][0])
                   + abs(b[0][1] - targets[i][0][1]))

    for i in order:
        if target_offsets[i] == (0, 0):
            claimed.append(targets[i])
            continue
        for attempt in range(4):
            sc, bars = bars_of(cur)
            cells = find_bar(i, bars)
            dy = targets[i][0][0] - cells[0][0]
            dx = targets[i][0][1] - cells[0][1]
            if (dy, dx) == (0, 0):
                break
            cur, actions_used = ensure_selected(env, cur, cells, sc, actions_used)
            axes = [("y", dy), ("x", dx)] if attempt % 2 == 0 else [("x", dx), ("y", dy)]
            for axis, delta in axes:
                key = (int(np.sign(delta)), 0) if axis == "y" else (0, int(np.sign(delta)))
                for _ in range(abs(delta)):
                    fd = env.step(ACTION_OF_ID[dir_map[key]])
                    actions_used += 1
                    cur = frame_np(fd)
        sc, bars = bars_of(cur)
        landed = any(sorted(b) == targets[i] for b in bars)
        if landed:
            claimed.append(targets[i])
        print(f"[realize] bar {i} -> {'OK' if landed else 'MISSED'} "
              f"(target head {targets[i][0]})", flush=True)

    # --- certify + pour ---
    verdict, km = kernel_certify(cur)
    print(f"[realize] kernel verdict on realized frame: {verdict}", flush=True)
    result = {"protocol": "r4 L3 realization: feasibility placement driven by legal "
                          "clicks, empirically-measured dir map, closed-loop landing "
                          "checks, kernel certification, single env pour",
              "seed": a.seed, "win_index": a.win_index,
              "target_offsets": [list(o) for o in target_offsets],
              "actions_used": actions_used, "kernel_verdict": verdict,
              "env_win": None}
    if verdict and (verdict["win"] or a.diagnostic_pour):
        from scene import WALL
        pre14_rows = (cur == WALL).mean(axis=1) > 0.5
        fd = env.step(ACTION_OF_ID[5]); actions_used += 1
        frames = fd.frame if isinstance(fd.frame, list) else [fd.frame]
        flash = any(int(((np.asarray(f) == WALL).mean(axis=1) > 0.5)
                        [~pre14_rows].sum()) >= 2 for f in frames)
        won = int(getattr(fd, "levels_completed", 0) or 0) > base_levels
        result["env_win"] = bool(won)
        result["env_flash"] = bool(flash)
        result["n_anim_frames"] = len(frames)
        np.savez_compressed(FIXTURES / f"r4_l3_diagpour_s{a.seed}_20260706.npz",
                            pre=cur, frames_flat=np.stack(
                                [np.asarray(f, dtype=np.int8) for f in frames]),
                            lens=np.asarray([len(frames)], dtype=np.int32),
                            wins=np.asarray([won], dtype=bool))
        print(f"[realize] POURED -> env_win={won} env_flash={flash} "
              f"(kernel said win={verdict['win']} floor={verdict['floor']})",
              flush=True)
    else:
        print("[realize] kernel does not certify a clean win — NOT pouring "
              "(no grinding; refine placement/realization first)", flush=True)

    out = FIXTURES / f"r4_l3_realize_s{a.seed}_20260706.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"-> {out}", flush=True)
    raise SystemExit(0 if result["env_win"] else 2)


if __name__ == "__main__":
    main()
