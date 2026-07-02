#!/usr/bin/env python3
"""Validate contra.sp80_spill_model against the live SP80 engine.

For each level and many randomized platform layouts: read sprite state, predict the
spill with simulate_spill, then ACTION5 on the real engine and capture its actual
resolution (filled buckets `cevwbinfgl`, overflow `kfdcqkodyy`, win via next_level)
through instance-method snapshot hooks. Compares filled-bucket sets, overflow, and
win. Exit 0 iff every case matches exactly.

Usage: PYTHONPATH=codexlab python -B codexlab/sp80_spill_model_validate.py [levels] [configs]
"""
from __future__ import annotations
import logging
logging.disable(logging.CRITICAL)
import random
import sys

from contra_human_replay_l1_testbed import reach_clean_level, _ACT, make_offline_env, load_human_wins
from arcengine import GameAction
from contra_head.sp80_spill_model import extract_state, simulate_spill


def reach_prewin(level_n, win_index=5):
    """Drive to the human's winning arrangement for level_n: the frame right BEFORE
    the ACTION5 spill that advances levels_completed from level_n to level_n+1."""
    wins = load_human_wins()
    w = wins[win_index]
    env = make_offline_env()
    fr = env.reset()
    lvl = int(getattr(fr, "levels_completed", 0) or 0)
    acts = w["actions"]
    # find the action index that takes us from level_n -> level_n+1
    probe = make_offline_env(); pfr = probe.reset(); plvl = 0; advance_i = None
    for i, a in enumerate(acts):
        data = {"x": a["x"], "y": a["y"]} if a["id"] == 6 else None
        pfr = probe.step(_ACT.get(a["id"], GameAction.RESET), data=data)
        nl = int(getattr(pfr, "levels_completed", 0) or 0)
        if plvl == level_n and nl == level_n + 1:
            advance_i = i
            break
        plvl = nl
    if advance_i is None:
        return None
    for a in acts[:advance_i]:  # stop right before the advancing spill
        data = {"x": a["x"], "y": a["y"]} if a["id"] == 6 else None
        env.step(_ACT.get(a["id"], GameAction.RESET), data=data)
    return env


def install_hooks(game):
    cap = {}
    orig_next = game.next_level
    orig_reset = game.lpqbikobah

    def snap():
        cap["filled"] = {(int(s.x), int(s.y)) for s in game.cevwbinfgl}
        cap["overflow"] = bool(game.kfdcqkodyy)

    def hook_next(*a, **k):
        if "filled" not in cap:
            snap()
        cap["win"] = True
        return orig_next(*a, **k)

    def hook_reset(*a, **k):
        if "filled" not in cap:
            snap()
        cap.setdefault("win", False)
        return orig_reset(*a, **k)

    game.next_level = hook_next
    game.lpqbikobah = hook_reset
    return cap


def scramble(game, rng):
    """Randomly reposition movable platforms/diverters within bounds, off buckets."""
    level = game.current_level
    gw, gh = level.grid_size
    bucket_cells = set()
    for s in level.get_sprites():
        if "repwkzbkhxl" in s.tags:
            import numpy as np
            px = np.asarray(s.render())
            for r in range(px.shape[0]):
                for c in range(px.shape[1]):
                    if int(px[r, c]) != -1:
                        bucket_cells.add((s.x + c, s.y + r))
    for s in level.get_sprites():
        if not (("plzwjbfyfli" in s.tags) or ("tuvkdkhdokr" in s.tags)):
            continue
        w, h = s.width, s.height
        for _ in range(12):
            nx = rng.randint(0, max(0, gw - w))
            ny = rng.randint(3, max(3, gh - 2 - (h - 1)))
            cells = {(nx + c, ny + r) for r in range(h) for c in range(w)}
            if cells & bucket_cells:
                continue
            s.set_position(nx, ny)
            break


def bucket_pos_by_idx(state):
    return {o["idx"]: (o["x"], o["y"]) for o in state["occupants"] if o["kind"] == "bucket"}


def run_case(level_n, rng, do_scramble, prewin=False):
    if prewin:
        env = reach_prewin(level_n)
        if env is None:
            return None, {"level": level_n, "skip": "no prewin"}
    else:
        env, _ = reach_clean_level(level_n)
    game = env._game
    if do_scramble:
        scramble(game, rng)
    state = extract_state(game)
    pred = simulate_spill(state)
    bidx = bucket_pos_by_idx(state)
    pred_filled = {bidx[i] for i in pred["filled_bucket_idxs"]}

    cap = install_hooks(game)
    lvl0 = int(getattr(env.step(GameAction.ACTION5), "levels_completed", 0) or 0)
    # cap is filled by the hooks during the auto-advanced spill resolution.
    actual_filled = cap.get("filled", set())
    actual_overflow = cap.get("overflow", False)
    actual_win = cap.get("win", False)

    ok = (pred_filled == actual_filled
          and bool(pred["overflow"]) == bool(actual_overflow)
          and bool(pred["win"]) == bool(actual_win))
    return ok, {
        "level": level_n, "scramble": do_scramble,
        "pred_filled": sorted(pred_filled), "act_filled": sorted(actual_filled),
        "pred_overflow": pred["overflow"], "act_overflow": actual_overflow,
        "pred_win": pred["win"], "act_win": actual_win,
        "ticks": pred["ticks"], "terminated": pred["terminated"],
        "hook_fired": "win" in cap,
    }


def main():
    levels = list(range(int(sys.argv[1]))) if len(sys.argv) > 1 else [0, 1, 2, 3]
    n_cfg = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rng = random.Random(1234)
    total = passed = 0
    mismatches = []
    outcomes = {"win": 0, "overflow": 0, "partial": 0}
    for L in levels:
        # prewin (human winning arrangement) + clean + n_cfg scrambled
        cases = [("prewin", False), ("clean", False)] + [("scr", True)] * n_cfg
        lp = lc = 0
        for kind, scr in cases:
            try:
                ok, info = run_case(L, rng, scr, prewin=(kind == "prewin"))
            except Exception as e:
                ok, info = False, {"level": L, "kind": kind, "error": repr(e)}
            if ok is None:  # skipped (no prewin for last level)
                continue
            total += 1; lc += 1
            lp += int(ok); passed += int(ok)
            if ok and "act_win" in info:
                if info["act_win"]:
                    outcomes["win"] += 1
                elif info["act_overflow"]:
                    outcomes["overflow"] += 1
                else:
                    outcomes["partial"] += 1
            if not ok:
                mismatches.append(info)
        print(f"level {L}: {lp}/{lc} match")
    print(f"\nTOTAL: {passed}/{total} exact matches")
    print(f"actual outcomes across matched cases: {outcomes}")
    if mismatches:
        print(f"\n{len(mismatches)} mismatches (first 12):")
        for m in mismatches[:12]:
            print("  ", m)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
