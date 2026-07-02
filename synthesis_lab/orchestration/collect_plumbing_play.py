"""SCOUT_EXPERIMENT: plumbing-play curriculum collection in local SP80.

The level-2 regime (see notes/HANDOFF.md §5): an upward jet from a bottom
spout, movable bar + two platforms, ~4 pours of supply per level, and the
offline env settles the whole flow inside step(5). The winning shape is
ARRANGE PLUMBING FIRST, POUR LAST. This collector produces experience for
outcome_v5:

1. REACH — complete the first level with the trained policy (learned
   glove finder + miss-gated careful placement, imported from
   level_probe). Model use here is collection policy, never labels.
2. ARRANGE — randomized plumbing play: rank components with the
   controllability head (or take them all), click one, issue random
   moves on ALL FOUR axes, repeat under a fixed action budget. The
   gauge is NOT read here — on the plumbing level it sits on row 63
   and drains color→color, so non-background counts are blind
   (HANDOFF §5); a fixed budget is the honest cap.
3. POUR LAST — one ACTION5, then the episode ends. Hindsight labels
   only: levels_completed lives in the recorded env fields; the
   dataset derives level_up. Per-pour catch records label every
   movable (roster), pixels only.
4. DEMO REPLAY — every Nth episode replays the human-demonstration
   arrangement mined from the corpus (platform2 right×2, platform1
   right×3, bar right×1, pour). Demonstrations are legitimate
   experience (HANDOFF §2.2/§2.5); these trajectories are tagged
   policy="demo_replay" so provenance is explicit, and they must never
   be presented as model capability.

Usage:
    python synthesis_lab/orchestration/collect_plumbing_play.py \
        --run-dir synthesis_lab/experiments/megarexus_cnn/runs/outcome_v4 \
        --out synthesis_lab/experiments/megarexus_cnn/data/plumbing_play_train.pt \
        --episodes 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LAB_ROOT))
sys.path.insert(0, str(_LAB_ROOT / "orchestration"))
sys.path.insert(0, str(_LAB_ROOT / "experiments" / "megarexus_cnn"))

from engine_adapters.portband_glove_adapter import (  # noqa: E402
    catch_label,
    component_displacement,
    frame_components,
    glove_ports,
)
from collect_catch_play import (  # noqa: E402
    GAME_ID,
    EpisodeRecorder,
    find_gloves,
    game_live,
    step,
    track_glove,
)
from level_probe import careful_place, find_gloves_learned  # noqa: E402
from eval_ghost import encode  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402

from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

_REPO_ROOT = _LAB_ROOT.parent

MAX_REACH_POURS = 5      # phase-A cap; the trained policy needs ~2
ARRANGE_BUDGET = 22      # phase-B action cap (clicks + moves), pour excluded
MOVE_ACTIONS = (1, 2, 3, 4)  # up, down, left, right — ALL FOUR axes


@torch.no_grad()
def rank_movables(model, device, cur, prev, top_k: int = 3):
    """Controllability-head ranking: max logit over all four move actions
    per component ("would this displace under SOME move"). Returns
    [(logit, comp)] sorted best-first."""
    frame = torch.cat([encode(cur), encode(prev)], dim=0).unsqueeze(0).to(device)
    scored = []
    for comp in frame_components(cur):
        plane = torch.zeros(1, 1, 16, 16, device=device)
        x0, y0, x1, y1 = (int(v) // 4 for v in comp["bbox_xyxy"])
        plane[0, 0, y0:y1 + 1, x0:x1 + 1] = 1.0
        logit = max(
            float(model.forward_control(frame, plane,
                                        torch.tensor([a], device=device)))
            for a in MOVE_ACTIONS
        )
        scored.append((logit, comp))
    scored.sort(key=lambda t: -t[0])
    return scored[:top_k]


def reach_next_level(model, device, env, rec, fd, cur, prev, rng):
    """Phase A: trained catch policy until levels_completed increments.

    Returns (fd, cur, prev, reached: bool, pours: int)."""
    start_levels = int(getattr(fd, "levels_completed", 0) or 0)
    gloves, fd, cur = find_gloves_learned(model, device, env, rec, fd, cur,
                                          prev, max_gloves=1)
    if not gloves:
        gloves, fd, cur = find_gloves(env, rec, fd, cur, rng, max_gloves=1)
    glove = gloves[0] if gloves else None
    pours = 0
    for _ in range(MAX_REACH_POURS):
        if not game_live(fd) or glove is None:
            break
        fd, cur, prev, glove, verdict = careful_place(
            model, device, env, rec, fd, cur, prev, glove)
        if not game_live(fd) or verdict["decision"] != "pour":
            break
        tracked = track_glove(cur, glove)
        if tracked is not None:
            glove = tracked
        prev_grid = cur
        fd, cur, frames = step(env, rec, cur, 5)
        # recorder convention: the NEXT action's prev frame is the pour's
        # second-to-last animation frame, not the pre-pour frame — the
        # level head is trained on that stack (arrange_probe's null plan
        # scored +12.8 on the out-of-convention (entry, pre-pour) stack)
        prev = (np.asarray(frames[-2], dtype=np.int8)
                if len(frames) >= 2 else prev_grid)
        if fd is None:
            break
        pours += 1
        now = int(getattr(fd, "levels_completed", 0) or 0)
        if now > start_levels:
            return fd, cur, prev, True, pours
    return fd, cur, prev, False, pours


def component_at(grid, x: int, y: int):
    """The component whose bbox contains (x, y), else None."""
    for comp in frame_components(grid):
        x0, y0, x1, y1 = comp["bbox_xyxy"]
        if x0 <= x <= x1 and y0 <= y <= y1:
            return comp
    return None


def relocate(grid, comp):
    """Re-locate a possibly-stale roster entry by position + area, NOT
    color: selection recolors movables (clicked → color 9 highlight,
    deselected → color 8 — observed live in the smoke run), so
    color-keyed tracking follows the wrong object across clicks."""
    candidates = [c for c in frame_components(grid)
                  if abs(c["area"] - comp["area"]) <= 8]
    if not candidates:
        return None
    gx, gy = comp["center_xy"]
    return min(candidates, key=lambda c: (c["center_xy"][0] - gx) ** 2
               + (c["center_xy"][1] - gy) ** 2)


def click_and_move(env, rec, fd, cur, comp, aid, n):
    """Select a component (ACTION6 at its center) then issue n moves,
    tracking it between moves. Selection RECOLORS the clicked object, so
    after the click the component is re-identified by what now sits at
    the click point. Returns (fd, cur, prev, comp, moved, actions_spent)."""
    prev = cur
    cx, cy = int(comp["center_xy"][0]), int(comp["center_xy"][1])
    fd, cur, _ = step(env, rec, cur, 6, cx, cy)
    spent = 1
    moved = 0
    if not game_live(fd):
        return fd, cur, prev, comp, moved, spent
    comp = component_at(cur, cx, cy) or comp  # pick up the recolored self
    for _ in range(n):
        before = cur
        fd, cur, _ = step(env, rec, cur, aid)
        prev = before
        spent += 1
        if not game_live(fd):
            return fd, cur, prev, comp, moved, spent
        if component_displacement(before, cur, comp) is None:
            break  # blocked (or not actually movable) — stop pushing
        moved += 1
        tracked = track_glove(cur, comp)
        if tracked is not None:
            comp = tracked
    return fd, cur, prev, comp, moved, spent


def arrange_random(model, device, env, rec, fd, cur, prev, rng):
    """Phase B (random): random clicks + random moves on all four axes
    under a fixed action budget.

    Component choice is 50% controllability-ranked (exploit the head),
    50% uniform over ALL components (explore) — the v4 head does not yet
    know the platforms are movable (they score ≤−8 while the bar scores
    +6.8), so a positive-logit filter would starve exactly the platform
    experience outcome_v5 needs. Hindsight control labels sort out the
    truth; clicking a non-movable is cheap negative evidence."""
    ranked = [c for _, c in rank_movables(model, device, cur, prev)]
    roster = list(frame_components(cur))
    log = []
    budget = ARRANGE_BUDGET
    while budget > 2 and game_live(fd) and roster:
        if ranked and rng.random() < 0.5:
            pick = ranked[int(rng.integers(len(ranked)))]
        else:
            pick = roster[int(rng.integers(len(roster)))]
        comp = relocate(cur, pick) or pick
        aid = int(rng.choice(MOVE_ACTIONS))
        n = int(rng.integers(1, 4))
        fd, cur, prev, comp, moved, spent = click_and_move(
            env, rec, fd, cur, comp, aid, min(n, budget - 1))
        budget -= spent
        log.append({"color": int(comp["color"]),
                    "bbox_after": list(comp["bbox_xyxy"]),
                    "aid": aid, "requested": n, "moved": moved})
    # roster for the pour's catch labels: the components as they stand in
    # the final pre-pour frame (frame_components sorts largest-first)
    return fd, cur, prev, frame_components(cur)[:6], log


# Human-demonstration arrangement (HANDOFF §5, mined from the corpus and
# verified offline): the three movables sit in the mid band (funnels are
# at y<=11, the spout at y>=56); sorted by vertical position they are
# [platform1 (upper), platform2 (middle), bar (lowest)]. The demo moves
# platform2 right×2, platform1 right×3, bar right×1, then pours. This is
# a REPLAYED DEMONSTRATION — legal experience, never model capability.
DEMO_SEQUENCE = ((1, 4, 2), (0, 4, 3), (2, 4, 1))  # (y-rank, aid, n_moves)
DEMO_BAND = (12, 48)  # y0 range that isolates bar + platforms (seen live)


def arrange_demo(model, device, env, rec, fd, cur, prev, rng=None):
    """Replay the human demonstration; with rng, jitter each move count by
    ±1 and add a vertical nudge in {-1,0,+1} (replay-with-noise: still
    demonstration-derived experience, tagged demo_jitter by the caller —
    it diversifies level-up positives and measures the basin's tolerance)."""
    band = [c for c in frame_components(cur)
            if DEMO_BAND[0] <= c["bbox_xyxy"][1] <= DEMO_BAND[1]]
    by_y = sorted(band, key=lambda c: c["center_xy"][1])
    log = []
    if len(by_y) != 3:
        log.append({"error": f"expected 3 mid-band movables, found "
                             f"{len(by_y)}: {[c['bbox_xyxy'] for c in by_y]}"})
        return fd, cur, prev, by_y, log
    seq = list(DEMO_SEQUENCE)
    if rng is not None:
        jittered = []
        for y_rank, aid, n in seq:
            n = max(0, n + int(rng.integers(-1, 2)))
            jittered.append((y_rank, aid, n))
            dy = int(rng.integers(-1, 2))
            if dy:
                jittered.append((y_rank, 2 if dy > 0 else 1, abs(dy)))
        seq = [s for s in jittered if s[2] > 0]
    for y_rank, aid, n in seq:
        comp = relocate(cur, by_y[y_rank]) or by_y[y_rank]
        fd, cur, prev, comp, moved, _ = click_and_move(
            env, rec, fd, cur, comp, aid, n)
        by_y[y_rank] = comp
        log.append({"color": int(comp["color"]),
                    "bbox_after": list(comp["bbox_xyxy"]),
                    "aid": aid, "requested": n, "moved": moved})
        if not game_live(fd):
            break
    return fd, cur, prev, by_y, log


def pour_last(env, rec, fd, cur, movables):
    """One final pour with full roster catch labels (pixels only)."""
    pour_step_index = len(rec.steps)
    fd, cur, frames = step(env, rec, cur, 5)
    if fd is None:
        return fd, cur, None
    roster = []
    for g_i, g in enumerate(movables):
        ports = glove_ports(g["bbox_xyxy"], g["component_id"])
        g_label = catch_label(frames, g["mask"], ports)
        roster.append({
            "glove_index": g_i,
            "active": g_i == 0,
            "glove_bbox_xyxy": g["bbox_xyxy"],
            "glove_color": g["color"],
            "caught": g_label["caught"],
            "contact_cells": g_label["contact_cells"],
        })
    active = roster[0] if roster else None
    record = {
        "pour_step_index": pour_step_index,
        "sweep_offset": 0,
        "glove_bbox_xyxy": active["glove_bbox_xyxy"] if active else None,
        "glove_color": active["glove_color"] if active else None,
        "port_count": 4,
        "roster": roster,
        "caught": active["caught"] if active else False,
        "contact_cells": active["contact_cells"] if active else 0,
    }
    return fd, cur, record


def collect(run_dir: Path, episodes: int, seed: int, demo_every: int,
            pour_first_prob: float = 0.0, demo_jitter_prob: float = 0.0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(run_dir / "best.pt", map_location=device,
                      weights_only=False)
    stack = ckpt["config"].get("stack_frames", 2)
    model = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)) \
        .to(device).eval()
    model.load_state_dict(ckpt["model"])

    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(_REPO_ROOT / "environment_files"),
    )
    rng = np.random.default_rng(seed)
    trajectories = []
    n_reached = n_level2 = 0
    for ep in range(episodes):
        if demo_every and ep % demo_every == 0:
            policy = "demo_replay"
        elif demo_jitter_prob and rng.random() < demo_jitter_prob:
            policy = "demo_jitter"
        elif rng.random() < pour_first_prob:
            # pour on the PRISTINE entry frame, no arrangement at all —
            # without these the level head never sees a negative for
            # "unarranged plumbing pour" and scores it positive (the
            # arrange-probe null-plan false positive, outcome_v7)
            policy = "pour_first"
        else:
            policy = "random_arrange"
        env = arcade.make(GAME_ID, seed=seed + ep)
        fd = env.reset()
        cur = np.asarray(fd.frame[-1] if isinstance(fd.frame, list)
                         else fd.frame, dtype=np.int8)
        prev = cur
        rec = EpisodeRecorder()
        catches: list[dict] = []
        arrangement: list[dict] = []

        fd, cur, prev, reached, reach_pours = reach_next_level(
            model, device, env, rec, fd, cur, prev, rng)
        n_reached += int(reached)
        if reached and game_live(fd):
            if policy == "demo_replay":
                fd, cur, prev, movables, arrangement = arrange_demo(
                    model, device, env, rec, fd, cur, prev)
            elif policy == "demo_jitter":
                fd, cur, prev, movables, arrangement = arrange_demo(
                    model, device, env, rec, fd, cur, prev, rng=rng)
            elif policy == "pour_first":
                movables, arrangement = frame_components(cur)[:6], []
            else:
                fd, cur, prev, movables, arrangement = arrange_random(
                    model, device, env, rec, fd, cur, prev, rng)
            if game_live(fd):
                fd, cur, record = pour_last(env, rec, fd, cur, movables)
                if record is not None:
                    catches.append(record)

        final_levels = int(getattr(fd, "levels_completed", 0) or 0) if fd else 0
        n_level2 += int(final_levels >= 2)
        rec.steps.append({
            "frame": cur, "action_id": 0, "action_x": 0, "action_y": 0,
            "levels_completed": final_levels,
            "state": str(getattr(fd, "state", None)) if fd else "None",
        })
        trajectories.append({
            "game_id": GAME_ID,
            "rollout_id": f"plumbing_seed{seed + ep}_{policy}",
            "policy": policy,
            "steps": rec.steps,
            "catches": catches,
            "arrangement": arrangement,
        })
        print(f"[plumb] ep {ep} ({policy}): steps={len(rec.steps)} "
              f"reach_pours={reach_pours} arrange_actions="
              f"{sum(a.get('requested', 0) + 1 for a in arrangement if 'aid' in a)} "
              f"levels={final_levels} "
              f"state={str(getattr(fd, 'state', None)) if fd else 'None'}")
    print(f"[plumb] {n_reached}/{episodes} episodes reached the plumbing "
          f"level; {n_level2}/{episodes} completed it (levels_completed>=2)")
    return trajectories


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=20290001)
    p.add_argument("--demo-every", type=int, default=5,
                   help="every Nth episode replays the human demonstration "
                        "(0 = never)")
    p.add_argument("--pour-first-prob", type=float, default=0.0,
                   help="probability of pouring on the pristine entry frame "
                        "with no arrangement (null-arrangement negatives)")
    p.add_argument("--demo-jitter-prob", type=float, default=0.0,
                   help="probability of a jittered demo replay (±1 move, "
                        "vertical nudges): diverse demonstration-derived "
                        "positives + basin-tolerance measurement")
    args = p.parse_args()
    trajectories = collect(args.run_dir, args.episodes, args.seed,
                           args.demo_every, args.pour_first_prob,
                           args.demo_jitter_prob)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajectories, args.out)
    n = sum(len(t["steps"]) for t in trajectories)
    n_demo = sum(1 for t in trajectories if t["policy"] == "demo_replay")
    print(f"[plumb] wrote {len(trajectories)} episodes ({n_demo} demo "
          f"replays) / {n} steps → {args.out}")


if __name__ == "__main__":
    main()
