"""SCOUT_EXPERIMENT: pure object-control curriculum in local SP80.

User-endorsed foundational curriculum (HANDOFF §6): selection mechanics
as their OWN subject, not a side effect of arranging. The arrange
planner provably depends on the composition "imagine the click, then
query control on the imagined frame" (rank_movables_if_clicked) — this
collector produces the experience that makes both halves robust:

- moves with NOTHING selected (control negatives for every component;
  dynamics learns the no-op),
- clicks on movables, non-movables, and EMPTY cells (click dynamics:
  what recolors, what doesn't),
- selection SWITCHING — click A then click B, then move (A deselects
  to color 8, B selects to color 9, only B moves — confirmed in the
  smoke run),
- re-clicks on the already-selected component (stays selected, still
  moves — no toggle-off; smoke run),
- clicks on EMPTY cells do NOT deselect (the selected component still
  moves afterward; smoke run),
- ordinary click+move on all four axes with varied repeat counts.

No new head, no new labels: everything lands in the EXISTING dynamics /
control / inverse-dynamics objectives via hindsight (dataset.py derives
control samples from move rows by component displacement). The trained
model is used only to REACH the plumbing level (collection policy,
never labels). Pours are scarce; this curriculum mostly does not pour —
`--pour-prob` optionally ends an episode with a pour for extra level
negatives on arbitrary post-play states.

Usage:
    python synthesis_lab/orchestration/collect_control_play.py \
        --run-dir <release-dir-with-best.pt> \
        --out synthesis_lab/experiments/megarexus_cnn/data/control_play_train.pt \
        --episodes 200 --seed 20300001
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
    component_displacement,
    frame_components,
)
from collect_catch_play import (  # noqa: E402
    GAME_ID,
    EpisodeRecorder,
    game_live,
    step,
)
from collect_plumbing_play import (  # noqa: E402
    DEMO_BAND,
    MOVE_ACTIONS,
    click_and_move,
    component_at,
    pour_last,
    reach_next_level,
    relocate,
)
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402

from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

_REPO_ROOT = _LAB_ROOT.parent

# Action budgets. The gauge drains ~1.5 cells/action and is unreadable
# on the plumbing level (row 63, color→color), so fixed caps are the
# honest limit; ARRANGE_BUDGET=22 survived 800+ plumbing episodes.
CONTROL_BUDGET = 26   # plumbing-level probe budget (clicks + moves)
# Level-1 probing is OFF by default: clicking a glove recolors it to
# selection color 9 and the learned glove finder no longer recognizes
# it — the reach phase falls back to interaction probing, drains the
# gauge, and the episode dies at level 0 (observed: smoke eps 0/3,
# seeds 20300001/4, ~120 steps, GAME_OVER). The finder's appearance
# sensitivity is itself a finding; fix the finder before re-enabling.
L1_BUDGET = 0         # level-1 probe budget before the reach phase


def pick_component(comps, rng):
    """50% mid-band (where the movables live — keeps positive-control
    coverage dense), 50% uniform over ALL components (non-movables are
    cheap negative evidence)."""
    band = [c for c in comps if DEMO_BAND[0] <= c["bbox_xyxy"][1] <= DEMO_BAND[1]]
    pool = band if (band and rng.random() < 0.5) else comps
    return pool[int(rng.integers(len(pool)))]


def empty_cell(grid, rng, tries: int = 20):
    """A random cell outside every component bbox, else None."""
    h, w = grid.shape
    for _ in range(tries):
        x, y = int(rng.integers(w)), int(rng.integers(h))
        if component_at(grid, x, y) is None:
            return x, y
    return None


def color_at_click(grid, comp):
    """Post-click color of a (possibly stale) roster entry, for the log."""
    found = relocate(grid, comp)
    return int(found["color"]) if found else None


def probe_moves(env, rec, fd, cur, rng, n, watch=None):
    """Issue n random-axis moves; report displacement of `watch` (a
    component) per move. Diagnostic only — labels are hindsight."""
    moved = 0
    spent = 0
    for _ in range(n):
        if not game_live(fd):
            break
        aid = int(rng.choice(MOVE_ACTIONS))
        before = cur
        fd, cur, _ = step(env, rec, cur, aid)
        spent += 1
        if watch is not None and game_live(fd):
            w = relocate(before, watch) or watch
            if component_displacement(before, cur, w) is not None:
                moved += 1
                watch = relocate(cur, w) or w
    return fd, cur, moved, spent, watch


def control_play(env, rec, fd, cur, rng, budget):
    """Selection/control probes under a fixed action budget."""
    log = []
    selected = None  # collector bookkeeping only, never a label

    # Entry state is guaranteed unselected: moves that displace nothing.
    fd, cur, moved, spent, _ = probe_moves(env, rec, fd, cur, rng, 2)
    budget -= spent
    log.append({"probe": "move_unselected", "spent": spent, "moved": moved})

    while budget > 1 and game_live(fd):
        comps = frame_components(cur)
        if not comps:
            break
        r = rng.random()
        if r < 0.15:
            cell = empty_cell(cur, rng)
            if cell is None:
                continue
            fd, cur, _ = step(env, rec, cur, 6, cell[0], cell[1])
            budget -= 1
            fd, cur, moved, spent, _ = probe_moves(
                env, rec, fd, cur, rng, 1, watch=selected)
            budget -= spent
            log.append({"probe": "click_empty", "cell": list(cell),
                        "selected_still_moves": moved})
        elif r < 0.30 and selected is not None:
            comp = relocate(cur, selected) or selected
            cx, cy = int(comp["center_xy"][0]), int(comp["center_xy"][1])
            fd, cur, _ = step(env, rec, cur, 6, cx, cy)
            budget -= 1
            fd, cur, moved, spent, _ = probe_moves(
                env, rec, fd, cur, rng, 1, watch=comp)
            budget -= spent
            log.append({"probe": "reclick",
                        "color_after": color_at_click(cur, comp),
                        "moved": moved})
            selected = None  # toggle behavior unknown — re-establish next click
        elif r < 0.55 and len(comps) >= 2:
            a = pick_component(comps, rng)
            b = pick_component(comps, rng)
            if b["component_id"] == a["component_id"]:
                continue
            ax, ay = int(a["center_xy"][0]), int(a["center_xy"][1])
            fd, cur, _ = step(env, rec, cur, 6, ax, ay)
            budget -= 1
            if not game_live(fd):
                break
            n = int(rng.integers(1, 3))
            fd, cur, prev, b, moved, spent = click_and_move(
                env, rec, fd, cur, relocate(cur, b) or b,
                int(rng.choice(MOVE_ACTIONS)), min(n, max(1, budget - 1)))
            budget -= spent
            log.append({"probe": "switch",
                        "a_color_after": color_at_click(cur, a),
                        "b_color_after": color_at_click(cur, b),
                        "b_moved": moved})
            selected = b
        else:
            comp = pick_component(comps, rng)
            n = int(rng.integers(1, 4))
            fd, cur, prev, comp, moved, spent = click_and_move(
                env, rec, fd, cur, comp, int(rng.choice(MOVE_ACTIONS)),
                min(n, max(1, budget - 1)))
            budget -= spent
            log.append({"probe": "click_move", "color": int(comp["color"]),
                        "moved": moved})
            selected = comp
    return fd, cur, log


def level1_play(env, rec, fd, cur, rng, budget):
    """A short probe burst on the level-1 layout before the reach phase:
    unselected moves + clicks on whatever is there. Different layout,
    same mechanics question."""
    log = []
    fd, cur, moved, spent, _ = probe_moves(env, rec, fd, cur, rng, 2)
    budget -= spent
    log.append({"probe": "l1_move_unselected", "spent": spent, "moved": moved})
    while budget > 1 and game_live(fd):
        comps = frame_components(cur)
        if not comps:
            break
        comp = comps[int(rng.integers(len(comps)))]
        fd, cur, prev, comp, moved, spent = click_and_move(
            env, rec, fd, cur, comp, int(rng.choice(MOVE_ACTIONS)),
            min(int(rng.integers(1, 3)), max(1, budget - 1)))
        budget -= spent
        log.append({"probe": "l1_click_move", "color": int(comp["color"]),
                    "moved": moved})
    return fd, cur, log


def collect(run_dir: Path, episodes: int, seed: int, pour_prob: float,
            l1_budget: int, budget: int):
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
    n_reached = 0
    for ep in range(episodes):
        env = arcade.make(GAME_ID, seed=seed + ep)
        fd = env.reset()
        cur = np.asarray(fd.frame[-1] if isinstance(fd.frame, list)
                         else fd.frame, dtype=np.int8)
        prev = cur
        rec = EpisodeRecorder()
        catches: list[dict] = []

        log_l1: list[dict] = []
        if l1_budget > 0:
            fd, cur, log_l1 = level1_play(env, rec, fd, cur, rng, l1_budget)

        log_ctl: list[dict] = []
        reached = False
        if game_live(fd):
            fd, cur, prev, reached, _ = reach_next_level(
                model, device, env, rec, fd, cur, prev, rng)
            n_reached += int(reached)
        if reached and game_live(fd):
            fd, cur, log_ctl = control_play(env, rec, fd, cur, rng, budget)
            if game_live(fd) and rng.random() < pour_prob:
                fd, cur, record = pour_last(
                    env, rec, fd, cur, frame_components(cur)[:6])
                if record is not None:
                    catches.append(record)

        final_levels = int(getattr(fd, "levels_completed", 0) or 0) if fd else 0
        rec.steps.append({
            "frame": cur, "action_id": 0, "action_x": 0, "action_y": 0,
            "levels_completed": final_levels,
            "state": str(getattr(fd, "state", None)) if fd else "None",
        })
        trajectories.append({
            "game_id": GAME_ID,
            "rollout_id": f"control_seed{seed + ep}",
            "policy": "control_play",
            "steps": rec.steps,
            "catches": catches,
            "arrangement": log_l1 + log_ctl,
        })
        probes = [e["probe"] for e in log_l1 + log_ctl]
        print(f"[ctl] ep {ep}: steps={len(rec.steps)} reached={reached} "
              f"probes={probes} levels={final_levels} "
              f"state={str(getattr(fd, 'state', None)) if fd else 'None'}")
    print(f"[ctl] {n_reached}/{episodes} episodes reached the plumbing level")
    return trajectories


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=20300001)
    p.add_argument("--pour-prob", type=float, default=0.25,
                   help="probability of ending an episode with a pour "
                        "(extra level negatives on arbitrary states)")
    p.add_argument("--l1-budget", type=int, default=L1_BUDGET)
    p.add_argument("--budget", type=int, default=CONTROL_BUDGET)
    args = p.parse_args()
    trajectories = collect(args.run_dir, args.episodes, args.seed,
                           args.pour_prob, args.l1_budget, args.budget)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajectories, args.out)
    n = sum(len(t["steps"]) for t in trajectories)
    print(f"[ctl] wrote {len(trajectories)} episodes / {n} steps → {args.out}")


if __name__ == "__main__":
    main()
