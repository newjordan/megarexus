"""Ghost-overlay confirmation pass for the trained MegarexusCNN.

The ghost role (per codexlab/contra_head/ghost.py): record relationship
evidence, never action authority. Here the trained CNN plays the ghost's
predictive half — it imagines the paint's trajectory H steps ahead on
held-out flow segments, and the overlay compares the imagined frames
against what actually happened. Sustained agreement on flow cells is the
"confirmed understanding of the paint mechanic" signal; disagreement
refutes it. Truth-boundary flags stay False: this confirms a *mechanic
hypothesis*, it admits no goal/solve/control truth.

Also runs Contra's lock/key asymmetry goal seeker
(``antisymmetric_relationship_hypotheses`` via
``analyze_scene_surface_hypotheses``) over the same held-out frames from
BOTH games, reporting how often it fires — the cross-game goal-seeking
signal, kept as informer output only.

Outputs (under --run-dir/ghost/):
- ghost_overlay_report.json — per-horizon metrics + ghost packet
- overlay_XX.png — [actual | predicted | agreement] panels per step
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))

from dataset import FLOW_CHANGED_CELLS, PALETTE, MegarexusTransitions  # noqa: E402
from engine_adapters.contra_perception_adapter import contra_planes  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402

GRID = 64
IOU_CONFIRM_THRESHOLD = 0.5


def encode(grid: np.ndarray) -> torch.Tensor:
    onehot = torch.nn.functional.one_hot(
        torch.from_numpy(grid.astype(np.int64)), 16
    ).permute(2, 0, 1).float()
    planes = torch.from_numpy(contra_planes(grid))
    return torch.cat([onehot, planes], dim=0)


def build_flow_windows(ds: MegarexusTransitions, horizon: int) -> list[list]:
    """Chains of `horizon` consecutive val transitions starting at a flow step."""
    windows = []
    i = 0
    while i < len(ds.transitions):
        t0 = ds.transitions[i]
        cur, nxt = ds.grids[t0.frame_key], ds.grids[t0.next_frame_key]
        if int((cur != nxt).sum()) >= FLOW_CHANGED_CELLS:
            chain = [t0]
            j = i
            while (
                len(chain) < horizon
                and j + 1 < len(ds.transitions)
                and ds.transitions[j + 1].frame_key == chain[-1].next_frame_key
            ):
                j += 1
                chain.append(ds.transitions[j])
            if len(chain) == horizon:
                windows.append(chain)
                i = j + 1
                continue
        i += 1
    return windows


def render_panel(actual: np.ndarray, predicted: np.ndarray, prev: np.ndarray,
                 scale: int = 6) -> Image.Image:
    """[actual | predicted | agreement]: green=changed cell predicted right,
    red=wrong cell, gray=unchanged cell predicted right."""
    a_rgb = PALETTE[actual]
    p_rgb = PALETTE[predicted]
    agree = np.zeros((GRID, GRID, 3), dtype=np.uint8)
    changed = actual != prev
    right = predicted == actual
    agree[right & ~changed] = (70, 70, 70)
    agree[right & changed] = (60, 200, 60)
    agree[~right] = (220, 50, 50)
    strip = np.concatenate([a_rgb, p_rgb, agree], axis=1)
    img = Image.fromarray(strip).resize(
        (strip.shape[1] * scale, strip.shape[0] * scale), Image.NEAREST
    )
    return img


@torch.no_grad()
def ghost_rollout(model, window, ds, device, save_prefix=None):
    """Autoregressive H-step prediction; returns per-step agreement stats."""
    stack_prev = ds.grids[window[0].prev_frame_key or window[0].frame_key]
    grid = ds.grids[window[0].frame_key]
    stats = []
    for h, t in enumerate(window):
        actual_next = ds.grids[t.next_frame_key]
        frame = torch.cat([encode(grid), encode(stack_prev)], dim=0)
        if model.config.n_input_frames == 1:
            frame = frame[: model.config.in_channels]
        logits = model.forward_dynamics(
            frame.unsqueeze(0).to(device),
            torch.tensor([t.action_id], device=device),
            torch.tensor([t.x], device=device),
            torch.tensor([t.y], device=device),
        )
        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int8)

        actual_changed = actual_next != grid
        pred_changed = pred != grid
        inter = int((actual_changed & pred_changed).sum())
        union = int((actual_changed | pred_changed).sum())
        n_changed = int(actual_changed.sum())
        stats.append({
            "horizon": h + 1,
            "action_id": t.action_id,
            "changed_cells": n_changed,
            "changed_cell_acc": float((pred[actual_changed] == actual_next[actual_changed]).mean())
            if n_changed else 1.0,
            "cell_acc": float((pred == actual_next).mean()),
            "persistence_cell_acc": float((grid == actual_next).mean()),
            "change_iou": inter / union if union else 1.0,
        })
        if save_prefix is not None:
            render_panel(actual_next, pred, grid).save(f"{save_prefix}_h{h + 1}.png")
        # feed the model's own prediction forward (ghost imagines, then checks)
        stack_prev, grid = grid, pred
    return stats


def lockkey_probe(grids: list[np.ndarray], label: str) -> dict:
    sys.path.insert(0, str(_HERE.parents[2] / "codexlab"))
    from contra_head.vision import analyze_scene_surface_hypotheses

    fired, examples = 0, []
    for g in grids:
        packet, _label_map, _lookup = analyze_scene_surface_hypotheses(
            np.asarray(g, dtype=np.int16)
        )
        hyps = packet.get("antisymmetric_relationship_hypotheses") or []
        if hyps:
            fired += 1
            best = hyps[0]
            examples.append({
                "score": best["score"], "axis": best["axis"],
                "color": best["raw_pixel_value"],
                "facing": [best["facing_direction_a"], best["facing_direction_b"]],
            })
    return {
        "game": label,
        "frames_probed": len(grids),
        "frames_with_lockkey_hypothesis": fired,
        "examples": examples[:5],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--max-windows", type=int, default=40)
    p.add_argument("--n-overlays", type=int, default=3)
    p.add_argument("--rollout-val-shard", type=Path, default=None,
                   help="evaluate on this fresh rollout shard (clean flow physics, "
                        "unseen seeds) instead of the human corpus val split")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.run_dir / "best.pt", map_location=device, weights_only=False)
    stack = ckpt["config"].get("stack_frames", 1)
    model = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)).to(device).eval()
    model.load_state_dict(ckpt["model"])

    if args.rollout_val_shard is not None:
        ds = MegarexusTransitions("train", rollout_shards=[args.rollout_val_shard],
                                  include_corpus=False, n_input_frames=stack)
        source = f"fresh rollout shard {args.rollout_val_shard.name}"
    else:
        ds = MegarexusTransitions("val", n_input_frames=stack)
        source = "human corpus val split"
    windows = build_flow_windows(ds, args.horizon)[: args.max_windows]
    print(f"[ghost] {len(windows)} flow windows of horizon {args.horizon} ({source})")

    out = args.run_dir / ("ghost_rollout" if args.rollout_val_shard else "ghost")
    out.mkdir(exist_ok=True)
    all_stats = []
    for w_i, window in enumerate(windows):
        prefix = str(out / f"overlay_{w_i:02d}") if w_i < args.n_overlays else None
        all_stats.append(ghost_rollout(model, window, ds, device, save_prefix=prefix))

    per_h = []
    for h in range(args.horizon):
        rows = [s[h] for s in all_stats]
        per_h.append({
            "horizon": h + 1,
            "mean_changed_cell_acc": float(np.mean([r["changed_cell_acc"] for r in rows])),
            "mean_cell_acc": float(np.mean([r["cell_acc"] for r in rows])),
            "mean_persistence_cell_acc": float(np.mean([r["persistence_cell_acc"] for r in rows])),
            "mean_change_iou": float(np.mean([r["change_iou"] for r in rows])),
        })
        print(f"[ghost] h={h + 1}: changed_acc={per_h[-1]['mean_changed_cell_acc']:.3f} "
              f"iou={per_h[-1]['mean_change_iou']:.3f} "
              f"cell={per_h[-1]['mean_cell_acc']:.4f} "
              f"(persist={per_h[-1]['mean_persistence_cell_acc']:.4f})")

    step1 = [s[0]["change_iou"] for s in all_stats]
    confirmed = sum(1 for v in step1 if v >= IOU_CONFIRM_THRESHOLD)

    # lock/key asymmetry goal seeker across both games' held-out frames
    sp80_grids = [ds.grids[k] for k in list(ds.grids) if "sp80" in k][:40]
    snake_grids = [ds.grids[k] for k in list(ds.grids) if "s5i5" in k][:40]
    lockkey = [lockkey_probe(sp80_grids, "sp80"), lockkey_probe(snake_grids, "s5i5")]
    for row in lockkey:
        print(f"[lockkey] {row['game']}: {row['frames_with_lockkey_hypothesis']}"
              f"/{row['frames_probed']} frames fired")

    report = {
        "schema": "synthesis_lab.megarexus_ghost_overlay.v0",
        "mechanic_hypothesis": "sp80_paint_flow_catch_and_redirect",
        "eval_source": source,
        "evidence_status": "hypothesis",
        "control_truth_admitted": False,
        "goal_truth_admitted": False,
        "solve_truth_admitted": False,
        "action_choice_admitted": False,
        "windows": len(all_stats),
        "step1_confirmed": confirmed,
        "step1_confirm_rate": confirmed / max(len(all_stats), 1),
        "iou_confirm_threshold": IOU_CONFIRM_THRESHOLD,
        "per_horizon": per_h,
        "lockkey_goal_seeker": lockkey,
    }
    (out / "ghost_overlay_report.json").write_text(json.dumps(report, indent=2))
    print(f"[ghost] report → {out / 'ghost_overlay_report.json'}")


if __name__ == "__main__":
    main()
