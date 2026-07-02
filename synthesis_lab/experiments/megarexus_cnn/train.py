"""SCOUT_EXPERIMENT: train MegarexusCNN on real ARC-AGI-3 transitions.

Objectives (all on real data, no synthetic answers):

- forward dynamics (LEWM_REX_ACTION_OWNER): per-cell CE of next-frame
  colors, with changed cells up-weighted so the model can't win by copying
  the input. Honesty check: validation reports the persistence baseline
  (predict next = current) alongside the model.
- Goose change objective (STOCHASTIC_GOOSE_RUNTIME): BCE on the taken
  action's change_logit vs. whether the frame actually changed — Goose's
  own self-supervised loss, ported verbatim.
- behavior cloning (human corpus only): CE of policy_logits vs. the human
  action index in the 4101-dim Goose head.
- inverse dynamics (diagnostic): CE of action id from (s_t, s_{t+1}),
  plus click-cell CE when the action was ACTION6.

Baselines reported on the SAME validation data:
- persistence per-cell accuracy (forward dynamics floor)
- majority-action accuracy (inverse dynamics / BC floor)
- change base rate (change-head floor)

Usage:
    python synthesis_lab/experiments/megarexus_cnn/train.py \
        [--epochs 40] [--batch-size 64] [--rollout-shards path.pt ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))

from dataset import GRID, MegarexusTransitions  # noqa: E402
from model import (  # noqa: E402
    ARC_CLICK_ACTION_ID,
    MegarexusCNN,
    MegarexusCNNConfig,
)

CHANGED_CELL_WEIGHT = 20.0


def dynamics_loss(logits: torch.Tensor, target: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
    """Per-cell CE with changed cells up-weighted (else persistence dominates)."""
    cur = frame[:, :16].argmax(dim=1)
    weights = torch.where(target != cur, CHANGED_CELL_WEIGHT, 1.0)
    ce = F.cross_entropy(logits, target, reduction="none")
    return (ce * weights).sum() / weights.sum()


SETTLE_POS_WEIGHT = 20.0  # settle masks are sparse (~1% of cells)
LEVEL_POS_WEIGHT = 10.0  # level-up pours are ~2% of pours


def batch_losses(model: MegarexusCNN, batch: dict, device: str) -> dict[str, torch.Tensor]:
    b = {k: v.to(device) for k, v in batch.items()}
    # control rows are auxiliary duplicates of their move transition — they
    # carry ONLY the controllability objective (else moves get over-weighted)
    main = b["has_control"] == 0
    losses: dict[str, torch.Tensor] = {}

    if main.any():
        m = {k: v[main] for k, v in b.items()}
        out = model(m["frame"], m["action_id"], m["x"], m["y"], next_frame=m["next_frame"])
        # level-completing pours keep their labels but their next_frame is a
        # NEW layout — mask every next-frame-dependent objective on them
        dv = m["dyn_valid"] > 0
        if dv.any():
            losses["dyn"] = dynamics_loss(
                out["next_frame_logits"][dv], m["target"][dv], m["frame"][dv])

        # Goose change objective on the taken action's logit
        has_goose = (m["goose_idx"] >= 0) & dv
        if has_goose.any():
            idx = m["goose_idx"].clamp(min=0)
            taken_logit = out["change_logits"].gather(1, idx.unsqueeze(1)).squeeze(1)
            losses["change"] = F.binary_cross_entropy_with_logits(
                taken_logit[has_goose], m["changed"][has_goose]
            )
            # behavior cloning on human actions only
            is_bc = has_goose & (m["is_human"] > 0)
            if is_bc.any():
                losses["bc"] = F.cross_entropy(out["policy_logits"][is_bc], idx[is_bc])

        # PORTBAND catch-play: will the pour touch the glove?
        has_catch = m["has_catch"] > 0
        if has_catch.any():
            catch_logit = model.forward_catch(m["frame"][has_catch],
                                              m["glove_plane"][has_catch])
            losses["catch"] = F.binary_cross_entropy_with_logits(
                catch_logit, m["catch"][has_catch]
            )

        # learned outcome semantics: settle mask (pixels) + level-up (env fields)
        has_settle = m["has_settle"] > 0
        has_level = m["has_level"] > 0
        if has_level.any():
            settle_logits, level_logit = model.forward_settle(m["frame"][has_level])
            losses["level"] = F.binary_cross_entropy_with_logits(
                level_logit, m["level_up"][has_level],
                pos_weight=torch.tensor(LEVEL_POS_WEIGHT, device=device),
            )
            sub_settle = has_settle[has_level]
            if sub_settle.any():
                losses["settle"] = F.binary_cross_entropy_with_logits(
                    settle_logits[sub_settle],
                    m["settle_mask"][has_level][sub_settle],
                    pos_weight=torch.tensor(SETTLE_POS_WEIGHT, device=device),
                )

        # inverse dynamics (needs a same-layout next frame)
        if dv.any():
            losses["inv"] = F.cross_entropy(
                out["inv_action_logits"][dv], m["action_id"][dv])
        is_click = (m["action_id"] == ARC_CLICK_ACTION_ID) & dv
        if is_click.any():
            cell = (m["y"] * GRID + m["x"])[is_click]
            losses["inv_click"] = F.cross_entropy(out["inv_click_logits"][is_click], cell)

    # controllability: does this component displace with this move?
    has_control = b["has_control"] > 0
    if has_control.any():
        control_logit = model.forward_control(
            b["frame"][has_control],
            b["control_plane"][has_control],
            b["action_id"][has_control],
        )
        losses["control"] = F.binary_cross_entropy_with_logits(
            control_logit, b["controlled"][has_control]
        )
    return losses


@torch.no_grad()
def evaluate(model: MegarexusCNN, loader: DataLoader, device: str) -> dict[str, float]:
    model.eval()
    cells_total = cells_correct = 0
    persist_correct = 0
    changed_total = changed_correct = 0
    flow_changed_total = flow_changed_correct = 0
    inv_total = inv_correct = 0
    click_total = click_hits = 0
    click_dist_sum = 0.0
    bc_total = bc_action_correct = 0
    change_total = change_correct = 0
    catch_total = catch_correct = 0
    catch_labels = []
    level_total = level_correct = level_tp = level_fp = 0
    level_labels = []
    settle_inter = settle_union = 0
    settle_n = 0
    control_total = control_correct = 0
    control_labels = []
    action_counter: Counter = Counter()
    changed_flags = []

    for batch in loader:
        full = {k: v.to(device) for k, v in batch.items()}
        has_control = full["has_control"] > 0
        if has_control.any():
            logit = model.forward_control(
                full["frame"][has_control],
                full["control_plane"][has_control],
                full["action_id"][has_control],
            )
            pred_c = (logit > 0).float()
            control_total += int(has_control.sum())
            control_correct += int((pred_c == full["controlled"][has_control]).sum())
            control_labels += full["controlled"][has_control].tolist()
        main = has_control == 0
        if not main.any():
            continue
        b = {k: v[main] for k, v in full.items()}
        out = model(b["frame"], b["action_id"], b["x"], b["y"], next_frame=b["next_frame"])
        # next-frame-dependent metrics only on same-layout rows
        dv = b["dyn_valid"] > 0
        pred = out["next_frame_logits"][dv].argmax(dim=1)
        cur = b["frame"][dv, :16].argmax(dim=1)
        tgt = b["target"][dv]

        cells_total += tgt.numel()
        cells_correct += int((pred == tgt).sum())
        persist_correct += int((cur == tgt).sum())
        ch = tgt != cur
        changed_total += int(ch.sum())
        changed_correct += int((pred[ch] == tgt[ch]).sum())
        is_flow = b["flow"][dv] > 0
        if is_flow.any():
            fch = ch & is_flow.view(-1, 1, 1)
            flow_changed_total += int(fch.sum())
            flow_changed_correct += int((pred[fch] == tgt[fch]).sum())

        inv_pred = out["inv_action_logits"][dv].argmax(dim=1)
        inv_total += len(inv_pred)
        inv_correct += int((inv_pred == b["action_id"][dv]).sum())
        action_counter.update(b["action_id"][dv].tolist())

        is_click = (b["action_id"] == ARC_CLICK_ACTION_ID) & dv
        if is_click.any():
            cell_pred = out["inv_click_logits"][is_click].argmax(dim=1)
            cy, cx = cell_pred // GRID, cell_pred % GRID
            ty, tx = b["y"][is_click], b["x"][is_click]
            click_total += int(is_click.sum())
            click_hits += int(((cy == ty) & (cx == tx)).sum())
            click_dist_sum += float(
                torch.sqrt(((cy - ty).float() ** 2 + (cx - tx).float() ** 2)).sum()
            )

        has_catch = b["has_catch"] > 0
        if has_catch.any():
            catch_logit = model.forward_catch(b["frame"][has_catch],
                                              b["glove_plane"][has_catch])
            catch_pred = (catch_logit > 0).float()
            catch_total += int(has_catch.sum())
            catch_correct += int((catch_pred == b["catch"][has_catch]).sum())
            catch_labels += b["catch"][has_catch].tolist()

        has_level = b["has_level"] > 0
        if has_level.any():
            settle_logits, level_logit = model.forward_settle(b["frame"][has_level])
            level_pred = (level_logit > 0).float()
            level_true = b["level_up"][has_level]
            level_total += int(has_level.sum())
            level_correct += int((level_pred == level_true).sum())
            level_tp += int(((level_pred > 0) & (level_true > 0)).sum())
            level_fp += int(((level_pred > 0) & (level_true == 0)).sum())
            level_labels += level_true.tolist()
            sub_settle = (b["has_settle"] > 0)[has_level]
            if sub_settle.any():
                pred_mask = settle_logits[sub_settle] > 0
                true_mask = b["settle_mask"][has_level][sub_settle] > 0
                settle_inter += int((pred_mask & true_mask).sum())
                settle_union += int((pred_mask | true_mask).sum())
                settle_n += int(sub_settle.sum())

        has_goose = (b["goose_idx"] >= 0) & (b["dyn_valid"] > 0)
        idx = b["goose_idx"].clamp(min=0)
        if has_goose.any():
            taken = out["change_logits"].gather(1, idx.unsqueeze(1)).squeeze(1)
            change_pred = (taken[has_goose] > 0).float()
            change_total += int(has_goose.sum())
            change_correct += int((change_pred == b["changed"][has_goose]).sum())
            changed_flags += b["changed"][has_goose].tolist()

            is_bc = has_goose & (b["is_human"] > 0)
            if is_bc.any():
                # action-TYPE accuracy: collapse 4096 click cells to one type
                pol = out["policy_logits"][is_bc]
                pred_idx = pol.argmax(dim=1)
                pred_type = torch.where(pred_idx < 5, pred_idx + 1,
                                        torch.full_like(pred_idx, ARC_CLICK_ACTION_ID))
                bc_total += int(is_bc.sum())
                bc_action_correct += int((pred_type == b["action_id"][is_bc]).sum())

    majority = action_counter.most_common(1)[0][1] / max(inv_total, 1)
    change_rate = sum(changed_flags) / max(len(changed_flags), 1)
    catch_rate = sum(catch_labels) / max(len(catch_labels), 1)
    level_rate = sum(level_labels) / max(len(level_labels), 1)
    control_rate = sum(control_labels) / max(len(control_labels), 1)
    return {
        "catch/acc": catch_correct / max(catch_total, 1),
        "catch/base_rate": max(catch_rate, 1 - catch_rate),
        "catch/n": catch_total,
        "level/acc": level_correct / max(level_total, 1),
        "level/base_rate": max(level_rate, 1 - level_rate),
        "level/n": level_total,
        "level/positives": int(sum(level_labels)),
        "level/true_pos": level_tp,
        "level/false_pos": level_fp,
        "settle/iou": settle_inter / max(settle_union, 1),
        "settle/n": settle_n,
        "control/acc": control_correct / max(control_total, 1),
        "control/base_rate": max(control_rate, 1 - control_rate),
        "control/n": control_total,
        "fwd/cell_acc": cells_correct / cells_total,
        "fwd/persistence_baseline": persist_correct / cells_total,
        "fwd/changed_cell_acc": changed_correct / max(changed_total, 1),
        "fwd/flow_changed_cell_acc": flow_changed_correct / max(flow_changed_total, 1),
        "fwd/changed_cells": changed_total,
        "fwd/flow_changed_cells": flow_changed_total,
        "inv/action_acc": inv_correct / max(inv_total, 1),
        "inv/majority_baseline": majority,
        "inv/click_cell_acc": click_hits / max(click_total, 1),
        "inv/click_mean_dist_px": click_dist_sum / max(click_total, 1),
        "goose/change_acc": change_correct / max(change_total, 1),
        "goose/change_base_rate": max(change_rate, 1 - change_rate),
        "bc/action_type_acc": bc_action_correct / max(bc_total, 1),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--rollout-shards", type=Path, nargs="*", default=None)
    p.add_argument("--val-rollout-shards", type=Path, nargs="*", default=None,
                   help="unseen-seed shards evaluated as val (e.g. catch-play val)")
    p.add_argument("--stack-frames", type=int, default=1, choices=(1, 2),
                   help="2 = stack the previous frame so paint velocity is visible")
    p.add_argument("--out-dir", type=Path, default=_HERE / "runs" / "scout_v0")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device}")

    train_ds = MegarexusTransitions("train", rollout_shards=args.rollout_shards,
                                    n_input_frames=args.stack_frames)
    val_ds = MegarexusTransitions("val", val_rollout_shards=args.val_rollout_shards,
                                  n_input_frames=args.stack_frames)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    model = MegarexusCNN(MegarexusCNNConfig(n_input_frames=args.stack_frames)).to(device)
    n_params = sum(t.numel() for t in model.parameters())
    print(f"[train] MegarexusCNN params: {n_params / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_changed_acc = -1.0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        sums: Counter = Counter()
        n_batches = 0
        for batch in train_loader:
            losses = batch_losses(model, batch, device)
            loss = sum(losses.values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k, v in losses.items():
                sums[k] += float(v.detach())
            n_batches += 1
        sched.step()

        metrics = evaluate(model, val_loader, device)
        train_str = " ".join(f"{k}={sums[k] / n_batches:.4f}" for k in sorted(sums))
        print(f"[epoch {epoch + 1:03d}] {train_str}")
        print(
            f"          val: cell={metrics['fwd/cell_acc']:.4f} "
            f"(persist={metrics['fwd/persistence_baseline']:.4f}) "
            f"changed={metrics['fwd/changed_cell_acc']:.3f} "
            f"flow={metrics['fwd/flow_changed_cell_acc']:.3f} "
            f"inv={metrics['inv/action_acc']:.3f} "
            f"(maj={metrics['inv/majority_baseline']:.3f}) "
            f"click_d={metrics['inv/click_mean_dist_px']:.1f}px "
            f"chg={metrics['goose/change_acc']:.3f} "
            f"bc={metrics['bc/action_type_acc']:.3f} "
            f"catch={metrics['catch/acc']:.3f}"
            f"(base={metrics['catch/base_rate']:.3f},n={metrics['catch/n']}) "
            f"level={metrics['level/acc']:.3f}"
            f"(base={metrics['level/base_rate']:.3f},"
            f"tp={metrics['level/true_pos']}/{metrics['level/positives']},"
            f"fp={metrics['level/false_pos']}) "
            f"settle_iou={metrics['settle/iou']:.3f} "
            f"ctl={metrics['control/acc']:.3f}"
            f"(base={metrics['control/base_rate']:.3f},n={metrics['control/n']})"
        )
        history.append({"epoch": epoch + 1,
                        "train": {k: sums[k] / n_batches for k in sums},
                        "val": metrics})
        if metrics["fwd/changed_cell_acc"] > best_changed_acc:
            best_changed_acc = metrics["fwd/changed_cell_acc"]
            torch.save(
                {"model": model.state_dict(), "config": vars(args), "val": metrics},
                args.out_dir / "best.pt",
            )
        # best.pt selects on world-model quality alone and can freeze on an
        # early epoch while the heads keep improving (outcome_v5: best.pt was
        # epoch 12; epoch 40 dominated every head). Always keep the final
        # weights too — release choice belongs to the external no-loss gate.
        torch.save(
            {"model": model.state_dict(), "config": vars(args), "val": metrics},
            args.out_dir / "last.pt",
        )

    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[train] done in {time.time() - t0:.0f}s; best changed-cell acc "
          f"{best_changed_acc:.3f}; artifacts → {args.out_dir}")


if __name__ == "__main__":
    main()
