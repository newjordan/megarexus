#!/usr/bin/env python3
"""Train the additive MegarexusCNN goal head on hindsight completion labels."""

from __future__ import annotations

import argparse
import html
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))

from dataset import GRID, NUM_COLORS, contra_planes  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402
from train import apply_trainable_prefixes, load_model_state_compatible, sha256_file  # noqa: E402
from orchestration.no_imitation_guard import assert_no_imitation_lineage  # noqa: E402


CHANNELS = ("target_lock", "key_block", "paint_output")
PALETTE = np.array(
    [
        [8, 10, 14],
        [230, 236, 244],
        [79, 124, 255],
        [255, 205, 84],
        [83, 215, 160],
        [246, 103, 103],
        [173, 123, 255],
        [255, 143, 65],
        [82, 190, 229],
        [216, 88, 167],
        [155, 170, 185],
        [40, 49, 66],
        [147, 222, 91],
        [252, 252, 164],
        [178, 108, 74],
        [119, 232, 224],
    ],
    dtype=np.uint8,
)


class GoalLabelDataset(Dataset):
    def __init__(self, dataset_path: Path, split: str, *, stack_frames: int = 2) -> None:
        payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
        assert_no_imitation_lineage(
            str(dataset_path),
            payload.get("manifest") if isinstance(payload, dict) else None,
            purpose="goal label dataset",
        )
        self.dataset_path = dataset_path
        self.channels = tuple(payload["channels"])
        if self.channels != CHANNELS:
            raise ValueError(f"unexpected channels: {self.channels}")
        self.samples = [s for s in payload["samples"] if s["split"] == split]
        self.stack_frames = int(stack_frames)
        if self.stack_frames not in (1, 2):
            raise ValueError("stack_frames must be 1 or 2")
        self._planes: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_grid(self, grid: np.ndarray, key: str) -> torch.Tensor:
        arr = np.asarray(grid, dtype=np.int64)
        onehot = torch.nn.functional.one_hot(torch.from_numpy(arr), NUM_COLORS).permute(2, 0, 1).float()
        if key not in self._planes:
            self._planes[key] = contra_planes(arr.astype(np.int8)).astype(np.float16)
        planes = torch.from_numpy(self._planes[key].astype(np.float32))
        return torch.cat([onehot, planes], dim=0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]
        cur = np.asarray(sample["frame"], dtype=np.int8)
        prev = np.asarray(sample["prev_frame"], dtype=np.int8)
        frame = self._encode_grid(cur, sample["pre_frame_sha256"])
        if self.stack_frames == 2:
            frame = torch.cat([frame, self._encode_grid(prev, sample["pre_frame_sha256"] + ":prev")], dim=0)
        return {
            "frame": frame,
            "goal_masks": torch.from_numpy(np.asarray(sample["goal_masks"], dtype=np.float32)),
            "sample_index": torch.tensor(idx, dtype=torch.long),
            "from_level": torch.tensor(int(sample["from_level"]), dtype=torch.long),
            "to_level": torch.tensor(int(sample["to_level"]), dtype=torch.long),
            "group_seen_in_train": torch.tensor(float(sample.get("group_seen_in_train", False))),
        }


def train_sample_weights(ds: GoalLabelDataset) -> torch.Tensor:
    counts = Counter(str(sample["group_key"]) for sample in ds.samples)
    return torch.tensor([1.0 / float(counts[str(sample["group_key"])]) for sample in ds.samples], dtype=torch.double)


def compute_pos_weight(ds: GoalLabelDataset, *, clamp: float = 200.0) -> torch.Tensor:
    pos = torch.zeros(3, dtype=torch.float64)
    total = 0
    for sample in ds.samples:
        masks = torch.from_numpy(np.asarray(sample["goal_masks"], dtype=np.float32))
        pos += masks.sum(dim=(1, 2)).double()
        total += int(masks.shape[1] * masks.shape[2])
    neg = float(total) - pos
    return torch.clamp((neg / pos.clamp(min=1.0)).float(), min=1.0, max=float(clamp))


def goal_loss(logits: torch.Tensor, masks: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    weight = pos_weight.view(1, 3, 1, 1).to(logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, masks, reduction="none")
    scale = torch.where(masks > 0, weight, torch.ones_like(loss))
    return (loss * scale).mean()


def mask_metrics(pred: torch.Tensor, true: torch.Tensor) -> dict[str, float]:
    pred_b = pred.bool()
    true_b = true.bool()
    inter = (pred_b & true_b).sum(dim=(0, 2, 3)).float()
    union = (pred_b | true_b).sum(dim=(0, 2, 3)).float()
    pred_sum = pred_b.sum(dim=(0, 2, 3)).float()
    true_sum = true_b.sum(dim=(0, 2, 3)).float()
    out: dict[str, float] = {}
    ious = []
    for idx, name in enumerate(CHANNELS):
        iou = float(inter[idx] / union[idx].clamp(min=1.0))
        precision = float(inter[idx] / pred_sum[idx].clamp(min=1.0))
        recall = float(inter[idx] / true_sum[idx].clamp(min=1.0))
        out[f"{name}/iou"] = iou
        out[f"{name}/precision"] = precision
        out[f"{name}/recall"] = recall
        out[f"{name}/pred_pixels"] = float(pred_sum[idx])
        out[f"{name}/true_pixels"] = float(true_sum[idx])
        ious.append(iou)
    out["mean_iou"] = float(sum(ious) / len(ious))
    return out


@torch.no_grad()
def evaluate_goal(
    model: MegarexusCNN,
    loader: DataLoader,
    device: str,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    model.eval()
    buckets: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
    losses = []
    pos_weight = torch.ones(3, device=device)
    for batch in loader:
        frame = batch["frame"].to(device)
        masks = batch["goal_masks"].to(device)
        logits = model.forward_goal(frame)
        losses.append(float(goal_loss(logits, masks, pos_weight).detach()))
        pred = (torch.sigmoid(logits) >= threshold).cpu()
        true = masks.cpu()
        buckets["all"].append((pred, true))
        seen = batch["group_seen_in_train"] > 0
        if seen.any():
            buckets["seen_group"].append((pred[seen], true[seen]))
        if (~seen).any():
            buckets["unseen_group"].append((pred[~seen], true[~seen]))
        from_level = batch["from_level"].cpu()
        to_level = batch["to_level"].cpu()
        for lvl in sorted(set(zip(from_level.tolist(), to_level.tolist()))):
            sel = (from_level == lvl[0]) & (to_level == lvl[1])
            if sel.any():
                buckets[f"level_{lvl[0]}_to_{lvl[1]}"].append((pred[sel], true[sel]))
    metrics: dict[str, Any] = {"loss": sum(losses) / max(1, len(losses))}
    for bucket, items in buckets.items():
        pred = torch.cat([p for p, _ in items], dim=0)
        true = torch.cat([t for _, t in items], dim=0)
        metrics[bucket] = mask_metrics(pred, true)
        metrics[bucket]["n"] = int(pred.shape[0])
    return metrics


def render_prediction_overlay(
    sample: dict[str, Any],
    pred: np.ndarray,
    path: Path,
    *,
    scale: int = 8,
) -> None:
    frame = np.asarray(sample["frame"], dtype=np.int16)
    true = np.asarray(sample["goal_masks"], dtype=np.uint8)
    arr = np.clip(frame, 0, len(PALETTE) - 1)
    rgb = PALETTE[arr].astype(np.float32)
    true_colors = [
        np.array([255, 47, 79], dtype=np.float32),
        np.array([255, 216, 74], dtype=np.float32),
        np.array([56, 217, 255], dtype=np.float32),
    ]
    pred_colors = [
        np.array([255, 155, 170], dtype=np.float32),
        np.array([255, 238, 145], dtype=np.float32),
        np.array([135, 235, 255], dtype=np.float32),
    ]
    for idx, color in enumerate(true_colors):
        mask = true[idx].astype(bool)
        rgb[mask] = 0.25 * rgb[mask] + 0.75 * color
    for idx, color in enumerate(pred_colors):
        mask = pred[idx].astype(bool) & ~true[idx].astype(bool)
        rgb[mask] = 0.20 * rgb[mask] + 0.80 * color
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    image = image.resize((frame.shape[1] * scale, frame.shape[0] * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), "true saturated; prediction-only pale", fill="#f0f4fb")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


@torch.no_grad()
def write_review(
    model: MegarexusCNN,
    val_ds: GoalLabelDataset,
    out_dir: Path,
    metrics: dict[str, Any],
    device: str,
    *,
    max_samples: int = 24,
    threshold: float = 0.5,
) -> None:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    cards = []
    for idx, sample in enumerate(val_ds.samples[:max_samples]):
        item = val_ds[idx]
        frame = item["frame"].unsqueeze(0).to(device)
        logits = model.forward_goal(frame)
        pred = (torch.sigmoid(logits)[0].cpu().numpy() >= threshold).astype(np.uint8)
        rel = Path("frames") / f"val_{idx:03d}_prediction_overlay.png"
        render_prediction_overlay(sample, pred, out_dir / rel)
        cards.append(
            f"""
            <article>
              <h3>{sample['from_level']} -&gt; {sample['to_level']}</h3>
              <img src="{html.escape(str(rel))}" alt="prediction overlay">
              <p><strong>seen group:</strong> {html.escape(str(sample.get('group_seen_in_train')))}</p>
              <p><strong>source:</strong> <code>{html.escape(sample['source_path'])}</code></p>
            </article>
            """
        )
    metric_rows = []
    for bucket, row in metrics.items():
        if not isinstance(row, dict):
            continue
        metric_rows.append(
            "<tr>"
            f"<th>{html.escape(bucket)}</th>"
            f"<td><code>{html.escape(json.dumps(row, sort_keys=True))}</code></td>"
            "</tr>"
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Goal Head Evaluation Review</title>
  <style>
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #111318; color: #e8edf4; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 28px; }}
    h1, h2, h3 {{ letter-spacing: 0; }}
    .boundary {{ color: #f4d278; max-width: 980px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0 26px; }}
    th, td {{ border: 1px solid #2d3440; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ width: 220px; background: #1d2430; }}
    td {{ background: #151920; }}
    .deck {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 14px; }}
    article {{ border: 1px solid #2d3440; border-radius: 6px; padding: 12px; background: #171b22; }}
    article img {{ width: 100%; image-rendering: pixelated; background: #080a0e; }}
    code {{ color: #d7e6ff; word-break: break-word; }}
    a {{ color: #8fc7ff; }}
  </style>
</head>
<body>
<main>
  <h1>Goal Head Evaluation Review</h1>
  <p class="boundary">CNN goal-head evaluation on hindsight completion labels. Not solve evidence and not runtime mastery unless heldout metrics and live wiring gates are separately passed.</p>
  <p>Raw metrics: <a href="goal_head_eval.json">goal_head_eval.json</a>. History: <a href="history.json">history.json</a>.</p>
  <h2>Metrics</h2>
  <table>{''.join(metric_rows)}</table>
  <h2>Validation Overlays</h2>
  <div class="deck">{''.join(cards)}</div>
</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def serializable_config(args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--goal-dataset", type=Path, required=True)
    p.add_argument("--init-ckpt", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stack-frames", type=int, default=2, choices=(1, 2))
    p.add_argument("--trainable-prefixes", nargs="*", default=["goal_conv", "head_goal"])
    p.add_argument("--no-balance-groups", action="store_true")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args(argv)

    assert_no_imitation_lineage(
        str(args.goal_dataset),
        str(args.init_ckpt),
        purpose="goal-head training inputs",
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[goal-train] device={device}")
    print(f"[goal-train] goal_dataset={args.goal_dataset} sha256={sha256_file(args.goal_dataset)}")
    train_ds = GoalLabelDataset(args.goal_dataset, "train", stack_frames=args.stack_frames)
    val_ds = GoalLabelDataset(args.goal_dataset, "val", stack_frames=args.stack_frames)
    print(f"[goal-train] train={len(train_ds)} val={len(val_ds)} channels={CHANNELS}")
    sampler = None
    shuffle = True
    if not args.no_balance_groups:
        sampler = WeightedRandomSampler(train_sample_weights(train_ds), num_samples=len(train_ds), replacement=True)
        shuffle = False
        print("[goal-train] sampler=group_balanced")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)
    pos_weight = compute_pos_weight(train_ds).to(device)
    print(f"[goal-train] pos_weight={pos_weight.cpu().tolist()}")

    model = MegarexusCNN(MegarexusCNNConfig(n_input_frames=args.stack_frames)).to(device)
    payload = torch.load(args.init_ckpt, map_location=device, weights_only=False)
    assert_no_imitation_lineage(
        payload.get("config") if isinstance(payload, dict) else None,
        payload.get("manifest") if isinstance(payload, dict) else None,
        purpose="goal-head init checkpoint metadata",
    )
    init_stack = int(payload["config"].get("stack_frames", args.stack_frames))
    if init_stack != args.stack_frames:
        raise ValueError(f"init_ckpt stack_frames={init_stack} != --stack-frames={args.stack_frames}")
    load_model_state_compatible(model, payload["model"])
    print(f"[goal-train] init_ckpt={args.init_ckpt} sha256={sha256_file(args.init_ckpt)}")
    n_trainable, n_total = apply_trainable_prefixes(model, args.trainable_prefixes)
    if n_trainable <= 0:
        raise ValueError(f"no trainable params for prefixes={args.trainable_prefixes}")
    print(f"[goal-train] trainable_prefixes={args.trainable_prefixes} trainable={n_trainable}/{n_total}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_mean_iou = -1.0
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            frame = batch["frame"].to(device)
            masks = batch["goal_masks"].to(device)
            logits = model.forward_goal(frame)
            loss = goal_loss(logits, masks, pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.detach())
            n_batches += 1
        sched.step()
        metrics = evaluate_goal(model, val_loader, device, threshold=args.threshold)
        mean_iou = float(metrics["all"]["mean_iou"])
        print(
            f"[epoch {epoch + 1:03d}] loss={total_loss / max(1, n_batches):.4f} "
            f"val_mean_iou={mean_iou:.4f} "
            f"target_iou={metrics['all']['target_lock/iou']:.4f} "
            f"key_iou={metrics['all']['key_block/iou']:.4f} "
        f"output_iou={metrics['all']['paint_output/iou']:.4f} "
            f"unseen_mean_iou={metrics.get('unseen_group', {}).get('mean_iou', 0.0):.4f}"
        )
        row = {
            "epoch": epoch + 1,
            "train_loss": total_loss / max(1, n_batches),
            "val": metrics,
        }
        history.append(row)
        ckpt = {
            "model": model.state_dict(),
            "config": serializable_config(args),
            "val": metrics,
            "channels": CHANNELS,
        }
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            torch.save(ckpt, args.out_dir / "best.pt")
        torch.save(ckpt, args.out_dir / "last.pt")
    final_metrics = evaluate_goal(model, val_loader, device, threshold=args.threshold)
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (args.out_dir / "goal_head_eval.json").write_text(
        json.dumps(
            {
                "schema": "synthesis_lab.megarexus.goal_head_eval.v1",
                "goal_dataset": str(args.goal_dataset),
                "goal_dataset_sha256": sha256_file(args.goal_dataset),
                "init_ckpt": str(args.init_ckpt),
                "init_ckpt_sha256": sha256_file(args.init_ckpt),
                "config": serializable_config(args),
                "channels": CHANNELS,
                "metrics": final_metrics,
                "best_mean_iou": best_mean_iou,
                "elapsed_seconds": round(time.time() - t0, 3),
                "boundary": (
                    "Goal-head label evaluation only. Not solve evidence, not runtime "
                    "wiring evidence, and not mastery without heldout/live gates."
                ),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    write_review(model, val_ds, args.out_dir, final_metrics, device, threshold=args.threshold)
    print(f"[goal-train] done in {time.time() - t0:.0f}s; best_mean_iou={best_mean_iou:.4f}; artifacts -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
