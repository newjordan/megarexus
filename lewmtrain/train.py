#!/usr/bin/env python3
"""LeWMTrain: Train a vanilla LeWM on human + random ARC-AGI-3 trajectories.

Loads pre-built trajectory shards from ``lewmtrain/data/`` and runs
paper-faithful LeWM training (L_pred + SIGReg). This is the validation
step — if the loss descends on human-solved trajectories, the data pipeline
is working and the world model is learning ARC dynamics.

Usage:
  # Smoke test (2 epochs, CPU, small batch):
  python lewmtrain/train.py --smoke

  # Full train (50 epochs, GPU):
  python lewmtrain/train.py --device cuda --num-epochs 50 --batch-size 128 --use-bf16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, ConcatDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lewm.data.arc_agi_3 import (
    ARCActionEmbedder,
    ARCTrajectoryDataset,
    load_trajectories,
    make_arc_encoder_config,
)
from lewm.loss import PAPER_REG_LAMBDA, lewm_loss
from lewm.model import LeWM, LeWMConfig
from lewm.predictor import PredictorConfig

DATA_DIR = ROOT / "lewmtrain" / "data"


def build_datasets(
    human_train: Path,
    human_val: Path,
    random_train: Path | None,
    random_val: Path | None,
    sequence_len: int,
) -> tuple[ARCTrajectoryDataset, ARCTrajectoryDataset]:
    """Load shards and build train/val ARCTrajectoryDataset instances."""

    def _load(path: Path) -> list:
        if not path.exists():
            print(f"  [warn] {path} not found, skipping")
            return []
        traj = load_trajectories(path)
        print(f"  Loaded {len(traj)} trajectories from {path.name}")
        return traj

    train_traj = _load(human_train)
    val_traj = _load(human_val)

    if random_train:
        train_traj += _load(random_train)
    if random_val:
        val_traj += _load(random_val)

    print(f"  Train: {len(train_traj)} trajectories")
    print(f"  Val:   {len(val_traj)} trajectories")

    train_ds = ARCTrajectoryDataset(train_traj, sequence_len=sequence_len)
    val_ds = ARCTrajectoryDataset(val_traj, sequence_len=sequence_len)

    print(f"  Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")
    return train_ds, val_ds


def train_epoch(
    model: LeWM,
    embedder: ARCActionEmbedder,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_bf16: bool,
    grad_clip: float,
    sigreg,
) -> dict[str, float]:
    """Run one training epoch. Returns average loss components."""
    model.train()
    embedder.train()

    total_pred = 0.0
    total_sigreg = 0.0
    total_loss = 0.0
    n_batches = 0

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )

    for batch in dataloader:
        obs = batch["obs"].to(device)
        action_emb = embedder(
            batch["action_id"].to(device),
            batch["action_x"].to(device),
            batch["action_y"].to(device),
            batch["action_complex_mask"].to(device),
        )

        with autocast_ctx:
            output = model(obs, action_emb)
            emb = output["emb"]
            next_emb = output["next_emb"]

            # Paper: pred_loss = MSE(emb[:,1:], next_emb[:,:-1])
            pred_loss = torch.nn.functional.mse_loss(
                emb[:, 1:], next_emb[:, :-1]
            )

            # SIGReg on flattened [B*T, D]
            sigreg_loss = sigreg(emb.reshape(-1, emb.shape[-1]))

            loss = pred_loss + PAPER_REG_LAMBDA * sigreg_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(embedder.parameters()),
                grad_clip,
            )
        optimizer.step()

        total_pred += pred_loss.item()
        total_sigreg += sigreg_loss.item()
        total_loss += loss.item()
        n_batches += 1

    return {
        "pred_loss": total_pred / n_batches,
        "sigreg_loss": total_sigreg / n_batches,
        "total_loss": total_loss / n_batches,
    }


@torch.no_grad()
def validate(
    model: LeWM,
    embedder: ARCActionEmbedder,
    dataloader: DataLoader,
    device: torch.device,
    use_bf16: bool,
    sigreg,
) -> dict[str, float]:
    """Validation pass."""
    model.eval()
    embedder.eval()

    total_pred = 0.0
    total_sigreg = 0.0
    total_loss = 0.0
    n_batches = 0

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_bf16
        else nullcontext()
    )

    for batch in dataloader:
        obs = batch["obs"].to(device)
        action_emb = embedder(
            batch["action_id"].to(device),
            batch["action_x"].to(device),
            batch["action_y"].to(device),
            batch["action_complex_mask"].to(device),
        )

        with autocast_ctx:
            output = model(obs, action_emb)
            emb = output["emb"]
            next_emb = output["next_emb"]

            pred_loss = torch.nn.functional.mse_loss(
                emb[:, 1:], next_emb[:, :-1]
            )

            sigreg_loss = sigreg(emb.reshape(-1, emb.shape[-1]))

            loss = pred_loss + PAPER_REG_LAMBDA * sigreg_loss

        total_pred += pred_loss.item()
        total_sigreg += sigreg_loss.item()
        total_loss += loss.item()
        n_batches += 1

    return {
        "pred_loss": total_pred / n_batches,
        "sigreg_loss": total_sigreg / n_batches,
        "total_loss": total_loss / n_batches,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true", help="Quick smoke test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--use-bf16", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--sequence-len", type=int, default=4)
    p.add_argument("--output-dir", type=Path, default=ROOT / "lewmtrain" / "runs" / "latest")
    p.add_argument("--human-train", type=Path, default=DATA_DIR / "human_trajectories_train.pt")
    p.add_argument("--human-val", type=Path, default=DATA_DIR / "human_trajectories_val.pt")
    p.add_argument("--random-train", type=Path, default=DATA_DIR / "random_trajectories_train.pt")
    p.add_argument("--random-val", type=Path, default=DATA_DIR / "random_trajectories_val.pt")
    p.add_argument("--no-random", action="store_true", help="Skip random trajectories")
    args = p.parse_args()

    if args.smoke:
        args.num_epochs = 2
        args.batch_size = 4
        args.device = "cpu"
        args.use_bf16 = False
        args.no_random = True
        print("[smoke mode] 2 epochs, batch=4, CPU, human-only")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    rand_train = None if args.no_random else args.random_train
    rand_val = None if args.no_random else args.random_val

    train_ds, val_ds = build_datasets(
        args.human_train, args.human_val,
        rand_train, rand_val,
        args.sequence_len,
    )

    if len(train_ds) == 0:
        raise SystemExit("No training windows! Check data paths.")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        drop_last=False, pin_memory=(device.type == "cuda"),
    )

    # Model
    encoder_config = make_arc_encoder_config()
    latent_dim = encoder_config.latent_dim
    predictor_config = PredictorConfig(
        latent_dim=latent_dim,
        num_layers=6,
        num_heads=16,
        dropout=0.1,
    )
    model_config = LeWMConfig(encoder=encoder_config, predictor=predictor_config)
    model = LeWM(model_config).to(device)
    embedder = ARCActionEmbedder(action_dim=latent_dim).to(device)

    # SIGReg (shared across train + val)
    from lewm.sigreg import SIGReg
    sigreg = SIGReg(latent_dim=latent_dim).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_emb_params = sum(p.numel() for p in embedder.parameters())
    print(f"Model params: {n_params:,} + embedder: {n_emb_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(embedder.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Training loop
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTraining {args.num_epochs} epochs...")
    print(f"{'Epoch':>6s}  {'train_pred':>10s}  {'train_sig':>10s}  "
          f"{'train_tot':>10s}  {'val_pred':>10s}  {'val_sig':>10s}  "
          f"{'val_tot':>10s}  {'best':>6s}")
    print("-" * 85)

    t_start = time.time()

    for epoch in range(1, args.num_epochs + 1):
        train_metrics = train_epoch(
            model, embedder, train_loader, optimizer,
            device, args.use_bf16, args.grad_clip, sigreg,
        )
        val_metrics = validate(
            model, embedder, val_loader,
            device, args.use_bf16, sigreg,
        )

        is_best = val_metrics["total_loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["total_loss"]

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })

        marker = " *" if is_best else ""
        print(
            f"{epoch:6d}  {train_metrics['pred_loss']:10.6f}  "
            f"{train_metrics['sigreg_loss']:10.6f}  "
            f"{train_metrics['total_loss']:10.6f}  "
            f"{val_metrics['pred_loss']:10.6f}  "
            f"{val_metrics['sigreg_loss']:10.6f}  "
            f"{val_metrics['total_loss']:10.6f}"
            f"{marker:>6s}"
        )

        # Save checkpoint every 10 epochs + best
        if epoch % 10 == 0 or is_best:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "embedder_state": embedder.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
            }
            suffix = "best" if is_best else f"epoch{epoch:03d}"
            torch.save(ckpt, args.output_dir / f"checkpoint_{suffix}.pt")

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed/60:.1f} min")
    print(f"Best val loss: {best_val_loss:.6f}")

    # Save final summary
    summary = {
        "best_val_loss": best_val_loss,
        "epochs": args.num_epochs,
        "elapsed_sec": elapsed,
        "history": history,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
