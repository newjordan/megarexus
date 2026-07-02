"""Goal inference: predict the goal state from partial ARC-AGI-3 trajectories.

Given a partial trajectory (obs_0, a_0, obs_1, ..., a_{k-1}, obs_k),
predict the goal state obs_goal where the win condition holds.

The model uses a lightweight CNN encoder + trajectory transformer to
aggregate temporal information, then predicts the goal frame either as
a full-frame reconstruction or via a contrastive goal-embedding loss.

Usage:
    from lewm.goal_inference import (
        GoalInferenceDataset, GoalInferenceModel, GoalInferenceTrainer,
    )
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── Constants ───────────────────────────────────────────────────────────────

ARC_GRID_SIZE = 64
ARC_NUM_COLORS = 16
ARC_NUM_ACTIONS = 7


# ── Dataset ─────────────────────────────────────────────────────────────────

@dataclass
class GoalInferenceSample:
    """One training sample for goal inference.

    Context: first K frames + actions from trajectory (up to start_idx).
    Target: the goal frame.

    Fields:
        context_frames: [K, 16, 64, 64] one-hot float
        context_actions: [K-1] long (action_ids), length K-1 (transitions between context frames)
        target_frame: [16, 64, 64] one-hot float — the goal state
        game_id: str
        goal_idx: int (index within full trajectory)
    """
    context_frames: torch.Tensor
    context_actions: torch.Tensor
    target_frame: torch.Tensor
    game_id: str
    goal_idx: int
    total_frames: int


class GoalInferenceDataset(Dataset):
    """ARC-AGI-3 trajectory dataset for goal inference.

    Each sample: given first K frames (default 3), predict the goal frame.

    The dataset samples random start positions up to goal_idx-K,
    providing multiple training examples per trajectory.
    """

    def __init__(
        self,
        pattern_dataset_path: str | Path,
        goals_dataset_path: str | Path,
        *,
        split: str = "train",
        context_len: int = 3,
        samples_per_game: int = 3,
        seed: int = 0,
    ) -> None:
        self.context_len = context_len
        self.samples_per_game = samples_per_game
        self.rng = random.Random(seed)

        # Load datasets
        pd = torch.load(Path(pattern_dataset_path), map_location="cpu", weights_only=False)
        gd = torch.load(Path(goals_dataset_path), map_location="cpu", weights_only=False)

        # Get pattern data (frame tensors + actions) for this split
        self.pattern_data = pd["data"][split]
        self.concept_names = pd.get("concept_names", [])

        # Get goal metadata (goal_idx, goal_type, etc.)
        self.goal_metadata = {m["game_id"]: m for m in gd[split]}

        # Build samples
        self.samples: list[GoalInferenceSample] = []
        for item in self.pattern_data:
            gid = item["game_id"]
            if gid not in self.goal_metadata:
                continue

            meta = self.goal_metadata[gid]
            frames = item["frames"]  # [T, 16, 64, 64]
            actions = item["actions"]  # [T-1]
            goal_idx = meta["goal_idx"]
            T = frames.shape[0]

            # Generate multiple samples per game
            for _ in range(samples_per_game):
                # Context must fit before the goal with room for context_len frames
                # context_actions length = context_len - 1 (one action per transition)
                max_start = max(0, goal_idx - context_len + 1)
                if max_start < 1:
                    continue

                start_idx = self.rng.randint(0, max_start - 1) if max_start > 1 else 0

                # Context frames: start_idx to start_idx + context_len - 1
                c_frames = frames[start_idx:start_idx + context_len]

                # Context actions: actions between those frames = start_idx to start_idx + context_len - 2
                if context_len > 1 and start_idx + context_len - 1 <= actions.shape[0]:
                    c_actions = actions[start_idx:start_idx + context_len - 1]
                else:
                    c_actions = actions.new_zeros((0,))

                assert c_frames.shape[0] == context_len, f"Expected {context_len} context frames, got {c_frames.shape[0]}"

                self.samples.append(GoalInferenceSample(
                    context_frames=c_frames.float(),
                    context_actions=c_actions.long(),
                    target_frame=frames[goal_idx].float(),
                    game_id=gid,
                    goal_idx=goal_idx,
                    total_frames=T,
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        s = self.samples[idx]
        return {
            "context_frames": s.context_frames,      # [K, 16, 64, 64]
            "context_actions": s.context_actions,    # [K-1]
            "target_frame": s.target_frame,          # [16, 64, 64]
            "game_id": s.game_id,
            "goal_idx": s.goal_idx,
            "total_frames": s.total_frames,
        }


# ── Model ───────────────────────────────────────────────────────────────────

class FrameCNNEncoder(nn.Module):
    """Lightweight CNN to encode individual frames.

    Same architecture as PatternHead's CNN trunk (known-good).
    Maps [B, 16, 64, 64] → [B, D].
    """

    def __init__(self, cnn_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
        self.cnn_dim = cnn_dim
        self.hidden_dim = hidden_dim

        c = cnn_dim
        self.conv1 = nn.Conv2d(ARC_NUM_COLORS, c, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c * 2, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(c * 2)
        self.conv3 = nn.Conv2d(c * 2, c * 4, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(c * 4)
        self.conv4 = nn.Conv2d(c * 4, c * 4, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(c * 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(c * 4, hidden_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frame(s).

        Args:
            frames: [B, 16, 64, 64] or [B, K, 16, 64, 64] with K extra dim
        Returns:
            [B, D] or [B, K, D] features
        """
        # Flatten batch+time dims if needed
        if frames.dim() == 5:
            B, K, C, H, W = frames.shape
            x = frames.flatten(0, 1)  # [B*K, C, H, W]
            flat = True
        else:
            B = frames.shape[0]
            x = frames
            flat = False
            K = 1

        h = F.gelu(self.bn1(self.conv1(x)))       # [B*K, c, 64, 64]
        h = F.max_pool2d(h, 2)                     # [B*K, c, 32, 32]
        h = F.gelu(self.bn2(self.conv2(h)))        # [B*K, c*2, 32, 32]
        h = F.max_pool2d(h, 2)                     # [B*K, c*2, 16, 16]
        h = F.gelu(self.bn3(self.conv3(h)))        # [B*K, c*4, 16, 16]
        h = F.max_pool2d(h, 2)                     # [B*K, c*4, 8, 8]
        h = F.gelu(self.bn4(self.conv4(h)))        # [B*K, c*4, 8, 8]
        h = self.pool(h).flatten(1)                # [B*K, c*4]
        h = self.proj(h)                           # [B*K, D]

        if flat:
            h = h.view(B, K, self.hidden_dim)

        return h


class ActionEmbedder(nn.Module):
    """Embed action IDs into continuous vectors."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.embed = nn.Embedding(ARC_NUM_ACTIONS + 1, hidden_dim)  # +1 for padding/null
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """Embed actions.

        Args:
            actions: [B, K-1] long or [B] long
        Returns:
            [B, K-1, D] or [B, D] embeddings
        """
        return self.embed(actions)


class GoalInferenceModel(nn.Module):
    """Goal inference model: trajectory → goal frame.

    Architecture:
    1. FrameCNNEncoder: each frame → latent [B, K, D]
    2. ActionEmbedder: each action → action_emb [B, K-1, D]
    3. Small transformer processes the token sequence
    4. Goal head predicts goal frame

    Two output modes:
    - "latent": outputs a latent vector (use with contrastive loss)
    - "reconstruct": outputs [16, 64, 64] frame reconstruction
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_transformer_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.frame_encoder = FrameCNNEncoder(cnn_dim=64, hidden_dim=hidden_dim)
        self.action_embedder = ActionEmbedder(hidden_dim=hidden_dim)

        # Positional embedding for trajectory positions (frames + actions interleaved)
        # Max trajectory length: we support up to 32 steps
        self.pos_embed = nn.Embedding(64, hidden_dim)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # Goal prediction head(s)
        self.goal_latent = nn.Linear(hidden_dim, hidden_dim)

        # Frame decoder (lightweight): latent → [16, 64, 64]
        # This is a tiny transposed-conv decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 256 * 8 * 8),
            nn.GELU(),
            nn.Unflatten(1, (256, 8, 8)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 16
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # 32
            nn.GELU(),
            nn.ConvTranspose2d(64, ARC_NUM_COLORS, 4, stride=2, padding=1),  # 64
        )

    def _build_trajectory_tokens(
        self,
        context_frames: torch.Tensor,
        context_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build interleaved token sequence from frames and actions.

        Sequence: [F0, A0, F1, A1, ..., F_{K-1}]
        where F_i are frame encodings and A_i are action embeddings.

        Args:
            context_frames: [B, K, 16, 64, 64]
            context_actions: [B, K-1] long

        Returns:
            tokens: [B, seq_len, D]
            mask: [B, seq_len] bool (True = valid, False = padding)
        """
        B, K = context_frames.shape[:2]

        # Encode frames: [B, K, D]
        frame_tokens = self.frame_encoder(context_frames)

        # Encode actions: [B, K-1, D]
        action_tokens = self.action_embedder(context_actions)

        # Interleave: [F0, A0, F1, A1, ..., A_{K-2}, F_{K-1}]
        # This gives seq_len = K + (K-1) = 2K - 1
        seq_len = 2 * K - 1
        tokens = torch.zeros(B, seq_len, self.hidden_dim, device=context_frames.device)

        # Place frames at even positions (0, 2, 4, ...)
        tokens[:, 0::2, :] = frame_tokens

        # Place actions at odd positions (1, 3, 5, ...)
        if K > 1:
            tokens[:, 1::2, :] = action_tokens

        # Add positional embeddings
        positions = torch.arange(seq_len, device=context_frames.device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.pos_embed(positions)

        # All tokens are valid (no padding in this simple version)
        mask = torch.ones(B, seq_len, dtype=torch.bool, device=context_frames.device)

        return tokens, mask

    def forward(
        self,
        context_frames: torch.Tensor,
        context_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            context_frames: [B, K, 16, 64, 64]
            context_actions: [B, K-1] long

        Returns:
            dict with:
                - goal_latent: [B, D] predicted goal representation
                - goal_frame: [B, 16, 64, 64] predicted goal frame
                - trajectory_features: [B, seq_len, D] full transformer output
        """
        B, K = context_frames.shape[:2]

        # Build interleaved token sequence
        tokens, mask = self._build_trajectory_tokens(context_frames, context_actions)

        # Transformer
        # Note: TransformerEncoder expects padding mask where True = valid
        # But we have no padding, so all True is fine
        x = self.transformer(tokens, src_key_padding_mask=~mask)

        # Use the last token (which corresponds to the last frame F_{K-1})
        # as the trajectory summary for goal prediction
        last_idx = 2 * K - 2  # even index: last frame position
        h = x[:, last_idx, :]  # [B, D]

        goal_latent = self.goal_latent(h)  # [B, D]

        # Decode to full frame
        goal_frame = self.decoder(goal_latent)  # [B, 16, 64, 64]

        return {
            "goal_latent": goal_latent,
            "goal_frame": goal_frame,
            "trajectory_features": x,
            "last_hidden": h,
        }

    def predict_goal(
        self,
        context_frames: torch.Tensor,
        context_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Inference: predict goal frame from partial trajectory.

        Args:
            context_frames: [K, 16, 64, 64] or [B, K, 16, 64, 64]
            context_actions: [K-1] or [B, K-1]

        Returns:
            [16, 64, 64] or [B, 16, 64, 64] predicted goal frame
        """
        # Handle single-sample case
        if context_frames.dim() == 4:
            context_frames = context_frames.unsqueeze(0)
            context_actions = context_actions.unsqueeze(0)

        with torch.no_grad():
            outputs = self.forward(context_frames, context_actions)
            goal_frame = outputs["goal_frame"]

        if goal_frame.shape[0] == 1:
            goal_frame = goal_frame.squeeze(0)

        return goal_frame


# ── Losses ─────────────────────────────────────────────────────────────────

def frame_reconstruction_loss(
    pred_frame: torch.Tensor,
    target_frame: torch.Tensor,
) -> torch.Tensor:
    """MSE + BCE on frame reconstruction.

    Args:
        pred_frame: [B, 16, 64, 64] logits (before softmax)
        target_frame: [B, 16, 64, 64] one-hot

    Returns:
        scalar loss
    """
    # Cross-entropy per pixel: [B, 16, 64, 64] → [B, 64, 64] → scalar
    bce = F.cross_entropy(pred_frame, target_frame.argmax(dim=1))

    # MSE on the softmax outputs encourages sharper predictions
    pred_softmax = F.softmax(pred_frame, dim=1)
    mse = F.mse_loss(pred_softmax, target_frame)

    return bce + mse


def contrastive_goal_loss(
    goal_latents: torch.Tensor,
    target_latents: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE loss between predicted and encoded goal latents.

    Args:
        goal_latents: [B, D] predicted latents
        target_latents: [B, D] target latents (encoded target frames)
        temperature: scaling temperature

    Returns:
        scalar InfoNCE loss
    """
    B = goal_latents.shape[0]
    if B <= 1:
        return goal_latents.new_zeros(())

    # Normalize
    q = F.normalize(goal_latents, dim=-1)
    k = F.normalize(target_latents, dim=-1)

    # Positive: diag; negatives: off-diagonal
    logits = q @ k.T / temperature  # [B, B]

    labels = torch.arange(B, device=goal_latents.device)
    return F.cross_entropy(logits, labels)


class GoalInferenceLoss(nn.Module):
    """Combined loss for goal inference training.

    L = L_recon + α * L_contrastive

    α controls the balance between frame reconstruction and contrastive alignment.
    """

    def __init__(self, contrastive_weight: float = 0.1, temperature: float = 0.1):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target_frame: torch.Tensor,
        target_latent: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute losses.

        Args:
            outputs: from GoalInferenceModel.forward()
            target_frame: [B, 16, 64, 64] one-hot
            target_latent: [B, D] optional target latents for contrastive loss

        Returns:
            dict with keys: total, recon, contrastive
        """
        recon = frame_reconstruction_loss(outputs["goal_frame"], target_frame)

        total = recon

        contrastive = torch.zeros_like(recon)
        if target_latent is not None and self.contrastive_weight > 0:
            contrastive = contrastive_goal_loss(
                outputs["goal_latent"],
                target_latent,
                temperature=self.temperature,
            )
            total = total + self.contrastive_weight * contrastive

        return {
            "total": total,
            "recon": recon.detach(),
            "contrastive": contrastive.detach(),
        }


# ── Trainer ─────────────────────────────────────────────────────────────────

@dataclass
class GoalTrainingConfig:
    num_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    contrastive_weight: float = 0.0  # Set >0 to enable contrastive loss
    temperature: float = 0.1
    seed: int = 0
    log_every: int = 10
    save_every: int = 25


class GoalInferenceTrainer:
    """Train a GoalInferenceModel."""

    def __init__(
        self,
        model: GoalInferenceModel,
        config: GoalTrainingConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or GoalTrainingConfig()
        torch.manual_seed(self.config.seed)

        self.loss_fn = GoalInferenceLoss(
            contrastive_weight=self.config.contrastive_weight,
            temperature=self.config.temperature,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.history: list[dict[str, float]] = []

    def train_step(
        self,
        context_frames: torch.Tensor,
        context_actions: torch.Tensor,
        target_frame: torch.Tensor,
    ) -> dict[str, float]:
        """Single training step.

        Args:
            context_frames: [B, K, 16, 64, 64]
            context_actions: [B, K-1] long
            target_frame: [B, 16, 64, 64]

        Returns:
            dict of loss values
        """
        self.model.train()

        outputs = self.model(context_frames, context_actions)
        losses = self.loss_fn(outputs, target_frame)

        self.optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {
            "total": float(losses["total"].item()),
            "recon": float(losses["recon"].item()),
            "contrastive": float(losses["contrastive"].item()),
        }

    @torch.no_grad()
    def evaluate(
        self,
        context_frames: torch.Tensor,
        context_actions: torch.Tensor,
        target_frame: torch.Tensor,
    ) -> dict[str, float]:
        """Evaluate on a batch.

        Args:
            context_frames: [B, K, 16, 64, 64]
            context_actions: [B, K-1] long
            target_frame: [B, 16, 64, 64]

        Returns:
            dict of loss + accuracy metrics
        """
        self.model.eval()

        outputs = self.model(context_frames, context_actions)
        losses = self.loss_fn(outputs, target_frame)

        # Frame prediction accuracy: argmax agreement per pixel
        pred_grid = outputs["goal_frame"].argmax(dim=1)  # [B, 64, 64]
        target_grid = target_frame.argmax(dim=1)  # [B, 64, 64]
        pixel_acc = (pred_grid == target_grid).float().mean().item()

        # Top-k pixel accuracy: pixels matching within k color tolerance
        # Treated as exact match for each pixel

        return {
            "total": float(losses["total"].item()),
            "recon": float(losses["recon"].item()),
            "pixel_acc": pixel_acc,
        }


# ── Evaluation utilities ────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(
    model: GoalInferenceModel,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Full evaluation over a dataloader (no contrastive loss).

    Returns:
        - mean_recon_loss: reconstruction loss over all pixels
        - mean_fg_acc: accuracy on non-background pixels only
        - mean_all_pixel_acc: accuracy over ALL pixels (inflated by background)
        - mean_changed_acc: accuracy on pixels that CHANGE between first and goal
        - mean_fg_mse: MSE on non-background pixels
    """
    model.eval()
    model.to(device)

    recon_losses = []
    fg_accs = []
    all_accs = []
    changed_accs = []
    fg_mses = []

    for batch in dataloader:
        cf = batch["context_frames"].to(device)
        ca = batch["context_actions"].to(device)
        tf = batch["target_frame"].to(device)

        outputs = model(cf, ca)
        recon = frame_reconstruction_loss(outputs["goal_frame"], tf)

        pred_grid = outputs["goal_frame"].argmax(dim=1)  # [B, 64, 64]
        target_grid = tf.argmax(dim=1)  # [B, 64, 64]

        # Per-sample metrics
        for b in range(pred_grid.shape[0]):
            pred = pred_grid[b]
            target = target_grid[b]

            # Full pixel accuracy
            all_acc = (pred == target).float().mean().item()
            all_accs.append(all_acc)

            # Foreground accuracy: only on non-background pixels of target
            fg_mask = target != 0
            if fg_mask.sum() > 0:
                fg_acc = (pred[fg_mask] == target[fg_mask]).float().mean().item()
                fg_accs.append(fg_acc)

                # Foreground MSE on softmax outputs
                tf_softmax = tf[b]  # [16, 64, 64]
                pred_softmax = torch.nn.functional.softmax(outputs["goal_frame"][b], dim=0)
                fg_mse = torch.nn.functional.mse_loss(
                    pred_softmax[:, fg_mask],
                    tf_softmax[:, fg_mask],
                )
                fg_mses.append(fg_mse.item())

            # Changed pixels accuracy: pixels that differ from first context frame
            first_frame = cf[b, 0]  # [16, 64, 64]
            first_grid = first_frame.argmax(dim=0)
            changed_mask = target != first_grid
            if changed_mask.sum() > 0:
                changed_acc = (pred[changed_mask] == target[changed_mask]).float().mean().item()
                changed_accs.append(changed_acc)

        recon_losses.append(recon.item())

    metrics = {"mean_recon_loss": float(torch.tensor(recon_losses).mean())}
    if fg_accs:
        metrics["mean_fg_acc"] = float(torch.tensor(fg_accs).mean())
    else:
        metrics["mean_fg_acc"] = 0.0
    if all_accs:
        metrics["mean_all_pixel_acc"] = float(torch.tensor(all_accs).mean())
    else:
        metrics["mean_all_pixel_acc"] = 0.0
    if changed_accs:
        metrics["mean_changed_acc"] = float(torch.tensor(changed_accs).mean())
    else:
        metrics["mean_changed_acc"] = 0.0
    if fg_mses:
        metrics["mean_fg_mse"] = float(torch.tensor(fg_mses).mean())
    else:
        metrics["mean_fg_mse"] = 0.0

    return metrics


def dataloader_from_dataset(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader from a GoalInferenceDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Simple single-process loading
        drop_last=True,
    )


__all__ = [
    "GoalInferenceSample",
    "GoalInferenceDataset",
    "GoalInferenceModel",
    "GoalTrainingConfig",
    "GoalInferenceTrainer",
    "GoalInferenceLoss",
    "frame_reconstruction_loss",
    "evaluate_model",
    "dataloader_from_dataset",
]
