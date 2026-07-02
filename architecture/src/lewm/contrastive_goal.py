"""Contrastive goal inference for ARC-AGI-3.

Instead of predicting the full goal frame (which is ~98.6% background),
learn a trajectory encoder that maps (obs_0, a_0, ..., obs_k) → goal_embedding
and uses contrastive loss to cluster embeddings from trajectories that share
the same goal state.

Architecture:
    FrameCNNEncoder → ActionEmbedder → TrajectoryTransformer → goal_embedding

Training:
    InfoNCE loss: trajectories from same game are positive pairs,
    different games are negatives. This pushes the model to learn
    game-specific goal representations without predicting full frames.

Evaluation:
    - Top-K retrieval: given query trajectory, retrieve the correct goal
      frame from a database of candidates
    - Game classification accuracy from trajectory embeddings
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ── Constants ───────────────────────────────────────────────────────────────

ARC_GRID_SIZE = 64
ARC_NUM_COLORS = 16
ARC_NUM_ACTIONS = 7


# ── Dataset ─────────────────────────────────────────────────────────────────

class ContrastiveGoalDataset(Dataset):
    """Trajectory contrastive dataset.

    Each sample: partial trajectory (context frames + actions) + game_id.
    Positive pairs come from the same game; negatives from different games.
    This means the model learns to embed trajectories into a space where
    same-game trajectories (same goal!) are close.

    We also return the goal frame for retrieval evaluation.
    """

    def __init__(
        self,
        pattern_dataset_path: str | Path,
        goals_dataset_path: str | Path,
        *,
        split: str = "train",
        context_len: int = 3,
        samples_per_game: int = 4,
        seed: int = 0,
    ) -> None:
        self.context_len = context_len
        self.rng = random.Random(seed)

        pd = torch.load(Path(pattern_dataset_path), map_location="cpu", weights_only=False)
        gd = torch.load(Path(goals_dataset_path), map_location="cpu", weights_only=False)

        self.pattern_data = pd["data"][split]
        self.goal_metadata = {m["game_id"]: m for m in gd[split]}

        self.samples: list[dict] = []
        for item in self.pattern_data:
            gid = item["game_id"]
            if gid not in self.goal_metadata:
                continue

            meta = self.goal_metadata[gid]
            frames = item["frames"]
            actions = item["actions"]
            goal_idx = meta["goal_idx"]
            T = frames.shape[0]

            for _ in range(samples_per_game):
                max_start = max(0, goal_idx - context_len + 1)
                if max_start < 1:
                    continue

                start_idx = self.rng.randint(0, max_start - 1) if max_start > 1 else 0

                c_frames = frames[start_idx:start_idx + context_len]
                if context_len > 1 and start_idx + context_len - 1 <= actions.shape[0]:
                    c_actions = actions[start_idx:start_idx + context_len - 1]
                else:
                    c_actions = actions.new_zeros((0,))

                self.samples.append({
                    "context_frames": c_frames.float(),
                    "context_actions": c_actions.long(),
                    "target_frame": frames[goal_idx].float(),
                    "game_id": gid,
                    "game_idx": meta["concept_idx"],
                    "goal_idx": goal_idx,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# ── Model ───────────────────────────────────────────────────────────────────

class FrameCNNEncoder(nn.Module):
    """Lightweight CNN to encode individual frames -> [B, D] or [B, K, D]."""

    def __init__(self, cnn_dim: int = 64, hidden_dim: int = 256) -> None:
        super().__init__()
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
        """Encode frames.

        Args:
            frames: [B, 16, 64, 64] or [B, K, 16, 64, 64]
        Returns:
            [B, D] or [B, K, D] features
        """
        if frames.dim() == 5:
            B, K, C, H, W = frames.shape
            x = frames.flatten(0, 1)
            flat = True
        else:
            B = frames.shape[0]
            x = frames
            flat = False
            K = 1

        h = F.gelu(self.bn1(self.conv1(x)))
        h = F.max_pool2d(h, 2)
        h = F.gelu(self.bn2(self.conv2(h)))
        h = F.max_pool2d(h, 2)
        h = F.gelu(self.bn3(self.conv3(h)))
        h = F.max_pool2d(h, 2)
        h = F.gelu(self.bn4(self.conv4(h)))
        h = self.pool(h).flatten(1)
        h = self.proj(h)

        if flat:
            h = h.view(B, K, self.hidden_dim)
        return h


class ActionEmbedder(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.embed = nn.Embedding(ARC_NUM_ACTIONS + 1, hidden_dim)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.embed(actions)


class ContrastiveGoalModel(nn.Module):
    """Trajectory encoder for contrastive goal inference.

    Encodes (obs_0, a_0, ..., a_{k-1}, obs_k) → goal embedding vector.
    Trained with InfoNCE so trajectories from the same game produce
    nearby embeddings.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.frame_encoder = FrameCNNEncoder(cnn_dim=64, hidden_dim=hidden_dim)
        self.action_embedder = ActionEmbedder(hidden_dim=hidden_dim)
        self.pos_embed = nn.Embedding(64, hidden_dim)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)

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
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False,
        )

        self.goal_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.trunc_normal_(self.goal_proj.weight, std=0.02)

    def _build_sequence(
        self, context_frames: torch.Tensor, context_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Build interleaved [F0, A0, F1, A1, ..., F_{K-1}] token sequence."""
        B, K = context_frames.shape[:2]

        frame_tokens = self.frame_encoder(context_frames)  # [B, K, D]
        if K > 1:
            action_tokens = self.action_embedder(context_actions)  # [B, K-1, D]
        else:
            return frame_tokens  # [B, 1, D]

        seq_len = 2 * K - 1
        tokens = torch.zeros(B, seq_len, self.hidden_dim, device=context_frames.device)
        tokens[:, 0::2, :] = frame_tokens
        tokens[:, 1::2, :] = action_tokens

        positions = torch.arange(seq_len, device=context_frames.device).unsqueeze(0).expand(B, -1)
        tokens = tokens + self.pos_embed(positions)

        return tokens

    def forward(
        self, context_frames: torch.Tensor, context_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode trajectory into goal embedding.

        Args:
            context_frames: [B, K, 16, 64, 64]
            context_actions: [B, K-1] long

        Returns:
            goal_embed: [B, D] L2-normalized trajectory embedding
            raw: [B, D] unnormalized
        """
        tokens = self._build_sequence(context_frames, context_actions)
        x = self.transformer(tokens)

        # Use last token's output as trajectory summary
        h = x[:, -1, :]
        raw = self.goal_proj(h)
        goal_embed = F.normalize(raw, p=2, dim=-1)

        return {"goal_embed": goal_embed, "raw": raw}

    @torch.no_grad()
    def encode_game(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Encode a full trajectory into an embedding (for database indexing)."""
        self.eval()
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
            actions = actions.unsqueeze(0)
        return self(frames, actions)["goal_embed"]


# ── InfoNCE Loss ───────────────────────────────────────────────────────────

def infonce_loss(
    embeddings: torch.Tensor,
    game_indices: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE contrastive loss.

    Positive pairs: samples from the same game.
    Negative pairs: samples from different games.

    Args:
        embeddings: [B, D] L2-normalized embeddings
        game_indices: [B] integer game IDs
        temperature: scaling temperature

    Returns:
        scalar InfoNCE loss
    """
    B = embeddings.shape[0]
    if B <= 1:
        return embeddings.new_zeros(())

    # Similarity matrix
    sim = embeddings @ embeddings.T / temperature  # [B, B]

    # Positive mask: same game (exclude self-pairs)
    game_idx = game_indices.view(-1, 1)
    pos_mask = (game_idx == game_idx.T).float()
    pos_mask.fill_diagonal_(0)

    # For each anchor, gather log-softmax over all positives
    # Use the InfoNCE formulation: -log( sum_pos exp(sim_ij) / sum_all exp(sim_ik) )
    exp_sim = torch.exp(sim)

    pos_sums = (exp_sim * pos_mask).sum(dim=1)  # [B]
    all_sums = exp_sim.sum(dim=1)  # [B]

    eps = 1e-8
    loss_per_anchor = -torch.log(pos_sums / (all_sums + eps) + eps)

    # Only anchors with at least one positive
    valid = pos_sums > 0
    if not valid.any():
        # Return zero loss on valid anchors but keep gradient graph alive
        # by returning loss * 0 on the first valid prediction
        dummy_pred = embeddings[0:1] - embeddings[0:1]
        return dummy_pred.sum() * 0.0

    return loss_per_anchor[valid].mean()


# ── Trainer ─────────────────────────────────────────────────────────────────

@dataclass
class ContrastiveGoalConfig:
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    temperature: float = 0.1
    seed: int = 0
    log_every: int = 5
    save_every: int = 25


class ContrastiveGoalTrainer:
    """Train a ContrastiveGoalModel with InfoNCE loss."""

    def __init__(
        self, model: ContrastiveGoalModel, config: ContrastiveGoalConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or ContrastiveGoalConfig()
        torch.manual_seed(self.config.seed)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.history: list[dict[str, float]] = []

    def train_step(
        self, context_frames: torch.Tensor, context_actions: torch.Tensor,
        game_indices: torch.Tensor,
    ) -> dict[str, float]:
        self.model.train()
        outputs = self.model(context_frames, context_actions)

        loss = infonce_loss(
            outputs["goal_embed"], game_indices, temperature=self.config.temperature,
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": float(loss.item())}

    @torch.no_grad()
    def evaluate(
        self, context_frames: torch.Tensor, context_actions: torch.Tensor,
        game_indices: torch.Tensor,
    ) -> dict[str, float]:
        self.model.eval()
        outputs = self.model(context_frames, context_actions)
        loss = infonce_loss(
            outputs["goal_embed"], game_indices, temperature=self.config.temperature,
        )
        return {"loss": float(loss.item())}


# ── Evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def retrieval_eval(
    model: ContrastiveGoalModel,
    query_loader: DataLoader,
    db_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Retrieval evaluation: for each query, find the correct goal frame.

    Builds a database of (goal_embed, target_frame, game_idx) from db_loader.
    For each query, retrieves top-K by cosine similarity.

    Returns:
        recall_at_1, recall_at_5, mean_rank
    """
    model.eval().to(device)

    # Build database
    db_embeds, db_frames, db_game_idxs = [], [], []
    for batch in db_loader:
        cf = batch["context_frames"].to(device)
        ca = batch["context_actions"].to(device)
        outputs = model(cf, ca)
        db_embeds.append(outputs["goal_embed"].cpu())
        db_frames.append(batch["target_frame"])
        db_game_idxs.append(batch["game_idx"])

    db_embeds = torch.cat(db_embeds, dim=0)  # [N, D]
    db_frames = torch.cat(db_frames, dim=0)  # [N, 16, 64, 64]
    db_game_idxs = torch.cat(db_game_idxs, dim=0)  # [N]

    top1, top5, ranks = 0, 0, 0
    n_queries = 0

    for batch in query_loader:
        cf = batch["context_frames"].to(device)
        ca = batch["context_actions"].to(device)
        outputs = model(cf, ca)
        q_embeds = outputs["goal_embed"].cpu()  # [B, D]
        q_game_idxs = batch["game_idx"]  # [B]

        for b in range(q_embeds.shape[0]):
            q_emb = q_embeds[b:b+1]
            q_game = q_game_idxs[b:b+1]

            # Similarities
            sim = q_emb @ db_embeds.T  # [1, N]
            # Exclude self (same sample index) — only matters if query == db
            # Get ranking
            sorted_indices = sim[0].argsort(descending=True)

            # Find correct game among top-K
            for rank, idx in enumerate(sorted_indices):
                if db_game_idxs[idx] == q_game.item():
                    if rank == 0:
                        top1 += 1
                    if rank < 5:
                        top5 += 1
                    ranks += rank + 1
                    break

            n_queries += 1

    return {
        "recall_at_1": top1 / n_queries if n_queries > 0 else 0.0,
        "recall_at_5": top5 / n_queries if n_queries > 0 else 0.0,
        "mean_rank": ranks / n_queries if n_queries > 0 else 0.0,
    }


def collate_contrastive(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate function for contrastive dataset batches."""
    cf = torch.stack([b["context_frames"] for b in batch])
    ca = torch.stack([b["context_actions"] for b in batch])
    tf = torch.stack([b["target_frame"] for b in batch])
    gi = torch.tensor([b["game_idx"] for b in batch], dtype=torch.long)
    return {
        "context_frames": cf,
        "context_actions": ca,
        "target_frame": tf,
        "game_idx": gi,
    }


__all__ = [
    "ContrastiveGoalDataset",
    "ContrastiveGoalModel",
    "ContrastiveGoalConfig",
    "ContrastiveGoalTrainer",
    "infonce_loss",
    "retrieval_eval",
    "collate_contrastive",
]
