"""Goal embedding prediction for ARC-AGI-3.

Train a model to predict the goal state embedding from partial trajectory
prefix. This is different from cross-game contrastive because:

- The target is the goal FRAME embedding, not the game ID
- Trajectories from the same game at different positions should
  produce embeddings that progressively approach the goal embedding
- The model must learn the dynamics of goal proximity

Architecture:
    FrameCNNEncoder (shared) encodes both:
        - partial trajectory frames → trajectory embedding (via transformer)
        - full goal frame → goal embedding (direct)
    
    Contrastive loss: trajectory_embedding should be close to goal_embedding
    for the same game, and far from different games' goal embeddings.

Key innovation: we explicitly encode the GOAL state and match against it,
rather than just clustering trajectory embeddings. This forces the model
to learn what the goal looks like.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


ARC_GRID_SIZE = 64
ARC_NUM_COLORS = 16
ARC_NUM_ACTIONS = 7


# ── Dataset ─────────────────────────────────────────────────────────────────

class GoalPredictionDataset(Dataset):
    """Each sample: trajectory prefix + goal frame.

    The goal embedding is computed on-the-fly by the shared encoder.
    The trajectory prefix embedding is computed by the transformer.
    Contrastive loss brings them together (same game) and pushes
    them apart (different games).
    """

    def __init__(
        self,
        pattern_dataset_path: str | Path,
        goals_dataset_path: str | Path,
        *,
        split: str = "train",
        context_len: int = 3,
        samples_per_game: int = 5,
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

            # Goal frame (the target)
            goal_frame = frames[goal_idx].float()

            for _ in range(samples_per_game):
                max_start = max(0, goal_idx - context_len + 1)
                if max_start < 1:
                    continue

                start_idx = self.rng.randint(0, max_start - 1) if max_start > 1 else 0

                c_frames = frames[start_idx:start_idx + context_len].float()
                if context_len > 1 and start_idx + context_len - 1 <= actions.shape[0]:
                    c_actions = actions[start_idx:start_idx + context_len - 1].long()
                else:
                    c_actions = actions.new_zeros((0,))

                self.samples.append({
                    "context_frames": c_frames,
                    "context_actions": c_actions,
                    "goal_frame": goal_frame,
                    "game_id": gid,
                    "game_idx": meta["concept_idx"],
                    "goal_idx": goal_idx,
                    "total_frames": T,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# ── Model Components ────────────────────────────────────────────────────────

class FrameCNNEncoder(nn.Module):
    """Lightweight CNN encoder for individual ARC frames."""

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


class GoalPredictionModel(nn.Module):
    """Predict goal frame embedding from trajectory prefix.

    Two encoding paths sharing the FrameCNNEncoder:

    1. Trajectory path: encodes K context frames + actions via transformer
    2. Goal path: directly encodes the target goal frame

    Trained to map trajectory_embed → goal_embed for same game
    (contrastive against other games' goal embeddings).
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

        # Shared frame encoder
        self.frame_encoder = FrameCNNEncoder(cnn_dim=64, hidden_dim=hidden_dim)
        self.action_embedder = ActionEmbedder(hidden_dim=hidden_dim)
        self.pos_embed = nn.Embedding(64, hidden_dim)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)

        # Transformer for trajectory encoding
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

        # Goal projection head (for trajectory → goal embedding)
        self.traj_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.trunc_normal_(self.traj_proj.weight, std=0.02)

        # Goal frame projection
        self.goal_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.trunc_normal_(self.goal_proj.weight, std=0.02)

    def encode_trajectory(
        self, context_frames: torch.Tensor, context_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Encode trajectory prefix into embedding.

        Args:
            context_frames: [B, K, 16, 64, 64]
            context_actions: [B, K-1] long

        Returns:
            [B, D] trajectory embedding
        """
        B, K = context_frames.shape[:2]
        frame_tokens = self.frame_encoder(context_frames)  # [B, K, D]

        if K > 1:
            action_tokens = self.action_embedder(context_actions)  # [B, K-1, D]
            seq_len = 2 * K - 1
            tokens = torch.zeros(B, seq_len, self.hidden_dim, device=context_frames.device)
            tokens[:, 0::2, :] = frame_tokens
            tokens[:, 1::2, :] = action_tokens
        else:
            tokens = frame_tokens

        positions = torch.arange(tokens.shape[1], device=context_frames.device)
        positions = positions.unsqueeze(0).expand(B, -1)
        tokens = tokens + self.pos_embed(positions)

        x = self.transformer(tokens)
        h = x[:, -1, :]
        return F.normalize(self.traj_proj(h), p=2, dim=-1)

    def encode_goal(self, goal_frame: torch.Tensor) -> torch.Tensor:
        """Encode goal frame into embedding.

        Args:
            goal_frame: [B, 16, 64, 64]

        Returns:
            [B, D] goal frame embedding
        """
        h = self.frame_encoder(goal_frame)  # [B, D]
        return F.normalize(self.goal_proj(h), p=2, dim=-1)

    def forward(
        self, context_frames: torch.Tensor, context_actions: torch.Tensor,
        goal_frame: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            context_frames: [B, K, 16, 64, 64]
            context_actions: [B, K-1] long
            goal_frame: [B, 16, 64, 64]

        Returns:
            traj_embed: [B, D] trajectory embedding
            goal_embed: [B, D] goal frame embedding
        """
        traj_embed = self.encode_trajectory(context_frames, context_actions)
        goal_embed = self.encode_goal(goal_frame)
        return {"traj_embed": traj_embed, "goal_embed": goal_embed}

    @torch.no_grad()
    def predict_goal(
        self, context_frames: torch.Tensor, context_actions: torch.Tensor,
        candidate_goals: torch.Tensor,
    ) -> torch.Tensor:
        """Retrieve the closest goal frame from candidates.

        Args:
            context_frames: [B, K, 16, 64, 64]
            context_actions: [B, K-1] long
            candidate_goals: [M, 16, 64, 64]

        Returns:
            scores: [B, M] cosine similarity to each candidate
        """
        self.eval()
        traj_embed = self.encode_trajectory(context_frames, context_actions)  # [B, D]
        goal_embeds = self.encode_goal(candidate_goals)  # [M, D]
        return traj_embed @ goal_embeds.T


# ── Loss ────────────────────────────────────────────────────────────────────

def goal_prediction_loss(
    traj_embed: torch.Tensor,
    goal_embed: torch.Tensor,
    game_indices: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Contrastive loss: trajectory→goal for same game.

    For each trajectory, positive = its game's goal embedding.
    Negatives = other games' goal embeddings.
    Uses in-batch classification with multi-positive labels: if a batch has
    multiple prefixes from the same game, every matching goal row is positive.

    Args:
        traj_embed: [B, D] normalized trajectory embeddings
        goal_embed: [B, D] normalized goal frame embeddings
        game_indices: [B] integer game IDs used to avoid same-game false negatives
        temperature: scaling

    Returns:
        scalar loss
    """
    B = traj_embed.shape[0]
    if B <= 1:
        return (traj_embed * 0).sum()

    # Similarity: each trajectory vs all goals in batch
    logits = traj_embed @ goal_embed.T / temperature  # [B, B]

    game_indices = game_indices.to(device=traj_embed.device)
    positive_mask = game_indices[:, None].eq(game_indices[None, :])
    if not bool(positive_mask.any()):
        labels = torch.arange(B, device=traj_embed.device)
        return F.cross_entropy(logits, labels)

    # Multi-positive InfoNCE:
    # - denominator is all in-batch goals
    # - numerator is every same-game goal embedding
    # This reduces to diagonal cross entropy when all game IDs are unique.
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    masked_log_prob = log_prob.masked_fill(~positive_mask, float("-inf"))
    return -torch.logsumexp(masked_log_prob, dim=1).mean()


# ── Trainer ─────────────────────────────────────────────────────────────────

@dataclass
class GoalPredConfig:
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    temperature: float = 0.1
    seed: int = 0
    log_every: int = 5
    save_every: int = 25


class GoalPredictionTrainer:
    def __init__(
        self, model: GoalPredictionModel, config: GoalPredConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or GoalPredConfig()
        torch.manual_seed(self.config.seed)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.history: list[dict[str, float]] = []

    def train_step(
        self, cf: torch.Tensor, ca: torch.Tensor,
        gf: torch.Tensor, gi: torch.Tensor,
    ) -> dict[str, float]:
        self.model.train()
        outputs = self.model(cf, ca, gf)
        loss = goal_prediction_loss(
            outputs["traj_embed"], outputs["goal_embed"], gi,
            temperature=self.config.temperature,
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return {"loss": float(loss.item())}

    @torch.no_grad()
    def evaluate(
        self, cf: torch.Tensor, ca: torch.Tensor,
        gf: torch.Tensor, gi: torch.Tensor,
    ) -> dict[str, float]:
        self.model.eval()
        outputs = self.model(cf, ca, gf)
        loss = goal_prediction_loss(
            outputs["traj_embed"], outputs["goal_embed"], gi,
            temperature=self.config.temperature,
        )
        return {
            "loss": float(loss.item()),
            "traj_embed": outputs["traj_embed"].cpu(),
            "goal_embed": outputs["goal_embed"].cpu(),
        }


# ── Evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def goal_retrieval_eval(
    model: GoalPredictionModel,
    query_loader: DataLoader,
    db_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate goal retrieval: given trajectory, retrieve correct goal frame.

    Builds database of (goal_embed, game_idx) from db_loader.
    For each query trajectory, finds nearest goal embeddings.

    Returns recall@1, recall@5, mean rank.
    """
    model.eval().to(device)

    # Build goal database
    db_embeds, db_game_idxs = [], []
    for batch in db_loader:
        gf = batch["goal_frame"].to(device)
        g_emb = model.encode_goal(gf)
        db_embeds.append(g_emb.cpu())
        db_game_idxs.append(batch["game_idx"])

    db_embeds = torch.cat(db_embeds, dim=0)
    db_game_idxs = torch.cat(db_game_idxs, dim=0)

    top1, top5, total_rank = 0, 0, 0
    n_queries = 0

    for batch in query_loader:
        cf = batch["context_frames"].to(device)
        ca = batch["context_actions"].to(device)
        t_emb = model.encode_trajectory(cf, ca).cpu()
        q_gi = batch["game_idx"]

        for b in range(t_emb.shape[0]):
            sim = t_emb[b:b+1] @ db_embeds.T
            sorted_idx = sim[0].argsort(descending=True)

            for rank, idx in enumerate(sorted_idx):
                if db_game_idxs[idx] == q_gi[b].item():
                    if rank == 0:
                        top1 += 1
                    if rank < 5:
                        top5 += 1
                    total_rank += rank + 1
                    break
            n_queries += 1

    return {
        "recall_at_1": top1 / n_queries if n_queries else 0.0,
        "recall_at_5": top5 / n_queries if n_queries else 0.0,
        "mean_rank": total_rank / n_queries if n_queries else 0.0,
    }


def collate_goal_pred(batch: list[dict]) -> dict:
    cf = torch.stack([b["context_frames"] for b in batch])
    ca = torch.stack([b["context_actions"] for b in batch])
    gf = torch.stack([b["goal_frame"] for b in batch])
    gi = torch.tensor([b["game_idx"] for b in batch], dtype=torch.long)
    return {
        "context_frames": cf, "context_actions": ca,
        "goal_frame": gf, "game_idx": gi,
    }


__all__ = [
    "GoalPredictionDataset",
    "GoalPredictionModel",
    "GoalPredConfig",
    "GoalPredictionTrainer",
    "goal_prediction_loss",
    "goal_retrieval_eval",
    "collate_goal_pred",
]
