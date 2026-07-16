"""Legal blend network: conv trunk + legal feature fuse + LoRA heads."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from legal_search.train.features import FEATURE_DIM
from legal_search.train.lora import LoRALinear, lora_parameters, swap_linear_with_lora

GRID = 64
N_COLORS = 16
GOOSE_SIMPLE = 5
GOOSE_DIM = GOOSE_SIMPLE + GRID * GRID  # 4101
GAME_BUCKETS = 64


def game_bucket(game_id: str, buckets: int = GAME_BUCKETS) -> int:
    """Stable game identity for small conditioned adapters (never Python hash)."""
    digest = hashlib.sha256(str(game_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % int(buckets)


def one_hot_frame(frame: torch.Tensor) -> torch.Tensor:
    """frame [B,H,W] long/int → [B,16,H,W] float."""
    if frame.dim() == 4 and frame.size(1) == N_COLORS:
        return frame.float()
    x = frame.long().clamp(0, N_COLORS - 1)
    return F.one_hot(x, N_COLORS).permute(0, 3, 1, 2).float()


def goose_index_tensor(action_id: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Map actions to [B] indices in 4101 space."""
    aid = action_id.long()
    xx = x.long().clamp(0, GRID - 1)
    yy = y.long().clamp(0, GRID - 1)
    simple_mask = (aid >= 1) & (aid <= 5)
    click_mask = aid == 6
    idx = torch.zeros_like(aid)
    idx = torch.where(simple_mask, aid - 1, idx)
    idx = torch.where(click_mask, GOOSE_SIMPLE + yy * GRID + xx, idx)
    return idx


@dataclass
class LegalBlendConfig:
    feature_dim: int = FEATURE_DIM
    base_channels: int = 48
    latent: int = 96
    fuse_dim: int = 192  # wider fuse for full-mine 392-D features
    lora_rank: int = 16
    lora_alpha: float = 32.0
    dropout: float = 0.05
    game_conditioned: bool = False
    game_buckets: int = GAME_BUCKETS
    game_embed_dim: int = 24
    interaction_conditioned: bool = False
    collapse_window: int = 8

    def to_dict(self) -> dict:
        return asdict(self)


class LegalBlendLoRA(nn.Module):
    """Trainable blend of pixels + mined legal features → Goose-style heads."""

    def __init__(self, config: LegalBlendConfig | None = None) -> None:
        super().__init__()
        self.config = config or LegalBlendConfig()
        cfg = self.config
        c, lat = cfg.base_channels, cfg.latent

        self.stem = nn.Sequential(
            nn.Conv2d(N_COLORS, c, 3, padding=1),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.SiLU(),
        )
        self.down1 = nn.Conv2d(c, c, 4, stride=2, padding=1)  # 32
        self.down2 = nn.Conv2d(c, lat, 4, stride=2, padding=1)  # 16
        self.mid = nn.Sequential(
            nn.GroupNorm(8, lat),
            nn.SiLU(),
            nn.Conv2d(lat, lat, 3, padding=1),
            nn.SiLU(),
        )

        self.feat_mlp = nn.Sequential(
            nn.Linear(cfg.feature_dim, cfg.fuse_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fuse_dim, cfg.fuse_dim),
            nn.SiLU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(lat + cfg.fuse_dim, cfg.fuse_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fuse_dim, cfg.fuse_dim),
            nn.SiLU(),
        )

        # Goose-like heads from fused vector + spatial coords from mid map
        self.simple_change = nn.Linear(cfg.fuse_dim, GOOSE_SIMPLE)
        self.simple_policy = nn.Linear(cfg.fuse_dim, GOOSE_SIMPLE)
        self.coord_change = nn.Sequential(
            nn.Conv2d(lat, lat // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(lat // 2, 1, 1),
            nn.Upsample(size=(GRID, GRID), mode="bilinear", align_corners=False),
        )
        self.coord_policy = nn.Sequential(
            nn.Conv2d(lat, lat // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(lat // 2, 1, 1),
            nn.Upsample(size=(GRID, GRID), mode="bilinear", align_corners=False),
        )
        # legal feature bias into click coords via 1x1 on pooled expand — keep simple:
        self.click_bias = nn.Linear(cfg.fuse_dim, 1)
        if cfg.game_conditioned:
            # Shared visual understanding, small game-specific action calibration.
            self.game_embed = nn.Embedding(cfg.game_buckets, cfg.game_embed_dim)
            self.game_simple_policy = nn.Linear(cfg.game_embed_dim, GOOSE_SIMPLE)
            self.game_click_bias = nn.Linear(cfg.game_embed_dim, 1)

        # Apply LoRA to Linear layers in fuse / heads / feat_mlp
        n = 0
        n += swap_linear_with_lora(self.feat_mlp, rank=cfg.lora_rank, alpha=cfg.lora_alpha)
        n += swap_linear_with_lora(self.fuse, rank=cfg.lora_rank, alpha=cfg.lora_alpha)
        self.simple_change = LoRALinear(self.simple_change, rank=cfg.lora_rank, alpha=cfg.lora_alpha)
        self.simple_policy = LoRALinear(self.simple_policy, rank=cfg.lora_rank, alpha=cfg.lora_alpha)
        self.click_bias = LoRALinear(self.click_bias, rank=cfg.lora_rank, alpha=cfg.lora_alpha)
        n += 3
        self.n_lora_layers = n

        # freeze early conv trunk optionally after warm start — default: train all non-LoRA lightly
        self._freeze_base = False

    def set_lora_only(self, enabled: bool = True) -> None:
        self._freeze_base = enabled
        for name, p in self.named_parameters():
            if "lora_" in name or "game_" in name:
                p.requires_grad = True
            else:
                p.requires_grad = not enabled

    def trainable_param_groups(self):
        lora = list(lora_parameters(self))
        other = [p for n, p in self.named_parameters() if p.requires_grad and "lora_" not in n]
        groups = []
        if lora:
            groups.append({"params": lora, "lr_scale": 1.0})
        if other:
            groups.append({"params": other, "lr_scale": 0.25})
        return groups

    def encode(self, frame: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = one_hot_frame(frame)
        h = self.stem(x)
        h = F.silu(self.down1(h))
        h = F.silu(self.down2(h))
        h16 = self.mid(h)  # [B,lat,16,16]
        pooled = h16.mean(dim=(2, 3))
        f = self.feat_mlp(features)
        fused = self.fuse(torch.cat([pooled, f], dim=-1))
        return h16, fused

    def forward(
        self,
        frame: torch.Tensor,
        features: torch.Tensor,
        game_idx: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        h16, fused = self.encode(frame, features)
        simple_c = self.simple_change(fused)
        simple_p = self.simple_policy(fused)
        coord_c = self.coord_change(h16).flatten(1) + self.click_bias(fused)
        coord_p = self.coord_policy(h16).flatten(1) + self.click_bias(fused)
        if self.config.game_conditioned:
            if game_idx is None:
                game_idx = torch.zeros(frame.size(0), dtype=torch.long, device=frame.device)
            game = self.game_embed(game_idx.long().clamp(0, self.config.game_buckets - 1))
            simple_p = simple_p + self.game_simple_policy(game)
            coord_p = coord_p + self.game_click_bias(game)
        change_logits = torch.cat([simple_c, coord_c], dim=1)
        policy_logits = torch.cat([simple_p, coord_p], dim=1)
        return {
            "change_logits": change_logits,
            "policy_logits": policy_logits,
            "fused": fused,
        }

    def loss(
        self,
        outputs: dict[str, torch.Tensor],
        *,
        action_id: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        did_change: torch.Tensor,
        policy_weight: torch.Tensor | None = None,
        change_weight: torch.Tensor | None = None,
        policy_coef: float = 0.35,
    ) -> dict[str, torch.Tensor]:
        idx = goose_index_tensor(action_id, x, y)
        # change BCE on taken action logit (per-sample weight for thrash / success)
        taken_change = outputs["change_logits"].gather(1, idx.unsqueeze(1)).squeeze(1)
        ch = F.binary_cross_entropy_with_logits(
            taken_change, did_change.float(), reduction="none"
        )
        if change_weight is not None:
            ch = ch * change_weight
        change_loss = ch.mean()
        # policy CE over full 4101
        logp = F.log_softmax(outputs["policy_logits"], dim=-1)
        pol = F.nll_loss(logp, idx, reduction="none")
        if policy_weight is not None:
            pol = pol * policy_weight
        policy_loss = pol.mean()
        total = change_loss + float(policy_coef) * policy_loss
        return {
            "loss": total,
            "change_loss": change_loss.detach(),
            "policy_loss": policy_loss.detach(),
        }
