"""MultiSenseEncoder — ViT + sprite perception + color-topology features.

Enriches the paper-faithful ViT-Tiny latent with structured frame-level features
that the ViT may underrepresent: sprite objects (color, position, size, shape)
and color-topology statistics (histogram, estimated wall density).

Architecture::

    frame [B, C, H, W]
        │
        ├─→ ViT-Tiny ──────────→ z_visual [B, D]
        │
        ├─→ extract_sprites() ─→ [K sprites × 8 feats] ─→ MLP_sprite ─→ z_sprite [B, Ds]
        │
        └─→ color_hist(16) + wall_frac(1) ──────────────→ MLP_topo ───→ z_topo [B, Dt]

    concat [z_visual, z_sprite, z_topo] → Linear → z_rich [B, D]

When ``sprite_dim=0`` and ``topo_dim=0`` the module is identical to
ViTTinyEncoder — fully backward-compatible.

Reference: Loom AGENTS.md, "3D renders / raycasting become image analysis tools."
The sprite extractor and color topology are the first compositional perception
signals; walls, affordances, and SAM masks follow the same pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from lewm.encoder import EncoderConfig, ViTTinyEncoder
from lewm.perception.sprites import Sprite, extract_sprites


# ── sprite feature encoding ──────────────────────────────────────────────

MAX_SPRITES = 8
SPRITE_FEAT_DIM = 8  # color, cx, cy, w, h, size, aspect, density


def _encode_sprites(sprites: list[Sprite], h: int, w: int) -> torch.Tensor:
    """Encode up to MAX_SPRITES sprites into fixed [MAX_SPRITES, SPRITE_FEAT_DIM].

    Each sprite → [color/15, cx/w, cy/h, w/64, h/64, size/256, aspect, density].
    Pads with zeros if fewer than MAX_SPRITES.
    """
    feats = torch.zeros(MAX_SPRITES, SPRITE_FEAT_DIM)
    for i, s in enumerate(sprites[:MAX_SPRITES]):
        feats[i, 0] = s.color / 15.0
        feats[i, 1] = s.centroid[1] / max(w, 1)  # cx
        feats[i, 2] = s.centroid[0] / max(h, 1)  # cy
        feats[i, 3] = s.width / 64.0
        feats[i, 4] = s.height / 64.0
        feats[i, 5] = min(s.size / 256.0, 1.0)
        feats[i, 6] = min(s.aspect_ratio / 4.0, 1.0)
        feats[i, 7] = s.density
    return feats


def _extract_sprites_batch(
    frames: np.ndarray,
    background_colors: tuple[int, ...] = (0,),
    min_size: int = 3,
) -> torch.Tensor:
    """Extract sprite features from a numpy batch [B, H, W] of color-index frames.

    Returns [B, MAX_SPRITES, SPRITE_FEAT_DIM].
    """
    b, h, w = frames.shape
    out = torch.zeros(b, MAX_SPRITES, SPRITE_FEAT_DIM)
    for i in range(b):
        sprites = extract_sprites(
            frames[i], background_colors=background_colors, min_size=min_size,
        )
        out[i] = _encode_sprites(sprites, h, w)
    return out


# ── color-topology features ──────────────────────────────────────────────

def _color_topo_features(frames: np.ndarray) -> torch.Tensor:
    """Extract color histogram (16 bins) + wall-fraction estimate from [B, H, W].

    Wall fraction is estimated as the proportion of cells with color 5 (grey),
    the most common wall color in ARC, plus cells with color 0 touching walls.

    Returns [B, 17].
    """
    b = frames.shape[0]
    feats = torch.zeros(b, 17)
    for i in range(b):
        f = frames[i]
        # 16-bin color histogram normalized by total cells
        for c in range(16):
            feats[i, c] = float((f == c).sum()) / max(f.size, 1)
        # Wall fraction: color 5 (grey) is the typical wall/barrier color in ARC
        feats[i, 16] = float((f == 5).sum()) / max(f.size, 1)
    return feats


# ── config ───────────────────────────────────────────────────────────────


@dataclass
class MultiSenseEncoderConfig:
    """Configuration for the multi-sense encoder.

    Attributes:
        vit: ViT-Tiny encoder configuration (paper defaults).
        sprite_dim: output dimension of sprite MLP. 0 disables the sprite path.
        sprite_hidden: hidden dimension of sprite MLP.
        topo_dim: output dimension of color-topology MLP. 0 disables the topo path.
        topo_hidden: hidden dimension of color-topology MLP.
        background_colors: palette indices to ignore in sprite extraction.
        sprite_min_size: minimum cells for a sprite candidate.
    """

    vit: EncoderConfig = field(default_factory=EncoderConfig)
    sprite_dim: int = 32
    sprite_hidden: int = 64
    topo_dim: int = 16
    topo_hidden: int = 32
    background_colors: tuple[int, ...] = (0,)
    sprite_min_size: int = 3

    def __post_init__(self) -> None:
        if self.sprite_dim < 0:
            raise ValueError(f"sprite_dim must be >= 0, got {self.sprite_dim}")
        if self.topo_dim < 0:
            raise ValueError(f"topo_dim must be >= 0, got {self.topo_dim}")
        # Total input to projection must be positive
        total_in = self.vit.latent_dim + self.sprite_dim + self.topo_dim
        if total_in <= 0:
            raise ValueError("at least one of vit.latent_dim, sprite_dim, topo_dim must be > 0")

    @property
    def latent_dim(self) -> int:
        return self.vit.latent_dim

    @property
    def num_sprite_feats(self) -> int:
        return MAX_SPRITES * SPRITE_FEAT_DIM

    @property
    def total_extra_dim(self) -> int:
        return self.sprite_dim + self.topo_dim


# ── module ───────────────────────────────────────────────────────────────


class MultiSenseEncoder(nn.Module):
    """Multi-sense encoder: ViT + sprite perception + color-topology.

    Args:
        config: MultiSenseEncoderConfig. Set sprite_dim=0 and topo_dim=0
            for pure ViT (paper-faithful) behavior.
    """

    def __init__(self, config: MultiSenseEncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or MultiSenseEncoderConfig()
        cfg = self.config

        self.vit = ViTTinyEncoder(cfg.vit)
        self.latent_dim = cfg.latent_dim

        # Sprite pathway
        if cfg.sprite_dim > 0:
            self.sprite_mlp = nn.Sequential(
                nn.Linear(cfg.num_sprite_feats, cfg.sprite_hidden),
                nn.GELU(),
                nn.Linear(cfg.sprite_hidden, cfg.sprite_dim),
                nn.LayerNorm(cfg.sprite_dim),
            )
        else:
            self.sprite_mlp = None

        # Color-topology pathway
        if cfg.topo_dim > 0:
            self.topo_mlp = nn.Sequential(
                nn.Linear(17, cfg.topo_hidden),
                nn.GELU(),
                nn.Linear(cfg.topo_hidden, cfg.topo_dim),
                nn.LayerNorm(cfg.topo_dim),
            )
        else:
            self.topo_mlp = None

        # Fusion projection: concat→latent_dim
        total_in = cfg.latent_dim + cfg.sprite_dim + cfg.topo_dim
        if total_in != cfg.latent_dim:
            self.fusion = nn.Sequential(
                nn.Linear(total_in, cfg.latent_dim),
                nn.LayerNorm(cfg.latent_dim),
            )
        else:
            self.fusion = nn.Identity()

        self._raw_frames_cache: np.ndarray | None = None  # for inspection

    def _to_numpy_frames(self, images: torch.Tensor) -> np.ndarray:
        """Convert [B, C, H, W] tensor to [B, H, W] numpy color-index frames.

        Assumes C=16 one-hot ARC palette encoding. Falls back to argmax if
        one-hot; otherwise takes first channel as-is.
        """
        if images.shape[1] == 16:
            # ARC one-hot palette: [B, 16, H, W] → [B, H, W] color indices
            return images.argmax(dim=1).cpu().numpy().astype(np.int64)
        elif images.shape[1] == 1:
            return images[:, 0].cpu().numpy().astype(np.int64)
        else:
            # Generic: take mean across channels as heuristic
            return images.mean(dim=1).cpu().numpy().astype(np.int64)

    def _prepare_visual(self, images: torch.Tensor) -> torch.Tensor:
        """Convert [B, C, H, W] to [B, 3, 224, 224] for ViT consumption.

        Supports ARC one-hot (C=16), grayscale (C=1), and RGB (C=3) inputs.
        For ARC frames, maps the integer color index to a deterministic RGB
        triplet so the ViT sees discriminative color structure.
        """
        b, c, h, w = images.shape
        if c == 3 and h == 224 and w == 224:
            return images  # already ViT-ready

        # Convert to integer color-index frame [B, H, W]
        if c == 16:
            # ARC one-hot → color indices
            idx = images.argmax(dim=1)  # [B, H, W]
        elif c == 1:
            idx = images[:, 0].long()  # [B, H, W]
        else:
            # Generic multi-channel: take argmax as color index
            idx = images.argmax(dim=1)  # [B, H, W]

        # Map color indices 0..15 to RGB triplets (same mapping used across
        # the Loom codebase — see perception/colors.py)
        color_map = torch.tensor(
            [
                [0, 0, 0],       # 0: black/background
                [0, 0, 170],     # 1: blue
                [170, 0, 0],     # 2: red
                [0, 170, 0],     # 3: green
                [170, 170, 0],   # 4: yellow
                [128, 128, 128], # 5: grey/wall
                [170, 0, 170],   # 6: magenta
                [255, 128, 0],   # 7: orange
                [128, 255, 255], # 8: cyan
                [160, 82, 45],   # 9: brown
                [128, 0, 128],   # 10: purple
                [0, 128, 128],   # 11: teal
                [255, 192, 203], # 12: pink
                [128, 128, 0],   # 13: olive
                [0, 255, 255],   # 14: aqua
                [255, 255, 255], # 15: white
            ],
            dtype=torch.float32, device=images.device,
        )
        rgb = color_map[idx.clamp(0, 15)]  # [B, H, W, 3]
        rgb = rgb.permute(0, 3, 1, 2)      # [B, 3, H, W]
        rgb = rgb / 255.0                   # normalize to [0, 1]

        # Resize to 224×224
        if h != 224 or w != 224:
            rgb = torch.nn.functional.interpolate(
                rgb, size=(224, 224), mode="bilinear", align_corners=False,
            )
        return rgb

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode frames [B, C, H, W] → [B, latent_dim].

        The ViT path always runs. Sprite and topo paths run only if configured
        with non-zero output dimensions.
        """
        if images.dim() != 4:
            raise ValueError(
                f"expected [B, C, H, W], got {tuple(images.shape)}"
            )

        # ViT path (always active) — convert to 3-channel 224×224 first
        z_visual = self.vit(self._prepare_visual(images))

        cfg = self.config
        parts = [z_visual]

        # Sprite path
        if self.sprite_mlp is not None:
            frames_np = self._to_numpy_frames(images)
            self._raw_frames_cache = frames_np
            sprite_feats = _extract_sprites_batch(
                frames_np,
                background_colors=cfg.background_colors,
                min_size=cfg.sprite_min_size,
            ).to(images.device)
            # Flatten [B, K, F] → [B, K*F]
            z_sprite = self.sprite_mlp(sprite_feats.flatten(1))
            parts.append(z_sprite)

        # Color-topology path
        if self.topo_mlp is not None:
            if self._raw_frames_cache is None:
                frames_np = self._to_numpy_frames(images)
            else:
                frames_np = self._raw_frames_cache
            topo_feats = _color_topo_features(frames_np).to(images.device)
            z_topo = self.topo_mlp(topo_feats)
            parts.append(z_topo)

        if len(parts) == 1:
            return z_visual  # pure ViT path, no fusion needed

        z_cat = torch.cat(parts, dim=-1)
        return self.fusion(z_cat)
