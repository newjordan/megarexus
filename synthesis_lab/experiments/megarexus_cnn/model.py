"""MegarexusCNN — a valid convolutional synthesis of the three engines.

Role labels (see synthesis_lab/README.md):

- ``LEWM_REX_ACTION_OWNER`` — the action-conditioned convolutional world
  model (encoder + FiLM-conditioned decoder → next-frame logits). This is
  LeWM's Algorithm-1 role (emb → action-conditioned next prediction) made
  fully convolutional: the ViT-Tiny encoder is replaced by a conv trunk and
  the AdaLN transformer predictor by FiLM-modulated residual conv blocks
  (same zero-init "identity at start" conditioning idea as AdaLN-zero).
  The structured action embedding is adapted from
  architecture/src/lewm/action_embedder.py (family base + offset + coords).
- ``CONTRA_THINKING_ENGINE`` — perception feature planes appended to the
  16-channel one-hot frame as extra input channels. The planes are produced
  by engine_adapters/contra_perception_adapter.py from Contra's real
  component packet (codexlab/contra_head); this module only declares how
  many it expects.
- ``STOCHASTIC_GOOSE_RUNTIME`` — the GooseHead mirrors Stochastic Goose's
  ActionModel output contract exactly: a combined 4101-logit vector =
  5 simple-action logits (ACTION1-5) + 4096 click logits (ACTION6 at
  y*64+x). Two twins share the trunk: ``change_logits`` trained with
  Goose's native self-supervised objective (BCE: did the taken action
  change the frame?) and ``policy_logits`` trained by behavior cloning on
  human corpus actions. Sampling semantics live in
  engine_adapters/goose_action_adapter.py.

Input/output contract (matches lewm.data.arc_agi_3 + codexlab ground truth):

- frame: ``[B, 16 + n_contra_planes, 64, 64]`` float (one-hot + planes)
- action: ``action_id`` in [0, 7] (0=RESET/null, 1-5 simple, 6=click with
  (x, y) on the 64×64 grid, 7=other)
- outputs:
  - ``next_frame_logits``: ``[B, 16, 64, 64]`` — forward dynamics
  - ``policy_logits`` / ``change_logits``: ``[B, 4101]`` — Goose head
  - ``inv_action_logits``: ``[B, 8]``, ``inv_click_logits``: ``[B, 4096]``
    — inverse dynamics from a frame pair (diagnostic: are action effects
    perceivable at all?)

Everything here is a standard trainable CNN: no lookup of stored answers,
no environment access at forward time.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

ARC_GRID_SIZE = 64
ARC_NUM_COLORS = 16
ARC_NUM_ACTIONS = 8  # 0=null/reset, 1-5 simple, 6=click(x,y), 7=other
ARC_CLICK_ACTION_ID = 6
GOOSE_NUM_SIMPLE = 5  # ACTION1..ACTION5
GOOSE_HEAD_DIM = GOOSE_NUM_SIMPLE + ARC_GRID_SIZE * ARC_GRID_SIZE  # 4101
def goose_index(action_id: int, x: int, y: int) -> int:
    """Map (action_id, x, y) → index in the 4101-dim Goose head.

    ACTION1..5 → 0..4; ACTION6 at (x, y) → 5 + y*64 + x (Goose's own
    y=idx//64, x=idx%64 convention). Other ids are not representable.
    """
    if 1 <= action_id <= 5:
        return action_id - 1
    if action_id == ARC_CLICK_ACTION_ID:
        return GOOSE_NUM_SIMPLE + y * ARC_GRID_SIZE + x
    raise ValueError(f"action_id {action_id} has no Goose-head index")


@dataclass(frozen=True)
class MegarexusCNNConfig:
    n_contra_planes: int = 6
    n_input_frames: int = 1  # 2 = stack previous frame so paint velocity is visible
    base_channels: int = 64
    latent_channels: int = 128
    action_dim: int = 128
    n_encoder_blocks: int = 2
    n_dynamics_blocks: int = 4
    dropout: float = 0.0

    @property
    def in_channels(self) -> int:
        # channel order per frame: [16 one-hot, n_contra_planes]; the CURRENT
        # frame occupies channels [0, 16+planes) and earlier frames follow, so
        # frame[:, :16] is always the current one-hot regardless of stacking
        return self.n_input_frames * (ARC_NUM_COLORS + self.n_contra_planes)

    def validate(self) -> None:
        if self.n_contra_planes < 0:
            raise ValueError("n_contra_planes must be >= 0")
        if self.n_input_frames not in (1, 2):
            raise ValueError("n_input_frames must be 1 or 2")
        if min(self.base_channels, self.latent_channels, self.action_dim) < 8:
            raise ValueError("channel/action dims must be >= 8")


class StructuredActionEmbedding(nn.Module):
    """Family-structured action embedding (adapted from lewm.action_embedder).

    emb = family_base[family(action)] + action_offset[action] + coord_emb(x, y)

    Families: 0=null, 1=simple (1-5), 2=placement (6), 3=other (7).
    Coordinate embeddings only contribute for the placement action.
    """

    FAMILY_OF_ACTION = (0, 1, 1, 1, 1, 1, 2, 3)

    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.family_base = nn.Embedding(4, action_dim)
        self.action_offset = nn.Embedding(ARC_NUM_ACTIONS, action_dim)
        nn.init.zeros_(self.action_offset.weight)
        self.x_emb = nn.Embedding(ARC_GRID_SIZE, action_dim)
        self.y_emb = nn.Embedding(ARC_GRID_SIZE, action_dim)
        nn.init.trunc_normal_(self.x_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.y_emb.weight, std=0.02)
        self.register_buffer(
            "family_map", torch.tensor(self.FAMILY_OF_ACTION, dtype=torch.long)
        )

    def forward(
        self, action_id: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        emb = self.family_base(self.family_map[action_id])
        emb = emb + self.action_offset(action_id)
        is_click = (action_id == ARC_CLICK_ACTION_ID).float().unsqueeze(-1)
        emb = emb + is_click * (self.x_emb(x) + self.y_emb(y))
        return emb


class FiLMResBlock(nn.Module):
    """Residual conv block modulated by the action embedding (FiLM).

    The conv analogue of the LeWM predictor's AdaLN-zero: the modulation MLP
    is zero-initialized so each block starts as identity and action
    conditioning enters progressively during training.
    """

    def __init__(self, channels: int, action_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.drop = nn.Dropout2d(dropout)
        self.film = nn.Linear(action_dim, 3 * channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, h: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        gamma, beta, alpha = self.film(action_emb).chunk(3, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        alpha = alpha.unsqueeze(-1).unsqueeze(-1)
        x = self.conv1(F.silu((1 + gamma) * self.norm1(h) + beta))
        x = self.drop(self.conv2(F.silu(self.norm2(x))))
        return h + alpha * x


class ConvTrunk(nn.Module):
    """Shared convolutional encoder: [B, C_in, 64, 64] → [B, C_lat, 16, 16]."""

    def __init__(self, cfg: MegarexusCNNConfig) -> None:
        super().__init__()
        c, lat = cfg.base_channels, cfg.latent_channels
        self.stem = nn.Conv2d(cfg.in_channels, c, 3, padding=1)
        enc = []
        for _ in range(cfg.n_encoder_blocks):
            enc += [
                nn.GroupNorm(8, c),
                nn.SiLU(),
                nn.Conv2d(c, c, 3, padding=1),
            ]
        self.enc = nn.Sequential(*enc)
        self.down1 = nn.Conv2d(c, c, 4, stride=2, padding=1)      # 64 → 32
        self.down2 = nn.Conv2d(c, lat, 4, stride=2, padding=1)    # 32 → 16
        self.act = nn.SiLU()

    def forward(self, frame: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h64 = self.enc(self.stem(frame))
        h32 = self.act(self.down1(h64))
        h16 = self.act(self.down2(h32))
        return h16, h64


class GooseHead(nn.Module):
    """Goose ActionModel head shape: pooled FC → 5 logits, conv → 4096 logits.

    Mirrors stochastic_goose/custom_agents/action.py: an action branch over
    the pooled latent and a spatial coordinate branch upsampled back to the
    64×64 grid, concatenated to one 4101-logit vector.
    """

    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        lat = latent_channels
        self.action_branch = nn.Sequential(
            nn.Linear(lat, 256), nn.SiLU(), nn.Linear(256, GOOSE_NUM_SIMPLE)
        )
        self.coord_branch = nn.Sequential(
            nn.Conv2d(lat, lat // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(lat // 2, 1, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
        )

    def forward(self, h16: torch.Tensor) -> torch.Tensor:
        pooled = h16.mean(dim=(2, 3))
        simple = self.action_branch(pooled)                    # [B, 5]
        coords = self.coord_branch(h16).flatten(1)             # [B, 4096]
        return torch.cat([simple, coords], dim=1)              # [B, 4101]


class MegarexusCNN(nn.Module):
    _ADDITIVE_MISSING_PREFIXES = (
        "goal_conv.",
        "head_goal.",
    )

    def __init__(self, config: MegarexusCNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or MegarexusCNNConfig()
        self.config.validate()
        cfg = self.config
        c, lat = cfg.base_channels, cfg.latent_channels

        self.action_embed = StructuredActionEmbedding(cfg.action_dim)
        self.trunk = ConvTrunk(cfg)

        # LEWM_REX_ACTION_OWNER: action-conditioned latent dynamics
        self.dynamics = nn.ModuleList(
            FiLMResBlock(lat, cfg.action_dim, cfg.dropout)
            for _ in range(cfg.n_dynamics_blocks)
        )
        # click position plane lets placement actions act spatially, not just
        # through the global FiLM vector
        self.click_plane_proj = nn.Conv2d(1, lat, 1)
        nn.init.zeros_(self.click_plane_proj.weight)
        nn.init.zeros_(self.click_plane_proj.bias)

        self.up1 = nn.ConvTranspose2d(lat, c, 4, stride=2, padding=1)   # 16 → 32
        self.up2 = nn.ConvTranspose2d(c, c, 4, stride=2, padding=1)     # 32 → 64
        self.skip_gate = nn.Parameter(torch.zeros(1))
        self.head_next_frame = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, ARC_NUM_COLORS, 3, padding=1),
        )

        # STOCHASTIC_GOOSE_RUNTIME: twin Goose heads off the shared trunk
        self.goose_policy = GooseHead(lat)
        self.goose_change = GooseHead(lat)

        # PORTBAND catch head: "will the pour touch the glove?" — the glove
        # (Portband's present-state moveable object) enters as a coarse plane.
        # Catch is a question about glove-vs-pour-column geometry, so the head
        # pools features *inside the glove region* (masked pool) alongside the
        # global pool — a global mean alone erases where the glove is.
        self.glove_plane_proj = nn.Conv2d(1, lat, 1)
        nn.init.zeros_(self.glove_plane_proj.weight)
        nn.init.zeros_(self.glove_plane_proj.bias)
        self.catch_conv = nn.Sequential(
            nn.Conv2d(lat, lat, 3, padding=1), nn.SiLU(),
            nn.Conv2d(lat, lat, 3, padding=1), nn.SiLU(),
        )
        self.head_catch = nn.Sequential(
            nn.Linear(2 * lat, 64), nn.SiLU(), nn.Linear(64, 1)
        )

        # learned outcome semantics (backlog Tier 1/3). Labels are hindsight
        # pixels + environment fields ONLY — no game-specific mechanic model
        # anywhere in the loop (see notes: answer-key ban).
        # settle head: where does this pour's change come to rest?
        self.settle_conv = nn.Sequential(
            nn.Conv2d(lat, lat, 3, padding=1), nn.SiLU(),
            nn.Conv2d(lat, lat, 3, padding=1), nn.SiLU(),
        )
        self.head_settle = nn.Sequential(
            nn.Conv2d(lat, lat // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(lat // 2, 1, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
        )
        # level-up head: P(levels_completed increments | this pour) — the
        # win condition as a learning target, shares the settle features
        self.head_level = nn.Sequential(
            nn.Linear(lat, 64), nn.SiLU(), nn.Linear(64, 1)
        )
        # controllability head: does THIS component displace under THIS move?
        # (catch-head recipe: candidate plane + masked pool; action enters
        # through the shared structured embedding)
        self.control_plane_proj = nn.Conv2d(1, lat, 1)
        nn.init.zeros_(self.control_plane_proj.weight)
        nn.init.zeros_(self.control_plane_proj.bias)
        self.control_conv = nn.Sequential(
            nn.Conv2d(lat, lat, 3, padding=1), nn.SiLU(),
            nn.Conv2d(lat, lat, 3, padding=1), nn.SiLU(),
        )
        self.head_control = nn.Sequential(
            nn.Linear(2 * lat + cfg.action_dim, 64), nn.SiLU(), nn.Linear(64, 1)
        )

        # Level-completion hindsight goal head. This is a static perception
        # head: from the current frame, predict masks for [target lock/bucket,
        # key block, paint output]. Labels come from recorded scoring windows,
        # not from a hand-coded simulator.
        self.goal_conv = nn.Sequential(
            nn.Conv2d(lat, lat, 3, padding=1), nn.SiLU(),
            nn.Conv2d(lat, lat, 3, padding=1), nn.SiLU(),
        )
        self.head_goal = nn.Sequential(
            nn.Conv2d(lat, lat // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(lat // 2, 3, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
        )

        # inverse dynamics: diagnostic head over frame pairs
        self.pair_fuse = nn.Sequential(
            nn.Conv2d(2 * lat, lat, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(lat, lat, 3, padding=1),
            nn.SiLU(),
        )
        self.head_inv_action = nn.Linear(lat, ARC_NUM_ACTIONS)
        self.head_inv_click = nn.Sequential(
            nn.Conv2d(lat, lat // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(lat // 2, 1, 1),
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Accept old checkpoints that predate additive semantic heads."""
        if not strict:
            return super().load_state_dict(state_dict, strict=False, assign=assign)
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        unexpected = list(result.unexpected_keys)
        missing = list(result.missing_keys)
        bad_missing = [
            key for key in missing
            if not key.startswith(self._ADDITIVE_MISSING_PREFIXES)
        ]
        if unexpected or bad_missing:
            raise RuntimeError(
                "checkpoint state mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return result

    def _click_plane(
        self, action_id: torch.Tensor, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """[B] ids/coords → [B, 1, 16, 16] one-hot at the (coarse) click cell."""
        b = action_id.shape[0]
        plane = torch.zeros(
            b, 1, ARC_GRID_SIZE // 4, ARC_GRID_SIZE // 4, device=action_id.device
        )
        is_click = action_id == ARC_CLICK_ACTION_ID
        if is_click.any():
            idx = is_click.nonzero(as_tuple=True)[0]
            plane[idx, 0, (y[idx] // 4).long(), (x[idx] // 4).long()] = 1.0
        return plane

    def forward_dynamics(
        self,
        frame: torch.Tensor,
        action_id: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """frame [B, C_in, 64, 64] + action → next_frame_logits [B, 16, 64, 64]."""
        a = self.action_embed(action_id, x, y)
        h16, h64 = self.trunk(frame)
        h16 = h16 + self.click_plane_proj(self._click_plane(action_id, x, y))
        for block in self.dynamics:
            h16 = block(h16, a)
        d32 = F.silu(self.up1(h16))
        d64 = F.silu(self.up2(d32))
        d64 = d64 + torch.tanh(self.skip_gate) * h64
        return self.head_next_frame(d64)

    def forward_goose(self, frame: torch.Tensor) -> dict[str, torch.Tensor]:
        """frame → Goose policy/change logits, both [B, 4101]."""
        h16, _ = self.trunk(frame)
        return {
            "policy_logits": self.goose_policy(h16),
            "change_logits": self.goose_change(h16),
        }

    def forward_catch(
        self, frame: torch.Tensor, glove_plane: torch.Tensor
    ) -> torch.Tensor:
        """(frame stack, glove plane [B, 1, 16, 16]) → catch logit [B]."""
        h16, _ = self.trunk(frame)
        h = self.catch_conv(h16 + self.glove_plane_proj(glove_plane))
        global_pool = h.mean(dim=(2, 3))
        w = glove_plane / glove_plane.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        masked_pool = (h * w).sum(dim=(2, 3))
        return self.head_catch(torch.cat([global_pool, masked_pool], dim=-1)).squeeze(-1)

    def forward_settle(
        self, frame: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pour-time frame stack → (settle_logits [B, 64, 64], level_logit [B]).

        settle: per-cell "will this pour's change persist here once the
        animation finishes"; level: "does this pour complete the level".
        """
        h16, _ = self.trunk(frame)
        h = self.settle_conv(h16)
        settle_logits = self.head_settle(h).squeeze(1)
        level_logit = self.head_level(h.mean(dim=(2, 3))).squeeze(-1)
        return settle_logits, level_logit

    def forward_control(
        self,
        frame: torch.Tensor,
        control_plane: torch.Tensor,
        action_id: torch.Tensor,
    ) -> torch.Tensor:
        """(frame stack, candidate plane [B, 1, 16, 16], move action) →
        controllability logit [B]: does this component displace with the move?"""
        h16, _ = self.trunk(frame)
        h = self.control_conv(h16 + self.control_plane_proj(control_plane))
        global_pool = h.mean(dim=(2, 3))
        w = control_plane / control_plane.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
        masked_pool = (h * w).sum(dim=(2, 3))
        zeros = torch.zeros_like(action_id)
        a = self.action_embed(action_id, zeros, zeros)
        return self.head_control(
            torch.cat([global_pool, masked_pool, a], dim=-1)
        ).squeeze(-1)

    def forward_goal(self, frame: torch.Tensor) -> torch.Tensor:
        """Static frame → goal logits [B, 3, 64, 64].

        Channel order: 0=target lock/bucket, 1=key block, 2=paint output.
        """
        h16, _ = self.trunk(frame)
        return self.head_goal(self.goal_conv(h16))

    def forward_pair(
        self, frame: torch.Tensor, next_frame: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """(s_t, s_{t+1}) → inverse-dynamics predictions."""
        h_t, _ = self.trunk(frame)
        h_n, _ = self.trunk(next_frame)
        fused = self.pair_fuse(torch.cat([h_t, h_n], dim=1))
        pooled = fused.mean(dim=(2, 3))
        return {
            "inv_action_logits": self.head_inv_action(pooled),
            "inv_click_logits": self.head_inv_click(fused).flatten(1),
        }

    def forward(
        self,
        frame: torch.Tensor,
        action_id: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        next_frame: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = {"next_frame_logits": self.forward_dynamics(frame, action_id, x, y)}
        out.update(self.forward_goose(frame))
        if next_frame is not None:
            out.update(self.forward_pair(frame, next_frame))
        return out
