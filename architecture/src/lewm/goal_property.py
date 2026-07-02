"""Goal property prediction from ARC-AGI-3 trajectory prefixes.

Instead of predicting full frames (too sparse), predict goal-relevant
properties that characterize the goal state:

1. Agent position delta (Δx, Δy) — where does the agent end up?
2. Goal type (classification: keys_depleted, pattern_completion, etc.)
3. Keys collected (binary per color: did a key of this color disappear?)
4. Goal delta magnitude (how many pixels change?)

These are tractable prediction targets with clear semantics.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from lewm.sigreg import SIGReg, SIGRegConfig
from lewm.turbosoap import SOAP


ARC_GRID_SIZE = 64
ARC_NUM_COLORS = 16
ARC_NUM_ACTIONS = 7
GOAL_TYPES = ["delta_convergence", "keys_depleted", "pattern_completion", "end_of_trajectory"]
NUM_GOAL_TYPES = len(GOAL_TYPES)
NUM_MECHANICS = 168  # Number of distinct mechanic classes (concept_idx 0-167)


# ── study_tags.v001 supervision constants ─────────────────────────────────
#
# Slot ordering MUST match docs/study_tags.v001.md § Codepoint layout
# (Block A in reserved_tokens.json). Bumping the schema means bumping
# these constants too — and re-training any head that was fit to them.

STUDY_TAGS_GOAL_FAMILIES: tuple[str, ...] = (
    "reach", "collect", "avoid", "cover", "clear", "restore",
    "transform", "sort", "count", "pattern_complete", "signal",
    "multi_objective", "multi_agent", "survive", "recall", "unknown",
) + tuple(f"random_{i:02d}" for i in range(32))
NUM_STUDY_TAGS_GOAL_FAMILIES = len(STUDY_TAGS_GOAL_FAMILIES)  # 16 named + 32 random = 48
GOAL_FAMILY_TO_IDX: dict[str, int] = {
    name: i for i, name in enumerate(STUDY_TAGS_GOAL_FAMILIES)
}

STUDY_TAGS_ROLE_FIELDS: tuple[str, ...] = (
    "movable_colors", "static_colors", "collectible_colors",
    "hazard_colors", "goal_target_colors", "pushable_colors",
    "switch_colors", "door_colors", "key_colors",
)
NUM_STUDY_TAGS_ROLES = len(STUDY_TAGS_ROLE_FIELDS)  # 9
ROLE_FIELD_TO_IDX: dict[str, int] = {
    name: i for i, name in enumerate(STUDY_TAGS_ROLE_FIELDS)
}

STUDY_TAGS_CAPABILITY_FIELDS: tuple[str, ...] = (
    "requires_memory", "requires_counting", "requires_contact",
    "requires_color_binding", "requires_topology",
    "requires_sequence_order", "requires_nonlocal_causality",
)
NUM_STUDY_TAGS_CAPABILITIES = len(STUDY_TAGS_CAPABILITY_FIELDS)  # 7

STUDY_TAGS_TOPOLOGY: tuple[str, ...] = (
    "dot", "line", "cross", "L", "block", "blob", "hollow", "compound", "unknown",
)
NUM_STUDY_TAGS_TOPOLOGY = len(STUDY_TAGS_TOPOLOGY)  # 9
TOPOLOGY_TO_IDX: dict[str, int] = {
    name: i for i, name in enumerate(STUDY_TAGS_TOPOLOGY)
}
ALLOWED_INPUT_MODES = {"prefix", "result_anchored"}
ALLOWED_VISIBLE_FIELDS = {"frames", "actions"}
FORBIDDEN_VISIBLE_FIELDS = {
    "mechanic", "family", "subclass", "stage", "goal_type", "won", "seed",
    "game_id", "source", "study_tags", "player", "objects", "goal",
    "transitions", "audit", "curriculum",
}


# ── Dataset ─────────────────────────────────────────────────────────────────

class GoalPropertyDataset(Dataset):
    """Predict goal properties from trajectory prefixes.

    Lazily loads and one-hot encodes frames in __getitem__ to avoid OOM.
    Stores only lightweight index + metadata in __init__.

    Targets:
        - agent_delta: [2] float (Δx, Δy) or [0,0] if agent didn't move
        - goal_type: int (classification)
        - keys_collected: [16] binary (which colors were collected)
        - goal_delta_px: int (how many pixels change)
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
        frame_encoding: str = "cpu_onehot",
        input_mode: str = "prefix",
        visible_fields: tuple[str, ...] | list[str] = ("frames", "actions"),
        _preloaded: tuple | None = None,
    ) -> None:
        if context_len < 1:
            raise ValueError("context_len must be >= 1")
        if input_mode not in ALLOWED_INPUT_MODES:
            raise ValueError(f"unsupported input_mode: {input_mode}")
        visible = tuple(str(f).strip() for f in visible_fields if str(f).strip())
        if "frames" not in visible:
            raise ValueError("visible_fields must include frames")
        unknown_visible = set(visible).difference(ALLOWED_VISIBLE_FIELDS)
        forbidden_visible = set(visible).intersection(FORBIDDEN_VISIBLE_FIELDS)
        if unknown_visible or forbidden_visible:
            raise ValueError(
                "visible_fields may only contain frames/actions; "
                f"unknown={sorted(unknown_visible)} forbidden={sorted(forbidden_visible)}"
            )
        self.context_len = context_len
        self.input_mode = input_mode
        self.visible_fields = visible
        self.rng = random.Random(seed)
        self.ARC_NUM_COLORS = ARC_NUM_COLORS
        if frame_encoding not in {"cpu_onehot", "gpu_onehot"}:
            raise ValueError(f"unsupported frame_encoding: {frame_encoding}")
        self.frame_encoding = frame_encoding

        if _preloaded is not None:
            pd, gd = _preloaded
        else:
            pd = torch.load(Path(pattern_dataset_path), map_location="cpu", weights_only=False)
            gd = torch.load(Path(goals_dataset_path), map_location="cpu", weights_only=False)

        self.pattern_data = pd["data"][split]
        self.goal_metadata = {m["game_id"]: m for m in gd[split]}

        # Build index: (episode_index, start_idx) + metadata
        self.index: list[dict] = []
        for ep_idx, item in enumerate(self.pattern_data):
            gid = item["game_id"]
            meta = self.goal_metadata.get(gid)
            if meta is None:
                continue

            goal_idx = meta["goal_idx"]

            sample_count = max(1, samples_per_game)
            for _ in range(sample_count):
                if input_mode == "prefix":
                    max_start = max(0, goal_idx - context_len + 1)
                    if max_start < 1:
                        continue
                    start_idx = self.rng.randint(0, max_start - 1) if max_start > 1 else 0
                    end_idx = start_idx + context_len
                    pad_left = 0
                else:
                    end_idx = goal_idx + 1
                    start_idx = max(0, end_idx - context_len)
                    pad_left = max(0, context_len - (end_idx - start_idx))

                # Store agent delta
                if meta.get("agent_moved", False) and meta.get("agent_start") and meta.get("agent_end"):
                    ax, ay = meta["agent_start"]
                    bx, by = meta["agent_end"]
                    agent_delta = (bx - ax, by - ay)
                else:
                    agent_delta = (0.0, 0.0)

                # Keys collected
                keys_collected = [0.0] * ARC_NUM_COLORS
                for k_color in meta.get("keys_disappeared", []):
                    if isinstance(k_color, int) and 0 <= k_color < ARC_NUM_COLORS:
                        keys_collected[k_color] = 1.0

                # Goal type
                goal_type_str = meta.get("goal_type", "end_of_trajectory")
                try:
                    goal_type_idx = GOAL_TYPES.index(goal_type_str)
                except ValueError:
                    goal_type_idx = GOAL_TYPES.index("end_of_trajectory")

                self.index.append({
                    "ep_idx": ep_idx,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "pad_left": pad_left,
                    "agent_delta": agent_delta,
                    "goal_type": goal_type_idx,
                    "keys_collected": keys_collected,
                    "goal_delta_px": float(meta.get("goal_delta_px", 0)),
                    "game_id": gid,
                    "game_idx": meta["concept_idx"],
                    "family_idx": meta.get("family_idx"),
                    "subclass_idx": meta.get("subclass_idx"),
                    "stage_idx": meta.get("stage_idx"),
                    "goal_idx": goal_idx,
                    # Affordance head supervision (iter 19) — per-action
                    # usefulness target + mask. Absent on pre-iter-19
                    # datasets; the loss term silently skips when missing.
                    "affordance": meta.get("affordance"),
                    "affordance_mask": meta.get("affordance_mask"),
                })

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        entry = self.index[idx]
        item = self.pattern_data[entry["ep_idx"]]

        frames = item["frames"]
        actions = item["actions"]
        start_idx = entry["start_idx"]
        end_idx = entry.get("end_idx", start_idx + self.context_len)
        pad_left = int(entry.get("pad_left", 0))

        chunk = frames[start_idx:end_idx]
        if pad_left > 0:
            if chunk.shape[0] == 0:
                pad_frame = frames[:1].expand(pad_left, *frames.shape[1:])
            else:
                pad_frame = chunk[:1].expand(pad_left, *chunk.shape[1:])
            chunk = torch.cat([pad_frame, chunk], dim=0)
        if chunk.shape[0] < self.context_len:
            pad_count = self.context_len - chunk.shape[0]
            pad_frame = chunk[:1].expand(pad_count, *chunk.shape[1:])
            chunk = torch.cat([pad_frame, chunk], dim=0)
        elif chunk.shape[0] > self.context_len:
            chunk = chunk[-self.context_len:]

        if self.frame_encoding == "gpu_onehot":
            context_frames = chunk.contiguous()
        elif chunk.dim() == 3:
            # One-hot encode on the fly (avoids storing 56GB of float tensors).
            context_frames = F.one_hot(
                chunk.long(), num_classes=self.ARC_NUM_COLORS
            ).permute(0, 3, 1, 2).contiguous().float()
        else:
            context_frames = chunk.float()

        if self.context_len > 1 and "actions" in self.visible_fields:
            action_end = max(start_idx, end_idx - 1)
            context_actions = actions[start_idx:action_end].long()
            if pad_left > 0:
                prepad = actions.new_zeros((pad_left,), dtype=torch.long)
                context_actions = torch.cat([prepad, context_actions], dim=0)
            if context_actions.shape[0] < self.context_len - 1:
                pad_count = self.context_len - 1 - context_actions.shape[0]
                postpad = actions.new_zeros((pad_count,), dtype=torch.long)
                context_actions = torch.cat([context_actions, postpad], dim=0)
            elif context_actions.shape[0] > self.context_len - 1:
                context_actions = context_actions[-(self.context_len - 1):]
        else:
            context_actions = actions.new_zeros((max(0, self.context_len - 1),), dtype=torch.long)

        item = {
            "context_frames": context_frames,
            "context_actions": context_actions,
            "agent_delta": torch.tensor(entry["agent_delta"], dtype=torch.float),
            "goal_type": entry["goal_type"],
            "keys_collected": torch.tensor(entry["keys_collected"], dtype=torch.float),
            "goal_delta_px": entry["goal_delta_px"],
            "game_id": entry["game_id"],
            "game_idx": entry["game_idx"],
            "family_idx": entry["family_idx"],
            "subclass_idx": entry["subclass_idx"],
            "stage_idx": entry["stage_idx"],
            "goal_idx": entry["goal_idx"],
        }
        # Affordance head supervision — only surfaced when the dataset
        # carries it (iter-19+ builds). The collate fn + loss term gate
        # on presence, so older datasets train unaffected.
        aff = entry.get("affordance")
        if aff is not None:
            item["affordance"] = torch.as_tensor(aff, dtype=torch.float)
            am = entry.get("affordance_mask")
            item["affordance_mask"] = (
                torch.as_tensor(am, dtype=torch.bool) if am is not None
                else ~torch.isnan(item["affordance"])
            )
        return item


# ── Model ───────────────────────────────────────────────────────────────────

class FrameCNNEncoder(nn.Module):
    """Lightweight CNN for ARC frames."""

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
            B, x = frames.shape[0], frames
            flat = False

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


class TokenRMSNorm(nn.Module):
    """Small RMSNorm implementation to keep the Mockingbird path self-contained."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight.to(dtype=x.dtype)


class MockingbirdSelfAttention(nn.Module):
    """Full-prefix attention with Mockingbird-style per-head QK gain."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        qk_gain_init: float = 5.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.qk_gain = nn.Parameter(torch.full((num_heads,), float(qk_gain_init)))
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        scale = (self.head_dim ** -0.5) * self.qk_gain.to(dtype=x.dtype).view(1, self.num_heads, 1, 1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        return self.out_proj(y)


class MockingbirdMLP(nn.Module):
    """Leaky-ReLU-square MLP used as a compact Mockingbird-inspired block MLP."""

    def __init__(self, hidden_dim: int, ff_mult: float = 3.75) -> None:
        super().__init__()
        ff_dim = max(1, int(hidden_dim * ff_mult))
        self.up = nn.Linear(hidden_dim, ff_dim, bias=False)
        self.down = nn.Linear(ff_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.up(x), negative_slope=0.1).square()
        return self.down(h)


class MockingbirdBlock(nn.Module):
    """Residual-mix block: RMSNorm, QK-gain attention, scaled MLP."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        ff_mult: float = 3.75,
        qk_gain_init: float = 5.0,
        dropout: float = 0.1,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.attn_norm = TokenRMSNorm(hidden_dim)
        self.mlp_norm = TokenRMSNorm(hidden_dim)
        self.attn = MockingbirdSelfAttention(
            hidden_dim, num_heads, qk_gain_init=qk_gain_init, dropout=dropout,
        )
        self.mlp = MockingbirdMLP(hidden_dim, ff_mult=ff_mult)
        self.attn_scale = nn.Parameter(torch.ones(hidden_dim))
        self.mlp_scale = nn.Parameter(torch.ones(hidden_dim))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(hidden_dim), torch.zeros(hidden_dim))).float())
        self.ln_scale_factor = 1.0 / ((layer_idx + 1) ** 0.5)

    def forward(self, x: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x_in) * self.ln_scale_factor)
        x_out = x_in + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        mlp_out = self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)
        return x_out + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * mlp_out


class MockingbirdTrajectoryEncoder(nn.Module):
    """Deep skip+loop token encoder inspired by the archived Mockingbird body."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        *,
        dropout: float = 0.1,
        ff_mult: float = 3.75,
        qk_gain_init: float = 5.0,
        num_loops: int = 2,
        loop_start: int = 3,
        loop_end: int = 5,
        skip_gates: bool = True,
    ) -> None:
        super().__init__()
        if not (0 <= loop_start <= loop_end < num_layers):
            raise ValueError("loop_start/loop_end must define a valid layer span")
        self.num_layers = num_layers
        self.num_loops = num_loops
        self.loop_start = loop_start
        self.loop_end = loop_end
        self.blocks = nn.ModuleList(
            [
                MockingbirdBlock(
                    hidden_dim,
                    num_heads,
                    ff_mult=ff_mult,
                    qk_gain_init=qk_gain_init,
                    dropout=dropout,
                    layer_idx=i,
                )
                for i in range(num_layers)
            ]
        )
        indices = self._active_indices()
        self.num_encoder_layers = len(indices) // 2
        self.encoder_indices = indices[: self.num_encoder_layers]
        self.decoder_indices = indices[self.num_encoder_layers :]
        self.num_skip_weights = min(len(self.encoder_indices), len(self.decoder_indices))
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, hidden_dim))
        self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, hidden_dim)) if skip_gates else None
        self.final_norm = TokenRMSNorm(hidden_dim)

    def _active_indices(self) -> list[int]:
        if self.num_loops <= 0:
            return list(range(self.num_layers))
        loop_segment = list(range(self.loop_start, self.loop_end + 1))
        out = list(range(self.loop_start))
        for _ in range(self.num_loops + 1):
            out.extend(loop_segment)
        out.extend(range(self.loop_end + 1, self.num_layers))
        return out

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = F.rms_norm(tokens, (tokens.shape[-1],))
        x0 = x
        skips: list[torch.Tensor] = []
        for idx in self.encoder_indices:
            x = self.blocks[idx](x, x0)
            skips.append(x)
        for skip_idx, idx in enumerate(self.decoder_indices):
            if skip_idx < self.num_skip_weights and skips:
                skip = skips.pop()
                weighted = self.skip_weights[skip_idx].to(dtype=x.dtype)[None, None, :] * skip
                if self.skip_gates is not None:
                    gate = torch.sigmoid(self.skip_gates[skip_idx].to(dtype=x.dtype))[None, None, :]
                    x = torch.lerp(weighted, x, gate)
                else:
                    x = x + weighted
            x = self.blocks[idx](x, x0)
        return self.final_norm(x)


class GoalPropertyModel(nn.Module):
    """Predict goal properties from trajectory prefix.

    Architecture: FrameCNNEncoder + ActionEmbedder → Transformer → multiple heads
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        model_arch: str = "standard",
        mockingbird_ff_mult: float = 3.75,
        mockingbird_qk_gain: float = 5.0,
        mockingbird_num_loops: int = 2,
        mockingbird_loop_start: int = 3,
        mockingbird_loop_end: int = 5,
        mockingbird_skip_gates: bool = True,
        num_mechanics: int = 0,
        num_families: int = 0,
        num_subclasses: int = 0,
        num_stages: int = 0,
        study_tags_heads: bool = False,
        conditional_mechanic_head: bool = False,
        hierarchical_mechanic_head: bool = False,
        mechanics_per_family: "list[int] | None" = None,
        concept_to_flat: "list[int] | torch.Tensor | None" = None,
        hier_additive_mode: bool = False,
        hier_mechanic_weight: float = 0.0,
        affordance_head: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        if model_arch not in {"standard", "mockingbird"}:
            raise ValueError(f"unsupported model_arch: {model_arch}")
        self.model_arch = model_arch

        self.frame_encoder = FrameCNNEncoder(cnn_dim=64, hidden_dim=hidden_dim)
        self.action_embedder = ActionEmbedder(hidden_dim=hidden_dim)
        self.pos_embed = nn.Embedding(64, hidden_dim)
        nn.init.trunc_normal_(self.pos_embed.weight, std=0.02)

        if model_arch == "standard":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers, enable_nested_tensor=False,
            )
        else:
            self.mockingbird_encoder = MockingbirdTrajectoryEncoder(
                hidden_dim,
                num_layers,
                num_heads,
                dropout=dropout,
                ff_mult=mockingbird_ff_mult,
                qk_gain_init=mockingbird_qk_gain,
                num_loops=mockingbird_num_loops,
                loop_start=mockingbird_loop_start,
                loop_end=mockingbird_loop_end,
                skip_gates=mockingbird_skip_gates,
            )

        # Prediction heads
        self.agent_head = nn.Linear(hidden_dim, 2)  # Δx, Δy
        self.goal_type_head = nn.Linear(hidden_dim, NUM_GOAL_TYPES)
        self.keys_head = nn.Linear(hidden_dim, ARC_NUM_COLORS)  # binary per color
        self.delta_px_head = nn.Linear(hidden_dim, 1)  # regression
        # Hierarchical-mechanic head replaces both flat mechanic_head and
        # flat family_head when enabled. Decomposes the 4096-way softmax
        # into a 33-way family softmax × per-family within-family softmax,
        # giving the long tail cleaner gradients per step.
        #
        # When hier_additive_mode=True the flat heads ALSO exist; the
        # forward pass combines flat + hier logits via
        # compute_hier_additive_logits and the loss uses
        # compute_hier_additive_loss. Set hier_mechanic_weight=0 to
        # silence the hier head at output time without removing it
        # from the graph.
        self.hierarchical_mechanic_head_enabled = bool(hierarchical_mechanic_head)
        self.hier_additive_mode = bool(hier_additive_mode)
        self.hier_mechanic_weight = float(hier_mechanic_weight)
        if hier_additive_mode and not hierarchical_mechanic_head:
            raise ValueError(
                "hier_additive_mode=True requires hierarchical_mechanic_head=True"
            )
        if hierarchical_mechanic_head:
            if mechanics_per_family is None or len(mechanics_per_family) == 0:
                raise ValueError(
                    "hierarchical_mechanic_head=True requires mechanics_per_family"
                )
            total = int(sum(mechanics_per_family))
            if num_mechanics > 0 and num_mechanics != total:
                raise ValueError(
                    f"sum(mechanics_per_family)={total} must match "
                    f"num_mechanics={num_mechanics} when both supplied"
                )
            if num_families > 0 and num_families != len(mechanics_per_family):
                raise ValueError(
                    f"len(mechanics_per_family)={len(mechanics_per_family)} must "
                    f"match num_families={num_families} when both supplied"
                )
            from .hierarchical_heads import HierarchicalMechanicHead
            self.hier_mechanic_head = HierarchicalMechanicHead(
                hidden_dim,
                num_families=len(mechanics_per_family),
                mechanics_per_family=list(mechanics_per_family),
            )
            # Permutation from concept_idx (dataset-native) to flat_idx
            # (family-contiguous slot inside hier head). When provided,
            # the model both (a) re-permutes hier_mechanic_logits back
            # to concept-axis when populating outputs["mechanic_logits"],
            # so downstream argmax / class-weights keep working; and
            # (b) loss path can remap concept targets to flat targets.
            if concept_to_flat is None:
                # Default identity permutation (caller asserts concepts are
                # already family-contiguous). Length must match total.
                c2f_tensor = torch.arange(total, dtype=torch.long)
            else:
                c2f_tensor = torch.as_tensor(concept_to_flat, dtype=torch.long)
                if c2f_tensor.shape != (total,):
                    raise ValueError(
                        f"concept_to_flat shape {tuple(c2f_tensor.shape)} must "
                        f"equal ({total},) = sum(mechanics_per_family)"
                    )
            # Inverse: flat slot k → concept_idx.
            flat_to_concept_tensor = torch.empty_like(c2f_tensor)
            flat_to_concept_tensor[c2f_tensor] = torch.arange(total, dtype=torch.long)
            self.register_buffer(
                "concept_to_flat", c2f_tensor, persistent=False,
            )
            self.register_buffer(
                "flat_to_concept", flat_to_concept_tensor, persistent=False,
            )
            if hier_additive_mode:
                # Additive mode: keep flat heads alongside the hier head.
                # forward() will combine outputs via compute_hier_additive_logits.
                self.mechanic_head = nn.Linear(hidden_dim, num_mechanics) if num_mechanics > 0 else None
                self.family_head = nn.Linear(hidden_dim, num_families) if num_families > 0 else None
            else:
                # Replacement mode (default): flat heads removed; hier is canonical.
                self.mechanic_head = None
                self.family_head = None
        else:
            self.hier_mechanic_head = None
            self.mechanic_head = nn.Linear(hidden_dim, num_mechanics) if num_mechanics > 0 else None
            self.family_head = nn.Linear(hidden_dim, num_families) if num_families > 0 else None
        self.subclass_head = nn.Linear(hidden_dim, num_subclasses) if num_subclasses > 0 else None
        self.stage_head = nn.Linear(hidden_dim, num_stages) if num_stages > 0 else None

        # study_tags.v001 supervision heads. Gated so existing training paths
        # are unaffected — only allocate when the caller opts in.
        self.study_tags_heads_enabled = bool(study_tags_heads)
        if study_tags_heads:
            self.goal_family_head = nn.Linear(
                hidden_dim, NUM_STUDY_TAGS_GOAL_FAMILIES,
            )
            # role_buckets_head outputs [B, ARC_NUM_COLORS * NUM_STUDY_TAGS_ROLES]
            # then forward() reshapes to [B, 16, 9]. Each entry is a binary
            # logit: "is color c assigned to role r?"
            self.role_buckets_head = nn.Linear(
                hidden_dim, ARC_NUM_COLORS * NUM_STUDY_TAGS_ROLES,
            )
            self.capability_head = nn.Linear(hidden_dim, NUM_STUDY_TAGS_CAPABILITIES)
            self.topology_head = nn.Linear(hidden_dim, NUM_STUDY_TAGS_TOPOLOGY)
        else:
            self.goal_family_head = None
            self.role_buckets_head = None
            self.capability_head = None
            self.topology_head = None

        # Capability-conditional mechanic head: predicts mechanic conditioned
        # on (trunk latent, predicted capability vector, predicted role table).
        # Reduces effective branching factor from num_mechanics to ~mechanics
        # sharing the same (capability, role) signature.
        #
        # Requires study_tags_heads=True (we condition on those head outputs).
        # The mechanic_head still exists for backward compat with
        # unconditional prediction; the conditional head is an ADDITIONAL
        # path consumed only when both flags are set.
        self.conditional_mechanic_head_enabled = bool(conditional_mechanic_head)
        if conditional_mechanic_head:
            if not study_tags_heads:
                raise ValueError(
                    "conditional_mechanic_head=True requires study_tags_heads=True"
                )
            if num_mechanics <= 0:
                raise ValueError(
                    "conditional_mechanic_head=True requires num_mechanics > 0"
                )
            cond_dim = (
                hidden_dim
                + NUM_STUDY_TAGS_CAPABILITIES
                + ARC_NUM_COLORS * NUM_STUDY_TAGS_ROLES
            )
            self.cond_mechanic_head = nn.Sequential(
                nn.Linear(cond_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, num_mechanics),
            )
        else:
            self.cond_mechanic_head = None

        # Affordance head (AGWM, arXiv 2605.06841): predict, per ARC
        # action slot, that action's likely usefulness — conditioned on
        # the capability vector (which Lewis already produces) and
        # decoupled from full dynamics. This is the head that connects
        # Lewis output to the action space: agent_delta predicts where
        # the *agent* moves, NOT which action/cell to pick — a category
        # error the eval bridge proved (iter 16-17). affordance_logits
        # is a per-action signal the play policy can argmax directly.
        self.affordance_head_enabled = bool(affordance_head)
        if affordance_head:
            if not study_tags_heads:
                raise ValueError(
                    "affordance_head=True requires study_tags_heads=True "
                    "(it conditions on the capability vector)"
                )
            aff_dim = hidden_dim + NUM_STUDY_TAGS_CAPABILITIES
            self.affordance_head = nn.Sequential(
                nn.Linear(aff_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, ARC_NUM_ACTIONS),
            )
        else:
            self.affordance_head = None

    def encode_trajectory(
        self, context_frames: torch.Tensor, context_actions: torch.Tensor,
    ) -> torch.Tensor:
        """Encode trajectory into hidden representation."""
        if context_frames.dim() == 4:
            onehot_dtype = torch.float32
            if context_frames.is_cuda and torch.is_autocast_enabled("cuda"):
                onehot_dtype = torch.get_autocast_dtype("cuda")
            frame_idx = context_frames.long().unsqueeze(2)
            context_frames = torch.zeros(
                (
                    context_frames.shape[0],
                    context_frames.shape[1],
                    ARC_NUM_COLORS,
                    context_frames.shape[2],
                    context_frames.shape[3],
                ),
                device=context_frames.device,
                dtype=onehot_dtype,
            )
            context_frames.scatter_(2, frame_idx, 1)
        B, K = context_frames.shape[:2]
        frame_tokens = self.frame_encoder(context_frames)

        if K > 1:
            action_tokens = self.action_embedder(context_actions)
            seq_len = 2 * K - 1
            tokens = torch.zeros(B, seq_len, self.hidden_dim, device=context_frames.device)
            tokens[:, 0::2, :] = frame_tokens
            tokens[:, 1::2, :] = action_tokens
        else:
            tokens = frame_tokens

        positions = torch.arange(tokens.shape[1], device=context_frames.device)
        tokens = tokens + self.pos_embed(positions.unsqueeze(0).expand(B, -1))
        if self.model_arch == "standard":
            x = self.transformer(tokens)
        else:
            x = self.mockingbird_encoder(tokens)
        return x[:, -1, :]  # [B, D]

    def forward(
        self,
        context_frames: torch.Tensor,
        context_actions: torch.Tensor,
        *,
        include_latent: bool = False,
    ) -> dict[str, torch.Tensor]:
        h = self.encode_trajectory(context_frames, context_actions)
        outputs = {
            "agent_delta": self.agent_head(h),             # [B, 2]
            "goal_type_logits": self.goal_type_head(h),     # [B, 4]
            "keys_logits": self.keys_head(h),               # [B, 16]
            "delta_px": self.delta_px_head(h).squeeze(-1),  # [B]
        }
        if self.hier_mechanic_head is not None:
            hier_out = self.hier_mechanic_head(h)
            # Re-permute hier logits to concept-axis (dataset-native).
            hier_in_concept = hier_out["hier_mechanic_logits"].index_select(
                1, self.concept_to_flat,
            )
            outputs["hier_mechanic_logits"] = hier_out["hier_mechanic_logits"]
            outputs["within_family_logits"] = hier_out["within_family_logits"]
            outputs["family_mask"] = hier_out["family_mask"]
            outputs["family_offsets"] = hier_out["family_offsets"]
            outputs["concept_to_flat"] = self.concept_to_flat
            outputs["flat_to_concept"] = self.flat_to_concept
            if self.hier_additive_mode and self.mechanic_head is not None:
                # Additive: flat is canonical, hier adds a weighted bias.
                flat_logits = self.mechanic_head(h)
                outputs["flat_mechanic_logits"] = flat_logits
                outputs["mechanic_logits"] = (
                    flat_logits + self.hier_mechanic_weight * hier_in_concept
                )
                if self.family_head is not None:
                    outputs["flat_family_logits"] = self.family_head(h)
                    outputs["family_logits"] = outputs["flat_family_logits"]
                else:
                    outputs["family_logits"] = hier_out["family_logits"]
                outputs["hier_family_logits"] = hier_out["family_logits"]
            else:
                # Replacement: hier IS the canonical mechanic / family path.
                outputs["mechanic_logits"] = hier_in_concept
                outputs["family_logits"] = hier_out["family_logits"]
        else:
            if self.mechanic_head is not None:
                outputs["mechanic_logits"] = self.mechanic_head(h)
            if self.family_head is not None:
                outputs["family_logits"] = self.family_head(h)
        if self.subclass_head is not None:
            outputs["subclass_logits"] = self.subclass_head(h)
        if self.stage_head is not None:
            outputs["stage_logits"] = self.stage_head(h)
        if self.study_tags_heads_enabled:
            B = h.shape[0]
            outputs["goal_family_logits"] = self.goal_family_head(h)         # [B, 46]
            role_logits_flat = self.role_buckets_head(h)                     # [B, 16*9]
            outputs["role_buckets_logits"] = role_logits_flat.view(
                B, ARC_NUM_COLORS, NUM_STUDY_TAGS_ROLES,
            )                                                                # [B, 16, 9]
            outputs["capability_logits"] = self.capability_head(h)           # [B, 7]
            outputs["topology_logits"] = self.topology_head(h)               # [B, 9]
            # Conditional mechanic head: concatenate trunk + sigmoided
            # study_tags features (detached so the conditional path can't
            # corrupt the supervised study_tags heads' gradients).
            if self.conditional_mechanic_head_enabled:
                cap_feats = torch.sigmoid(outputs["capability_logits"]).detach()
                role_feats = torch.sigmoid(role_logits_flat).detach()
                cond_input = torch.cat([h, cap_feats, role_feats], dim=-1)
                outputs["cond_mechanic_logits"] = self.cond_mechanic_head(cond_input)
            # Affordance head: per-action usefulness, conditioned on the
            # capability vector (detached — affordance learning must not
            # corrupt the supervised capability head's gradients).
            if self.affordance_head_enabled:
                cap_feats = torch.sigmoid(outputs["capability_logits"]).detach()
                aff_input = torch.cat([h, cap_feats], dim=-1)
                outputs["affordance_logits"] = self.affordance_head(aff_input)
        if include_latent:
            outputs["latent"] = h
        return outputs


# ── Loss ────────────────────────────────────────────────────────────────────

def goal_property_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    agent_weight: float = 1.0,
    goal_type_weight: float = 0.5,
    keys_weight: float = 0.5,
    delta_px_weight: float = 0.1,
    mechanic_weight: float = 1.0,
    family_weight: float = 1.0,
    subclass_weight: float = 1.0,
    stage_weight: float = 0.25,
    cond_mechanic_weight: float = 1.0,
    affordance_weight: float = 0.5,
    hier_family_ce_weight: float = 1.0,
    hier_within_family_ce_weight: float = 1.0,
    goal_type_class_weights: torch.Tensor | None = None,
    mechanic_class_weights: torch.Tensor | None = None,
    family_class_weights: torch.Tensor | None = None,
    active_mechanic_indices: torch.Tensor | None = None,
    mechanic_label_smoothing: float = 0.0,
    family_label_smoothing: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Multi-task loss for goal property prediction.

    Args:
        outputs: model outputs
        targets: ground truth dict with keys:
            agent_delta: [B, 2]
            goal_type: [B] long
            keys_collected: [B, 16] float binary
            goal_delta_px: [B] float
            mechanic: [B] long — global mechanic classification
            family: [B] long — mechanic-family classification
            subclass: [B] long — mechanic-subclass classification
            stage: [B] long — curriculum-stage classification

    Returns:
        dict with total and per-task losses
    """
    losses = {}

    # Agent delta: MSE
    ad = F.mse_loss(outputs["agent_delta"], targets["agent_delta"])
    losses["agent"] = ad * agent_weight

    # Goal type: cross-entropy
    gt = F.cross_entropy(
        outputs["goal_type_logits"],
        targets["goal_type"],
        weight=goal_type_class_weights,
    )
    losses["goal_type"] = gt * goal_type_weight

    # Keys collected: binary cross-entropy
    kc = F.binary_cross_entropy_with_logits(
        outputs["keys_logits"], targets["keys_collected"]
    )
    losses["keys"] = kc * keys_weight

    # Delta pixels: MSE (scaled)
    dp = F.mse_loss(outputs["delta_px"], targets["goal_delta_px"])
    losses["delta_px"] = dp * delta_px_weight

    # Mechanic + family: three regimes —
    #   * hier-additive: flat CE on flat_mechanic_logits + hier CE on hier
    #     outputs, both contribute to total. flat is canonical for argmax.
    #   * hier-replace: hier CE decomposes family + within-family.
    #   * flat: standard flat CE on mechanic_logits + family CE.
    hier_present = (
        "hier_mechanic_logits" in outputs
        and "family_offsets" in outputs
        and "family_mask" in outputs
        and "mechanic" in targets
        and "family" in targets
        and active_mechanic_indices is None
    )
    hier_additive = hier_present and "flat_mechanic_logits" in outputs
    hier_active = hier_present and not hier_additive

    if hier_additive:
        # Flat CE on the explicit flat head; hier CE on the hier head.
        flat_logits = outputs["flat_mechanic_logits"]
        mc = F.cross_entropy(
            flat_logits,
            targets["mechanic"],
            weight=mechanic_class_weights,
            label_smoothing=mechanic_label_smoothing,
        )
        losses["mechanic"] = mc * mechanic_weight
        flat_fam = outputs.get("flat_family_logits", outputs["family_logits"])
        fc = F.cross_entropy(
            flat_fam,
            targets["family"],
            weight=family_class_weights,
            label_smoothing=family_label_smoothing,
        )
        losses["family"] = fc * family_weight
        # Hier CE — same machinery as replacement path; emitted as
        # separate loss terms so the operator can dial via *_weight.
        from .hierarchical_heads import hierarchical_mechanic_loss
        offsets = outputs["family_offsets"]
        total_mech = int(offsets[-1].item())
        m2f = torch.zeros(total_mech, dtype=torch.long, device=offsets.device)
        for f in range(offsets.shape[0] - 1):
            lo, hi = int(offsets[f].item()), int(offsets[f + 1].item())
            m2f[lo:hi] = f
        concept_to_flat = outputs["concept_to_flat"]
        flat_mechanic_target = concept_to_flat.to(
            device=targets["mechanic"].device,
        ).gather(0, targets["mechanic"].long())
        hier_outputs = {
            "family_logits": outputs["hier_family_logits"],
            "within_family_logits": outputs["within_family_logits"],
            "family_offsets": outputs["family_offsets"],
            "family_mask": outputs["family_mask"],
        }
        hier_losses = hierarchical_mechanic_loss(
            hier_outputs,
            targets["family"],
            flat_mechanic_target,
            m2f,
            family_ce_weight=hier_family_ce_weight,
            within_family_ce_weight=hier_within_family_ce_weight,
        )
        losses["hier_family"] = hier_losses["family_ce"] * family_weight
        losses["hier_mechanic"] = (
            hier_losses["within_family_ce"] * mechanic_weight
        )
    elif hier_active:
        from .hierarchical_heads import hierarchical_mechanic_loss
        hier_outputs = {
            "family_logits": outputs["family_logits"],
            "within_family_logits": outputs["within_family_logits"],
            "family_offsets": outputs["family_offsets"],
            "family_mask": outputs["family_mask"],
        }
        # Build mechanic_to_family on the fly from family_offsets (small
        # tensor, ~num_mechanics longs; constructed in the loss closure
        # so it lives on the right device).
        offsets = outputs["family_offsets"]
        total_mech = int(offsets[-1].item())
        m2f = torch.zeros(total_mech, dtype=torch.long, device=offsets.device)
        for f in range(offsets.shape[0] - 1):
            lo, hi = int(offsets[f].item()), int(offsets[f + 1].item())
            m2f[lo:hi] = f
        # Remap concept-axis mechanic targets to flat-axis (family-contiguous)
        # before computing within-family CE.
        concept_to_flat = outputs["concept_to_flat"]
        flat_mechanic_target = concept_to_flat.to(
            device=targets["mechanic"].device,
        ).gather(0, targets["mechanic"].long())
        hier_losses = hierarchical_mechanic_loss(
            hier_outputs,
            targets["family"],
            flat_mechanic_target,
            m2f,
            family_ce_weight=hier_family_ce_weight,
            within_family_ce_weight=hier_within_family_ce_weight,
        )
        # Emit per-component losses with familiar names so downstream
        # logs / balancer task-keys keep working (family / mechanic),
        # weighted by family_weight / mechanic_weight respectively.
        losses["family"] = hier_losses["family_ce"] * family_weight
        losses["mechanic"] = hier_losses["within_family_ce"] * mechanic_weight
    else:
        if "mechanic_logits" in outputs and "mechanic" in targets:
            mechanic_logits = outputs["mechanic_logits"]
            mechanic_targets = targets["mechanic"]
            if active_mechanic_indices is not None:
                active_mechanic_indices = active_mechanic_indices.to(mechanic_logits.device)
                mechanic_logits = mechanic_logits.index_select(1, active_mechanic_indices)
                remap = torch.full(
                    (outputs["mechanic_logits"].shape[1],),
                    -1,
                    dtype=torch.long,
                    device=mechanic_logits.device,
                )
                remap[active_mechanic_indices] = torch.arange(
                    active_mechanic_indices.numel(),
                    dtype=torch.long,
                    device=mechanic_logits.device,
                )
                mechanic_targets = remap[mechanic_targets]
                if (mechanic_targets < 0).any():
                    raise ValueError("mechanic target missing from active mechanic mask")
            mc = F.cross_entropy(
                mechanic_logits,
                mechanic_targets,
                weight=mechanic_class_weights,
                label_smoothing=mechanic_label_smoothing,
            )
            losses["mechanic"] = mc * mechanic_weight
        if "family_logits" in outputs and "family" in targets:
            fc = F.cross_entropy(
                outputs["family_logits"],
                targets["family"],
                weight=family_class_weights,
                label_smoothing=family_label_smoothing,
            )
            losses["family"] = fc * family_weight
    if "subclass_logits" in outputs and "subclass" in targets:
        sc = F.cross_entropy(outputs["subclass_logits"], targets["subclass"])
        losses["subclass"] = sc * subclass_weight
    if "stage_logits" in outputs and "stage" in targets:
        st = F.cross_entropy(outputs["stage_logits"], targets["stage"])
        losses["stage"] = st * stage_weight

    # Capability-conditional mechanic head: cross-entropy against the same
    # mechanic targets as the unconditional head, using the active-mask
    # reduction when supplied. Gradient flow is structured so this loss
    # trains the conditional head only — the capability/role heads were
    # detached in forward().
    if "cond_mechanic_logits" in outputs and "mechanic" in targets:
        cond_logits = outputs["cond_mechanic_logits"]
        cond_targets = targets["mechanic"]
        if active_mechanic_indices is not None:
            active_idxs = active_mechanic_indices.to(cond_logits.device)
            cond_logits = cond_logits.index_select(1, active_idxs)
            remap = torch.full(
                (outputs["cond_mechanic_logits"].shape[1],),
                -1, dtype=torch.long, device=cond_logits.device,
            )
            remap[active_idxs] = torch.arange(
                active_idxs.numel(), dtype=torch.long, device=cond_logits.device,
            )
            cond_targets = remap[cond_targets]
            if (cond_targets < 0).any():
                raise ValueError("mechanic target missing from active mechanic mask")
        cm = F.cross_entropy(
            cond_logits, cond_targets, weight=mechanic_class_weights,
        )
        losses["cond_mechanic"] = cm * cond_mechanic_weight

    # Affordance head: per-action usefulness. Target is binary per
    # action slot (1 = action led to state-change/progress in this
    # context). Sparse — only supervised where the rollout recorded an
    # effect — so an optional mask zeroes unsupervised slots.
    if "affordance_logits" in outputs and "affordance" in targets:
        aff_logits = outputs["affordance_logits"]
        aff_target = targets["affordance"].float()
        aff_mask = targets.get("affordance_mask")
        if aff_mask is None:
            aff_mask = ~torch.isnan(aff_target)
        aff_safe = torch.where(aff_mask, aff_target, torch.zeros_like(aff_target))
        elem = F.binary_cross_entropy_with_logits(
            aff_logits, aff_safe, reduction="none",
        )
        denom = aff_mask.float().sum().clamp_min(1.0)
        losses["affordance"] = (elem * aff_mask.float()).sum() / denom * affordance_weight

    losses["total"] = sum(losses.values())
    return losses


# ── study_tags.v001 supervision loss ──────────────────────────────────────


def study_tags_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    goal_family_weight: float = 1.0,
    role_buckets_weight: float = 0.5,
    capability_weight: float = 0.5,
    topology_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Multi-task loss for study_tags supervision heads.

    Expected targets (any subset; the corresponding loss term is skipped
    when a key is missing):

      goal_family:        [B] long, in [0, 46). Class indices into
                          STUDY_TAGS_GOAL_FAMILIES. No mask needed —
                          every episode has a goal_family (``unknown``
                          covers the fallback case).
      role_buckets:       [B, 16, 9] float in {0, 1}. Binary per
                          (color, role) cell. Set NaN to skip that
                          (color, role) entry.
      role_buckets_mask:  [B, 16, 9] bool, optional. True means "include
                          this cell in the loss"; defaults to all-True
                          if absent (uses NaN-detection on role_buckets).
      capability_flags:   [B, 7] float in {0, 1}, with NaN for "not
                          declared by mechanic".
      capability_mask:    [B, 7] bool, optional. True for "declared".
      topology:           [B] long in [0, 9). Use TOPOLOGY_TO_IDX
                          ["unknown"] for "not declared" — but pair
                          with topology_mask=False to skip the loss
                          for that episode entirely.
      topology_mask:      [B] bool, optional. True for "declared".

    Returns a dict with one entry per included head plus ``"total"``.
    """
    losses: dict[str, torch.Tensor] = {}

    if "goal_family_logits" in outputs and "goal_family" in targets:
        gf = F.cross_entropy(
            outputs["goal_family_logits"], targets["goal_family"],
        )
        losses["goal_family"] = gf * goal_family_weight

    if "role_buckets_logits" in outputs and "role_buckets" in targets:
        logits = outputs["role_buckets_logits"]
        rb = targets["role_buckets"]
        mask = targets.get("role_buckets_mask")
        if mask is None:
            mask = ~torch.isnan(rb)
        rb_safe = torch.where(mask, rb, torch.zeros_like(rb))
        elem = F.binary_cross_entropy_with_logits(
            logits, rb_safe, reduction="none",
        )
        denom = mask.float().sum().clamp_min(1.0)
        rb_loss = (elem * mask.float()).sum() / denom
        losses["role_buckets"] = rb_loss * role_buckets_weight

    if "capability_logits" in outputs and "capability_flags" in targets:
        logits = outputs["capability_logits"]
        cf = targets["capability_flags"]
        mask = targets.get("capability_mask")
        if mask is None:
            mask = ~torch.isnan(cf)
        cf_safe = torch.where(mask, cf, torch.zeros_like(cf))
        elem = F.binary_cross_entropy_with_logits(
            logits, cf_safe, reduction="none",
        )
        denom = mask.float().sum().clamp_min(1.0)
        cap_loss = (elem * mask.float()).sum() / denom
        losses["capability"] = cap_loss * capability_weight

    if "topology_logits" in outputs and "topology" in targets:
        logits = outputs["topology_logits"]
        topo = targets["topology"]
        mask = targets.get("topology_mask")
        if mask is None:
            top_loss = F.cross_entropy(logits, topo)
        else:
            if mask.any():
                top_loss = F.cross_entropy(logits[mask], topo[mask])
            else:
                top_loss = logits.new_zeros(())
        losses["topology"] = top_loss * topology_weight

    if losses:
        losses["total"] = sum(losses.values())
    return losses


# ── Checkpoint loading: add study_tags heads to an existing model ─────────


def from_pretrained_add_study_tags_heads(
    checkpoint_path: str | Path,
    *,
    freeze_trunk: bool = True,
    **model_kwargs,
) -> GoalPropertyModel:
    """Load an existing GoalPropertyModel checkpoint and add study_tags heads.

    The trunk weights (encoder, transformer, original heads) load from
    the checkpoint; the four new study_tags heads are freshly
    initialised by PyTorch's default schemes.

    Args:
        checkpoint_path: path to a torch.save() artefact. Accepts either
            a bare state_dict OR a dict containing ``model_state_dict``
            (and optionally ``config``).
        freeze_trunk: if True, ``requires_grad=False`` on every parameter
            that came from the checkpoint. Only the new heads will train.
        **model_kwargs: overrides for the GoalPropertyModel constructor.
            Use this when the checkpoint dict doesn't carry a ``config``.

    Raises:
        RuntimeError if the loaded state_dict has unexpected missing
            keys beyond the four new heads.
    """
    state = torch.load(
        str(checkpoint_path), map_location="cpu", weights_only=False,
    )
    if isinstance(state, dict) and "model_state_dict" in state:
        sd = state["model_state_dict"]
        ckpt_cfg = dict(state.get("config", {}) or {})
    elif isinstance(state, dict) and "model_state" in state:
        sd = state["model_state"]
        raw_cfg = dict(state.get("config", {}) or {})
        ckpt_cfg = dict(raw_cfg.get("model", raw_cfg) or {})
    elif isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
        raw_cfg = dict(state.get("config", {}) or {})
        ckpt_cfg = dict(raw_cfg.get("model", raw_cfg) or {})
    else:
        sd = state
        ckpt_cfg = {}
    cfg = {**ckpt_cfg, **model_kwargs, "study_tags_heads": True}

    model = GoalPropertyModel(**cfg)
    incompatible = model.load_state_dict(sd, strict=False)

    expected_new_params = {
        "goal_family_head.weight", "goal_family_head.bias",
        "role_buckets_head.weight", "role_buckets_head.bias",
        "capability_head.weight", "capability_head.bias",
        "topology_head.weight", "topology_head.bias",
    }
    # Conditional mechanic head adds Sequential(Linear, GELU, Linear) →
    # keys are cond_mechanic_head.{0,2}.{weight,bias}. Add to the
    # accepted-missing set when that head is enabled.
    if cfg.get("conditional_mechanic_head"):
        expected_new_params |= {
            "cond_mechanic_head.0.weight", "cond_mechanic_head.0.bias",
            "cond_mechanic_head.2.weight", "cond_mechanic_head.2.bias",
        }
    unexpected_missing = set(incompatible.missing_keys) - expected_new_params
    if unexpected_missing:
        raise RuntimeError(
            f"checkpoint missing trunk parameters: {sorted(unexpected_missing)[:5]}"
            f" (total {len(unexpected_missing)})"
        )
    if incompatible.unexpected_keys:
        # Extra keys in the checkpoint that don't fit the model are non-fatal,
        # but flag them so the operator notices schema drift.
        import warnings
        warnings.warn(
            f"checkpoint has unexpected keys (ignored): "
            f"{sorted(incompatible.unexpected_keys)[:5]}"
            f" (total {len(incompatible.unexpected_keys)})"
        )

    if freeze_trunk:
        for name, param in model.named_parameters():
            if name not in expected_new_params:
                param.requires_grad = False

    return model


def _masked_mechanic_predictions(
    logits: torch.Tensor,
    targets: torch.Tensor,
    active_mechanic_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if active_mechanic_indices is None:
        return logits, targets
    active_mechanic_indices = active_mechanic_indices.to(logits.device)
    active_logits = logits.index_select(1, active_mechanic_indices)
    remap = torch.full((logits.shape[1],), -1, dtype=torch.long, device=logits.device)
    remap[active_mechanic_indices] = torch.arange(
        active_mechanic_indices.numel(), dtype=torch.long, device=logits.device,
    )
    active_targets = remap[targets]
    if (active_targets < 0).any():
        raise ValueError("mechanic target missing from active mechanic mask")
    return active_logits, active_targets


# ── Evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_property_model(
    model: GoalPropertyModel,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: str = "none",
    non_blocking: bool = False,
    active_mechanic_indices: torch.Tensor | None = None,
) -> dict[str, float]:
    """Evaluate all prediction heads over a dataloader."""
    model.eval().to(device)
    amp_enabled = amp != "none" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16

    agent_errors, type_accs, keys_accs, delta_errors = [], [], [], []
    mechanic_accs = []
    mechanic_top5_accs = []
    family_accs = []
    subclass_accs = []
    stage_accs = []
    type_correct = torch.zeros(NUM_GOAL_TYPES, dtype=torch.long)
    type_total = torch.zeros(NUM_GOAL_TYPES, dtype=torch.long)
    n = 0

    for batch in loader:
        cf = batch["context_frames"].to(device, non_blocking=non_blocking)
        ca = batch["context_actions"].to(device, non_blocking=non_blocking)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(cf, ca)

        # Agent: L2 error
        ad_pred = outputs["agent_delta"]
        ad_target = batch["agent_delta"].to(device, non_blocking=non_blocking)
        agent_errors.append((ad_pred - ad_target).norm(dim=1).mean().item())

        # Goal type accuracy
        type_pred = outputs["goal_type_logits"].argmax(dim=1)
        type_target = batch["goal_type"].to(device, non_blocking=non_blocking)
        type_accs.append((type_pred == type_target).float().mean().item())
        for goal_type_idx in range(NUM_GOAL_TYPES):
            mask = type_target == goal_type_idx
            count = int(mask.sum().item())
            if count:
                type_total[goal_type_idx] += count
                type_correct[goal_type_idx] += int((type_pred[mask] == goal_type_idx).sum().item())

        # Keys accuracy
        keys_pred = (outputs["keys_logits"].sigmoid() > 0.5).float()
        keys_target = batch["keys_collected"].to(device, non_blocking=non_blocking)
        keys_accs.append((keys_pred == keys_target).float().mean().item())

        # Delta pixels
        dp_pred = outputs["delta_px"]
        dp_target = batch["goal_delta_px"].to(device, non_blocking=non_blocking)
        delta_errors.append((dp_pred - dp_target).abs().mean().item())

        if "mechanic_logits" in outputs:
            mech_target = batch["mechanic"].to(device, non_blocking=non_blocking)
            mech_logits, mech_target = _masked_mechanic_predictions(
                outputs["mechanic_logits"], mech_target, active_mechanic_indices,
            )
            mech_pred = mech_logits.argmax(dim=1)
            mechanic_accs.append((mech_pred == mech_target).float().mean().item())
            k = min(5, mech_logits.shape[1])
            if k > 1:
                mech_topk = mech_logits.topk(k, dim=1).indices
                mechanic_top5_accs.append((mech_topk == mech_target.unsqueeze(1)).any(dim=1).float().mean().item())
        if "family_logits" in outputs and "family" in batch:
            family_pred = outputs["family_logits"].argmax(dim=1)
            family_target = batch["family"].to(device, non_blocking=non_blocking)
            family_accs.append((family_pred == family_target).float().mean().item())
        if "subclass_logits" in outputs and "subclass" in batch:
            subclass_pred = outputs["subclass_logits"].argmax(dim=1)
            subclass_target = batch["subclass"].to(device, non_blocking=non_blocking)
            subclass_accs.append((subclass_pred == subclass_target).float().mean().item())
        if "stage_logits" in outputs and "stage" in batch:
            stage_pred = outputs["stage_logits"].argmax(dim=1)
            stage_target = batch["stage"].to(device, non_blocking=non_blocking)
            stage_accs.append((stage_pred == stage_target).float().mean().item())

        n += 1

    per_type_acc = {}
    macro_values = []
    for idx, name in enumerate(GOAL_TYPES):
        total = int(type_total[idx].item())
        correct = int(type_correct[idx].item())
        acc = (correct / total) if total else 0.0
        per_type_acc[f"goal_type_acc_{name}"] = acc
        per_type_acc[f"goal_type_count_{name}"] = total
        if total:
            macro_values.append(acc)

    metrics = {
        "agent_l2_error": sum(agent_errors) / n,
        "goal_type_acc": sum(type_accs) / n,
        "goal_type_macro_acc": sum(macro_values) / max(len(macro_values), 1),
        "keys_acc": sum(keys_accs) / n,
        "delta_px_abs_error": sum(delta_errors) / n,
        **per_type_acc,
    }
    if mechanic_accs:
        metrics["mechanic_acc"] = sum(mechanic_accs) / len(mechanic_accs)
    if mechanic_top5_accs:
        metrics["mechanic_top5_acc"] = sum(mechanic_top5_accs) / len(mechanic_top5_accs)
    if family_accs:
        metrics["family_acc"] = sum(family_accs) / len(family_accs)
    if subclass_accs:
        metrics["subclass_acc"] = sum(subclass_accs) / len(subclass_accs)
    if stage_accs:
        metrics["stage_acc"] = sum(stage_accs) / len(stage_accs)
    return metrics


def collate_property(batch: list[dict]) -> dict:
    cf = torch.stack([b["context_frames"] for b in batch])
    ca = torch.stack([b["context_actions"] for b in batch])
    out = {
        "context_frames": cf,
        "context_actions": ca,
        "agent_delta": torch.stack([b["agent_delta"] for b in batch]),
        "goal_type": torch.tensor([b["goal_type"] for b in batch], dtype=torch.long),
        "keys_collected": torch.stack([b["keys_collected"] for b in batch]),
        "goal_delta_px": torch.tensor([b["goal_delta_px"] for b in batch], dtype=torch.float),
        "mechanic": torch.tensor([b["game_idx"] for b in batch], dtype=torch.long),
        "game_id": [b["game_id"] for b in batch],
    }
    if all(b.get("family_idx") is not None for b in batch):
        out["family"] = torch.tensor([b["family_idx"] for b in batch], dtype=torch.long)
    if all(b.get("subclass_idx") is not None for b in batch):
        out["subclass"] = torch.tensor([b["subclass_idx"] for b in batch], dtype=torch.long)
    if all(b.get("stage_idx") is not None for b in batch):
        out["stage"] = torch.tensor([b["stage_idx"] for b in batch], dtype=torch.long)
    # study_tags supervision tensors — forwarded only when every item has them
    # (i.e. the dataset was wrapped with StudyTagsSupervisionWrapper). Missing
    # any one item means the batch is mixed and should NOT supervise.
    if all("goal_family" in b for b in batch):
        out["goal_family"] = torch.stack([b["goal_family"] for b in batch])
    if all("role_buckets" in b for b in batch):
        out["role_buckets"] = torch.stack([b["role_buckets"] for b in batch])
        out["role_buckets_mask"] = torch.stack([b["role_buckets_mask"] for b in batch])
    if all("capability_flags" in b for b in batch):
        out["capability_flags"] = torch.stack([b["capability_flags"] for b in batch])
        out["capability_mask"] = torch.stack([b["capability_mask"] for b in batch])
    if all("topology" in b for b in batch):
        out["topology"] = torch.stack([b["topology"] for b in batch])
        out["topology_mask"] = torch.stack([b["topology_mask"] for b in batch])
    # Affordance supervision — iter-19+ datasets only.
    if all("affordance" in b for b in batch):
        out["affordance"] = torch.stack([b["affordance"] for b in batch])
        out["affordance_mask"] = torch.stack([b["affordance_mask"] for b in batch])
    return out


# ── Trainer ─────────────────────────────────────────────────────────────────

@dataclass
class PropertyTrainConfig:
    num_epochs: int = 150
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    agent_weight: float = 1.0
    goal_type_weight: float = 0.5
    keys_weight: float = 0.5
    delta_px_weight: float = 0.1
    mechanic_weight: float = 1.0
    family_weight: float = 1.0
    subclass_weight: float = 1.0
    stage_weight: float = 0.25
    seed: int = 0
    log_every: int = 5
    amp: str = "none"
    non_blocking: bool = False
    lr_schedule: str = "constant"
    total_train_steps: int = 0
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    goal_type_class_weights: list[float] | None = None
    active_mechanic_indices: list[int] | None = None
    sigreg_weight: float = 0.0
    sigreg_projections: int = 256
    sigreg_weight_lambda: float = 1.0
    optimizer: str = "adamw"
    # SODA tail-of-run safety (precision regime iter 13). The raw
    # 1/(k+2) centering decays to ~0 over a long run, losing the
    # skipless-collapse rescue exactly where the model has trained
    # longest. soda_reanchor_every>0 re-snapshots the z0 anchor every N
    # steps and restarts the decay window so centering stays active;
    # soda_alpha_floor clamps the centering coefficient's minimum.
    # Both default off (legacy 1/(k+2) toward frozen init).
    soda_reanchor_every: int = 0
    soda_alpha_floor: float = 0.0
    soap_betas: tuple[float, float] = (0.95, 0.999)
    soap_shampoo_beta: float = 0.999
    soap_precondition_frequency: int = 10
    soap_max_precond_dim: int = 10000
    soap_merge_dims: bool = False
    soap_precondition_1d: bool = False
    soap_basis_stage1_method: str = "tangent"
    soap_basis_warmup_steps: int = 250
    soap_basis_compute_dtype: str = "bfloat16"
    soap_basis_storage_dtype: str = "bfloat16"
    soap_covariance_compute_dtype: str = "bfloat16"
    # study_tags.v001 supervision (only used when the model was built
    # with study_tags_heads=True AND the dataset is wrapped with
    # StudyTagsSupervisionWrapper so the batch carries goal_family/
    # role_buckets/capability_flags/topology + masks).
    study_tags_heads: bool = False
    goal_family_weight: float = 1.0
    role_buckets_weight: float = 0.5
    capability_weight: float = 0.5
    topology_weight: float = 0.25
    # Capability-conditional mechanic head loss weight (cond head only used
    # when GoalPropertyModel was built with conditional_mechanic_head=True).
    cond_mechanic_weight: float = 1.0
    # Affordance head (iter 19): build flag + loss weight. The head that
    # connects Lewis output to the action space (per-action usefulness,
    # conditioned on capability). Loss term fires only when the dataset
    # carries the `affordance` target (iter-19+ builds).
    affordance_head: bool = False
    affordance_weight: float = 0.5
    # Contrastive trunk loss — applies when PropertyTrainer is given a
    # ContrastiveBatchAugmenter. 0.0 disables the term even when the
    # augmenter is wired up.
    contrastive_weight: float = 0.0
    # Shannon-entropy-driven adaptive loss balancer. When True, the
    # per-head *_weight values above become BASE weights; the runtime
    # balancer multiplies them by adaptive multipliers derived from
    # rolling EMA of per-task prediction entropy. See
    # lewm.adaptive_loss_balance.ShannonAdaptiveBalancer.
    adaptive_loss_balance: bool = False
    adaptive_ema_alpha: float = 0.95
    adaptive_temperature: float = 1.0
    adaptive_min_multiplier: float = 0.1
    adaptive_max_multiplier: float = 5.0
    adaptive_warmup_steps: int = 50
    # Differential learning-rate multipliers (multiplied onto config.learning_rate
    # for each parameter group). Defaults of 1.0 preserve the legacy
    # single-LR behaviour. Used for post-train scenarios where the trunk
    # should move slowly while new heads adapt fast.
    trunk_lr_mult: float = 1.0
    existing_heads_lr_mult: float = 1.0
    new_heads_lr_mult: float = 1.0
    # Label smoothing on the 3043-way mechanic head and 33-way family head.
    # 0.0 disables. Recommended ~0.05–0.1 for the mechanic head — long-tail
    # robustness against rare-class memorization.
    mechanic_label_smoothing: float = 0.0
    family_label_smoothing: float = 0.0
    # Hierarchical mechanic head (family-gated). When True, the model is
    # built with HierarchicalMechanicHead replacing flat mechanic_head +
    # family_head; loss switches to hierarchical_mechanic_loss
    # (family CE + within-family CE). Requires mechanics_per_family.
    hierarchical_mechanic_head: bool = False
    hier_family_ce_weight: float = 1.0
    hier_within_family_ce_weight: float = 1.0
    # Iter 8: additive mode. When True the flat mechanic_head +
    # family_head coexist with the hier head; mechanic_logits is
    # produced via compute_hier_additive_logits (flat + w*hier).
    # hier_mechanic_weight=0 silences the hier signal at output time.
    hier_additive_mode: bool = False
    hier_mechanic_weight: float = 0.5
    # When >0, the trainer anneals the model's current hier weight
    # linearly from 0 → hier_mechanic_weight over the first N global
    # steps. Lets the flat head establish itself before the hier signal
    # starts nudging the joint argmax.
    hier_mechanic_weight_anneal_steps: int = 0
    # PCGrad-style paired-gradient orthogonalisation for trunk-shared
    # heads. Adds N retain-graph backwards per step (one per head),
    # so cost ~N× the baseline backward; deploy selectively.
    orthogonal_paired_gradients: bool = False
    # Heads supplied to the orthogonalizer. Default covers the
    # canonical Lewis head set; loss keys missing from the model are
    # silently skipped (see PropertyTrainer.train_step).
    orthogonal_head_names: tuple[str, ...] = (
        "agent", "goal_type", "keys", "delta_px",
        "mechanic", "family", "subclass", "stage",
        "cond_mechanic", "goal_family", "role_buckets",
        "capability", "topology", "contrastive",
    )
    # Gradient-entropy + conflict-penalty extensions on the adaptive
    # balancer. These wrap ShannonAdaptiveBalancer transparently — when
    # both False, the GradientAwareBalancer is identical to the inner
    # ShannonAdaptiveBalancer. use_gradient_entropy requires
    # orthogonal_paired_gradients=True (it needs per-head trunk grads).
    use_gradient_entropy: bool = False
    use_conflict_penalty: bool = False
    gradient_entropy_temperature: float = 1.0
    conflict_penalty_alpha: float = 1.0
    # Optional path to a JSON file mapping task name -> conflict rate
    # (typically produced by an offline RelationalGradientProbe run).
    # Consumed by GradientAwareBalancer.update_conflicts() at init.
    conflict_rate_file: str | None = None
    # Iter 7: live per-pair gradient probe + selective PCGrad. When
    # enabled, the probe EMA-tracks per-pair cosines every step and
    # (a) feeds per-task conflict rates into the balancer dynamically
    # (overrides any static conflict_rate_file load), (b) gates the
    # orthogonaliser's inner projection loop so non-conflicting pairs
    # skip the projection cost. Requires orthogonal_paired_gradients=True.
    live_relational_probe: bool = False
    live_probe_ema_alpha: float = 0.9
    live_probe_warmup_steps: int = 50
    live_probe_conflict_threshold: float = 0.0
    # When True, the orthogonaliser is built as SelectivePCGradOrthogonalizer
    # gated by the live probe's should_project predicate. Without this flag
    # the live probe still updates the balancer but PCGrad runs every pair.
    selective_pcgrad: bool = False


class PropertyTrainer:
    def __init__(
        self,
        model: GoalPropertyModel,
        config: PropertyTrainConfig | None = None,
        *,
        contrastive_augmenter: "object | None" = None,
        contrastive_mechanic_names: "list[str] | None" = None,
    ):
        self.model = model
        self.config = config or PropertyTrainConfig()
        # Contrastive supervision: optional augmenter that draws partner
        # episodes from a confusion-mined or family-mined pair registry.
        # ``contrastive_mechanic_names`` is the int→name lookup used to
        # resolve batch['mechanic'] integers to the registry's name keys.
        self.contrastive_augmenter = contrastive_augmenter
        self.contrastive_mechanic_names = contrastive_mechanic_names
        if self.contrastive_augmenter is not None and self.contrastive_mechanic_names is None:
            raise ValueError(
                "PropertyTrainer.contrastive_augmenter requires "
                "contrastive_mechanic_names (int → name lookup)"
            )

        # Shannon-entropy adaptive loss balancer (optional). When enabled,
        # the per-head loss weights in config become BASE weights and the
        # balancer applies adaptive multipliers per step. Tasks whose
        # logits aren't in the model's outputs (e.g. study_tags heads on
        # a model built without them) silently skip.
        #
        # When use_gradient_entropy or use_conflict_penalty is on, we
        # use GradientAwareBalancer which composes the same Shannon
        # logic with per-task grad-distribution entropy and conflict
        # factors. With both flags off the balancers are equivalent.
        self.adaptive_balancer = None
        if self.config.adaptive_loss_balance:
            from lewm.adaptive_loss_balance import (
                ShannonAdaptiveBalancer,
                default_lewis_tasks,
            )
            base_tasks = default_lewis_tasks(
                mechanic_weight=self.config.mechanic_weight,
                family_weight=self.config.family_weight,
                subclass_weight=self.config.subclass_weight,
                stage_weight=self.config.stage_weight,
                goal_type_weight=self.config.goal_type_weight,
                keys_weight=self.config.keys_weight,
                agent_weight=self.config.agent_weight,
                delta_px_weight=self.config.delta_px_weight,
                goal_family_weight=self.config.goal_family_weight,
                role_buckets_weight=self.config.role_buckets_weight,
                capability_weight=self.config.capability_weight,
                topology_weight=self.config.topology_weight,
                cond_mechanic_weight=self.config.cond_mechanic_weight,
            )
            if self.config.use_gradient_entropy or self.config.use_conflict_penalty:
                from lewm.gradient_aware_balancer import GradientAwareBalancer
                self.adaptive_balancer = GradientAwareBalancer(
                    base_tasks,
                    use_prediction_entropy=True,
                    use_gradient_entropy=self.config.use_gradient_entropy,
                    use_conflict_penalty=self.config.use_conflict_penalty,
                    gradient_entropy_temperature=self.config.gradient_entropy_temperature,
                    conflict_penalty_alpha=self.config.conflict_penalty_alpha,
                    ema_alpha=self.config.adaptive_ema_alpha,
                    temperature=self.config.adaptive_temperature,
                    min_multiplier=self.config.adaptive_min_multiplier,
                    max_multiplier=self.config.adaptive_max_multiplier,
                    warmup_steps=self.config.adaptive_warmup_steps,
                )
                # Seed conflict rates from an offline file when available.
                if self.config.conflict_rate_file:
                    import json
                    from pathlib import Path as _Path
                    crf = _Path(self.config.conflict_rate_file)
                    if crf.exists():
                        with open(crf) as fh:
                            rates = json.load(fh)
                        # Accept either {task: rate} or
                        # {"per_task": {task: rate}, ...}
                        if isinstance(rates, dict) and "per_task" in rates:
                            rates = rates["per_task"]
                        self.adaptive_balancer.update_conflicts(
                            {str(k): float(v) for k, v in rates.items()},
                        )
            else:
                self.adaptive_balancer = ShannonAdaptiveBalancer(
                    base_tasks,
                    ema_alpha=self.config.adaptive_ema_alpha,
                    temperature=self.config.adaptive_temperature,
                    min_multiplier=self.config.adaptive_min_multiplier,
                    max_multiplier=self.config.adaptive_max_multiplier,
                    warmup_steps=self.config.adaptive_warmup_steps,
                )

        # Live relational probe (optional, Iter 7). Tracks per-pair
        # gradient cosines via EMA and exposes a should_project predicate.
        self.live_probe = None
        if self.config.live_relational_probe:
            from lewm.live_relational_probe import LiveRelationalProbe
            self.live_probe = LiveRelationalProbe(
                list(self.config.orthogonal_head_names),
                ema_alpha=self.config.live_probe_ema_alpha,
                warmup_steps=self.config.live_probe_warmup_steps,
                conflict_threshold=self.config.live_probe_conflict_threshold,
            )

        # PCGrad orthogonalizer (optional). Built once, reused every
        # step. Head_names should NOT include heads the trainer never
        # produces losses for — see PropertyTrainer.train_step where we
        # filter to losses present in the dict.
        self.orthogonalizer = None
        if self.config.orthogonal_paired_gradients:
            if self.config.selective_pcgrad:
                if self.live_probe is None:
                    raise ValueError(
                        "selective_pcgrad=True requires live_relational_probe=True "
                        "(the probe supplies the should_project predicate)"
                    )
                from lewm.selective_orthogonalizer import SelectivePCGradOrthogonalizer
                self.orthogonalizer = SelectivePCGradOrthogonalizer(
                    self.model,
                    trunk_prefixes=self._TRUNK_PREFIXES,
                    head_names=list(self.config.orthogonal_head_names),
                    should_project_fn=self.live_probe.should_project,
                    rng_seed=self.config.seed,
                )
            else:
                from lewm.relational_gradients import PairedGradientOrthogonalizer
                self.orthogonalizer = PairedGradientOrthogonalizer(
                    self.model,
                    trunk_prefixes=self._TRUNK_PREFIXES,
                    head_names=list(self.config.orthogonal_head_names),
                    rng_seed=self.config.seed,
                )

        torch.manual_seed(self.config.seed)
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(base_model, "_orig_mod"):
            base_model = base_model._orig_mod

        param_groups = self._make_param_groups()
        if not any(g["params"] for g in param_groups):
            raise ValueError("PropertyTrainer received a model with no trainable parameters")
        self.optimizer = self._make_optimizer(param_groups)
        self.history: list[dict[str, float]] = []
        self.global_step = 0
        self.sigreg = None
        if self.config.sigreg_weight > 0:
            self.sigreg = SIGReg(config=SIGRegConfig(
                latent_dim=base_model.hidden_dim,
                num_projections=self.config.sigreg_projections,
                weight_lambda=self.config.sigreg_weight_lambda,
                seed=self.config.seed,
            ))

    # Trunk vs. existing-heads vs. new-heads parameter partitioning. Used
    # to drive differential learning rates (configured via *_lr_mult).
    _TRUNK_PREFIXES = (
        "frame_encoder.", "action_embedder.", "pos_embed.",
        "transformer.", "mockingbird_encoder.",
    )
    _EXISTING_HEAD_PREFIXES = (
        "agent_head.", "goal_type_head.", "keys_head.", "delta_px_head.",
        "mechanic_head.", "family_head.", "subclass_head.", "stage_head.",
    )
    _NEW_HEAD_PREFIXES = (
        "goal_family_head.", "role_buckets_head.", "capability_head.",
        "topology_head.", "cond_mechanic_head.", "affordance_head.",
        # Iter 10: hier head classified as "new head" (parallel to
        # cond_mechanic_head). Without this entry the hier head's params
        # fall into the "other" group and silently ride on
        # existing_heads_lr_mult — a static lock when the operator dials
        # existing_heads_lr_mult for warmstart-style fine-tunes.
        "hier_mechanic_head.",
    )

    def _partition_trainable(self) -> tuple[list, list, list, list]:
        """Returns ``(trunk, existing_heads, new_heads, other)`` lists of
        trainable parameters, partitioned by name prefix. Unwraps DDP /
        torch.compile if needed.
        """
        base = self.model
        # Unwrap DDP
        if hasattr(base, "module"):
            base = base.module
        # Unwrap torch.compile
        if hasattr(base, "_orig_mod"):
            base = base._orig_mod
        trunk, existing, new, other = [], [], [], []
        for name, p in base.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(self._TRUNK_PREFIXES):
                trunk.append(p)
            elif name.startswith(self._EXISTING_HEAD_PREFIXES):
                existing.append(p)
            elif name.startswith(self._NEW_HEAD_PREFIXES):
                new.append(p)
            else:
                other.append(p)
        return trunk, existing, new, other

    def _make_param_groups(self) -> list[dict]:
        """Build optimizer param-groups honouring differential LR multipliers.

        If all *_lr_mult == 1.0, returns a single legacy group (preserves
        prior behaviour). Otherwise emits up to four groups (trunk,
        existing_heads, new_heads, other) each with its own scaled LR
        and an ``lr_mult`` field used by the schedule loop.
        """
        cfg = self.config
        same = (
            abs(cfg.trunk_lr_mult - 1.0) < 1e-12
            and abs(cfg.existing_heads_lr_mult - 1.0) < 1e-12
            and abs(cfg.new_heads_lr_mult - 1.0) < 1e-12
        )
        base_lr = cfg.learning_rate
        if same:
            params = [p for p in self.model.parameters() if p.requires_grad]
            return [{"params": params, "lr": base_lr}]
        trunk, existing, new, other = self._partition_trainable()
        groups: list[dict] = []
        if trunk:
            groups.append({
                "params": trunk,
                "lr": base_lr * cfg.trunk_lr_mult,
                "lr_mult": cfg.trunk_lr_mult,
                "name": "trunk",
            })
        if existing:
            groups.append({
                "params": existing,
                "lr": base_lr * cfg.existing_heads_lr_mult,
                "lr_mult": cfg.existing_heads_lr_mult,
                "name": "existing_heads",
            })
        if new:
            groups.append({
                "params": new,
                "lr": base_lr * cfg.new_heads_lr_mult,
                "lr_mult": cfg.new_heads_lr_mult,
                "name": "new_heads",
            })
        if other:
            # Unclassified params (rare) ride on the existing-heads schedule.
            groups.append({
                "params": other,
                "lr": base_lr * cfg.existing_heads_lr_mult,
                "lr_mult": cfg.existing_heads_lr_mult,
                "name": "other",
            })
        return groups

    def _make_optimizer(
        self, param_groups: list[dict],
    ) -> torch.optim.Optimizer:
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        if self.config.optimizer == "soap":
            return SOAP(
                param_groups,
                lr=self.config.learning_rate,
                betas=self.config.soap_betas,
                shampoo_beta=self.config.soap_shampoo_beta,
                weight_decay=self.config.weight_decay,
                precondition_frequency=self.config.soap_precondition_frequency,
                max_precond_dim=self.config.soap_max_precond_dim,
                merge_dims=self.config.soap_merge_dims,
                precondition_1d=self.config.soap_precondition_1d,
                basis_stage1_method=self.config.soap_basis_stage1_method,
                basis_warmup_steps=self.config.soap_basis_warmup_steps,
                basis_compute_dtype=self.config.soap_basis_compute_dtype,
                basis_storage_dtype=self.config.soap_basis_storage_dtype,
                covariance_compute_dtype=self.config.soap_covariance_compute_dtype,
            )
        if self.config.optimizer == "soda":
            # SODA (Pethick et al. 2026) wraps a base optimizer with a
            # 1/(k+2) centering schedule toward initialization. Per the
            # SODA research report, the base optimizer runs WITHOUT
            # weight decay — SODA's centering provides the implicit
            # regularization. Critically: SODA's centering keeps the
            # Jacobian well-conditioned, which rescues skipless
            # Mockingbird from the AdamW mid-training collapse and is
            # the scaling-safety guarantee for large game counts.
            from lewm.soda import SODA as SODAOptimizer
            base = torch.optim.AdamW(
                param_groups,
                lr=self.config.learning_rate,
                weight_decay=0.0,  # SODA centering replaces weight decay
            )
            soda = SODAOptimizer(
                param_groups, base_optimizer=base,
                reanchor_every=self.config.soda_reanchor_every,
                alpha_floor=self.config.soda_alpha_floor,
            )
            # base AdamW and SODA each build their own param_group dicts
            # over the same param tensors. The trainer's LR scheduler
            # writes group["lr"] on self.optimizer.param_groups (SODA's);
            # AdamW.step() reads base's. Share base's dicts so the
            # schedule actually reaches the inner optimizer.
            soda.param_groups = base.param_groups
            return soda
        raise ValueError(f"unsupported optimizer: {self.config.optimizer}")

    def _scheduled_lr(self, step: int) -> float:
        if self.config.lr_schedule == "constant" or self.config.total_train_steps <= 0:
            return self.config.learning_rate
        warmup_steps = max(0, min(self.config.warmup_steps, self.config.total_train_steps - 1))
        min_lr = self.config.learning_rate * self.config.min_lr_ratio
        if warmup_steps > 0 and step < warmup_steps:
            return self.config.learning_rate * float(step + 1) / float(warmup_steps)
        denom = max(1, self.config.total_train_steps - warmup_steps)
        progress = min(1.0, max(0.0, float(step - warmup_steps) / float(denom)))
        if self.config.lr_schedule == "cosine":
            import math
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr + (self.config.learning_rate - min_lr) * scale
        raise ValueError(f"unsupported lr_schedule: {self.config.lr_schedule}")

    def _scheduled_hier_weight(self) -> float | None:
        """Linearly anneal hier_mechanic_weight from 0 → config target over
        ``hier_mechanic_weight_anneal_steps`` global steps. Returns None
        when no anneal is configured (caller leaves model weight as-is).
        """
        n = int(self.config.hier_mechanic_weight_anneal_steps)
        if n <= 0:
            return None
        target = float(self.config.hier_mechanic_weight)
        # Anneal runs at the START of train_step (before global_step
        # increments). Read step+1 so the weight applied during step k
        # is target * k/N — full target reached by step N.
        progress = min(1.0, float(self.global_step + 1) / float(n))
        return target * progress

    def train_step(self, batch: dict) -> dict[str, float]:
        device = next(self.model.parameters()).device
        scheduled_lr = self._scheduled_lr(self.global_step)
        # Anneal hier additive weight on the underlying model. Effective
        # weight is read by GoalPropertyModel.forward each step.
        annealed = self._scheduled_hier_weight()
        if annealed is not None:
            base = self.model.module if hasattr(self.model, "module") else self.model
            if hasattr(base, "_orig_mod"):
                base = base._orig_mod
            if hasattr(base, "hier_mechanic_weight"):
                base.hier_mechanic_weight = annealed
        # Honour per-group lr_mult so differential LR groups are preserved
        # across the schedule (e.g. trunk at 0.1x, new heads at 1x).
        for group in self.optimizer.param_groups:
            mult = float(group.get("lr_mult", 1.0))
            group["lr"] = scheduled_lr * mult
        cf = batch["context_frames"].to(device, non_blocking=self.config.non_blocking)
        ca = batch["context_actions"].to(device, non_blocking=self.config.non_blocking)
        targets = {
            "agent_delta": batch["agent_delta"].to(device, non_blocking=self.config.non_blocking),
            "goal_type": batch["goal_type"].to(device, non_blocking=self.config.non_blocking),
            "keys_collected": batch["keys_collected"].to(device, non_blocking=self.config.non_blocking),
            "goal_delta_px": batch["goal_delta_px"].to(device, non_blocking=self.config.non_blocking),
            "mechanic": batch["mechanic"].to(device, non_blocking=self.config.non_blocking),
        }
        if "family" in batch:
            targets["family"] = batch["family"].to(device, non_blocking=self.config.non_blocking)
        if "subclass" in batch:
            targets["subclass"] = batch["subclass"].to(device, non_blocking=self.config.non_blocking)
        if "stage" in batch:
            targets["stage"] = batch["stage"].to(device, non_blocking=self.config.non_blocking)
        if "affordance" in batch:
            targets["affordance"] = batch["affordance"].to(device, non_blocking=self.config.non_blocking)
            if "affordance_mask" in batch:
                targets["affordance_mask"] = batch["affordance_mask"].to(
                    device, non_blocking=self.config.non_blocking)

        self.model.train()
        amp_enabled = self.config.amp != "none" and device.type == "cuda"
        amp_dtype = torch.bfloat16 if self.config.amp == "bf16" else torch.float16

        # Run the contrastive augmenter BEFORE the forward so we can
        # concatenate partner episodes into a single forward batch. Two
        # separate forward calls under DDP triggers an autograd inplace
        # error (some BN/qk_gain/RMSNorm buffer of shape [256] gets
        # versioned past what the first forward's backward expects).
        aug_payload = None
        if (
            self.contrastive_augmenter is not None
            and self.config.contrastive_weight > 0.0
            and "mechanic" in batch
        ):
            aug_payload = self.contrastive_augmenter.augment(
                batch, mechanic_names=self.contrastive_mechanic_names,
            )

        need_latent = self.sigreg is not None or aug_payload is not None

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            if aug_payload is not None:
                partner_cf = aug_payload["partner_context_frames"].to(
                    device, non_blocking=self.config.non_blocking,
                )
                partner_ca = aug_payload["partner_context_actions"].to(
                    device, non_blocking=self.config.non_blocking,
                )
                combined_cf = torch.cat([cf, partner_cf], dim=0)
                combined_ca = torch.cat([ca, partner_ca], dim=0)
                all_outputs = self.model(
                    combined_cf, combined_ca, include_latent=True,
                )
                n_main = cf.shape[0]
                # Split the combined outputs: first n_main rows are the
                # main batch (consumed by goal_property_loss + study_tags
                # heads), remainder are partner latents (contrastive only).
                outputs = {}
                for k, v in all_outputs.items():
                    if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == combined_cf.shape[0]:
                        outputs[k] = v[:n_main]
                    else:
                        outputs[k] = v
                _partner_latents = all_outputs["latent"][n_main:]
                _anchor_idxs = aug_payload["anchor_indices"].to(device)
                _margins = aug_payload["margins"].to(device)
            else:
                outputs = self.model(cf, ca, include_latent=need_latent)
                _partner_latents = None
                _anchor_idxs = None
                _margins = None
            class_weights = None
            if self.config.goal_type_class_weights is not None:
                class_weights = torch.tensor(
                    self.config.goal_type_class_weights,
                    dtype=torch.float32,
                    device=device,
                )
            active_mechanic_indices = None
            if self.config.active_mechanic_indices is not None:
                active_mechanic_indices = torch.tensor(
                    self.config.active_mechanic_indices,
                    dtype=torch.long,
                    device=device,
                )
            # Resolve loss weights: balancer override (if enabled) else
            # static config values. The balancer EMA is updated from THIS
            # step's outputs before reading effective weights — so the
            # tasks the model is currently uncertain about get up-weighted.
            w = {
                "agent_weight": self.config.agent_weight,
                "goal_type_weight": self.config.goal_type_weight,
                "keys_weight": self.config.keys_weight,
                "delta_px_weight": self.config.delta_px_weight,
                "mechanic_weight": self.config.mechanic_weight,
                "family_weight": self.config.family_weight,
                "subclass_weight": self.config.subclass_weight,
                "stage_weight": self.config.stage_weight,
                "cond_mechanic_weight": self.config.cond_mechanic_weight,
                "goal_family_weight": self.config.goal_family_weight,
                "role_buckets_weight": self.config.role_buckets_weight,
                "capability_weight": self.config.capability_weight,
                "topology_weight": self.config.topology_weight,
            }
            if self.adaptive_balancer is not None:
                self.adaptive_balancer.update(outputs, batch=batch)
                # Sync EMAs across DDP ranks so all ranks compute losses
                # with identical effective weights. No-op when not DDP.
                self.adaptive_balancer.sync_across_ranks()
                eff = self.adaptive_balancer.effective_weights()
                _task_to_weight = {
                    "agent":         "agent_weight",
                    "goal_type":     "goal_type_weight",
                    "keys":          "keys_weight",
                    "delta_px":      "delta_px_weight",
                    "mechanic":      "mechanic_weight",
                    "family":        "family_weight",
                    "subclass":      "subclass_weight",
                    "stage":         "stage_weight",
                    "cond_mechanic": "cond_mechanic_weight",
                    "goal_family":   "goal_family_weight",
                    "role_buckets":  "role_buckets_weight",
                    "capability":    "capability_weight",
                    "topology":      "topology_weight",
                }
                for task_name, wkey in _task_to_weight.items():
                    if task_name in eff:
                        w[wkey] = float(eff[task_name])
            losses = goal_property_loss(
                outputs, targets,
                agent_weight=w["agent_weight"],
                goal_type_weight=w["goal_type_weight"],
                keys_weight=w["keys_weight"],
                delta_px_weight=w["delta_px_weight"],
                mechanic_weight=w["mechanic_weight"],
                family_weight=w["family_weight"],
                subclass_weight=w["subclass_weight"],
                stage_weight=w["stage_weight"],
                cond_mechanic_weight=w["cond_mechanic_weight"],
                affordance_weight=self.config.affordance_weight,
                goal_type_class_weights=class_weights,
                active_mechanic_indices=active_mechanic_indices,
                mechanic_label_smoothing=self.config.mechanic_label_smoothing,
                family_label_smoothing=self.config.family_label_smoothing,
            )
            # study_tags.v001 supervision — only added when model has the heads
            # AND the batch carries the supervision tensors.
            if self.config.study_tags_heads and "goal_family_logits" in outputs:
                st_targets: dict[str, torch.Tensor] = {}
                for key in (
                    "goal_family", "role_buckets", "role_buckets_mask",
                    "capability_flags", "capability_mask",
                    "topology", "topology_mask",
                ):
                    if key in batch:
                        st_targets[key] = batch[key].to(
                            device, non_blocking=self.config.non_blocking,
                        )
                if st_targets:
                    st_losses = study_tags_loss(
                        outputs, st_targets,
                        goal_family_weight=w["goal_family_weight"],
                        role_buckets_weight=w["role_buckets_weight"],
                        capability_weight=w["capability_weight"],
                        topology_weight=w["topology_weight"],
                    )
                    # Merge per-head losses (prefixed) and roll total into the
                    # primary loss so the optimizer sees one objective.
                    for name in ("goal_family", "role_buckets", "capability", "topology"):
                        if name in st_losses:
                            losses[f"st_{name}"] = st_losses[name]
                    losses["total"] = losses["total"] + st_losses.get(
                        "total", losses["total"].new_zeros(()),
                    )
        if self.sigreg is not None:
            self.sigreg = self.sigreg.to(device)
            latent = outputs["latent"].float()
            sigreg_loss = self.sigreg(latent) * self.config.sigreg_weight
            losses["sigreg"] = sigreg_loss
            losses["latent_norm"] = latent.norm(dim=-1).mean().detach()
            losses["total"] = losses["total"] + sigreg_loss

        # Contrastive trunk loss — pull-apart partner episodes whose
        # mechanics confuse the model (or share a goal_family with
        # capability divergence). Latents were produced by the single
        # combined forward above; partner latents live in
        # ``_partner_latents`` and anchor latents are sliced out of
        # ``outputs["latent"]`` here.
        if _partner_latents is not None:
            from lewm.contrastive_supervision import contrastive_trunk_loss
            anchor_latents = outputs["latent"].index_select(0, _anchor_idxs)
            contrastive_loss_term = contrastive_trunk_loss(
                anchor_latents.float(),
                _partner_latents.float(),
                _margins,
            )
            weighted = contrastive_loss_term * self.config.contrastive_weight
            losses["contrastive"] = weighted
            losses["contrastive_pairs"] = torch.tensor(
                float(_anchor_idxs.shape[0]),
                device=device, dtype=torch.float32,
            )
            losses["total"] = losses["total"] + weighted

        if self.orthogonalizer is not None:
            # PCGrad path: each head's loss is back-propped under
            # retain_graph; trunk grads are projected against
            # conflicting peers and reassembled. Head-local grads come
            # from the single total_loss backward and are restored
            # after the orthogonaliser's scratch passes.
            from lewm.orthogonal_train_step import orthogonal_step
            # Filter to losses present in the dict (study_tags heads may
            # be absent on models built without them, contrastive may
            # be absent when weight=0, etc).
            head_losses = {
                k: v for k, v in losses.items()
                if k in self.config.orthogonal_head_names
                and isinstance(v, torch.Tensor)
                and v.requires_grad
            }
            self.optimizer.zero_grad()
            orthogonal_step(
                self.model,
                head_losses,
                losses["total"],
                self._TRUNK_PREFIXES,
                self.orthogonalizer,
            )
            # Feed per-head trunk grads to the balancer so it can
            # update per-task gradient-distribution Shannon entropy.
            if (
                self.adaptive_balancer is not None
                and hasattr(self.adaptive_balancer, "update_gradients")
                and self.orthogonalizer.last_per_head_grads
            ):
                self.adaptive_balancer.update_gradients(
                    self.orthogonalizer.last_per_head_grads,
                )
            # Live relational probe: EMA-update per-pair cosines, push
            # live per-task conflict rates into the balancer. This
            # OVERRIDES any static conflict_rate_file values per step.
            if (
                self.live_probe is not None
                and self.orthogonalizer.last_per_head_grads
            ):
                self.live_probe.update(self.orthogonalizer.last_per_head_grads)
                if (
                    self.adaptive_balancer is not None
                    and hasattr(self.adaptive_balancer, "update_conflicts")
                ):
                    self.adaptive_balancer.update_conflicts(
                        self.live_probe.per_task_conflict_rate(),
                    )
        else:
            self.optimizer.zero_grad()
            losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.global_step += 1

        return {k: float(v.item()) for k, v in losses.items()}


__all__ = [
    "GoalPropertyDataset",
    "GoalPropertyModel",
    "MockingbirdTrajectoryEncoder",
    "PropertyTrainConfig",
    "PropertyTrainer",
    "goal_property_loss",
    "study_tags_loss",
    "from_pretrained_add_study_tags_heads",
    "evaluate_property_model",
    "collate_property",
    "STUDY_TAGS_GOAL_FAMILIES",
    "STUDY_TAGS_ROLE_FIELDS",
    "STUDY_TAGS_CAPABILITY_FIELDS",
    "STUDY_TAGS_TOPOLOGY",
    "NUM_STUDY_TAGS_GOAL_FAMILIES",
    "NUM_STUDY_TAGS_ROLES",
    "NUM_STUDY_TAGS_CAPABILITIES",
    "NUM_STUDY_TAGS_TOPOLOGY",
    "GOAL_FAMILY_TO_IDX",
    "ROLE_FIELD_TO_IDX",
    "TOPOLOGY_TO_IDX",
]
