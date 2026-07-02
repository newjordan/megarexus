"""Goal vocabulary — discrete goal tokens for REX 3.0.

Turns continuous goal properties into a compositional VOCABULARY of tokens.
Each token represents a semantic aspect of the goal state (type, direction,
keys, magnitude). The vocabulary is built from the dataset and provides:

1. **Tokenization**: ground-truth properties → sequence of token ids
2. **Detection**: predicted properties → token ids (via argmax/bucket)
3. **Embedding**: token ids → combined goal latent (embedding sum)

This is fundamentally better than regressing continuous values because:
- Cross-entropy loss over discrete tokens is much easier than MSE regression
- The vocabulary is composable — tokens naturally combine
- Similar property combinations produce similar latents via embedding similarity
- Confidence is naturally captured by softmax probabilities

Vocabulary structure:
    type:       {0: delta_convergence, 1: keys_depleted, 2: pattern_completion, 3: end_of_trajectory}
    dx:         {0: <<-20, 1: -20..-5, 2: -5..+5, 3: +5..+20, 4: >>+20, 5: stationary}
    dy:         {0: <<-20, 1: -20..-5, 2: -5..+5, 3: +5..+20, 4: >>+20, 5: stationary}
    keys:       {0: none, 1: red, 2: blue, 3: green, 4: yellow, 5: purple, 6: orange, 
                 7: cyan, 8: magenta, 9: lime, 10: pink, 11: teal, 12: lavender,
                 13: brown, 14: beige, 15: maroon}
    delta_px:   {0: none(0), 1: low(1-5), 2: medium(5-20), 3: high(>20)}

USAGE:
    from lewm.goal_vocab import GOAL_VOCAB, tokenize_properties, detokenize_predictions

    # During data processing: convert ground-truth to tokens
    tokens = tokenize_properties(goal_type, agent_delta, keys_disappeared, delta_px)
    # tokens = {"type": 3, "dx": 2, "dy": 5, "keys": 0, "delta_px": 1}

    # During training (goal epoch): predict logits per vocab field
    # During training (LeWM epoch): convert predicted logits → embeddings → condition predictor
    embedder = GoalVocabEmbedder(vocab_sizes=GOAL_VOCAB.sizes(), latent_dim=192)
    z_goal = embedder(type_logits, dx_logits, dy_logits, keys_logits, delta_px_logits, confidence)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Vocabulary definition ────────────────────────────────────────────────────

@dataclass(frozen=True)
class VocabField:
    """A single vocabulary field."""
    name: str
    tokens: tuple[str, ...]
    # Bounds for discrete bucketing (only for continuous fields)
    buckets: tuple[tuple[float | None, float | None], ...] | None = None

    def size(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class GoalVocabulary:
    """Complete goal vocabulary — 5 fields with semantic tokens."""

    type: VocabField = field(default_factory=lambda: VocabField(
        name="goal_type",
        tokens=("delta_convergence", "keys_depleted", "pattern_completion", "end_of_trajectory"),
    ))
    dx: VocabField = field(default_factory=lambda: VocabField(
        name="agent_dx",
        tokens=("<< -20", "-20 to -5", "-5 to +5", "+5 to +20", ">> +20", "stationary"),
        buckets=((None, -20), (-20, -5), (-5, 5), (5, 20), (20, None), (0, 0)),
    ))
    dy: VocabField = field(default_factory=lambda: VocabField(
        name="agent_dy",
        tokens=("<< -20", "-20 to -5", "-5 to +5", "+5 to +20", ">> +20", "stationary"),
        buckets=((None, -20), (-20, -5), (-5, 5), (5, 20), (20, None), (0, 0)),
    ))
    keys: VocabField = field(default_factory=lambda: VocabField(
        name="keys_disappeared",
        tokens=("none", "red", "blue", "green", "yellow", "purple", "orange",
                "cyan", "magenta", "lime", "pink", "teal", "lavender",
                "brown", "beige", "maroon"),
    ))
    delta_px: VocabField = field(default_factory=lambda: VocabField(
        name="delta_px",
        tokens=("none (0)", "low (1-5)", "medium (5-20)", "high (>20)"),
        buckets=((0, 0), (1, 5), (5, 20), (20, None)),
    ))

    def fields(self) -> tuple[VocabField, ...]:
        return (self.type, self.dx, self.dy, self.keys, self.delta_px)

    def sizes(self) -> dict[str, int]:
        return {f.name: f.size() for f in self.fields()}


GOAL_VOCAB = GoalVocabulary()


# ── Tokenization ─────────────────────────────────────────────────────────────

def bucket_value(value: float | int, buckets: tuple[tuple[float | None, float | None], ...]) -> int:
    """Bucket a continuous value into a discrete token index."""
    for i, (lo, hi) in enumerate(buckets):
        if lo is None and hi is not None and value < hi:
            return i
        if lo is not None and hi is None and value >= lo:
            return i
        if lo is not None and hi is not None and lo <= value < hi:
            return i
    return len(buckets) - 1  # fallback to last bucket


def tokenize_properties(
    goal_type: str,
    agent_delta: tuple[float, float] | list[float],
    keys_disappeared: list[int],
    delta_px: int,
    vocab: GoalVocabulary = GOAL_VOCAB,
) -> dict[str, int]:
    """Convert ground-truth goal properties → discrete token ids.

    Returns dict of {field_name: token_id}.
    """
    dx, dy = agent_delta

    # Goal type
    try:
        type_idx = vocab.type.tokens.index(goal_type)
    except ValueError:
        type_idx = vocab.type.size() - 1  # fallback to last type

    # Agent delta — bucket dx, dy
    dx_idx = bucket_value(dx, vocab.dx.buckets)
    dy_idx = bucket_value(dy, vocab.dy.buckets)

    # Keys — take the FIRST key color that disappeared, or 0 (none)
    if keys_disappeared and len(keys_disappeared) > 0:
        key_color = keys_disappeared[0]
        keys_idx = key_color + 1 if 0 <= key_color < vocab.keys.size() - 1 else 1
    else:
        keys_idx = 0  # none

    # Delta px
    dp_idx = bucket_value(delta_px, vocab.delta_px.buckets)

    return {
        "goal_type": type_idx,
        "agent_dx": dx_idx,
        "agent_dy": dy_idx,
        "keys_disappeared": keys_idx,
        "delta_px": dp_idx,
    }


# ── Vocabulary-based goal latent embedder ────────────────────────────────────

class GoalVocabEmbedder(nn.Module):
    """Replace the MLP reconstructor with a vocab embedding table per field.

    Each field gets its own embedding table. The final goal latent is the
    sum of all field embeddings (learned, no MLP needed).

    To handle prediction uncertainty, we use SOFT embeddings from logits:
        embed = Σ_i softmax(logits_field)[i] · embedding_table[i]
    This is a differentiable way to handle partially confident predictions.

    Architecture:
        type_logits (4)    ──→ softmax → sum(emb[type]) ─┐
        dx_logits (6)      ──→ softmax → sum(emb[dx])   ├── sum → z_goal
        dy_logits (6)      ──→ softmax → sum(emb[dy])   │
        keys_logits (16)   ──→ softmax → sum(emb[keys])  │
        delta_px_logits (4)──→ softmax → sum(emb[dpx])   ┘
    """

    def __init__(
        self,
        vocab_sizes: dict[str, int],
        latent_dim: int = 192,
        embedding_dim: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.field_names = ["goal_type", "agent_dx", "agent_dy", "keys_disappeared", "delta_px"]

        # Per-field embedding tables
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_sizes[name], embedding_dim)
            for name in self.field_names
        })

        # Project sum of field embeddings → latent_dim
        self.project = nn.Linear(embedding_dim * len(self.field_names), latent_dim)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

        # Confidence head (learned scalar per sample)
        self.confidence_head = nn.Sequential(
            nn.Linear(latent_dim, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.confidence_head[0].weight)
        nn.init.zeros_(self.confidence_head[0].bias)

    def forward(
        self,
        type_logits: torch.Tensor,      # [B, 4]
        dx_logits: torch.Tensor,        # [B, 6]
        dy_logits: torch.Tensor,        # [B, 6]
        keys_logits: torch.Tensor,      # [B, 16]
        delta_px_logits: torch.Tensor,  # [B, 4]
    ) -> dict[str, torch.Tensor]:
        """Embed predicted logits → goal latent.

        Uses soft (distributional) embeddings for differentiable training.
        """
        # Soft-embed each field: sum_i softmax(logits)[i] · emb[i]
        field_embs = []
        for name, logits in [
            ("goal_type", type_logits),
            ("agent_dx", dx_logits),
            ("agent_dy", dy_logits),
            ("keys_disappeared", keys_logits),
            ("delta_px", delta_px_logits),
        ]:
            weights = F.softmax(logits, dim=-1)  # [B, V]
            emb = self.embeddings[name].weight  # [V, emb_dim]
            field_emb = weights @ emb  # [B, emb_dim]
            field_embs.append(field_emb)

        # Concatenate all field embeddings
        combined = torch.cat(field_embs, dim=-1)  # [B, 5 * emb_dim]
        z_goal = self.project(combined)  # [B, latent_dim]
        confidence = self.confidence_head(z_goal).squeeze(-1)  # [B]

        return {
            "z_goal": z_goal,
            "confidence": confidence,
        }


__all__ = [
    "GOAL_VOCAB",
    "GoalVocabulary",
    "VocabField",
    "GoalVocabEmbedder",
    "tokenize_properties",
    "bucket_value",
]
