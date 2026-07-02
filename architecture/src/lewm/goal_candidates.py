"""Pareto-ranked goal candidates with Shannon-entropy uncertainty signal.

Per LeWM-v2 architecture: instead of a single ``goal_latent``, expose a
distribution over plausible goals so the planner can:
  - exploit (low entropy → plan toward the mode)
  - explore (high entropy → sample a goal per simulation)
  - boost c_puct under uncertainty (entropy-modulated exploration)

Candidates are ranked by a non-learned Pareto product over three signals
(each in [0, 1]):

  - **distinctness**:  color_rarity × shape_uniqueness (visually pops out)
  - **extremity**:     distance from grid center / grid diagonal (corners/edges)
  - **persistence**:   fraction of multi-frame structures this sprite-color
                        appears in (stable goals > transient artifacts)

Pareto product = distinctness × extremity × persistence; centered/softmaxed
to a probability distribution over the top-K candidates.

Shannon entropy ``H = -Σ p log p`` (in nats) is the planner's uncertainty
signal:
  - ``H < 0.5 nats`` → one clear winner; plan toward it
  - ``H > 2.0 nats``  → genuinely ambiguous; sample per simulation
  - in between → linear blend

Integrates with MCTS via ``goal_inference_fn`` (returns the mode's z_goal)
and feeds value_fn slot if used as a multi-goal aggregator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import torch

from typing import TYPE_CHECKING

from lewm.frame_structure import (
    ARC_GRID_SIZE,
    BBox,
    FrameStructure,
    SpriteRegion,
    parse_frame,
)

if TYPE_CHECKING:
    from loom_layers.gamemaster_procedural import GameInstance


ZGoalFn = Callable[[np.ndarray, BBox], torch.Tensor]
"""(frame, sprite_bbox) → z_goal [D]. Caller supplies — typically wraps
the LeWM encoder with a sprite-highlight mask."""


@dataclass
class GoalCandidate:
    """One candidate goal: a sprite + its Pareto score + lazy z_goal."""

    sprite: SpriteRegion
    score: float                                     # Pareto product (pre-softmax)
    probability: float                               # softmaxed across top-K
    evidence: dict                                   # debug: distinctness/extremity/persistence
    z_goal: Optional[torch.Tensor] = None            # lazily filled via ZGoalFn

    @property
    def centroid(self) -> tuple[float, float]:
        return self.sprite.centroid


@dataclass
class GoalCandidateDistribution:
    """Top-K candidates + Shannon entropy + uncertainty flag."""

    candidates: list[GoalCandidate]                  # sorted by probability descending
    entropy: float                                   # nats
    is_uncertain: bool

    @property
    def mode(self) -> Optional[GoalCandidate]:
        return self.candidates[0] if self.candidates else None

    def sample(self, rng: np.random.Generator) -> Optional[GoalCandidate]:
        if not self.candidates:
            return None
        probs = np.array([c.probability for c in self.candidates])
        idx = int(rng.choice(len(self.candidates), p=probs))
        return self.candidates[idx]


# ─────────────────────────────────────────────────────────────────────
# Pareto scoring components
# ─────────────────────────────────────────────────────────────────────


def _color_rarity(sprite: SpriteRegion, sprites: list[SpriteRegion]) -> float:
    """1 / count(sprites_with_same_color). Rare-colored sprites score higher."""

    same = sum(1 for s in sprites if s.color == sprite.color)
    return 1.0 / max(1, same)


def _shape_uniqueness(sprite: SpriteRegion, sprites: list[SpriteRegion]) -> float:
    """1 / count(sprites_with_similar_bbox). Distinctive shapes pop."""

    # Similarity = bbox area within 1.5× of this sprite's area.
    target_area = max(1, sprite.pixel_count)
    similar = sum(
        1 for s in sprites
        if 0.667 * target_area <= s.pixel_count <= 1.5 * target_area
    )
    return 1.0 / max(1, similar)


def _distinctness(sprite: SpriteRegion, sprites: list[SpriteRegion]) -> float:
    return _color_rarity(sprite, sprites) * _shape_uniqueness(sprite, sprites)


def _extremity(centroid: tuple[float, float], extent: BBox) -> float:
    """Normalized Manhattan distance from grid center.

    Corners and edges score near 1; dead center scores near 0.
    """

    cy = (extent.y0 + extent.y1 - 1) / 2.0
    cx = (extent.x0 + extent.x1 - 1) / 2.0
    dy = abs(centroid[0] - cy) / max(1.0, (extent.y1 - extent.y0) / 2.0)
    dx = abs(centroid[1] - cx) / max(1.0, (extent.x1 - extent.x0) / 2.0)
    # Manhattan in normalized space, clamped to [0, 1].
    return float(min(1.0, (dy + dx) / 2.0))


def _persistence(
    sprite: SpriteRegion,
    historical_structures: Optional[list[FrameStructure]],
) -> float:
    """Fraction of historical frames where a sprite of this color appears.

    Without history, defaults to 1.0 (single-frame mode).
    """

    if not historical_structures:
        return 1.0
    n = len(historical_structures)
    appears = sum(
        1 for fs in historical_structures
        if any(s.color == sprite.color for s in fs.sprites)
    )
    return appears / max(1, n)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


def propose_goals(
    frame: np.ndarray,
    *,
    structure: Optional[FrameStructure] = None,
    historical_structures: Optional[list[FrameStructure]] = None,
    top_k: int = 5,
    z_goal_fn: Optional[ZGoalFn] = None,
    softmax_temperature: float = 0.5,
    uncertainty_threshold_nats: float = 1.0,
) -> GoalCandidateDistribution:
    """Build a top-K Pareto-ranked goal-candidate distribution from one frame.

    Parameters
    ----------
    frame:
        Current observation, ``(64, 64)`` uint8/int.
    structure:
        Pre-computed structure for this frame (e.g. cached multi-frame
        parse). If absent, parsed from ``frame`` alone.
    historical_structures:
        Optional list of structures from earlier frames in the same game,
        used to score persistence.
    top_k:
        Maximum candidates returned.
    z_goal_fn:
        If provided, each candidate's ``z_goal`` is computed via this
        callback. Otherwise candidates' ``z_goal`` is ``None`` (caller
        fills lazily).
    softmax_temperature:
        Higher → flatter distribution; lower → sharper.
    uncertainty_threshold_nats:
        ``entropy > threshold`` → distribution flagged uncertain.
    """

    struct = structure or parse_frame(frame)
    if not struct.sprites:
        return GoalCandidateDistribution(candidates=[], entropy=0.0, is_uncertain=False)

    # Score every sprite, then pick the top_k by raw Pareto product.
    scored: list[tuple[SpriteRegion, float, dict]] = []
    for sp in struct.sprites:
        d = _distinctness(sp, struct.sprites)
        e = _extremity(sp.centroid, struct.grid_extent)
        p = _persistence(sp, historical_structures)
        prod = d * e * p
        scored.append((sp, prod, {"distinctness": d, "extremity": e, "persistence": p}))

    scored.sort(key=lambda s: -s[1])
    top = scored[:top_k]

    # Softmax to a probability distribution.
    raw = np.array([s[1] for s in top], dtype=np.float64)
    if raw.sum() == 0:
        probs = np.full(len(top), 1.0 / len(top))
    else:
        logits = np.log(raw + 1e-12) / max(1e-6, softmax_temperature)
        logits = logits - logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()

    entropy_nats = float(-np.sum(probs * np.log(probs + 1e-12)))
    is_uncertain = entropy_nats > uncertainty_threshold_nats

    candidates = []
    for (sp, prod, evidence), pr in zip(top, probs):
        z_goal = z_goal_fn(frame, sp.bbox) if z_goal_fn is not None else None
        candidates.append(GoalCandidate(
            sprite=sp, score=float(prod), probability=float(pr),
            evidence=evidence, z_goal=z_goal,
        ))

    return GoalCandidateDistribution(
        candidates=candidates, entropy=entropy_nats, is_uncertain=is_uncertain,
    )


def shannon_entropy(probs: np.ndarray | list[float]) -> float:
    """Helper: Shannon entropy in nats from a probability vector."""

    p = np.asarray(probs, dtype=np.float64)
    p = p / max(1e-12, p.sum())
    return float(-np.sum(p * np.log(p + 1e-12)))


def from_game_instance(
    instance: "GameInstance",
    *,
    frame: Optional[np.ndarray] = None,
    z_goal_fn: Optional[ZGoalFn] = None,
) -> GoalCandidateDistribution:
    """Authoritative goal distribution for a GameMaster-generated game.

    Since `GameInstance` exposes `goal_cell` directly, the distribution
    has exactly one candidate with probability 1.0 and entropy 0. No
    Pareto ranking needed — the goal is known. The single candidate's
    sprite is built from the goal cell's color in the glyph_grid.

    If a separate `frame` is given, sprites are extracted from that
    frame (e.g. the live observation); otherwise the instance's
    `glyph_grid` is used.
    """

    use_frame = np.asarray(
        frame if frame is not None else instance.glyph_grid, dtype=np.int64,
    )
    gy, gx = (int(v) for v in instance.goal_cell)
    color = int(use_frame[gy, gx])
    goal_sprite = SpriteRegion(
        color=color,
        bbox=BBox(y0=gy, x0=gx, y1=gy + 1, x1=gx + 1),
        pixel_count=1,
        centroid=(float(gy), float(gx)),
    )
    z = z_goal_fn(use_frame, goal_sprite.bbox) if z_goal_fn is not None else None
    candidate = GoalCandidate(
        sprite=goal_sprite, score=1.0, probability=1.0,
        evidence={"source": "GameInstance.goal_cell"},
        z_goal=z,
    )
    return GoalCandidateDistribution(
        candidates=[candidate], entropy=0.0, is_uncertain=False,
    )


__all__ = [
    "GoalCandidate", "GoalCandidateDistribution",
    "propose_goals", "from_game_instance",
    "shannon_entropy", "ZGoalFn",
]
