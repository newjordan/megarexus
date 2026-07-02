"""Heuristic goal inference (Tier 0) — pure-vision, no extra training.

Sibling module to `lewm.goal_inference` (which is the LEARNED Tier-3
trajectory-conditioned goal predictor). This module is Tier 0: zero
training, runs at inference via existing sprite + Pareto-ranking infra.

Tiering rationale (v2b roadmap):
- Tier 0 (here): heuristic from frame_structure + goal_candidates. Cheap,
  no model loading. Validates the goal-inference-as-bottleneck hypothesis.
- Tier 1: SAM cached masks from `experiments/2026-05-14_sam2_map_pack_001_smoke_v1/`.
- Tier 2: Nemotron judge over top-K candidates (via `loom_layers.nemo_judge`).
- Tier 3: learned model in `lewm.goal_inference` (existing, trains a transformer).

The world model was trained on hindsight tuples
``(obs, action, goal=actual_next_frame)``. At inference the actual next
frame is unknown — using `encoder(current_frame)` as the goal latent was
the v2b-1 finding's gate-blocker (the model is then asked "given current
state and current state as goal, what action?" → null-action).

This module's algorithm:
  1. Parse the current frame via `lewm.frame_structure.parse_frame`.
  2. Rank candidates with `lewm.goal_candidates.propose_goals` (Pareto over
     distinctness × extremity × persistence; Shannon-entropy uncertainty).
  3. Take the highest-scoring candidate (mode of the distribution).
  4. Build a "highlighted" frame where only that sprite's bbox is visible
     (rest is filled with background color).
  5. Encode the highlighted frame → z_goal.

The signal: the planner is told "aim for the visually-distinctive thing
in the corner / edge / unique color" rather than "aim for current state."

Integrates with `MCTSConfig.goal_inference_fn` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from lewm.data.arc_agi_3 import frame_to_onehot
from lewm.frame_structure import (
    ARC_GRID_SIZE,
    BBox,
    FrameStructure,
    SpriteRegion,
    parse_frame,
)
from lewm.goal_candidates import (
    GoalCandidateDistribution,
    propose_goals,
)


def _highlight_sprite(
    frame: np.ndarray,
    bbox: BBox,
    background_color: int,
) -> np.ndarray:
    """Return a frame where only `bbox` keeps its original pixels; rest = background."""

    out = np.full_like(frame, background_color, dtype=frame.dtype)
    out[bbox.y0:bbox.y1, bbox.x0:bbox.x1] = frame[bbox.y0:bbox.y1, bbox.x0:bbox.x1]
    return out


def _coerce_to_frame_np(frame: torch.Tensor | np.ndarray) -> np.ndarray:
    """One-hot [C, H, W] tensor or color-index [H, W] array → int8 [64, 64] color-index."""

    if isinstance(frame, torch.Tensor):
        if frame.dim() == 3:
            arr = frame.argmax(dim=0).detach().cpu().numpy()
        elif frame.dim() == 2:
            arr = frame.detach().cpu().numpy()
        else:
            raise ValueError(f"frame tensor must be 2D or 3D; got {tuple(frame.shape)}")
    else:
        arr = np.asarray(frame)
    return arr.astype(np.int8)


@dataclass
class HeuristicGoalInferenceFn:
    """Callable adapter satisfying ``MCTSConfig.goal_inference_fn`` signature.

    Build once per planning session; MCTS invokes it once per ``plan()``.
    The encoder must accept ``[B, C, H, W]`` one-hot frames (matches
    ViTTinyEncoder + MegaREX-small).
    """

    encoder: nn.Module
    device: str = "cpu"
    bf16: bool = True
    top_k: int = 5
    historical_structures: Optional[list[FrameStructure]] = None
    # If True, fall back to encoding the FULL current frame (degenerate proxy)
    # when no sprite candidate is found. False = raise.
    fallback_to_current: bool = True
    # Telemetry — updated per call.
    last_entropy: float = field(default=0.0, init=False)
    last_n_candidates: int = field(default=0, init=False)
    last_top_candidate_score: float = field(default=0.0, init=False)
    last_used_fallback: bool = field(default=False, init=False)

    def __call__(self, frame: torch.Tensor | np.ndarray) -> torch.Tensor:
        frame_np = _coerce_to_frame_np(frame)
        if frame_np.shape != (ARC_GRID_SIZE, ARC_GRID_SIZE):
            raise ValueError(
                f"frame must be ({ARC_GRID_SIZE}, {ARC_GRID_SIZE}); got {frame_np.shape}"
            )

        struct = parse_frame(frame_np)
        distribution: GoalCandidateDistribution = propose_goals(
            frame_np,
            structure=struct,
            historical_structures=self.historical_structures,
            top_k=self.top_k,
        )
        self.last_entropy = float(distribution.entropy)
        self.last_n_candidates = len(distribution.candidates)

        if not distribution.candidates:
            self.last_used_fallback = True
            self.last_top_candidate_score = 0.0
            if not self.fallback_to_current:
                raise RuntimeError(
                    f"no goal candidates found in frame (background={struct.background_color}, "
                    f"n_sprites={len(struct.sprites)})"
                )
            target_frame = frame_np
        else:
            self.last_used_fallback = False
            top = distribution.mode
            assert top is not None
            self.last_top_candidate_score = float(top.score)
            target_frame = _highlight_sprite(
                frame_np, top.sprite.bbox, struct.background_color,
            )

        obs = frame_to_onehot(target_frame).to(self.device)
        if self.bf16:
            obs = obs.to(torch.bfloat16)
        with torch.no_grad():
            z = self.encoder(obs.unsqueeze(0))  # [1, D]
        return z


__all__ = ["HeuristicGoalInferenceFn", "_highlight_sprite", "_coerce_to_frame_np"]
