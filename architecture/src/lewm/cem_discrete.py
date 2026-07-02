"""Discrete-action CEM planner for ARC-AGI-3.

Adapted from LeWM paper Section 3.2 (latent planning with CEM) for ARC's
categorical action space (action_id ∈ {0..7}).

Paper CEM (continuous):
  1. Initialize μ_0 = 0, Σ_0 = I  (Gaussian over action sequences)
  2. Sample N sequences from N(μ, diag(Σ²))
  3. Score via cost_fn (terminal latent MSE: ‖ẑ_H - z_goal‖²₂)
  4. Pick K elites (lowest cost)
  5. Update μ, σ from elite statistics
  6. Repeat 2-5 for I iterations

Discrete CEM (categorical):
  1. Initialize uniform probs per timestep: P(action_id = k) = 1/|A|
  2. Sample N action_id sequences from categorical(probs)
  3. Roll out frozen Rex predictor in latent space to get ẑ_H
  4. Score each: cost = ‖ẑ_H - z_goal‖²₂  (paper equation 4)
  5. Pick K elites (lowest cost)
  6. Update probs from elite action counts (+ smoothing)
  7. Repeat 2-6 for I iterations

Key differences from paper:
  - Categorical sampling (not Gaussian) — ARC actions are discrete
  - Action space varies per game (|A| ∈ {1..7}) — we query env.action_space
  - Cost is latent-space MSE between predicted terminal latent and encoded goal
  - Goal must be defined externally (e.g., target grid from task spec)

Reference: LeWM paper arXiv:2603.19312, Section 3.2, Equation 4, Appendix D.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from lewm.model import LeWM
from lewm.data.arc_agi_3 import ARCActionEmbedder, frame_to_onehot


PAPER_CEM_DEFAULTS = dict(
    num_samples=300, num_iterations=30, elite_count=30, horizon=5,
)


@dataclass(frozen=True)
class DiscreteCEMConfig:
    """Configuration for discrete-action CEM planner on ARC-AGI-3.

    Defaults match paper Push-T setting (Appendix D):
      num_samples=300, num_iterations=30, elite_count=30, horizon=5
    """
    horizon: int = PAPER_CEM_DEFAULTS["horizon"]
    num_samples: int = PAPER_CEM_DEFAULTS["num_samples"]
    num_iterations: int = PAPER_CEM_DEFAULTS["num_iterations"]
    elite_count: int = PAPER_CEM_DEFAULTS["elite_count"]
    num_actions: int = 8  # action_id ∈ {0..7}, but effective count varies per game
    smoothing: float = 0.1  # mix elite probs with prior to avoid collapse
    temp: float = 1.0  # softmax temperature for probability updates (higher = more exploration)

    def validate(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.num_samples < 2:
            raise ValueError("num_samples must be >= 2")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be >= 1")
        if self.elite_count < 1 or self.elite_count > self.num_samples:
            raise ValueError("elite_count must be in [1, num_samples]")
        if self.num_actions < 1:
            raise ValueError("num_actions must be >= 1")
        if not 0.0 <= self.smoothing <= 1.0:
            raise ValueError("smoothing must be in [0, 1]")


@dataclass
class DiscreteCEMResult:
    best_sequence: torch.Tensor         # [horizon] long int, action_id sequence
    best_cost: float
    final_probs: torch.Tensor           # [horizon, num_actions]
    iterations: int
    all_samples: torch.Tensor | None = None     # optional: [num_iterations, num_samples, horizon]
    all_costs: torch.Tensor | None = None       # optional: [num_iterations, num_samples]


class DiscreteCEMPlanner:
    """Discrete-action CEM planner for ARC-AGI-3.

    Samples action_id sequences from a categorical distribution, scores them
    via predicted-vs-goal latent MSE, and iteratively refines the distribution.

    Usage:
        planner = DiscreteCEMPlanner()
        result = planner.plan(
            model=rex_model,
            embedder=action_embedder,
            current_frame=obs_frame,      # np.ndarray (64, 64)
            goal_frame=target_frame,      # np.ndarray (64, 64) — optional
            goal_latent=target_latent,    # torch.Tensor [1, D] — alternative
            available_actions=[0, 1, 4],  # valid action_ids for this game
            device='cuda',
        )
    """

    def __init__(
        self,
        config: DiscreteCEMConfig | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.config = config or DiscreteCEMConfig()
        self.config.validate()
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    @torch.no_grad()
    def plan(
        self,
        *,
        model: LeWM,
        embedder: ARCActionEmbedder,
        current_frame: torch.Tensor,    # [C, H, W] one-hot float (from frame_to_onehot)
        goal_latent: torch.Tensor,      # [1, D] encoder embedding of goal frame
        available_actions: list[int],   # valid action_ids for this game (e.g., [1,2,3,4])
        device: str = "cpu",
        return_trace: bool = False,
        seed: int | None = None,
    ) -> DiscreteCEMResult:
        """Plan an action sequence toward goal_latent.

        Args:
            model: Frozen LeWM model.
            embedder: ARCActionEmbedder.
            current_frame: [C, H, W] one-hot tensor of current observation.
            goal_latent: [1, D] encoder embedding of goal/reference frame.
            available_actions: List of valid action_ids (e.g., [1, 4]).
            device: Torch device.
            return_trace: If True, store all sample sequences and costs.

        Returns:
            DiscreteCEMResult with best sequence and costs.
        """
        cfg = self.config
        num_avail = len(available_actions)
        if num_avail == 0:
            raise ValueError("available_actions must not be empty")

        # Create a generator on the correct device
        gen = torch.Generator(device=device)
        seed_val = seed if seed is not None else (
            self.generator.initial_seed() if hasattr(self, 'generator') else 42
        )
        gen.manual_seed(seed_val)

        # Map action IDs to indices 0..num_avail-1 for sampling efficiency
        idx_to_action = torch.tensor(available_actions, dtype=torch.long, device=device)

        # Encode current frame → z_current
        z_current = model.encoder(current_frame.unsqueeze(0).to(device))  # [1, D]

        # Initialize uniform probabilities over available actions
        probs = torch.full(
            (cfg.horizon, num_avail),
            1.0 / num_avail,
            device=device,
        )

        best_cost = float("inf")
        best_sequence: torch.Tensor | None = None

        # Optional trace storage
        all_samples_list = [] if return_trace else None
        all_costs_list = [] if return_trace else None

        for _ in range(cfg.num_iterations):
            # 1. Sample categorical action sequences
            #    samples: [num_samples, horizon] in index space (0..num_avail-1)
            flat_probs = probs.unsqueeze(0).expand(cfg.num_samples, -1, -1)  # [N, H, |A|]
            flat_probs_2d = flat_probs.reshape(-1, num_avail)                # [N*H, |A|]
            sample_indices = torch.multinomial(
                flat_probs_2d, 1, generator=gen,
            ).reshape(cfg.num_samples, cfg.horizon)                         # [N, H]
            action_seqs = idx_to_action[sample_indices]                      # [N, H]

            # 2. Roll out predictor in latent space for each candidate
            #    z_current: [1, D] → expand to [N, 1, D]
            z_hist = z_current.unsqueeze(0).expand(cfg.num_samples, -1, -1)  # [N, 1, D]

            # Build action embeddings for all candidate sequences
            # action_seqs: [N, H], need action_x, action_y, complex_mask
            zeros = torch.zeros(cfg.num_samples, cfg.horizon, device=device, dtype=torch.long)
            # We need to know which actions are complex to set the mask properly
            # For now, use action_embedder directly: action_x=0, action_y=0, mask=1 if action==6
            complex_mask = (action_seqs == 6).float()  # ACTION6 is complex
            action_emb = embedder(
                action_seqs,
                zeros,        # action_x=0
                zeros,        # action_y=0
                complex_mask,
            )  # [N, H, action_dim]

            # Autoregressive rollout: predictor expects z_history and actions
            # to have SAME sequence length T. So we build both incrementally:
            #   z_hist = [z_current, z_pred_1, ..., z_pred_{t-1}]  [N, t+1, D]
            #   actions = [a_0, a_1, ..., a_t]                      [N, t+1, A]
            # At each step we feed the full history so far and read the last
            # output (position t) as the next prediction z_pred_t.
            z_hist = z_current.unsqueeze(1).expand(cfg.num_samples, -1, -1)  # [N, 1, D]
            for t in range(cfg.horizon):
                # Actions used so far: [a_0, ..., a_t]  [N, t+1, A]
                action_sofar = action_emb[:, :t+1, :]  # [N, t+1, A]
                out = model.predictor(z_hist, action_sofar)  # [N, t+1, D]
                # Position t of output is the next latent prediction
                z_pred = out[:, -1:, :]  # [N, 1, D]
                if t < cfg.horizon - 1:
                    z_hist = torch.cat([z_hist, z_pred], dim=1)
                else:
                    z_hist = z_pred  # last step's prediction = terminal latent

            # Predicted terminal latent: z_hist[:, -1, :]  [N, D]
            z_terminal = z_hist[:, -1, :]  # [N, D]

            # 3. Score: cost = ||predicted_terminal - goal_latent||²₂  (paper eq 4)
            costs = (z_terminal - goal_latent.to(device)).pow(2).mean(dim=-1)  # [N]

            # Track best
            min_cost, min_idx = torch.min(costs, dim=0)
            min_cost_val = float(min_cost.item())
            if min_cost_val < best_cost:
                best_cost = min_cost_val
                best_sequence = action_seqs[min_idx].clone()

            # 4. Pick elites (lowest cost)
            elite_idx = torch.topk(costs, k=cfg.elite_count, largest=False).indices  # [K]
            elite_seqs = action_seqs[elite_idx]  # [K, H]

            # 5. Update probs from elite action counts
            #    One-hot encode elite actions → count → normalize
            elite_onehot = nn.functional.one_hot(
                sample_indices[elite_idx], num_classes=num_avail,
            ).float()  # [K, H, |A|]
            elite_counts = elite_onehot.sum(dim=0)  # [H, |A|]
            # Smooth: mix with prior to avoid collapse on small elite counts
            new_probs = elite_counts / elite_counts.sum(dim=-1, keepdim=True).clamp_min(1)
            probs = (1.0 - cfg.smoothing) * new_probs + cfg.smoothing * probs

            # Optional trace
            if return_trace:
                all_samples_list.append(action_seqs.cpu())
                all_costs_list.append(costs.cpu())

        if best_sequence is None:
            raise RuntimeError("DiscreteCEM failed to find any valid sequence")

        all_samples = torch.stack(all_samples_list) if return_trace else None
        all_costs = torch.stack(all_costs_list) if return_trace else None

        return DiscreteCEMResult(
            best_sequence=best_sequence,
            best_cost=best_cost,
            final_probs=probs,
            iterations=cfg.num_iterations,
            all_samples=all_samples,
            all_costs=all_costs,
        )


__all__ = ["DiscreteCEMConfig", "DiscreteCEMResult", "DiscreteCEMPlanner"]
