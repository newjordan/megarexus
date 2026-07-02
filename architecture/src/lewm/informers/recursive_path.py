"""GRAM Recursive Path Informer for Rexus.

Implements arXiv:2605.19376 'Generative Recursive Reasoning' as a PUCT prior.
Shifts inference from autoregressive guessing to deterministic latent search.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import torch
import numpy as np

from lewm.imagination import Informer, InformerContext
from lab.probes.active_inference_logic import ActiveInferenceProbe
from lab.probes.lattice_deduction_logic import LatticeDeductionProbe

if TYPE_CHECKING:
    from lewm.model import LeWorldModel


@dataclass(frozen=True)
class MechanicConstraint:
    action_id: int
    invariant: str
    confidence: float = 1.0


class MechanicRegistry:
    """Per-game store for verified action mechanics."""

    def __init__(self) -> None:
        self._by_game: dict[str, dict[int, MechanicConstraint]] = {}

    def update(self, game_id: str, action_id: int, outcome: np.ndarray, confidence: float = 1.0) -> None:
        nonzero = int(np.count_nonzero(outcome))
        invariant = f"verified_nonzero_count:{nonzero}"
        self._by_game.setdefault(game_id, {})[action_id] = MechanicConstraint(
            action_id=action_id,
            invariant=invariant,
            confidence=float(confidence),
        )

    def get(self, game_id: str | None, action_id: int) -> MechanicConstraint | None:
        if not game_id:
            return None
        return self._by_game.get(game_id, {}).get(action_id)

    def constraints_for(self, game_id: str | None, action_id: int) -> list[str]:
        constraint = self.get(game_id, action_id)
        return [constraint.invariant] if constraint else []

    def logit_adjustment(self, game_id: str | None, action_id: int) -> float:
        constraint = self.get(game_id, action_id)
        if constraint is None:
            return 0.0
        return min(1.0, max(0.0, constraint.confidence))


class RecursivePathInformer:
    """Informer that performs internal latent beam search (GRAM) for Rexus."""

    def __init__(
        self, 
        model: LeWorldModel, 
        fan_width: int = 8, 
        refinement_steps: int = 3
    ) -> None:
        self.model = model
        self.fan_width = fan_width
        self.refinement_steps = refinement_steps
        self.epistemic_probe = ActiveInferenceProbe()
        self.lattice_probe = LatticeDeductionProbe()
        self.mechanic_registry = MechanicRegistry()

    def update_mechanic_registry(self, game_id: str, action_id: int, outcome: np.ndarray):
        """Anchors a verified mechanic to prevent narrative drift."""
        self.mechanic_registry.update(game_id, action_id, outcome)
        print(f"REXUS: Mechanic Registry Updated for {game_id} | Action {action_id}")

    def __call__(self, ctx: InformerContext, avail: list[int]) -> torch.Tensor:
        """
        Perform GRAM: Width (fan out) + Depth (recursive refinement).
        Returns logits over available actions based on path consistency for Rexus.
        """
        if ctx.z_root is None:
            return torch.zeros(len(avail))

        device = ctx.z_root.device
        z_t = ctx.z_root.unsqueeze(0) # [1, D]
        
        action_scores = []
        for action_id in avail:
            # 1. Width Sampling (Fan Out)
            # Sample multiple stochastic trajectories for this action
            path_outcomes = []
            constraints = ["permanence"]
            constraints.extend(self.mechanic_registry.constraints_for(ctx.game_id, action_id))
            for _ in range(self.fan_width):
                # 2. Recursive Refinement (Depth)
                # Iteratively refine the latent outcome O
                z_o = z_t.clone()
                for _ in range(self.refinement_steps):
                    # Mock transition pass
                    z_o = self.model.predict_next(z_o, action_id)
                
                # Apply Lattice Projection (LDT logic)
                z_o = self.lattice_probe.apply_lattice_projection(
                    z_o.detach().cpu().numpy(), 
                    constraints=constraints
                )
                path_outcomes.append(z_o)

            # 3. Active Inference Scoring
            epistemic_val = self.epistemic_probe.calculate_epistemic_value(path_outcomes)
            
            # Action score = log(Epistemic Value) + verified mechanic prior.
            registry_bonus = self.mechanic_registry.logit_adjustment(ctx.game_id, action_id)
            action_scores.append(np.log(1.0 + epistemic_val) + registry_bonus)

        # Convert to logits
        logits = torch.tensor(action_scores, dtype=torch.float32, device=device)
        return logits

def make_gram_informer(model: LeWorldModel, weight: float = 1.5) -> Informer:
    """Factory to drop GRAM into the PriorMixer for Rexus."""
    gram = RecursivePathInformer(model)
    return Informer(name="gram_recursive_path", fn=gram, weight=weight)
