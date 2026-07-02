#!/usr/bin/env python3
"""Multi-View LeWM (Fly-Eye Architecture) — prototype.

Architecture:
    Frame ─┬─ Encoder(p=16) → z_coarse ─┬─ Predictor(H=2) → plan_coarse ─┐
            ├─ Encoder(p=8)  → z_medium ─┼─ Predictor(H=5) → plan_medium ─┼─ Shannon Gate → Action
            └─ Encoder(p=4)  → z_fine   ─┴─ Predictor(H=8) → plan_fine   ─┘

Gating:
    1. Each branch runs CEM independently
    2. Compute per-branch entropy: H(branch) = -sum(p(action) * log(p(action)))
    3. Branch weight = softmax(-H / temperature)
    4. Weighted action selection
    5. Track branch win rate over recent N steps
    6. Prune branches with win_rate < threshold for M consecutive steps
    7. Reactivate pruned branches if active branches start failing

This is a drop-in replacement for the single-view CEM planner.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from lewm.cem_discrete import DiscreteCEMConfig, DiscreteCEMPlanner
from lewm.encoder import EncoderConfig
from lewm.model import LeWM, LeWMConfig
from lewm.predictor import PredictorConfig


# ═══════════════════════════════════════════════════════════════════════════
# Multi-View Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ViewConfig:
    """Configuration for one eye in the compound eye."""
    name: str
    patch_size: int          # 4, 8, or 16
    horizon: int             # planning horizon for this view
    num_samples: int = 50    # CEM samples
    num_iters: int = 5       # CEM iterations
    elite_count: int = 10    # CEM elite count
    smoothing: float = 0.0   # CEM smoothing
    temperature: float = 1.0  # for entropy weighting (lower = more selective)
    
    def make_encoder_config(self) -> EncoderConfig:
        return EncoderConfig(
            image_size=64, in_channels=16, patch_size=self.patch_size,
            num_layers=12, num_heads=3, hidden_dim=192, latent_dim=192,
        )
    
    def make_predictor_config(self, action_dim: int) -> PredictorConfig:
        return PredictorConfig(
            latent_dim=192, action_dim=action_dim,
            num_layers=6, num_heads=16, dropout=0.1,
        )
    
    def make_cem_config(self) -> DiscreteCEMConfig:
        return DiscreteCEMConfig(
            horizon=self.horizon,
            num_samples=self.num_samples,
            num_iterations=self.num_iters,
            elite_count=self.elite_count,
            num_actions=8,
            smoothing=self.smoothing,
        )


@dataclass
class FlyEyeConfig:
    """Configuration for the full compound-eye planner."""
    views: list[ViewConfig]
    
    # Gating
    entropy_temperature: float = 0.5   # lower = more selective
    pruning_window: int = 20           # steps to evaluate branch performance
    pruning_threshold: float = 0.2     # win rate below this → prune
    pruning_patience: int = 5          # consecutive steps below threshold
    reactivation_interval: int = 50    # steps before trying a pruned branch again
    min_active_views: int = 1          # never prune below this many views
    
    # Debug
    verbose: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Per-View Model
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ViewState:
    """Runtime state for one eye."""
    config: ViewConfig
    model: LeWM
    embedder: Any
    planner: DiscreteCEMPlanner
    active: bool = True
    
    # Statistics
    win_history: deque = field(default_factory=lambda: deque(maxlen=50))
    pruning_counter: int = 0
    last_reactivated: int = -1
    total_steps: int = 0
    total_wins: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Shannon Entropy Gate
# ═══════════════════════════════════════════════════════════════════════════

class ShannonGate:
    """Entropy-weighted action selection across multiple views.
    
    For each view that produces action probabilities p(a):
        H(view) = -sum(p(a) * log(p(a)))
        weight = softmax(-H / T) across views
    
    Final action: weighted vote across views.
    """
    
    def __init__(self, temperature: float = 0.5):
        self.temperature = temperature
    
    def compute_entropy(self, probs: torch.Tensor) -> float:
        """Compute Shannon entropy of action distribution. [H, num_actions] → scalar."""
        # Average entropy across horizon steps
        eps = 1e-8
        entropies = -(probs * torch.log(probs + eps)).sum(dim=-1)  # [H]
        return float(entropies.mean().item())
    
    def select_action(
        self,
        view_results: list[dict],
        view_states: list[ViewState],
        step_idx: int,
    ) -> tuple[int, dict[str, Any]]:
        """Select best action across views using entropy-weighted voting.
        
        Args:
            view_results: list of {plan, cost, probs, entropy} per active view
            view_states: corresponding ViewState objects
            step_idx: current step number
        
        Returns:
            (action_id, metadata) where metadata includes full gate diagnostics
        """
        active = [(i, r, vs) for i, (r, vs) in enumerate(zip(view_results, view_states))
                  if vs.active]
        
        if not active:
            # Fallback: use first view
            best_seq = view_results[0]["plan"].tolist()
            return best_seq[0], {"gate": "fallback", "active_views": 0}
        
        if len(active) == 1:
            i, r, vs = active[0]
            best_seq = r["plan"].tolist()
            return best_seq[0], {"gate": "single_view", "active_view": vs.config.name}
        
        # Compute weights from entropy
        entropies = [r["entropy"] for _, r, _ in active]
        neg_entropies = [-e / self.temperature for e in entropies]
        
        # Softmax
        max_ne = max(neg_entropies)
        exp_weights = [math.exp(ne - max_ne) for ne in neg_entropies]
        total = sum(exp_weights)
        weights = [w / total for w in exp_weights]
        
        # Weighted vote: each view proposes its first action, weight by confidence
        vote = {}
        for (i, r, vs), w in zip(active, weights):
            action = int(r["plan"][0].item())
            vote[action] = vote.get(action, 0.0) + w
        
        best_action = max(vote, key=vote.get)
        
        metadata = {
            "gate": "shannon_entropy",
            "active_views": len(active),
            "view_weights": {vs.config.name: round(w, 3) for (_, _, vs), w in zip(active, weights)},
            "view_entropies": {vs.config.name: round(e, 3) for (_, r, vs), e in zip(active, entropies)},
            "view_actions": {vs.config.name: int(r["plan"][0].item()) for (_, r, vs) in active},
            "vote_distribution": {int(k): round(v, 3) for k, v in vote.items()},
            "selected_by_weight": round(vote[best_action], 3),
        }
        
        return best_action, metadata


# ═══════════════════════════════════════════════════════════════════════════
# Eyedentity Pruner
# ═══════════════════════════════════════════════════════════════════════════

class EyedentityPruner:
    """Track and prune stale view branches.
    
    "Revolve eyedentity": branches that consistently fail get deactivated.
    They can be reactivated later to test if conditions have changed.
    """
    
    def __init__(
        self,
        window: int = 20,
        threshold: float = 0.2,
        patience: int = 5,
        reactivation_interval: int = 50,
        min_active: int = 1,
    ):
        self.window = window
        self.threshold = threshold
        self.patience = patience
        self.reactivation_interval = reactivation_interval
        self.min_active = min_active
    
    def update(self, view_states: list[ViewState], step_idx: int) -> list[str]:
        """Evaluate all views and prune/reactivate as needed.
        
        Returns list of events for logging.
        """
        events = []
        
        # Count active views
        n_active = sum(1 for vs in view_states if vs.active)
        
        for vs in view_states:
            if not vs.active:
                # Check reactivation
                if (step_idx - vs.last_reactivated) >= self.reactivation_interval:
                    vs.active = True
                    vs.pruning_counter = 0
                    vs.last_reactivated = step_idx
                    events.append(f"reactivated {vs.config.name}")
                continue
            
            # Compute recent win rate
            recent = list(vs.win_history)[-self.window:]
            if len(recent) < self.window:
                continue
            
            win_rate = sum(recent) / len(recent)
            
            if win_rate < self.threshold:
                vs.pruning_counter += 1
                if vs.pruning_counter >= self.patience and n_active > self.min_active:
                    vs.active = False
                    vs.pruning_counter = 0
                    vs.last_reactivated = step_idx
                    n_active -= 1
                    events.append(
                        f"pruned {vs.config.name} (win_rate={win_rate:.2f} "
                        f"over {len(recent)} steps)"
                    )
            else:
                vs.pruning_counter = max(0, vs.pruning_counter - 1)
        
        return events
    
    def record_outcome(self, view_states: list[ViewState], action_success: bool):
        """Record whether the selected action improved the state."""
        for vs in view_states:
            if vs.active:
                vs.win_history.append(1 if action_success else 0)
                vs.total_steps += 1
                if action_success:
                    vs.total_wins += 1


# ═══════════════════════════════════════════════════════════════════════════
# Fly-Eye Planner (orchestrator)
# ═══════════════════════════════════════════════════════════════════════════

class FlyEyePlanner:
    """Multi-view CEM planner with entropy gating and branch pruning."""
    
    def __init__(self, config: FlyEyeConfig, device: str = "cuda"):
        self.config = config
        self.device = device
        self.gate = ShannonGate(temperature=config.entropy_temperature)
        self.pruner = EyedentityPruner(
            window=config.pruning_window,
            threshold=config.pruning_threshold,
            patience=config.pruning_patience,
            reactivation_interval=config.reactivation_interval,
            min_active=config.min_active_views,
        )
        self.views: list[ViewState] = []
        self.step_idx = 0
        self.log: list[dict] = []
    
    def add_view(self, view_state: ViewState):
        self.views.append(view_state)
    
    def plan(
        self,
        current_frame: torch.Tensor,  # [C, H, W] one-hot
        goal_latent: torch.Tensor,    # [1, D] from primary encoder
        available_actions: list[int],
    ) -> tuple[int, dict[str, Any]]:
        """Plan next action using all active views.
        
        Args:
            current_frame: one-hot encoded frame [16, 64, 64]
            goal_latent: goal latent from the primary (coarse) encoder
            available_actions: list of valid action IDs
        
        Returns:
            (action_id, metadata) with full gate diagnostics
        """
        view_results = []
        
        for vs in self.views:
            if not vs.active:
                view_results.append({"plan": torch.tensor([1]), "cost": 0.0, "probs": None, "entropy": 999.0})
                continue
            
            # Run CEM for this view
            cem_result = vs.planner.plan(
                model=vs.model,
                embedder=vs.embedder,
                current_frame=current_frame,
                goal_latent=goal_latent,
                available_actions=available_actions,
                device=self.device,
            )
            
            # Compute entropy from CEM action probabilities
            if cem_result.final_probs is not None:
                entropy = self.gate.compute_entropy(cem_result.final_probs)
            else:
                entropy = 999.0
            
            view_results.append({
                "plan": cem_result.best_sequence,
                "cost": cem_result.best_cost,
                "probs": cem_result.final_probs,
                "entropy": entropy,
            })
        
        # Prune stale branches
        prune_events = self.pruner.update(self.views, self.step_idx)
        
        # Select action via Shannon gate
        action, metadata = self.gate.select_action(view_results, self.views, self.step_idx)
        metadata["prune_events"] = prune_events
        metadata["step"] = self.step_idx
        metadata["n_active"] = sum(1 for vs in self.views if vs.active)
        
        if self.config.verbose:
            self.log.append(metadata)
        
        self.step_idx += 1
        return action, metadata
    
    def record_outcome(self, success: bool):
        """Call after executing the action to update branch statistics."""
        self.pruner.record_outcome(self.views, success)
    
    def get_branch_stats(self) -> dict[str, dict]:
        """Return per-branch statistics."""
        return {
            vs.config.name: {
                "active": vs.active,
                "total_steps": vs.total_steps,
                "total_wins": vs.total_wins,
                "win_rate": vs.total_wins / max(vs.total_steps, 1),
                "recent_win_rate": (
                    sum(list(vs.win_history)[-20:]) / min(len(vs.win_history), 20)
                    if vs.win_history else 0.0
                ),
            }
            for vs in self.views
        }
    
    def get_log(self) -> list[dict]:
        return self.log


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

def make_fly_eye_planner(
    action_dim: int = 192,
    device: str = "cuda",
) -> FlyEyePlanner:
    """Create a fly-eye planner with 3 views at different timescales."""
    
    views = [
        ViewConfig(
            name="coarse",
            patch_size=16,
            horizon=2,
            num_samples=50,
            num_iters=5,
            elite_count=10,
            smoothing=0.0,
            temperature=1.0,
        ),
        ViewConfig(
            name="medium",
            patch_size=8,
            horizon=5,
            num_samples=100,
            num_iters=10,
            elite_count=15,
            smoothing=0.0,
            temperature=0.7,
        ),
        ViewConfig(
            name="fine",
            patch_size=4,
            horizon=8,
            num_samples=200,
            num_iters=15,
            elite_count=25,
            smoothing=0.0,
            temperature=0.3,
        ),
    ]
    
    config = FlyEyeConfig(
        views=views,
        entropy_temperature=0.5,
        pruning_window=20,
        pruning_threshold=0.2,
        pruning_patience=5,
        reactivation_interval=50,
        min_active_views=1,
        verbose=True,
    )
    
    planner = FlyEyePlanner(config, device=device)
    
    # Build each view's model (shared architecture, different patch sizes)
    for vc in views:
        enc_cfg = vc.make_encoder_config()
        pred_cfg = vc.make_predictor_config(action_dim)
        model = LeWM(LeWMConfig(encoder=enc_cfg, predictor=pred_cfg)).to(device)
        from lewm.data.arc_agi_3 import ARCActionEmbedder
        embedder = ARCActionEmbedder(action_dim=action_dim).to(device)
        cem_planner = DiscreteCEMPlanner(vc.make_cem_config(), seed=42)
        
        vs = ViewState(
            config=vc,
            model=model,
            embedder=embedder,
            planner=cem_planner,
        )
        planner.add_view(vs)
    
    return planner


# ═══════════════════════════════════════════════════════════════════════════
# Demo / smoke test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Fly-Eye Planner — Architecture Overview")
    print("=" * 60)
    
    planner = make_fly_eye_planner()
    
    print(f"\nViews:")
    for vs in planner.views:
        n_patches = (64 // vs.config.patch_size) ** 2
        n_params = sum(p.numel() for p in vs.model.parameters())
        print(f"  {vs.config.name:>8s}: patch={vs.config.patch_size} "
              f"→ {n_patches} patches, H={vs.config.horizon}, "
              f"{n_params//1000}K params")
    
    print(f"\nGate: Shannon entropy (T={planner.config.entropy_temperature})")
    print(f"Pruner: window={planner.config.pruning_window}, "
          f"threshold={planner.config.pruning_threshold}, "
          f"patience={planner.config.pruning_patience}")
    
    print(f"\nTotal params: {sum(sum(p.numel() for p in vs.model.parameters()) for vs in planner.views)//1000}K")
    print(f"(vs single-view: {sum(p.numel() for p in planner.views[0].model.parameters())//1000}K)")
    print(f"\nNext: train each view on ARC-AGI-3 at its native patch size.")
    print(f"Then run CEM with fly-eye gating and measure vs single-view baseline.")
