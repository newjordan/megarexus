"""Joint LeWM + Goal Inference training with evidential heads and belief state.

Architecture (REX 3.1):

1. **Goal model** (LeWMGoalInferenceModel) with evidential heads:
   - goal_type: Dirichlet distribution over 4 classes
   - agent_delta / delta_px: Normal-Inverse-Gamma (evidential regression)
   - keys: Independent Beta distributions

2. **BeliefState** — Kalman-style belief over goal properties:
   - Observation step: goal model output → Bayesian conjugate update
   - Error correction step: LeWM prediction mismatch → learned gate adjusts beliefs
   - Confidence lives at the property belief level (not the reconstructor)

3. **GoalLatentReconstructor** — deterministic mapping from MAP beliefs → z_goal
   - Always outputs 100% signal, no confidence gating
   - Confidence is queryable from BeliefState

4. **LeWM feedback loop**: MSE(ẑ_H, z_goal) → corrects belief state → better goals
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.belief_state import BeliefState
from lewm.encoder import EncoderConfig, ViTTinyEncoder
from lewm.evidential_heads import (
    ARC_NUM_COLORS,
    NUM_GOAL_TYPES,
    dirichlet_nll_loss,
    dirichlet_regularization,
    nig_nll_loss,
    nig_reg_loss,
    beta_nll_loss,
)
from lewm.goal_latent import GoalLatentConfig, GoalLatentReconstructor
from lewm.goal_property import (
    GoalPropertyDataset,
    collate_property,
    evaluate_property_model,
    goal_property_loss,
)
from lewm.goal_lewm import LeWMGoalInferenceModel
from lewm.loss import lewm_loss
from lewm.model import LeWM, LeWMConfig
from lewm.predictor import AdaLNTransformerPredictor, PredictorConfig
from lewm.sigreg import SIGReg, SIGRegConfig


# ── Evidential loss function ────────────────────────────────────────────

# Number of properties in the belief state (4 goal_type + 2 agent + 16 keys + 1 delta_px)
N_GOAL_PROPERTIES = NUM_GOAL_TYPES + 2 + ARC_NUM_COLORS + 1  # 4 + 2 + 16 + 1 = 23


def evidential_goal_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    goal_type_weight: float = 0.5,
    agent_weight: float = 1.0,
    keys_weight: float = 0.05,  # Beta NLL is ~10-100× BCE, scale down
    delta_px_weight: float = 0.1,
    evidence_annealing: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Combined evidential loss for all goal property heads.

    Uses distributional losses (Dirichlet NLL, NIG NLL, Beta NLL)
    plus evidence regularization terms.

    NOTE: Beta NLL has a much larger magnitude than BCE (~10-100×), so
    keys_weight is set low (0.05) to compensate.

    Args:
        outputs: dict with evidential parameters from LeWMGoalInferenceModel
        targets: dict with ground-truth labels
        evidence_annealing: scale for evidence regularizers (start low, ramp up)

    Returns:
        dict with per-task and total losses
    """
    losses = {}

    # --- Dirichlet: goal_type ---
    gt_nll = dirichlet_nll_loss(outputs["goal_type_alpha"], targets["goal_type"])
    gt_reg = dirichlet_regularization(outputs["goal_type_alpha"], targets["goal_type"])
    losses["goal_type"] = gt_nll * goal_type_weight + gt_reg * evidence_annealing * 0.1

    # --- NIG: agent_delta ---
    ad_nll = nig_nll_loss(
        targets["agent_delta"],
        outputs["agent_gamma"],
        outputs["agent_nu"],
        outputs["agent_alpha"],
        outputs["agent_beta"],
    )
    ad_reg = nig_reg_loss(
        targets["agent_delta"],
        outputs["agent_gamma"],
        outputs["agent_nu"],
        outputs["agent_alpha"],
    )
    losses["agent"] = (ad_nll + ad_reg * evidence_annealing * 0.01) * agent_weight

    # --- Beta: keys ---
    keys_nll = beta_nll_loss(
        outputs["keys_alpha1"],
        outputs["keys_alpha2"],
        targets["keys_collected"],
    )
    losses["keys"] = keys_nll * keys_weight

    # --- NIG: delta_px ---
    dp_nll = nig_nll_loss(
        targets["goal_delta_px"].unsqueeze(-1),
        outputs["delta_px_gamma"],
        outputs["delta_px_nu"],
        outputs["delta_px_alpha"],
        outputs["delta_px_beta"],
    )
    dp_reg = nig_reg_loss(
        targets["goal_delta_px"].unsqueeze(-1),
        outputs["delta_px_gamma"],
        outputs["delta_px_nu"],
        outputs["delta_px_alpha"],
    )
    losses["delta_px"] = (dp_nll + dp_reg * evidence_annealing * 0.01) * delta_px_weight

    losses["total"] = sum(losses.values())
    return losses


@dataclass
class JointConfig:
    """Configuration for joint training (REX 3.1)."""

    # Shared
    latent_dim: int = 192
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    # Encoder (matching LeWM paper spec adapted for ARC)
    encoder_image_size: int = 64
    encoder_in_channels: int = 16
    encoder_patch_size: int = 16
    encoder_num_layers: int = 12
    encoder_num_heads: int = 3
    encoder_hidden_dim: int = 192

    # Predictor (LeWM)
    predictor_num_layers: int = 6
    predictor_num_heads: int = 16
    predictor_dropout: float = 0.1
    predictor_action_dim: int = 192
    predictor_goal_dim: int = 192  # z_goal matches encoder latent dim

    # SIGReg (LeWM)
    sigreg_num_projections: int = 5_000
    reg_lambda: float = 0.1

    # Goal inference
    goal_pool: str = "mean"

    # Training
    num_epochs: int = 200
    lewm_batch_size: int = 32
    goal_batch_size: int = 32
    lewm_sequence_len: int = 8
    goal_context_len: int = 3
    lewm_samples_per_game: int = 3
    goal_samples_per_game: int = 10

    lewm_lr: float = 1e-3
    goal_lr: float = 3e-4
    goal_reconstructor_lr: float = 3e-4
    weight_decay: float = 1e-5

    # Oscillation schedule
    oscillation: str = "weighted"
    lewm_first_epochs: int = 0

    # Loss weights
    goal_agent_weight: float = 1.0
    goal_goal_type_weight: float = 0.5
    goal_keys_weight: float = 0.05  # Beta NLL is ~10-100× BCE
    goal_delta_px_weight: float = 0.1

    # Goal-match weight (REX 3.0 style MSE between ẑ_H and z_goal)
    goal_match_weight: float = 1.0

    # Pretrained encoder
    pretrained_encoder_path: str | None = None
    freeze_encoder_init: bool = False

    # Logging
    log_every: int = 5
    eval_every: int = 10

    def validate(self) -> None:
        if self.lewm_sequence_len < 2:
            raise ValueError("lewm_sequence_len must be >= 2")
        if self.goal_context_len < 1:
            raise ValueError("goal_context_len must be >= 1")
        assert self.oscillation in ("alternate", "weighted", "two_phase")


class JointTrainer:
    """Joint LeWM + Goal Trainer with evidential heads and belief state.

    REX 3.1 architecture:
    - Goal model outputs evidential distributions (Dirichlet, NIG, Beta)
    - BeliefState maintains running Gaussian beliefs over properties
    - Reconstructor is deterministic from MAP beliefs
    - LeWM prediction mismatch corrects belief state
    - Confidence at the property belief level, not reconstructor
    """

    def __init__(self, config: JointConfig | None = None) -> None:
        self.config = config or JointConfig()
        self.config.validate()
        cfg = self.config

        torch.manual_seed(cfg.seed)
        self.device = torch.device(cfg.device)

        # --- Build shared encoder ---
        encoder_config = EncoderConfig(
            image_size=cfg.encoder_image_size,
            in_channels=cfg.encoder_in_channels,
            patch_size=cfg.encoder_patch_size,
            num_layers=cfg.encoder_num_layers,
            num_heads=cfg.encoder_num_heads,
            hidden_dim=cfg.encoder_hidden_dim,
            latent_dim=cfg.latent_dim,
        )

        # --- LeWM ---
        lewm_config = LeWMConfig(encoder=encoder_config)
        lewm_config.predictor = PredictorConfig(
            latent_dim=cfg.latent_dim,
            action_dim=cfg.latent_dim,
            goal_dim=cfg.predictor_goal_dim,
            num_layers=cfg.predictor_num_layers,
            num_heads=cfg.predictor_num_heads,
            dropout=cfg.predictor_dropout,
        )

        self.lewm = LeWM(lewm_config)
        self.sigreg = SIGReg(
            config=SIGRegConfig(
                latent_dim=cfg.latent_dim,
                num_projections=cfg.sigreg_num_projections,
                seed=cfg.seed,
            ),
        ).to(self.device)
        self.lewm = self.lewm.to(self.device)

        # --- Goal inference model (evidential heads, shares encoder) ---
        self.goal_model = LeWMGoalInferenceModel(
            encoder_config=encoder_config,
            hidden_dim=cfg.latent_dim,
            pool=cfg.goal_pool,
        )
        self.goal_model.encoder = self.lewm.encoder
        self.goal_model = self.goal_model.to(self.device)

        # --- Goal latent reconstructor (DETERMINISTIC from MAP) ---
        self.goal_reconstructor = GoalLatentReconstructor(
            GoalLatentConfig(latent_dim=cfg.latent_dim),
        ).to(self.device)

        # --- Belief state (Kalman + error correction) ---
        self.belief_state = BeliefState(n_properties=N_GOAL_PROPERTIES).to(self.device)

        # --- Load pretrained encoder ---
        if cfg.pretrained_encoder_path:
            self.goal_model.load_lewm_encoder(cfg.pretrained_encoder_path)
            if cfg.freeze_encoder_init:
                for p in self.lewm.encoder.parameters():
                    p.requires_grad = False

        # --- Optimizers ---
        self.lewm_optimizer = torch.optim.AdamW(
            [
                {"params": self.lewm.encoder.parameters(), "lr": cfg.lewm_lr},
                {"params": self.lewm.predictor.parameters(), "lr": cfg.lewm_lr},
            ],
            weight_decay=cfg.weight_decay,
        )

        # Goal property heads (evidential heads)
        self.goal_optimizer = torch.optim.AdamW(
            [
                {"params": self.goal_model.agent_head.parameters(), "lr": cfg.goal_lr},
                {"params": self.goal_model.goal_type_head.parameters(), "lr": cfg.goal_lr},
                {"params": self.goal_model.keys_head.parameters(), "lr": cfg.goal_lr},
                {"params": self.goal_model.delta_px_head.parameters(), "lr": cfg.goal_lr},
            ],
            weight_decay=cfg.weight_decay,
        )

        # Goal latent reconstructor
        self.reconstructor_optimizer = torch.optim.AdamW(
            self.goal_reconstructor.parameters(),
            lr=cfg.goal_reconstructor_lr,
            weight_decay=cfg.weight_decay,
        )

        # Belief state error gate optimizer
        self.belief_optimizer = torch.optim.AdamW(
            self.belief_state.error_gate.parameters(),
            lr=1e-4,
            weight_decay=cfg.weight_decay,
        )

        # --- History ---
        self.history: list[dict[str, float]] = []
        self.best_type_acc = 0.0
        self.best_epoch = -1
        self.best_state = None

    # ── Helper to build full 23-dim MAP from outputs ─────────────────────

    @staticmethod
    def _build_properties_map(goal_out: dict[str, torch.Tensor]) -> torch.Tensor:
        """Build 23-dim MAP property vector from goal model outputs.

        Order: [goal_type (4), agent_delta (2), keys (16), delta_px (1)]
        """
        # goal_type: Dirichlet mean
        alpha = goal_out["goal_type_alpha"]
        goal_type_map = alpha / alpha.sum(dim=-1, keepdim=True)  # [B, 4]

        # agent_delta: NIG gamma (MAP)
        agent_map = goal_out["agent_gamma"]  # [B, 2]

        # keys: Beta mean
        keys_map = goal_out["keys_alpha1"] / (
            goal_out["keys_alpha1"] + goal_out["keys_alpha2"] + 1e-8
        )  # [B, 16]

        # delta_px: NIG gamma
        delta_px_map = goal_out["delta_px_gamma"]  # [B, 1]

        return torch.cat([goal_type_map, agent_map, keys_map, delta_px_map], dim=-1)

    # ── Training steps ──────────────────────────────────────────────────

    def lewm_step(
        self,
        obs: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> dict[str, float]:
        """Standard LeWM step (no goal conditioning)."""
        self.lewm.train()
        outputs = self.lewm(obs, action_emb)
        loss = lewm_loss(
            outputs["emb"], outputs["next_emb"],
            self.sigreg, reg_lambda=self.config.reg_lambda,
        )

        self.lewm_optimizer.zero_grad()
        loss.total.backward()
        torch.nn.utils.clip_grad_norm_(self.lewm.parameters(), 1.0)
        self.lewm_optimizer.step()

        return {
            "lewm_pred": float(loss.pred.detach()),
            "lewm_sigreg": float(loss.sigreg.detach()),
            "lewm_total": float(loss.total.detach()),
        }

    def goal_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Goal inference training step with evidential losses + belief update.

        1. Forward through goal model → evidential parameters
        2. Evidential loss (NLL + regularization)
        3. Update belief state from observation
        4. Cache MAP properties for reconstructor
        """
        device = self.device
        cf, ca = batch["context_frames"].to(device), batch["context_actions"].to(device)
        targets = {
            "agent_delta": batch["agent_delta"].to(device),
            "goal_type": batch["goal_type"].to(device),
            "keys_collected": batch["keys_collected"].to(device),
            "goal_delta_px": batch["goal_delta_px"].to(device),
        }

        self.goal_model.train()
        B = cf.shape[0]

        # Forward through goal model → evidential parameters
        outputs = self.goal_model(cf, ca)

        # Evidential loss
        losses = evidential_goal_loss(
            outputs, targets,
            goal_type_weight=self.config.goal_goal_type_weight,
            agent_weight=self.config.goal_agent_weight,
            keys_weight=self.config.goal_keys_weight,
            delta_px_weight=self.config.goal_delta_px_weight,
        )

        # Update property heads
        params_to_update = (
            list(self.goal_model.agent_head.parameters())
            + list(self.goal_model.goal_type_head.parameters())
            + list(self.goal_model.keys_head.parameters())
            + list(self.goal_model.delta_px_head.parameters())
        )
        if not self.config.freeze_encoder_init:
            params_to_update += list(self.goal_model.encoder.parameters())

        self.goal_optimizer.zero_grad()
        losses["total"].backward(retain_graph=True)
        goal_params = [p for p in params_to_update if p.requires_grad]
        if goal_params:
            torch.nn.utils.clip_grad_norm_(goal_params, 1.0)
        self.goal_optimizer.step()

        # Update belief state from observation (Kalman step)
        if self.belief_state.mu is None or self.belief_state.mu.shape[0] != B:
            self.belief_state.reset(B, self.device)
        self.belief_state.update_from_observation(outputs)

        # Compute confidence from belief state (mean precision across properties)
        conf = self.belief_state.confidence
        mean_conf = conf.mean().item() if conf is not None else 0.0

        return {
            "goal_total": float(losses["total"].detach()),
            "goal_goal_type": float(losses["goal_type"].detach()),
            "goal_agent": float(losses["agent"].detach()),
            "goal_keys": float(losses["keys"].detach()),
            "goal_delta_px": float(losses["delta_px"].detach()),
            "belief_conf": mean_conf,
        }

    def weighted_step(
        self,
        lewm_batch: dict[str, torch.Tensor],
        goal_batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Combined step: evidential goal loss + LeWM + belief correction.

        This is the full REX 3.1 loop:
        1. Goal model → evidential outputs
        2. Evidential loss (NLL + regularization)
        3. Belief state: update from observation
        4. Reconstructor: MAP → z_goal (deterministic, no confidence gating)
        5. LeWM forward: predicts ẑ_H
        6. Goal-match loss: MSE(ẑ_H, z_goal) — gradient flows through everything
        7. Belief state: error correction from prediction mismatch
        """
        device = self.device
        self.lewm.train()
        self.goal_model.train()

        obs = lewm_batch["obs"].to(device)
        action_emb = lewm_batch["action_emb"].to(device)

        cf = goal_batch["context_frames"].to(device)
        ca = goal_batch["context_actions"].to(device)
        targets = {
            "agent_delta": goal_batch["agent_delta"].to(device),
            "goal_type": goal_batch["goal_type"].to(device),
            "keys_collected": goal_batch["keys_collected"].to(device),
            "goal_delta_px": goal_batch["goal_delta_px"].to(device),
        }

        # 1. Goal model forward
        goal_out = self.goal_model(cf, ca)

        # 2. Evidential loss
        goal_losses = evidential_goal_loss(
            goal_out, targets,
            goal_type_weight=self.config.goal_goal_type_weight,
            agent_weight=self.config.goal_agent_weight,
            keys_weight=self.config.goal_keys_weight,
            delta_px_weight=self.config.goal_delta_px_weight,
        )

        # 3. Belief state: Kalman update from observation
        B = cf.shape[0]
        if self.belief_state.mu is None or self.belief_state.mu.shape[0] != B:
            self.belief_state.reset(B, self.device)
        self.belief_state.update_from_observation(goal_out)

        # 4. Reconstructor: MAP → z_goal (DETERMINISTIC, no confidence)
        props_map = self._build_properties_map(goal_out)
        z_goal = self.goal_reconstructor(props_map)

        # 5. LeWM forward
        lewm_out = self.lewm.forward_with_goal(
            obs, action_emb, goal_emb=z_goal,
        )
        lewm_l = lewm_loss(
            lewm_out["emb"], lewm_out["next_emb"],
            self.sigreg, reg_lambda=self.config.reg_lambda,
        )

        # 6. Goal-match loss: MSE(ẑ_H, z_goal)
        z_hat_H = lewm_out["next_emb"][:, -1, :]  # [B, D]
        goal_match_loss = F.mse_loss(z_hat_H, z_goal)
        # The prediction error PER BATCH ITEM for belief correction
        batch_pred_errors = (z_hat_H - z_goal).norm(dim=-1)  # [B]

        # 7. Combined loss: evidential + lewm + goal-match
        total_loss = (
            lewm_l.total
            + self.config.goal_match_weight * goal_match_loss
            + goal_losses["total"]
        )

        # Build all trainable params
        all_params = (
            list(self.lewm.encoder.parameters())
            + list(self.lewm.predictor.parameters())
            + list(self.goal_model.agent_head.parameters())
            + list(self.goal_model.goal_type_head.parameters())
            + list(self.goal_model.keys_head.parameters())
            + list(self.goal_model.delta_px_head.parameters())
            + list(self.goal_reconstructor.parameters())
            + list(self.belief_state.error_gate.parameters())
        )

        optim = torch.optim.AdamW(
            [p for p in all_params if p.requires_grad],
            lr=self.config.lewm_lr,
            weight_decay=self.config.weight_decay,
        )

        optim.zero_grad()
        total_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(
            [p for p in all_params if p.requires_grad], 1.0,
        )
        optim.step()

        # 7b. Belief state: error correction from LeWM prediction mismatch
        # This runs AFTER the backward pass — it's a forward correction
        # that will be felt in the NEXT iteration's belief state.
        with torch.no_grad():
            self.belief_state.correct_from_error(batch_pred_errors)

        # Confidence from belief state
        conf = self.belief_state.confidence
        mean_conf = conf.mean().item() if conf is not None else 0.0

        return {
            "lewm_pred": float(lewm_l.pred.detach()),
            "lewm_sigreg": float(lewm_l.sigreg.detach()),
            "lewm_total": float(lewm_l.total.detach()),
            "goal_match": float(goal_match_loss.detach()),
            "goal_total": float(goal_losses["total"].detach()),
            "goal_goal_type": float(goal_losses["goal_type"].detach()),
            "goal_agent": float(goal_losses["agent"].detach()),
            "goal_keys": float(goal_losses["keys"].detach()),
            "goal_delta_px": float(goal_losses["delta_px"].detach()),
            "belief_conf": mean_conf,
        }

    def evaluate_goal(
        self,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
    ) -> dict[str, float]:
        return evaluate_property_model(self.goal_model, test_loader, device)

    def state_dict(self) -> dict:
        return {
            "lewm": self.lewm.state_dict(),
            "goal_model": self.goal_model.state_dict(),
            "goal_reconstructor": self.goal_reconstructor.state_dict(),
            "sigreg": self.sigreg.state_dict(),
            "belief_state": self.belief_state.get_state_dict(),
            "lewm_optimizer": self.lewm_optimizer.state_dict(),
            "best_type_acc": self.best_type_acc,
            "best_epoch": self.best_epoch,
            "best_state": self.best_state,
        }


def build_goal_dataloader(
    pattern_dataset_path: str | Path,
    goals_dataset_path: str | Path,
    split: str,
    context_len: int,
    samples_per_game: int,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
    _preloaded: tuple | None = None,
) -> torch.utils.data.DataLoader:
    """Build a goal-property DataLoader."""
    dataset = GoalPropertyDataset(
        pattern_dataset_path, goals_dataset_path,
        split=split, context_len=context_len,
        samples_per_game=samples_per_game, seed=seed,
        _preloaded=_preloaded,
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, collate_fn=collate_property, drop_last=shuffle,
    )


__all__ = [
    "JointConfig",
    "JointTrainer",
    "evidential_goal_loss",
    "build_goal_dataloader",
]
