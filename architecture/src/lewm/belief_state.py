"""Belief state over goal properties with Kalman-style updates.

Architecture:
    Each goal property has a belief state: (μ, σ²) representing a Gaussian belief
    over the property's true value.

    Two update pathways:
    1. Observation update (forward): Goal model produces evidential parameters
       (γ, ν, α, β) from context frames → Bayesian conjugate update of belief.
       This is a Kalman filter step where the observation likelihood is NIG.

    2. Error-driven correction (backward): LeWM prediction mismatch
       e_t = MSE(ẑ_H, z_goal) signals that our goal guess was wrong.
       A learned gating network adjusts beliefs based on the prediction error.

    The reconstructor always gets the MAP estimate from current beliefs,
    with NO confidence gating. Confidence lives at the belief level.

Usage:
    belief = BeliefState(n_properties=23)  # 4 + 2 + 16 + 1
    belief.reset(B, device)

    # Forward step: incorporate goal model observation
    belief.update_from_observation(goal_output)

    # LeWM runs: reconstructor receives belief.mu (MAP)

    # Backward step: correct beliefs from prediction error
    belief.correct_from_error(prediction_error)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BeliefState(nn.Module):
    """Bayesian belief state over goal properties with Kalman + error correction.

    Each property is modeled as a Gaussian with learned mean and variance.
    Updates are performed via conjugate Bayesian inference (observation step)
    followed by a learned error-correction gate (prediction mismatch step).

    Properties (23 total):
        goal_type: 4 (Dirichlet → 4 means, one per class prob)
        agent_delta: 2 (Δx, Δy)
        keys: 16 (one per color, probability collected)
        delta_px: 1 (scalar)
    """

    def __init__(
        self,
        n_properties: int = 23,
        hidden_dim: int = 64,
        min_variance: float = 1e-4,
        max_variance: float = 100.0,
    ) -> None:
        super().__init__()
        self.n_properties = n_properties
        self.min_var = min_variance
        self.max_var = max_variance

        # Learned error-correction gate: maps prediction error + current
        # uncertainty → adjustment to mean and variance
        self.error_gate = nn.Sequential(
            nn.Linear(n_properties + 2, hidden_dim),  # uncert + error + error_smoothed
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_properties * 2),  # Δμ, Δlog(σ²)
        )

        # Running beliefs — stored externally, reset via reset()
        self._mu: torch.Tensor | None = None
        self._log_var: torch.Tensor | None = None

        # Smoothing for prediction error (low-pass filter)
        self.register_buffer("_smoothed_error", torch.zeros(1, 1))
        self._error_smoothing = 0.9  # EMA decay

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize error gate to near-zero output so it doesn't
        interfere with the conjugate Bayesian update early in training."""
        for m in self.error_gate.modules():
            if isinstance(m, nn.Linear):
                # Small weights: start with minimal correction
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def mu(self) -> torch.Tensor | None:
        return self._mu

    @property
    def log_var(self) -> torch.Tensor | None:
        return self._log_var

    @property
    def var(self) -> torch.Tensor | None:
        if self._log_var is None:
            return None
        return self._log_var.exp().clamp(self.min_var, self.max_var)

    @property
    def confidence(self) -> torch.Tensor | None:
        """Precision = 1/variance — confidence per property. [B, n_prop]"""
        if self._log_var is None:
            return None
        return 1.0 / self.var

    def reset(self, batch_size: int, device: torch.device) -> None:
        """Initialize with uninformative prior: μ=0, σ²=10."""
        self._mu = torch.zeros(batch_size, self.n_properties, device=device)
        self._log_var = torch.full(
            (batch_size, self.n_properties), fill_value=torch.log(torch.tensor(10.0)),
            device=device,
        )
        self._smoothed_error = torch.zeros(1, 1, device=device)

    def update_from_observation(
        self,
        goal_output: dict[str, torch.Tensor],
    ) -> None:
        """Bayesian conjugate update from evidential goal model output.

        The goal model produces evidential parameters per property.
        We convert these into a Kalman-style observation update.

        Args:
            goal_output: dict with keys from evidential heads.
                For NIG heads (agent_delta, delta_px): gamma, nu, alpha, beta
                For Dirichlet head (goal_type): alpha (concentrations)
                For Beta head (keys): alpha1, alpha2
        """
        if self._mu is None:
            return  # not initialized — skip

        # Build observation mean and variance per property
        obs_mu_list = []
        obs_var_list = []

        # goal_type (indices 0-3): Dirichlet → expected probabilities
        alpha = goal_output["goal_type_alpha"]  # [B, 4]
        p = alpha / alpha.sum(dim=-1, keepdim=True)  # expected probs
        # Variance of Beta marginal: α_k(α₀ - α_k) / (α₀²(α₀ + 1))
        alpha_0 = alpha.sum(dim=-1, keepdim=True)
        var_p = alpha * (alpha_0 - alpha) / (alpha_0 ** 2 * (alpha_0 + 1) + 1e-8)
        obs_mu_list.append(p)
        obs_var_list.append(var_p)

        # agent_delta (indices 4-5): NIG → γ is MAP
        gamma_ad = goal_output["agent_gamma"]  # [B, 2]
        nu_ad = goal_output["agent_nu"]
        alpha_ad = goal_output["agent_alpha"]
        beta_ad = goal_output["agent_beta"]
        # Aleatoric variance: β/(α-1)
        var_ad = beta_ad / (alpha_ad - 1).clamp(min=1e-6)
        # Effective observation variance (accounting for epistemic): β/(ν(α-1))
        obs_var_ad = var_ad / nu_ad.clamp(min=1e-6)
        obs_mu_list.append(gamma_ad)
        obs_var_list.append(obs_var_ad)

        # keys (indices 6-21): Beta → expected probability per color
        alpha1 = goal_output["keys_alpha1"]  # [B, 16]
        alpha2 = goal_output["keys_alpha2"]
        p_keys = alpha1 / (alpha1 + alpha2)
        # Beta variance: α₁α₂ / ((α₁+α₂)²(α₁+α₂+1))
        var_keys = alpha1 * alpha2 / ((alpha1 + alpha2) ** 2 * (alpha1 + alpha2 + 1) + 1e-8)
        obs_mu_list.append(p_keys)
        obs_var_list.append(var_keys)

        # delta_px (index 22): NIG
        gamma_dp = goal_output["delta_px_gamma"]  # [B, 1]
        nu_dp = goal_output["delta_px_nu"]
        alpha_dp = goal_output["delta_px_alpha"]
        beta_dp = goal_output["delta_px_beta"]
        var_dp = beta_dp / (alpha_dp - 1).clamp(min=1e-6)
        obs_var_dp = var_dp / nu_dp.clamp(min=1e-6)
        obs_mu_list.append(gamma_dp)
        obs_var_list.append(obs_var_dp)

        obs_mu = torch.cat(obs_mu_list, dim=-1)  # [B, 23]
        obs_var = torch.cat(obs_var_list, dim=-1)  # [B, 23]

        # Kalman gain: K = σ²_prior / (σ²_prior + σ²_obs)
        prior_var = self.var.clamp(self.min_var, self.max_var)
        kalman_gain = prior_var / (prior_var + obs_var.clamp(min=1e-6))

        # Posterior update
        mu_new = self._mu + kalman_gain * (obs_mu - self._mu)
        var_new = (1 - kalman_gain) * prior_var

        self._mu = mu_new
        self._log_var = torch.log(var_new.clamp(self.min_var, self.max_var))

    def correct_from_error(
        self,
        prediction_error: torch.Tensor,
    ) -> None:
        """Learned correction from LeWM prediction mismatch.

        The prediction error e_t = MSE(ẑ_H, z_goal) provides an indirect
        signal that our goal beliefs might be wrong. A learned gating
        network adjusts both μ and σ² based on the error magnitude and
        current uncertainty levels.

        Args:
            prediction_error: [B] scalar — MSE(ẑ_H, z_goal) per batch item
        """
        if self._mu is None or self._log_var is None:
            return

        B = self._mu.shape[0]

        # Smooth the error signal (low-pass filter)
        if self._smoothed_error.shape[0] != B:
            self._smoothed_error = torch.zeros(B, 1, device=self._mu.device)
        self._smoothed_error = (
            self._error_smoothing * self._smoothed_error
            + (1 - self._error_smoothing) * prediction_error.detach().unsqueeze(-1)
        )

        # Gate input: [current uncertainty + raw error + smoothed error]
        gate_input = torch.cat([
            self.var.detach(),                              # [B, n_prop] — how uncertain we are
            prediction_error.detach().unsqueeze(-1),         # [B, 1] — raw error
            self._smoothed_error.detach(),                   # [B, 1] — smoothed error
        ], dim=-1)

        correction = self.error_gate(gate_input)  # [B, n_prop * 2]

        # Split: Δμ and Δlog(σ²)
        delta_mu = correction[:, :self.n_properties]
        delta_log_var = correction[:, self.n_properties:]

        # Update: nudge mean toward correction, increase uncertainty proportional
        # to error (always increase, never decrease — error means we're less sure)
        self._mu = self._mu + delta_mu
        self._log_var = self._log_var + F.softplus(delta_log_var)

        # Clamp
        self._mu = self._mu.clamp(-5, 5)
        self._log_var = self._log_var.clamp(
            torch.log(torch.tensor(self.min_var, device=self._mu.device)),
            torch.log(torch.tensor(self.max_var, device=self._mu.device)),
        )

    def get_map(self) -> torch.Tensor:
        """Return MAP estimate from current belief. [B, n_prop]

        This is what the reconstructor receives — always 100% signal,
        no confidence gating. Confidence is queried separately via
        the confidence property.
        """
        return self._mu.clone()

    def get_state_dict(self) -> dict:
        """Serialize belief state for checkpointing."""
        return {
            "mu": self._mu,
            "log_var": self._log_var,
            "smoothed_error": self._smoothed_error,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._mu = state_dict["mu"]
        self._log_var = state_dict["log_var"]
        self._smoothed_error = state_dict.get("smoothed_error", torch.zeros(1, 1))


__all__ = ["BeliefState"]
