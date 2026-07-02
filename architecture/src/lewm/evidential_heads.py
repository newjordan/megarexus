"""Evidential (distributional) goal property prediction heads.

Replaces point-estimate heads with uncertainty-aware distributions:

- goal_type (categorical): Dirichlet distribution — outputs concentration α_k > 0
- agent_delta, delta_px (regression): Normal-Inverse-Gamma — (γ, ν, α, β)
- keys (multi-label): Independent Beta distributions per color

Reference: Deep Evidential Regression (Amini et al. 2020, NeurIPS)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.goal_property import ARC_NUM_COLORS, NUM_GOAL_TYPES


# ---------------------------------------------------------------------------
# Dirichlet head for categorical goal_type
# ---------------------------------------------------------------------------

class DirichletHead(nn.Module):
    """Predict Dirichlet concentration parameters for K-class classification.

    Output: α_k > 0 (concentration for each class)
    Prediction (MAP): argmax_k(α_k - 1) = argmax_k(logits)
    Uncertainty: α₀ = Σα_k — total evidence (lower = more uncertain)
    """

    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Linear(in_dim, num_classes)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns dict with 'alpha' [B, K] and 'logits' [B, K]."""
        logits = self.net(h)
        alpha = F.softplus(logits) + 1.0  # α > 1 always
        return {"alpha": alpha, "logits": logits}

    @staticmethod
    def mean(alpha: torch.Tensor) -> torch.Tensor:
        """Expected probability for each class."""
        return alpha / alpha.sum(dim=-1, keepdim=True)

    @staticmethod
    def uncertainty(alpha: torch.Tensor) -> torch.Tensor:
        """Total evidence: lower = more uncertain."""
        return alpha.sum(dim=-1)  # [B]

    @staticmethod
    def predictive_entropy(alpha: torch.Tensor) -> torch.Tensor:
        """Entropy of the expected categorical distribution."""
        p = DirichletHead.mean(alpha)
        return -(p * torch.log(p.clamp(min=1e-8))).sum(dim=-1)


def dirichlet_nll_loss(
    alpha: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood for Dirichlet distribution.

    Args:
        alpha: [B, K] concentration parameters (from DirichletHead)
        target: [B] class indices

    Returns:
        scalar loss
    """
    alpha_0 = alpha.sum(dim=-1)  # [B]
    alpha_t = alpha.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [B]

    # NLL = ln Γ(α₀) - Σ ln Γ(α_k) - (α_t - 1) * ln(p_t) at MAP
    # But since we're training alpha directly, use the standard negative log
    # likelihood of the Dirichlet-multinomial / categorical:
    # L = Σ_k (α_k - 1) * (ln Γ - digamma trick)
    # Simpler: softmax cross-entropy on logits + KL regularizer

    # Standard cross-entropy on logits (works because expected p = α/α₀
    # and α = softplus(logits) + 1, so MAP = argmax(α - 1) = argmax(logits))
    logits = alpha.log()  # log(α) which preserves argmax
    return F.cross_entropy(logits, target)


def dirichlet_regularization(
    alpha: torch.Tensor,
    target: torch.Tensor,
    annealing_coef: float = 1.0,
) -> torch.Tensor:
    """KL divergence regularizer: push α toward 1 for non-target classes.

    KL[Dir(α̃) ‖ Dir(1)] where α̃ = α ⊙ (1 - y_onehot) + 1 ⊙ y_onehot
    (only regularize non-target evidence).

    This penalizes evidence for wrong classes, keeping the model
    appropriately uncertain when it doesn't know.
    """
    K = alpha.shape[-1]
    target_onehot = F.one_hot(target, num_classes=K).float()

    # α̃: keep α for target class, set to 1 for non-target
    alpha_tilde = target_onehot * alpha + (1 - target_onehot) * 1.0
    alpha_0_tilde = alpha_tilde.sum(dim=-1)

    # KL[Dir(α̃) ‖ Dir(1)] = ln Γ(α₀) - Σ ln Γ(α_k) + Σ(α_k - 1)[ψ(α_k) - ψ(α₀)]
    # where ψ = digamma function, Dir(1) = uniform (all α_k = 1)
    # Simplified: KL ≈ Σ_k (α_k - 1) * [ψ(α_k) - ψ(α₀)] for k != target

    # Numerically stable computation
    kl = (
        torch.lgamma(alpha_0_tilde)
        - torch.lgamma(alpha_tilde).sum(dim=-1)
        + ((alpha_tilde - 1) * torch.digamma(alpha_tilde)).sum(dim=-1)
        - (alpha_0_tilde - K) * torch.digamma(alpha_0_tilde)
    )

    return (kl * annealing_coef).mean()


# ---------------------------------------------------------------------------
# Normal-Inverse-Gamma head for regression properties
# ---------------------------------------------------------------------------

class NIGHead(nn.Module):
    """Normal-Inverse-Gamma evidential regression head.

    Outputs 4 parameters per target dimension:
        γ (mean), ν > 0 (precision of mean), α > 1 (shape), β > 0 (scale)

    MAP estimate: γ
    Aleatoric uncertainty: E[σ²] = β/(α-1)
    Epistemic uncertainty: Var[μ] = β/(ν(α-1))
    Total uncertainty: aleatoric + epistemic
    """

    def __init__(self, in_dim: int, out_dim: int = 1) -> None:
        super().__init__()
        self.out_dim = out_dim
        # Output: [γ, log(ν), log(α-1), log(β)]  — 4 parameters per dimension
        self.net = nn.Linear(in_dim, out_dim * 4)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns dict with evidential parameters.

        Args:
            h: [B, D] hidden representation

        Returns:
            gamma: [B, out_dim] — MAP estimate
            nu: [B, out_dim] — precision of mean (> 0)
            alpha: [B, out_dim] — shape parameter (> 1)
            beta: [B, out_dim] — scale parameter (> 0)
        """
        raw = self.net(h)  # [B, out_dim * 4]
        B = h.shape[0]

        gamma = raw[:, 0 * self.out_dim : 1 * self.out_dim]
        log_nu = raw[:, 1 * self.out_dim : 2 * self.out_dim]
        log_alpha_minus_1 = raw[:, 2 * self.out_dim : 3 * self.out_dim]
        log_beta = raw[:, 3 * self.out_dim : 4 * self.out_dim]

        nu = F.softplus(log_nu) + 1e-6
        alpha = F.softplus(log_alpha_minus_1) + 1.0 + 1e-6  # α > 1
        beta = F.softplus(log_beta) + 1e-6

        return {
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
        }


def nig_nll_loss(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood for Normal-Inverse-Gamma.

    Closed-form Student-t likelihood:
        L = ½ ln(π/ν) - α ln(Ω) + (α + ½) ln((y-γ)²ν + Ω)
          + ln Γ(α) - ln Γ(α + ½)

    where Ω = 2β(1 + ν)
    """
    omega = 2 * beta * (1 + nu)

    loss = (
        0.5 * torch.log(torch.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    return loss.mean()


def nig_reg_loss(
    y: torch.Tensor,
    gamma: torch.Tensor,
    nu: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Evidence regularization for NIG regression.

    Penalizes evidence on wrong predictions:
        L_reg = |y - γ| * (2ν + α)

    This encourages the model to reduce evidence (increase uncertainty)
    when predictions are wrong.
    """
    error = (y - gamma).abs()
    return (error * (2 * nu + alpha)).mean()


# ---------------------------------------------------------------------------
# Beta head for multi-label keys (independent binary per color)
# ---------------------------------------------------------------------------

class BetaHead(nn.Module):
    """Independent Beta distributions for multi-label classification.

    Each of K binary labels gets its own (α₁, α₂) pair.
    MAP: α₁ / (α₁ + α₂)
    Uncertainty: α₁ + α₂ (total evidence)
    """

    def __init__(self, in_dim: int, num_labels: int) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.net = nn.Linear(in_dim, num_labels * 2)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns dict with 'alpha1', 'alpha2' [B, K]."""
        raw = self.net(h)
        alpha1 = F.softplus(raw[:, :self.num_labels]) + 1.0
        alpha2 = F.softplus(raw[:, self.num_labels:]) + 1.0
        return {"alpha1": alpha1, "alpha2": alpha2}

    @staticmethod
    def mean(alpha1: torch.Tensor, alpha2: torch.Tensor) -> torch.Tensor:
        return alpha1 / (alpha1 + alpha2)

    @staticmethod
    def logits_from_alpha(alpha1: torch.Tensor, alpha2: torch.Tensor) -> torch.Tensor:
        """Convert (α₁, α₂) to log-odds for BCE compatibility."""
        return torch.log(alpha1 / alpha2 + 1e-8)


def beta_nll_loss(
    alpha1: torch.Tensor,
    alpha2: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood for independent Beta distributions.

    Args:
        alpha1, alpha2: [B, K] concentration parameters
        target: [B, K] binary (float)

    Returns:
        scalar loss
    """
    # Beta NLL: -ln Γ(α₁+α₂) + ln Γ(α₁) + ln Γ(α₂)
    #           - (α₁-1)*ln(y) - (α₂-1)*ln(1-y)
    K = alpha1.shape[-1]
    y = target.clamp(1e-6, 1 - 1e-6)

    nll = (
        -torch.lgamma(alpha1 + alpha2)
        + torch.lgamma(alpha1)
        + torch.lgamma(alpha2)
        - (alpha1 - 1) * torch.log(y)
        - (alpha2 - 1) * torch.log(1 - y)
    )

    return nll.mean()


__all__ = [
    "DirichletHead",
    "NIGHead",
    "BetaHead",
    "dirichlet_nll_loss",
    "dirichlet_regularization",
    "nig_nll_loss",
    "nig_reg_loss",
    "beta_nll_loss",
]
