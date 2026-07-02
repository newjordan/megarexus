"""Latent action-effect basis diagnostics for LeWM predictors.

This module is an adapted-paper-grounded extension: the paper trains a
JEPA-style predictor ``p(z_t, a_t) -> z_{t+1}``; this diagnostic asks a local
question about that learned transition map. Given a state history and candidate
action embeddings, it estimates predicted latent displacements and selects a
small orthogonal action basis for exploration or planner proposal generation.

It does not change the LeWM training objective and does not use rewards,
scorecards, goal labels, or environment cloning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


PredictorFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class LatentActionBasisReport:
    """Summary of local action controllability around one latent history.

    Shapes:
      effects: ``[B, N, D]`` predicted displacement for each candidate action.
      effect_norms: ``[B, N]`` L2 norm of each displacement.
      mean_effects: ``[N, D]`` batch-mean effect vectors.
      cosine_matrix: ``[N, N]`` cosine similarity among mean effects.
      selected_indices: greedy orthogonal action basis indices.
    """

    effects: torch.Tensor
    effect_norms: torch.Tensor
    mean_effects: torch.Tensor
    cosine_matrix: torch.Tensor
    selected_indices: tuple[int, ...]
    singular_values: torch.Tensor
    effective_rank: float
    target_step: int


def _validate_z_history(z_history: torch.Tensor) -> tuple[int, int, int]:
    if z_history.dim() != 3:
        raise ValueError(f"z_history must be [B, T, D], got {tuple(z_history.shape)}")
    batch, steps, dim = z_history.shape
    if batch < 1 or steps < 1 or dim < 1:
        raise ValueError("z_history must have non-empty B, T, and D dimensions")
    if not torch.is_floating_point(z_history):
        raise TypeError("z_history must be floating point")
    if not torch.isfinite(z_history).all():
        raise ValueError("z_history contains non-finite values")
    return int(batch), int(steps), int(dim)


def _candidate_actions_for_batch(
    candidate_actions: torch.Tensor,
    *,
    batch: int,
) -> torch.Tensor:
    if not torch.is_floating_point(candidate_actions):
        raise TypeError("candidate_actions must be floating point")
    if not torch.isfinite(candidate_actions).all():
        raise ValueError("candidate_actions contains non-finite values")
    if candidate_actions.dim() == 2:
        if candidate_actions.shape[0] < 1:
            raise ValueError("candidate_actions must include at least one action")
        return candidate_actions.unsqueeze(0).expand(batch, -1, -1)
    if candidate_actions.dim() == 3:
        if candidate_actions.shape[0] != batch:
            raise ValueError(
                f"candidate_actions batch {candidate_actions.shape[0]} != z_history batch {batch}"
            )
        if candidate_actions.shape[1] < 1:
            raise ValueError("candidate_actions must include at least one action")
        return candidate_actions
    raise ValueError(
        f"candidate_actions must be [N, A] or [B, N, A], got {tuple(candidate_actions.shape)}"
    )


def _base_action_sequence(
    *,
    batch: int,
    steps: int,
    action_dim: int,
    z_history: torch.Tensor,
    history_action_emb: torch.Tensor | None,
) -> torch.Tensor:
    if history_action_emb is None:
        return z_history.new_zeros((batch, steps, action_dim))
    if history_action_emb.shape != (batch, steps, action_dim):
        raise ValueError(
            "history_action_emb must match [B, T, A], "
            f"expected {(batch, steps, action_dim)}, got {tuple(history_action_emb.shape)}"
        )
    if not torch.is_floating_point(history_action_emb):
        raise TypeError("history_action_emb must be floating point")
    if not torch.isfinite(history_action_emb).all():
        raise ValueError("history_action_emb contains non-finite values")
    return history_action_emb.to(device=z_history.device, dtype=z_history.dtype)


def predict_candidate_effects(
    predictor: PredictorFn,
    z_history: torch.Tensor,
    candidate_actions: torch.Tensor,
    *,
    history_action_emb: torch.Tensor | None = None,
    target_step: int = -1,
    baseline_action: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return predicted latent displacements for candidate actions.

    The predictor is called on a baseline action sequence and on one sequence
    per candidate, where only ``target_step`` is replaced by the candidate
    action. Returned effects are ``candidate_pred - baseline_pred`` at the
    target step, shaped ``[B, N, D]``.
    """

    batch, steps, latent_dim = _validate_z_history(z_history)
    actions = _candidate_actions_for_batch(candidate_actions, batch=batch).to(
        device=z_history.device,
        dtype=z_history.dtype,
    )
    _, num_actions, action_dim = actions.shape
    if target_step < 0:
        target_step = steps + target_step
    if target_step < 0 or target_step >= steps:
        raise ValueError(f"target_step {target_step} out of range for T={steps}")

    base_actions = _base_action_sequence(
        batch=batch,
        steps=steps,
        action_dim=action_dim,
        z_history=z_history,
        history_action_emb=history_action_emb,
    )
    if baseline_action is not None:
        if not torch.is_floating_point(baseline_action):
            raise TypeError("baseline_action must be floating point")
        if not torch.isfinite(baseline_action).all():
            raise ValueError("baseline_action contains non-finite values")
        if baseline_action.shape != (action_dim,):
            raise ValueError(
                f"baseline_action must be [A={action_dim}], got {tuple(baseline_action.shape)}"
            )
        base_actions = base_actions.clone()
        base_actions[:, target_step, :] = baseline_action.to(
            device=z_history.device,
            dtype=z_history.dtype,
        )

    with torch.no_grad():
        baseline_pred = predictor(z_history, base_actions)
        if baseline_pred.shape != (batch, steps, latent_dim):
            raise ValueError(
                "predictor baseline output must be [B, T, D], "
                f"expected {(batch, steps, latent_dim)}, got {tuple(baseline_pred.shape)}"
            )

        z_rep = z_history.unsqueeze(1).expand(batch, num_actions, steps, latent_dim)
        z_rep = z_rep.reshape(batch * num_actions, steps, latent_dim)
        a_rep = base_actions.unsqueeze(1).expand(batch, num_actions, steps, action_dim).clone()
        a_rep[:, :, target_step, :] = actions
        a_rep = a_rep.reshape(batch * num_actions, steps, action_dim)
        candidate_pred = predictor(z_rep, a_rep)
        if candidate_pred.shape != (batch * num_actions, steps, latent_dim):
            raise ValueError(
                "predictor candidate output must be [B*N, T, D], "
                f"expected {(batch * num_actions, steps, latent_dim)}, "
                f"got {tuple(candidate_pred.shape)}"
            )

    candidate_step = candidate_pred.reshape(
        batch,
        num_actions,
        steps,
        latent_dim,
    )[:, :, target_step, :]
    baseline_step = baseline_pred[:, target_step, :].unsqueeze(1)
    return candidate_step - baseline_step


def action_effect_cosine_matrix(effects: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Return cosine similarities among action-effect vectors.

    ``effects`` may be ``[N, D]`` or ``[B, N, D]``. Batched effects are averaged
    before cosine computation.
    """

    if effects.dim() == 3:
        vectors = effects.mean(dim=0)
    elif effects.dim() == 2:
        vectors = effects
    else:
        raise ValueError(f"effects must be [N, D] or [B, N, D], got {tuple(effects.shape)}")
    if vectors.shape[0] < 1 or vectors.shape[1] < 1:
        raise ValueError("effects must have non-empty N and D dimensions")
    vectors = F.normalize(vectors.float(), dim=-1, eps=eps)
    return vectors @ vectors.T


def latent_effective_rank(effects: torch.Tensor, *, eps: float = 1e-12) -> tuple[torch.Tensor, float]:
    """Return singular values and entropy effective rank for mean effects."""

    if effects.dim() == 3:
        matrix = effects.mean(dim=0).float()
    elif effects.dim() == 2:
        matrix = effects.float()
    else:
        raise ValueError(f"effects must be [N, D] or [B, N, D], got {tuple(effects.shape)}")
    singular_values = torch.linalg.svdvals(matrix)
    total = singular_values.sum()
    if float(total) <= eps:
        return singular_values, 0.0
    probs = singular_values / total.clamp_min(eps)
    entropy = -(probs * probs.clamp_min(eps).log()).sum()
    return singular_values, float(entropy.exp().item())


def select_orthogonal_action_basis(
    effects: torch.Tensor,
    *,
    max_actions: int,
    min_effect_norm: float = 0.0,
    eps: float = 1e-8,
) -> tuple[int, ...]:
    """Greedily select high-norm action effects with orthogonal residuals."""

    if max_actions < 1:
        raise ValueError("max_actions must be >= 1")
    if min_effect_norm < 0.0:
        raise ValueError("min_effect_norm must be >= 0")
    if effects.dim() == 3:
        vectors = effects.mean(dim=0).float()
    elif effects.dim() == 2:
        vectors = effects.float()
    else:
        raise ValueError(f"effects must be [N, D] or [B, N, D], got {tuple(effects.shape)}")
    num_actions = int(vectors.shape[0])
    if num_actions < 1:
        raise ValueError("effects must include at least one action")

    selected: list[int] = []
    basis: list[torch.Tensor] = []
    remaining = set(range(num_actions))
    norms = vectors.norm(dim=-1)
    while remaining and len(selected) < max_actions:
        best_idx = None
        best_score = float("-inf")
        best_residual = None
        for idx in sorted(remaining):
            vector = vectors[idx]
            residual = vector.clone()
            for unit in basis:
                residual = residual - torch.dot(residual, unit) * unit
            residual_norm = float(residual.norm().item())
            if float(norms[idx].item()) < min_effect_norm:
                score = float("-inf")
            else:
                score = residual_norm
            if score > best_score:
                best_score = score
                best_idx = idx
                best_residual = residual
        if best_idx is None or best_score <= eps or best_residual is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)
        basis.append(best_residual / best_residual.norm().clamp_min(eps))
    return tuple(selected)


def latent_action_basis_report(
    predictor: PredictorFn,
    z_history: torch.Tensor,
    candidate_actions: torch.Tensor,
    *,
    history_action_emb: torch.Tensor | None = None,
    target_step: int = -1,
    max_basis_actions: int = 4,
    min_effect_norm: float = 0.0,
) -> LatentActionBasisReport:
    """Compute predicted effects and a compact local action basis report."""

    effects = predict_candidate_effects(
        predictor,
        z_history,
        candidate_actions,
        history_action_emb=history_action_emb,
        target_step=target_step,
    )
    _, steps, _ = _validate_z_history(z_history)
    resolved_step = target_step if target_step >= 0 else steps + target_step
    mean_effects = effects.mean(dim=0)
    singular_values, rank = latent_effective_rank(mean_effects)
    return LatentActionBasisReport(
        effects=effects,
        effect_norms=effects.norm(dim=-1),
        mean_effects=mean_effects,
        cosine_matrix=action_effect_cosine_matrix(mean_effects),
        selected_indices=select_orthogonal_action_basis(
            mean_effects,
            max_actions=max_basis_actions,
            min_effect_norm=min_effect_norm,
        ),
        singular_values=singular_values,
        effective_rank=rank,
        target_step=resolved_step,
    )


__all__ = [
    "LatentActionBasisReport",
    "action_effect_cosine_matrix",
    "latent_action_basis_report",
    "latent_effective_rank",
    "predict_candidate_effects",
    "select_orthogonal_action_basis",
]
