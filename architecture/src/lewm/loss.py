"""LeWM training objective: only two loss terms (paper Section 3, eq 1-3).

Paper Algorithm 1 (page 6, pseudocode)::

    emb = encoder(obs)                      # (B, T, D), TRAINABLE
    next_emb = predictor(emb, actions)      # (B, T, D)
    pred_loss = F.mse_loss(emb[:, 1:], next_emb[:, :-1])   # NO STOP-GRAD
    sigreg_loss = mean(SIGReg(emb.transpose(0, 1)))        # step-wise
    return pred_loss + lambd * sigreg_loss

Paper page 6, top: *"We do not employ stop-gradient, exponential moving averages,
or additional stabilization heuristics. Gradients are propagated through all
components of the loss, and all parameters are optimized jointly end-to-end."*

The whole point of LeWM is that SIGReg substitutes for stop-gradient — making
the encoder receive prediction-loss gradient on **both** sides of the MSE
(via ``emb[:, 1:]`` AND via ``emb`` flowing into the predictor through
``emb[:, :-1]``). Adding a ``.detach()`` here would defeat the paper's
methodological contribution, so we never detach.

Default ``λ = 0.1`` (paper Section 3, also App. G ablations).

This module also defines the **augmented** four-term loss
``lewm_corrected_loss`` used for Gaussian Correction Depth calibration —
see ``docs/gaussian_correction_depth.md`` for the architectural motivation
and ``correction.py`` / ``correction_transformer.py`` for the ``G_ψ``
implementation. The augmented loss freezes the LeWM backbone and trains a
``GPsi`` correction stack so that ``J(base+G_ψ)`` improves on the calibration
domain without regressing on the source domain (the lab's promotion gate).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


PAPER_REG_LAMBDA = 0.1


@dataclass
class LossOutput:
    pred: torch.Tensor
    sigreg: torch.Tensor
    total: torch.Tensor


@dataclass
class CorrectedLossOutput:
    """Per-step components of the four-term GCD calibration loss.

    Fields:

    - ``pred_corr``: ``MSE(g[:, 1:], ĝ[:, :-1])`` — prediction loss on the
      ``G_ψ``-corrected sequence; equals base ``L_pred`` at init (when
      ``G_ψ`` is identity).
    - ``sigreg_g``: step-wise SIGReg statistic on the corrected encoder
      outputs ``g``; equals ``SIGReg(emb)`` at init.
    - ``proximal``: ``MSE(g, emb)`` — keeps the correction near identity on
      the calibration data; equals 0 at init.
    - ``retention``: ``MSE(G_ψ(B_0), B_0)`` over the source-domain replay
      anchor set ``B_0``; equals 0 at init or when ``replay_anchor_emb=None``.
    - ``total``: ``pred_corr + λ_g·sigreg_g + β·proximal + ρ·retention``.
    """

    pred_corr: torch.Tensor
    sigreg_g: torch.Tensor
    proximal: torch.Tensor
    retention: torch.Tensor
    total: torch.Tensor


def prediction_loss(
    emb: torch.Tensor,
    next_emb: torch.Tensor,
) -> torch.Tensor:
    """``L_pred = MSE(emb[:, 1:], next_emb[:, :-1])`` over (B, T, D) sequences.

    ``emb`` is the encoder's output across the trajectory; ``next_emb`` is the
    predictor's per-position next-latent predictions. Position ``t`` of
    ``next_emb`` is the prediction of ``emb[:, t+1]``. The loss compares
    matching positions for ``t ∈ [0, T-2]``. **No stop-gradient on either
    side** (paper page 6, top).
    """

    if emb.dim() != 3:
        raise ValueError(f"emb must be [B, T, D], got {tuple(emb.shape)}")
    if next_emb.shape != emb.shape:
        raise ValueError(
            f"emb and next_emb must have the same shape, got "
            f"{tuple(emb.shape)} vs {tuple(next_emb.shape)}"
        )
    if emb.shape[1] < 2:
        raise ValueError(
            f"prediction_loss needs T >= 2 to form transitions, got T={emb.shape[1]}"
        )
    return F.mse_loss(next_emb[:, :-1], emb[:, 1:])


def lewm_loss(
    emb: torch.Tensor,
    next_emb: torch.Tensor,
    sigreg_module: Any,
    *,
    reg_lambda: float = PAPER_REG_LAMBDA,
) -> LossOutput:
    """Full LeWM objective for a (B, T, D) sequence batch.

    ``sigreg_module`` must expose ``forward_sequence(z_btd)`` returning a
    scalar — the step-wise mean of per-T SIGReg values, per paper pseudocode
    ``mean(SIGReg(emb.transpose(0, 1)))``. Anti-collapse pressure is applied to
    the encoder's outputs only; ``next_emb`` is not regularized (the paper does
    not regularize the predictor's outputs separately).
    """

    pred = prediction_loss(emb, next_emb)
    reg = sigreg_module.forward_sequence(emb)
    total = pred + reg_lambda * reg
    return LossOutput(pred=pred, sigreg=reg, total=total)


def lewm_corrected_loss(
    emb: torch.Tensor,
    next_emb: torch.Tensor,
    gpsi: Any,
    sigreg_module: Any,
    *,
    reg_lambda: float = PAPER_REG_LAMBDA,
    prox_beta: float = 1e-2,
    ret_rho: float = 1e-2,
    replay_anchor_emb: torch.Tensor | None = None,
    cond: torch.Tensor | None = None,
    replay_cond: torch.Tensor | None = None,
) -> CorrectedLossOutput:
    """Four-term GCD calibration loss (paper-doc eq ``L_aug``).

    ::

        L_aug = L_pred_corr + λ_g · SIGReg(g) + β · L_prox + ρ · L_ret

    where:

    - ``g = G_ψ(emb, cond)`` and ``ĝ = G_ψ(next_emb, cond)`` (shared weights)
    - ``L_pred_corr = MSE(g[:, 1:], ĝ[:, :-1])`` (paper Algorithm 1 shape,
      now applied to corrected latents)
    - ``L_prox = MSE(g, emb)`` (proximal term — forces ``G_ψ`` to remain
      near identity on the calibration data; the doc's "principled correction
      vs arbitrary extra depth" distinction)
    - ``L_ret = MSE(G_ψ(B_0, replay_cond), B_0)`` over the source-domain
      replay anchor set ``B_0 = replay_anchor_emb``; if ``None``, this term
      is exactly 0 (no retention pressure)

    At init, ``G_ψ`` is the identity, so ``g = emb``, ``ĝ = next_emb``,
    ``L_prox = 0``, and ``L_ret = 0`` — meaning ``L_aug`` reduces to base
    LeWM ``L_pred + λ · SIGReg(emb)``. Calibration training only changes
    ``G_ψ`` parameters; the backbone stays frozen.

    Notes:

    - This function does NOT detach anything. Per the LeWM no-stop-grad
      convention, gradients flow through every term that has trainable
      parameters (i.e., ``G_ψ``). The backbone is frozen by the **caller**
      via ``model.requires_grad_(False)`` + ``torch.no_grad()`` around the
      backbone forward, NOT by detaching here.
    - ``replay_anchor_emb`` should be source-domain encoder outputs (not
      raw observations) cached once before training begins; per the doc's
      "Training and Calibration on Separate Datasets" section.
    """

    if emb.dim() != 3:
        raise ValueError(f"emb must be [B, T, D], got {tuple(emb.shape)}")
    if next_emb.shape != emb.shape:
        raise ValueError(
            f"next_emb shape {tuple(next_emb.shape)} must match emb "
            f"{tuple(emb.shape)}"
        )
    if emb.shape[1] < 2:
        raise ValueError(
            f"need T >= 2 for L_pred_corr, got T={emb.shape[1]}"
        )

    # Apply correction (shared weights via the same gpsi instance for both
    # encoder and predictor outputs — per doc "Placement" recommendation).
    if cond is not None:
        g = gpsi(emb, cond)
        g_hat = gpsi(next_emb, cond)
    else:
        g = gpsi(emb)
        g_hat = gpsi(next_emb)

    pred_corr = F.mse_loss(g_hat[:, :-1], g[:, 1:])
    sigreg_g = sigreg_module.forward_sequence(g)
    proximal = F.mse_loss(g, emb)

    if replay_anchor_emb is not None:
        if replay_anchor_emb.dim() != 3:
            raise ValueError(
                f"replay_anchor_emb must be [M, T_a, D], got "
                f"{tuple(replay_anchor_emb.shape)}"
            )
        if replay_cond is not None:
            rep_g = gpsi(replay_anchor_emb, replay_cond)
        else:
            rep_g = gpsi(replay_anchor_emb)
        retention = F.mse_loss(rep_g, replay_anchor_emb)
    else:
        retention = emb.new_zeros(())

    total = (
        pred_corr
        + reg_lambda * sigreg_g
        + prox_beta * proximal
        + ret_rho * retention
    )
    return CorrectedLossOutput(
        pred_corr=pred_corr,
        sigreg_g=sigreg_g,
        proximal=proximal,
        retention=retention,
        total=total,
    )


def lewm_paired_calibration_loss(
    emb: torch.Tensor,
    next_emb: torch.Tensor,
    target_y: torch.Tensor,
    gpsi: Any,
    sigreg_module: Any,
    *,
    reg_lambda: float = PAPER_REG_LAMBDA,
    prox_beta: float = 1e-2,
    ret_rho: float = 1e-2,
    replay_anchor_emb: torch.Tensor | None = None,
    cond: torch.Tensor | None = None,
    replay_cond: torch.Tensor | None = None,
) -> CorrectedLossOutput:
    """Paired-calibration loss for GCD (paper-doc, "Training and Calibration on
    Separate Datasets, paired" case).

    ::

        L_cal = L_dyn + λ_g · SIGReg(g) + β · L_prox + ρ · L_ret

    where:

    - ``L_dyn = MSE(target_y[:, 1:], ĝ_t[:, :-1])`` — the predictor's
      corrected output ``ĝ_t = G_ψ(ĥ_t)`` is matched to the calibration
      target's *next* latent ``y_{t+1}``. This is the doc's
      ``L_cal`` formulation (eq just before the linear-correction
      proposition).
    - ``L_prox = MSE(g, emb)`` — proximal pull toward identity on the
      calibration data. Same as in :func:`lewm_corrected_loss`.
    - ``L_ret = MSE(G_ψ(B_0), B_0)`` — retention on a separate source-domain
      replay anchor set. Optional via ``replay_anchor_emb``.

    Used by Phase C synthetic-shift experiments where ``target_y =
    shift(emb)`` for a known ``shift`` (rotation, translation, affine) — the
    test of whether ``G_ψ`` can learn the shift while respecting the proximal
    and retention regularizers.

    At init (``G_ψ`` is identity): ``g_hat == next_emb``, so
    ``L_dyn = MSE(target_y[:, 1:], next_emb[:, :-1])``. For a chassis-quality
    predictor (``next_emb[:, t] ≈ emb[:, t+1]``), this approximately equals
    ``MSE(target_y[:, 1:], emb[:, 1:]) ≈ ||shift||²`` — the shift magnitude.
    ``L_prox == 0`` and ``L_ret == 0`` at init.
    """

    if emb.dim() != 3:
        raise ValueError(f"emb must be [B, T, D], got {tuple(emb.shape)}")
    if next_emb.shape != emb.shape:
        raise ValueError(
            f"next_emb shape {tuple(next_emb.shape)} must match emb "
            f"{tuple(emb.shape)}"
        )
    if target_y.shape != emb.shape:
        raise ValueError(
            f"target_y shape {tuple(target_y.shape)} must match emb "
            f"{tuple(emb.shape)}"
        )
    if emb.shape[1] < 2:
        raise ValueError(f"need T >= 2 for L_dyn, got T={emb.shape[1]}")

    if cond is not None:
        g = gpsi(emb, cond)
        g_hat = gpsi(next_emb, cond)
    else:
        g = gpsi(emb)
        g_hat = gpsi(next_emb)

    pred_corr = F.mse_loss(g_hat[:, :-1], target_y[:, 1:])
    sigreg_g = sigreg_module.forward_sequence(g)
    proximal = F.mse_loss(g, emb)

    if replay_anchor_emb is not None:
        if replay_anchor_emb.dim() != 3:
            raise ValueError(
                f"replay_anchor_emb must be [M, T_a, D], got "
                f"{tuple(replay_anchor_emb.shape)}"
            )
        if replay_cond is not None:
            rep_g = gpsi(replay_anchor_emb, replay_cond)
        else:
            rep_g = gpsi(replay_anchor_emb)
        retention = F.mse_loss(rep_g, replay_anchor_emb)
    else:
        retention = emb.new_zeros(())

    total = (
        pred_corr
        + reg_lambda * sigreg_g
        + prox_beta * proximal
        + ret_rho * retention
    )
    return CorrectedLossOutput(
        pred_corr=pred_corr,
        sigreg_g=sigreg_g,
        proximal=proximal,
        retention=retention,
        total=total,
    )
