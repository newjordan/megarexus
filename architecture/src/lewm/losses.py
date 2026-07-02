"""LeWM-v2 losses — InfoNCE next-frame contrastive + helpers.

Companion to the paper-faithful :mod:`lewm.loss` module. That module
holds the original ``prediction_loss`` (MSE on predicted vs actual
next-frame latent) plus SIGReg, which is the canonical LeWM training
objective. This module adds the v2-proposal §3.4 alternative: an
InfoNCE contrastive objective that swaps "predict the *exact* next
latent" for "rank the actual next latent above K negatives."

Why a separate file: the paper-faithful module is locked at the paper's
spec for reproducibility. v2 modifications belong in a sibling module
that the trainer can opt into via a flag.

Existing related code in the repo (don't duplicate):
- ``src/lewm/loss.py``                    — paper-faithful MSE+SIGReg
- ``src/lewm/contrastive_supervision.py`` — hinge contrastive over trunk
  latents for hard-negative pair training (different signal)
- ``src/lewm/family_contrastive.py``     — capability-family pair mining
- ``src/loom_layers/lewm_clean/v1.py``    — uses
  ``model.action_embedder.contrastive_loss`` on action embeddings (also
  different signal — over actions, not next-frame latents)

Public API:
- :func:`info_nce_next_frame` — drop-in alternative to ``prediction_loss``
- :func:`info_nce_top1_accuracy` — eval metric for the §4.2 gate
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class InfoNCEOutput:
    """Result of one ``info_nce_next_frame`` call."""

    loss: torch.Tensor                          # scalar; mean InfoNCE
    top1_accuracy: torch.Tensor                 # scalar in [0, 1]; eval-style
    n_anchors: int                              # batch.size * (T-1)
    n_negatives_per_anchor: int                 # K for the log denominator


def info_nce_next_frame(
    emb: torch.Tensor,                          # [B, T, D]  encoder latents
    next_emb: torch.Tensor,                     # [B, T, D]  predictor outputs
    *,
    temperature: float = 0.1,
    normalize: bool = True,
) -> InfoNCEOutput:
    """InfoNCE on next-frame latents with in-batch negatives.

    For each anchor ``a = next_emb[b, t]`` (the predictor's t-th output),
    the positive is ``p = emb[b, t+1]`` (the encoder's actual (t+1)-th
    latent — the ground-truth target). Negatives are ``emb[b', t'+1]``
    for every other ``(b', t')`` pair in the batch — same number as
    anchors minus one.

    The loss is the standard symmetric InfoNCE in one direction
    (anchor → positive), normalized over the in-batch negative set::

        L = -log( exp(sim(a, p) / τ) / Σ_k exp(sim(a, k) / τ) )

    where ``sim`` is cosine (when ``normalize=True``) or dot product.

    Returns also ``top1_accuracy`` — the fraction of anchors whose
    actual positive is the closest match among all candidates. This is
    the metric the LeWM-v2 §4.2 gate measures (random chance is
    ``1 / (B·(T-1))`` which is tiny → gate threshold > 0.5 is meaningful).

    Note: This uses IN-BATCH negatives, not per-game negatives. Adding
    a ``game_id`` mask to restrict negatives to the same game is a
    natural follow-up; for the iter-4 smoke we keep the contract simple.
    """

    if emb.shape != next_emb.shape:
        raise ValueError(
            f"emb {tuple(emb.shape)} and next_emb {tuple(next_emb.shape)} "
            f"must have identical shape (B, T, D)"
        )
    if emb.dim() != 3:
        raise ValueError(f"emb must be 3D (B, T, D); got shape {tuple(emb.shape)}")
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    B, T, D = emb.shape
    if T < 2:
        raise ValueError(f"sequence_len must be >= 2; got T={T}")

    # Anchors: predictor outputs at positions 0..T-2 → predict positions 1..T-1.
    anchors = next_emb[:, :-1, :].reshape(B * (T - 1), D)  # [N, D]
    # Targets (positives): encoder latents at positions 1..T-1.
    positives = emb[:, 1:, :].reshape(B * (T - 1), D)      # [N, D]

    if normalize:
        anchors = F.normalize(anchors.float(), dim=-1)
        positives = F.normalize(positives.float(), dim=-1)
    else:
        anchors = anchors.float()
        positives = positives.float()

    # Similarity matrix [N, N]: row i = anchor i's sim to every candidate.
    # Candidates pool = ALL positives across the batch (including the matching
    # one). The diagonal is the positive; off-diagonal entries are negatives.
    logits = (anchors @ positives.t()) / temperature        # [N, N]
    targets = torch.arange(logits.size(0), device=logits.device)

    loss = F.cross_entropy(logits, targets)
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        top1_acc = (preds == targets).float().mean()

    return InfoNCEOutput(
        loss=loss,
        top1_accuracy=top1_acc,
        n_anchors=int(logits.size(0)),
        n_negatives_per_anchor=int(logits.size(1) - 1),
    )


def info_nce_top1_accuracy(
    emb: torch.Tensor,
    next_emb: torch.Tensor,
    *,
    temperature: float = 0.1,
    normalize: bool = True,
) -> float:
    """Convenience: just the top-1 accuracy scalar (the §4.2 gate metric)."""

    with torch.no_grad():
        out = info_nce_next_frame(
            emb, next_emb, temperature=temperature, normalize=normalize,
        )
    return float(out.top1_accuracy.item())


__all__ = ["InfoNCEOutput", "info_nce_next_frame", "info_nce_top1_accuracy"]
