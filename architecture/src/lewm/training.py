"""LeWM training loop — paper Algorithm 1.

Iterates over a sequence dataset (each item is a ``(T, C, H, W)`` obs +
``(T, ...)`` action fields), encodes the full sequence, runs the predictor,
and minimizes::

    L_LeWM = MSE(emb[:, 1:], next_emb[:, :-1]) + λ · mean_T(SIGReg(emb[:, t, :]))

AdamW. No stop-gradient (``loss.py`` enforces this), no EMA, no pre-trained
encoder, no auxiliary heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from lewm.loss import PAPER_REG_LAMBDA, lewm_loss
from lewm.model import LeWM, LeWMConfig
from lewm.sigreg import PAPER_NUM_PROJECTIONS, SIGReg, SIGRegConfig


PAPER_BATCH_SIZE = 128
PAPER_SEQUENCE_LEN = 8
PAPER_LEARNING_RATE = 1e-3
PAPER_WEIGHT_DECAY = 1e-5


@dataclass
class TrainingConfig:
    num_epochs: int = 50
    batch_size: int = PAPER_BATCH_SIZE
    sequence_len: int = PAPER_SEQUENCE_LEN
    learning_rate: float = PAPER_LEARNING_RATE
    weight_decay: float = PAPER_WEIGHT_DECAY
    reg_lambda: float = PAPER_REG_LAMBDA
    sigreg_num_projections: int = PAPER_NUM_PROJECTIONS
    seed: int = 0
    log_every: int = 1

    def validate(self) -> None:
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.sequence_len < 2:
            raise ValueError(
                "sequence_len must be >= 2 to form transitions for L_pred"
            )


class LeWMTrainer:
    """Holds the model, SIGReg module, optimizer, and training history."""

    def __init__(
        self,
        *,
        model: LeWM | None = None,
        model_config: LeWMConfig | None = None,
        training_config: TrainingConfig | None = None,
    ) -> None:
        self.training_config = training_config or TrainingConfig()
        self.training_config.validate()
        if model is None:
            model = LeWM(model_config)
        self.model = model
        self.sigreg = SIGReg(
            config=SIGRegConfig(
                latent_dim=self.model.latent_dim,
                num_projections=self.training_config.sigreg_num_projections,
                seed=self.training_config.seed,
            ),
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        torch.manual_seed(self.training_config.seed)
        self.history: list[dict[str, float]] = []

    def train_step(
        self,
        obs: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> dict[str, float]:
        """One LeWM training step over a ``(B, T)`` batch.

        - ``obs``: ``[B, T, C, H, W]`` raw pixel observations
        - ``action_emb``: ``[B, T, action_dim]`` per-step action embeddings

        Returns ``{"pred", "sigreg", "total"}`` per-step floats.
        """

        self.model.train()
        outputs = self.model(obs, action_emb)
        loss = lewm_loss(
            outputs["emb"],
            outputs["next_emb"],
            self.sigreg,
            reg_lambda=self.training_config.reg_lambda,
        )
        self.optimizer.zero_grad()
        loss.total.backward()
        self.optimizer.step()
        return {
            "pred": float(loss.pred.detach()),
            "sigreg": float(loss.sigreg.detach()),
            "total": float(loss.total.detach()),
        }

    def fit(self, batches: Iterable[dict[str, torch.Tensor]]) -> dict:
        cfg = self.training_config
        for epoch in range(cfg.num_epochs):
            epoch_losses = {"pred": 0.0, "sigreg": 0.0, "total": 0.0}
            count = 0
            for batch in batches:
                step_losses = self.train_step(
                    batch["obs"],
                    batch["action_emb"],
                )
                for key in epoch_losses:
                    epoch_losses[key] += step_losses[key]
                count += 1
            if count > 0:
                avg = {k: v / count for k, v in epoch_losses.items()}
            else:
                avg = epoch_losses
            avg["epoch"] = epoch
            self.history.append(avg)
        return {
            "history": list(self.history),
            "final_total": self.history[-1]["total"] if self.history else None,
        }
