"""Novel predictor variants for controlled Loom experiments.

These modules are not paper-faithful LeWM components. They live behind the
Novel lab protocols and should be promoted only with controlled evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from lewm.predictor import PAPER_DROPOUT, PAPER_HIDDEN_DIM
from lewm.predictor import AdaLNTransformerPredictor, PredictorConfig


def _validate_sequence_inputs(
    z_history: torch.Tensor,
    action_emb: torch.Tensor,
    *,
    latent_dim: int,
    action_dim: int,
) -> None:
    if z_history.dim() != 3:
        raise ValueError(
            f"z_history must be [B, T, latent_dim], got {tuple(z_history.shape)}"
        )
    if action_emb.dim() != 3 or action_emb.shape[:2] != z_history.shape[:2]:
        raise ValueError(
            "action_emb must be [B, T, action_dim] matching z_history, "
            f"got {tuple(action_emb.shape)}"
        )
    if z_history.shape[-1] != latent_dim:
        raise ValueError(
            f"latent_dim mismatch: predictor has {latent_dim}, got {z_history.shape[-1]}"
        )
    if action_emb.shape[-1] != action_dim:
        raise ValueError(
            f"action_dim mismatch: predictor has {action_dim}, got {action_emb.shape[-1]}"
        )


@dataclass(frozen=True)
class MarkovMLPPredictorConfig:
    latent_dim: int = PAPER_HIDDEN_DIM
    action_dim: int = PAPER_HIDDEN_DIM
    hidden_ratio: float = 2.0
    dropout: float = PAPER_DROPOUT

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.hidden_ratio <= 0:
            raise ValueError("hidden_ratio must be > 0")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")


class MarkovMLPPredictor(nn.Module):
    """Two-layer Markovian predictor with no temporal attention.

    Predicts each next latent from only the current latent and current action:

        ``z_hat[t+1] = MLP(norm(z[t]), norm(a[t]))``

    This intentionally removes history and the transformer stack to test
    whether the action pathway improves when the predictor cannot hide behind
    temporal context.
    """

    def __init__(self, config: MarkovMLPPredictorConfig | None = None) -> None:
        super().__init__()
        self.config = config or MarkovMLPPredictorConfig()
        self.config.validate()
        cfg = self.config
        hidden = int(cfg.latent_dim * cfg.hidden_ratio)
        self.z_norm = nn.LayerNorm(cfg.latent_dim)
        self.action_norm = nn.LayerNorm(cfg.action_dim)
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.action_dim, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.latent_dim),
        )

    def forward(self, z_history: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        _validate_sequence_inputs(
            z_history,
            action_emb,
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
        )
        x = torch.cat([self.z_norm(z_history), self.action_norm(action_emb)], dim=-1)
        return self.net(x)


@dataclass(frozen=True)
class RelationalDeltaMultiHeadMLPAdapterConfig:
    latent_dim: int = PAPER_HIDDEN_DIM
    action_dim: int = PAPER_HIDDEN_DIM
    num_layers: int = 6
    num_heads: int = 16
    dropout: float = PAPER_DROPOUT
    num_delta_heads: int = 4
    hidden_ratio: float = 1.0
    init_scale: float = 0.02
    refinement_iterations: int = 1

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.latent_dim % self.num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")
        if self.num_delta_heads < 2:
            raise ValueError("num_delta_heads must be >= 2")
        if self.hidden_ratio <= 0:
            raise ValueError("hidden_ratio must be > 0")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.init_scale < 0:
            raise ValueError("init_scale must be >= 0")
        if self.refinement_iterations < 1:
            raise ValueError("refinement_iterations must be >= 1")


class RelationalDeltaMultiHeadMLPAdapterPredictor(nn.Module):
    """Residual-fixed parent plus multiple causal MLP relational-delta heads.

    This is a targeted repair for the Markov MLP lateral. The standalone MLP
    produced useful action-effect geometry but regressed prediction accuracy.
    This variant keeps the best residual-fixed transformer parent and adds a
    zero-initialized mixture of MLP delta heads:

        ``relation_t = [z_t, a_t, z_t - z_{t-1}]``
        ``delta_t = Σ softmax(g(relation_t))_h * MLP_h(relation_t)``
        ``z_hat[t+1] = residual_fixed_parent(z_t, a_t) + scale * delta_t``

    The gate entropy is a Shannon diagnostic for whether the heads specialize
    or collapse to one expert. It is exposed for the Novel harness, not used as
    a paper-faithful LeWM claim.
    """

    def __init__(
        self,
        config: RelationalDeltaMultiHeadMLPAdapterConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or RelationalDeltaMultiHeadMLPAdapterConfig()
        self.config.validate()
        cfg = self.config
        self.base_predictor = ResidualTransformerDeltaPredictor(
            ResidualTransformerDeltaConfig(
                latent_dim=cfg.latent_dim,
                action_dim=cfg.action_dim,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                init_gain=0.05,
                trainable_gain=False,
            )
        )
        relation_dim = cfg.latent_dim + cfg.action_dim + cfg.latent_dim
        hidden = int(cfg.latent_dim * cfg.hidden_ratio)
        self.z_norm = nn.LayerNorm(cfg.latent_dim)
        self.action_norm = nn.LayerNorm(cfg.action_dim)
        self.delta_norm = nn.LayerNorm(cfg.latent_dim)
        self.gate = nn.Linear(relation_dim, cfg.num_delta_heads)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(relation_dim, hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(hidden, cfg.latent_dim),
            )
            for _ in range(cfg.num_delta_heads)
        ])
        self.scale = nn.Parameter(torch.tensor(float(cfg.init_scale)))
        self.scale.requires_grad_(False)

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        for head in self.heads:
            last = head[-1]
            assert isinstance(last, nn.Linear)
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

        self._last_gate_entropy: torch.Tensor | None = None
        self._last_gate_entropy_norm: torch.Tensor | None = None
        self._last_head_usage: torch.Tensor | None = None
        self._last_delta_norm: torch.Tensor | None = None
        self._last_head_delta_norm: torch.Tensor | None = None
        self._last_refinement_predictions: tuple[torch.Tensor, ...] = ()

    def _relation(self, z_history: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        _validate_sequence_inputs(
            z_history,
            action_emb,
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
        )
        prev = torch.cat([z_history[:, :1], z_history[:, :-1]], dim=1)
        temporal_delta = z_history - prev
        return torch.cat(
            [
                self.z_norm(z_history),
                self.action_norm(action_emb),
                self.delta_norm(temporal_delta),
            ],
            dim=-1,
        )

    def _forward_once(self, z_history: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        base = self.base_predictor(z_history, action_emb)
        relation = self._relation(z_history, action_emb)
        gate_probs = torch.softmax(self.gate(relation), dim=-1)
        head_outputs = torch.stack([head(relation) for head in self.heads], dim=-2)
        delta = (gate_probs.unsqueeze(-1) * head_outputs).sum(dim=-2)

        eps = torch.finfo(gate_probs.dtype).eps
        entropy = -(gate_probs * (gate_probs + eps).log()).sum(dim=-1).mean()
        max_entropy = torch.log(
            torch.tensor(
                float(self.config.num_delta_heads),
                device=gate_probs.device,
                dtype=gate_probs.dtype,
            )
        )
        self._last_gate_entropy = entropy
        self._last_gate_entropy_norm = entropy / max_entropy.clamp_min(eps)
        self._last_head_usage = gate_probs.mean(dim=(0, 1)).detach()
        self._last_delta_norm = delta.norm(dim=-1).mean().detach()
        self._last_head_delta_norm = head_outputs.norm(dim=-1).mean(dim=(0, 1)).detach()
        return base + self.scale * delta

    @staticmethod
    def _refined_history(
        original_history: torch.Tensor,
        predicted_next: torch.Tensor,
    ) -> torch.Tensor:
        refined = original_history.clone()
        if refined.shape[1] > 1:
            refined[:, 1:] = predicted_next[:, :-1]
        return refined

    def forward_refinement_steps(
        self,
        z_history: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        predictions: list[torch.Tensor] = []
        working_history = z_history
        for step_idx in range(self.config.refinement_iterations):
            predicted_next = self._forward_once(working_history, action_emb)
            predictions.append(predicted_next)
            if step_idx + 1 < self.config.refinement_iterations:
                working_history = self._refined_history(z_history, predicted_next)
        self._last_refinement_predictions = tuple(predictions)
        return self._last_refinement_predictions

    def forward(self, z_history: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        return self.forward_refinement_steps(z_history, action_emb)[-1]

    def last_refinement_predictions(self) -> tuple[torch.Tensor, ...]:
        return self._last_refinement_predictions

    def head_entropy_balance_loss(self) -> torch.Tensor:
        if self._last_gate_entropy_norm is None:
            return torch.tensor(0.0, device=self.scale.device, dtype=self.scale.dtype)
        return 1.0 - self._last_gate_entropy_norm.to(device=self.scale.device)

    def diagnostics(self) -> dict[str, float | list[float]]:
        diagnostics: dict[str, float | list[float]] = {}
        if self._last_gate_entropy is not None:
            diagnostics["head_gate_entropy"] = float(self._last_gate_entropy.item())
        if self._last_gate_entropy_norm is not None:
            diagnostics["head_gate_entropy_norm"] = float(
                self._last_gate_entropy_norm.item()
            )
        if self._last_delta_norm is not None:
            diagnostics["delta_norm"] = float(self._last_delta_norm.item())
        if self._last_head_usage is not None:
            diagnostics["head_usage"] = [
                float(x) for x in self._last_head_usage.detach().cpu().tolist()
            ]
        if self._last_head_delta_norm is not None:
            diagnostics["head_delta_norm"] = [
                float(x) for x in self._last_head_delta_norm.detach().cpu().tolist()
            ]
        diagnostics["refinement_iterations"] = float(self.config.refinement_iterations)
        return diagnostics


@dataclass(frozen=True)
class ResidualActionDeltaConfig:
    latent_dim: int = PAPER_HIDDEN_DIM
    action_dim: int = PAPER_HIDDEN_DIM
    hidden_ratio: float = 2.0
    dropout: float = PAPER_DROPOUT
    init_std: float = 1e-4

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.hidden_ratio <= 0:
            raise ValueError("hidden_ratio must be > 0")
        if self.init_std < 0:
            raise ValueError("init_std must be >= 0")


class ResidualActionDeltaPredictor(nn.Module):
    """Predict next latent as current latent plus a learned action delta.

    The paper predictor directly emits ``z_{t+1}``. This Novel variant makes
    the strong adjacent-frame prior explicit: ``z_hat_{t+1} = z_t + Δ(z_t,a_t)``.
    It is useful only if action-shuffle guards show the delta is action-aware.
    """

    def __init__(self, config: ResidualActionDeltaConfig | None = None) -> None:
        super().__init__()
        self.config = config or ResidualActionDeltaConfig()
        self.config.validate()
        cfg = self.config
        hidden = int(cfg.latent_dim * cfg.hidden_ratio)
        self.z_norm = nn.LayerNorm(cfg.latent_dim)
        self.action_norm = nn.LayerNorm(cfg.action_dim)
        self.delta = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.action_dim, hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(hidden, cfg.latent_dim),
        )
        last = self.delta[-1]
        assert isinstance(last, nn.Linear)
        nn.init.normal_(last.weight, mean=0.0, std=cfg.init_std)
        nn.init.zeros_(last.bias)

    def forward(self, z_history: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        if z_history.dim() != 3:
            raise ValueError(
                f"z_history must be [B, T, latent_dim], got {tuple(z_history.shape)}"
            )
        if action_emb.dim() != 3 or action_emb.shape[:2] != z_history.shape[:2]:
            raise ValueError(
                "action_emb must be [B, T, action_dim] matching z_history, "
                f"got {tuple(action_emb.shape)}"
            )
        if z_history.shape[-1] != self.config.latent_dim:
            raise ValueError(
                f"latent_dim mismatch: predictor has {self.config.latent_dim}, "
                f"got {z_history.shape[-1]}"
            )
        if action_emb.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"action_dim mismatch: predictor has {self.config.action_dim}, "
                f"got {action_emb.shape[-1]}"
            )
        x = torch.cat([self.z_norm(z_history), self.action_norm(action_emb)], dim=-1)
        return z_history + self.delta(x)


@dataclass(frozen=True)
class ResidualTransformerDeltaConfig:
    latent_dim: int = PAPER_HIDDEN_DIM
    action_dim: int = PAPER_HIDDEN_DIM
    num_layers: int = 6
    num_heads: int = 16
    dropout: float = PAPER_DROPOUT
    init_gain: float = 0.05
    trainable_gain: bool = True

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.latent_dim % self.num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")
        if self.init_gain < 0:
            raise ValueError("init_gain must be >= 0")


class ResidualTransformerDeltaPredictor(nn.Module):
    """Identity-anchored variant of the paper action transformer.

    The base AdaLN transformer remains the action-conditioned component, but
    its output is interpreted as a delta around ``z_t``:

        ``z_hat_{t+1} = z_t + gain * transformer_delta(z_t, a_t)``

    This tests whether the paper predictor is struggling because it has to
    emit the whole next latent instead of the small transition delta.
    """

    def __init__(self, config: ResidualTransformerDeltaConfig | None = None) -> None:
        super().__init__()
        self.config = config or ResidualTransformerDeltaConfig()
        self.config.validate()
        cfg = self.config
        self.delta_predictor = AdaLNTransformerPredictor(
            PredictorConfig(
                latent_dim=cfg.latent_dim,
                action_dim=cfg.action_dim,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
            )
        )
        self.gain = nn.Parameter(torch.tensor(float(cfg.init_gain)))
        self.gain.requires_grad_(cfg.trainable_gain)

    def forward(self, z_history: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        delta = self.delta_predictor(z_history, action_emb)
        return z_history + self.gain * delta


@dataclass(frozen=True)
class ActionLowRankDeltaConfig:
    latent_dim: int = PAPER_HIDDEN_DIM
    action_dim: int = PAPER_HIDDEN_DIM
    rank: int = 32
    dropout: float = PAPER_DROPOUT
    init_std: float = 1e-3
    init_scale: float = 0.1
    trainable_scale: bool = False

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.init_std < 0:
            raise ValueError("init_std must be >= 0")
        if self.init_scale < 0:
            raise ValueError("init_scale must be >= 0")


class ActionLowRankDeltaPredictor(nn.Module):
    """Identity-anchored action-conditioned low-rank latent transition.

    This Novel variant tests a different hypothesis from the residual
    transformer: action effects may live in a compact controllable subspace.
    It predicts

        ``z_hat[t+1] = z[t] + scale * U((V z[t]) * g(a[t]))``

    where ``g(a[t])`` is an action-dependent rank gate. The zero-initialized gate
    makes the module start as exact identity, while the low-rank operator keeps
    the action pathway small enough to expose collapse in shuffle/zero guards.
    """

    def __init__(self, config: ActionLowRankDeltaConfig | None = None) -> None:
        super().__init__()
        self.config = config or ActionLowRankDeltaConfig()
        self.config.validate()
        cfg = self.config

        self.z_norm = nn.LayerNorm(cfg.latent_dim)
        self.action_norm = nn.LayerNorm(cfg.action_dim)
        self.down = nn.Linear(cfg.latent_dim, cfg.rank, bias=False)
        self.action_gate = nn.Linear(cfg.action_dim, cfg.rank)
        self.dropout = nn.Dropout(cfg.dropout)
        self.up = nn.Linear(cfg.rank, cfg.latent_dim, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(cfg.init_scale)))
        self.scale.requires_grad_(cfg.trainable_scale)

        nn.init.normal_(self.down.weight, mean=0.0, std=cfg.init_std)
        nn.init.normal_(self.up.weight, mean=0.0, std=cfg.init_std)
        nn.init.zeros_(self.action_gate.weight)
        nn.init.zeros_(self.action_gate.bias)

    def forward(self, z_history: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        if z_history.dim() != 3:
            raise ValueError(
                f"z_history must be [B, T, latent_dim], got {tuple(z_history.shape)}"
            )
        if action_emb.dim() != 3 or action_emb.shape[:2] != z_history.shape[:2]:
            raise ValueError(
                "action_emb must be [B, T, action_dim] matching z_history, "
                f"got {tuple(action_emb.shape)}"
            )
        if z_history.shape[-1] != self.config.latent_dim:
            raise ValueError(
                f"latent_dim mismatch: predictor has {self.config.latent_dim}, "
                f"got {z_history.shape[-1]}"
            )
        if action_emb.shape[-1] != self.config.action_dim:
            raise ValueError(
                f"action_dim mismatch: predictor has {self.config.action_dim}, "
                f"got {action_emb.shape[-1]}"
            )

        z_rank = self.down(self.z_norm(z_history))
        action_gate = torch.tanh(self.action_gate(self.action_norm(action_emb)))
        delta_rank = self.dropout(z_rank * action_gate)
        delta = self.up(delta_rank)
        return z_history + self.scale * delta


@dataclass(frozen=True)
class ActionLowRankAdapterConfig:
    latent_dim: int = PAPER_HIDDEN_DIM
    action_dim: int = PAPER_HIDDEN_DIM
    num_layers: int = 6
    num_heads: int = 16
    dropout: float = PAPER_DROPOUT
    rank: int = 32
    init_std: float = 1e-3
    init_scale: float = 0.1
    trainable_scale: bool = False

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.latent_dim % self.num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.init_std < 0:
            raise ValueError("init_std must be >= 0")
        if self.init_scale < 0:
            raise ValueError("init_scale must be >= 0")


class ActionLowRankAdapterPredictor(nn.Module):
    """Paper predictor plus a compact action-conditioned residual adapter.

    The base paper AdaLN transformer still predicts the next latent. A small
    low-rank action operator adds an initialized-zero residual:

        ``z_hat[t+1] = base(z[t], a[t]) + scale * U((V z[t]) * g(a[t]))``

    This keeps the candidate from replacing dynamics with a pure identity-copy
    path while testing whether action effects benefit from an explicit compact
    transition subspace.
    """

    def __init__(self, config: ActionLowRankAdapterConfig | None = None) -> None:
        super().__init__()
        self.config = config or ActionLowRankAdapterConfig()
        self.config.validate()
        cfg = self.config
        self.base_predictor = AdaLNTransformerPredictor(
            PredictorConfig(
                latent_dim=cfg.latent_dim,
                action_dim=cfg.action_dim,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
            )
        )
        self.adapter = ActionLowRankDeltaPredictor(
            ActionLowRankDeltaConfig(
                latent_dim=cfg.latent_dim,
                action_dim=cfg.action_dim,
                rank=cfg.rank,
                dropout=cfg.dropout,
                init_std=cfg.init_std,
                init_scale=cfg.init_scale,
                trainable_scale=cfg.trainable_scale,
            )
        )

    def forward(self, z_history: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        base = self.base_predictor(z_history, action_emb)
        adapted_identity = self.adapter(z_history, action_emb)
        return base + (adapted_identity - z_history)


@dataclass(frozen=True)
class FunctionalTransitionExpansionConfig:
    latent_dim: int = PAPER_HIDDEN_DIM
    action_dim: int = PAPER_HIDDEN_DIM
    num_layers: int = 6
    num_heads: int = 16
    dropout: float = PAPER_DROPOUT
    rank: int = 32
    init_std: float = 1e-3
    init_scale: float = 0.05
    trainable_scale: bool = False

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.latent_dim % self.num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")
        if self.rank < 2:
            raise ValueError("rank must be >= 2")
        if self.rank % 2 != 0:
            raise ValueError("rank must be even for paired rotations")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.init_std < 0:
            raise ValueError("init_std must be >= 0")
        if self.init_scale < 0:
            raise ValueError("init_scale must be >= 0")


class FunctionalTransitionExpansionPredictor(nn.Module):
    """Paper predictor plus explicit transition-function expansion.

    This Novel variant is intentionally a larger function-class change than a
    tuned loss. It leaves the paper AdaLN predictor intact, then adds three
    action-conditioned transition families:

    - additive action translation ``T(a)``
    - bilinear low-rank dynamics ``U((V z) * G(a))``
    - action-phase rotation in an even-rank latent subspace

    The branches initialize to exact zero contribution, so the first forward
    pass matches the base paper predictor before training.
    """

    def __init__(
        self,
        config: FunctionalTransitionExpansionConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or FunctionalTransitionExpansionConfig()
        self.config.validate()
        cfg = self.config

        self.base_predictor = AdaLNTransformerPredictor(
            PredictorConfig(
                latent_dim=cfg.latent_dim,
                action_dim=cfg.action_dim,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
            )
        )
        self.z_norm = nn.LayerNorm(cfg.latent_dim)
        self.action_norm = nn.LayerNorm(cfg.action_dim)
        self.translation = nn.Linear(cfg.action_dim, cfg.latent_dim)

        self.bilinear_down = nn.Linear(cfg.latent_dim, cfg.rank, bias=False)
        self.bilinear_gate = nn.Linear(cfg.action_dim, cfg.rank)
        self.bilinear_up = nn.Linear(cfg.rank, cfg.latent_dim, bias=False)

        self.rotation_down = nn.Linear(cfg.latent_dim, cfg.rank, bias=False)
        self.rotation_phase = nn.Linear(cfg.action_dim, cfg.rank // 2)
        self.rotation_up = nn.Linear(cfg.rank, cfg.latent_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)
        self.scale = nn.Parameter(torch.tensor(float(cfg.init_scale)))
        self.scale.requires_grad_(cfg.trainable_scale)

        nn.init.zeros_(self.translation.weight)
        nn.init.zeros_(self.translation.bias)
        nn.init.normal_(self.bilinear_down.weight, mean=0.0, std=cfg.init_std)
        nn.init.normal_(self.bilinear_up.weight, mean=0.0, std=cfg.init_std)
        nn.init.zeros_(self.bilinear_gate.weight)
        nn.init.zeros_(self.bilinear_gate.bias)
        nn.init.normal_(self.rotation_down.weight, mean=0.0, std=cfg.init_std)
        nn.init.normal_(self.rotation_up.weight, mean=0.0, std=cfg.init_std)
        nn.init.zeros_(self.rotation_phase.weight)
        nn.init.zeros_(self.rotation_phase.bias)

    @staticmethod
    def _rotate_pairs(x: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        x0 = x[..., 0::2]
        x1 = x[..., 1::2]
        cos = torch.cos(phase)
        sin = torch.sin(phase)
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
        return torch.stack((y0, y1), dim=-1).flatten(start_dim=-2)

    def forward(self, z_history: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        _validate_sequence_inputs(
            z_history,
            action_emb,
            latent_dim=cfg.latent_dim,
            action_dim=cfg.action_dim,
        )
        base = self.base_predictor(z_history, action_emb)
        z = self.z_norm(z_history)
        action = self.action_norm(action_emb)

        translation = self.translation(action)
        bilinear_rank = self.bilinear_down(z) * torch.tanh(self.bilinear_gate(action))
        bilinear = self.bilinear_up(self.dropout(bilinear_rank))

        rotation_rank = self.rotation_down(z)
        phase = self.rotation_phase(action)
        rotated_rank = self._rotate_pairs(rotation_rank, phase)
        rotation = self.rotation_up(self.dropout(rotated_rank - rotation_rank))

        return base + self.scale * (translation + bilinear + rotation)


@dataclass(frozen=True)
class ResidualFixedLowRankAdapterConfig:
    latent_dim: int = PAPER_HIDDEN_DIM
    action_dim: int = PAPER_HIDDEN_DIM
    num_layers: int = 6
    num_heads: int = 16
    dropout: float = PAPER_DROPOUT
    rank: int = 32
    init_std: float = 1e-3
    init_scale: float = 0.02

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if self.latent_dim % self.num_heads != 0:
            raise ValueError("latent_dim must be divisible by num_heads")
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.init_std < 0:
            raise ValueError("init_std must be >= 0")
        if self.init_scale < 0:
            raise ValueError("init_scale must be >= 0")


class ResidualFixedLowRankAdapterPredictor(nn.Module):
    """Residual-fixed predictor with a compact action-subspace adapter.

    This repairs the wide functional expansion by keeping the strongest current
    prediction-side parent:

        ``base = z[t] + 0.05 * transformer_delta(z[t], a[t])``

    and adding only a zero-initialized low-rank action residual. At
    initialization, output matches the residual-fixed parent exactly.
    """

    def __init__(
        self,
        config: ResidualFixedLowRankAdapterConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or ResidualFixedLowRankAdapterConfig()
        self.config.validate()
        cfg = self.config
        self.base_predictor = ResidualTransformerDeltaPredictor(
            ResidualTransformerDeltaConfig(
                latent_dim=cfg.latent_dim,
                action_dim=cfg.action_dim,
                num_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                init_gain=0.05,
                trainable_gain=False,
            )
        )
        self.adapter = ActionLowRankDeltaPredictor(
            ActionLowRankDeltaConfig(
                latent_dim=cfg.latent_dim,
                action_dim=cfg.action_dim,
                rank=cfg.rank,
                dropout=cfg.dropout,
                init_std=cfg.init_std,
                init_scale=cfg.init_scale,
                trainable_scale=False,
            )
        )

    def forward(self, z_history: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        base = self.base_predictor(z_history, action_emb)
        adapted_identity = self.adapter(z_history, action_emb)
        return base + (adapted_identity - z_history)


# ---------------------------------------------------------------------------
# Symbiotic relational-delta architectures (2026-05-16)
#
# These combine insights from the URM paper (SwiGLU nonlinearity, iterative
# refinement) with the relational-delta routing mechanism from the top-ranked
# Novel candidates. The goal is symbiotic improvement: each component amplifies
# the other beyond what either achieves alone.
# ---------------------------------------------------------------------------


class _SwiGLUMLP(nn.Module):
    """SwiGLU MLP block as recommended by URM (Shazeer 2020, arXiv:2512.14693).

    SwiGLU(x) = (SiLU(x @ W_gate) * (x @ W_up)) @ W_down
    This provides richer nonlinearity than GELU/ReLU, critical for ARC reasoning.
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


@dataclass(frozen=True)
class SwiGLUResidualActionDeltaConfig:
    """SwiGLU variant of ResidualActionDelta."""

    latent_dim: int = PAPER_HIDDEN_DIM
    action_dim: int = PAPER_HIDDEN_DIM
    hidden_ratio: float = 2.0
    dropout: float = PAPER_DROPOUT
    init_std: float = 1e-4

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.hidden_ratio <= 0:
            raise ValueError("hidden_ratio must be > 0")
        if self.init_std < 0:
            raise ValueError("init_std must be >= 0")


class SwiGLUResidualActionDeltaPredictor(nn.Module):
    """SwiGLU version of ResidualActionDelta.

    Architecture: z_hat_{t+1} = z_t + SwiGLU([z_t, a_t])
    The SwiGLU replaces GELU for richer feature interactions per URM.
    """

    def __init__(self, config: SwiGLUResidualActionDeltaConfig | None = None) -> None:
        super().__init__()
        self.config = config or SwiGLUResidualActionDeltaConfig()
        self.config.validate()
        cfg = self.config
        hidden = int(cfg.latent_dim * cfg.hidden_ratio)
        self.z_norm = nn.LayerNorm(cfg.latent_dim)
        self.action_norm = nn.LayerNorm(cfg.action_dim)
        self.delta = _SwiGLUMLP(
            in_dim=cfg.latent_dim + cfg.action_dim,
            hidden_dim=hidden,
            out_dim=cfg.latent_dim,
            dropout=cfg.dropout,
        )
        nn.init.normal_(self.delta.w_down.weight, mean=0.0, std=cfg.init_std)
        nn.init.zeros_(self.delta.w_down.bias)

    def forward(self, z_history: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        _validate_sequence_inputs(
            z_history, action_emb,
            latent_dim=self.config.latent_dim, action_dim=self.config.action_dim,
        )
        x = torch.cat([self.z_norm(z_history), self.action_norm(action_emb)], dim=-1)
        return z_history + self.delta(x)


@dataclass(frozen=True)
class SwiGLURelationalDeltaMultiHeadMLPAdapterConfig:
    """SwiGLU variant of relational-delta multi-head adapter."""

    latent_dim: int = PAPER_HIDDEN_DIM
    action_dim: int = PAPER_HIDDEN_DIM
    num_layers: int = 6
    num_heads: int = 16
    dropout: float = PAPER_DROPOUT
    num_delta_heads: int = 4
    hidden_ratio: float = 1.0
    init_scale: float = 0.02

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        if self.num_delta_heads < 2:
            raise ValueError("num_delta_heads must be >= 2")


class SwiGLURelationalDeltaMultiHeadMLPAdapterPredictor(nn.Module):
    """SwiGLU variant: SwiGLU heads instead of GELU for richer nonlinearity.

    Architecture:
        relation_t = [z_t, a_t, z_t - z_{t-1}]
        delta_t = Σ softmax(g(relation_t))_h * SwiGLU_MLP_h(relation_t)
        z_hat[t+1] = residual_fixed_parent(z_t, a_t) + scale * delta_t
    """

    def __init__(
        self,
        config: SwiGLURelationalDeltaMultiHeadMLPAdapterConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SwiGLURelationalDeltaMultiHeadMLPAdapterConfig()
        self.config.validate()
        cfg = self.config
        self.base_predictor = ResidualTransformerDeltaPredictor(
            ResidualTransformerDeltaConfig(
                latent_dim=cfg.latent_dim, action_dim=cfg.action_dim,
                num_layers=cfg.num_layers, num_heads=cfg.num_heads,
                dropout=cfg.dropout, init_gain=0.05, trainable_gain=False,
            )
        )
        relation_dim = cfg.latent_dim + cfg.action_dim + cfg.latent_dim
        hidden = int(cfg.latent_dim * cfg.hidden_ratio)
        self.z_norm = nn.LayerNorm(cfg.latent_dim)
        self.action_norm = nn.LayerNorm(cfg.action_dim)
        self.delta_norm = nn.LayerNorm(cfg.latent_dim)
        self.gate = nn.Linear(relation_dim, cfg.num_delta_heads)
        self.heads = nn.ModuleList([
            _SwiGLUMLP(relation_dim, hidden, cfg.latent_dim, cfg.dropout)
            for _ in range(cfg.num_delta_heads)
        ])
        self.scale = nn.Parameter(torch.tensor(float(cfg.init_scale)))
        self.scale.requires_grad_(False)

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        for head in self.heads:
            nn.init.zeros_(head.w_down.weight)
            nn.init.zeros_(head.w_down.bias)

        self._last_gate_entropy_norm: torch.Tensor | None = None
        self._last_delta_norm: torch.Tensor | None = None
        self._last_head_usage: torch.Tensor | None = None

    def _relation(self, z_history: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        _validate_sequence_inputs(z_history, action_emb,
                                   latent_dim=cfg.latent_dim, action_dim=cfg.action_dim)
        prev = torch.cat([z_history[:, :1], z_history[:, :-1]], dim=1)
        temporal_delta = z_history - prev
        return torch.cat([
            self.z_norm(z_history),
            self.action_norm(action_emb),
            self.delta_norm(temporal_delta),
        ], dim=-1)

    def forward(self, z_history: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        base = self.base_predictor(z_history, action_emb)
        relation = self._relation(z_history, action_emb)
        gate_probs = torch.softmax(self.gate(relation), dim=-1)
        head_outputs = torch.stack([head(relation) for head in self.heads], dim=-2)
        delta = (gate_probs.unsqueeze(-1) * head_outputs).sum(dim=-2)
        eps = torch.finfo(gate_probs.dtype).eps
        entropy = -(gate_probs * (gate_probs + eps).log()).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(
            float(self.config.num_delta_heads), device=gate_probs.device, dtype=gate_probs.dtype))
        self._last_gate_entropy_norm = entropy / max_entropy.clamp_min(eps)
        self._last_head_usage = gate_probs.mean(dim=(0, 1)).detach()
        self._last_delta_norm = delta.norm(dim=-1).mean().detach()
        return base + self.scale * delta

    def head_entropy_balance_loss(self) -> torch.Tensor:
        if self._last_gate_entropy_norm is None:
            return torch.tensor(0.0, device=self.scale.device, dtype=self.scale.dtype)
        return 1.0 - self._last_gate_entropy_norm.to(device=self.scale.device)

    def diagnostics(self) -> dict[str, float | list[float]]:
        d: dict[str, float | list[float]] = {}
        if self._last_gate_entropy_norm is not None:
            d["head_gate_entropy_norm"] = float(self._last_gate_entropy_norm.item())
        if self._last_delta_norm is not None:
            d["delta_norm"] = float(self._last_delta_norm.item())
        if self._last_head_usage is not None:
            d["head_usage"] = [float(x) for x in self._last_head_usage.cpu().tolist()]
        return d


@dataclass(frozen=True)
class EnrichedRelationalDeltaMultiHeadMLPAdapterConfig:
    """Relational-delta adapter with enriched relational features.

    Adds cosine similarity and element-wise interaction terms to the relational
    gate input, giving the router richer temporal-change signals.
    """

    latent_dim: int = PAPER_HIDDEN_DIM
    action_dim: int = PAPER_HIDDEN_DIM
    num_layers: int = 6
    num_heads: int = 16
    dropout: float = PAPER_DROPOUT
    num_delta_heads: int = 4
    hidden_ratio: float = 1.0
    init_scale: float = 0.02
    use_swiglu: bool = True

    def validate(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.num_delta_heads < 2:
            raise ValueError("num_delta_heads must be >= 2")


class EnrichedRelationalDeltaMultiHeadMLPAdapterPredictor(nn.Module):
    """Enriched relational features: adds cosine sim and element-wise product.

    The standard relational gate sees: [z_t, a_t, z_t - z_{t-1}].
    The enriched gate additionally sees:
      - cos(z_t, z_{t-1}) — direction of change (scalar)
      - z_t * z_{t-1} — element-wise interaction

    These give the router richer signals about HOW the state is changing,
    not just the magnitude of the delta. Combined with optional SwiGLU heads
    for stronger nonlinearity.
    """

    def __init__(
        self,
        config: EnrichedRelationalDeltaMultiHeadMLPAdapterConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or EnrichedRelationalDeltaMultiHeadMLPAdapterConfig()
        self.config.validate()
        cfg = self.config
        self.base_predictor = ResidualTransformerDeltaPredictor(
            ResidualTransformerDeltaConfig(
                latent_dim=cfg.latent_dim, action_dim=cfg.action_dim,
                num_layers=cfg.num_layers, num_heads=cfg.num_heads,
                dropout=cfg.dropout, init_gain=0.05, trainable_gain=False,
            )
        )
        # Enriched: base 3 vectors + scalar cosine + element-wise product
        self.relation_dim = cfg.latent_dim + cfg.action_dim + cfg.latent_dim + 1 + cfg.latent_dim
        hidden = int(cfg.latent_dim * cfg.hidden_ratio)
        self.z_norm = nn.LayerNorm(cfg.latent_dim)
        self.action_norm = nn.LayerNorm(cfg.action_dim)
        self.delta_norm = nn.LayerNorm(cfg.latent_dim)
        self.interaction_norm = nn.LayerNorm(cfg.latent_dim)
        self.gate = nn.Linear(self.relation_dim, cfg.num_delta_heads)
        if cfg.use_swiglu:
            self.heads = nn.ModuleList([
                _SwiGLUMLP(self.relation_dim, hidden, cfg.latent_dim, cfg.dropout)
                for _ in range(cfg.num_delta_heads)
            ])
        else:
            self.heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.relation_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(hidden, cfg.latent_dim),
                )
                for _ in range(cfg.num_delta_heads)
            ])
        self.scale = nn.Parameter(torch.tensor(float(cfg.init_scale)))
        self.scale.requires_grad_(False)

        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        for head in self.heads:
            if isinstance(head, _SwiGLUMLP):
                nn.init.zeros_(head.w_down.weight)
                nn.init.zeros_(head.w_down.bias)
            else:
                last = head[-1]
                assert isinstance(last, nn.Linear)
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

        self._last_gate_entropy_norm: torch.Tensor | None = None
        self._last_delta_norm: torch.Tensor | None = None
        self._last_head_usage: torch.Tensor | None = None

    def _relation(self, z_history: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        _validate_sequence_inputs(z_history, action_emb,
                                   latent_dim=cfg.latent_dim, action_dim=cfg.action_dim)
        prev = torch.cat([z_history[:, :1], z_history[:, :-1]], dim=1)
        temporal_delta = z_history - prev
        # Cosine similarity
        z_n = torch.nn.functional.normalize(z_history, dim=-1)
        p_n = torch.nn.functional.normalize(prev, dim=-1)
        cosine_sim = (z_n * p_n).sum(dim=-1, keepdim=True)
        # Element-wise interaction
        interaction = z_history * prev
        return torch.cat([
            self.z_norm(z_history),
            self.action_norm(action_emb),
            self.delta_norm(temporal_delta),
            cosine_sim,
            self.interaction_norm(interaction),
        ], dim=-1)

    def forward(self, z_history: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        base = self.base_predictor(z_history, action_emb)
        relation = self._relation(z_history, action_emb)
        gate_probs = torch.softmax(self.gate(relation), dim=-1)
        head_outputs = torch.stack([head(relation) for head in self.heads], dim=-2)
        delta = (gate_probs.unsqueeze(-1) * head_outputs).sum(dim=-2)
        eps = torch.finfo(gate_probs.dtype).eps
        entropy = -(gate_probs * (gate_probs + eps).log()).sum(dim=-1).mean()
        max_entropy = torch.log(torch.tensor(
            float(self.config.num_delta_heads), device=gate_probs.device, dtype=gate_probs.dtype))
        self._last_gate_entropy_norm = entropy / max_entropy.clamp_min(eps)
        self._last_head_usage = gate_probs.mean(dim=(0, 1)).detach()
        self._last_delta_norm = delta.norm(dim=-1).mean().detach()
        return base + self.scale * delta

    def head_entropy_balance_loss(self) -> torch.Tensor:
        if self._last_gate_entropy_norm is None:
            return torch.tensor(0.0, device=self.scale.device, dtype=self.scale.dtype)
        return 1.0 - self._last_gate_entropy_norm.to(device=self.scale.device)

    def diagnostics(self) -> dict[str, float | list[float]]:
        d: dict[str, float | list[float]] = {}
        if self._last_gate_entropy_norm is not None:
            d["head_gate_entropy_norm"] = float(self._last_gate_entropy_norm.item())
        if self._last_delta_norm is not None:
            d["delta_norm"] = float(self._last_delta_norm.item())
        if self._last_head_usage is not None:
            d["head_usage"] = [float(x) for x in self._last_head_usage.cpu().tolist()]
        return d
