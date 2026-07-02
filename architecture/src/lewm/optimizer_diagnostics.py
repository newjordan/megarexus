"""Optimizer diagnostics for rectangular matrix update pathologies.

These helpers are deliberately diagnostic-only. They measure the row-norm and
row-leverage imbalance that Aurora-style optimizers are meant to address, but
they do not change optimizer behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RectangularMatrixDiagnostic:
    """Single 2-D matrix diagnostic after orienting wide matrices as tall."""

    name: str
    original_shape: tuple[int, int]
    analysis_shape: tuple[int, int]
    transposed: bool
    source: str
    target_row_sq: float
    row_norm_mean: float
    row_norm_cv: float
    leverage_mean: float
    leverage_cv: float
    leverage_min: float
    leverage_max: float
    low_leverage_fraction: float
    near_zero_row_fraction: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _coefficient_of_variation(x: torch.Tensor, eps: float = 1e-12) -> float:
    x = x.to(torch.float64)
    mean = x.mean()
    if mean.abs() < eps:
        return 0.0
    return float(x.std(unbiased=False) / mean.abs())


def _orient_tall(matrix: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-D matrix, got shape {tuple(matrix.shape)}")
    if matrix.shape[0] >= matrix.shape[1]:
        return matrix, False
    return matrix.mT, True


def row_leverage_scores(matrix: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    """Return row leverage scores for a 2-D matrix oriented as tall.

    For a tall matrix ``M = U S V^T``, the row leverage scores are
    ``||U[i, :]||^2``. For wide matrices, callers should either transpose first
    or use :func:`rectangular_matrix_diagnostic`, which does that explicitly.
    """

    tall, _ = _orient_tall(matrix.detach())
    tall = tall.to(torch.float64)
    if tall.numel() == 0:
        raise ValueError("cannot compute leverage scores for an empty matrix")
    if float(tall.norm()) <= eps:
        return torch.zeros(tall.shape[0], dtype=torch.float64, device=tall.device)
    u, _, _ = torch.linalg.svd(tall, full_matrices=False)
    return u.square().sum(dim=1)


def rectangular_matrix_diagnostic(
    name: str,
    matrix: torch.Tensor,
    *,
    source: str = "matrix",
    low_leverage_ratio: float = 0.1,
    near_zero_row_ratio: float = 1e-6,
) -> RectangularMatrixDiagnostic:
    """Measure row/update imbalance for one 2-D matrix.

    ``low_leverage_ratio`` is relative to the uniform target ``n / m`` after
    orienting the matrix as tall. This is a single-step warning signal, not the
    persistent neuron-death criterion used in the Aurora report.
    """

    tall, transposed = _orient_tall(matrix.detach())
    tall = tall.to(torch.float64)
    m, n = tall.shape
    target_row_sq = n / m
    row_norm = tall.norm(dim=1)
    leverage = row_leverage_scores(tall)

    low_threshold = target_row_sq * low_leverage_ratio
    mean_row_norm = row_norm.mean()
    near_zero_threshold = max(float(mean_row_norm) * near_zero_row_ratio, 1e-12)

    return RectangularMatrixDiagnostic(
        name=name,
        original_shape=(int(matrix.shape[0]), int(matrix.shape[1])),
        analysis_shape=(int(m), int(n)),
        transposed=transposed,
        source=source,
        target_row_sq=float(target_row_sq),
        row_norm_mean=float(mean_row_norm),
        row_norm_cv=_coefficient_of_variation(row_norm),
        leverage_mean=float(leverage.mean()),
        leverage_cv=_coefficient_of_variation(leverage),
        leverage_min=float(leverage.min()),
        leverage_max=float(leverage.max()),
        low_leverage_fraction=float((leverage < low_threshold).to(torch.float64).mean()),
        near_zero_row_fraction=float((row_norm <= near_zero_threshold).to(torch.float64).mean()),
    )


def iter_model_rectangular_diagnostics(
    model: nn.Module,
    *,
    source: str = "grad",
    name_contains: Iterable[str] | None = None,
    low_leverage_ratio: float = 0.1,
) -> list[RectangularMatrixDiagnostic]:
    """Collect diagnostics for 2-D model parameters or gradients.

    Args:
        model: model to inspect.
        source: ``"grad"`` to inspect gradients, or ``"weight"`` to inspect
            parameter values.
        name_contains: optional substrings; if supplied, only matching
            parameter names are included.
        low_leverage_ratio: threshold passed to
            :func:`rectangular_matrix_diagnostic`.
    """

    if source not in {"grad", "weight"}:
        raise ValueError(f"source must be 'grad' or 'weight', got {source!r}")
    filters = tuple(name_contains or ())
    rows: list[RectangularMatrixDiagnostic] = []
    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        if filters and not any(part in name for part in filters):
            continue
        tensor = param.grad if source == "grad" else param.data
        if tensor is None:
            continue
        rows.append(
            rectangular_matrix_diagnostic(
                name,
                tensor,
                source=source,
                low_leverage_ratio=low_leverage_ratio,
            )
        )
    return rows
