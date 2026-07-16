"""Generic saliency — mined from CONTRA symmetry / vision / edges."""

from __future__ import annotations

import numpy as np


def edge_map(frame: np.ndarray) -> np.ndarray:
    grid = np.asarray(frame)
    edge = np.zeros(grid.shape, dtype=bool)
    edge[:-1, :] |= grid[:-1, :] != grid[1:, :]
    edge[1:, :] |= grid[1:, :] != grid[:-1, :]
    edge[:, :-1] |= grid[:, :-1] != grid[:, 1:]
    edge[:, 1:] |= grid[:, 1:] != grid[:, :-1]
    return edge.astype(np.float32)


def symmetry_score_horizontal(frame: np.ndarray) -> float:
    """How close the frame is to left-right mirror — hypothesis only."""
    grid = np.asarray(frame)
    flipped = np.flip(grid, axis=1)
    return float(np.mean(grid == flipped))


def symmetry_score_vertical(frame: np.ndarray) -> float:
    grid = np.asarray(frame)
    flipped = np.flip(grid, axis=0)
    return float(np.mean(grid == flipped))


def color_histogram(frame: np.ndarray, n_colors: int = 16) -> np.ndarray:
    flat = np.asarray(frame).reshape(-1).astype(np.int64)
    hist = np.bincount(np.clip(flat, 0, n_colors - 1), minlength=n_colors).astype(np.float64)
    hist /= max(1.0, hist.sum())
    return hist
