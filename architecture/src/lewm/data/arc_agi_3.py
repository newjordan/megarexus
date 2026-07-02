"""ARC-AGI-3 dataset adapter for LeWM.

Wraps :mod:`arc_agi.Arcade` for trajectory rollout and exposes a
``torch.utils.data.Dataset`` that yields LeWM-shaped batches over those
trajectories, plus a companion ``ARCActionEmbedder`` ``nn.Module`` that
turns raw action fields into the ``action_emb`` tensor the predictor
takes.

Observation format (ground truth from ``scripts/arc_agi_smoke_test.py``):
- ``obs.frame`` is a ``list[np.ndarray]`` — list because turns can return
  multiple frames as a transition animation
- each array is ``shape=(64, 64), dtype=int8`` with values 0..15 (the 16
  colors as integer indices, *not* RGB)
- the per-environment ``action_space`` is a subset of
  ``GameAction.ACTION1..ACTION7``; ``ACTION6`` is the only one for which
  ``action.is_complex()`` is True (it takes ``{"x", "y"}`` on the 64×64
  grid)

Encoding choices:
- frames are converted to one-hot ``[16, 64, 64]`` float tensors at
  ``__getitem__`` time. The encoder's Conv2d patchify (``in_channels=16``,
  ``patch_size=16``) then performs the per-color projection implicitly,
  so no encoder code change is needed beyond the config
- actions are stored raw (action_id, x, y, complex_mask) and embedded by
  ``ARCActionEmbedder`` so gradients flow into the embedding tables
- only the *last* frame of each animation sequence is retained for
  training pairs; mid-animation frames are dropped. Animation-aware
  modeling is out of scope for this adapter

Storage: trajectories serialize via ``torch.save`` to a single ``.pt``
file. Per-game total data is small (the largest realistic rollout budget
puts a single game at well under 100 MB), so a single shard works fine.
Switch to zarr or per-game shards if and when scale demands it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import arc_agi
from arcengine import GameAction, GameState

from lewm.encoder import EncoderConfig


ARC_GRID_SIZE = 64
ARC_NUM_COLORS = 16
ARC_NUM_ACTION_KINDS = 7
ARC_NULL_ACTION_ID = 0
ARC_TOTAL_ACTION_TOKENS = ARC_NUM_ACTION_KINDS + 1
ARC_COMPLEX_ACTION_ID = int(GameAction.ACTION6.value)
ARC_DEFAULT_PATCH_SIZE = 16


def make_arc_encoder_config(**overrides: object) -> EncoderConfig:
    """``EncoderConfig`` pre-set for 64×64×16-color ARC-AGI-3 frames."""

    defaults: dict[str, object] = dict(
        image_size=ARC_GRID_SIZE,
        in_channels=ARC_NUM_COLORS,
        patch_size=ARC_DEFAULT_PATCH_SIZE,
    )
    defaults.update(overrides)
    return EncoderConfig(**defaults)  # type: ignore[arg-type]


def frame_to_onehot(frame: np.ndarray) -> torch.Tensor:
    """``[64, 64]`` int → ``[16, 64, 64]`` float one-hot."""

    if frame.shape != (ARC_GRID_SIZE, ARC_GRID_SIZE):
        raise ValueError(
            f"expected frame shape ({ARC_GRID_SIZE}, {ARC_GRID_SIZE}), "
            f"got {frame.shape}"
        )
    indices = torch.from_numpy(frame.astype(np.int64))
    if indices.min() < 0 or indices.max() >= ARC_NUM_COLORS:
        raise ValueError(
            f"frame values must be in [0, {ARC_NUM_COLORS}), got "
            f"[{int(indices.min())}, {int(indices.max())}]"
        )
    return nn.functional.one_hot(indices, ARC_NUM_COLORS).permute(2, 0, 1).float()


# Canonical ARC-AGI-3 RGB palette (matches arc_agi.rendering.COLOR_MAP).
# Used by frame_to_canonical_rgb to render frames as 3-channel images for
# the paper-faithful LeWM RGB-input variant (vs the 16-channel one-hot
# we default to on this branch — see frame_to_onehot above).
ARC_CANONICAL_RGB_PALETTE = np.array([
    [255, 255, 255], [204, 204, 204], [153, 153, 153], [102, 102, 102],
    [ 51,  51,  51], [  0,   0,   0], [229,  58, 163], [255, 123, 204],
    [249,  60,  49], [ 30, 147, 255], [136, 216, 241], [255, 220,   0],
    [255, 133,  27], [146,  18,  49], [ 79, 204,  48], [163,  86, 214],
], dtype=np.uint8)


def frame_to_canonical_rgb(frame: np.ndarray) -> torch.Tensor:
    """``[64, 64]`` int → ``[3, 64, 64]`` float RGB in [0, 1] using the canonical
    arc_agi.rendering.COLOR_MAP palette. Paper-faithful LeWM input format
    (vs the [16, 64, 64] one-hot in frame_to_onehot)."""

    if frame.shape != (ARC_GRID_SIZE, ARC_GRID_SIZE):
        raise ValueError(
            f"expected frame shape ({ARC_GRID_SIZE}, {ARC_GRID_SIZE}), "
            f"got {frame.shape}"
        )
    arr = np.asarray(frame, dtype=np.int64)
    if arr.min() < 0 or arr.max() >= ARC_NUM_COLORS:
        raise ValueError(
            f"frame values must be in [0, {ARC_NUM_COLORS}), got "
            f"[{int(arr.min())}, {int(arr.max())}]"
        )
    rgb = ARC_CANONICAL_RGB_PALETTE[arr]  # [64, 64, 3] uint8
    # Rearrange to [3, 64, 64] and normalize to [0, 1] float.
    return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0


def compute_dense_reward(
    frame_prev: np.ndarray,
    frame_next: np.ndarray,
    *,
    level_up: bool = False,
    alpha: float = 1.0,
    beta: float = 0.01,
    outlier_threshold: int = 1000,
) -> float:
    """Dense per-step reward for LeWM-v2.

    Formula (proposal §3.1): ``log(1+pixel_diff) + alpha*level_up - beta*noise_variance``.

    - ``pixel_diff`` = number of cells where ``frame_prev != frame_next``
      (0..4096 on a 64x64 grid). When ``pixel_diff > outlier_threshold``
      the count is capped at ``outlier_threshold`` (assumed to be an
      animation artifact in which most/all cells churn).
    - ``noise_variance`` = variance of the per-row pixel-diff counts. This
      is a cheap proxy for "many rows changed uniformly" (animation
      sweep / global flash) vs "a small region changed coherently" (real
      agent-driven transition).
    - ``level_up`` preserves the sparse level-completion signal as an
      additive bonus weighted by ``alpha``.

    Both frames must be ``np.ndarray`` of shape ``(64, 64)`` (the ARC-AGI-3
    grid). Returns a Python ``float``.

    Parameters
    ----------
    frame_prev, frame_next:
        Settled frames before/after the action; shape ``(64, 64)``.
        ``dtype`` is free (int8/uint8/int64 all accepted) — comparison
        only requires equality.
    level_up:
        ``True`` when ``levels_completed`` strictly increased on this
        step. Contributes ``alpha`` to the reward.
    alpha:
        Weight on the level-up indicator. Default ``1.0``.
    beta:
        Weight on the per-row noise-variance penalty. Default ``0.01``.
    outlier_threshold:
        Pixel-diff counts above this are capped before the ``log``. This
        prevents whole-frame animation artifacts (often > 3000 changed
        cells) from dominating the reward tail. Default ``1000``.
    """

    if frame_prev.shape != (ARC_GRID_SIZE, ARC_GRID_SIZE):
        raise ValueError(
            f"frame_prev shape must be ({ARC_GRID_SIZE}, {ARC_GRID_SIZE}), "
            f"got {frame_prev.shape}"
        )
    if frame_next.shape != (ARC_GRID_SIZE, ARC_GRID_SIZE):
        raise ValueError(
            f"frame_next shape must be ({ARC_GRID_SIZE}, {ARC_GRID_SIZE}), "
            f"got {frame_next.shape}"
        )

    # Cast to a common signed type so comparison is well-defined across
    # the int8 (ARCStep.frame) / uint8 callers.
    diff_mask = frame_prev.astype(np.int64) != frame_next.astype(np.int64)
    pixel_diff = int(diff_mask.sum())
    # Per-row change counts (length 64); used both for the variance term
    # and the final aggregate (after capping).
    row_diffs = diff_mask.sum(axis=1).astype(np.float64)
    noise_variance = float(row_diffs.var())

    capped_pixel_diff = min(pixel_diff, int(outlier_threshold))
    reward = float(np.log1p(capped_pixel_diff))
    if level_up:
        reward += float(alpha)
    reward -= float(beta) * noise_variance
    return float(reward)


@dataclass
class ARCStep:
    """One agent turn: action taken and its resulting (last) frame.

    ``frame`` is the *last* frame of the post-action animation sequence —
    i.e. the settled state the next action will operate on. The full
    animation sequence is dropped at collection time.
    """

    action_id: int
    action_x: int
    action_y: int
    action_complex: bool
    frame: np.ndarray
    state: str
    levels_completed: int


@dataclass
class ARCTrajectory:
    """One playthrough of one environment.

    ``steps[i].frame`` is the state *after* ``steps[i]``'s action was
    applied. Hence training a transition is: predict ``steps[i].frame``
    given prior frames ``steps[i-T..i-1].frame`` and action
    ``steps[i].(action_id, x, y)``.

    There is no separate "initial frame" recorded — we only see frames
    that result from agent actions, and the first usable transition is
    therefore from step 0 to step 1.
    """

    game_id: str
    win_levels: int
    steps: list[ARCStep]


def _action_fields(action: GameAction, data: dict) -> tuple[int, int, int, bool]:
    is_complex = bool(action.is_complex())
    x = int(data["x"]) if is_complex and "x" in data else 0
    y = int(data["y"]) if is_complex and "y" in data else 0
    return int(action.value), x, y, is_complex


def _state_to_str(state: object) -> str:
    if hasattr(state, "value"):
        return str(state.value)  # type: ignore[union-attr]
    return str(state)


PolicyFn = Callable[
    [list[GameAction], np.ndarray, random.Random],
    tuple[GameAction, dict],
]


def random_policy(
    action_space: list[GameAction],
    frame: np.ndarray,
    rng: random.Random,
) -> tuple[GameAction, dict]:
    """Uniform-random action; uniform-random (x, y) for complex actions."""

    del frame
    action = rng.choice(action_space)
    if action.is_complex():
        return action, {
            "x": rng.randint(0, ARC_GRID_SIZE - 1),
            "y": rng.randint(0, ARC_GRID_SIZE - 1),
        }
    return action, {}


def collect_trajectory(
    arcade: arc_agi.Arcade,
    game_id: str,
    *,
    max_steps: int = 1000,
    policy: PolicyFn = random_policy,
    seed: int = 0,
) -> ARCTrajectory:
    """Roll out one trajectory in ``game_id`` using ``policy``.

    Stops on ``WIN``, ``GAME_OVER``, ``max_steps``, or any step that
    returns ``None``. The returned trajectory's ``game_id`` is the
    *versioned* id (e.g. ``"ls20-9607627b"``) read from the first obs.
    """

    rng = random.Random(seed)
    env = arcade.make(game_id, render_mode=None)
    if env is None:
        raise RuntimeError(f"failed to instantiate environment {game_id!r}")

    steps: list[ARCStep] = []
    versioned_id: str = game_id
    win_levels: int = 0

    last_frame = np.zeros((ARC_GRID_SIZE, ARC_GRID_SIZE), dtype=np.int8)
    for _ in range(max_steps):
        action, data = policy(env.action_space, last_frame, rng)
        obs = env.step(action, data=data)
        if obs is None:
            break
        if not obs.frame:
            break
        last_frame = obs.frame[-1]
        action_id, x, y, is_complex = _action_fields(action, data)
        steps.append(
            ARCStep(
                action_id=action_id,
                action_x=x,
                action_y=y,
                action_complex=is_complex,
                frame=last_frame.copy(),
                state=_state_to_str(obs.state),
                levels_completed=int(obs.levels_completed),
            )
        )
        versioned_id = str(obs.game_id)
        win_levels = int(obs.win_levels)
        if obs.state == GameState.WIN or obs.state == GameState.GAME_OVER:
            break

    return ARCTrajectory(game_id=versioned_id, win_levels=win_levels, steps=steps)


# Module-level worker state for the multiprocessing pool. Each worker
# process gets its own Arcade instance (initialized once via the pool
# initializer) and reuses it for all tasks the worker handles.
_WORKER_ARCADE: arc_agi.Arcade | None = None


def _worker_init() -> None:
    global _WORKER_ARCADE
    _WORKER_ARCADE = arc_agi.Arcade()


def _worker_collect(task: tuple[str, int, int]) -> ARCTrajectory:
    game_id, max_steps, task_seed = task
    assert _WORKER_ARCADE is not None
    return collect_trajectory(
        _WORKER_ARCADE,
        game_id,
        max_steps=max_steps,
        policy=random_policy,
        seed=task_seed,
    )


def collect_trajectories(
    *,
    game_ids: Iterable[str] | None = None,
    rollouts_per_game: int = 4,
    max_steps_per_rollout: int = 1000,
    policy: PolicyFn = random_policy,
    seed: int = 0,
    num_workers: int = 1,
) -> list[ARCTrajectory]:
    """Roll out trajectories across several games.

    ``game_ids=None`` defaults to all locally-or-remotely available games
    (the 25 public-demo envs when running with the anonymous key).

    With ``num_workers > 1`` collection runs in a ``multiprocessing.Pool``,
    one ``arc_agi.Arcade`` per worker (held alive by the pool initializer).
    Parallel mode is restricted to ``policy=random_policy`` because pickling
    arbitrary callables across spawned processes is fragile; pass
    ``num_workers=1`` for custom policies. All games are pre-warmed in the
    parent process before the pool starts so workers don't race on the
    on-disk ``environment_files/`` cache.
    """

    arcade = arc_agi.Arcade()
    if game_ids is None:
        game_ids = [info.game_id for info in arcade.available_environments]
    else:
        game_ids = list(game_ids)

    if num_workers > 1:
        for gid in game_ids:
            if arcade.make(gid, render_mode=None) is None:
                raise RuntimeError(f"could not pre-warm environment {gid!r}")

    tasks: list[tuple[str, int, int]] = []
    for game_idx, gid in enumerate(game_ids):
        for r in range(rollouts_per_game):
            tasks.append((gid, max_steps_per_rollout, seed + 10_000 * game_idx + r))

    if num_workers <= 1:
        out: list[ARCTrajectory] = []
        for gid, max_s, s in tasks:
            out.append(collect_trajectory(
                arcade, gid, max_steps=max_s, policy=policy, seed=s,
            ))
        return out

    if policy is not random_policy:
        raise ValueError(
            "num_workers > 1 only supports policy=random_policy "
            "(pickling arbitrary callables across processes is fragile). "
            "Use num_workers=1 for custom policies."
        )

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(num_workers, initializer=_worker_init) as pool:
        return list(pool.map(_worker_collect, tasks))


def save_trajectories(
    trajectories: list[ARCTrajectory], path: str | Path,
) -> None:
    """Serialize trajectories to a single ``.pt`` shard."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "game_id": t.game_id,
            "win_levels": t.win_levels,
            "steps": [
                {
                    "action_id": s.action_id,
                    "action_x": s.action_x,
                    "action_y": s.action_y,
                    "action_complex": s.action_complex,
                    "frame": s.frame,
                    "state": s.state,
                    "levels_completed": s.levels_completed,
                }
                for s in t.steps
            ],
        }
        for t in trajectories
    ]
    torch.save(payload, path)


def load_trajectories(path: str | Path) -> list[ARCTrajectory]:
    raw = torch.load(Path(path), weights_only=False)
    out: list[ARCTrajectory] = []
    for d in raw:
        out.append(
            ARCTrajectory(
                game_id=d["game_id"],
                win_levels=d["win_levels"],
                steps=[ARCStep(**s) for s in d["steps"]],
            )
        )
    return out


class ARCActionEmbedder(nn.Module):
    """Maps raw action fields to a continuous ``action_emb`` tensor.

    Shape-agnostic over leading dims — the same module is used for:

    - **Training** with ``(B, T, action_dim)`` outputs from ``(B, T)``-shaped
      ``action_id`` / ``action_x`` / ``action_y`` / ``action_complex_mask`` inputs
    - **Inference** with ``(N, action_dim)`` outputs from ``(N,)``-shaped inputs
      (e.g. when scoring N candidate actions for one current state)

    Both action-id and (x, y) embeddings are learned. The (x, y) term is
    multiplied by the mask so it cleanly drops out for non-complex
    actions, leaving the gradient route through ``action_id`` only.

    Field semantics:

    - ``action_id``: long, ``0=null``; ``1..7=ACTION1..ACTION7``
    - ``action_x``, ``action_y``: long in ``[0, 64)``; only meaningful when
      ``action_complex_mask`` is non-zero
    - ``action_complex_mask``: float, 0 or 1; gates the (x, y) term
    """

    def __init__(self, action_dim: int = 192) -> None:
        super().__init__()
        if action_dim < 1:
            raise ValueError("action_dim must be >= 1")
        self.action_dim = action_dim
        self.action_emb = nn.Embedding(ARC_TOTAL_ACTION_TOKENS, action_dim)
        self.x_emb = nn.Embedding(ARC_GRID_SIZE, action_dim)
        self.y_emb = nn.Embedding(ARC_GRID_SIZE, action_dim)
        nn.init.trunc_normal_(self.action_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.x_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.y_emb.weight, std=0.02)

    def forward(
        self,
        action_id: torch.Tensor,
        action_x: torch.Tensor,
        action_y: torch.Tensor,
        action_complex_mask: torch.Tensor,
    ) -> torch.Tensor:
        a = self.action_emb(action_id)
        xy = self.x_emb(action_x) + self.y_emb(action_y)
        mask = action_complex_mask.to(device=xy.device, dtype=xy.dtype).unsqueeze(-1)
        return a + mask * xy


class ARCTrajectoryDataset(Dataset):
    """Slice trajectories into ``T``-step sequence windows for paper Algorithm 1.

    Each item is a dict with:

    - ``obs``: ``[T, 16, 64, 64]`` one-hot float
    - ``action_id``: ``[T]`` long
    - ``action_x``, ``action_y``: ``[T]`` long
    - ``action_complex_mask``: ``[T]`` float

    Mapping from our ``ARCStep`` schema (``step[i]`` records the action ``a_i``
    that was taken from the prior frame and produced ``step[i].frame`` =
    ``o_{i+1}``) to the paper's ``(o_0, ..., o_{T-1})`` and ``(a_0, ..., a_{T-1})``::

        o_t  ↔  step[k-1+t].frame   for t in 0..T-1
        a_t  ↔  step[k+t].action    for t in 0..T-1

    So the window touches steps ``[k-1, k+T-1]`` inclusive (length ``T+1``). The
    loss ``MSE(emb[:, 1:], next_emb[:, :-1])`` covers ``T-1`` transitions per item.

    Windows are dropped if any step in the obs span ``[k-1, k+T-2]`` is
    terminal — that would mean the env was reset mid-window and the
    transition would cross an episode boundary.
    """

    def __init__(
        self,
        trajectories: list[ARCTrajectory],
        *,
        sequence_len: int = 8,
    ) -> None:
        if sequence_len < 2:
            raise ValueError(
                "sequence_len must be >= 2 (need at least one transition for L_pred)"
            )
        self.sequence_len = sequence_len
        self.trajectories = trajectories
        self.windows: list[tuple[int, int]] = []
        for t_idx, traj in enumerate(trajectories):
            n = len(traj.steps)
            for k in range(1, n - sequence_len + 1):
                if self._window_valid(traj, k, sequence_len):
                    self.windows.append((t_idx, k))

    @staticmethod
    def _window_valid(traj: ARCTrajectory, k: int, sequence_len: int) -> bool:
        # Obs span is step[k-1 : k+T-1]; require all of these NOT_FINISHED so
        # no episode boundary is crossed mid-window.
        for i in range(k - 1, k + sequence_len - 1):
            if traj.steps[i].state != "NOT_FINISHED":
                return False
        return True

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        t_idx, k = self.windows[idx]
        traj = self.trajectories[t_idx]
        T = self.sequence_len
        obs_steps = traj.steps[k - 1 : k + T - 1]      # T frames: o_0..o_{T-1}
        action_steps = traj.steps[k : k + T]            # T actions: a_0..a_{T-1}

        obs = torch.stack([frame_to_onehot(s.frame) for s in obs_steps])
        action_id = torch.tensor(
            [s.action_id for s in action_steps], dtype=torch.long,
        )
        action_x = torch.tensor(
            [s.action_x for s in action_steps], dtype=torch.long,
        )
        action_y = torch.tensor(
            [s.action_y for s in action_steps], dtype=torch.long,
        )
        action_complex_mask = torch.tensor(
            [1.0 if s.action_complex else 0.0 for s in action_steps],
            dtype=torch.float,
        )
        # Dense per-step reward (LeWM-v2 §3.1): reward[t] is the reward of
        # transition (o_t, a_t) → o_{t+1} where o_{t+1} = action_steps[t].frame.
        reward = torch.tensor(
            [
                compute_dense_reward(
                    obs_steps[t].frame,
                    action_steps[t].frame,
                    level_up=(action_steps[t].levels_completed > obs_steps[t].levels_completed),
                )
                for t in range(T)
            ],
            dtype=torch.float32,
        )
        return {
            "obs": obs,
            "action_id": action_id,
            "action_x": action_x,
            "action_y": action_y,
            "action_complex_mask": action_complex_mask,
            "reward": reward,
        }
