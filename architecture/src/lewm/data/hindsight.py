"""HindsightARCDataset — every transition is positive for its achieved goal.

Per LeWM-v2 proposal §3.5: classic BC fails when positive trajectories are
rare (0.056% on iter-0). Hindsight relabeling sidesteps the rarity problem
by making EVERY transition positive — for the goal it actually achieved.
For each window ``(o_0, ..., o_{T-1})`` with action sequence ``(a_0, ..., a_{T-1})``
that produces final frame ``o_T``, we treat ``o_T`` as the goal and train::

    model(obs=(o_0..o_{T-1}), actions=(a_0..a_{T-1}), goal_latent=encoder(o_T))
        → predicts (o_1..o_T) at every step

This is HER (Andrychowicz et al. 2017) adapted to LeWM-v2's continuous-goal
encoder substrate. Filtering: windows where ``pixel_diff(o_0, o_T) == 0``
are dropped — those don't move the agent at all and would teach the
model that "the null action achieves the null goal" (trivially true,
gradient-free).

Connections:
  - Reuses ``compute_dense_reward`` (v2-iter 1) for filtering: the pixel-diff
    function is the same one used for reward, so filter and reward are
    coherent.
  - Feeds the redesigned MegaREXConfig.goal_mode='latent' (v2-iter 3 part 2).
  - The goal latent is computed lazily at training time: this dataset just
    emits the goal *frame*; the trainer encodes it. Keeps disk/memory small.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from typing import Callable

from lewm.data.arc_agi_3 import (
    ARC_GRID_SIZE,
    ARCTrajectory,
    ARCTrajectoryDataset,
    compute_dense_reward,
    frame_to_onehot,
)


def _pixel_diff(a: np.ndarray, b: np.ndarray) -> int:
    return int(
        (np.asarray(a, dtype=np.int64) != np.asarray(b, dtype=np.int64)).sum()
    )


class HindsightARCDataset(Dataset):
    """Hindsight goal-conditioned window dataset.

    Each item emits:

    - ``obs``                  : ``[T, 16, 64, 64]`` one-hot
    - ``action_id``            : ``[T]`` long
    - ``action_x``, ``action_y``: ``[T]`` long
    - ``action_complex_mask``  : ``[T]`` float
    - ``reward``               : ``[T]`` float (dense per-step reward)
    - ``goal_frame``           : ``[16, 64, 64]`` one-hot — the final frame ``o_T``
    - ``goal_pixel_diff``      : ``int`` — pixel_diff(o_0, o_T); always > 0 after filter
    - ``window_origin``        : ``(t_idx, k)`` tuple (provenance for debugging)

    Construction:

    - ``trajectories``: same shape the existing ``ARCTrajectoryDataset`` takes
    - ``sequence_len``: T (must be ≥ 2)
    - ``pixel_diff_threshold``: drop windows where ``pixel_diff(o_0, o_T)`` is
      below this. Default 1 (any change at all).
    - ``cap_per_game``: optional limit on windows per game_id (for balanced
      sampling during ablation). ``None`` = no cap.
    - ``family_index``: optional ``{game_id: mechanic_family_id}`` map (the
      ``game_to_family_id`` field of a family_conditioned_dataset family_index
      sidecar). When provided, ``__getitem__`` emits ``mechanic_family_id``
      (int; ``-1`` for game_ids absent from the map). Default ``None`` leaves
      the output dict unchanged (backward-compatible).
    """

    def __init__(
        self,
        trajectories: list[ARCTrajectory],
        *,
        sequence_len: int = 8,
        pixel_diff_threshold: int = 1,
        cap_per_game: Optional[int] = None,
        frame_transform: Callable[[np.ndarray], torch.Tensor] = frame_to_onehot,
        family_index: Optional[dict[str, int]] = None,
    ) -> None:
        if sequence_len < 2:
            raise ValueError("sequence_len must be >= 2")
        if pixel_diff_threshold < 0:
            raise ValueError("pixel_diff_threshold must be >= 0")

        self.sequence_len = sequence_len
        self.pixel_diff_threshold = pixel_diff_threshold
        self.trajectories = trajectories
        self.family_index = family_index
        # Pluggable frame encoder: defaults to frame_to_onehot (16-channel
        # one-hot, the iter_05b baseline). Pass frame_to_canonical_rgb for
        # the paper-faithful RGB variant (3-channel).
        self.frame_transform = frame_transform
        # windows: list of (t_idx, k, goal_step_idx) where goal_step_idx is
        # the absolute step index of o_T in traj.steps. Caching the index
        # avoids repeated arithmetic in __getitem__.
        self.windows: list[tuple[int, int, int]] = []
        per_game_counts: dict[str, int] = {}

        # Stats for the iter-3 check JSON.
        self.stats: dict[str, int] = {
            "n_candidate_windows": 0,
            "n_filtered_trivial": 0,         # pixel_diff < threshold
            "n_filtered_terminal": 0,        # window crossed an episode boundary
            "n_filtered_cap": 0,             # per-game cap reached
            "n_emitted": 0,
        }

        T = sequence_len
        for t_idx, traj in enumerate(trajectories):
            n = len(traj.steps)
            for k in range(1, n - T + 1):
                self.stats["n_candidate_windows"] += 1
                # Reject if any obs-span step is terminal (matches ARCTrajectoryDataset).
                if not ARCTrajectoryDataset._window_valid(traj, k, T):
                    self.stats["n_filtered_terminal"] += 1
                    continue
                o_0 = traj.steps[k - 1].frame
                # o_T = the frame AFTER the last action in the window
                o_T_step = k + T - 1
                if o_T_step >= n:
                    self.stats["n_filtered_terminal"] += 1
                    continue
                o_T = traj.steps[o_T_step].frame
                if _pixel_diff(o_0, o_T) < pixel_diff_threshold:
                    self.stats["n_filtered_trivial"] += 1
                    continue
                if cap_per_game is not None:
                    used = per_game_counts.get(traj.game_id, 0)
                    if used >= cap_per_game:
                        self.stats["n_filtered_cap"] += 1
                        continue
                    per_game_counts[traj.game_id] = used + 1
                self.windows.append((t_idx, k, o_T_step))
                self.stats["n_emitted"] += 1

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        t_idx, k, goal_idx = self.windows[idx]
        traj = self.trajectories[t_idx]
        T = self.sequence_len
        obs_steps = traj.steps[k - 1 : k + T - 1]
        action_steps = traj.steps[k : k + T]
        goal_step = traj.steps[goal_idx]

        obs = torch.stack([self.frame_transform(s.frame) for s in obs_steps])
        action_id = torch.tensor([s.action_id for s in action_steps], dtype=torch.long)
        action_x = torch.tensor([s.action_x for s in action_steps], dtype=torch.long)
        action_y = torch.tensor([s.action_y for s in action_steps], dtype=torch.long)
        action_complex_mask = torch.tensor(
            [1.0 if s.action_complex else 0.0 for s in action_steps],
            dtype=torch.float,
        )
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
        goal_frame = self.frame_transform(goal_step.frame)
        goal_pixel_diff = _pixel_diff(obs_steps[0].frame, goal_step.frame)

        item = {
            "obs": obs,
            "action_id": action_id,
            "action_x": action_x,
            "action_y": action_y,
            "action_complex_mask": action_complex_mask,
            "reward": reward,
            "goal_frame": goal_frame,
            "goal_pixel_diff": goal_pixel_diff,
            "game_id": traj.game_id,
            "window_origin": (t_idx, k),
            # Negative-class flag: win_levels==0 => bad run (loss/timeout);
            # >0 => winning trajectory. Lets a downstream head distinguish
            # success vs failure dynamics even though hindsight relabels every
            # transition as positive-for-its-achieved-goal.
            "is_win": bool(traj.win_levels > 0),
        }
        if self.family_index is not None:
            item["mechanic_family_id"] = int(
                self.family_index.get(traj.game_id, -1)
            )
        return item


__all__ = ["HindsightARCDataset"]
