"""ChapterDataset — training-wheels BC over cached winning chapters.

Drop-in sibling of :class:`lewm.data.hindsight.HindsightARCDataset`. Same
``__getitem__`` shape so the v2 trainer's ``collate()`` works unchanged.

Source: ``chapters.pt`` produced by
``scripts/extract_chapters_from_wins.py`` — 234 chapters (each
chapter = one winning sub-trajectory ending at a level-up event).
Median chapter length 40 actions, p90 237.

Why training-wheels:
- Hindsight on iter-0 random data ≈ 0.056% real-positive density.
- Hindsight on chapters = 100% of windows lead toward an ACTUAL winning
  terminal state (the chapter's last frame = the env's level-1-clear).
- The model gets dense, real supervised signal instead of self-supervised
  "predict whatever the random policy did next."

Each item:
- ``obs``:                 [T, C, 64, 64] (C=16 one-hot or 3 RGB per transform)
- ``action_id``:           [T] long
- ``action_x``, ``action_y``: [T] long
- ``action_complex_mask``: [T] float
- ``reward``:              [T] float (compute_dense_reward over (obs_t, obs_{t+1}))
- ``goal_frame``:          [C, 64, 64] — the WINDOW'S last frame (the
                             actual short-horizon next-step target,
                             matching HindsightARCDataset semantics —
                             varies per window so the encoder doesn't
                             collapse onto a single fixed terminal).
- ``game_id``:             str (provenance)
- ``window_origin``:       (chapter_idx, window_start) — provenance
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from lewm.data.arc_agi_3 import (
    ARC_GRID_SIZE,
    compute_dense_reward,
    frame_to_onehot,
)


class ChapterDataset(Dataset):
    """Drop-in compatible with HindsightARCDataset for the v2 trainer.

    Loads chapters.pt (produced by extract_chapters_from_wins.py) and
    emits sliding windows over each chapter. The goal_frame is the
    chapter's terminal (level-up) frame — every window in the chapter
    shares that goal, giving the model dense supervision toward known
    winning states.
    """

    def __init__(
        self,
        chapters_pt_path: str | Path,
        *,
        sequence_len: int = 8,
        cap_per_game: Optional[int] = None,
        frame_transform: Callable[[np.ndarray], torch.Tensor] = frame_to_onehot,
        chapter_kind_filter: Optional[set[str]] = None,
    ) -> None:
        if sequence_len < 2:
            raise ValueError("sequence_len must be >= 2")
        self.sequence_len = sequence_len
        self.frame_transform = frame_transform

        blob = torch.load(Path(chapters_pt_path), map_location="cpu", weights_only=False)
        if blob.get("schema") != "lewm.chapter_dataset.v1":
            raise ValueError(
                f"unexpected schema {blob.get('schema')!r}; "
                f"expected 'lewm.chapter_dataset.v1' (from extract_chapters_from_wins.py)"
            )
        chapters = blob["chapters"]
        if chapter_kind_filter is not None:
            chapters = [c for c in chapters if c["chapter_kind"] in chapter_kind_filter]

        # Per-game cap (matches HindsightARCDataset's cap_per_game semantics).
        if cap_per_game is not None:
            per_game_used: dict[str, int] = {}
            filtered: list[dict] = []
            for c in chapters:
                gid = c["game_id"]
                if per_game_used.get(gid, 0) >= cap_per_game:
                    continue
                per_game_used[gid] = per_game_used.get(gid, 0) + 1
                filtered.append(c)
            chapters = filtered

        self.chapters = chapters
        # Build window index: (chapter_idx, window_start_k) for each valid window.
        # A window covers obs_steps = chapter.frames[k-1 .. k+T-2] (length T)
        # and action_steps = chapter.actions[k .. k+T-1] (length T).
        # Requires k >= 1 (need a "before" frame) and k+T-1 < len(chapter.actions).
        self.windows: list[tuple[int, int]] = []
        for c_idx, c in enumerate(chapters):
            n_actions = len(c["actions"])
            # Need: k >= 1 AND k + T - 1 < n_actions  →  k in [1, n_actions - T]
            for k in range(1, max(1, n_actions - sequence_len + 1)):
                self.windows.append((c_idx, k))

        self.stats = {
            "n_chapters": len(chapters),
            "n_windows": len(self.windows),
            "sequence_len": sequence_len,
        }

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        c_idx, k = self.windows[idx]
        c = self.chapters[c_idx]
        T = self.sequence_len
        # frames[i] is the state AFTER actions[i]. So frames[k-1] is BEFORE
        # actions[k] — the obs the model sees at window position 0.
        obs_frames = c["frames"][k - 1 : k - 1 + T]            # [T] frames (BEFORE each action)
        next_frames = c["frames"][k : k + T]                   # [T] frames (AFTER each action — used for reward)
        actions = c["actions"][k : k + T]                      # [T] action dicts

        obs = torch.stack([self.frame_transform(np.asarray(f)) for f in obs_frames])
        action_id = torch.tensor([int(a.get("id", 0)) for a in actions], dtype=torch.long)
        action_x = torch.tensor([int(a.get("x", 0)) for a in actions], dtype=torch.long)
        action_y = torch.tensor([int(a.get("y", 0)) for a in actions], dtype=torch.long)
        action_complex_mask = torch.tensor(
            [1.0 if a.get("complex") else 0.0 for a in actions],
            dtype=torch.float,
        )
        # Dense reward per step (same as HindsightARCDataset). Chapter context
        # makes level_up flag available; we use a coarse approximation: True on
        # the chapter's terminal action only.
        chapter_end_step = c["end_step"] - c["start_step"]  # local step index within chapter
        rewards = []
        for t in range(T):
            local_t = k + t  # global within chapter
            level_up = (local_t == chapter_end_step)
            rewards.append(compute_dense_reward(
                np.asarray(obs_frames[t]), np.asarray(next_frames[t]),
                level_up=level_up,
            ))
        reward = torch.tensor(rewards, dtype=torch.float32)
        # Goal frame: the WINDOW'S LAST frame (matches HindsightARCDataset
        # semantics). Earlier version used the chapter's terminal which gave
        # every window the same goal → encoder collapsed onto the terminal
        # state and val_top1 plateaued at 0.21 (vs iter_05b 0.486).
        goal_frame = self.frame_transform(np.asarray(next_frames[-1]))

        return {
            "obs": obs,
            "action_id": action_id,
            "action_x": action_x,
            "action_y": action_y,
            "action_complex_mask": action_complex_mask,
            "reward": reward,
            "goal_frame": goal_frame,
            "game_id": c["game_id"],
            "window_origin": (c_idx, k),
        }


__all__ = ["ChapterDataset"]
