"""Transition dataset for MegarexusCNN.

Sources (ARC_SCORECARD_EVIDENCE stays untouched — this is training data,
not evidence of solving anything):

1. Human corpus recordings (corpus/s5i5_snake_modulo, corpus/sp80_liquid_pour):
   PNG frames decoded via the canonical ARC palette + manifest.jsonl actions.
   Verified alignment convention: row t's action transitions frame_t →
   frame_{t+1} (checked against SP80's known directional display deltas —
   see notes/megarexus_cnn_scout.md).
2. Optional locally collected SP80 rollout shards (.pt written by
   orchestration/collect_sp80_rollouts.py) using the same convention.

Transitions never cross episode boundaries, level changes, or
non-consecutive frame indices. The validation split is held out at the
episode/level granularity (no within-episode leakage):

- sp80_liquid_pour episode_005 → val
- s5i5_snake_modulo final level → val
- rollout shards → train only

Each item:
- ``frame``:      [16 + 6, 64, 64] float32 (one-hot + Contra planes)
- ``next_frame``: [16 + 6, 64, 64] float32 (input for inverse dynamics)
- ``target``:     [64, 64] long (next-frame color indices)
- ``action_id``, ``x``, ``y``: long scalars
- ``goose_idx``:  long (index in the 4101 Goose head; -1 if action outside 1..6)
- ``changed``:    float scalar (did the frame change?)
- ``is_human``:   float scalar (behavior-cloning mask)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_LAB_ROOT = Path(__file__).resolve().parents[2]
_REPO_ROOT = _LAB_ROOT.parent
if str(_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAB_ROOT))

from engine_adapters.contra_perception_adapter import N_PLANES, contra_planes  # noqa: E402

GRID = 64
NUM_COLORS = 16

# canonical ARC-AGI-3 palette (arc_agi.rendering.COLOR_MAP; also in
# architecture/src/lewm/data/arc_agi_3.py and corpus/CORPUS_SPEC.md)
PALETTE = np.array([
    [255, 255, 255], [204, 204, 204], [153, 153, 153], [102, 102, 102],
    [51, 51, 51], [0, 0, 0], [229, 58, 163], [255, 123, 204],
    [249, 60, 49], [30, 147, 255], [136, 216, 241], [255, 220, 0],
    [255, 133, 27], [146, 18, 49], [79, 204, 48], [163, 86, 214],
], dtype=np.uint8)

_RGB_LUT: np.ndarray | None = None


def _rgb_lut() -> np.ndarray:
    global _RGB_LUT
    if _RGB_LUT is None:
        lut = np.full((256, 256, 256), -1, dtype=np.int16)
        for i, c in enumerate(PALETTE):
            lut[c[0], c[1], c[2]] = i
        _RGB_LUT = lut
    return _RGB_LUT


def decode_frame_png(path: Path) -> np.ndarray:
    """Corpus PNG → [64, 64] int8 color-index grid (lossless, exact palette)."""
    img = np.array(Image.open(path).convert("RGB"))
    if img.shape != (GRID, GRID, 3):
        raise ValueError(f"{path}: expected (64, 64, 3), got {img.shape}")
    idx = _rgb_lut()[img[..., 0], img[..., 1], img[..., 2]]
    if (idx < 0).any():
        raise ValueError(f"{path}: {int((idx < 0).sum())} pixels off-palette")
    return idx.astype(np.int8)


@dataclass
class Transition:
    frame_key: str
    next_frame_key: str
    action_id: int
    x: int
    y: int
    is_human: bool
    split: str  # "train" | "val"
    prev_frame_key: str | None = None  # same-episode predecessor for stacking
    glove_bbox: tuple | None = None  # PORTBAND glove at a catch-play pour step
    catch: float | None = None  # hindsight label: did the pour touch the glove?
    # learned outcome semantics (backlog Tier 1/3) — labels from pixels and
    # environment fields ONLY; no game-specific mechanic model anywhere
    level_up: float | None = None  # pour rows: did levels_completed increment?
    settle_key: str | None = None  # pour rows: frame after the pour settles
    control_bbox: tuple | None = None  # move rows: candidate component bbox
    controlled: float | None = None  # did that component displace with the move?
    # False when next_frame belongs to a NEW layout (a level-completing pour):
    # the row keeps its labels but must not teach dynamics/inverse/change
    dyn_valid: bool = True


def _episode_transitions(
    rows: list[dict],
    frame_key_of: "callable",
    is_human: bool,
    split_of: "callable",
) -> list[Transition]:
    out = []
    for i, (a, b) in enumerate(zip(rows, rows[1:])):
        if a.get("level") != b.get("level"):
            continue  # level change: next frame is a new layout, not a dynamics step
        prev = rows[i - 1] if i > 0 and rows[i - 1].get("level") == a.get("level") else None
        act = a["action"]
        aid = int(act["id"])
        out.append(
            Transition(
                frame_key=frame_key_of(a),
                next_frame_key=frame_key_of(b),
                action_id=aid,
                x=int(act.get("x") or 0),
                y=int(act.get("y") or 0),
                is_human=is_human,
                split=split_of(a),
                prev_frame_key=frame_key_of(prev) if prev is not None else None,
            )
        )
    return out


def load_corpus_transitions(corpus_root: Path) -> list[Transition]:
    transitions: list[Transition] = []

    # ── sp80_liquid_pour: 6 episodes, full ACTION1-6 mix ──
    sp80 = corpus_root / "sp80_liquid_pour"
    rows = [json.loads(l) for l in open(sp80 / "manifest.jsonl")]
    by_ep: dict[str, list[dict]] = {}
    for r in rows:
        by_ep.setdefault(r["image"].split("/")[1], []).append(r)
    for ep, ep_rows in sorted(by_ep.items()):
        ep_rows.sort(key=lambda r: r["episode_frame_index"])
        split = "val" if ep == "episode_005" else "train"
        transitions += _episode_transitions(
            ep_rows,
            frame_key_of=lambda r, _sp=sp80: str(_sp / r["image"]),
            is_human=True,
            split_of=lambda r, _s=split: _s,
        )

    # ── s5i5_snake_modulo: one playthrough, click-only; final level → val ──
    snake = corpus_root / "s5i5_snake_modulo"
    rows = [json.loads(l) for l in open(snake / "manifest.jsonl")]
    rows.sort(key=lambda r: r["frame"])
    max_level = max(r["level"] for r in rows)
    transitions += _episode_transitions(
        rows,
        frame_key_of=lambda r, _sn=snake: str(_sn / r["image"]),
        is_human=True,
        split_of=lambda r: "val" if r["level"] == max_level else "train",
    )
    return transitions


MOVE_ACTION_AXIS = {1: (1, -1), 2: (1, +1), 3: (0, -1), 4: (0, +1)}  # aid → (axis, sign)
MAX_CONTROL_NEGATIVES = 2  # per move transition, alongside every positive


def _frame_level(steps: list[dict], j: int) -> int:
    """levels_completed of the LAYOUT shown in row j's stored frame.

    Row j stores the frame BEFORE action j together with the POST-action
    env fields, so the frame's layout belongs to the PREVIOUS row's
    fields (reset frames are level 0). Comparing rows j/j+1 directly —
    as this loader did before outcome_v6 — tests "did action j complete
    a level", which is the level_up LABEL: it leaked into prev-frame
    selection and self-stacked exactly the positive pour rows."""
    return int(steps[j - 1].get("levels_completed", 0) or 0) if j > 0 else 0


def _settle_index(steps: list[dict], pour_i: int) -> int:
    """Index of the settled frame after the pour: the first non-tick row
    (its stored frame is the pour's final animation frame) or episode end."""
    j = pour_i + 1
    while j < len(steps) - 1 and steps[j]["action_id"] == 0:
        j += 1
    return j


def _control_samples(
    base: Transition, pre: np.ndarray, post: np.ndarray, aid: int,
    rng: np.random.Generator,
) -> list[Transition]:
    """Hindsight controllability labels: which component displaced with the
    move? Positive = axis displacement >= 0.4 in the action direction with
    off-axis <= 1.0 (the fixed glove-probe rule); negative = did not."""
    from engine_adapters.portband_glove_adapter import (
        component_displacement,
        frame_components,
    )

    axis, sign = MOVE_ACTION_AXIS[aid]
    positives, negatives = [], []
    for comp in frame_components(pre):
        moved = component_displacement(pre, post, comp)
        if moved is None:
            ok = False
        else:
            on_axis, off_axis = moved[axis], moved[1 - axis]
            ok = on_axis * sign >= 0.4 and abs(off_axis) <= 1.0
        (positives if ok else negatives).append(
            replace(base, control_bbox=tuple(comp["bbox_xyxy"]), controlled=float(ok))
        )
    if len(negatives) > MAX_CONTROL_NEGATIVES:
        keep = rng.choice(len(negatives), size=MAX_CONTROL_NEGATIVES, replace=False)
        negatives = [negatives[k] for k in keep]
    return positives + negatives


def load_rollout_transitions(
    shard_paths: list[Path], split: str = "train"
) -> list[Transition]:
    """Load .pt shards from the orchestration collectors.

    Catch-play shards additionally carry per-pour records (glove bbox +
    hindsight caught label), attached to their ACTION5 transitions. Every
    pour row also gets level_up (environment fields) + settle_key (pixels)
    labels; every move row spawns controllability samples.
    """
    transitions: list[Transition] = []
    rng = np.random.default_rng(0)
    for path in shard_paths:
        for traj in torch.load(path, weights_only=False):
            steps = traj["steps"]
            key_of = lambda j: f"{path}::{traj['rollout_id']}::{j}"  # noqa: E731
            catch_by_index = {
                int(c["pour_step_index"]): c for c in traj.get("catches", [])
            }
            for i, (a, b) in enumerate(zip(steps, steps[1:])):
                aid = int(a["action_id"])
                level_up = settle_key = None
                if aid == 5:
                    prev_lvl = steps[i - 1].get("levels_completed", 0) if i > 0 else 0
                    level_up = float((a.get("levels_completed") or 0) > (prev_lvl or 0))
                    if not level_up:  # a level change makes the settle frame a new layout
                        settle_key = key_of(_settle_index(steps, i))
                # Pair inclusion matches v4/v5: the completing pour's next
                # frame is the first ANIMATION frame (still the old layout) —
                # a valid 1-step target; the real scene swap happens inside
                # the tick rows, which the env fields cannot localize.
                # Masking the pour pair instead (tried in outcome_v6) left
                # pour-at-completion imagination unsupervised: it extrapolated
                # a giant scene change and the level probe collapsed
                # 20/20 → 0/20 on decline:gauge (imagined drain 102 > 64).
                if a.get("levels_completed") != b.get("levels_completed"):
                    continue  # drops only the pre-pour move row, as in v4/v5
                prev_ok = (
                    i > 0
                    and steps[i - 1].get("levels_completed") == a.get("levels_completed")
                )
                if level_up:
                    # the leakage fix (the ONE change vs v5): frame i is
                    # still the OLD layout, so its true predecessor is row
                    # i-1's frame. The field test above compares POST-action
                    # levels and fails exactly on positives, self-stacking
                    # [cur, cur] — the label leaked into the encoding.
                    prev_ok = (i > 0 and
                               _frame_level(steps, i - 1) == _frame_level(steps, i))
                pour = catch_by_index.get(i)
                base = Transition(
                    frame_key=key_of(i),
                    next_frame_key=key_of(i + 1),
                    action_id=aid,
                    x=int(a["action_x"]),
                    y=int(a["action_y"]),
                    is_human=False,
                    split=split,
                    prev_frame_key=key_of(i - 1) if prev_ok else None,
                    glove_bbox=tuple(pour["glove_bbox_xyxy"]) if pour else None,
                    catch=float(pour["caught"]) if pour else None,
                    level_up=level_up,
                    settle_key=settle_key,
                )
                transitions.append(base)
                # multi-object rosters: one catch sample per non-active glove
                # (same frames, different glove plane + label)
                for row in (pour or {}).get("roster", []):
                    if row.get("active"):
                        continue
                    transitions.append(replace(
                        base,
                        glove_bbox=tuple(row["glove_bbox_xyxy"]),
                        catch=float(row["caught"]),
                        level_up=None, settle_key=None,  # labeled once, on base
                    ))
                if aid in MOVE_ACTION_AXIS:
                    pre = np.asarray(a["frame"], dtype=np.int8)
                    post = np.asarray(b["frame"], dtype=np.int8)
                    transitions += _control_samples(base, pre, post, aid, rng)
    return transitions


FLOW_CHANGED_CELLS = 20  # transitions with >= this many changed cells count as "flow"


class MegarexusTransitions(Dataset):
    def __init__(
        self,
        split: str,
        corpus_root: Path | None = None,
        rollout_shards: list[Path] | None = None,
        val_rollout_shards: list[Path] | None = None,
        n_input_frames: int = 1,
        include_corpus: bool = True,
        verbose: bool = True,
    ) -> None:
        self.n_input_frames = n_input_frames
        corpus_root = corpus_root or (_REPO_ROOT / "corpus")
        transitions = load_corpus_transitions(corpus_root) if include_corpus else []

        # frame cache: key → int8 grid; corpus keys are PNG paths,
        # rollout keys index into the shard's stored frames
        self.grids: dict[str, np.ndarray] = {}
        for t in transitions:
            for key in (t.frame_key, t.next_frame_key, t.prev_frame_key):
                if key is not None and key not in self.grids:
                    self.grids[key] = decode_frame_png(Path(key))

        for path in list(rollout_shards or []) + list(val_rollout_shards or []):
            for traj in torch.load(path, weights_only=False):
                for i, step in enumerate(traj["steps"]):
                    self.grids[f"{path}::{traj['rollout_id']}::{i}"] = np.asarray(
                        step["frame"], dtype=np.int8
                    )
        if rollout_shards:
            transitions += load_rollout_transitions(rollout_shards, split="train")
        if val_rollout_shards:
            transitions += load_rollout_transitions(val_rollout_shards, split="val")

        self.transitions = [t for t in transitions if t.split == split]

        # CONTRA_THINKING_ENGINE: precompute perception planes per unique frame
        self.planes: dict[str, np.ndarray] = {}
        needed = {
            k
            for t in self.transitions
            for k in (t.frame_key, t.next_frame_key, t.prev_frame_key)
            if k is not None
        }
        for key in needed:
            self.planes[key] = contra_planes(self.grids[key]).astype(np.float16)

        if verbose:
            n_human = sum(t.is_human for t in self.transitions)
            print(
                f"[dataset] split={split}: {len(self.transitions)} transitions "
                f"({n_human} human, {len(self.transitions) - n_human} rollout), "
                f"{len(needed)} unique frames"
            )

    def __len__(self) -> int:
        return len(self.transitions)

    def _encode(self, key: str) -> torch.Tensor:
        grid = torch.from_numpy(self.grids[key].astype(np.int64))
        onehot = torch.nn.functional.one_hot(grid, NUM_COLORS).permute(2, 0, 1).float()
        planes = torch.from_numpy(self.planes[key].astype(np.float32))
        return torch.cat([onehot, planes], dim=0)

    def _encode_stack(self, key: str, prev_key: str | None) -> torch.Tensor:
        """Current frame first (channels [0, 22)), then the previous frame."""
        cur = self._encode(key)
        if self.n_input_frames == 1:
            return cur
        return torch.cat([cur, self._encode(prev_key or key)], dim=0)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        t = self.transitions[i]
        cur = self.grids[t.frame_key]
        nxt = self.grids[t.next_frame_key]
        if 1 <= t.action_id <= 5:
            goose_idx = t.action_id - 1
        elif t.action_id == 6:
            goose_idx = 5 + t.y * GRID + t.x
        else:
            goose_idx = -1
        # PORTBAND glove plane (coarse 16x16 bbox mask) for the catch head
        glove_plane = torch.zeros(1, GRID // 4, GRID // 4)
        if t.glove_bbox is not None:
            x0, y0, x1, y1 = (int(v) // 4 for v in t.glove_bbox)
            glove_plane[0, y0:y1 + 1, x0:x1 + 1] = 1.0
        # controllability candidate plane (same coarse-bbox convention)
        control_plane = torch.zeros(1, GRID // 4, GRID // 4)
        if t.control_bbox is not None:
            x0, y0, x1, y1 = (int(v) // 4 for v in t.control_bbox)
            control_plane[0, y0:y1 + 1, x0:x1 + 1] = 1.0
        # settle mask: cells still different once the pour finished (pixels)
        settle_mask = torch.zeros(GRID, GRID)
        if t.settle_key is not None:
            settle_mask = torch.from_numpy(
                (self.grids[t.settle_key] != cur).astype(np.float32)
            )
        return {
            "frame": self._encode_stack(t.frame_key, t.prev_frame_key),
            "next_frame": self._encode_stack(t.next_frame_key, t.frame_key),
            "target": torch.from_numpy(nxt.astype(np.int64)),
            "glove_plane": glove_plane,
            "catch": torch.tensor(t.catch if t.catch is not None else 0.0),
            "has_catch": torch.tensor(float(t.catch is not None)),
            "control_plane": control_plane,
            "controlled": torch.tensor(t.controlled if t.controlled is not None else 0.0),
            # control rows are auxiliary duplicates of their move transition:
            # they carry ONLY the controllability objective (main losses gate
            # on has_control == 0 so move dynamics aren't over-weighted)
            "has_control": torch.tensor(float(t.controlled is not None)),
            "settle_mask": settle_mask,
            "has_settle": torch.tensor(float(t.settle_key is not None)),
            "level_up": torch.tensor(t.level_up if t.level_up is not None else 0.0),
            "has_level": torch.tensor(float(t.level_up is not None)),
            "action_id": torch.tensor(t.action_id, dtype=torch.long),
            "x": torch.tensor(t.x, dtype=torch.long),
            "y": torch.tensor(t.y, dtype=torch.long),
            "goose_idx": torch.tensor(goose_idx, dtype=torch.long),
            "changed": torch.tensor(float((cur != nxt).any())),
            "flow": torch.tensor(float(int((cur != nxt).sum()) >= FLOW_CHANGED_CELLS)),
            "is_human": torch.tensor(float(t.is_human)),
            "dyn_valid": torch.tensor(float(t.dyn_valid)),
        }


assert N_PLANES == 6, "model config n_contra_planes must match the adapter"
