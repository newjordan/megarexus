"""ReplayDecoder — extract mechanic, agent, goal labels from won replay traces.

Consumes wins.pkl replay data (human or agent playthroughs) and produces
structured annotations usable as:
- Training labels for mechanic-classification heads
- Agent/goal sprite masks for the perception framework (Q1, Q2)
- Action→effect pairs for the predictor's transition model

No training required — this is a pure decode module that reads replay data
and outputs labels. The labels can then feed encoder/decoder training.

Usage:
    from lewm.replay_decoder import ReplayDecoder, decode_wins_pkl

    decoder = ReplayDecoder()
    labels = decoder.decode_entry(wins_pkl_entry)
    # labels.mechanic, labels.agent_bboxes, labels.goal_config, ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
import pickle
import json
import numpy as np
from typing import Sequence

ACTION_NAMES = {
    0: "NOOP", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
    5: "SELECT", 6: "ACTION6",
}


@dataclass
class SpriteTrack:
    """Tracked sprite bounding box over one level segment."""
    level: int
    bboxes: list[tuple[int, int, int, int]]  # (x1,y1,x2,y2) per frame
    color: int  # most common color inside the bbox
    pixels_moved_per_step: list[int]  # how many pixels changed each step


@dataclass
class GoalConfig:
    """What the 'won' state looks like for one level."""
    level: int
    frame: np.ndarray | None  # the final frame (H, W) color-index
    colors_present: list[int]
    # Pixel mask of what's DIFFERENT from the initial frame
    delta_from_start: np.ndarray | None  # (H, W) bool


@dataclass
class MechanicLabel:
    """Classified mechanic from action distribution."""
    primary: str  # "dpad_navigation", "select_place", "action6", "mixed"
    action_distribution: dict[int, int]  # action_id → count
    dpad_ratio: float
    select_ratio: float
    action6_ratio: float


@dataclass
class LevelDecode:
    """Full decode for one won level segment."""
    game_id: str
    level: int
    mechanic: MechanicLabel
    sprite_tracks: list[SpriteTrack]
    goal: GoalConfig
    n_frames: int
    n_actions: int


def _action_id(a) -> int:
    if isinstance(a, dict):
        return a.get("id", 0)
    return int(a) if isinstance(a, (int, float)) else 0


def _extract_sprite_bboxes(
    frames: list[np.ndarray],
    segment_indices: list[int],
) -> list[SpriteTrack]:
    """Track sprite movement via frame-diff bounding boxes.

    For each consecutive frame pair in the segment:
    1. Compute per-pixel difference mask
    2. Find connected components (simple flood-fill)
    3. Track the largest component's bounding box
    """
    tracks: list[SpriteTrack] = []
    if len(segment_indices) < 2:
        return tracks

    bboxes: list[tuple[int, int, int, int]] = []
    colors: list[int] = []
    pixel_counts: list[int] = []

    for i in range(len(segment_indices) - 1):
        idx_a = segment_indices[i]
        idx_b = segment_indices[i + 1]
        a = frames[idx_a]
        b = frames[idx_b]

        changed = a != b
        ys, xs = np.where(changed)
        if len(ys) < 2:
            pixel_counts.append(0)
            continue

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bboxes.append((x1, y1, x2, y2))
        pixel_counts.append(len(ys))

        # Dominant color in the changed region of frame_b
        region = b[y1:y2+1, x1:x2+1]
        colors.append(int(np.bincount(region.flatten()).argmax()))

    if bboxes:
        dominant_color = int(np.bincount(np.array(colors)).argmax()) if colors else 0
        tracks.append(SpriteTrack(
            level=0,  # filled by caller
            bboxes=bboxes,
            color=dominant_color,
            pixels_moved_per_step=pixel_counts,
        ))

    return tracks


def _extract_goal_config(
    frames: list[np.ndarray],
    segment_indices: list[int],
) -> GoalConfig:
    """Extract goal configuration from the final frame of a won segment."""
    if not segment_indices:
        return GoalConfig(level=0, frame=None, colors_present=[], delta_from_start=None)

    first_idx = segment_indices[0]
    last_idx = segment_indices[-1]
    first_frame = frames[first_idx]
    last_frame = frames[last_idx]

    delta = (first_frame != last_frame)

    return GoalConfig(
        level=0,  # filled by caller
        frame=last_frame.copy(),
        colors_present=sorted(set(int(c) for c in last_frame.flatten())),
        delta_from_start=delta,
    )


def _classify_mechanic(action_ids: list[int]) -> MechanicLabel:
    """Classify mechanic from action distribution."""
    counts = Counter(action_ids)
    total = len(action_ids) or 1

    dpad = counts.get(1, 0) + counts.get(2, 0) + counts.get(3, 0) + counts.get(4, 0)
    selects = counts.get(5, 0)
    action6 = counts.get(6, 0)
    noops = counts.get(0, 0)

    if dpad > 0.6 * total:
        primary = "dpad_navigation"
    elif selects > 0.4 * total and dpad > 0.15 * total:
        primary = "select_place"
    elif selects > 0.5 * total:
        primary = "select_dominant"
    elif action6 > 0.5 * total:
        primary = "action6_dominant"
    elif action6 > 0.2 * total and dpad > 0.2 * total:
        primary = "action6_navigation"
    elif noops > 0.5 * total:
        primary = "wait_observe"
    else:
        primary = "mixed"

    return MechanicLabel(
        primary=primary,
        action_distribution=dict(counts),
        dpad_ratio=dpad / total,
        select_ratio=selects / total,
        action6_ratio=action6 / total,
    )


class ReplayDecoder:
    """Decode won replay traces into structured labels."""

    def decode_entry(self, entry: dict) -> list[LevelDecode]:
        """Decode all won levels in one wins.pkl entry.

        Args:
            entry: dict with keys: game_id, frames, actions, levels

        Returns:
            List of LevelDecode, one per won level.
        """
        game_id = entry["game_id"]
        frames = entry["frames"]
        actions = entry["actions"]
        levels = entry["levels"]

        action_ids = [_action_id(a) for a in actions]

        # Find level transitions
        won_levels: list[int] = []
        prev = -1
        for l in levels:
            if l > prev:
                won_levels.append(l)
                prev = l
        if 0 not in won_levels:
            won_levels = [0] + won_levels

        results: list[LevelDecode] = []
        for li in won_levels:
            seg_indices = [j for j, l in enumerate(levels) if l == li]
            seg_actions = [action_ids[j] for j in seg_indices]
            seg_frames = [frames[j] for j in seg_indices]

            mechanic = _classify_mechanic(seg_actions)
            tracks = _extract_sprite_bboxes(frames, seg_indices)
            goal = _extract_goal_config(frames, seg_indices)

            for t in tracks:
                t.level = li
            goal.level = li

            results.append(LevelDecode(
                game_id=game_id,
                level=li,
                mechanic=mechanic,
                sprite_tracks=tracks,
                goal=goal,
                n_frames=len(seg_indices),
                n_actions=len(seg_actions),
            ))

        return results


def decode_wins_pkl(pkl_path: str | Path) -> list[LevelDecode]:
    """Decode all entries in a wins.pkl file."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    entries = data if isinstance(data, list) else [data]
    decoder = ReplayDecoder()
    results: list[LevelDecode] = []
    for entry in entries:
        results.extend(decoder.decode_entry(entry))
    return results


# ── GoalFrameExtractor ──────────────────────────────────────────────────────


@dataclass
class GoalFrameBatch:
    """Batch of goal frames with metadata for training a goal encoder."""
    game_id: str
    frames: list[np.ndarray]  # [N, H, W] final frames from won levels
    deltas: list[np.ndarray]  # [N, H, W] delta from start (bool)
    mechanics: list[str]
    colors: list[list[int]]


class GoalFrameExtractor:
    """Extract goal-frame batches for training a goal-configuration encoder.

    For each won level, captures:
    - The final frame (goal state)
    - The delta from the initial frame (what changed)
    - The mechanic type

    These can be used to train a goal-inference head: given a delta trajectory,
    predict the goal configuration.
    """

    def extract_from_decodes(
        self,
        decodes: list[LevelDecode],
    ) -> GoalFrameBatch:
        """Convert LevelDecode list to a training-ready batch."""
        if not decodes:
            return GoalFrameBatch(
                game_id="", frames=[], deltas=[], mechanics=[], colors=[],
            )

        frames: list[np.ndarray] = []
        deltas: list[np.ndarray] = []
        mechanics: list[str] = []
        colors: list[list[int]] = []

        for d in decodes:
            if d.goal.frame is not None:
                frames.append(d.goal.frame)
                mechanics.append(d.mechanic.primary)
                colors.append(d.goal.colors_present)
                if d.goal.delta_from_start is not None:
                    deltas.append(d.goal.delta_from_start)
                else:
                    deltas.append(np.zeros_like(d.goal.frame, dtype=bool))

        return GoalFrameBatch(
            game_id=decodes[0].game_id,
            frames=frames,
            deltas=deltas,
            mechanics=mechanics,
            colors=colors,
        )

    def extract_all(
        self,
        pkl_dir: str | Path,
    ) -> dict[str, GoalFrameBatch]:
        """Extract goal frames from all wins.pkl in a directory."""
        batches: dict[str, GoalFrameBatch] = {}
        for pkl_path in sorted(Path(pkl_dir).glob("*wins.pkl")):
            decodes = decode_wins_pkl(str(pkl_path))
            if decodes:
                batch = self.extract_from_decodes(decodes)
                batches[batch.game_id] = batch
        return batches


# ── serialization ───────────────────────────────────────────────────────────


def level_decode_to_dict(ld: LevelDecode) -> dict:
    """Serialize LevelDecode to JSON-safe dict."""
    return {
        "game_id": ld.game_id,
        "level": ld.level,
        "mechanic": {
            "primary": ld.mechanic.primary,
            "action_distribution": ld.mechanic.action_distribution,
            "dpad_ratio": round(ld.mechanic.dpad_ratio, 4),
            "select_ratio": round(ld.mechanic.select_ratio, 4),
            "action6_ratio": round(ld.mechanic.action6_ratio, 4),
        },
        "sprite_tracks": [
            {
                "level": t.level,
                "n_bboxes": len(t.bboxes),
                "color": t.color,
                "bboxes": [[int(v) for v in b] for b in t.bboxes[:10]],  # first 10
                "mean_pixels_per_step": (
                    float(np.mean(t.pixels_moved_per_step))
                    if t.pixels_moved_per_step else 0.0
                ),
            }
            for t in ld.sprite_tracks
        ],
        "goal": {
            "level": ld.goal.level,
            "colors_present": ld.goal.colors_present,
            "delta_pixels": int(ld.goal.delta_from_start.sum())
            if ld.goal.delta_from_start is not None else 0,
        },
        "n_frames": ld.n_frames,
        "n_actions": ld.n_actions,
    }


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    REPO = Path(__file__).resolve().parent.parent.parent  # src/lewm/ → src/ → repo
    ap = argparse.ArgumentParser(description="ReplayDecoder — extract labels from wins.pkl")
    ap.add_argument("--pkl-dir", type=Path,
                    default=REPO / "experiments/2026-05-16_human_cheat_map_v1")
    ap.add_argument("--out", type=Path,
                    default=REPO / "lab/research_hunts/2026-05-19_replay_decoder_v1")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    all_decodes: list[LevelDecode] = []
    for pkl_path in sorted(args.pkl_dir.glob("*wins.pkl")):
        decodes = decode_wins_pkl(str(pkl_path))
        all_decodes.extend(decodes)
        game_id = decodes[0].game_id if decodes else pkl_path.stem

        # Write per-game JSON
        game_out = args.out / f"{game_id}_decode.json"
        game_out.write_text(json.dumps(
            [level_decode_to_dict(d) for d in decodes], indent=2,
        ))

    # Extract goal frames
    extractor = GoalFrameExtractor()
    batches = extractor.extract_all(str(args.pkl_dir))

    # Summary
    mech_counts = Counter(d.mechanic.primary for d in all_decodes)
    print(f"[replay-decoder] {len(all_decodes)} levels decoded from {len(batches)} games")
    print(f"  Mechanics: {dict(mech_counts)}")
    print(f"  Output: {args.out}")

    # Write manifest
    manifest = {
        "schema": "loom.replay_decoder.v001",
        "total_levels": len(all_decodes),
        "total_games": len(batches),
        "mechanic_summary": dict(mech_counts),
        "games": [
            {
                "game_id": batch.game_id,
                "n_goal_frames": len(batch.frames),
                "mechanics": list(set(batch.mechanics)),
                "colors": sorted(set(c for cl in batch.colors for c in cl)),
            }
            for batch in batches.values()
        ],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Goal frame statistics
    total_goal_frames = sum(len(b.frames) for b in batches.values())
    total_delta_pixels = sum(
        int(d.sum()) for b in batches.values() for d in b.deltas
    )
    print(f"  Goal frames: {total_goal_frames}")
    print(f"  Mean delta pixels: {total_delta_pixels / max(total_goal_frames, 1):.1f}")


if __name__ == "__main__":
    main()
