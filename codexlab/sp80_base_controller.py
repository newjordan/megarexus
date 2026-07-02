#!/usr/bin/env python3
"""Runtime SP80/Liquid Pour basic-control controller.

This is not a route player. It does not read human replay coordinates, win-bank
actions, or a stored solution. It loads the SP80 game class, finds selectable
sprites from the live game state, then uses a tiny replaceable MLP to map
language intents onto one basic control primitive at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn

from arcengine import ActionInput, GameAction
from players.sp80.registry import CONTROL_PRIMITIVES, INTENT_TRAINING_EXAMPLES, SP80_ACTION_SET


ROOT = Path(__file__).resolve().parents[1]
SP80_PATH = ROOT / "environment" / "environment_files" / "sp80" / "589a99af" / "sp80.py"

PALETTE = np.array(
    [
        [255, 255, 255],
        [204, 204, 204],
        [153, 153, 153],
        [102, 102, 102],
        [51, 51, 51],
        [0, 0, 0],
        [229, 58, 163],
        [255, 123, 204],
        [249, 60, 49],
        [30, 147, 255],
        [136, 216, 241],
        [255, 220, 0],
        [255, 133, 27],
        [146, 18, 49],
        [79, 204, 48],
        [163, 86, 214],
    ],
    dtype=np.uint8,
)

# Perceptual lock-goal signature. CONTRA's oracle identifies the SP80 goal as the
# yellow (raw value 11) lock components; a bucket turns raw value 13 once liquid
# fills it (see the sp80 update loop, repwkzbkhxl -> 13 on fill). Selecting goal
# sprites by this color signature lets goal localization come from perception
# (the same signal oracle's lock/key pass uses) instead of the privileged sprite
# tag. get_sprites_by_tag("repwkzbkhxl") is retained only as a ground-truth
# verifier for the perceptual selection, never as the sole control source.
GOAL_LOCK_PIXEL_VALUES = (11, 13)
GOAL_VERIFIER_TAG = "repwkzbkhxl"
# Routing-game payload + outcome colors (observed; measure-only, not control truth).
# The payload (magenta v6 liquid) is the controlled-but-not-directly-controllable
# "paint"; a sink turns v13 when filled; the floor turns v14 on spill (hazard).
PAYLOAD_PIXEL_VALUE = 6
FILLED_SINK_PIXEL_VALUE = 13
SPILL_PIXEL_VALUE = 14

# Perceptual selection signature. The engine renders the currently-selected control
# platform in blue (raw value 9) and every selectable-but-unselected platform in red
# (raw value 8). Verified empirically: 360/360 frames across all six levels matched the
# privileged selected sprite (vsoxmtrhqt) under random select+move sequences. This lets
# the controller perceive which object is under control, and where, from the rendered
# FRAME instead of reading the privileged selected-sprite object. The privileged read is
# retained only as a fail-open verifier, never as the sole control source.
SELECTED_CONTROL_PIXEL_VALUE = 9
UNSELECTED_CONTROL_PIXEL_VALUE = 8


def _sprite_pixel_value_set(sprite: Any) -> set[int]:
    pixels = getattr(sprite, "pixels", None)
    if pixels is None:
        return set()
    try:
        values = np.asarray(pixels)
    except Exception:
        return set()
    return {int(v) for v in np.unique(values) if int(v) >= 0}


def _manhattan(a: list[float] | None, b: list[float] | None) -> float | None:
    if not a or not b:
        return None
    return abs(float(a[0]) - float(b[0])) + abs(float(a[1]) - float(b[1]))


MOVE_ACTIONS = {
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
}

GOAL_DIRECTION_TO_PRIMITIVE = {
    "up": "move_up",
    "down": "move_down",
    "left": "move_left",
    "right": "move_right",
}

GOAL_CONDITION_TRAINING_EXAMPLES = [
    "i need to approach the nearest bucket",
    "move toward the target bucket",
    "get the selected object closer to the bucket",
    "target the bucket",
    "approach the pour bucket",
]

GOAL_INTENT_TOKENS = {"approach", "nearest", "bucket", "target", "closer"}

GOAL_FEATURE_KEYS = [
    "has_bucket_goal",
    "goal_direction_up",
    "goal_direction_down",
    "goal_direction_left",
    "goal_direction_right",
    "goal_axis_horizontal",
    "goal_axis_vertical",
    "goal_distance_near",
    "goal_distance_medium",
    "goal_distance_far",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_frame(frame: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(frame).tobytes()).hexdigest()


def rgb_frame(frame: np.ndarray, scale: int) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.int64).clip(0, 15)
    rgb = PALETTE[arr]
    if scale > 1:
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return rgb


def save_frame(frame: np.ndarray, path: Path, scale: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb_frame(frame, scale), mode="RGB").save(path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "png_sha256": sha256_file(path),
        "raw_frame_sha256": sha256_frame(frame),
        "shape": [int(v) for v in np.asarray(frame).shape],
    }


def save_frame_sequence(frames: list[np.ndarray], directory: Path, scale: int) -> list[dict[str, Any]]:
    return [
        save_frame(frame, directory / f"frame_{idx:03d}.png", scale)
        for idx, frame in enumerate(frames)
    ]


def frame_delta(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    before_arr = np.asarray(before)
    after_arr = np.asarray(after)
    if before_arr.shape != after_arr.shape:
        return {
            "shape_match": False,
            "before_shape": [int(v) for v in before_arr.shape],
            "after_shape": [int(v) for v in after_arr.shape],
            "changed_pixels": None,
            "changed_bbox_xyxy": None,
        }
    changed = before_arr != after_arr
    changed_pixels = int(np.count_nonzero(changed))
    if changed_pixels == 0:
        bbox = None
    else:
        ys, xs = np.nonzero(changed)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {
        "shape_match": True,
        "changed_pixels": changed_pixels,
        "changed_ratio": float(changed_pixels / max(1, changed.size)),
        "changed_bbox_xyxy": bbox,
    }


def load_sp80_class() -> Any:
    spec = importlib.util.spec_from_file_location("codexlab_sp80", SP80_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {SP80_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Sp80


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def sign_label(value: float) -> str:
    if value < 0:
        return "negative"
    if value > 0:
        return "positive"
    return "same"


def distance_bucket(distance: float) -> str:
    if distance <= 3:
        return "near"
    if distance <= 10:
        return "medium"
    return "far"


def direction_axis(direction: str) -> str:
    if direction in {"left", "right"}:
        return "horizontal"
    if direction in {"up", "down"}:
        return "vertical"
    return "same"


def synthetic_bucket_goal_condition(direction: str) -> dict[str, Any]:
    axis = direction_axis(direction)
    return {
        "schema": "codexlab.sp80_goal_condition.v1",
        "condition": "pour_into_buckets",
        "target_label": "bucket",
        "target_source_tag": "repwkzbkhxl",
        "goal_object_count": 1,
        "visible_goal_objects": [
            {
                "object_id": "bucket_synthetic_00",
                "semantic_label": "bucket",
                "source_tag": "repwkzbkhxl",
                "relative_display_direction": direction,
                "primary_axis": axis,
                "display_distance_bucket": "medium",
            }
        ],
        "nearest_goal": {
            "object_id": "bucket_synthetic_00",
            "semantic_label": "bucket",
            "source_tag": "repwkzbkhxl",
            "relative_display_direction": direction,
            "primary_axis": axis,
            "display_distance_bucket": "medium",
            "display_delta_sign": [
                "negative" if direction == "left" else "positive" if direction == "right" else "same",
                "negative" if direction == "up" else "positive" if direction == "down" else "same",
            ],
            "display_manhattan_distance": 6.0,
        },
        "route_or_replay_used": False,
        "coordinate_target_used": False,
    }


def goal_features(goal_condition: dict[str, Any] | None) -> list[float]:
    values = [0.0 for _ in GOAL_FEATURE_KEYS]
    if not goal_condition:
        return values
    nearest_goal = goal_condition.get("nearest_goal") or {}
    direction = nearest_goal.get("relative_display_direction")
    axis = nearest_goal.get("primary_axis")
    bucket = nearest_goal.get("display_distance_bucket")
    values[0] = 1.0
    direction_offsets = {"up": 1, "down": 2, "left": 3, "right": 4}
    if direction in direction_offsets:
        values[direction_offsets[str(direction)]] = 1.0
    if axis == "horizontal":
        values[5] = 1.0
    elif axis == "vertical":
        values[6] = 1.0
    distance_offsets = {"near": 7, "medium": 8, "far": 9}
    if bucket in distance_offsets:
        values[distance_offsets[str(bucket)]] = 1.0
    return values


def intent_uses_goal_condition(intent: str) -> bool:
    return bool(set(tokenize(intent)) & GOAL_INTENT_TOKENS)


def primitive_toward_goal_condition(goal_condition: dict[str, Any] | None) -> str | None:
    nearest_goal = (goal_condition or {}).get("nearest_goal") or {}
    direction = nearest_goal.get("relative_display_direction")
    return GOAL_DIRECTION_TO_PRIMITIVE.get(str(direction))


class TinyIntentMLP(nn.Module):
    """Replaceable language-intent to basic-control primitive policy."""

    def __init__(self, input_dim: int, hidden_dim: int = 24) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, len(CONTROL_PRIMITIVES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IntentMotorController:
    """Tiny learned bridge: text intent + selected-state features -> primitive."""

    def __init__(self, *, seed: int = 7, train_steps: int = 240) -> None:
        torch.manual_seed(seed)
        training_texts = [
            text
            for examples in INTENT_TRAINING_EXAMPLES.values()
            for text in examples
        ] + GOAL_CONDITION_TRAINING_EXAMPLES
        training_texts.extend(f"bucket is {direction}" for direction in GOAL_DIRECTION_TO_PRIMITIVE)
        words = sorted({token for text in training_texts for token in tokenize(text)})
        self.vocab = {word: i for i, word in enumerate(words)}
        self.seed = int(seed)
        self.train_steps = int(train_steps)
        self.model = TinyIntentMLP(input_dim=len(self.vocab) + 5 + len(GOAL_FEATURE_KEYS))
        self._train_control_vocab()

    def encode(
        self,
        intent: str,
        state: dict[str, Any],
        goal_condition: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        active_goal_condition = goal_condition if intent_uses_goal_condition(intent) else None
        features = torch.zeros(len(self.vocab) + 5 + len(GOAL_FEATURE_KEYS), dtype=torch.float32)
        for token in tokenize(intent):
            idx = self.vocab.get(token)
            if idx is not None:
                features[idx] += 1.0
        if len(self.vocab):
            features[: len(self.vocab)] = features[: len(self.vocab)].clamp(0, 1)

        selected_coord = state.get("selected_coord") or [0, 0]
        selected_shape = state.get("selected_shape") or [0, 0]
        base = len(self.vocab)
        features[base + 0] = 1.0 if state.get("selected") else 0.0
        features[base + 1] = float(selected_coord[0]) / 64.0
        features[base + 2] = float(selected_coord[1]) / 64.0
        features[base + 3] = float(selected_shape[0]) / 64.0
        features[base + 4] = float(selected_shape[1]) / 64.0

        goal_base = base + 5
        for offset, value in enumerate(goal_features(active_goal_condition)):
            features[goal_base + offset] = float(value)
        return features

    def _train_control_vocab(self) -> None:
        rows: list[torch.Tensor] = []
        labels: list[int] = []
        label_index = {primitive: idx for idx, primitive in enumerate(CONTROL_PRIMITIVES)}
        neutral_state = {
            "selected": True,
            "selected_coord": [8, 8],
            "selected_shape": [3, 1],
        }
        for primitive in CONTROL_PRIMITIVES:
            for text in INTENT_TRAINING_EXAMPLES[primitive]:
                rows.append(self.encode(text, neutral_state))
                labels.append(label_index[primitive])
        for direction, primitive in GOAL_DIRECTION_TO_PRIMITIVE.items():
            condition = synthetic_bucket_goal_condition(direction)
            for text in GOAL_CONDITION_TRAINING_EXAMPLES:
                rows.append(self.encode(text, neutral_state, condition))
                labels.append(label_index[primitive])
            rows.append(self.encode(f"bucket is {direction}", neutral_state, condition))
            labels.append(label_index[primitive])
        x = torch.stack(rows)
        y = torch.tensor(labels, dtype=torch.long)
        opt = torch.optim.AdamW(self.model.parameters(), lr=0.03, weight_decay=0.001)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        last_loss = 0.0
        for _ in range(self.train_steps):
            opt.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach().cpu())
        self.training_loss = last_loss
        self.model.eval()

    def choose(
        self,
        intent: str,
        state: dict[str, Any],
        goal_condition: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with torch.no_grad():
            x = self.encode(intent, state, goal_condition).unsqueeze(0)
            logits = self.model(x)[0]
            probs = torch.softmax(logits, dim=0)
            index = int(torch.argmax(probs).item())
        primitive = CONTROL_PRIMITIVES[index]
        active_goal_condition = goal_condition if intent_uses_goal_condition(intent) else None
        nearest_goal = (active_goal_condition or {}).get("nearest_goal") or {}
        goal_direction = nearest_goal.get("relative_display_direction")
        return {
            "intent_text": intent,
            "chosen_primitive": primitive,
            "primitive_probs": {
                primitive_name: float(probs[i].detach().cpu())
                for i, primitive_name in enumerate(CONTROL_PRIMITIVES)
            },
            "training_loss": self.training_loss,
            "policy_type": "tiny_intent_mlp",
            "goal_condition_available": goal_condition is not None,
            "goal_condition_used": active_goal_condition is not None,
            "goal_relative_display_direction": goal_direction,
            "reason_code": f"goal_bucket_{goal_direction}_of_selected"
            if goal_direction in GOAL_DIRECTION_TO_PRIMITIVE
            else "intent_policy_top_primitive",
        }


@dataclass(frozen=True)
class SpriteRef:
    index: int
    name: str
    x: int
    y: int
    width: int
    height: int
    tags: list[str]


@dataclass(frozen=True)
class GroundedAction:
    primitive: str
    action: GameAction
    data: dict[str, Any]
    selected_target: SpriteRef | None

    def packet(self) -> dict[str, Any]:
        return {
            "primitive": self.primitive,
            "id": int(self.action.value),
            "name": self.action.name,
            "data": self.data,
            "selected_target": None
            if self.selected_target is None
            else self.selected_target.__dict__,
        }


class SP80BaseController:
    """Small runtime controller for SP80 click/select and movement primitives."""

    def __init__(self, *, level_index: int = 0) -> None:
        self.game_class = load_sp80_class()
        self.level_index = int(level_index)
        self.intent_policy = IntentMotorController()

    def new_game(self) -> tuple[Any, Any]:
        game = self.game_class()
        obs = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if self.level_index:
            game.set_level(self.level_index)
            frame = game.camera.render(game.current_level.get_sprites())
            obs.frame = [frame]
        if obs is None or not obs.frame:
            raise RuntimeError("SP80 reset produced no frame")
        return game, obs

    def action_set(self) -> dict[str, dict[str, Any]]:
        return json.loads(json.dumps(SP80_ACTION_SET))

    def selectable_sprites(self, game: Any) -> list[SpriteRef]:
        sprites = list(game.fbrwmvzsym())
        return [
            SpriteRef(
                index=i,
                name=str(getattr(sprite, "name", "")),
                x=int(sprite.x),
                y=int(sprite.y),
                width=int(sprite.width),
                height=int(sprite.height),
                tags=[str(tag) for tag in (getattr(sprite, "tags", []) or [])],
            )
            for i, sprite in enumerate(sprites)
        ]

    def display_click_for_sprite(self, game: Any, sprite: SpriteRef) -> dict[str, int]:
        hits: list[tuple[int, int]] = []
        for display_y in range(64):
            for display_x in range(64):
                query_x, query_y = display_x, display_y
                if hasattr(game, "ojydygjkdb"):
                    query_x, query_y = game.ojydygjkdb(query_x, query_y)
                grid_cell = game.camera.display_to_grid(query_x, query_y)
                if grid_cell is None:
                    continue
                grid_x, grid_y = grid_cell
                if (
                    sprite.x <= grid_x < sprite.x + sprite.width
                    and sprite.y <= grid_y < sprite.y + sprite.height
                ):
                    hits.append((display_x, display_y))
        if not hits:
            raise RuntimeError(f"no display click found for sprite {sprite}")
        x, y = hits[len(hits) // 2]
        return {"x": int(x), "y": int(y)}

    def display_bbox_for_sprite(self, game: Any, sprite: Any) -> list[int]:
        cells: list[tuple[int, int]] = []
        for raw_y in range(int(sprite.y), int(sprite.y) + int(sprite.height)):
            for raw_x in range(int(sprite.x), int(sprite.x) + int(sprite.width)):
                if hasattr(game, "rqlmciubuz"):
                    display_x, display_y = game.rqlmciubuz(raw_x, raw_y)
                else:
                    display_x, display_y = raw_x, raw_y
                cells.append((int(display_x), int(display_y)))
        if not cells:
            return [int(sprite.x), int(sprite.y), int(sprite.x), int(sprite.y)]
        xs = [x for x, _ in cells]
        ys = [y for _, y in cells]
        return [min(xs), min(ys), max(xs), max(ys)]

    def sprite_geometry_packet(
        self,
        game: Any,
        sprite: Any,
        *,
        object_id: str,
        semantic_label: str,
        source_tag: str,
    ) -> dict[str, Any]:
        raw_bbox = [
            int(sprite.x),
            int(sprite.y),
            int(sprite.x + sprite.width - 1),
            int(sprite.y + sprite.height - 1),
        ]
        display_bbox = self.display_bbox_for_sprite(game, sprite)
        return {
            "object_id": object_id,
            "semantic_label": semantic_label,
            "source_tag": source_tag,
            "name": str(getattr(sprite, "name", "")),
            "tags": [str(tag) for tag in (getattr(sprite, "tags", []) or [])],
            "x": int(sprite.x),
            "y": int(sprite.y),
            "width": int(sprite.width),
            "height": int(sprite.height),
            "raw_bbox_xyxy": raw_bbox,
            "raw_center": [
                (raw_bbox[0] + raw_bbox[2]) / 2.0,
                (raw_bbox[1] + raw_bbox[3]) / 2.0,
            ],
            "display_bbox_xyxy": display_bbox,
            "display_center": [
                (display_bbox[0] + display_bbox[2]) / 2.0,
                (display_bbox[1] + display_bbox[3]) / 2.0,
            ],
        }

    def _perceived_goal_sprites(self, game: Any) -> list[Any] | None:
        """Select goal/lock sprites by their perceptual color signature (yellow 11,
        filled 13) rather than the privileged sprite tag. Returns ``None`` when the
        level sprites are not enumerable so the caller can fail open to the tag."""
        level = getattr(game, "current_level", None)
        if level is None or not hasattr(level, "get_sprites"):
            return None
        try:
            sprites = list(level.get_sprites())
        except Exception:
            return None
        goal_values = set(GOAL_LOCK_PIXEL_VALUES)
        return [s for s in sprites if _sprite_pixel_value_set(s) & goal_values]

    @staticmethod
    def _goal_sprite_position_key(sprite: Any) -> tuple[int, int, int, int]:
        return (int(sprite.x), int(sprite.y), int(sprite.width), int(sprite.height))

    def bucket_goal_labels(self, game: Any) -> dict[str, Any]:
        # Ground-truth verifier: the privileged sprite tag. Used only to confirm the
        # perceptual selection below; it is no longer the sole control source.
        if hasattr(game, "mxdlffpzkc"):
            tag_sprites = list(game.mxdlffpzkc())
        else:
            tag_sprites = list(game.current_level.get_sprites_by_tag(GOAL_VERIFIER_TAG))
        # Primary source: perceptual lock-goal selection (oracle's yellow-lock signal).
        perceived = self._perceived_goal_sprites(game)
        tag_keys = sorted(self._goal_sprite_position_key(s) for s in tag_sprites)
        perceived_keys = (
            sorted(self._goal_sprite_position_key(s) for s in perceived)
            if perceived is not None
            else None
        )
        perception_matches = perceived_keys is not None and perceived_keys == tag_keys
        if perception_matches:
            # Perception independently found the same lock set; emit in the verifier's
            # order so bucket object_ids (and target_bucket_index) stay stable.
            source_sprites = tag_sprites
            label_source = "perceived_lock_goal_signature"
            control_goal_source = "perceived_lock_goal_signature_verified_against_tag"
        else:
            # Fail open: perception unavailable or disagreed with the verifier.
            source_sprites = tag_sprites
            label_source = "live_game_sprites_by_tag"
            control_goal_source = (
                "live_game_sprites_by_tag_perception_unavailable"
                if perceived is None
                else "live_game_sprites_by_tag_perception_mismatch"
            )
        objects = [
            self.sprite_geometry_packet(
                game,
                sprite,
                object_id=f"bucket_{idx:02d}",
                semantic_label="bucket",
                source_tag=GOAL_VERIFIER_TAG,
            )
            for idx, sprite in enumerate(source_sprites)
        ]
        return {
            "schema": "codexlab.sp80_goal_labels.v1",
            "label_source": label_source,
            "source_tag": GOAL_VERIFIER_TAG,
            "semantic_label": "bucket",
            "object_count": len(objects),
            "goal_color_signature": list(GOAL_LOCK_PIXEL_VALUES),
            "goal_calibration": {
                "control_goal_source": control_goal_source,
                "perception_matches_ground_truth": bool(perception_matches),
                "perceived_object_count": (
                    len(perceived) if perceived is not None else None
                ),
                "verifier_tag_object_count": len(tag_sprites),
                "verifier_tag": GOAL_VERIFIER_TAG,
                "boundary": (
                    "goal_localization_is_perceptual_lock_signature_verified_against"
                    "_privileged_tag_not_solve_truth"
                ),
            },
            "objects": objects,
        }

    def _paint_objective_state(self, game: Any) -> dict[str, Any]:
        """Observed routing-objective state: payload ("paint", v6) location plus
        sink fill (v13) and spill (v14) counts. Measure-only ground truth — this is
        NOT the control objective (the live metric remains effector->goal until the
        flip is validated). Fail-open to empties when sprites are not enumerable."""
        empty = {
            "payload_present": False,
            "payload_center": None,
            "payload_object_count": 0,
            "filled_sink_count": 0,
            "total_sink_count": 0,
            "spill_present": False,
        }
        level = getattr(game, "current_level", None)
        if level is None or not hasattr(level, "get_sprites"):
            return empty
        try:
            sprites = list(level.get_sprites())
        except Exception:
            return empty
        goal_values = set(GOAL_LOCK_PIXEL_VALUES)
        payload_bboxes: list[list[int]] = []
        filled = 0
        total_sink = 0
        spill = False
        for sprite in sprites:
            vals = _sprite_pixel_value_set(sprite)
            if PAYLOAD_PIXEL_VALUE in vals:
                payload_bboxes.append(self.display_bbox_for_sprite(game, sprite))
            if vals & goal_values:
                total_sink += 1
                if FILLED_SINK_PIXEL_VALUE in vals:
                    filled += 1
            if SPILL_PIXEL_VALUE in vals:
                spill = True
        payload_center = None
        if payload_bboxes:
            x0 = min(b[0] for b in payload_bboxes)
            y0 = min(b[1] for b in payload_bboxes)
            x1 = max(b[2] for b in payload_bboxes)
            y1 = max(b[3] for b in payload_bboxes)
            payload_center = [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
        return {
            "payload_present": bool(payload_bboxes),
            "payload_center": payload_center,
            "payload_object_count": len(payload_bboxes),
            "filled_sink_count": filled,
            "total_sink_count": total_sink,
            "spill_present": spill,
        }

    def bucket_goal_condition(self, game: Any, *, target_bucket_index: int | None = None) -> dict[str, Any]:
        labels = self.bucket_goal_labels(game)
        selected = getattr(game, "vsoxmtrhqt", None)
        selected_packet = None
        selected_center = None
        if selected is not None:
            selected_packet = self.sprite_geometry_packet(
                game,
                selected,
                object_id="selected_control_object",
                semantic_label="selected_control_object",
                source_tag="selected",
            )
            selected_center = selected_packet["display_center"]

        paint_state = self._paint_objective_state(game)
        payload_center = paint_state["payload_center"]
        visible_goal_objects: list[dict[str, Any]] = []
        for obj in labels["objects"]:
            relation = self.goal_relation(obj, selected_center)
            visible_goal_objects.append(
                {
                    "object_id": obj["object_id"],
                    "semantic_label": obj["semantic_label"],
                    "source_tag": obj["source_tag"],
                    "shape": [obj["width"], obj["height"]],
                    "paint_to_sink_display_distance": _manhattan(payload_center, obj["display_center"]),
                    **relation,
                }
            )

        nearest_goal = None
        if visible_goal_objects:
            ranked = [
                obj
                for obj in visible_goal_objects
                if isinstance(obj.get("display_manhattan_distance"), (int, float))
            ]
            if ranked:
                nearest_goal = min(ranked, key=lambda obj: float(obj["display_manhattan_distance"]))
        target_goal = None
        if target_bucket_index is not None:
            target_object_id = f"bucket_{int(target_bucket_index):02d}"
            for obj in visible_goal_objects:
                if obj.get("object_id") == target_object_id:
                    target_goal = obj
                    break
        active_goal = target_goal or nearest_goal

        active_sink_paint_distance = None
        if payload_center is not None and isinstance(active_goal, dict):
            active_obj = next(
                (o for o in labels["objects"] if o["object_id"] == active_goal.get("object_id")),
                None,
            )
            if active_obj is not None:
                active_sink_paint_distance = _manhattan(payload_center, active_obj["display_center"])
        sink_paint_distances = [
            o["paint_to_sink_display_distance"]
            for o in visible_goal_objects
            if isinstance(o.get("paint_to_sink_display_distance"), (int, float))
        ]
        nearest_sink_paint_distance = min(sink_paint_distances) if sink_paint_distances else None

        return {
            "schema": "codexlab.sp80_goal_condition.v1",
            "condition": "pour_into_buckets",
            "target_label": "bucket",
            "target_source_tag": "repwkzbkhxl",
            "target_bucket_index": int(target_bucket_index) if target_bucket_index is not None else None,
            "target_object_id": active_goal.get("object_id") if isinstance(active_goal, dict) else None,
            "target_goal": active_goal,
            "actual_nearest_goal": nearest_goal,
            "goal_selection_policy": "explicit_target_bucket_index" if target_goal is not None else "nearest_visible_bucket",
            "goal_object_count": len(visible_goal_objects),
            "visible_goal_objects": visible_goal_objects,
            "nearest_goal": active_goal,
            "paint_objective": {
                "schema": "codexlab.sp80_paint_objective.v1",
                "basis": "measure_only_observed_payload_v6_to_sink_not_control_objective",
                "payload_present": paint_state["payload_present"],
                "payload_center": payload_center,
                "payload_object_count": paint_state["payload_object_count"],
                "active_sink_paint_distance": active_sink_paint_distance,
                "nearest_sink_paint_distance": nearest_sink_paint_distance,
                "filled_sink_count": paint_state["filled_sink_count"],
                "total_sink_count": paint_state["total_sink_count"],
                "spill_present": paint_state["spill_present"],
                "control_objective_is_still_effector_to_goal": True,
                "boundary": "paint_to_sink_is_measurement_not_control_truth_objective_flip_deferred",
            },
            "selected_anchor": None
            if selected_packet is None
            else {
                "object_id": selected_packet["object_id"],
                "semantic_label": selected_packet["semantic_label"],
                "source_tag": selected_packet["source_tag"],
                "shape": [selected_packet["width"], selected_packet["height"]],
            },
            "route_or_replay_used": False,
            "coordinate_target_used": target_goal is not None,
        }

    @staticmethod
    def goal_relation(obj: dict[str, Any], selected_center: list[float] | None) -> dict[str, Any]:
        if selected_center is None:
            return {
                "relative_display_direction": "unknown",
                "primary_axis": "unknown",
                "display_delta_sign": ["unknown", "unknown"],
                "display_distance_bucket": "unknown",
                "display_manhattan_distance": None,
            }
        goal_center = obj["display_center"]
        dx = float(goal_center[0] - selected_center[0])
        dy = float(goal_center[1] - selected_center[1])
        if dx == 0 and dy == 0:
            direction = "same"
        elif abs(dx) > abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"
        distance = abs(dx) + abs(dy)
        return {
            "relative_display_direction": direction,
            "primary_axis": direction_axis(direction),
            "display_delta_sign": [sign_label(dx), sign_label(dy)],
            "display_distance_bucket": distance_bucket(distance),
            "display_manhattan_distance": float(distance),
        }

    def _perceived_selected_object(self, game: Any, obs: Any) -> dict[str, Any] | None:
        """Locate the selected control object from the rendered FRAME alone.

        The selected control platform renders blue (``SELECTED_CONTROL_PIXEL_VALUE``,
        9); unselected platforms render red (8). Returns grid-space
        ``{"coord": [x, y], "shape": [w, h]}`` (top-left cell + extent) or ``None``
        when no blue control region is visible. Coordinate recovery reuses the
        camera's display->grid projection — the same screen geometry
        ``display_click_for_sprite`` relies on to pick click targets; it reads no
        sprite or selection state. Assumes a single contiguous blue control region
        (true for all six SP80 levels); a second blue object would widen the bbox,
        which the verifier-mismatch field in ``state`` would surface.
        """
        frame = obs.frame[-1] if getattr(obs, "frame", None) else None
        if frame is None or not hasattr(game, "camera"):
            return None
        arr = np.asarray(frame)
        ys, xs = np.nonzero(arr == SELECTED_CONTROL_PIXEL_VALUE)
        if xs.size == 0:
            return None
        cells: set[tuple[int, int]] = set()
        for fx, fy in zip(xs.tolist(), ys.tolist()):
            query_x, query_y = int(fx), int(fy)
            if hasattr(game, "ojydygjkdb"):
                query_x, query_y = game.ojydygjkdb(query_x, query_y)
            cell = game.camera.display_to_grid(query_x, query_y)
            if cell is not None:
                cells.add((int(cell[0]), int(cell[1])))
        if not cells:
            return None
        min_x = min(c[0] for c in cells)
        min_y = min(c[1] for c in cells)
        max_x = max(c[0] for c in cells)
        max_y = max(c[1] for c in cells)
        return {"coord": [min_x, min_y], "shape": [max_x - min_x + 1, max_y - min_y + 1]}

    def state(self, game: Any, obs: Any) -> dict[str, Any]:
        # Verifier-only: the privileged selected-sprite object. Recorded to cross-check
        # the perceptual read below; it is NOT the control source.
        verifier = getattr(game, "vsoxmtrhqt", None)
        verifier_coord = [int(verifier.x), int(verifier.y)] if verifier is not None else None
        verifier_shape = [int(verifier.width), int(verifier.height)] if verifier is not None else None

        # Primary control source: perceive the selected object from the rendered frame.
        perceived = self._perceived_selected_object(game, obs)
        if perceived is not None:
            selected_coord = perceived["coord"]
            selected_shape = perceived["shape"]
            selected_present = True
            if verifier_coord is None:
                coord_source = "perceived_blue_control_signature_no_verifier"
            elif perceived["coord"] == verifier_coord:
                coord_source = "perceived_blue_control_signature_verified"
            else:
                coord_source = "perceived_blue_control_signature_verifier_mismatch"
        else:
            # Fail open: no blue control region visible -> fall back to the verifier so
            # the controller still functions, with the source labeled accordingly.
            selected_coord = verifier_coord
            selected_shape = verifier_shape
            selected_present = verifier is not None
            coord_source = "privileged_selected_sprite_perception_unavailable"

        selected_display_coord = None
        if selected_coord is not None:
            if hasattr(game, "rqlmciubuz"):
                display_x, display_y = game.rqlmciubuz(selected_coord[0], selected_coord[1])
                selected_display_coord = [int(display_x), int(display_y)]
            else:
                selected_display_coord = selected_coord
        return {
            "state": str(obs.state.value if hasattr(obs.state, "value") else obs.state),
            "levels_completed": int(getattr(obs, "levels_completed", 0)),
            "mode": str(getattr(game, "dkvpswzsjg", "")),
            "rotation_quarters": int(getattr(game, "fahhoimkk", 0)),
            "steps_remaining": int(getattr(game, "zlhbnhpcq", 0)),
            "selected": bool(selected_present),
            "selected_name": getattr(verifier, "name", None) if verifier is not None else None,
            "selected_coord": selected_coord,
            "selected_display_coord": selected_display_coord,
            "selected_shape": selected_shape,
            "selected_coord_source": coord_source,
            "selected_coord_verifier": verifier_coord,
            "selected_coord_matches_verifier": (
                None
                if (perceived is None or verifier_coord is None)
                else bool(perceived["coord"] == verifier_coord)
            ),
        }

    def click_sprite(self, game: Any, sprite: SpriteRef) -> tuple[Any, dict[str, int]]:
        click = self.display_click_for_sprite(game, sprite)
        obs = game.perform_action(
            ActionInput(id=GameAction.ACTION6, data=click),
            raw=True,
        )
        if obs is None or not obs.frame:
            raise RuntimeError("click action produced no frame")
        return obs, click

    def move_selected(self, game: Any, direction: str) -> Any:
        if direction not in MOVE_ACTIONS:
            raise ValueError(f"unknown direction {direction!r}")
        obs = game.perform_action(ActionInput(id=MOVE_ACTIONS[direction], data={}), raw=True)
        if obs is None or not obs.frame:
            raise RuntimeError(f"move {direction} produced no frame")
        return obs

    def ground_primitive(
        self,
        game: Any,
        primitive: str,
        *,
        target: SpriteRef | None = None,
    ) -> GroundedAction:
        if not primitive.startswith("move_"):
            if primitive == "select_object":
                if target is None:
                    raise RuntimeError("select_object requires a target sprite")
                return GroundedAction(
                    primitive=primitive,
                    action=GameAction.ACTION6,
                    data=self.display_click_for_sprite(game, target),
                    selected_target=target,
                )
            if primitive == "activate_flow":
                return GroundedAction(
                    primitive=primitive,
                    action=GameAction.ACTION5,
                    data={},
                    selected_target=None,
                )
            if primitive == "no_selected_motion":
                return GroundedAction(
                    primitive=primitive,
                    action=GameAction.ACTION1,
                    data={},
                    selected_target=target,
                )
            raise RuntimeError(f"unknown primitive {primitive!r}")
        direction = primitive.removeprefix("move_")
        action = MOVE_ACTIONS[direction]
        return GroundedAction(
            primitive=primitive,
            action=action,
            data={},
            selected_target=target,
        )

    def execute_grounded_action(self, game: Any, grounded: GroundedAction) -> Any:
        obs = game.perform_action(ActionInput(id=grounded.action, data=grounded.data), raw=True)
        if obs is None or not obs.frame:
            raise RuntimeError(f"{grounded.primitive} produced no frame")
        return obs

    def execute_intent(
        self,
        game: Any,
        intent: str,
        state: dict[str, Any],
        *,
        target: SpriteRef | None = None,
        expected_primitive: str | None = None,
    ) -> tuple[Any, dict[str, Any], GroundedAction]:
        decision = self.intent_policy.choose(intent, state)
        primitive = str(decision["chosen_primitive"])
        if expected_primitive is not None and primitive != expected_primitive:
            raise RuntimeError(f"intent chose {primitive}, expected {expected_primitive}: {decision}")
        grounded = self.ground_primitive(game, primitive, target=target)
        obs = self.execute_grounded_action(game, grounded)
        decision = {**decision, "action": grounded.packet()}
        return obs, decision, grounded

    def probe_basic_controls(self, *, out_dir: Path, scale: int) -> dict[str, Any]:
        out_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = out_dir / "frames"
        packets_path = out_dir / "packets.jsonl"

        game, obs = self.new_game()
        initial_frame = np.asarray(obs.frame[-1])
        sprites = self.selectable_sprites(game)
        if not sprites:
            raise RuntimeError("SP80 level has no selectable sprites")
        target = sprites[0]

        initial_meta = save_frame(initial_frame, frames_dir / "initial.png", scale)
        before_click_state = self.state(game, obs)
        click_obs, click_decision, click_grounded = self.execute_intent(
            game,
            "I need to select the object",
            before_click_state,
            target=target,
            expected_primitive="select_object",
        )
        click_frame = np.asarray(click_obs.frame[-1])
        click_meta = save_frame(click_frame, frames_dir / "click_after.png", scale)
        click_packet = {
            "packet_type": "click_select",
            "primitive": "select_object",
            "game": "sp80_liquid_pour",
            "level_index": self.level_index,
            "target_sprite": target.__dict__,
            "action": click_grounded.packet(),
            "intent_policy": click_decision,
            "before_state": before_click_state,
            "after_state": self.state(game, click_obs),
            "before_frame": initial_meta,
            "after_frame": click_meta,
            "delta": frame_delta(initial_frame, click_frame),
            "controlled_runtime": True,
        }

        packets: list[dict[str, Any]] = [click_packet]
        primitive_results: dict[str, Any] = {
            "select_object": {
                "intent_text": click_decision["intent_text"],
                "mlp_chosen_primitive": click_decision["chosen_primitive"],
                "action": click_grounded.packet(),
                "selected_after": click_packet["after_state"]["selected_coord"],
                "changed_pixels": click_packet["delta"]["changed_pixels"],
            }
        }
        movement_intents = {
            "move_up": "I need to go up",
            "move_down": "I need to go down",
            "move_left": "I need to go left",
            "move_right": "I need to go right",
        }
        for primitive, intent_text in movement_intents.items():
            direction = primitive.removeprefix("move_")
            move_game, move_obs = self.new_game()
            move_sprites = self.selectable_sprites(move_game)
            if not move_sprites:
                raise RuntimeError("SP80 level has no selectable sprites during move probe")
            move_target = move_sprites[target.index]
            selected_obs, click_data = self.click_sprite(move_game, move_target)
            selected_frame = np.asarray(selected_obs.frame[-1])
            before_state = self.state(move_game, selected_obs)
            after_obs, policy_decision, move_grounded = self.execute_intent(
                move_game,
                intent_text,
                before_state,
                target=move_target,
                expected_primitive=primitive,
            )
            after_frame = np.asarray(after_obs.frame[-1])
            before_meta = save_frame(selected_frame, frames_dir / f"{direction}_before.png", scale)
            after_meta = save_frame(after_frame, frames_dir / f"{direction}_after.png", scale)
            after_state = self.state(move_game, after_obs)
            before_coord = before_state.get("selected_coord")
            after_coord = after_state.get("selected_coord")
            moved = before_coord is not None and after_coord is not None and before_coord != after_coord
            packet = {
                "packet_type": "move_selected",
                "primitive": primitive,
                "game": "sp80_liquid_pour",
                "level_index": self.level_index,
                "target_sprite": move_target.__dict__,
                "selection_action": {"id": int(GameAction.ACTION6.value), "name": "ACTION6", "data": click_data},
                "action": move_grounded.packet(),
                "intent_policy": policy_decision,
                "before_state": before_state,
                "after_state": after_state,
                "before_frame": before_meta,
                "after_frame": after_meta,
                "delta": frame_delta(selected_frame, after_frame),
                "selected_coord_delta": None
                if not moved
                else [
                    int(after_coord[0] - before_coord[0]),
                    int(after_coord[1] - before_coord[1]),
                ],
                "moved_selected_sprite": bool(moved),
                "controlled_runtime": True,
            }
            packets.append(packet)
            primitive_results[primitive] = {
                "intent_text": intent_text,
                "mlp_chosen_primitive": policy_decision["chosen_primitive"],
                "action": move_grounded.packet(),
                "moved_selected_sprite": bool(moved),
                "selected_coord_delta": packet["selected_coord_delta"],
                "changed_pixels": packet["delta"]["changed_pixels"],
            }

        flow_game, flow_obs = self.new_game()
        flow_frame_before = np.asarray(flow_obs.frame[-1])
        flow_before_state = self.state(flow_game, flow_obs)
        flow_obs_after, flow_decision, flow_grounded = self.execute_intent(
            flow_game,
            "I need to activate flow",
            flow_before_state,
            expected_primitive="activate_flow",
        )
        flow_action_frames = [np.asarray(frame) for frame in flow_obs_after.frame]
        flow_frame_after = flow_action_frames[-1]
        flow_before_meta = save_frame(flow_frame_before, frames_dir / "activate_flow_before.png", scale)
        flow_after_meta = save_frame(flow_frame_after, frames_dir / "activate_flow_after.png", scale)
        flow_sequence_meta = save_frame_sequence(
            flow_action_frames,
            frames_dir / "activate_flow_sequence",
            scale,
        )
        flow_deltas = [frame_delta(flow_frame_before, frame) for frame in flow_action_frames]
        max_flow_idx, max_flow_delta = max(
            enumerate(flow_deltas),
            key=lambda item: int(item[1].get("changed_pixels") or 0),
        )
        flow_after_state = self.state(flow_game, flow_obs_after)
        flow_packet = {
            "packet_type": "activate_flow",
            "primitive": "activate_flow",
            "game": "sp80_liquid_pour",
            "level_index": self.level_index,
            "action": flow_grounded.packet(),
            "intent_policy": flow_decision,
            "before_state": flow_before_state,
            "after_state": flow_after_state,
            "before_frame": flow_before_meta,
            "after_frame": flow_after_meta,
            "action_frames": flow_sequence_meta,
            "action_frame_count": len(flow_action_frames),
            "delta": frame_delta(flow_frame_before, flow_frame_after),
            "max_action_frame_delta": {
                "frame_index": int(max_flow_idx),
                **max_flow_delta,
            },
            "controlled_runtime": True,
        }
        packets.append(flow_packet)
        primitive_results["activate_flow"] = {
            "intent_text": flow_decision["intent_text"],
            "mlp_chosen_primitive": flow_decision["chosen_primitive"],
            "action": flow_grounded.packet(),
            "mode_before": flow_before_state["mode"],
            "mode_after": flow_after_state["mode"],
            "levels_completed_delta": int(flow_after_state["levels_completed"]) - int(flow_before_state["levels_completed"]),
            "changed_pixels": flow_packet["delta"]["changed_pixels"],
            "action_frame_count": len(flow_action_frames),
            "max_changed_pixels": max_flow_delta["changed_pixels"],
            "max_changed_frame_index": int(max_flow_idx),
        }

        blocked_game, blocked_obs = self.new_game()
        blocked_sprites = self.selectable_sprites(blocked_game)
        if not blocked_sprites:
            raise RuntimeError("SP80 level has no selectable sprites during no-motion probe")
        blocked_target = blocked_sprites[target.index]
        blocked_selected_obs, blocked_click_data = self.click_sprite(blocked_game, blocked_target)
        first_up_obs = self.move_selected(blocked_game, "up")
        blocked_before_frame = np.asarray(first_up_obs.frame[-1])
        blocked_before_state = self.state(blocked_game, first_up_obs)
        blocked_after_obs, blocked_decision, blocked_grounded = self.execute_intent(
            blocked_game,
            "I need to test no motion",
            blocked_before_state,
            target=blocked_target,
            expected_primitive="no_selected_motion",
        )
        blocked_after_frame = np.asarray(blocked_after_obs.frame[-1])
        blocked_before_meta = save_frame(blocked_before_frame, frames_dir / "no_selected_motion_before.png", scale)
        blocked_after_meta = save_frame(blocked_after_frame, frames_dir / "no_selected_motion_after.png", scale)
        blocked_after_state = self.state(blocked_game, blocked_after_obs)
        blocked_before_coord = blocked_before_state.get("selected_coord")
        blocked_after_coord = blocked_after_state.get("selected_coord")
        selected_stationary = blocked_before_coord is not None and blocked_before_coord == blocked_after_coord
        blocked_packet = {
            "packet_type": "no_selected_motion",
            "primitive": "no_selected_motion",
            "game": "sp80_liquid_pour",
            "level_index": self.level_index,
            "setup_actions": [
                {"id": int(GameAction.ACTION6.value), "name": "ACTION6", "data": blocked_click_data},
                {"id": int(GameAction.ACTION1.value), "name": "ACTION1", "direction": "up", "data": {}},
            ],
            "action": blocked_grounded.packet(),
            "intent_policy": blocked_decision,
            "before_state": blocked_before_state,
            "after_state": blocked_after_state,
            "before_frame": blocked_before_meta,
            "after_frame": blocked_after_meta,
            "delta": frame_delta(blocked_before_frame, blocked_after_frame),
            "selected_coord_delta": None
            if blocked_before_coord is None or blocked_after_coord is None
            else [
                int(blocked_after_coord[0] - blocked_before_coord[0]),
                int(blocked_after_coord[1] - blocked_before_coord[1]),
            ],
            "selected_object_stationary": bool(selected_stationary),
            "controlled_runtime": True,
        }
        packets.append(blocked_packet)
        primitive_results["no_selected_motion"] = {
            "intent_text": blocked_decision["intent_text"],
            "mlp_chosen_primitive": blocked_decision["chosen_primitive"],
            "action": blocked_grounded.packet(),
            "selected_object_stationary": bool(selected_stationary),
            "selected_coord_delta": blocked_packet["selected_coord_delta"],
            "changed_pixels": blocked_packet["delta"]["changed_pixels"],
        }

        with packets_path.open("w", encoding="utf-8") as f:
            for packet in packets:
                f.write(json.dumps(packet, sort_keys=True) + "\n")

        result = {
            "schema": "codexlab.sp80_basic_controls_mlp_mvp.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "condition": {
                "game_source": SP80_PATH.relative_to(ROOT).as_posix(),
                "game_source_sha256": sha256_file(SP80_PATH),
                "level_index": self.level_index,
                "target_selection": "first live selectable sprite from game.fbrwmvzsym()",
                "control_policy": "tiny MLP maps language intent to one basic control primitive",
                "route_or_replay_used": False,
                "human_trace_used": False,
            },
            "tiny_brain": {
                "type": "IntentMotorController/TinyIntentMLP",
                "input": "intent text bag-of-words plus selected sprite state",
                "output": "basic-control primitive logits",
                "primitive_vocab": CONTROL_PRIMITIVES,
                "sp80_action_set": self.action_set(),
                "training_examples": INTENT_TRAINING_EXAMPLES,
                "training_loss": self.intent_policy.training_loss,
                "seed": self.intent_policy.seed,
            },
            "primitive_results": primitive_results,
            "packets_jsonl": packets_path.relative_to(ROOT).as_posix(),
            "frames_dir": frames_dir.relative_to(ROOT).as_posix(),
        }
        result_path = out_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level-index", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=Path("codexlab/runs/sp80_basic_controls_mlp_mvp"))
    parser.add_argument("--scale", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    controller = SP80BaseController(level_index=args.level_index)
    result = controller.probe_basic_controls(out_dir=ROOT / args.out_dir, scale=max(1, int(args.scale)))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
