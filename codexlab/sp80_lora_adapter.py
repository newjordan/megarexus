"""Load and run the offline SP80 LoRA dry-run adapter."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import nn

from players.sp80.registry import CONTROL_PRIMITIVES


NUMERIC_KEYS = [
    "setup_action_count",
    "rotation_quarters",
    "steps_remaining",
    "selected",
    "selected_shape_w",
    "selected_shape_h",
    "has_goal_condition",
    "goal_object_count",
    "nearest_goal_distance",
]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def row_target(row: dict[str, Any]) -> str:
    primitive = (row.get("target") or {}).get("primitive")
    if primitive not in CONTROL_PRIMITIVES:
        raise ValueError(f"{row.get('row_id')}: invalid primitive {primitive!r}")
    return str(primitive)


def input_feature_text(input_obj: dict[str, Any]) -> str:
    observation = input_obj.get("observation") or {}
    state = observation.get("state") or {}
    shape = state.get("selected_shape") or [0, 0]
    view_state = input_obj.get("view_state") or {}
    goal_condition = input_obj.get("goal_condition") or {}
    nearest_goal = goal_condition.get("nearest_goal") or {}
    parts = [
        f"game {input_obj.get('game')}",
        f"task {input_obj.get('task')}",
        f"intent {input_obj.get('intent')}",
        f"view_schema {view_state.get('schema')}",
        f"view_phase {view_state.get('phase')}",
        f"view_look_order {view_state.get('look_order')}",
        f"view_raw_pixels_ground_truth {view_state.get('raw_pixels_are_observation_ground_truth')}",
        f"setup {observation.get('setup')}",
        f"setup_action_count {observation.get('setup_action_count')}",
        f"state {state.get('state')}",
        f"mode {state.get('mode')}",
        f"rotation_quarters {state.get('rotation_quarters')}",
        f"selected {state.get('selected')}",
        f"selected_shape_w {shape[0] if len(shape) > 0 else 0}",
        f"selected_shape_h {shape[1] if len(shape) > 1 else 0}",
        f"goal_condition {goal_condition.get('condition')}",
        f"goal_target_label {goal_condition.get('target_label')}",
        f"goal_target_source_tag {goal_condition.get('target_source_tag')}",
        f"goal_object_count {goal_condition.get('goal_object_count')}",
        f"nearest_goal_direction {nearest_goal.get('relative_display_direction')}",
        f"nearest_goal_axis {nearest_goal.get('primary_axis')}",
        f"nearest_goal_distance_bucket {nearest_goal.get('display_distance_bucket')}",
        "contract primitive_only no_coordinates no_routes motor_executes",
    ]
    return " ".join(parts)


def coordinate_free_text(row: dict[str, Any]) -> str:
    return input_feature_text(row["input"])


def numeric_features_from_input(input_obj: dict[str, Any]) -> list[float]:
    observation = input_obj.get("observation") or {}
    state = observation.get("state") or {}
    shape = state.get("selected_shape") or [0, 0]
    goal_condition = input_obj.get("goal_condition") or {}
    nearest_goal = goal_condition.get("nearest_goal") or {}
    return [
        float(observation.get("setup_action_count") or 0) / 32.0,
        float(state.get("rotation_quarters") or 0) / 3.0,
        float(state.get("steps_remaining") or 0) / 64.0,
        1.0 if state.get("selected") else 0.0,
        float(shape[0] if len(shape) > 0 else 0) / 64.0,
        float(shape[1] if len(shape) > 1 else 0) / 64.0,
        1.0 if goal_condition else 0.0,
        float(goal_condition.get("goal_object_count") or 0) / 8.0,
        float(nearest_goal.get("display_manhattan_distance") or 0) / 64.0,
    ]


def numeric_features(row: dict[str, Any]) -> list[float]:
    return numeric_features_from_input(row["input"])


def build_vocab(train_rows: list[dict[str, Any]], max_vocab: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in train_rows:
        counts.update(tokenize(coordinate_free_text(row)))
    words = [word for word, _ in counts.most_common(max_vocab)]
    return {word: idx for idx, word in enumerate(sorted(words))}


def encode_input(input_obj: dict[str, Any], vocab: dict[str, int]) -> torch.Tensor:
    x = torch.zeros(len(vocab) + len(NUMERIC_KEYS), dtype=torch.float32)
    for token in tokenize(input_feature_text(input_obj)):
        col = vocab.get(token)
        if col is not None:
            x[col] = 1.0
    for offset, value in enumerate(numeric_features_from_input(input_obj)):
        x[len(vocab) + offset] = float(value)
    return x


def encode_rows(rows: list[dict[str, Any]], vocab: dict[str, int]) -> torch.Tensor:
    if not rows:
        return torch.zeros((0, len(vocab) + len(NUMERIC_KEYS)), dtype=torch.float32)
    return torch.stack([encode_input(row["input"], vocab) for row in rows])


class FrozenLinearLoRA(nn.Module):
    def __init__(self, *, input_dim: int, output_dim: int, rank: int, alpha: float, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        base_weight = torch.randn(output_dim, input_dim, generator=generator) * 0.01
        base_bias = torch.zeros(output_dim)
        self.register_buffer("base_weight", base_weight)
        self.register_buffer("base_bias", base_bias)
        self.lora_a = nn.Parameter(torch.randn(rank, input_dim, generator=generator) * 0.02)
        self.lora_b = nn.Parameter(torch.zeros(output_dim, rank))
        self.rank = int(rank)
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = x @ self.base_weight.T + self.base_bias
        adapter = (x @ self.lora_a.T) @ self.lora_b.T
        return base + adapter * (self.alpha / float(self.rank))

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {
            "lora_a": self.lora_a.detach().cpu(),
            "lora_b": self.lora_b.detach().cpu(),
        }


def input_from_brain_request(request: dict[str, Any]) -> dict[str, Any]:
    observation = request.get("observation") or {}
    setup_actions = observation.get("setup_actions") or []
    return {
        "game": request.get("game"),
        "task": "choose_one_control_primitive",
        "intent": request.get("intent"),
        "view_state": request.get("view_state") or {},
        "observation": {
            "setup": observation.get("setup"),
            "setup_action_count": observation.get("setup_action_count", len(setup_actions)),
            "state": observation.get("state") or {},
        },
        "goal_condition": request.get("goal_condition") or {},
        "controller_contract": {
            "output": "one primitive",
            "may_choose_coordinates": False,
            "may_emit_route": False,
            "motor_executes_actions": True,
        },
    }


def load_adapter_bundle(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    condition = payload["condition"]
    vocab = payload["vocab"]
    state = payload["state_dict"]
    lora_a = state["lora_a"]
    lora_b = state["lora_b"]
    model = FrozenLinearLoRA(
        input_dim=int(lora_a.shape[1]),
        output_dim=int(lora_b.shape[0]),
        rank=int(lora_a.shape[0]),
        alpha=float(condition["alpha"]),
        seed=int(condition["seed"]),
    )
    with torch.no_grad():
        model.lora_a.copy_(lora_a)
        model.lora_b.copy_(lora_b)
    model.eval()
    return {
        "payload": payload,
        "condition": condition,
        "vocab": vocab,
        "model": model,
    }


def predict_input(bundle: dict[str, Any], input_obj: dict[str, Any]) -> dict[str, Any]:
    model = bundle["model"]
    vocab = bundle["vocab"]
    x = encode_input(input_obj, vocab).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=0)
        pred_index = int(torch.argmax(probs).item())
    return {
        "primitive": CONTROL_PRIMITIVES[pred_index],
        "confidence": float(probs[pred_index].item()),
        "probabilities": {
            primitive: float(probs[idx].item())
            for idx, primitive in enumerate(CONTROL_PRIMITIVES)
        },
    }


def predict_request(bundle: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    input_obj = input_from_brain_request(request)
    prediction = predict_input(bundle, input_obj)
    return {
        **prediction,
        "input_obj": input_obj,
        "adapter_condition": {
            "run_label": bundle["condition"].get("run_label"),
            "qwen_weights_used": bundle["condition"].get("qwen_weights_used"),
            "qwen_lora_claim": bundle["condition"].get("qwen_lora_claim"),
            "feature_space": bundle["condition"].get("feature_space"),
        },
    }
