"""Goal/belief packet contract for Rex-conditioned planning.

Rex owns action selection. Lewis/LeWMSight may only provide this packet:
goal and mechanic beliefs, study-tag semantics, and confidence. The packet is
small on purpose so oracle-goal pretraining and Lewis-inferred deployment use
the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.goal_latent import GoalLatentConfig, GoalLatentReconstructor
from lewm.goal_property import ARC_NUM_COLORS, NUM_GOAL_TYPES


RexGoalSource = Literal["oracle", "lewis", "manual", "unknown"]

_TENSOR_FIELDS = (
    "confidence",
    "goal_type_probs",
    "mechanic_probs",
    "family_probs",
    "subclass_probs",
    "stage_probs",
    "keys_probs",
    "delta_px",
    "agent_delta",
    "goal_family_probs",
    "role_buckets_probs",
    "capability_probs",
    "topology_probs",
)


@dataclass(frozen=True)
class RexGoalPacket:
    """Typed goal/belief input to a Rex successor.

    All tensors are batch-first. Probability tensors are already normalized or
    sigmoided by the builder. `confidence` is `[B]` and names how strongly Rex
    should trust this packet, not whether Rex must act.
    """

    source: RexGoalSource
    confidence: torch.Tensor
    goal_type_probs: torch.Tensor | None = None
    mechanic_probs: torch.Tensor | None = None
    family_probs: torch.Tensor | None = None
    subclass_probs: torch.Tensor | None = None
    stage_probs: torch.Tensor | None = None
    keys_probs: torch.Tensor | None = None
    delta_px: torch.Tensor | None = None
    agent_delta: torch.Tensor | None = None
    goal_family_probs: torch.Tensor | None = None
    role_buckets_probs: torch.Tensor | None = None
    capability_probs: torch.Tensor | None = None
    topology_probs: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.source not in ("oracle", "lewis", "manual", "unknown"):
            raise ValueError(f"unsupported RexGoalPacket source: {self.source}")
        conf = self.confidence
        if conf.dim() == 2 and conf.shape[1] == 1:
            conf = conf.squeeze(1)
            object.__setattr__(self, "confidence", conf)
        if conf.dim() != 1:
            raise ValueError(f"confidence must be [B] or [B, 1], got {tuple(conf.shape)}")
        if not torch.is_floating_point(conf):
            raise TypeError("confidence must be floating point")
        if not torch.isfinite(conf).all():
            raise ValueError("confidence contains non-finite values")
        if conf.numel() == 0:
            raise ValueError("confidence must not be empty")
        if bool((conf < 0).any() or (conf > 1).any()):
            raise ValueError("confidence must be in [0, 1]")

        batch = conf.shape[0]
        for field in fields(self):
            if field.name not in _TENSOR_FIELDS or field.name == "confidence":
                continue
            value = getattr(self, field.name)
            if value is None:
                continue
            if value.shape[0] != batch:
                raise ValueError(
                    f"{field.name} batch {value.shape[0]} does not match confidence batch {batch}"
                )
            if not torch.isfinite(value).all():
                raise ValueError(f"{field.name} contains non-finite values")

    @property
    def batch_size(self) -> int:
        return int(self.confidence.shape[0])

    def to(self, device: torch.device | str) -> "RexGoalPacket":
        values = {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }
        for name in _TENSOR_FIELDS:
            value = values.get(name)
            if value is not None:
                values[name] = value.to(device)
        return RexGoalPacket(**values)


@dataclass(frozen=True)
class RexGoalProjection:
    """Projected goal latent consumed by a goal-conditioned Rex predictor."""

    z_goal: torch.Tensor
    lambda_conf: torch.Tensor
    properties_map: torch.Tensor
    source: RexGoalSource


def _softmax(outputs: dict[str, torch.Tensor], key: str) -> torch.Tensor | None:
    logits = outputs.get(key)
    if logits is None:
        return None
    return F.softmax(logits.float(), dim=-1)


def _sigmoid(outputs: dict[str, torch.Tensor], key: str) -> torch.Tensor | None:
    logits = outputs.get(key)
    if logits is None:
        return None
    return torch.sigmoid(logits.float())


def _mean_confidence(parts: list[torch.Tensor]) -> torch.Tensor:
    if not parts:
        raise ValueError(
            "cannot infer RexGoalPacket confidence: pass confidence or provide logits"
        )
    return torch.stack(parts, dim=0).mean(dim=0).clamp(0.0, 1.0)


def confidence_from_goal_property_outputs(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    """Conservative confidence from available categorical heads.

    This is not a policy decision. It is a deterministic bridge so Rex can use
    the same packet shape for oracle and Lewis paths until a calibrated
    confidence head exists.
    """

    parts: list[torch.Tensor] = []
    for key in (
        "goal_type_logits",
        "mechanic_logits",
        "family_logits",
        "subclass_logits",
        "stage_logits",
        "goal_family_logits",
        "topology_logits",
    ):
        probs = _softmax(outputs, key)
        if probs is not None:
            parts.append(probs.max(dim=-1).values)
    return _mean_confidence(parts)


def packet_from_goal_property_outputs(
    outputs: dict[str, torch.Tensor],
    *,
    source: RexGoalSource = "lewis",
    confidence: torch.Tensor | None = None,
) -> RexGoalPacket:
    """Build a RexGoalPacket from GoalPropertyModel-style outputs."""

    conf = (
        confidence.float()
        if confidence is not None
        else confidence_from_goal_property_outputs(outputs)
    )
    return RexGoalPacket(
        source=source,
        confidence=conf,
        goal_type_probs=_softmax(outputs, "goal_type_logits"),
        mechanic_probs=_softmax(outputs, "mechanic_logits"),
        family_probs=_softmax(outputs, "family_logits"),
        subclass_probs=_softmax(outputs, "subclass_logits"),
        stage_probs=_softmax(outputs, "stage_logits"),
        keys_probs=_sigmoid(outputs, "keys_logits"),
        delta_px=outputs.get("delta_px"),
        agent_delta=outputs.get("agent_delta"),
        goal_family_probs=_softmax(outputs, "goal_family_logits"),
        role_buckets_probs=_sigmoid(outputs, "role_buckets_logits"),
        capability_probs=_sigmoid(outputs, "capability_logits"),
        topology_probs=_softmax(outputs, "topology_logits"),
    )


def packet_from_tensor_bundle(bundle: dict) -> RexGoalPacket:
    """Rehydrate a RexGoalPacket from an exported tensor bundle.

    The bundle schema is produced by ``scripts/export_lewis_goal_packets.py``.
    This is the lossless runtime path; the JSONL export is only a readable
    summary for inspection.
    """

    schema = bundle.get("schema")
    if schema != "lewm.rex_goal_packet.tensor_bundle.v001":
        raise ValueError(f"unsupported RexGoalPacket tensor bundle schema: {schema!r}")
    source = bundle.get("source", "unknown")
    tensors = bundle.get("tensors")
    if not isinstance(tensors, dict):
        raise ValueError("RexGoalPacket tensor bundle missing tensors dict")
    if "confidence" not in tensors:
        raise ValueError("RexGoalPacket tensor bundle missing confidence")

    values = {
        "source": source,
        **{
            name: tensors.get(name)
            for name in _TENSOR_FIELDS
        },
    }
    return RexGoalPacket(**values)


def load_packet_tensor_bundle(path: str | Path) -> RexGoalPacket:
    """Load an exported RexGoalPacket tensor bundle from disk."""

    bundle = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise ValueError(f"RexGoalPacket tensor bundle is not a dict: {path}")
    return packet_from_tensor_bundle(bundle)


def _packet_tensor_or_zeros(
    packet: RexGoalPacket,
    name: str,
    shape_tail: tuple[int, ...],
) -> torch.Tensor:
    value = getattr(packet, name)
    if value is None:
        return packet.confidence.new_zeros((packet.batch_size, *shape_tail))
    expected = (packet.batch_size, *shape_tail)
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} shape {tuple(value.shape)} != expected {expected}")
    return value.to(device=packet.confidence.device, dtype=packet.confidence.dtype)


def packet_to_properties_map(packet: RexGoalPacket) -> torch.Tensor:
    """Return the 23-dim MAP vector expected by GoalLatentReconstructor.

    Order: goal_type(4), agent_delta(2), keys(16), delta_px(1). Missing
    components are explicit zeros, so oracle and Lewis paths can share the same
    vector shape even while some heads are not available yet.
    """

    goal_type = _packet_tensor_or_zeros(packet, "goal_type_probs", (NUM_GOAL_TYPES,))
    agent_delta = _packet_tensor_or_zeros(packet, "agent_delta", (2,))
    keys = _packet_tensor_or_zeros(packet, "keys_probs", (ARC_NUM_COLORS,))
    delta_px = _packet_tensor_or_zeros(packet, "delta_px", ())
    if delta_px.dim() == 1:
        delta_px = delta_px.unsqueeze(-1)
    return torch.cat([goal_type, agent_delta, keys, delta_px], dim=-1)


class RexGoalPacketProjector(nn.Module):
    """Project a RexGoalPacket to ``(z_goal, lambda_conf)``.

    This is the physical handoff into the goal-conditioned Rex garage. It does
    not inspect or emit actions; action search remains owned by Rex/LeWM.
    """

    def __init__(self, config: GoalLatentConfig | None = None) -> None:
        super().__init__()
        self.reconstructor = GoalLatentReconstructor(config)

    def forward(self, packet: RexGoalPacket) -> RexGoalProjection:
        properties_map = packet_to_properties_map(packet)
        z_goal = self.reconstructor(properties_map)
        return RexGoalProjection(
            z_goal=z_goal,
            lambda_conf=packet.confidence.to(device=z_goal.device, dtype=z_goal.dtype),
            properties_map=properties_map,
            source=packet.source,
        )


__all__ = [
    "RexGoalPacket",
    "RexGoalPacketProjector",
    "RexGoalProjection",
    "RexGoalSource",
    "confidence_from_goal_property_outputs",
    "load_packet_tensor_bundle",
    "packet_from_tensor_bundle",
    "packet_to_properties_map",
    "packet_from_goal_property_outputs",
]
