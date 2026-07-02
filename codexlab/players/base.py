"""Shared player contracts for primitive-control game harnesses."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    action_id: int
    action_name: str
    requires_selection: bool = False
    requires_click_target: bool = False
    effect_tags: tuple[str, ...] = ()
    description: str = ""

    def action_entry(self) -> dict[str, Any]:
        return {
            "action_id": int(self.action_id),
            "action_name": self.action_name,
            "requires_selection": self.requires_selection,
            "requires_click_target": self.requires_click_target,
            "effect_tags": list(self.effect_tags),
            "description": self.description,
        }


@dataclass(frozen=True)
class ControlCase:
    case_id: str
    intent: str
    expected_primitive: str
    setup: str
    effect_tags: tuple[str, ...] = ()

    def to_harness_case(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "intent": self.intent,
            "expected_primitive": self.expected_primitive,
            "setup": self.setup,
            "effect_tags": list(self.effect_tags),
        }


@dataclass(frozen=True)
class GamePlayerDescriptor:
    game_id: str
    package: str
    interface_schema: str
    result_schema: str
    packet_schema: str
    brain_request_schema: str
    controller_source: str
    registry_source: str
    verifier_source: str

    def to_packet(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BrainDecision:
    backend: str
    primitive: str | None
    confidence: float | None
    raw: Any
    latency_ms: float
    error: str | None = None

    def to_packet(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "primitive": self.primitive,
            "confidence": self.confidence,
            "raw": self.raw,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class PrimitiveRegistry:
    def __init__(self, specs: list[PrimitiveSpec], aliases: dict[str, str] | None = None) -> None:
        self._specs = {spec.name: spec for spec in specs}
        self._aliases = aliases or {}

    def names(self) -> list[str]:
        return list(self._specs)

    def action_set(self) -> dict[str, dict[str, Any]]:
        return {name: spec.action_entry() for name, spec in self._specs.items()}

    def direct_action_expectations(self) -> dict[str, tuple[int, str]]:
        return {name: (spec.action_id, spec.action_name) for name, spec in self._specs.items()}

    def normalize(self, value: Any) -> str | None:
        if value is None:
            return None
        key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        key = key.removeprefix("gameaction.")
        if key in self._specs:
            return key
        return self._aliases.get(key)


class PacketRecorder:
    """Writes the standard run artifacts for a player harness."""

    def __init__(self, *, root: Path, out_dir: Path) -> None:
        self.root = root.resolve()
        self.out_dir = out_dir.resolve()
        self.packets_path = self.out_dir / "packets.jsonl"
        self.brain_calls_path = self.out_dir / "brain_calls.jsonl"
        self.result_path = self.out_dir / "result.json"

    def relpath(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def write_packets(self, packets: list[dict[str, Any]]) -> None:
        self._write_jsonl(self.packets_path, packets)

    def write_brain_calls(self, brain_calls: list[dict[str, Any]]) -> None:
        self._write_jsonl(self.brain_calls_path, brain_calls)

    def write_result(self, result: dict[str, Any]) -> None:
        self.result_path.parent.mkdir(parents=True, exist_ok=True)
        self.result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

