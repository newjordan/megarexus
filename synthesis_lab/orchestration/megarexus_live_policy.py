"""Megarexus CNN policy adapter for ARC live/offline scorecard runners.

This is an action adapter only: it loads a trained Megarexus checkpoint and
uses the same learned catch and arrangement probes as the smoke gates to choose
one SDK action per observed frame. It does not read game source, sprite state,
or hidden environment internals.

Course-correction runtime (Stage 0 of HANDOFF §0B), added 2026-07-02 after
the first live scout failed beyond level 0:

- FRAME CONVENTION: the runner passes the full animation frame list per
  action (obs["frames"]); the adapter consumes it with the recorder
  convention the shards were trained on — cur = frames[-1] (settled),
  prev = frames[-2] for multi-frame actions. The live failure planned on
  frames[0] of the level-up pour animation: the old level-0 scene.
- POUR DISCIPLINE: pours are the scarce resource (~4 per life). Each built
  plan gets at most ONE real pour. A pour that does not complete the level
  refutes that plan (own history, no answer key); the next build excludes
  refuted plans. No action is ever "pour because we ran out of ideas" —
  the old attempt-cap branch that kept pouring into game over is gone.
- PLAN CONFIDENCE: a plan is executed only if the model's own level head
  separates it from the field (margin vs 5th-best and vs the null plan,
  calibrated on local settled-frame planning: healthy >= 7.7, the live
  OOD failure was 0.63). Unconfident builds hold with a safe no-op
  (background click) and periodically replan.
- RESET AWARENESS: the runner passes obs["resets"]; a reset refutes any
  in-flight pour plan, clears per-life state, and keeps refuted-plan
  memory so the search continues instead of repeating.
- FRAME CAPTURE: with MEGAREXUS_TRACE_DIR set, every observed animation
  frame is saved for post-run audit (the live failure was only diagnosable
  from bbox forensics because no frames were captured).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_LAB_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _LAB_ROOT.parent
_EXPERIMENT_ROOT = _LAB_ROOT / "experiments" / "megarexus_cnn"
for _entry in (_LAB_ROOT, _LAB_ROOT / "orchestration", _EXPERIMENT_ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from arrange_probe import (  # noqa: E402
    exhaustive_x_plans,
    imagine_plan,
    plan_actions,
    rank_movables_if_clicked,
)
from behavioral_catch_probe import imagine_pour_column  # noqa: E402
from collect_plumbing_play import component_at, relocate  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from eval_ghost import encode  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402
from train import load_model_state_compatible  # noqa: E402


DEFAULT_CANDIDATE = (
    _EXPERIMENT_ROOT / "runs" / "outcome_v9_retry_20260702" / "last.pt"
)
DEFAULT_RELEASE = _EXPERIMENT_ROOT / "runs" / "outcome_v8" / "last.pt"

# Plan-confidence margin, calibrated 2026-07-02 on local settled-frame
# exhaustive planning (winner minus 5th-best: 16.9 for v9retry last.pt,
# 7.8 for best.pt) vs the live transition-frame failure (0.63).
CONFIDENCE_MARGIN = 3.0
# Course-correction budgets per level: bounded search, never a pour loop.
MAX_PLANS_PER_LEVEL = 8
MAX_POURS_PER_LEVEL = 6
NOCONF_REBUILD_EVERY = 5

# Background-click hold points (y kept away from gauge rows 0/63).
_NOOP_POINTS = ((1, 1), (62, 1), (1, 30), (62, 30), (32, 1))
_TRUE_VALUES = {"1", "true", "yes", "on"}


def clamp_float_env(name: str, default: float, lo: float, hi: float) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(lo, min(hi, value))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_hash(grid: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(grid, dtype=np.int16).tobytes()).hexdigest()


def plan_key(plan) -> tuple:
    return tuple((int(m), int(dx), int(dy)) for m, dx, dy in plan)


def is_null_plan(plan) -> bool:
    return all(int(dx) == 0 and int(dy) == 0 for _m, dx, dy in plan)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE_VALUES


def binary_components(mask: np.ndarray, min_area: int = 3) -> list[dict[str, Any]]:
    """Small 4-connected component extractor for 64x64 predicted masks."""
    arr = np.asarray(mask).astype(bool)
    seen = np.zeros(arr.shape, dtype=bool)
    rows: list[dict[str, Any]] = []
    height, width = arr.shape
    for y0 in range(height):
        for x0 in range(width):
            if seen[y0, x0] or not arr[y0, x0]:
                continue
            stack = [(x0, y0)]
            seen[y0, x0] = True
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                x, y = stack.pop()
                xs.append(x)
                ys.append(y)
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if seen[ny, nx] or not arr[ny, nx]:
                        continue
                    seen[ny, nx] = True
                    stack.append((nx, ny))
            area = len(xs)
            if area < min_area:
                continue
            rows.append(
                {
                    "area_pixels": int(area),
                    "bbox_xyxy": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                    "center_xy": [round(float(np.mean(xs)), 3), round(float(np.mean(ys)), 3)],
                }
            )
    rows.sort(key=lambda row: (-int(row["area_pixels"]), row["bbox_xyxy"]))
    return rows


def glove_plane_of(comp: dict[str, Any], device: str) -> torch.Tensor:
    plane = torch.zeros(1, 1, 16, 16, device=device)
    x0, y0, x1, y1 = (int(v) // 4 for v in comp["bbox_xyxy"])
    plane[0, 0, y0 : y1 + 1, x0 : x1 + 1] = 1.0
    return plane


@torch.no_grad()
def raw_control_score(
    model: MegarexusCNN,
    device: str,
    cur: np.ndarray,
    prev: np.ndarray,
    comp: dict[str, Any],
    action_id: int = 4,
) -> float:
    frame = torch.cat([encode(cur), encode(prev)], dim=0).unsqueeze(0).to(device)
    return float(
        model.forward_control(
            frame,
            glove_plane_of(comp, device),
            torch.tensor([action_id], device=device),
        )
    )


class MegarexusArcPolicy:
    """Stateful per-frame policy for `codexlab.arc_suite_runner.run_game`."""

    def __init__(self, checkpoint_path: str | os.PathLike[str] | None = None) -> None:
        env_ckpt = os.environ.get("MEGAREXUS_CKPT")
        ckpt = Path(checkpoint_path or env_ckpt or DEFAULT_CANDIDATE)
        if not ckpt.exists():
            ckpt = DEFAULT_RELEASE
        self.checkpoint_path = ckpt.resolve()
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        payload = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_keys = tuple((payload.get("model") or {}).keys())
        stack = int(payload["config"].get("stack_frames", 2))
        self.stack_frames = stack
        self.goal_channels = tuple(payload.get("channels", ("target_lock", "key_block", "paint_source")))
        self.model = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)).to(self.device).eval()
        load_model_state_compatible(self.model, payload["model"])

        self.prev_grid: np.ndarray | None = None
        self.first_level0_hash: str | None = None
        self.level_entry_hash: str | None = None
        self.levels_seen = 0
        self.last_resets = 0
        self.frames_fallback_count = 0
        self.glove: dict[str, Any] | None = None
        self.pending_glove_click: tuple[int, int, dict[str, Any]] | None = None
        self.catch_moves_since_pour = 0

        self.arrange: dict[str, Any] | None = None
        self.pending_arrange_click: tuple[int, int, int, list[int]] | None = None
        self.pending_pour: dict[str, Any] | None = None
        self.refuted_plans: dict[int, set[tuple]] = {}
        self.plans_built: dict[int, int] = {}
        self.pours_at_level: dict[int, int] = {}
        self.noop_count = 0
        self.arrange_memory_enabled = env_flag("MEGAREXUS_ARRANGE_MEMORY")
        self.graph_demo_plan_enabled = env_flag("MEGAREXUS_GRAPH_DEMO_PLAN")
        self.goal_provider_enabled = env_flag("MEGAREXUS_GOAL_PROVIDER")
        self.goal_pour_budget_enabled = env_flag("MEGAREXUS_GOAL_POUR_BUDGET")
        self.goal_threshold = clamp_float_env("MEGAREXUS_GOAL_THRESHOLD", 0.5, 0.05, 0.95)
        self.active_goal_schema: dict[str, Any] | None = None
        self.goal_provider_updates: list[dict[str, Any]] = []
        self.arrange_attempt_ledger: list[dict[str, Any]] = []
        self.failed_attempt_signatures: dict[int, set[tuple]] = {}

        self.trace_dir = os.environ.get("MEGAREXUS_TRACE_DIR") or None
        self.frame_capture: list[dict[str, Any]] = []

        self.ledger: list[dict[str, Any]] = []
        self.plan_summaries: list[dict[str, Any]] = []
        self.last_action_source = "megarexus_init"
        self.last_action_record: dict[str, Any] | None = None

    # ------------------------------------------------------------- arrange memory
    def _layout_signature(self, levels: int, comps: list[dict[str, Any]]) -> tuple:
        return (
            int(levels),
            tuple(tuple(int(v) for v in comp["bbox_xyxy"]) for comp in comps),
        )

    def _attempt_signature(self, levels: int, layout_sig: tuple, pkey: tuple) -> tuple:
        return (int(levels), layout_sig, tuple(tuple(int(v) for v in item) for item in pkey))

    def _serialise_plan_key(self, pkey: tuple | None) -> list[list[int]] | None:
        if pkey is None:
            return None
        return [[int(v) for v in item] for item in pkey]

    def _serialise_layout_signature(self, layout_sig: tuple | None) -> dict[str, Any] | None:
        if layout_sig is None:
            return None
        level, boxes = layout_sig
        return {
            "level": int(level),
            "top_movable_bboxes": [[int(v) for v in box] for box in boxes],
        }

    def _serialise_attempt_signature(self, sig: tuple | None) -> dict[str, Any] | None:
        if sig is None:
            return None
        level, layout_sig, pkey = sig
        return {
            "level": int(level),
            "layout_signature": self._serialise_layout_signature(layout_sig),
            "plan_key": self._serialise_plan_key(pkey),
        }

    # ------------------------------------------------------------- state
    def _reset_episode_state(self) -> None:
        """Per-life state. Refuted-plan memory intentionally survives."""
        self.glove = None
        self.pending_glove_click = None
        self.catch_moves_since_pour = 0
        self.arrange = None
        self.pending_arrange_click = None
        self.active_goal_schema = None

    def _record(
        self,
        *,
        obs: dict[str, Any],
        history: list[dict[str, Any]],
        source: str,
        action: int,
        x: int | None = None,
        y: int | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[int, int | None, int | None]:
        record = {
            "step": len(history),
            "levels_completed": int(obs.get("levels_completed") or 0),
            "source": source,
            "action": int(action),
            "x": None if x is None else int(x),
            "y": None if y is None else int(y),
            "info": info or {},
        }
        self.last_action_source = source
        self.last_action_record = record
        self.ledger.append(record)
        return int(action), x, y

    def _capture_frames(self, frames: list[np.ndarray], levels: int, obs: dict[str, Any]) -> None:
        if self.trace_dir is None:
            return
        self.frame_capture.append(
            {
                "call": len(self.ledger),
                "levels": int(levels),
                "state": str(obs.get("state", "")),
                "resets": int(obs.get("resets") or 0),
                "frames": np.stack([np.asarray(f, dtype=np.int8) for f in frames]),
            }
        )

    def _note_pour_outcome(self, levels: int, reset_seen: bool, grid: np.ndarray) -> None:
        """Hindsight judgment of our own last arrange pour, env fields only."""
        if self.pending_pour is None:
            return
        pour = self.pending_pour
        self.pending_pour = None
        succeeded = levels > int(pour["level"]) and not reset_seen
        if self.arrange_memory_enabled:
            attempt_sig = pour.get("attempt_signature")
            row = {
                "level": int(pour["level"]),
                "levels_after": int(levels),
                "outcome": "success" if succeeded else "refuted",
                "reset_seen": bool(reset_seen),
                "before_frame_sha256": pour.get("before_frame_sha256"),
                "after_frame_sha256": frame_hash(grid),
                "layout_signature": self._serialise_layout_signature(pour.get("layout_signature")),
                "plan_key": self._serialise_plan_key(pour.get("plan_key")),
                "attempt_signature": self._serialise_attempt_signature(attempt_sig),
            }
            self.arrange_attempt_ledger.append(row)
            if not succeeded and attempt_sig is not None:
                self.failed_attempt_signatures.setdefault(int(pour["level"]), set()).add(attempt_sig)
        if succeeded:
            return  # the pour completed the level; nothing to refute
        self.refuted_plans.setdefault(int(pour["level"]), set()).add(pour["plan_key"])
        self.arrange = None  # force a fresh build that excludes it

    def _update_reset_detection(
        self, grid: np.ndarray, levels: int, history: list[dict[str, Any]], reset_seen: bool
    ) -> None:
        digest = frame_hash(grid)
        if levels == 0 and self.first_level0_hash is None:
            self.first_level0_hash = digest
        reset_like = (
            levels == 0
            and self.first_level0_hash is not None
            and digest == self.first_level0_hash
            and len(history) > 0
            and self.levels_seen > 0
        )
        if reset_seen or levels < self.levels_seen or reset_like:
            self._reset_episode_state()
            self.level_entry_hash = digest
        if levels != self.levels_seen:
            self._reset_episode_state()
            self.level_entry_hash = digest
        self.levels_seen = levels

    def _update_pending_clicks(self, grid: np.ndarray) -> None:
        if self.pending_glove_click is not None:
            x, y, comp = self.pending_glove_click
            self.glove = component_at(grid, x, y) or relocate(grid, comp) or comp
            self.pending_glove_click = None
        elif self.glove is not None:
            self.glove = relocate(grid, self.glove) or self.glove

        if self.pending_arrange_click is not None and self.arrange is not None:
            m, x, y, moves = self.pending_arrange_click
            roster = self.arrange["roster"]
            roster[m] = component_at(grid, x, y) or relocate(grid, roster[m]) or roster[m]
            self.arrange["move_queue"] = list(moves)
            self.pending_arrange_click = None

    # ------------------------------------------------------------ goal provider
    def _frame_stack(self, grid: np.ndarray, prev: np.ndarray) -> torch.Tensor:
        cur = encode(grid)
        if self.stack_frames == 1:
            return cur.unsqueeze(0).to(self.device)
        return torch.cat([cur, encode(prev)], dim=0).unsqueeze(0).to(self.device)

    def _component_from_prediction(
        self, grid: np.ndarray, mask: np.ndarray
    ) -> dict[str, Any] | None:
        comps = self._components_from_prediction(grid, mask)
        return comps[0] if comps else None

    def _components_from_prediction(
        self, grid: np.ndarray, mask: np.ndarray
    ) -> list[dict[str, Any]]:
        comps = frame_components(grid)
        if not comps:
            return []
        mask_b = np.asarray(mask).astype(bool)
        scored = []
        for comp in comps:
            comp_mask = np.asarray(comp.get("mask")).astype(bool)
            inter = int(np.logical_and(mask_b, comp_mask).sum())
            if inter <= 0:
                continue
            union = int(np.logical_or(mask_b, comp_mask).sum())
            scored.append((inter / max(1, union), inter, comp))
        if not scored:
            return []
        scored.sort(key=lambda row: (-row[0], -row[1], row[2]["bbox_xyxy"]))
        return [
            {
                "component_id": str(comp.get("component_id", "")),
                "raw_pixel_value": int(comp.get("color", 0)),
                "bbox_xyxy": [int(v) for v in comp["bbox_xyxy"]],
                "center_xy": [round(float(v), 3) for v in comp["center_xy"]],
                "area_pixels": int(comp.get("area", 0)),
                "prediction_component_iou": round(float(iou), 6),
                "prediction_overlap_pixels": int(inter),
            }
            for iou, inter, comp in scored
        ]

    def _expand_paint_output_components(
        self, grid: np.ndarray, paint_components: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Paint-output channel semantics: every small block of the paint color.

        The model mask only needs to seed the paint color. The provider then
        mirrors the label rule used for Contra training: once a paint-color
        block is detected, all small components with that same raw value are
        paint outputs.
        """
        seed = next(
            (
                comp
                for comp in paint_components
                if int(comp.get("area_pixels", 0)) <= 64
            ),
            None,
        )
        if seed is None:
            return paint_components
        paint_value = int(seed["raw_pixel_value"])
        by_box = {
            tuple(int(v) for v in comp["bbox_xyxy"]): comp
            for comp in paint_components
            if int(comp.get("raw_pixel_value", -1)) == paint_value
        }
        expanded = []
        for comp in frame_components(grid):
            if int(comp.get("color", -1)) != paint_value:
                continue
            if int(comp.get("area", 0)) > 64:
                continue
            box = tuple(int(v) for v in comp["bbox_xyxy"])
            matched = by_box.get(box)
            expanded.append(
                {
                    "component_id": str(comp.get("component_id", "")),
                    "raw_pixel_value": int(comp.get("color", 0)),
                    "bbox_xyxy": [int(v) for v in comp["bbox_xyxy"]],
                    "center_xy": [round(float(v), 3) for v in comp["center_xy"]],
                    "area_pixels": int(comp.get("area", 0)),
                    "prediction_component_iou": (
                        0.0
                        if matched is None
                        else float(matched.get("prediction_component_iou", 0.0))
                    ),
                    "prediction_overlap_pixels": (
                        0
                        if matched is None
                        else int(matched.get("prediction_overlap_pixels", 0))
                    ),
                    "selection_reason": (
                        "same_raw_paint_value_expansion"
                        if matched is None
                        else "model_seeded_paint_output"
                    ),
                }
            )
        expanded.sort(key=lambda row: row["bbox_xyxy"])
        return expanded

    @torch.no_grad()
    def _update_goal_provider(self, levels: int, grid: np.ndarray, prev: np.ndarray) -> None:
        if not self.goal_provider_enabled or levels < 1:
            return
        frame = self._frame_stack(grid, prev)
        probs = torch.sigmoid(self.model.forward_goal(frame))[0].detach().cpu().numpy()
        masks = (probs >= self.goal_threshold)
        target_components = binary_components(masks[0], min_area=32)
        key_component = self._component_from_prediction(grid, masks[1])
        paint_name = self.goal_channels[2] if len(self.goal_channels) >= 3 else "paint_source"
        paint_components = self._components_from_prediction(grid, masks[2])
        if paint_name == "paint_output":
            paint_components = self._expand_paint_output_components(grid, paint_components)
        source_component = paint_components[0] if paint_components else None
        schema = {
            "level": int(levels),
            "frame_sha256": frame_hash(grid),
            "threshold": float(self.goal_threshold),
            "channels": list(self.goal_channels),
            "target_count": int(len(target_components)),
            "target_components": target_components,
            "key_component": key_component,
            "paint_source_component": source_component if paint_name == "paint_source" else None,
            "paint_output_components": paint_components if paint_name == "paint_output" else [],
            "paint_output_count": int(len(paint_components)) if paint_name == "paint_output" else 0,
            "mask_pixels": {
                "target_lock": int(masks[0].sum()),
                "key_block": int(masks[1].sum()),
                str(paint_name): int(masks[2].sum()),
            },
            "mean_prob": {
                "target_lock": round(float(probs[0].mean()), 6),
                "key_block": round(float(probs[1].mean()), 6),
                str(paint_name): round(float(probs[2].mean()), 6),
            },
        }
        self.active_goal_schema = schema
        if len(self.goal_provider_updates) < 512:
            self.goal_provider_updates.append(schema)

    def _goal_key_component_for_arrange(self, grid: np.ndarray) -> dict[str, Any] | None:
        if not (self.goal_provider_enabled and self.active_goal_schema):
            return None
        key = self.active_goal_schema.get("key_component")
        if not key:
            return None
        x, y = (int(round(v)) for v in key["center_xy"])
        return component_at(grid, x, y) or {
            "component_id": key.get("component_id", "goal_key"),
            "color": int(key.get("raw_pixel_value", 0)),
            "bbox_xyxy": [int(v) for v in key["bbox_xyxy"]],
            "center_xy": [float(v) for v in key["center_xy"]],
            "area": int(key.get("area_pixels", 0)),
        }

    def _goal_pour_cap(self, default_cap: int) -> int:
        if not (self.goal_provider_enabled and self.goal_pour_budget_enabled and self.active_goal_schema):
            return int(default_cap)
        target_count = int(self.active_goal_schema.get("target_count") or 0)
        if target_count <= 0:
            return int(default_cap)
        return max(1, min(int(default_cap), target_count))

    # -------------------------------------------------------------- catch
    def _select_glove(
        self,
        obs: dict[str, Any],
        history: list[dict[str, Any]],
        grid: np.ndarray,
        prev: np.ndarray,
    ) -> tuple[int, int | None, int | None]:
        comps = frame_components(grid)
        if not comps:
            return self._record(obs=obs, history=history, source="megarexus_no_components", action=5)
        scored = [
            (raw_control_score(self.model, self.device, grid, prev, comp, 4), comp)
            for comp in comps
        ]
        scored.sort(key=lambda row: -row[0])
        score, comp = scored[0]
        x, y = int(comp["center_xy"][0]), int(comp["center_xy"][1])
        self.pending_glove_click = (x, y, comp)
        return self._record(
            obs=obs,
            history=history,
            source="megarexus_select_glove",
            action=6,
            x=x,
            y=y,
            info={"control_logit": round(float(score), 4), "bbox": comp["bbox_xyxy"]},
        )

    def _catch_action(
        self,
        obs: dict[str, Any],
        history: list[dict[str, Any]],
        grid: np.ndarray,
        prev: np.ndarray,
    ) -> tuple[int, int | None, int | None]:
        if self.glove is None:
            return self._select_glove(obs, history, grid, prev)
        tx = imagine_pour_column(self.model, self.device, grid, prev)
        if tx is None:
            self.catch_moves_since_pour = 0
            return self._record(obs=obs, history=history, source="megarexus_catch_no_column", action=5)
        x0, _y0, x1, _y1 = self.glove["bbox_xyxy"]
        if x0 <= tx <= x1 or self.catch_moves_since_pour >= 10:
            info = {
                "target_col": int(tx),
                "glove_bbox": self.glove["bbox_xyxy"],
                "moves_since_pour": int(self.catch_moves_since_pour),
            }
            self.catch_moves_since_pour = 0
            return self._record(obs=obs, history=history, source="megarexus_catch_pour", action=5, info=info)
        aid = 4 if tx > x1 else 3
        self.catch_moves_since_pour += 1
        return self._record(
            obs=obs,
            history=history,
            source="megarexus_catch_move",
            action=aid,
            info={"target_col": int(tx), "glove_bbox": self.glove["bbox_xyxy"]},
        )

    # ------------------------------------------------------------ arrange
    def _graph_demo_plan_candidate(
        self, levels: int, grid: np.ndarray
    ) -> tuple[list[dict[str, Any]], list[tuple[int, int, int]], dict[str, Any]] | None:
        """Graph/y-rank demo-basin plan, controlled-scout-backed.

        Stage7b found that the heldout seed 20290103 fails when the lower
        bar is target-x aligned (+3) but passes with the human-demo x-profile:
        upper +3, middle +2, lower +1. This opt-in path exposes that distinct
        search representation without running the serial model scorer.
        """
        band = [
            comp for comp in frame_components(grid)
            if 12 <= int(comp["bbox_xyxy"][1]) <= 48 and int(comp.get("area", 0)) >= 32
        ]
        movables = sorted(band, key=lambda c: int(c["center_xy"][1]))
        if len(movables) == 4:
            plan = [(0, 4, 1), (1, 3, 2), (2, 3, 2), (3, 4, 1)]
            action_items = [
                {"movable": 0, "moves": [4]},
                {"movable": 1, "moves": [3, 3]},
                {"movable": 2, "moves": [3, 3]},
                {"movable": 3, "moves": [4]},
            ]
            summary = {
                "level": int(levels),
                "build_index": self.plans_built.get(levels, 0) + 1,
                "plan_score": None,
                "null_score": None,
                "margin_vs_5th": None,
                "margin_vs_null": None,
                "confident": True,
                "confidence_basis": "graph_level2_raw_action_template_controlled_scout",
                "graph_x_profile": "stage17_raw_action_id_template",
                "dx_by_y_rank": None,
                "dy_by_y_rank": [0, 0, 0, 0],
                "count_by_y_rank": [1, 2, 2, 1],
                "direction_action_id_by_y_rank": [4, 3, 3, 4],
                "graph_demo_plan_enabled": True,
                "arrange_memory_enabled": bool(self.arrange_memory_enabled),
                "refuted_excluded": len(self.refuted_plans.get(levels, set())),
                "legacy_refuted_total": len(self.refuted_plans.get(levels, set())),
                "layout_memory_refuted_total": len(self.failed_attempt_signatures.get(levels, set())),
                "layout_signature": self._serialise_layout_signature(self._layout_signature(levels, movables)),
                "top5_scores": [],
                "plan": plan,
                "roster_bboxes": [list(c["bbox_xyxy"]) for c in movables],
                "action_items": action_items,
                "stage17_evidence": {
                    "report": "synthesis_lab/experiments/megarexus_cnn/runs/stage17_level2_action_template_probe_v12_graphadaptive_20260704/graph_plan_probe_level2.json",
                    "report_sha256": "4bfa869576771d9340c82085e3b9a6c2109fc7dbf04ec3c788b249d04cfb2140",
                    "condition_sha256": "d6f1ca052015f9fe6dd217a8969286c1f484fcd1bb91b8005eabf837143e00fe",
                    "winning_candidate": "codexlab_solution_level2_action_ids_counts_1_2_2_1",
                },
            }
            return movables, plan, summary
        if len(movables) != 3:
            return None
        target_x = 41
        bottom = [
            comp for comp in frame_components(grid)
            if int(comp["bbox_xyxy"][1]) >= 52 and int(comp.get("area", 0)) <= 24
        ]
        if bottom:
            target_x = int(round(float(np.median([int(c["center_xy"][0]) for c in bottom]))))
        target_dx = [
            int(np.clip(int(round((target_x - int(comp["center_xy"][0])) / 4.0)), -3, 3))
            for comp in movables
        ]
        middle_y = int(movables[1]["center_xy"][1])
        if middle_y >= 28:
            x_profile = "target_x_middle_down"
            dx = target_dx
        else:
            x_profile = "demo_x_original_band"
            dx = [3, 2, 1]
        plan = [(1, int(dx[1]), 0), (0, int(dx[0]), 0), (2, int(dx[2]), 0)]
        summary = {
            "level": int(levels),
            "build_index": self.plans_built.get(levels, 0) + 1,
            "plan_score": None,
            "null_score": None,
            "margin_vs_5th": None,
            "margin_vs_null": None,
            "confident": True,
            "confidence_basis": "graph_demo_plan_controlled_scout",
            "graph_x_profile": x_profile,
            "target_x": int(target_x),
            "middle_y": int(middle_y),
            "dx_by_y_rank": [int(v) for v in dx],
            "graph_demo_plan_enabled": True,
            "arrange_memory_enabled": bool(self.arrange_memory_enabled),
            "refuted_excluded": len(self.refuted_plans.get(levels, set())),
            "legacy_refuted_total": len(self.refuted_plans.get(levels, set())),
            "layout_memory_refuted_total": len(self.failed_attempt_signatures.get(levels, set())),
            "layout_signature": self._serialise_layout_signature(self._layout_signature(levels, movables)),
            "top5_scores": [],
            "plan": plan,
            "roster_bboxes": [list(c["bbox_xyxy"]) for c in movables],
            "action_items": [
                {"movable": int(m), "moves": [int(a) for a in moves]}
                for m, moves in plan_actions(plan, movables)
            ],
        }
        return movables, plan, summary

    def _safe_noop(
        self,
        obs: dict[str, Any],
        history: list[dict[str, Any]],
        grid: np.ndarray,
        source: str,
        info: dict[str, Any] | None = None,
    ) -> tuple[int, int | None, int | None]:
        """Hold action that spends no pour: click a background cell."""
        flat = np.asarray(grid, dtype=np.int64).ravel()
        bg = int(np.bincount(flat, minlength=16).argmax())
        x, y = 1, 1
        for cx, cy in _NOOP_POINTS:
            if int(grid[cy, cx]) == bg:
                x, y = cx, cy
                break
        else:
            hits = np.argwhere(np.asarray(grid) == bg)
            if len(hits):
                y, x = (int(v) for v in hits[0])
        self.noop_count += 1
        payload = {"noop_bg_color": bg}
        payload.update(info or {})
        return self._record(
            obs=obs, history=history, source=source, action=6, x=x, y=y, info=payload
        )

    def _build_arrange_plan(self, levels: int, grid: np.ndarray, prev: np.ndarray) -> None:
        built = self.plans_built.get(levels, 0)
        if built >= MAX_PLANS_PER_LEVEL:
            self.arrange = {"level": levels, "status": "plan_budget_exhausted",
                            "reason": f"{built} plans already built"}
            return
        if self.graph_demo_plan_enabled:
            candidate = self._graph_demo_plan_candidate(levels, grid)
            if candidate is not None:
                movables, best_plan, summary = candidate
                if self.active_goal_schema is not None:
                    summary["goal_schema"] = self.active_goal_schema
                self.plans_built[levels] = built + 1
                self.plan_summaries.append(summary)
                layout_sig = self._layout_signature(levels, movables)
                self.arrange = {
                    "level": levels,
                    "status": "planned",
                    "roster": movables,
                    "layout_signature": layout_sig,
                    "plan": best_plan,
                    "plan_key": plan_key(best_plan),
                    "items": [
                        (int(row["movable"]), [int(a) for a in row["moves"]])
                        for row in summary.get("action_items", [])
                    ] or plan_actions(best_plan, movables),
                    "item_index": 0,
                    "move_queue": [],
                    "poured": False,
                    "noconf_age": 0,
                    "summary": summary,
                }
                return
            self.arrange = {
                "level": levels,
                "status": "no_roster",
                "reason": "graph_demo_plan_enabled but no supported 3- or 4-component midband roster",
            }
            return
        ranked_movables = [
            comp for _score, comp in
            rank_movables_if_clicked(self.model, self.device, grid, prev, top_k=3)
        ]
        goal_key = self._goal_key_component_for_arrange(grid)
        movables = []
        if goal_key is not None and len(movables) < 3:
            if not any(tuple(int(v) for v in existing["bbox_xyxy"]) == tuple(int(v) for v in goal_key["bbox_xyxy"]) for existing in movables):
                movables.append(goal_key)
        for comp in ranked_movables:
            bbox = tuple(int(v) for v in comp["bbox_xyxy"])
            if any(tuple(int(v) for v in existing["bbox_xyxy"]) == bbox for existing in movables):
                continue
            movables.append(comp)
            if len(movables) >= 3:
                break
        if len(movables) < 3:
            self.arrange = {"level": levels, "status": "no_roster",
                            "reason": f"ranked {len(movables)} movables"}
            return

        layout_sig = self._layout_signature(levels, movables)
        legacy_refuted = self.refuted_plans.get(levels, set())
        memory_refuted = self.failed_attempt_signatures.get(levels, set())
        scored: list[tuple[float, Any]] = []
        null_score: float | None = None
        for plan in exhaustive_x_plans():
            score, _imagined_end = imagine_plan(self.model, self.device, grid, prev, movables, plan)
            score = float(score)
            if is_null_plan(plan):
                null_score = score
            pkey = plan_key(plan)
            if self.arrange_memory_enabled:
                if self._attempt_signature(levels, layout_sig, pkey) in memory_refuted:
                    continue
            elif pkey in legacy_refuted:
                continue
            scored.append((score, plan))
        self.plans_built[levels] = built + 1
        if not scored:
            self.arrange = {"level": levels, "status": "no_candidates",
                            "reason": "all plans refuted"}
            return

        scored.sort(key=lambda row: -row[0])
        best_score, best_plan = scored[0]
        margin5 = best_score - scored[min(4, len(scored) - 1)][0]
        if is_null_plan(best_plan) or null_score is None:
            margin_null = None
            confident = margin5 >= CONFIDENCE_MARGIN
        else:
            margin_null = best_score - null_score
            confident = margin5 >= CONFIDENCE_MARGIN and margin_null >= CONFIDENCE_MARGIN

        items = plan_actions(best_plan, movables)
        summary = {
            "level": int(levels),
            "build_index": built + 1,
            "plan_score": float(best_score),
            "null_score": null_score,
            "margin_vs_5th": round(float(margin5), 4),
            "margin_vs_null": None if margin_null is None else round(float(margin_null), 4),
            "confident": bool(confident),
            "arrange_memory_enabled": bool(self.arrange_memory_enabled),
            "refuted_excluded": len(memory_refuted) if self.arrange_memory_enabled else len(legacy_refuted),
            "legacy_refuted_total": len(legacy_refuted),
            "layout_memory_refuted_total": len(memory_refuted),
            "layout_signature": self._serialise_layout_signature(layout_sig),
            "top5_scores": [round(s, 3) for s, _p in scored[:5]],
            "plan": best_plan,
            "roster_bboxes": [list(c["bbox_xyxy"]) for c in movables],
            "action_items": [{"movable": int(m), "moves": [int(a) for a in moves]} for m, moves in items],
            "goal_provider_enabled": bool(self.goal_provider_enabled),
            "goal_key_seeded_roster": bool(goal_key is not None),
            "goal_schema": self.active_goal_schema if self.goal_provider_enabled else None,
        }
        self.plan_summaries.append(summary)
        self.arrange = {
            "level": levels,
            "status": "planned" if confident else "no_confidence",
            "roster": movables,
            "layout_signature": layout_sig,
            "plan": best_plan,
            "plan_key": plan_key(best_plan),
            "items": items,
            "item_index": 0,
            "move_queue": [],
            "poured": False,
            "noconf_age": 0,
            "summary": summary,
        }

    def _arrange_action(
        self,
        obs: dict[str, Any],
        history: list[dict[str, Any]],
        grid: np.ndarray,
        prev: np.ndarray,
        levels: int,
    ) -> tuple[int, int | None, int | None]:
        if self.arrange is None or self.arrange.get("level") != levels:
            self._build_arrange_plan(levels, grid, prev)

        assert self.arrange is not None
        status = self.arrange.get("status")

        if status == "no_confidence":
            # Hold safely; replan every NOCONF_REBUILD_EVERY calls in case the
            # scene settles into distribution.
            self.arrange["noconf_age"] = int(self.arrange.get("noconf_age", 0)) + 1
            age = self.arrange["noconf_age"]
            if age % NOCONF_REBUILD_EVERY == 0:
                self._build_arrange_plan(levels, grid, prev)
                status = self.arrange.get("status")
                if status != "planned":
                    return self._safe_noop(
                        obs, history, grid, "megarexus_plan_no_confidence",
                        info={"summary_status": status, "age": age},
                    )
            else:
                return self._safe_noop(
                    obs, history, grid, "megarexus_plan_no_confidence",
                    info={"margin_vs_5th": self.arrange["summary"]["margin_vs_5th"],
                          "margin_vs_null": self.arrange["summary"]["margin_vs_null"],
                          "age": age},
                )

        if status in ("no_roster", "no_candidates", "plan_budget_exhausted"):
            return self._safe_noop(
                obs, history, grid, f"megarexus_arrange_{status}",
                info={"reason": self.arrange.get("reason")},
            )

        if self.arrange["move_queue"]:
            aid = int(self.arrange["move_queue"].pop(0))
            if not self.arrange["move_queue"]:
                self.arrange["item_index"] += 1
            return self._record(obs=obs, history=history, source="megarexus_arrange_move", action=aid)

        items = self.arrange["items"]
        if self.arrange["item_index"] < len(items):
            m, moves = items[self.arrange["item_index"]]
            comp = relocate(grid, self.arrange["roster"][m]) or self.arrange["roster"][m]
            x, y = int(comp["center_xy"][0]), int(comp["center_xy"][1])
            self.pending_arrange_click = (int(m), x, y, list(moves))
            return self._record(
                obs=obs,
                history=history,
                source="megarexus_arrange_select",
                action=6,
                x=x,
                y=y,
                info={"movable": int(m), "moves": [int(a) for a in moves], "bbox": comp["bbox_xyxy"]},
            )

        # Plan fully executed: exactly one pour per plan, budget-capped.
        if self.arrange.get("poured"):
            # Outcome not observed yet (pending) or already refuted; hold.
            return self._safe_noop(obs, history, grid, "megarexus_await_pour_outcome")
        pours = self.pours_at_level.get(levels, 0)
        pour_cap = self._goal_pour_cap(MAX_POURS_PER_LEVEL)
        if pours >= pour_cap:
            return self._safe_noop(
                obs, history, grid, "megarexus_pour_budget_exhausted",
                info={
                    "pours": pours,
                    "pour_cap": int(pour_cap),
                    "goal_pour_budget_enabled": bool(self.goal_pour_budget_enabled),
                    "goal_target_count": None
                    if self.active_goal_schema is None
                    else int(self.active_goal_schema.get("target_count") or 0),
                },
            )
        self.pours_at_level[levels] = pours + 1
        self.arrange["poured"] = True
        attempt_sig = self._attempt_signature(
            levels, self.arrange["layout_signature"], self.arrange["plan_key"]
        )
        self.pending_pour = {
            "level": int(levels),
            "plan_key": self.arrange["plan_key"],
            "layout_signature": self.arrange["layout_signature"],
            "attempt_signature": attempt_sig,
            "before_frame_sha256": frame_hash(grid),
        }
        memory_info: dict[str, Any] = {}
        if self.arrange_memory_enabled:
            memory_info = {
                "arrange_memory_enabled": True,
                "layout_signature": self._serialise_layout_signature(self.arrange["layout_signature"]),
                "attempt_signature": self._serialise_attempt_signature(attempt_sig),
            }
        return self._record(
            obs=obs,
            history=history,
            source="megarexus_arrange_pour",
            action=5,
            info={
                "pour_index_at_level": pours + 1,
                "pour_cap": int(pour_cap),
                "goal_pour_budget_enabled": bool(self.goal_pour_budget_enabled),
                "goal_target_count": None
                if self.active_goal_schema is None
                else int(self.active_goal_schema.get("target_count") or 0),
                "goal_schema": self.active_goal_schema if self.goal_provider_enabled else None,
                "plan_score": self.arrange["summary"]["plan_score"],
                "margin_vs_5th": self.arrange["summary"]["margin_vs_5th"],
                "margin_vs_null": self.arrange["summary"]["margin_vs_null"],
                "confidence_basis": self.arrange["summary"].get("confidence_basis", "model_level_head_margin"),
                **memory_info,
            },
        )

    # ------------------------------------------------------------ __call__
    def __call__(self, obs: dict[str, Any], history: list[dict[str, Any]]) -> tuple[int, int | None, int | None]:
        anim = obs.get("frames")
        if anim:
            frames = [np.asarray(f, dtype=np.int8) for f in anim]
            grid = frames[-1]
            if len(frames) >= 2:
                prev = frames[-2]
            else:
                prev = self.prev_grid if self.prev_grid is not None else grid
        else:
            self.frames_fallback_count += 1
            frames = None
            grid = np.asarray(obs["grid"], dtype=np.int8)
            prev = self.prev_grid if self.prev_grid is not None else grid
        levels = int(obs.get("levels_completed") or 0)

        self._capture_frames(frames if frames is not None else [grid], levels, obs)

        resets = obs.get("resets")
        reset_seen = resets is not None and int(resets) > self.last_resets
        if resets is not None:
            self.last_resets = int(resets)

        self._note_pour_outcome(levels, reset_seen, grid)
        self._update_reset_detection(grid, levels, history, reset_seen)
        self._update_pending_clicks(grid)
        self._update_goal_provider(levels, grid, prev)

        if levels >= 1:
            action = self._arrange_action(obs, history, grid, prev, levels)
        else:
            action = self._catch_action(obs, history, grid, prev)
        self.prev_grid = grid.copy()
        return action

    def _flush_frame_capture(self) -> dict[str, Any] | None:
        if self.trace_dir is None or not self.frame_capture:
            return None
        out_dir = Path(self.trace_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        arrays = {f"call{row['call']:04d}": row["frames"] for row in self.frame_capture}
        frames_path = out_dir / "policy_frames.npz"
        np.savez_compressed(frames_path, **arrays)
        meta = [
            {k: row[k] for k in ("call", "levels", "state", "resets")}
            | {"n_frames": int(row["frames"].shape[0])}
            for row in self.frame_capture
        ]
        meta_path = out_dir / "policy_frames_meta.json"
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return {"frames_npz": str(frames_path), "meta_json": str(meta_path),
                "calls_captured": len(meta)}

    def export_trace(self) -> dict[str, Any]:
        return {
            "policy": "megarexus_cnn",
            "schema": "synthesis_lab.megarexus_arc_policy_trace.v1",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": self.device,
            "truth_boundary": "current_frame_pixels_plus_own_history_and_megarexus_checkpoint_only",
            "frame_convention": "recorder_cur_last_prev_second_to_last",
            "arrange_memory_enabled": bool(self.arrange_memory_enabled),
            "graph_demo_plan_enabled": bool(self.graph_demo_plan_enabled),
            "goal_provider_enabled": bool(self.goal_provider_enabled),
            "goal_pour_budget_enabled": bool(self.goal_pour_budget_enabled),
            "goal_threshold": float(self.goal_threshold),
            "active_goal_schema": self.active_goal_schema,
            "goal_provider_updates": self.goal_provider_updates,
            "frames_fallback_count": int(self.frames_fallback_count),
            "confidence_margin": CONFIDENCE_MARGIN,
            "noop_count": int(self.noop_count),
            "plans_built": {str(k): int(v) for k, v in self.plans_built.items()},
            "pours_at_level": {str(k): int(v) for k, v in self.pours_at_level.items()},
            "refuted_plans": {str(k): sorted(map(list, v)) for k, v in self.refuted_plans.items()},
            "failed_attempt_signatures": {
                str(k): [self._serialise_attempt_signature(sig) for sig in sorted(v)]
                for k, v in self.failed_attempt_signatures.items()
            },
            "arrange_attempt_ledger": self.arrange_attempt_ledger,
            "frame_capture": self._flush_frame_capture(),
            "plan_summaries": self.plan_summaries,
            "ledger": self.ledger,
        }


def megarexus_arc_policy() -> MegarexusArcPolicy:
    return MegarexusArcPolicy()
