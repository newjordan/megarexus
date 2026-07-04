#!/usr/bin/env python3
"""Probe whether the goal head designates the next target after level 0.

The probe uses recorded policy-frame archives only:

1. Find a recorded 0->1 completion window.
2. Take the first stable frame after that completion as the post-level-0 input.
3. Find the later 1->2 completion window in the same episode.
4. Derive hindsight target/key/source masks from that later completion.
5. Run the CNN goal head on the post-level-0 input and compare to those masks.

This is a recorded-frame designation probe. It is not solve evidence and not
runtime action wiring evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXP_ROOT = _REPO_ROOT / "synthesis_lab" / "experiments" / "megarexus_cnn"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXP_ROOT))
if str(_EXP_ROOT.parents[1]) not in sys.path:
    sys.path.insert(0, str(_EXP_ROOT.parents[1]))

from dataset import NUM_COLORS, contra_planes  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402
from train import load_model_state_compatible, sha256_file  # noqa: E402
from synthesis_lab.orchestration.build_goal_label_dataset import (  # noqa: E402
    CHANNELS,
    analyze_completion,
    source_sha,
)
from synthesis_lab.orchestration.extract_level0_goal_schema import (  # noqa: E402
    BOUNDARY as STATIC_BOUNDARY,
    PALETTE,
    component_mask,
    find_level_windows,
    frame_hash,
    load_episode,
    raw_pixel_components_with_labels,
)


SCHEMA = "synthesis_lab.megarexus.goal_head_post_level0_probe.v1"
BOUNDARY = (
    "Recorded-frame post-level-0 designation probe. The CNN goal head is run "
    "on a real archived frame after the 0->1 completion, and the comparison "
    "target is a hindsight mask derived from the later 1->2 completion in the "
    "same archive. This is not solve evidence, not live runtime goal-provider "
    "wiring, and not action-choice evidence."
)
DEFAULT_PASS_TARGET_IOU = 0.90
DEFAULT_PASS_MEAN_IOU = 0.75
LABEL_SPACES = ("completion_pre", "current_frame_components")


def mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
    else:
        p = Path(value)
        label = p.parent.name or p.name
        path = value
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("label cannot be empty")
    return label, Path(path)


def encode_grid(grid: np.ndarray) -> torch.Tensor:
    arr = np.asarray(grid, dtype=np.int64)
    onehot = torch.nn.functional.one_hot(torch.from_numpy(arr), NUM_COLORS).permute(2, 0, 1).float()
    planes = torch.from_numpy(contra_planes(arr.astype(np.int8)).astype(np.float32))
    return torch.cat([onehot, planes], dim=0)


def encode_stack(current: np.ndarray, previous: np.ndarray | None, stack_frames: int) -> torch.Tensor:
    cur = encode_grid(current)
    if stack_frames == 1:
        return cur
    prev = encode_grid(previous if previous is not None else current)
    return torch.cat([cur, prev], dim=0)


def render_grid(grid: np.ndarray, path: Path, *, scale: int = 8) -> None:
    arr = np.clip(np.asarray(grid, dtype=np.int16), 0, len(PALETTE) - 1)
    image = Image.fromarray(PALETTE[arr], mode="RGB")
    image = image.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.Resampling.NEAREST)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def colorize_masks(base: np.ndarray, masks: np.ndarray, *, alpha: float = 0.72) -> np.ndarray:
    arr = np.clip(np.asarray(base, dtype=np.int16), 0, len(PALETTE) - 1)
    rgb = PALETTE[arr].astype(np.float32)
    colors = [
        np.array([255, 47, 79], dtype=np.float32),
        np.array([255, 216, 74], dtype=np.float32),
        np.array([56, 217, 255], dtype=np.float32),
    ]
    for idx, color in enumerate(colors):
        mask = np.asarray(masks[idx]).astype(bool)
        rgb[mask] = (1.0 - alpha) * rgb[mask] + alpha * color
    return np.clip(rgb, 0, 255).astype(np.uint8)


def render_mask_overlay(
    base: np.ndarray,
    masks: np.ndarray,
    path: Path,
    *,
    title: str,
    scale: int = 8,
) -> None:
    rgb = colorize_masks(base, masks)
    image = Image.fromarray(rgb, mode="RGB")
    image = image.resize((rgb.shape[1] * scale, rgb.shape[0] * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), title, fill="#f0f4fb")
    draw.text((8, 26), "target=red  key=yellow  source=cyan", fill="#f0f4fb")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def render_comparison_overlay(
    base: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    path: Path,
    *,
    scale: int = 8,
) -> None:
    arr = np.clip(np.asarray(base, dtype=np.int16), 0, len(PALETTE) - 1)
    rgb = PALETTE[arr].astype(np.float32)
    true_colors = [
        np.array([255, 47, 79], dtype=np.float32),
        np.array([255, 216, 74], dtype=np.float32),
        np.array([56, 217, 255], dtype=np.float32),
    ]
    pred_colors = [
        np.array([255, 150, 165], dtype=np.float32),
        np.array([255, 238, 145], dtype=np.float32),
        np.array([140, 235, 255], dtype=np.float32),
    ]
    for idx, color in enumerate(true_colors):
        mask = np.asarray(true[idx]).astype(bool)
        rgb[mask] = 0.25 * rgb[mask] + 0.75 * color
    for idx, color in enumerate(pred_colors):
        mask = np.asarray(pred[idx]).astype(bool) & ~np.asarray(true[idx]).astype(bool)
        rgb[mask] = 0.18 * rgb[mask] + 0.82 * color
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    image = image.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    draw.text((8, 8), "true saturated; prediction-only pale", fill="#f0f4fb")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def mask_metrics(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ious = []
    for idx, name in enumerate(CHANNELS):
        p = np.asarray(pred[idx]).astype(bool)
        t = np.asarray(true[idx]).astype(bool)
        inter = int(np.logical_and(p, t).sum())
        union = int(np.logical_or(p, t).sum())
        pred_sum = int(p.sum())
        true_sum = int(t.sum())
        iou = float(inter / max(1, union))
        precision = float(inter / max(1, pred_sum))
        recall = float(inter / max(1, true_sum))
        out[name] = {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "intersection_pixels": inter,
            "union_pixels": union,
            "pred_pixels": pred_sum,
            "true_pixels": true_sum,
        }
        ious.append(iou)
    out["mean_iou"] = float(sum(ious) / max(1, len(ious)))
    return out


def bbox_for_mask(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.nonzero(np.asarray(mask).astype(bool))
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def centroid_for_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(np.asarray(mask).astype(bool))
    if len(xs) == 0:
        return None
    return [round(float(xs.mean()), 3), round(float(ys.mean()), 3)]


def designation_summary(prob: np.ndarray, pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for idx, name in enumerate(CHANNELS):
        t = np.asarray(true[idx]).astype(bool)
        p = np.asarray(pred[idx]).astype(bool)
        channel = np.asarray(prob[idx], dtype=np.float32)
        background = ~t
        out[name] = {
            "true_bbox_xyxy": bbox_for_mask(t),
            "pred_bbox_xyxy": bbox_for_mask(p),
            "true_center_xy": centroid_for_mask(t),
            "pred_center_xy": centroid_for_mask(p),
            "mean_prob_on_true_mask": float(channel[t].mean()) if t.any() else 0.0,
            "max_prob_on_true_mask": float(channel[t].max()) if t.any() else 0.0,
            "max_prob_off_true_mask": float(channel[background].max()) if background.any() else 0.0,
            "mask_sha256": mask_hash(t),
        }
    return out


def frame_drift_summary(probe_frame: np.ndarray, label_frame: np.ndarray, masks: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {
        "full_frame_changed_pixels": int(np.sum(np.asarray(probe_frame) != np.asarray(label_frame))),
        "full_frame_total_pixels": int(np.asarray(probe_frame).size),
    }
    for idx, name in enumerate(CHANNELS):
        mask = np.asarray(masks[idx]).astype(bool)
        changed = int(np.sum(np.asarray(probe_frame)[mask] != np.asarray(label_frame)[mask]))
        total = int(mask.sum())
        out[name] = {
            "label_mask_pixels": total,
            "changed_pixels_between_probe_and_label_pre": changed,
            "changed_fraction_between_probe_and_label_pre": float(changed / max(1, total)),
        }
    return out


def select_probe_component_mask(
    probe_frame: np.ndarray,
    *,
    raw_pixel_value: int,
    target_area: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    components, label_map = raw_pixel_components_with_labels(np.asarray(probe_frame, dtype=np.int16))
    same_value = [
        comp for comp in components
        if int(comp["raw_pixel_value"]) == int(raw_pixel_value)
    ]
    if not same_value:
        return None, {
            "mapped": False,
            "reason": "raw_pixel_value_not_present_in_probe_frame",
            "raw_pixel_value": int(raw_pixel_value),
            "target_area": int(target_area),
        }
    selected = min(
        same_value,
        key=lambda comp: (
            abs(int(comp["area_pixels"]) - int(target_area)),
            -int(comp["area_pixels"]),
            str(comp["component_id"]),
        ),
    )
    mask = component_mask(label_map, selected).astype(np.uint8)
    return mask, {
        "mapped": True,
        "reason": "selected_same_raw_value_component_closest_area_in_probe_frame",
        "raw_pixel_value": int(raw_pixel_value),
        "target_area": int(target_area),
        "selected_component_id": str(selected["component_id"]),
        "selected_area_pixels": int(selected["area_pixels"]),
        "selected_bbox_xyxy": [int(v) for v in selected["bbox_xyxy"]],
        "selected_center_xy": [round(float(v), 3) for v in selected["center_xy"]],
        "candidate_count_same_value": int(len(same_value)),
    }


def current_frame_component_masks(
    probe_frame: np.ndarray,
    completion_pre_frame: np.ndarray,
    completion_masks: np.ndarray,
    candidate_summary: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Map dynamic goal roles from the later completion-pre frame to now.

    Target locks are spatial goal locations and are expected to be stable.
    The key/source roles identify current components, so this maps their raw
    values back into the post-level-0 frame.
    """
    adjusted = np.asarray(completion_masks, dtype=np.uint8).copy()
    mapping: dict[str, Any] = {
        "label_space": "current_frame_components",
        "target_lock": {
            "mapped": True,
            "reason": "target_lock_is_spatial_goal_location_kept_from_completion_pre_mask",
            "changed_pixels_between_probe_and_completion_pre_on_mask": int(
                np.sum(
                    np.asarray(probe_frame)[completion_masks[0].astype(bool)]
                    != np.asarray(completion_pre_frame)[completion_masks[0].astype(bool)]
                )
            ),
        },
    }
    key = candidate_summary.get("key") or {}
    key_mask, key_info = select_probe_component_mask(
        probe_frame,
        raw_pixel_value=int(key.get("raw_pixel_value", -9999)),
        target_area=int(key.get("area_pixels", int(completion_masks[1].sum()))),
    )
    if key_mask is not None:
        adjusted[1] = key_mask
    mapping["key_block"] = key_info

    source = candidate_summary.get("source") or {}
    source_mask, source_info = select_probe_component_mask(
        probe_frame,
        raw_pixel_value=int(source.get("raw_pixel_value", -9999)),
        target_area=int(source.get("area_pixels", int(completion_masks[2].sum()))),
    )
    if source_mask is not None:
        adjusted[2] = source_mask
    mapping["paint_source"] = source_info
    return adjusted, mapping


def find_first_stable_after(rows: list[dict[str, Any]], start_idx: int, before_idx: int, level: int) -> int | None:
    for idx in range(start_idx + 1, before_idx):
        if int(rows[idx].get("levels") or 0) != level:
            continue
        n_frames = int(rows[idx].get("n_frames") or len(rows[idx]["frames"]))
        if n_frames == 1:
            return idx
    return None


def build_cases(
    *,
    episode_label: str,
    episode_dir: Path,
    after_level: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = load_episode(episode_dir)
    episode_hash = source_sha(episode_dir)
    windows = find_level_windows(rows, from_level=None)
    completion_samples = []
    skipped = []
    for pre_idx, win_idx in windows:
        pre_frame = np.asarray(rows[pre_idx]["frames"][-1], dtype=np.int16)
        prev_frame = np.asarray(rows[pre_idx - 1]["frames"][-1], dtype=np.int16) if pre_idx > 0 else pre_frame
        sample = analyze_completion(
            source_kind="policy_frames",
            source_label=episode_label,
            source_path=episode_dir,
            source_hash=episode_hash,
            sample_id=f"policy::{episode_label}::call{int(rows[win_idx]['call']):04d}",
            pre_frame=pre_frame,
            prev_frame=prev_frame,
            completion_frames=np.asarray(rows[win_idx]["frames"], dtype=np.int16),
            from_level=int(rows[pre_idx].get("levels") or 0),
            to_level=int(rows[win_idx].get("levels") or 0),
            location={
                "pre_call": int(rows[pre_idx]["call"]),
                "completion_call": int(rows[win_idx]["call"]),
                "pre_row_index": int(pre_idx),
                "completion_row_index": int(win_idx),
            },
        )
        if sample is None:
            skipped.append({
                "episode_label": episode_label,
                "completion_call": int(rows[win_idx]["call"]),
                "reason": "completion_window_did_not_yield_all_goal_roles",
            })
            continue
        completion_samples.append({
            "pre_idx": pre_idx,
            "win_idx": win_idx,
            "sample": sample,
        })

    cases = []
    for prev, current in zip(completion_samples, completion_samples[1:]):
        prev_sample = prev["sample"]
        cur_sample = current["sample"]
        if int(prev_sample["from_level"]) != after_level:
            continue
        if int(prev_sample["to_level"]) != int(cur_sample["from_level"]):
            skipped.append({
                "episode_label": episode_label,
                "previous_completion_call": int(rows[prev["win_idx"]]["call"]),
                "current_completion_call": int(rows[current["win_idx"]]["call"]),
                "reason": "completion_levels_are_not_adjacent",
            })
            continue
        probe_idx = find_first_stable_after(
            rows,
            int(prev["win_idx"]),
            int(current["pre_idx"]),
            int(cur_sample["from_level"]),
        )
        if probe_idx is None:
            skipped.append({
                "episode_label": episode_label,
                "previous_completion_call": int(rows[prev["win_idx"]]["call"]),
                "current_completion_call": int(rows[current["win_idx"]]["call"]),
                "reason": "no_stable_post_completion_frame_before_next_completion",
            })
            continue
        probe_frame = np.asarray(rows[probe_idx]["frames"][-1], dtype=np.int16)
        prev_probe_frame = (
            np.asarray(rows[probe_idx - 1]["frames"][-1], dtype=np.int16)
            if probe_idx > 0
            else probe_frame
        )
        cases.append({
            "episode_label": episode_label,
            "episode_dir": str(episode_dir),
            "episode_sha256": episode_hash,
            "post_level0_row_index": int(probe_idx),
            "post_level0_call": int(rows[probe_idx]["call"]),
            "post_level0_frame": probe_frame,
            "post_level0_prev_frame": prev_probe_frame,
            "previous_completion": {
                "from_level": int(prev_sample["from_level"]),
                "to_level": int(prev_sample["to_level"]),
                "completion_call": int(rows[prev["win_idx"]]["call"]),
                "completion_row_index": int(prev["win_idx"]),
            },
            "target_completion": {
                "from_level": int(cur_sample["from_level"]),
                "to_level": int(cur_sample["to_level"]),
                "pre_call": int(rows[current["pre_idx"]]["call"]),
                "completion_call": int(rows[current["win_idx"]]["call"]),
                "pre_row_index": int(current["pre_idx"]),
                "completion_row_index": int(current["win_idx"]),
            },
            "target_sample": cur_sample,
        })
    return cases, skipped


@torch.no_grad()
def run_model_on_cases(
    *,
    cases: list[dict[str, Any]],
    ckpt_path: Path,
    device: str,
    threshold: float,
    label_space: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = payload.get("config", {})
    stack_frames = int(config.get("stack_frames", config.get("n_input_frames", 2)))
    model = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack_frames)).to(device)
    load_model_state_compatible(model, payload["model"])
    model.eval()
    rows = []
    for idx, case in enumerate(cases):
        frame = encode_stack(
            np.asarray(case["post_level0_frame"], dtype=np.int16),
            np.asarray(case["post_level0_prev_frame"], dtype=np.int16),
            stack_frames,
        ).unsqueeze(0).to(device)
        logits = model.forward_goal(frame)
        prob = torch.sigmoid(logits)[0].cpu().numpy()
        pred = (prob >= threshold).astype(np.uint8)
        completion_true = np.asarray(case["target_sample"]["goal_masks"], dtype=np.uint8)
        completion_label_frame = np.asarray(case["target_sample"]["frame"], dtype=np.int16)
        label_mapping: dict[str, Any] = {"label_space": "completion_pre"}
        if label_space == "current_frame_components":
            true, label_mapping = current_frame_component_masks(
                np.asarray(case["post_level0_frame"], dtype=np.int16),
                completion_label_frame,
                completion_true,
                case["target_sample"]["candidate_summary"],
            )
            comparison_label_frame = np.asarray(case["post_level0_frame"], dtype=np.int16)
        else:
            true = completion_true
            comparison_label_frame = completion_label_frame
        metrics = mask_metrics(pred, true)
        completion_pre_metrics = mask_metrics(pred, completion_true)
        rows.append({
            "case_id": f"case_{idx:03d}",
            "episode_label": case["episode_label"],
            "episode_dir": case["episode_dir"],
            "episode_sha256": case["episode_sha256"],
            "post_level0_call": case["post_level0_call"],
            "target_completion": case["target_completion"],
            "previous_completion": case["previous_completion"],
            "post_level0_frame_sha256": frame_hash(case["post_level0_frame"]),
            "target_label_pre_frame_sha256": frame_hash(completion_label_frame),
            "goal_masks_sha256": case["target_sample"]["goal_masks_sha256"],
            "comparison_label_space": label_space,
            "comparison_masks_sha256": hashlib.sha256(true.astype(np.uint8).tobytes()).hexdigest(),
            "label_mapping": label_mapping,
            "metrics": metrics,
            "completion_pre_metrics": completion_pre_metrics,
            "designation": designation_summary(prob, pred, true),
            "completion_pre_designation": designation_summary(prob, pred, completion_true),
            "frame_drift": frame_drift_summary(case["post_level0_frame"], completion_label_frame, completion_true),
            "candidate_summary": case["target_sample"]["candidate_summary"],
            "prediction_masks_sha256": hashlib.sha256(pred.astype(np.uint8).tobytes()).hexdigest(),
            "prediction_prob_sha256": hashlib.sha256(prob.astype(np.float32).tobytes()).hexdigest(),
            "_images": {
                "post_level0_frame": case["post_level0_frame"],
                "target_label_frame": comparison_label_frame,
                "completion_label_frame": completion_label_frame,
                "pred_masks": pred,
                "true_masks": true,
                "completion_true_masks": completion_true,
            },
        })
    model_info = {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "stack_frames": stack_frames,
        "checkpoint_channels": list(payload.get("channels", CHANNELS)),
    }
    return rows, model_info


def aggregate(rows: list[dict[str, Any]], *, pass_target_iou: float, pass_mean_iou: float) -> dict[str, Any]:
    if not rows:
        return {
            "case_count": 0,
            "status": "failed_no_cases",
            "pass_target_iou": pass_target_iou,
            "pass_mean_iou": pass_mean_iou,
        }
    channel_means = {}
    for name in CHANNELS:
        channel_means[name] = {
            "mean_iou": float(np.mean([row["metrics"][name]["iou"] for row in rows])),
            "min_iou": float(np.min([row["metrics"][name]["iou"] for row in rows])),
            "mean_precision": float(np.mean([row["metrics"][name]["precision"] for row in rows])),
            "mean_recall": float(np.mean([row["metrics"][name]["recall"] for row in rows])),
        }
    mean_iou = float(np.mean([row["metrics"]["mean_iou"] for row in rows]))
    min_mean_iou = float(np.min([row["metrics"]["mean_iou"] for row in rows]))
    min_target_iou = float(np.min([row["metrics"]["target_lock"]["iou"] for row in rows]))
    passed = min_target_iou >= pass_target_iou and min_mean_iou >= pass_mean_iou
    return {
        "case_count": len(rows),
        "status": "passed_recorded_frame_probe" if passed else "failed_recorded_frame_probe",
        "pass_target_iou": pass_target_iou,
        "pass_mean_iou": pass_mean_iou,
        "all_cases_passed": bool(passed),
        "mean_iou": mean_iou,
        "min_mean_iou": min_mean_iou,
        "min_target_lock_iou": min_target_iou,
        "channels": channel_means,
    }


def json_case(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "_images"}


def write_review(
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    result: dict[str, Any],
    max_cards: int,
) -> None:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for idx, row in enumerate(rows[:max_cards]):
        images = row["_images"]
        prefix = row["case_id"]
        raw_rel = Path("frames") / f"{prefix}_post_level0_raw.png"
        pred_rel = Path("frames") / f"{prefix}_prediction_on_post_level0.png"
        label_rel = Path("frames") / f"{prefix}_comparison_label.png"
        completion_label_rel = Path("frames") / f"{prefix}_completion_pre_label.png"
        cmp_rel = Path("frames") / f"{prefix}_comparison_on_post_level0.png"
        render_grid(images["post_level0_frame"], out_dir / raw_rel)
        render_mask_overlay(
            images["post_level0_frame"],
            images["pred_masks"],
            out_dir / pred_rel,
            title="CNN prediction on post-level-0 frame",
        )
        render_mask_overlay(
            images["target_label_frame"],
            images["true_masks"],
            out_dir / label_rel,
            title=f"Comparison label: {row['comparison_label_space']}",
        )
        render_mask_overlay(
            images["completion_label_frame"],
            images["completion_true_masks"],
            out_dir / completion_label_rel,
            title="Original later completion-pre label",
        )
        render_comparison_overlay(
            images["post_level0_frame"],
            images["pred_masks"],
            images["true_masks"],
            out_dir / cmp_rel,
        )
        metrics = row["metrics"]
        drift = row["frame_drift"]
        cards.append(
            f"""
            <article>
              <h3>{html.escape(row['case_id'])}: call {row['post_level0_call']} -&gt; target call {row['target_completion']['completion_call']}</h3>
              <p><strong>episode:</strong> <code>{html.escape(row['episode_dir'])}</code></p>
              <div class="quad">
                <figure><img src="{html.escape(str(raw_rel))}" alt="raw post-level-0 frame"><figcaption>post-level-0 raw</figcaption></figure>
                <figure><img src="{html.escape(str(pred_rel))}" alt="prediction overlay"><figcaption>CNN prediction on post-level-0</figcaption></figure>
                <figure><img src="{html.escape(str(label_rel))}" alt="comparison label overlay"><figcaption>comparison label: {html.escape(row['comparison_label_space'])}</figcaption></figure>
                <figure><img src="{html.escape(str(completion_label_rel))}" alt="completion-pre label overlay"><figcaption>original later completion-pre label</figcaption></figure>
                <figure><img src="{html.escape(str(cmp_rel))}" alt="comparison overlay"><figcaption>comparison on post-level-0 frame</figcaption></figure>
              </div>
              <table>
                <tr><th>mean IoU</th><td>{metrics['mean_iou']:.6f}</td></tr>
                <tr><th>target IoU</th><td>{metrics['target_lock']['iou']:.6f}</td></tr>
                <tr><th>key IoU</th><td>{metrics['key_block']['iou']:.6f}</td></tr>
                <tr><th>source IoU</th><td>{metrics['paint_source']['iou']:.6f}</td></tr>
                <tr><th>frame drift</th><td>{drift['full_frame_changed_pixels']} changed pixels between post-level-0 and label pre-frame</td></tr>
              </table>
            </article>
            """
        )
    summary_rows = []
    for key, value in result["summary"].items():
        summary_rows.append(
            f"<tr><th>{html.escape(str(key))}</th><td><code>{html.escape(json.dumps(value, sort_keys=True))}</code></td></tr>"
        )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Goal Head Post-Level-0 Probe</title>
  <style>
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #111318; color: #e8edf4; }}
    main {{ max-width: 1560px; margin: 0 auto; padding: 28px; }}
    h1, h2, h3 {{ letter-spacing: 0; }}
    .boundary {{ color: #f4d278; max-width: 1100px; }}
    a {{ color: #8fc7ff; }}
    code {{ color: #d7e6ff; word-break: break-word; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 22px; }}
    th, td {{ border: 1px solid #2d3440; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ width: 260px; background: #1d2430; }}
    td {{ background: #151920; }}
    article {{ border: 1px solid #2d3440; border-radius: 6px; padding: 14px; background: #171b22; margin: 16px 0; }}
    .quad {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; }}
    figcaption {{ color: #cbd6e2; font-size: 13px; margin-top: 4px; }}
    img {{ width: 100%; image-rendering: pixelated; background: #080a0e; }}
  </style>
</head>
<body>
<main>
  <h1>Goal Head Post-Level-0 Probe</h1>
  <p class="boundary">{html.escape(BOUNDARY)}</p>
  <p>Raw result: <a href="result.json">result.json</a>. Condition: <a href="CONDITION.json">CONDITION.json</a>.</p>
  <h2>Summary</h2>
  <table>{''.join(summary_rows)}</table>
  <h2>Frame Deck</h2>
  {''.join(cards)}
</main>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    out = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, list):
            out[key] = [
                [label, str(path)] if isinstance(item, tuple) else str(item)
                for item in value
                for label, path in ([item] if isinstance(item, tuple) else [])
            ] if key == "policy_frames_dir" else [str(v) for v in value]
        else:
            out[key] = value
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--goal-ckpt", type=Path, required=True)
    p.add_argument("--policy-frames-dir", type=parse_labeled_path, action="append", default=[])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--condition", type=Path, default=None)
    p.add_argument("--after-level", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--label-space", choices=LABEL_SPACES, default="completion_pre")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--pass-target-iou", type=float, default=DEFAULT_PASS_TARGET_IOU)
    p.add_argument("--pass-mean-iou", type=float, default=DEFAULT_PASS_MEAN_IOU)
    p.add_argument("--max-review-cards", type=int, default=12)
    args = p.parse_args(argv)

    if not args.policy_frames_dir:
        raise ValueError("at least one --policy-frames-dir is required")

    t0 = time.time()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_cases: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    inputs = []
    for label, path in args.policy_frames_dir:
        cases, skip = build_cases(episode_label=label, episode_dir=path, after_level=args.after_level)
        all_cases.extend(cases)
        skipped.extend(skip)
        inputs.append({
            "label": label,
            "path": str(path),
            "sha256": source_sha(path),
            "case_count": len(cases),
        })

    rows, model_info = run_model_on_cases(
        cases=all_cases,
        ckpt_path=args.goal_ckpt,
        device=device,
        threshold=args.threshold,
        label_space=args.label_space,
    )
    summary = aggregate(
        rows,
        pass_target_iou=float(args.pass_target_iou),
        pass_mean_iou=float(args.pass_mean_iou),
    )
    result = {
        "schema": SCHEMA,
        "run_label": args.out_dir.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "working_directory": str(Path.cwd()),
        "command": " ".join(sys.argv),
        "condition_artifact": str(args.condition) if args.condition else None,
        "source_script": str(Path(__file__)),
        "source_script_sha256": sha256_file(Path(__file__)),
        "device": device,
        "threshold": float(args.threshold),
        "label_space": args.label_space,
        "after_level": int(args.after_level),
        "model": model_info,
        "inputs": inputs,
        "summary": summary,
        "cases": [json_case(row) for row in rows],
        "skipped": skipped,
        "elapsed_seconds": round(time.time() - t0, 3),
        "boundary": BOUNDARY,
        "static_extraction_boundary": STATIC_BOUNDARY,
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    write_review(out_dir=args.out_dir, rows=rows, result=result, max_cards=args.max_review_cards)
    result["outputs"] = {
        "result_json": {
            "path": str(args.out_dir / "result.json"),
            "sha256": sha256_file(args.out_dir / "result.json"),
        },
        "review_index_html": {
            "path": str(args.out_dir / "index.html"),
            "sha256": sha256_file(args.out_dir / "index.html"),
        },
    }
    (args.out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "summary": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
