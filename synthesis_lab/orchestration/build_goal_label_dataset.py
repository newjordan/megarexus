#!/usr/bin/env python3
"""Build a hindsight goal-label dataset from recorded completion windows.

Labels are derived from real scoring animations: late-filled components become
target-lock candidates, early-changing opposite components become key
candidates, and paint-color blocks adjacent to expanding source tubes become
paint-output candidates.
The output is a training dataset for the additive CNN goal head; it is not
itself solve evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from synthesis_lab.orchestration.extract_level0_goal_schema import (  # noqa: E402
    BOUNDARY as STATIC_BOUNDARY,
    PALETTE,
    component_change_rows,
    component_mask,
    find_level_windows,
    find_scoring_end,
    frame_hash,
    load_episode,
    raw_pixel_components_with_labels,
    select_keys,
    select_sources,
    select_targets,
    sha256_file,
)
from synthesis_lab.orchestration.no_imitation_guard import (  # noqa: E402
    assert_no_imitation_lineage,
)


SCHEMA = "synthesis_lab.megarexus.goal_label_dataset.v2"
BOUNDARY = (
    "Hindsight labels from recorded completion animations. Not control truth, "
    "not solve evidence, not a wired runtime goal provider, and not CNN mastery."
)
CHANNELS = ("target_lock", "key_block", "paint_output")


def mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def stack_hash(masks: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(masks, dtype=np.uint8).tobytes()).hexdigest()


def source_sha(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    h = hashlib.sha256()
    for name in ("policy_frames_meta.json", "policy_frames.npz"):
        p = path / name
        h.update(name.encode("utf-8"))
        h.update(bytes.fromhex(sha256_file(p)))
    return h.hexdigest()


def parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
    else:
        p = Path(value)
        label = p.stem
        path = value
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("label cannot be empty")
    return label, Path(path)


def discover_policy_dirs(roots: Iterable[Path]) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for root in roots:
        for meta_path in sorted(root.glob("**/policy_frames_meta.json")):
            episode_dir = meta_path.parent
            if (episode_dir / "policy_frames.npz").exists():
                label = str(episode_dir).replace("/", "__").replace(":", "_")
                found.append((label, episode_dir))
    return found


def render_label_overlay(frame: np.ndarray, masks: np.ndarray, path: Path, scale: int = 8) -> None:
    arr = np.clip(np.asarray(frame, dtype=np.int16), 0, len(PALETTE) - 1)
    rgb = PALETTE[arr].astype(np.float32)
    colors = [
        np.array([255, 47, 79], dtype=np.float32),
        np.array([255, 216, 74], dtype=np.float32),
        np.array([56, 217, 255], dtype=np.float32),
    ]
    for idx, color in enumerate(colors):
        mask = masks[idx].astype(bool)
        rgb[mask] = 0.25 * rgb[mask] + 0.75 * color
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    image = image.resize((arr.shape[1] * scale, arr.shape[0] * scale), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    labels = [("target", "#ff2f4f"), ("key", "#ffd84a"), ("paint output", "#38d9ff")]
    x = 8
    for text, color in labels:
        draw.text((x, 8), text, fill=color)
        x += 72
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def adjacent_label_indices(label_map: np.ndarray, component: dict[str, Any]) -> set[int]:
    mask = component_mask(label_map, component)
    near = np.zeros(mask.shape, dtype=bool)
    near[1:, :] |= mask[:-1, :]
    near[:-1, :] |= mask[1:, :]
    near[:, 1:] |= mask[:, :-1]
    near[:, :-1] |= mask[:, 1:]
    ring = near & ~mask
    labels = {int(v) for v in np.unique(label_map[ring]).tolist()}
    labels.discard(0)
    labels.discard(int(component["label_index"]))
    return labels


def select_paint_outputs(
    components: list[dict[str, Any]],
    label_map: np.ndarray,
    source_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(component["component_id"]): component for component in components}
    by_label = {int(component["label_index"]): component for component in components}
    paint_values: set[int] = set()
    paint_seed_components: list[dict[str, Any]] = []
    for row in source_candidates:
        source = by_id.get(str(row["component_id"]))
        if source is None:
            continue
        source_value = int(source["raw_pixel_value"])
        adjacent = [
            by_label[label]
            for label in adjacent_label_indices(label_map, source)
            if label in by_label and int(by_label[label]["raw_pixel_value"]) != source_value
        ]
        adjacent.sort(key=lambda comp: (int(comp["area_pixels"]), comp["bbox_xyxy"]))
        for comp in adjacent:
            # The paint color block is the small color chip touching the old
            # expanding source tube. All chips with that raw value are outputs.
            if int(comp["area_pixels"]) <= 64:
                paint_values.add(int(comp["raw_pixel_value"]))
                paint_seed_components.append(comp)
    outputs = [
        comp
        for comp in components
        if int(comp["raw_pixel_value"]) in paint_values and int(comp["area_pixels"]) <= 64
    ]
    outputs.sort(key=lambda comp: (int(comp["raw_pixel_value"]), comp["bbox_xyxy"]))
    return [
        {
            "component_id": str(comp["component_id"]),
            "raw_pixel_value": int(comp["raw_pixel_value"]),
            "area_pixels": int(comp["area_pixels"]),
            "bbox_xyxy": [int(v) for v in comp["bbox_xyxy"]],
            "center_xy": [round(float(v), 3) for v in comp["center_xy"]],
            "candidate_role": "paint_color_block_output_candidate",
            "seeded_by_adjacent_source_tube": any(
                str(comp["component_id"]) == str(seed["component_id"])
                for seed in paint_seed_components
            ),
            "evidence_status": "static_hindsight_hypothesis",
            "admission_status": "not_admitted",
            "reasons": [
                "paint_color_block_adjacent_to_expanding_source_tube",
                "all_same_paint_color_blocks_are_outputs",
            ],
        }
        for comp in outputs
    ]


def make_masks(
    pre_frame: np.ndarray,
    components: list[dict[str, Any]],
    label_map: np.ndarray,
    target_candidates: list[dict[str, Any]],
    key_candidates: list[dict[str, Any]],
    paint_output_candidates: list[dict[str, Any]],
) -> np.ndarray:
    masks = np.zeros((3, 64, 64), dtype=np.uint8)
    by_id = {str(component["component_id"]): component for component in components}
    for row in target_candidates:
        comp = by_id[str(row["component_id"])]
        masks[0][component_mask(label_map, comp)] = 1
    for row in key_candidates[:1]:
        comp = by_id[str(row["component_id"])]
        masks[1][component_mask(label_map, comp)] = 1
    for row in paint_output_candidates:
        comp = by_id[str(row["component_id"])]
        masks[2][component_mask(label_map, comp)] = 1
    return masks


def analyze_completion(
    *,
    source_kind: str,
    source_label: str,
    source_path: Path,
    source_hash: str,
    sample_id: str,
    pre_frame: np.ndarray,
    prev_frame: np.ndarray | None,
    completion_frames: np.ndarray,
    from_level: int,
    to_level: int,
    location: dict[str, Any],
) -> dict[str, Any] | None:
    pre_frame = np.asarray(pre_frame, dtype=np.int16)
    completion_frames = np.asarray(completion_frames, dtype=np.int16)
    scoring_end, reset_info = find_scoring_end(completion_frames)
    scoring_frames = completion_frames[:scoring_end]
    if len(scoring_frames) == 0:
        return None
    components, label_map = raw_pixel_components_with_labels(pre_frame)
    component_count_by_value = Counter(int(c["raw_pixel_value"]) for c in components)
    change_rows = component_change_rows(pre_frame, scoring_frames, components, label_map)
    target_candidates = select_targets(change_rows, len(scoring_frames))
    key_candidates = select_keys(change_rows, target_candidates, dict(component_count_by_value))
    source_candidates = select_sources(pre_frame, scoring_frames, components, dict(component_count_by_value))
    paint_output_candidates = select_paint_outputs(components, label_map, source_candidates)
    if not target_candidates or not key_candidates or not source_candidates or not paint_output_candidates:
        return None
    masks = make_masks(
        pre_frame,
        components,
        label_map,
        target_candidates,
        key_candidates,
        paint_output_candidates,
    )
    group_key = "|".join(
        [
            str(from_level),
            str(to_level),
            frame_hash(pre_frame),
            stack_hash(masks),
        ]
    )
    return {
        "sample_id": sample_id,
        "source_kind": source_kind,
        "source_label": source_label,
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "location": location,
        "from_level": int(from_level),
        "to_level": int(to_level),
        "frame": np.asarray(pre_frame, dtype=np.int8),
        "prev_frame": np.asarray(prev_frame if prev_frame is not None else pre_frame, dtype=np.int8),
        "goal_masks": masks,
        "pre_frame_sha256": frame_hash(pre_frame),
        "goal_masks_sha256": stack_hash(masks),
        "group_key": group_key,
        "scoring_frame_count": int(len(scoring_frames)),
        "completion_frame_count": int(len(completion_frames)),
        "reset_detection": reset_info,
        "candidate_summary": {
            "targets": [
                {
                    "component_id": row["component_id"],
                    "raw_pixel_value": row["raw_pixel_value"],
                    "bbox_xyxy": row["bbox_xyxy"],
                    "center_xy": row["center_xy"],
                    "final_dominant_value": row["final_dominant_value"],
                    "score": row["score"],
                }
                for row in target_candidates
            ],
            "key": {
                "component_id": key_candidates[0]["component_id"],
                "raw_pixel_value": key_candidates[0]["raw_pixel_value"],
                "bbox_xyxy": key_candidates[0]["bbox_xyxy"],
                "center_xy": key_candidates[0]["center_xy"],
                "target_centroid_relation": key_candidates[0]["target_centroid_relation"],
                "score": key_candidates[0]["score"],
            },
            "source_tube": {
                "component_id": source_candidates[0]["component_id"],
                "raw_pixel_value": source_candidates[0]["raw_pixel_value"],
                "bbox_xyxy": source_candidates[0]["bbox_xyxy"],
                "center_xy": source_candidates[0]["center_xy"],
                "value_growth_pixels_during_scoring": source_candidates[0]["value_growth_pixels_during_scoring"],
                "score": source_candidates[0]["score"],
            },
            "paint_outputs": paint_output_candidates,
            "mask_pixels": {
                "target_lock": int(masks[0].sum()),
                "key_block": int(masks[1].sum()),
                "paint_output": int(masks[2].sum()),
            },
        },
    }


def iter_policy_samples(label: str, episode_dir: Path) -> list[dict[str, Any]]:
    rows = load_episode(episode_dir)
    h = source_sha(episode_dir)
    samples: list[dict[str, Any]] = []
    for pre_idx, win_idx in find_level_windows(rows, from_level=None):
        pre = np.asarray(rows[pre_idx]["frames"][-1], dtype=np.int16)
        prev = np.asarray(rows[pre_idx - 1]["frames"][-1], dtype=np.int16) if pre_idx > 0 else pre
        frames = np.asarray(rows[win_idx]["frames"], dtype=np.int16)
        sample = analyze_completion(
            source_kind="policy_frames",
            source_label=label,
            source_path=episode_dir,
            source_hash=h,
            sample_id=f"policy::{label}::call{int(rows[win_idx]['call']):04d}",
            pre_frame=pre,
            prev_frame=prev,
            completion_frames=frames,
            from_level=int(rows[pre_idx].get("levels") or 0),
            to_level=int(rows[win_idx].get("levels") or 0),
            location={
                "pre_call": int(rows[pre_idx]["call"]),
                "completion_call": int(rows[win_idx]["call"]),
            },
        )
        if sample is not None:
            samples.append(sample)
    return samples


def iter_shard_samples(label: str, shard_path: Path) -> list[dict[str, Any]]:
    h = source_sha(shard_path)
    samples: list[dict[str, Any]] = []
    trajectories = torch.load(shard_path, map_location="cpu", weights_only=False)
    for traj_idx, traj in enumerate(trajectories):
        steps = traj.get("steps", [])
        for idx in range(1, len(steps)):
            from_level = int(steps[idx - 1].get("levels_completed") or 0)
            to_level = int(steps[idx].get("levels_completed") or 0)
            if to_level <= from_level:
                continue
            pre = np.asarray(steps[idx - 1]["frame"], dtype=np.int16)
            prev = np.asarray(steps[idx - 2]["frame"], dtype=np.int16) if idx > 1 else pre
            seq = []
            for j in range(idx, min(len(steps), idx + 40)):
                seq.append(np.asarray(steps[j]["frame"], dtype=np.int16))
                if j > idx + 1 and int(np.sum(seq[-1] != seq[-2])) > 512:
                    break
            sample = analyze_completion(
                source_kind="rollout_shard",
                source_label=label,
                source_path=shard_path,
                source_hash=h,
                sample_id=f"shard::{label}::traj{traj_idx:04d}::step{idx:04d}",
                pre_frame=pre,
                prev_frame=prev,
                completion_frames=np.stack(seq),
                from_level=from_level,
                to_level=to_level,
                location={
                    "rollout_id": str(traj.get("rollout_id", "")),
                    "trajectory_index": int(traj_idx),
                    "levelup_step_index": int(idx),
                },
            )
            if sample is not None:
                samples.append(sample)
    return samples


def assign_splits(samples: list[dict[str, Any]]) -> dict[str, Any]:
    for sample in samples:
        path = sample["source_path"]
        if sample["source_kind"] == "rollout_shard" and "_val" in Path(path).name:
            split = "val"
            reason = "rollout_val_shard"
        elif sample["source_kind"] == "policy_frames" and path.startswith("codexlab/runs/"):
            split = "val"
            reason = "live_policy_frame_holdout"
        else:
            split = "train"
            reason = "default_train_source"
        sample["split"] = split
        sample["split_reason"] = reason

    train_groups = {s["group_key"] for s in samples if s["split"] == "train"}
    val_groups = {s["group_key"] for s in samples if s["split"] == "val"}
    for sample in samples:
        sample["group_seen_in_train"] = bool(sample["group_key"] in train_groups)
    return {
        "train_samples": sum(1 for s in samples if s["split"] == "train"),
        "val_samples": sum(1 for s in samples if s["split"] == "val"),
        "unique_groups_total": len({s["group_key"] for s in samples}),
        "unique_groups_train": len(train_groups),
        "unique_groups_val": len(val_groups),
        "val_groups_seen_in_train": len(val_groups & train_groups),
        "val_groups_unseen_in_train": len(val_groups - train_groups),
        "by_level": {
            f"{a}->{b}": n
            for (a, b), n in sorted(Counter((s["from_level"], s["to_level"]) for s in samples).items())
        },
        "by_split": dict(Counter(s["split"] for s in samples)),
    }


def json_safe_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in sample.items()
        if key not in {"frame", "prev_frame", "goal_masks"}
    }


def write_review(out_dir: Path, samples: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    frames_dir = out_dir / "frames"
    groups: dict[str, dict[str, Any]] = {}
    for sample in samples:
        groups.setdefault(sample["group_key"], sample)
    rows = []
    for idx, sample in enumerate(list(groups.values())[:24]):
        rel = Path("frames") / f"group_{idx:03d}_overlay.png"
        render_label_overlay(sample["frame"], sample["goal_masks"], out_dir / rel)
        rows.append((sample, str(rel)))
    cards = []
    for sample, rel in rows:
        pixels = sample["candidate_summary"]["mask_pixels"]
        cards.append(
            f"""
            <article>
              <h3>{html.escape(sample['from_level'].__str__())} -&gt; {html.escape(sample['to_level'].__str__())}</h3>
              <img src="{html.escape(rel)}" alt="label overlay">
              <p><strong>split:</strong> {html.escape(sample['split'])} ({html.escape(sample['split_reason'])})</p>
              <p><strong>pixels:</strong> target {pixels['target_lock']}, key {pixels['key_block']}, output {pixels['paint_output']}</p>
              <p><strong>source:</strong> <code>{html.escape(sample['source_path'])}</code></p>
            </article>
            """
        )
    summary_rows = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(json.dumps(v, sort_keys=True))}</td></tr>"
        for k, v in manifest["summary"].items()
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Goal Label Dataset Review</title>
  <style>
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #111318; color: #e8edf4; }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 28px; }}
    h1, h2, h3 {{ letter-spacing: 0; }}
    .boundary {{ color: #f4d278; max-width: 980px; }}
    a {{ color: #8fc7ff; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 26px; }}
    th, td {{ border: 1px solid #2d3440; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ width: 260px; color: #cbd6e2; background: #1d2430; }}
    td {{ background: #151920; }}
    .deck {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 14px; }}
    article {{ border: 1px solid #2d3440; border-radius: 6px; padding: 12px; background: #171b22; }}
    article img {{ width: 100%; image-rendering: pixelated; background: #080a0e; }}
    code {{ color: #d7e6ff; word-break: break-word; }}
  </style>
</head>
<body>
<main>
  <h1>Goal Label Dataset Review</h1>
  <p class="boundary">{html.escape(BOUNDARY)}</p>
  <p>Raw manifest: <a href="manifest.json">manifest.json</a>. Dataset: <code>goal_label_dataset.pt</code>.</p>
  <h2>Summary</h2>
  <table>{summary_rows}</table>
  <h2>Unique Label Group Overlays</h2>
  <div class="deck">{''.join(cards)}</div>
</main>
</body>
</html>
"""
    frames_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html_text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-frames-root", type=Path, action="append", default=[])
    p.add_argument("--policy-frames-dir", type=parse_labeled_path, action="append", default=[])
    p.add_argument("--rollout-shard", type=parse_labeled_path, action="append", default=[])
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--condition", type=Path, default=None)
    args = p.parse_args(argv)

    explicit_inputs = {
        "policy_frames_root": [str(p) for p in args.policy_frames_root],
        "policy_frames_dir": [
            {"label": label, "path": str(path)}
            for label, path in args.policy_frames_dir
        ],
        "rollout_shard": [
            {"label": label, "path": str(path)}
            for label, path in args.rollout_shard
        ],
    }
    assert_no_imitation_lineage(
        explicit_inputs,
        purpose="goal-label dataset input",
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_dirs = list(args.policy_frames_dir) + discover_policy_dirs(args.policy_frames_root)
    discovered_inputs = [
        {"label": label, "path": str(path)}
        for label, path in policy_dirs
    ]
    assert_no_imitation_lineage(
        discovered_inputs,
        purpose="discovered policy-frame input",
    )
    all_samples: list[dict[str, Any]] = []
    skipped = []
    inputs = []
    for label, episode_dir in policy_dirs:
        try:
            samples = iter_policy_samples(label, episode_dir)
            all_samples.extend(samples)
            inputs.append({
                "kind": "policy_frames",
                "label": label,
                "path": str(episode_dir),
                "sha256": source_sha(episode_dir),
                "sample_count": len(samples),
            })
        except Exception as exc:
            skipped.append({"kind": "policy_frames", "label": label, "path": str(episode_dir), "error": repr(exc)})
    for label, shard in args.rollout_shard:
        try:
            samples = iter_shard_samples(label, shard)
            all_samples.extend(samples)
            inputs.append({
                "kind": "rollout_shard",
                "label": label,
                "path": str(shard),
                "sha256": source_sha(shard),
                "sample_count": len(samples),
            })
        except Exception as exc:
            skipped.append({"kind": "rollout_shard", "label": label, "path": str(shard), "error": repr(exc)})
    split_summary = assign_splits(all_samples)
    manifest = {
        "schema": SCHEMA,
        "run_label": out_dir.name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "working_directory": str(Path.cwd()),
        "command": " ".join(sys.argv),
        "condition_artifact": str(args.condition) if args.condition else None,
        "source_script": str(Path(__file__)),
        "source_script_sha256": sha256_file(Path(__file__)),
        "channels": CHANNELS,
        "inputs": inputs,
        "skipped_inputs": skipped,
        "summary": {
            "sample_count": len(all_samples),
            **split_summary,
            "boundary": BOUNDARY,
            "static_extraction_boundary": STATIC_BOUNDARY,
        },
    }
    dataset = {
        "schema": SCHEMA,
        "channels": CHANNELS,
        "samples": all_samples,
        "manifest": manifest,
    }
    torch.save(dataset, out_dir / "goal_label_dataset.pt")
    manifest["outputs"] = {
        "dataset_pt": {
            "path": str(out_dir / "goal_label_dataset.pt"),
            "sha256": sha256_file(out_dir / "goal_label_dataset.pt"),
        },
        "manifest_json": "manifest.json",
        "review_index_html": "index.html",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    write_review(out_dir, all_samples, manifest)
    print(json.dumps({"out_dir": str(out_dir), "sample_count": len(all_samples), **split_summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
