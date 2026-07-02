#!/usr/bin/env python3
"""One-shot sanity probe: does LeWM latent distance track SP80 solve progress?

Encodes a solved episode's frame sequence through the Loom LeWM Phase-0 encoder
and measures squared-L2 latent distance from each frame to the final (solved)
frame. If the JEPA latent space tracks semantic progress, that distance should
fall roughly monotonically over the trajectory. This is the falsification test
for the planned CONTRA->LeWM goal-cost bridge; it admits no truth and feeds no
solver path.

Requires the Loom repo (default /home/frosty40/Loom) and its Phase-0 checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_CODEXLAB = Path(__file__).resolve().parents[1]
if str(_CODEXLAB) not in sys.path:
    sys.path.insert(0, str(_CODEXLAB))

from contra_head.crucible.canon import canonicalize_grid  # noqa: E402
from contra_head.crucible.corpus import (  # noqa: E402
    ROOT,
    read_jsonl,
    rel,
    resolve_repo_path,
    sha256_file,
    source_hashes,
)

DEFAULT_EPISODES_DIR = _CODEXLAB / "datasets" / "sp80_lewm_episodes_v0"
DEFAULT_LOOM_ROOT = Path("/home/frosty40/Loom")
DEFAULT_CHECKPOINT_REL = Path("experiments/2026-05-23_phase0_production_rtx6k/bs64_lr1e-4/best.pt")
DEFAULT_OUT_ROOT = _CODEXLAB / "runs" / "_lewm_bridge"
REPORT_SCHEMA = "codexlab.lewm_latent_distance_probe.v1"
TRUTH_BOUNDARY_LITERAL = "latent_distance_probe_is_diagnostic_evidence_not_control_truth"
NUM_COLORS = 16

# Verdict thresholds are heuristics for reading the report, not admission gates.
SUPPORT_SPEARMAN_MAX = -0.5
SUPPORT_DECREASING_FRACTION_MIN = 0.6
AMBIGUOUS_SPEARMAN_MAX = -0.2


def pick_episode(episodes_jsonl: Path, episode_id: str | None) -> dict[str, Any]:
    episodes = read_jsonl(episodes_jsonl)
    if episode_id:
        for episode in episodes:
            if episode.get("episode_id") == episode_id:
                return episode
        raise SystemExit(f"episode_id not found in {episodes_jsonl}: {episode_id}")
    solved = [ep for ep in episodes if ep.get("solved")]
    if not solved:
        raise SystemExit(f"no solved episodes in {episodes_jsonl}; run export_episodes.py first")
    # Deterministic default: the longest solved episode gives the probe the
    # most steps to expose (non-)monotonicity.
    solved.sort(key=lambda ep: (-int(ep.get("step_count") or 0), str(ep.get("episode_id"))))
    return solved[0]


def episode_frames(episode: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Return [T+1, H, W] frame stack in display row-major orientation."""
    steps = episode["steps"]
    refs: list[tuple[dict[str, Any], int]] = [(steps[0]["before_grid"], int(steps[0]["rotation_quarters"] or 0))]
    refs.extend((step["after_grid"], int(step["rotation_quarters"] or 0)) for step in steps)
    frames: list[np.ndarray] = []
    frame_meta: list[dict[str, Any]] = []
    for ref, rotation in refs:
        path = resolve_repo_path(ref["path"])
        grid = np.load(path)
        if grid.ndim != 2:
            raise SystemExit(f"unexpected grid shape {grid.shape} at {path}")
        if grid.min() < 0 or grid.max() >= NUM_COLORS:
            raise SystemExit(f"grid values outside [0, {NUM_COLORS}) at {path}")
        # Saved grids are indexed grid[x, y]; the encoder expects row-major
        # [H, W] images, so canonicalize to display orientation then transpose.
        display = canonicalize_grid(grid, rotation).T
        frames.append(display.astype(np.int64))
        frame_meta.append({"path": ref["path"], "npy_sha256": ref["npy_sha256"], "rotation_quarters": rotation})
    return np.stack(frames, axis=0), frame_meta


def load_encoder(loom_root: Path, checkpoint: Path, device: str):
    import torch

    loom_src = loom_root / "src"
    if not loom_src.is_dir():
        raise SystemExit(f"Loom src not found: {loom_src}")
    if str(loom_src) not in sys.path:
        sys.path.insert(0, str(loom_src))
    from lewm.encoder import EncoderConfig
    from lewm.model import LeWM, LeWMConfig
    from lewm.predictor import PredictorConfig

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    encoder_config = dict(ckpt["encoder_config"])
    model = LeWM(
        LeWMConfig(
            encoder=EncoderConfig(**encoder_config),
            predictor=PredictorConfig(**dict(ckpt["predictor_config"])),
        )
    )
    model.load_state_dict(ckpt["model_state"])
    encoder = model.encoder.to(device).eval()
    meta = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "encoder_config": encoder_config,
        "epoch": ckpt.get("epoch"),
        "best_val_pred": ckpt.get("best_val_pred"),
    }
    return encoder, meta


def encode_frames(encoder, frames: np.ndarray, device: str, batch_size: int = 32) -> np.ndarray:
    """[T, H, W] int frames -> [T, latent_dim] float latents."""
    import torch

    # Mirrors Loom's frame_to_onehot(): 16-channel one-hot, [C, H, W] float.
    indices = torch.from_numpy(frames)
    onehot = torch.nn.functional.one_hot(indices, NUM_COLORS).permute(0, 3, 1, 2).float()
    chunks = []
    with torch.inference_mode():
        for start in range(0, onehot.shape[0], batch_size):
            chunk = onehot[start : start + batch_size].to(device)
            chunks.append(encoder(chunk).cpu())
    return torch.cat(chunks, dim=0).numpy().astype(np.float64)


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 3:
        return None

    def ranks(values: np.ndarray) -> np.ndarray:
        order = np.argsort(values, kind="stable")
        ranked = np.empty(len(values), dtype=np.float64)
        ranked[order] = np.arange(1, len(values) + 1, dtype=np.float64)
        for value in np.unique(values):
            mask = values == value
            if mask.sum() > 1:
                ranked[mask] = ranked[mask].mean()
        return ranked

    rx, ry = ranks(x), ranks(y)
    sx, sy = rx.std(), ry.std()
    if sx == 0 or sy == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def distance_stats(distances: np.ndarray) -> dict[str, Any]:
    """Stats over d[t] for t=0..T; d[T] == 0 by construction, so headline
    monotonicity stats exclude the final anchor point."""
    t_excl = np.arange(len(distances) - 1, dtype=np.float64)
    d_excl = distances[:-1]
    diffs = np.diff(distances)
    decreasing_fraction = float((diffs < 0).mean()) if len(diffs) else None
    return {
        "frame_count": int(len(distances)),
        "d_first": float(distances[0]),
        "d_min": float(distances.min()),
        "d_max": float(distances.max()),
        "spearman_rho_vs_time_excl_final": spearman_rho(t_excl, d_excl),
        "spearman_rho_vs_time_incl_final": spearman_rho(
            np.arange(len(distances), dtype=np.float64), distances
        ),
        "decreasing_step_fraction": decreasing_fraction,
        "distances": [float(v) for v in distances],
    }


def verdict(latent_final: dict[str, Any]) -> str:
    rho = latent_final.get("spearman_rho_vs_time_excl_final")
    frac = latent_final.get("decreasing_step_fraction")
    if rho is None or frac is None:
        return "insufficient_data"
    if rho <= SUPPORT_SPEARMAN_MAX and frac >= SUPPORT_DECREASING_FRACTION_MIN:
        return "supports_latent_progress"
    if rho <= AMBIGUOUS_SPEARMAN_MAX:
        return "ambiguous"
    return "does_not_support_latent_progress"


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    episodes_dir = args.episodes_dir if args.episodes_dir.is_absolute() else ROOT / args.episodes_dir
    episode = pick_episode(episodes_dir / "episodes.jsonl", args.episode_id)
    frames, frame_meta = episode_frames(episode)

    checkpoint = args.checkpoint if args.checkpoint is not None else args.loom_root / DEFAULT_CHECKPOINT_REL
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    encoder, encoder_meta = load_encoder(args.loom_root, checkpoint, device)
    latents = encode_frames(encoder, frames, device, batch_size=args.batch_size)

    d_final = ((latents - latents[-1]) ** 2).sum(axis=1)
    d_init = ((latents - latents[0]) ** 2).sum(axis=1)
    pixel_hamming_final = (frames != frames[-1]).reshape(len(frames), -1).sum(axis=1).astype(np.float64)

    latent_final_stats = distance_stats(d_final)
    report = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "episode": {
            "episode_id": episode["episode_id"],
            "run_dir": episode["run_dir"],
            "level_id": episode["level_id"],
            "solved": episode["solved"],
            "step_count": episode["step_count"],
            "discontinuity_count": episode["discontinuity_count"],
        },
        "frames": {"count": int(len(frames)), "first": frame_meta[0], "last": frame_meta[-1]},
        "encoder": encoder_meta,
        "device": device,
        "loom_root": str(args.loom_root),
        "distance_metric": "squared_l2_sum_on_projected_cls_latent",
        "latent_distance_to_final": latent_final_stats,
        "latent_distance_to_initial": distance_stats_no_anchor(d_init),
        "pixel_hamming_to_final": distance_stats(pixel_hamming_final),
        "verdict": verdict(latent_final_stats),
        "verdict_thresholds": {
            "supports": {
                "spearman_rho_vs_time_excl_final_max": SUPPORT_SPEARMAN_MAX,
                "decreasing_step_fraction_min": SUPPORT_DECREASING_FRACTION_MIN,
            },
            "ambiguous_spearman_rho_max": AMBIGUOUS_SPEARMAN_MAX,
        },
        "source_hashes": source_hashes(["codexlab/lewm_bridge/latent_distance_probe.py"]),
        "truth_boundary": {"boundary": TRUTH_BOUNDARY_LITERAL},
    }
    return report


def distance_stats_no_anchor(distances: np.ndarray) -> dict[str, Any]:
    """Distance-to-initial has its trivial zero at t=0, not t=T; report the
    full-series correlation (should be positive if latents track progress)."""
    series = [float(v) for v in distances]
    rho = spearman_rho(np.arange(1, len(distances), dtype=np.float64), distances[1:])
    return {
        "frame_count": int(len(distances)),
        "d_last": float(distances[-1]),
        "d_min": float(distances.min()),
        "d_max": float(distances.max()),
        "spearman_rho_vs_time_excl_initial": rho,
        "distances": series,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, default=DEFAULT_EPISODES_DIR)
    parser.add_argument("--episode-id", type=str, default=None, help="default: longest solved episode")
    parser.add_argument("--loom-root", type=Path, default=DEFAULT_LOOM_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None, help=f"default: <loom-root>/{DEFAULT_CHECKPOINT_REL}")
    parser.add_argument("--device", type=str, default=None, help="default: cuda if available else cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_probe(args)
    out_root = args.out_root if args.out_root.is_absolute() else ROOT / args.out_root
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = out_root / f"latent_distance_probe_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    report["report_json"] = rel(report_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "report_json": report["report_json"],
        "episode_id": report["episode"]["episode_id"],
        "level_id": report["episode"]["level_id"],
        "frame_count": report["frames"]["count"],
        "device": report["device"],
        "spearman_rho_vs_time_excl_final": report["latent_distance_to_final"]["spearman_rho_vs_time_excl_final"],
        "decreasing_step_fraction": report["latent_distance_to_final"]["decreasing_step_fraction"],
        "verdict": report["verdict"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
