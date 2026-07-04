"""Evaluate multiple Megarexus checkpoints on one explicit valset.

This is the same no-loss gate as gate_eval.py, but it loads the validation
dataset once and evaluates a list of checkpoints in sequence. It uses
train.py's evaluate() unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LAB_ROOT / "experiments" / "megarexus_cnn"))

from dataset import MegarexusTransitions  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402
from train import evaluate, load_model_state_compatible  # noqa: E402


def load_shards(args: argparse.Namespace) -> list[Path]:
    if args.val_rollout_shards:
        return list(args.val_rollout_shards)
    if args.val_shards_from is None:
        raise ValueError("need --val-rollout-shards or --val-shards-from")
    ref = torch.load(args.val_shards_from, map_location="cpu", weights_only=False)
    return [Path(s) for s in ref["config"]["val_rollout_shards"]]


def load_model(path: Path, device: str) -> tuple[MegarexusCNN, int]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    stack = int(ckpt["config"].get("stack_frames", 2))
    model = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)).to(device).eval()
    load_model_state_compatible(model, ckpt["model"])
    return model, stack


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", type=Path, nargs="+", required=True)
    p.add_argument("--val-shards-from", type=Path, default=None)
    p.add_argument("--val-rollout-shards", type=Path, nargs="*", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers; use 0 on memory-constrained rigs")
    p.add_argument("--out-json", type=Path, required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    first_payload = torch.load(args.ckpts[0], map_location="cpu", weights_only=False)
    stack = int(first_payload["config"].get("stack_frames", 2))
    shards = load_shards(args)
    print(f"[gate-many] val shards: {[str(s) for s in shards]}")
    val_ds = MegarexusTransitions("val", val_rollout_shards=shards,
                                  n_input_frames=stack)
    loader = DataLoader(val_ds, batch_size=args.batch_size,
                        num_workers=args.num_workers)
    print(f"[gate-many] dataloader num_workers={args.num_workers}")

    results = []
    for ckpt_path in args.ckpts:
        print(f"[gate-many] ckpt={ckpt_path}", flush=True)
        model, ckpt_stack = load_model(ckpt_path, device)
        if ckpt_stack != stack:
            raise ValueError(
                f"{ckpt_path}: stack_frames={ckpt_stack}, expected {stack}"
            )
        metrics = evaluate(model, loader, device)
        rounded = {k: round(float(v), 6) for k, v in sorted(metrics.items())}
        result = {
            "ckpt": str(ckpt_path),
            "metrics": rounded,
        }
        results.append(result)
        print(json.dumps({
            "ckpt": str(ckpt_path),
            "bc": rounded["bc/action_type_acc"],
            "catch": rounded["catch/acc"],
            "level_tp": rounded["level/true_pos"],
            "level_pos": rounded["level/positives"],
            "level_fp": rounded["level/false_pos"],
            "settle_iou": rounded["settle/iou"],
            "changed": rounded["fwd/changed_cell_acc"],
        }))

    report = {
        "schema": "synthesis_lab.gate_eval_many.v0",
        "val_shards": [str(s) for s in shards],
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "results": results,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[gate-many] report -> {args.out_json}")


if __name__ == "__main__":
    main()
