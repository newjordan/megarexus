#!/usr/bin/env python3
"""Offline LoRA mechanics dry-run for the SP80 primitive-control dataset.

This is not a Qwen LoRA claim. It trains a small coordinate-free primitive
chooser with a frozen base linear map plus trainable low-rank adapter matrices.
The point is to prove the LoRA data/control harness path before attaching a
real Qwen-family base model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn

from contra_head import contra_spec
from players.sp80.registry import CONTROL_PRIMITIVES
from sp80_lora_adapter import (
    NUMERIC_KEYS,
    FrozenLinearLoRA,
    build_vocab,
    coordinate_free_text,
    encode_rows,
    row_target,
)
from sp80_check_dataset_leakage import check_leakage
from sp80_validate_dataset_rows import validate_dataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "codexlab" / "sp80_lora_dry_run.py"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    resolved.relative_to(ROOT)
    return resolved


def relpath(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise SystemExit(f"{path}:{line_no}: row is not an object")
        rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def accuracy_and_predictions(
    *,
    model: nn.Module,
    x: torch.Tensor,
    rows: list[dict[str, Any]],
    y: torch.Tensor,
    split: str,
) -> tuple[float, list[dict[str, Any]]]:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)
    predictions: list[dict[str, Any]] = []
    correct = 0
    for idx, row in enumerate(rows):
        target_index = int(y[idx].item())
        pred_index = int(pred[idx].item())
        if target_index == pred_index:
            correct += 1
        predictions.append(
            {
                "schema": "codexlab.sp80_lora_prediction.v1",
                "split": split,
                "row_id": row.get("row_id"),
                "calibration_slice_index": row.get("calibration_slice_index"),
                "case_id": row.get("case_id"),
                "target_primitive": CONTROL_PRIMITIVES[target_index],
                "predicted_primitive": CONTROL_PRIMITIVES[pred_index],
                "correct": target_index == pred_index,
                "confidence": float(probs[idx, pred_index].item()),
                "probabilities": {
                    primitive: float(probs[idx, primitive_idx].item())
                    for primitive_idx, primitive in enumerate(CONTROL_PRIMITIVES)
                },
            }
        )
    return correct / max(1, len(rows)), predictions


def build_confusion(predictions: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        target: {pred: 0 for pred in CONTROL_PRIMITIVES}
        for target in CONTROL_PRIMITIVES
    }
    for row in predictions:
        matrix[row["target_primitive"]][row["predicted_primitive"]] += 1
    return matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("codexlab/datasets/sp80_control_sft_v0"))
    parser.add_argument("--out-dir", type=Path, default=Path("codexlab/runs/sp80_lora_dry_run_v0"))
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-vocab", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--min-heldout-accuracy", type=float, default=1.0)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset_dir = resolve_repo_path(args.dataset_dir)
    out_dir = resolve_repo_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    validation = validate_dataset(dataset_dir)
    leakage = check_leakage(dataset_dir, allow_input_overlap=False)
    if validation["status"] != "passed":
        raise SystemExit(f"dataset validation failed: {validation['errors']}")
    if leakage["status"] != "passed":
        raise SystemExit(f"dataset leakage check failed: {leakage['failures']}")

    manifest_path = dataset_dir / "manifest.json"
    train_path = dataset_dir / "train.jsonl"
    heldout_path = dataset_dir / "heldout.jsonl"
    manifest = load_json(manifest_path)
    train_rows = load_jsonl(train_path)
    heldout_rows = load_jsonl(heldout_path)
    if manifest.get("status") != "exported":
        raise SystemExit("dataset manifest status is not exported")

    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    vocab = build_vocab(train_rows, max_vocab=max(1, int(args.max_vocab)))
    x_train = encode_rows(train_rows, vocab)
    x_heldout = encode_rows(heldout_rows, vocab)
    label_index = {primitive: idx for idx, primitive in enumerate(CONTROL_PRIMITIVES)}
    y_train = torch.tensor([label_index[row_target(row)] for row in train_rows], dtype=torch.long)
    y_heldout = torch.tensor([label_index[row_target(row)] for row in heldout_rows], dtype=torch.long)

    model = FrozenLinearLoRA(
        input_dim=x_train.shape[1],
        output_dim=len(CONTROL_PRIMITIVES),
        rank=max(1, int(args.rank)),
        alpha=float(args.alpha),
        seed=int(args.seed),
    )
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_param_count = model.base_weight.numel() + model.base_bias.numel()

    condition = {
        "schema": "codexlab.sp80_lora_dry_run_condition.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_label": "new_experiment",
        "evidence_status": "controlled",
        "qwen_weights_used": False,
        "qwen_lora_claim": False,
        "contra": contra_spec(
            backend="offline_lora_adapter",
            game="sp80_liquid_pour",
            allowed_primitives=CONTROL_PRIMITIVES,
        ),
        "model_family": "offline_frozen_linear_lora_classifier",
        "feature_space": "coordinate_free_intent_state_goal_condition_features",
        "cwd": str(ROOT),
        "command": " ".join(sys.argv),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": "cpu",
        "gpu_count": 0,
        "world_size": 1,
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "rank": int(args.rank),
        "alpha": float(args.alpha),
        "max_vocab": int(args.max_vocab),
        "vocab_size": len(vocab),
        "numeric_feature_keys": NUMERIC_KEYS,
        "allowed_primitives": CONTROL_PRIMITIVES,
        "trainable_param_count": int(trainable_param_count),
        "frozen_param_count": int(frozen_param_count),
        "source_script": relpath(SCRIPT_PATH),
        "source_script_sha256": sha256_file(SCRIPT_PATH),
        "dataset_manifest": relpath(manifest_path),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "train_jsonl": relpath(train_path),
        "train_jsonl_sha256": sha256_file(train_path),
        "heldout_jsonl": relpath(heldout_path),
        "heldout_jsonl_sha256": sha256_file(heldout_path),
        "dataset_validation": validation,
        "dataset_leakage_check": leakage,
        "controller_boundary": "adapter predicts one primitive only from intent/state/goal-condition relation fields; SP80 motor/controller remains unchanged",
    }
    condition_path = out_dir / "condition.json"
    write_json(condition_path, condition)

    optimizer = torch.optim.AdamW(
        [model.lora_a, model.lora_b],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    loss_fn = nn.CrossEntropyLoss()
    log_rows: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch == int(args.epochs) or epoch % max(1, int(args.log_every)) == 0:
            train_acc, _ = accuracy_and_predictions(model=model, x=x_train, rows=train_rows, y=y_train, split="train")
            heldout_acc, _ = accuracy_and_predictions(
                model=model,
                x=x_heldout,
                rows=heldout_rows,
                y=y_heldout,
                split="heldout",
            )
            log_rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(loss.detach().cpu().item()),
                    "train_accuracy": train_acc,
                    "heldout_accuracy": heldout_acc,
                }
            )

    training_log_path = out_dir / "training_log.jsonl"
    write_jsonl(training_log_path, log_rows)

    train_accuracy, train_predictions = accuracy_and_predictions(
        model=model,
        x=x_train,
        rows=train_rows,
        y=y_train,
        split="train",
    )
    heldout_accuracy, heldout_predictions = accuracy_and_predictions(
        model=model,
        x=x_heldout,
        rows=heldout_rows,
        y=y_heldout,
        split="heldout",
    )
    predictions = train_predictions + heldout_predictions
    predictions_path = out_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)

    adapter_payload = {
        "schema": "codexlab.sp80_lora_adapter.v1",
        "condition": condition,
        "vocab": vocab,
        "numeric_feature_keys": NUMERIC_KEYS,
        "allowed_primitives": CONTROL_PRIMITIVES,
        "state_dict": model.adapter_state(),
    }
    adapter_path = out_dir / "adapter.pt"
    torch.save(adapter_payload, adapter_path)

    result = {
        "schema": "codexlab.sp80_lora_dry_run_result.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "passed" if heldout_accuracy >= float(args.min_heldout_accuracy) else "failed",
        "run_label": "new_experiment",
        "qwen_weights_used": False,
        "qwen_lora_claim": False,
        "contra": condition["contra"],
        "condition_json": relpath(condition_path),
        "condition_json_sha256": sha256_file(condition_path),
        "adapter_pt": relpath(adapter_path),
        "adapter_pt_sha256": sha256_file(adapter_path),
        "training_log_jsonl": relpath(training_log_path),
        "training_log_jsonl_sha256": sha256_file(training_log_path),
        "predictions_jsonl": relpath(predictions_path),
        "predictions_jsonl_sha256": sha256_file(predictions_path),
        "train_count": len(train_rows),
        "heldout_count": len(heldout_rows),
        "train_accuracy": train_accuracy,
        "heldout_accuracy": heldout_accuracy,
        "min_heldout_accuracy": float(args.min_heldout_accuracy),
        "train_confusion": build_confusion(train_predictions),
        "heldout_confusion": build_confusion(heldout_predictions),
        "final_log_row": log_rows[-1] if log_rows else None,
        "boundary": "LoRA dry-run validates adapter mechanics over primitive rows only; it does not solve SP80 and does not train Qwen.",
    }
    result_path = out_dir / "result.json"
    write_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.strict and result["status"] != "passed":
        raise SystemExit("strict LoRA dry-run failed heldout threshold")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
