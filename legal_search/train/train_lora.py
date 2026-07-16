#!/usr/bin/env python3
"""Train LegalBlendLoRA on legal_search experience. GPU if available."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_search.train.features import FEATURE_DIM  # noqa: E402
from legal_search.train.model import LegalBlendConfig, LegalBlendLoRA  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batchify(store: dict, device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in store.items() if torch.is_tensor(v)}


@torch.no_grad()
def evaluate(model: LegalBlendLoRA, data: dict, device: torch.device, batch_size: int = 64) -> dict:
    model.eval()
    n = data["frame"].size(0)
    total_loss = 0.0
    total_change = 0.0
    correct_change = 0.0
    n_batches = 0
    change_pos = 0.0
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        frame = data["frame"][sl].to(device)
        features = data["features"][sl].to(device)
        action_id = data["action_id"][sl].to(device)
        x = data["x"][sl].to(device)
        y = data["y"][sl].to(device)
        did_change = data["did_change"][sl].to(device)
        game_idx = data.get("game_idx")
        game_idx = game_idx[sl].to(device) if game_idx is not None else None
        out = model(frame, features, game_idx=game_idx)
        losses = model.loss(out, action_id=action_id, x=x, y=y, did_change=did_change)
        total_loss += float(losses["loss"])
        total_change += float(losses["change_loss"])
        # accuracy: sigmoid(logit) > 0.5 vs did_change
        from legal_search.train.model import goose_index_tensor

        idx = goose_index_tensor(action_id, x, y)
        logit = out["change_logits"].gather(1, idx.unsqueeze(1)).squeeze(1)
        pred = (logit.sigmoid() >= 0.5).float()
        correct_change += float((pred == did_change).float().sum())
        change_pos += float(did_change.sum())
        n_batches += 1
    return {
        "loss": total_loss / max(1, n_batches),
        "change_loss": total_change / max(1, n_batches),
        "change_acc": correct_change / max(1, n),
        "change_base_rate": change_pos / max(1, n),
        "n": n,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "legal_search" / "runs" / "blend_lora_gated_v1",
    )
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--lora-only-after", type=int, default=2, help="epoch index to freeze base (0=immediate)")
    p.add_argument("--collect-if-missing", action="store_true", default=True)
    p.add_argument("--force-collect", action="store_true", help="Re-collect even if data.pt exists")
    p.add_argument("--collect-episodes", type=int, default=20, help="synthetic episodes")
    p.add_argument("--arcade-episodes", type=int, default=3, help="offline Arcade episodes")
    p.add_argument("--max-actions", type=int, default=48, help="max steps per collect episode")
    p.add_argument(
        "--regime",
        type=str,
        default="",
        choices=["", "medium", "huge", "level"],
        help="Preset scale: medium, huge, or level (path B: levels_up-shaped)",
    )
    p.add_argument(
        "--level-shaped",
        action="store_true",
        help="Path B: success-window policy weights + thrash change downweight",
    )
    p.add_argument("--policy-coef", type=float, default=0.35, help="policy loss coefficient")
    p.add_argument(
        "--levels-up-boost",
        type=float,
        default=2.0,
        help="extra policy weight multiplier on levels_up rows (added to sample_weight)",
    )
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--game-conditioned", action="store_true", help="Use shared trunk plus game-conditioned policy residuals")
    p.add_argument("--interaction-conditioned", action="store_true", help="Enable generic gas/brake exploration from recent env transitions")
    p.add_argument(
        "--allow-ungated-scout",
        action="store_true",
        help="DANGER: bypass mining gate for scout only; forces runs/_ungated_scout/",
    )
    args = p.parse_args()

    # Regime presets override scale knobs (explicit CLI wins if set after — apply first)
    if args.regime == "medium":
        args.epochs = max(args.epochs, 40)
        args.batch_size = max(args.batch_size, 96)
        args.collect_episodes = max(args.collect_episodes, 200)
        args.arcade_episodes = max(args.arcade_episodes, 16)
        args.max_actions = max(args.max_actions, 64)
        args.lr = min(args.lr, 1.5e-3)
        args.lora_only_after = max(args.lora_only_after, 4)
    elif args.regime == "huge":
        args.epochs = max(args.epochs, 100)
        args.batch_size = max(args.batch_size, 128)
        args.collect_episodes = max(args.collect_episodes, 600)
        args.arcade_episodes = max(args.arcade_episodes, 40)
        args.max_actions = max(args.max_actions, 80)
        args.lr = min(args.lr, 1e-3)
        args.lora_only_after = max(args.lora_only_after, 8)
    elif args.regime == "level":
        # Path B: many completed synth levels + thrash-aware weights
        args.level_shaped = True
        args.epochs = max(args.epochs, 60)
        args.batch_size = max(args.batch_size, 96)
        args.collect_episodes = max(args.collect_episodes, 400)
        args.arcade_episodes = max(args.arcade_episodes, 24)
        args.max_actions = max(args.max_actions, 64)
        args.lr = min(args.lr, 1e-3)
        args.lora_only_after = max(args.lora_only_after, 6)
        args.policy_coef = max(args.policy_coef, 0.85)
        args.levels_up_boost = max(args.levels_up_boost, 8.0)

    from legal_search.train.mining_gate import MiningIncompleteError, assert_mining_complete

    try:
        gate = assert_mining_complete(allow_ungated_scout=bool(args.allow_ungated_scout))
    except MiningIncompleteError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    set_seed(args.seed)
    out_dir = args.out_dir
    if gate.get("scout"):
        # Force scout outputs away from gated run dirs
        out_dir = ROOT / "legal_search" / "runs" / "_ungated_scout" / out_dir.name
        print(f"[train] {gate['message']}")
        print(f"[train] scout out_dir forced → {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = args.data or (out_dir / "data.pt")

    need_collect = args.force_collect or (not data_path.exists() and args.collect_if_missing)
    if need_collect:
        if args.force_collect and data_path.exists():
            print(f"[train] force-collect: removing existing {data_path}")
            data_path.unlink()
        print(
            f"[train] collecting data → {data_path} "
            f"(synth_ep={args.collect_episodes} arcade_ep={args.arcade_episodes} "
            f"max_actions={args.max_actions} regime={args.regime or 'custom'})"
        )
        from legal_search.train.collect import main as collect_main

        collect_argv = [
            "collect.py",
            "--out",
            str(data_path),
            "--episodes",
            str(args.collect_episodes),
            "--arcade-episodes",
            str(args.arcade_episodes),
            "--max-actions",
            str(args.max_actions),
            "--seed",
            str(args.seed),
        ]
        if args.level_shaped:
            collect_argv.append("--level-shaped")
        if args.allow_ungated_scout:
            collect_argv.append("--allow-ungated-scout")
        sys.argv = collect_argv
        collect_main()

    if not data_path.exists():
        raise SystemExit(f"missing data {data_path}")

    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    assert int(payload["feature_dim"]) == FEATURE_DIM
    train = payload["train"]
    val = payload["val"]

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    print(
        f"[train] device={device} train_n={train['frame'].size(0)} val_n={val['frame'].size(0)} "
        f"epochs={args.epochs} batch={args.batch_size} lr={args.lr} regime={args.regime or 'custom'}"
    )

    cfg = LegalBlendConfig(
        game_conditioned=bool(args.game_conditioned),
        interaction_conditioned=bool(args.interaction_conditioned),
    )
    model = LegalBlendLoRA(cfg).to(device)
    print(f"[train] lora_layers={model.n_lora_layers} feature_dim={FEATURE_DIM}")

    n_tr = int(train["frame"].size(0))
    sample_w = train.get("sample_weight")
    if sample_w is None or sample_w.numel() != n_tr:
        sample_w = torch.ones(n_tr, dtype=torch.float32)
        train["sample_weight"] = sample_w
    change_w = train.get("change_weight")
    if change_w is None or change_w.numel() != n_tr:
        change_w = torch.ones(n_tr, dtype=torch.float32)
        train["change_weight"] = change_w
    success_w = train.get("success_window")
    if success_w is None or success_w.numel() != n_tr:
        success_w = torch.zeros(n_tr, dtype=torch.float32)
        train["success_window"] = success_w
    game_idx = train.get("game_idx")
    if game_idx is None or game_idx.numel() != n_tr:
        game_idx = torch.zeros(n_tr, dtype=torch.long)
        train["game_idx"] = game_idx

    n_up = int(train["levels_up"].sum().item()) if "levels_up" in train else 0
    n_sw = int(success_w.sum().item())
    print(
        f"[train] level_shaped={bool(args.level_shaped)} levels_up_rows={n_up} "
        f"success_window_rows={n_sw} policy_coef={args.policy_coef} "
        f"levels_up_boost={args.levels_up_boost}"
    )

    ds = TensorDataset(
        train["frame"],
        train["features"],
        train["action_id"],
        train["x"],
        train["y"],
        train["did_change"],
        train["levels_up"],
        sample_w,
        change_w,
        success_w,
        game_idx,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    def build_opt():
        groups = []
        for g in model.trainable_param_groups():
            groups.append({"params": g["params"], "lr": args.lr * g["lr_scale"]})
        if not groups:
            groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr}]
        return torch.optim.AdamW(groups, weight_decay=1e-4)

    opt = build_opt()
    best_val = float("inf")
    best_path = out_dir / "best.pt"
    log_path = out_dir / "train_log.jsonl"
    metrics_path = out_dir / "metrics.json"
    config_path = out_dir / "config.json"

    config = {
        "schema": "legal_search.blend_lora_train.v1",
        "feature_dim": FEATURE_DIM,
        "model": cfg.to_dict(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "device": str(device),
        "data": str(data_path),
        "n_train": int(train["frame"].size(0)),
        "n_val": int(val["frame"].size(0)),
        "legal": True,
        "banned_lineage": payload.get("meta", {}).get("banned_lineage"),
        "mining_docs": [
            "legal_search/MINING_MANIFEST.md",
            "legal_search/docs/MINING_HANDOFF.md",
            "legal_search/docs/TRAIN_PLAN.md",
            "legal_search/docs/FULL_MINE_TO_LORA.md",
            "legal_search/docs/ADAJEPA_PORT.md",
            "PRIMARY_LANE.md",
        ],
        "full_mine": True,
        "feature_schema": "legal_search.features.v2_fullmine",
        "gated": bool(gate.get("gated")),
        "scout": bool(gate.get("scout")),
        "mining_gate": {
            "path": gate.get("gate_path"),
            "doc": "legal_search/MINING_THEN_TRAIN.md",
            "complete": bool(gate.get("complete")),
        },
        "level_shaped": bool(args.level_shaped),
        "game_conditioned": bool(args.game_conditioned),
        "interaction_conditioned": bool(args.interaction_conditioned),
        "policy_coef": float(args.policy_coef),
        "levels_up_boost": float(args.levels_up_boost),
        "n_levels_up_train": n_up,
        "n_success_window_train": n_sw,
        "adajepa": {
            "enabled_at_deploy": True,
            "module": "legal_search.train.ada_adapt.AdaBlendPolicy",
            "ideas": [
                "plan_act_adapt_replan",
                "online_buffer",
                "one_grad_step_lora_only",
                "ssl_change_loss_no_reward",
            ],
            "source": "https://agenticlearning.ai/adajepa/",
        },
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    # baseline val before train
    base_val = evaluate(model, val, device, args.batch_size)
    print(f"[train] val@0 loss={base_val['loss']:.4f} change_acc={base_val['change_acc']:.3f} base_rate={base_val['change_base_rate']:.3f}")

    t0 = time.time()
    global_step = 0
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write(json.dumps({"event": "start", "val": base_val}) + "\n")
        for epoch in range(args.epochs):
            if epoch == args.lora_only_after:
                model.set_lora_only(True)
                opt = build_opt()
                print(f"[train] epoch {epoch}: switched to LoRA-only params")

            model.train()
            running = 0.0
            n_b = 0
            for batch in loader:
                (
                    frame,
                    features,
                    action_id,
                    x,
                    y,
                    did_change,
                    levels_up,
                    sample_weight,
                    change_weight,
                    success_window,
                    game_idx,
                ) = [t.to(device) for t in batch]
                # Path B: sample_weight (success window) + levels_up boost; thrash already low
                pol_w = sample_weight * (1.0 + float(args.levels_up_boost) * levels_up)
                if args.level_shaped:
                    pol_w = pol_w * (1.0 + 2.0 * success_window)
                out = model(frame, features, game_idx=game_idx)
                losses = model.loss(
                    out,
                    action_id=action_id,
                    x=x,
                    y=y,
                    did_change=did_change,
                    policy_weight=pol_w,
                    change_weight=change_weight,
                    policy_coef=float(args.policy_coef),
                )
                opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                running += float(losses["loss"].detach())
                n_b += 1
                global_step += 1

            train_loss = running / max(1, n_b)
            val_m = evaluate(model, val, device, args.batch_size)
            row = {
                "epoch": epoch,
                "step": global_step,
                "train_loss": train_loss,
                "val": val_m,
                "seconds": time.time() - t0,
            }
            logf.write(json.dumps(row) + "\n")
            logf.flush()
            print(
                f"[train] epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} "
                f"val_loss={val_m['loss']:.4f} change_acc={val_m['change_acc']:.3f}"
            )
            if val_m["loss"] < best_val:
                best_val = val_m["loss"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg.to_dict(),
                        "val": val_m,
                        "epoch": epoch,
                        "feature_dim": FEATURE_DIM,
                        "schema": "legal_search.blend_lora_ckpt.v1",
                        "legal": True,
                    },
                    best_path,
                )
                print(f"[train] saved best → {best_path}")

    final_val = evaluate(model, val, device, args.batch_size)
    metrics = {
        "schema": "legal_search.blend_lora_metrics.v1",
        "base_val": base_val,
        "final_val": final_val,
        "best_val_loss": best_val,
        "improved": final_val["loss"] < base_val["loss"] or best_val < base_val["loss"],
        "change_acc_vs_base_rate": final_val["change_acc"] - final_val["change_base_rate"],
        "seconds": time.time() - t0,
        "device": str(device),
        "best_path": str(best_path),
        "legal": True,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0 if metrics["improved"] or final_val["change_acc"] >= final_val["change_base_rate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
