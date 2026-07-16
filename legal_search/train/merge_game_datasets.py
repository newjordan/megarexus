#!/usr/bin/env python3
"""Build a balanced multi-game dataset from explicit per-game source artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_search.train.model import game_bucket


def _all_rows(payload: dict) -> dict[str, torch.Tensor]:
    keys = set(payload["train"]).intersection(payload["val"])
    return {key: torch.cat([payload["train"][key], payload["val"][key]], dim=0) for key in keys}


def _slice(data: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[idx] for key, value in data.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, help="GAME_ID=DATA.pt")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=71008)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    selected: dict[str, tuple[dict, dict[str, torch.Tensor], torch.Tensor]] = {}
    counts: dict[str, int] = {}
    for spec in args.source:
        game_id, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        rows = _all_rows(payload)
        bucket = game_bucket(game_id)
        idx = torch.nonzero(rows["game_idx"] == bucket, as_tuple=False).flatten()
        if not len(idx):
            raise SystemExit(f"{path}: no rows for {game_id} bucket={bucket}")
        selected[game_id] = (payload, rows, idx)
        counts[game_id] = int(len(idx))
    target = min(counts.values())
    train_parts: dict[str, list[torch.Tensor]] = {}
    val_parts: dict[str, list[torch.Tensor]] = {}
    receipt: dict[str, dict] = {}
    for game_id, (payload, rows, idx) in sorted(selected.items()):
        priority_mask = (rows["levels_up"][idx] > 0) | (rows["success_window"][idx] > 0)
        priority = idx[priority_mask]
        ordinary = idx[~priority_mask]
        ordinary = ordinary[torch.from_numpy(rng.permutation(len(ordinary))).long()]
        keep = torch.cat([priority[:target], ordinary[: max(0, target - len(priority))]])
        # Keep every success row in train. Validation is ordinary held-out rows.
        n_val = max(16, target // 10)
        is_priority = (rows["levels_up"][keep] > 0) | (rows["success_window"][keep] > 0)
        priority_keep = keep[is_priority]
        ordinary_keep = keep[~is_priority]
        val_idx = ordinary_keep[-n_val:]
        train_idx = torch.cat([priority_keep, ordinary_keep[:-n_val]])
        train_data = _slice(rows, train_idx)
        val_data = _slice(rows, val_idx)
        for key, value in train_data.items():
            train_parts.setdefault(key, []).append(value)
        for key, value in val_data.items():
            val_parts.setdefault(key, []).append(value)
        receipt[game_id] = {
            "source": str(args.source[[s.split("=", 1)[0] for s in args.source].index(game_id)].split("=", 1)[1]),
            "available_rows": counts[game_id],
            "selected_rows": int(len(keep)),
            "train_rows": int(len(train_idx)),
            "val_rows": int(len(val_idx)),
            "train_level_ups": int(rows["levels_up"][train_idx].sum().item()),
            "train_success_window": int(rows["success_window"][train_idx].sum().item()),
        }
    train = {key: torch.cat(parts, dim=0) for key, parts in train_parts.items()}
    val = {key: torch.cat(parts, dim=0) for key, parts in val_parts.items()}
    payload0 = next(iter(selected.values()))[0]
    out = {
        "schema": payload0["schema"],
        "feature_schema": payload0["feature_schema"],
        "feature_dim": payload0["feature_dim"],
        "feature_layout": payload0["feature_layout"],
        "n_train": int(train["frame"].size(0)),
        "n_val": int(val["frame"].size(0)),
        "train": train,
        "val": val,
        "meta": {
            "builder": "balanced_explicit_per_game_sources_v1",
            "seed": args.seed,
            "target_rows_per_game": target,
            "receipt": receipt,
            "legal": True,
            "banned_lineage": ["sp80_spill_model", "dropletsim", "human_mined_routes", "fixture_win_banks_as_labels"],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    args.out.with_suffix(".manifest.json").write_text(json.dumps(out["meta"], indent=2) + "\n")
    print(json.dumps(out["meta"], indent=2))
    print(f"wrote {args.out} train={out['n_train']} val={out['n_val']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
