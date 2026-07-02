"""SCOUT_EXPERIMENT: the same-valset no-loss gate, as a command.

Evaluates a checkpoint on an EXPLICIT list of val shards with train.py's
own evaluate() — the exact metrics the training loop prints. Purpose
(ADDING_CNN_ELEMENTS.md "Gates"): val shards change between runs, so
headline numbers aren't comparable; the gate is to evaluate the NEW
checkpoint (both last.pt and best.pt) on the PREVIOUS release's val
shards and compare head-by-head against the previous release's numbers.
The release checkpoint is whichever passes this + the behavioral probes
(level_probe 20/20 vs 5/20, arrange_probe 10/10 vs 1/10) — never a
training-time heuristic.

Recover the previous release's val shard list from its checkpoint:
    ckpt["config"]["val_rollout_shards"]

Usage (compare v9 last.pt against v8 on v8's valset):
    python synthesis_lab/orchestration/gate_eval.py \
        --ckpt synthesis_lab/experiments/megarexus_cnn/runs/outcome_v9/last.pt \
        --val-shards-from synthesis_lab/experiments/megarexus_cnn/runs/outcome_v8/last.pt
then the same with --ckpt .../outcome_v8/last.pt for the baseline side.
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
from train import evaluate  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True,
                   help="checkpoint .pt to evaluate (last.pt or best.pt)")
    p.add_argument("--val-shards-from", type=Path, default=None,
                   help="checkpoint whose config supplies the val shard "
                        "list (the PREVIOUS release, for the no-loss gate)")
    p.add_argument("--val-rollout-shards", type=Path, nargs="*", default=None,
                   help="explicit val shards (overrides --val-shards-from)")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    stack = ckpt["config"].get("stack_frames", 2)
    model = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)) \
        .to(device).eval()
    model.load_state_dict(ckpt["model"])

    shards = args.val_rollout_shards
    if not shards:
        if args.val_shards_from is None:
            p.error("need --val-rollout-shards or --val-shards-from")
        ref = torch.load(args.val_shards_from, map_location="cpu",
                         weights_only=False)
        shards = [Path(s) for s in ref["config"]["val_rollout_shards"]]
    print(f"[gate] ckpt={args.ckpt}")
    print(f"[gate] val shards: {[str(s) for s in shards]}")

    # mirror train.py's val dataset exactly (corpus val split included)
    val_ds = MegarexusTransitions("val", val_rollout_shards=shards,
                                  n_input_frames=stack)
    loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)
    metrics = evaluate(model, loader, device)
    print(json.dumps({k: round(v, 4) for k, v in sorted(metrics.items())},
                     indent=1))


if __name__ == "__main__":
    main()
