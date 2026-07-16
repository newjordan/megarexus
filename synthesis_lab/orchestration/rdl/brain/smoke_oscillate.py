#!/usr/bin/env python3
"""Synthetic smoke: dual-hemisphere oscillation trains without Arcade.

Run:
  /home/frosty40/miniconda3/bin/python3 -B \\
    synthesis_lab/orchestration/rdl/brain/smoke_oscillate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# repo root on path
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synthesis_lab.orchestration.rdl.brain.oscillate import OscillationConfig, OscillationTrainer


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    B, H, W = 16, 64, 64
    frames = torch.randint(0, 16, (B, H, W))
    # synthetic dynamics: next frame is a noisy shift of palette
    frames_next = (frames + 1).remainder(16)
    actions = torch.randint(1, 7, (B,))

    trainer = OscillationTrainer(cfg=OscillationConfig(device=device))
    print(f"device={device} emb={trainer.cfg.emb}")

    # warm baselines
    logs = trainer.train_cycle(frames, actions, frames_next, steps_per_phase=5)
    by_phase: dict[str, list[float]] = {}
    for row in logs:
        by_phase.setdefault(row["phase"], []).append(row["loss"])

    print("phase loss curves (5 steps each):")
    ok = True
    for phase, losses in by_phase.items():
        first, last = losses[0], losses[-1]
        improved = last <= first * 1.05  # allow noise; prefer non-explosion
        print(f"  {phase:8s}  first={first:.4f}  last={last:.4f}  stable={improved}")
        if last > first * 5.0 or last != last:  # explode or nan
            ok = False

    # one more full cycle to show E watches I disagreement
    more = trainer.step_batch(frames, actions, frames_next, phase=trainer.cfg.phases[1])
    if "mean_disagreement" in more:
        print(f"internal phase mean_disagreement={more['mean_disagreement']:.4f}")

    c = trainer.step_batch(frames, actions, frames_next, phase=trainer.cfg.phases[2])
    if "trust_E_mean" in c:
        print(f"central trust_E_mean={c['trust_E_mean']:.4f}")

    if ok:
        print("SMOKE PASS — oscillation loop stable on synthetic transitions")
        return 0
    print("SMOKE FAIL — loss exploded or NaN")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
