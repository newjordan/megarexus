#!/usr/bin/env python3
"""End-to-end MegaMind smoke: observe + train_transition, truth_flags all false.

Run (prefer CPU — vLLM often OOMs the GPU):
  CUDA_VISIBLE_DEVICES= /home/frosty40/miniconda3/bin/python3 -B \\
    synthesis_lab/orchestration/rdl/brain/smoke_megamind.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synthesis_lab.orchestration.rdl.brain.megamind import MegaMind
from synthesis_lab.orchestration.rdl.brain.types import TruthFlags


def _flags_all_false(tf) -> bool:
    if isinstance(tf, TruthFlags):
        return not tf.any_true()
    if isinstance(tf, dict):
        return not any(bool(v) for v in tf.values())
    return False


def main() -> int:
    device = "cpu"
    torch.manual_seed(0)
    print(f"device={device}")

    mind = MegaMind(emb=128, n_actions=8, device=device)

    frame = torch.randint(0, 16, (64, 64))
    state = mind.observe(frame, available_actions=[1, 2, 3, 4, 5, 6])
    print(
        f"observe: trusted={state.trusted} status={state.status} "
        f"h_I={None if state.h_I is None else tuple(state.h_I.shape)} "
        f"propose={None if state.propose_logits is None else tuple(state.propose_logits.shape)}"
    )
    print(f"  gate_failures={state.gate_failures}")
    print(
        "  dream_summary keys="
        f"{sorted(k for k in state.dream_summary if not str(k).startswith('_'))}"
    )
    print(f"  truth_flags={state.truth_flags_dict()}")

    if not state.all_truth_flags_false() or not _flags_all_false(state.truth_flags):
        print("SMOKE FAIL — truth_flags not all false after observe")
        return 1

    action = 3
    judged = mind.judge_plan(frame, action=action, plan={"hyp_action": action})
    print(
        f"judge_plan: judge_I={judged.judge_I:.4f} disagreement={judged.disagreement:.4f} "
        f"logits={tuple(judged.action_logits.shape) if judged.action_logits is not None else None}"
    )
    if not judged.all_truth_flags_false():
        print("SMOKE FAIL — truth_flags not all false after judge_plan")
        return 1

    logits = mind.act_logits(judged)
    print(f"act_logits: shape={tuple(logits.shape)} argmax={int(logits[0].argmax())}")

    s = frame.unsqueeze(0)
    a = torch.tensor([action])
    s_next = (s + 1).remainder(16)
    logs = mind.train_transition(s, a, s_next, steps_per_phase=1)
    phases = [row["phase"] for row in logs]
    print(f"train_transition phases={phases}")
    for row in logs:
        print(f"  {row['phase']:8s} loss={row['loss']:.4f}")

    expected = ["external", "internal", "central"]
    if phases != expected:
        print(f"SMOKE FAIL — expected phases {expected}, got {phases}")
        return 1

    # banned import guard: only real import/from statements (not ban comments)
    import re

    brain_dir = Path(__file__).resolve().parent
    banned = ("dropletsim", "spill_model", "native_scene")
    import_re = re.compile(r"^\s*(from\s+\S+\s+import|import\s+)")
    for py in list(brain_dir.glob("*.py")) + list((brain_dir / "operators").glob("*.py")):
        text = py.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if not import_re.match(line):
                continue
            for b in banned:
                if b in stripped:
                    print(f"SMOKE FAIL — banned import in {py.name}: {stripped}")
                    return 1

    print("truth_flags all false: OK")
    print("SMOKE PASS — megamind observe + train_transition cycle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
