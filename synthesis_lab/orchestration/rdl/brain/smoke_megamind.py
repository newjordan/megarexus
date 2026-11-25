#!/usr/bin/env python3
"""End-to-end MegaMind smoke: observe, judge, act, and one train cycle.

Run (CPU is enough; this does not download weights):

  python -B synthesis_lab/orchestration/rdl/brain/smoke_megamind.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synthesis_lab.orchestration.rdl.brain.brain_types import TruthFlags
from synthesis_lab.orchestration.rdl.brain.megamind import MegaMind

TRAIN_PHASES = ("external", "internal", "central")


def _say(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def _flags_all_false(tf: Any) -> bool:
    if isinstance(tf, TruthFlags):
        return not tf.any_true()
    if isinstance(tf, dict):
        return not any(bool(v) for v in tf.values())
    return False


def phase_line(row: dict[str, Any]) -> str:
    """One status line. Coverage and idle Ada rows have no loss."""
    phase = str(row.get("phase", "?"))
    loss = row.get("loss")
    if isinstance(loss, (int, float)) and not isinstance(loss, bool):
        return f"  {phase:<12} loss={float(loss):.4f}"
    return f"  {phase:<12} loss=n/a"


def training_phases(logs: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("phase")) for row in logs if row.get("phase") in TRAIN_PHASES]


def _operator_counts(dream_summary: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, value in dream_summary.items():
        if str(key).startswith("_") or not isinstance(value, dict):
            continue
        if "status" not in value:
            continue
        status = str(value.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return counts


def main() -> int:
    device = "cpu"
    torch.manual_seed(0)
    _say(f"MegaMind smoke  device={device}")

    mind = MegaMind(emb=128, n_actions=8, device=device)
    frame = torch.randint(0, 16, (64, 64))
    state = mind.observe(frame, available_actions=[1, 2, 3, 4, 5, 6])
    counts = _operator_counts(state.dream_summary or {})
    count_text = ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
    h_shape = None if state.h_I is None else tuple(state.h_I.shape)
    p_shape = None if state.propose_logits is None else tuple(state.propose_logits.shape)
    _say(
        f"observe     trusted={state.trusted} status={state.status} "
        f"h_I={h_shape} propose={p_shape}"
    )
    _say(f"operators   {count_text or 'none'}")
    if state.gate_failures:
        _say(f"gates       {state.gate_failures}")
    _say(f"truth       {state.truth_flags_dict()}")

    if not state.all_truth_flags_false() or not _flags_all_false(state.truth_flags):
        _say("FAIL  truth flags were not all false after observe")
        return 1
    if state.propose_logits is not None:
        illegal = state.propose_logits[0, 0]
        if torch.isfinite(illegal):
            _say("FAIL  action 0 was offered even though it was not available")
            return 1

    action = 3
    judged = mind.judge_plan(frame, action=action, plan={"hyp_action": action})
    logit_shape = None if judged.action_logits is None else tuple(judged.action_logits.shape)
    _say(
        f"judge       judge_I={judged.judge_I:.4f} "
        f"disagreement={judged.disagreement:.4f} logits={logit_shape}"
    )
    if not judged.all_truth_flags_false():
        _say("FAIL  truth flags were not all false after judge_plan")
        return 1

    logits = mind.act_logits(judged)
    _say(f"act         shape={tuple(logits.shape)} argmax={int(logits[0].argmax())}")

    logs = mind.train_transition(
        frame.unsqueeze(0),
        torch.tensor([action]),
        (frame.unsqueeze(0) + 1).remainder(16),
        steps_per_phase=1,
    )
    _say("train")
    for row in logs:
        _say(phase_line(row))

    phases = training_phases(logs)
    if phases != list(TRAIN_PHASES):
        _say(f"FAIL  training phases {phases}, expected {list(TRAIN_PHASES)}")
        return 1
    names = [str(row.get("phase")) for row in logs]
    if "ada_online" not in names or names[-1] != "coverage":
        _say(f"FAIL  expected ada_online and a trailing coverage row, got {names}")
        return 1

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
            for name in banned:
                if name in stripped:
                    _say(f"FAIL  banned import in {py.name}: {stripped}")
                    return 1

    _say("PASS  observe, judge, act, and one train cycle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
