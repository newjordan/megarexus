"""One MegaMind cycle on a synthetic board. No checkpoints are loaded."""

from __future__ import annotations

import json

import torch

from synthesis_lab.orchestration.rdl.brain.brain_types import BrainState
from synthesis_lab.orchestration.rdl.brain.megamind import MegaMind
from synthesis_lab.orchestration.rdl.brain.smoke_megamind import phase_line, training_phases


def test_judge_plan_batches_a_scalar_action():
    mind = MegaMind(emb=16, n_actions=8, device="cpu")
    frames = torch.randint(0, 16, (3, 16, 16))
    judged = mind.judge_plan(
        frames,
        action=11,
        state=BrainState(dream_summary={"status": "ok"}),
    )
    assert tuple(judged.action_logits.shape) == (3, 8)
    assert judged.dream_summary["_action_for_dream"] == 3
    payload = judged.to_dict()
    json.dumps(payload)
    assert "_action_for_dream" not in payload["dream_summary"]
    assert "_dream_next" not in payload["dream_summary"]


def test_empty_available_actions_mask_every_logit():
    mind = MegaMind(emb=16, n_actions=8, device="cpu")
    state = mind.observe(torch.randint(0, 16, (24, 24)), available_actions=[])
    assert state.propose_logits is not None
    assert not torch.isfinite(state.propose_logits).any()


def test_train_cycle_logs_are_printable_without_a_loss_key():
    mind = MegaMind(emb=16, n_actions=8, device="cpu")
    frame = torch.randint(0, 16, (1, 24, 24))
    logs = mind.train_transition(frame, 2, (frame + 1) % 16, steps_per_phase=1)
    lines = [phase_line(row) for row in logs]
    assert any("loss=n/a" in line for line in lines)
    assert training_phases(logs) == ["external", "internal", "central"]
    assert logs[-1]["phase"] == "coverage"
    assert any(row["phase"] == "ada_online" for row in logs)
