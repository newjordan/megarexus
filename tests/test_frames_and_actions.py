"""Frame layout and action alignment."""

from __future__ import annotations

import torch

from synthesis_lab.orchestration.rdl.brain.hemisphere_e import ExternalHemisphereFull
from synthesis_lab.orchestration.rdl.brain.hemisphere_i import (
    _as_frame_tensor,
    _result_to_dict,
    align_actions,
)
from synthesis_lab.orchestration.rdl.brain.transcript_macros import macro_action_bias


def test_color_index_batch_with_width_3_stays_a_batch():
    frames = torch.arange(4 * 64 * 3).reshape(4, 64, 3) % 10
    out = _as_frame_tensor(frames, torch.device("cpu"))
    assert out.shape == (4, 64, 3)
    assert torch.equal(out, frames.long())


def test_channel_last_board_becomes_one_frame():
    image = torch.randint(0, 10, (64, 64, 3))
    out = _as_frame_tensor(image, torch.device("cpu"))
    assert out.shape == (1, 64, 64)
    assert torch.equal(out[0], image[..., 0].long())


def test_two_dimensional_board_gains_a_batch_axis():
    board = torch.randint(0, 16, (32, 32))
    out = _as_frame_tensor(board, torch.device("cpu"))
    assert out.shape == (1, 32, 32)


def test_scalar_action_is_repeated_across_the_batch():
    actions = align_actions(3, 4, 8, torch.device("cpu"))
    assert actions.tolist() == [3, 3, 3, 3]


def test_negative_action_wraps_into_range():
    actions = align_actions(-1, 1, 8, torch.device("cpu"))
    assert int(actions) == 7


def test_mismatched_action_batch_is_rejected():
    try:
        align_actions(torch.tensor([1, 2]), 3, 8, torch.device("cpu"))
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_external_run_accepts_one_action_for_a_batch():
    external = ExternalHemisphereFull(
        emb=8, n_actions=4, device="cpu", use_world_model=False
    )
    out = external.run(torch.zeros(3, 8, 8, dtype=torch.long), 2)
    assert out["h_E"].shape == (3, 8)


def test_learn_step_rejects_mismatched_next_frames():
    external = ExternalHemisphereFull(
        emb=8, n_actions=4, device="cpu", use_world_model=False
    )
    try:
        external.learn_step(torch.zeros(2, 8, 8), 1, torch.zeros(1, 8, 8))
    except ValueError as exc:
        assert "batch sizes differ" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_bool_scores_stay_booleans():
    packet = _result_to_dict(
        {"name": "x", "scores": {"ok": True, "n": 2}, "status": "ok"},
        "x",
    )
    assert packet["scores"]["ok"] is True
    assert packet["scores"]["n"] == 2.0


def test_empty_action_list_adds_no_macro_bias():
    bias = macro_action_bias([], n_actions=8)
    assert float(bias.sum()) == 0.0
    default = macro_action_bias(None, n_actions=8)
    assert float(default.sum()) > 0.0
