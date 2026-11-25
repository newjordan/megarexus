"""Central fusion, phase loss, and the private keep gate."""

from __future__ import annotations

import torch
import torch.nn as nn

from synthesis_lab.orchestration.rdl.brain.ada_online import CentralAdaOnline
from synthesis_lab.orchestration.rdl.brain.brain_types import BrainState
from synthesis_lab.orchestration.rdl.brain.central import (
    CentralConfig,
    CentralCortex,
    LoRALinear,
)
from synthesis_lab.orchestration.rdl.brain.hemisphere_i import EXTRA_FEAT_DIM
from synthesis_lab.orchestration.rdl.brain.hemispheres import (
    ExternalHemisphere,
    InternalHemisphere,
)
from synthesis_lab.orchestration.rdl.brain.oscillate import OscillationConfig, OscillationTrainer, Phase
from synthesis_lab.orchestration.rdl.brain.rsi_outer_gate import private_axes
from synthesis_lab.orchestration.rdl.brain.train_level_groups import _mean_phase_loss


def test_default_feature_pad_matches_the_operator_garden():
    assert CentralConfig().extra_feats == EXTRA_FEAT_DIM
    assert EXTRA_FEAT_DIM == 3696


def test_one_feature_row_broadcasts_across_the_batch():
    cfg = CentralConfig(emb=8, n_actions=4, extra_feats=4, hidden=8)
    cortex = CentralCortex(cfg)
    hidden = torch.zeros(3, 8)
    extra = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    fused = cortex.fuse_features(
        hidden,
        hidden,
        judge_I=torch.tensor([0.5]),
        disagreement=torch.tensor([0.25]),
        extra=extra,
    )
    assert fused.shape == (3, 8 * 4 + 2 + 4)
    assert torch.allclose(fused[:, -4:], extra.expand(3, 4))
    assert torch.allclose(fused[:, 32], torch.full((3,), 0.5))
    assert torch.allclose(fused[:, 33], torch.full((3,), 0.25))


def test_last_garden_feature_is_not_truncated():
    cortex = CentralCortex(CentralConfig(emb=4, hidden=8))
    hidden = torch.zeros(1, 4)
    extra = torch.zeros(1, EXTRA_FEAT_DIM)
    extra[0, -1] = 7.0
    fused = cortex.fuse_features(hidden, hidden, extra=extra)
    assert fused.shape[-1] == 4 * 4 + 2 + EXTRA_FEAT_DIM
    assert float(fused[0, -1]) == 7.0


def test_lora_rank_must_be_positive():
    try:
        LoRALinear(nn.Linear(4, 4), rank=0)
    except ValueError as exc:
        assert "rank" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_ada_repeats_a_single_feature_row():
    central = CentralCortex(CentralConfig(emb=8, n_actions=4, extra_feats=3, hidden=8))
    ada = CentralAdaOnline(
        central,
        InternalHemisphere(emb=8, n_actions=4),
        ExternalHemisphere(emb=8, n_actions=4),
        device="cpu",
    )
    frames = torch.randint(0, 4, (2, 8, 8))
    ada.observe_transition(frames, torch.tensor([1]), frames + 1, extra=torch.ones(1, 3))
    assert len(ada.buffer) == 2
    assert tuple(ada.buffer[0]["extra"].shape) == (3,)
    assert tuple(ada.buffer[1]["extra"].shape) == (3,)
    assert float(ada.buffer[1]["extra"].sum()) == 3.0


def test_single_row_central_phase_skips_median_trust_target():
    trainer = OscillationTrainer(cfg=OscillationConfig(emb=8, n_actions=4, device="cpu"))
    frames = torch.randint(0, 8, (1, 8, 8))
    metrics = trainer.step_batch(
        frames,
        torch.tensor([1]),
        (frames + 1) % 8,
        phase=Phase.C,
    )
    assert metrics["loss_trust"] == 0.0


def test_mean_phase_loss_counts_each_step_once():
    logs = [
        {"phase": "external", "loss": 1.0, "loss_E": 1.0},
        {
            "phase": "central",
            "loss": 3.0,
            "loss_C": 3.0,
            "loss_pol": 2.0,
            "loss_trust": 5.0,
        },
        {"phase": "coverage", "extra_dim": 4},
    ]
    assert _mean_phase_loss(logs) == 2.0


def _cohort(ups, levels, deltas, games_with):
    return {
        "collect": {
            "n_transitions": 10,
            "n_games_ok": len(ups),
            "per_game": [
                {
                    "status": "ok",
                    "max_levels_seen": levels[i],
                    "n_level_ups": ups[i],
                    "mean_delta_frac": deltas[i],
                }
                for i in range(len(ups))
            ],
            "mean_max_levels": sum(levels) / len(levels),
            "total_level_ups": float(sum(ups)),
            "games_with_level_up": games_with,
            "mean_delta_frac": sum(deltas) / len(deltas),
        },
        "train": {
            "final_mean_loss": 1.0,
            "n_games": len(ups),
            "games_hit": len(ups),
            "robust_coverage": 1.0,
            "epoch_curve": [{"mean_loss": 2.0}, {"mean_loss": 1.0}],
        },
        "pattern_robustness": {"pattern_robust_frac": 1.0},
    }


def test_level_ups_are_not_added_twice():
    metrics = {
        "results": [_cohort([1, 2], [0, 2], [0.1, 0.1], 2)],
        "suite": {"n_games": 2},
    }
    axes = private_axes(metrics)
    assert axes["total_level_ups"] == 3.0
    assert axes["games_up_frac"] == 1.0
    assert axes["mean_max_levels"] == 1.0


def test_level_up_games_sum_across_groups():
    axes = private_axes(
        {
            "results": [
                _cohort([1, 0], [1, 0], [0.0, 0.0], 1),
                _cohort([0, 1], [0, 1], [0.0, 0.0], 1),
            ],
            "suite": {"n_games": 2},
        }
    )
    assert axes["total_level_ups"] == 2.0
    assert axes["games_up_frac"] == 0.5


def test_games_up_fraction_uses_the_reported_suite_size():
    axes = private_axes(
        {
            "results": [
                {
                    "collect": {
                        "games_with_level_up": 2,
                        "n_games_ok": 2,
                        "n_transitions": 4,
                    },
                    "train": {
                        "final_mean_loss": 1.0,
                        "n_games": 2,
                        "robust_coverage": 1.0,
                    },
                    "pattern_robustness": {"pattern_robust_frac": 1.0},
                }
            ],
            "suite": {"n_games": 2},
        }
    )
    assert axes["games_up_frac"] == 1.0


def test_brain_state_snapshot_is_json_and_omits_dream_tensors():
    import json

    state = BrainState(
        dream_summary={"status": "ok", "_dream_next": torch.zeros(2, 3)},
        disagreement=float("nan"),
    )
    payload = state.to_dict()
    json.dumps(payload)
    assert "_dream_next" not in payload["dream_summary"]
    assert payload["dream_summary"]["status"] == "ok"
    assert payload["disagreement"] is None
    assert payload["truth_flags"]["solve"] is False
