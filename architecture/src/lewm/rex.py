"""Named runnable LeWM chassis releases.

``LeWM`` names the mathematical architecture. ``Rex`` names the first
versioned runnable chassis artifact we trust enough to build on.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChassisRelease:
    codename: str
    version: str
    release_id: str
    architecture: str
    checkpoint_path: str
    checkpoint_sha256: str
    summary_path: str
    summary_schema: str
    trained_at_utc: str
    seed: int
    sequence_len: int
    latent_dim: int
    num_parameters: int
    best_val_pred: float
    best_epoch: int
    locked: bool
    successor_rule: str
    notes: tuple[str, ...]


@dataclass(frozen=True)
class GuidedMotionRelease:
    codename: str
    version: str
    release_id: str
    parent_release_id: str
    adapter_path: str
    adapter_sha256: str
    report_path: str
    report_sha256: str
    guidance_mode: str
    controlled_examples: int
    gate_verdict: str
    gate_target_rank_1: str
    locked_parent_unchanged: bool
    notes: tuple[str, ...]


REX_V1 = ChassisRelease(
    codename="Rex",
    version="1.0.0",
    release_id="rex-1.0.0",
    architecture="paper-faithful LeWM encoder + action-conditioned predictor",
    checkpoint_path="experiments/2026-05-09_chassis_paper_faithful_v1/best.pt",
    checkpoint_sha256=(
        "6141b2bf1f3dedb2798699082fdee35ea5d71b6f30c5f36e7955269dd117a18d"
    ),
    summary_path="experiments/2026-05-09_chassis_paper_faithful_v1/summary.json",
    summary_schema="lewm.arc_pretrain.v003",
    trained_at_utc="2026-05-10T01:20:30.721996+00:00",
    seed=0,
    sequence_len=8,
    latent_dim=192,
    num_parameters=10_228_800,
    best_val_pred=0.0003167311903477336,
    best_epoch=3,
    locked=True,
    successor_rule=(
        "Do not mutate Rex 1.0.0. Any guided-control, fine-tuned, or "
        "planner-integrated derivative must receive a new release id."
    ),
    notes=(
        "First named runnable chassis release.",
        "No goal/planner semantics are implied by this release.",
        "External-agent and NEMO layers are outside the trusted LeWM core.",
        "Locked baseline as of game_theory_v1 commit 6b41025.",
    ),
)

REX_1_0_1_GUIDED_MOTION = GuidedMotionRelease(
    codename="Rex",
    version="1.0.1",
    release_id="rex-1.0.1-guided-motion",
    parent_release_id=REX_V1.release_id,
    adapter_path="experiments/2026-05-10_rex_1_0_1_guided_motion/adapter.pt",
    adapter_sha256=(
        "748a80548b94b1172e4ec8f0506093fb9637e8f5036d8653b2f5b6d92644e7b6"
    ),
    report_path=(
        "experiments/2026-05-10_rex_1_0_1_guided_motion/motion_probe/report.json"
    ),
    report_sha256=(
        "7e90e19ec9fb8719e3266e9ce1513abeb65583e4a2456144c2eb6f713f755d99"
    ),
    guidance_mode="frozen Rex + residual adapter + controlled trace memory",
    controlled_examples=48,
    gate_verdict="pass",
    gate_target_rank_1="48/48",
    locked_parent_unchanged=True,
    notes=(
        "Guided motion literacy candidate for controlled up/right traces.",
        "This release proves ingestion of recorded traces, not unseen-motion generalization.",
        "Rex 1.0.0 remains the locked base chassis.",
    ),
)

REX_2_0_0 = ChassisRelease(
    codename="Rex",
    version="2.0.0",
    release_id="rex-2.0.0-structured-actions",
    architecture=(
        "paper-faithful LeWM encoder + action-conditioned predictor + "
        "family-structured action embedder"
    ),
    checkpoint_path="experiments/2026-05-10_rex_2_0_0_structured_actions/best.pt",
    checkpoint_sha256=(
        "c705e0b00e48a59bb4f5de88976a206fd9797b35a32f2d9ec2b9a2b242c55e1f"
    ),
    summary_path="experiments/2026-05-10_rex_2_0_0_structured_actions/summary.json",
    summary_schema="lewm.rrrr_pretrain.v002",
    trained_at_utc="2026-05-10T16:45:00+00:00",
    seed=0,
    sequence_len=8,
    latent_dim=192,
    num_parameters=10_229_568,
    best_val_pred=1.3482e-5,
    best_epoch=49,
    locked=True,
    successor_rule=(
        "Structured action embedder is now part of the base architecture. "
        "Any follow-up that modifies the embedder structure must create a "
        "new release. This release supersedes Rex 1.0.0 as the default chassis."
    ),
    notes=(
        "Structured action embedder with family bases (null/directional/placement/other). "
        "Action separation ratio: 29.8 (was 0.58 in Rex 1.0.0 — 50× improvement). "
        "Val pred loss: 1.35e-5 (was 3.17e-4 — 24× improvement). "
        "Action probe accuracy: 49.2% (was 41.5%). "
        "T4 goal sensitivity ratio: 0.71 (was 1.04 — terminal = lower error now). "
        "Trained 50 epochs on 180 trajectories from 25 games.",
    ),
)

REX_3_0_0 = ChassisRelease(
    codename="Rex",
    version="3.0.0",
    release_id="rex-3.0.0-goal-conditioned",
    architecture=(
        "REX 2.0 architecture + goal-conditioned AdaLN predictor + "
        "GoalLatentReconstructor + joint goal/LeWM oscillation on "
        "staged_dataset_v1 (16,800 episodes)"
    ),
    checkpoint_path="experiments/2026-05-11_rex_3_0_0_goal_conditioned/rex3_joint_model.pt",
    checkpoint_sha256="TRAINING_IN_PROGRESS",
    summary_path="experiments/2026-05-11_rex_3_0_0_goal_conditioned/metrics.json",
    summary_schema="loom.rex3_joint.v001",
    trained_at_utc="2026-05-11T00:00:00+00:00",
    seed=0,
    sequence_len=8,
    latent_dim=192,
    num_parameters=10_229_568,
    best_val_pred=0.0,
    best_epoch=-1,
    locked=False,
    successor_rule=(
        "This release introduces goal conditioning into the LeWM predictor. "
        "The predictor's AdaLN blocks now accept (action_emb, z_goal, λ_conf) "
        "as conditioning, making p(t+1) = f(z_t, a_t, z_goal). "
        "Any follow-up that changes the goal latent interface must create "
        "a new release. Training in progress — checkpoint values are provisional."
    ),
    notes=(
        "REX 3.0.0 — goal-conditioned chassis candidate, not promoted.",
        "Do not use as CURRENT_CHASSIS until its checkpoint path, SHA, metrics, "
        "and ARC gate report are updated from a verified promotion packet.",
        "Goal latent reconstructor maps (goal_type, agent_delta, keys, delta_px) → 192-dim latent.",
        "Confidence gate λ_conf scales goal influence per-sample (learned).",
        "Oscillation: joint backward pass (weighted) so gradients flow through "
        "LeWM → reconstructor → goal heads → encoder (mutual bootstrapping).",
        "Trained on staged_dataset_v1: 16,800 episodes, 168 mechanics, 100 seeds each.",
        "Predictor same size as REX 2.0 (~10.2M) + GoalLatentReconstructor (~0.5M additional).",
        "Training: python scripts/train_rex3_joint.py",
    ),
)

CURRENT_CHASSIS = REX_2_0_0
__version__ = REX_2_0_0.version

__all__ = [
    "CURRENT_CHASSIS",
    "REX_1_0_1_GUIDED_MOTION",
    "REX_V1",
    "REX_2_0_0",
    "REX_3_0_0",
    "ChassisRelease",
    "GuidedMotionRelease",
    "__version__",
]
