"""LeWorldModel (LeWM) — clean paper-faithful implementation.

Reference: Maes et al., "LeWorldModel: Stable End-to-End Joint-Embedding
Predictive Architecture from Pixels," arXiv:2603.19312.

This module is a self-contained implementation of LeWM. It has no
dependencies on the Dreamweaver code base. Inputs are raw RGB image
tensors; outputs are compact Gaussian-regularized latents and
action-conditioned next-latent predictions.

GO-TO MODEL: :class:`MegaREX` — Mockingbird-infused LeWM with RMSNorm,
QK-gain attention, leaky-ReLU² MLP, residual mixing, loop blocks, and
skip gates. Three tiers: mega_rex_small (2.1M), mega_rex_medium (12.1M),
mega_rex_full (61.4M).

LEGACY: :class:`LeWM` — baseline paper-faithful implementation kept for
reference and regression testing. Use MegaREX for all new training.
"""

from lewm.cem import CEMConfig, CEMPlanner, CEMResult, PAPER_CEM_DEFAULTS
from lewm.encoder import EncoderConfig, ViTTinyEncoder
from lewm.guided_motion import GuidedMotionAdapter, GuidedMotionConfig
from lewm.loss import PAPER_REG_LAMBDA, LossOutput, lewm_loss, prediction_loss
from lewm.mega_rex import MegaREX, MegaREXConfig  # ← go-to model
from lewm.model import LeWM, LeWMConfig            # ← legacy, kept for reference
from lewm.predictor import (
    AdaLNTransformerPredictor,
    AdaLNZeroBlock,
    PredictorConfig,
)
from lewm.rex import (
    CURRENT_CHASSIS,
    REX_1_0_1_GUIDED_MOTION,
    REX_V1,
    REX_2_0_0,
    REX_3_0_0,
    ChassisRelease,
    GuidedMotionRelease,
    __version__,
)
from lewm.rex_goal_packet import (
    RexGoalPacket,
    RexGoalPacketProjector,
    RexGoalProjection,
    RexGoalSource,
    confidence_from_goal_property_outputs,
    load_packet_tensor_bundle,
    packet_from_tensor_bundle,
    packet_to_properties_map,
    packet_from_goal_property_outputs,
)
from lewm.sigreg import (
    PAPER_NUM_PROJECTIONS,
    PAPER_QUAD_MAX,
    PAPER_QUAD_MIN,
    PAPER_QUAD_NODES,
    PAPER_WEIGHT_LAMBDA,
    SIGReg,
    SIGRegConfig,
)
from lewm.training import LeWMTrainer, TrainingConfig

__all__ = [
    "AdaLNTransformerPredictor",
    "AdaLNZeroBlock",
    "CEMConfig",
    "CEMPlanner",
    "CEMResult",
    "CURRENT_CHASSIS",
    "ChassisRelease",
    "EncoderConfig",
    "GuidedMotionAdapter",
    "GuidedMotionConfig",
    "GuidedMotionRelease",
    "LeWM",                     # legacy
    "LeWMConfig",               # legacy
    "LeWMTrainer",
    "LossOutput",
    "MegaREX",                  # ← go-to model
    "MegaREXConfig",            # ← go-to config
    "PAPER_CEM_DEFAULTS",
    "PAPER_NUM_PROJECTIONS",
    "PAPER_QUAD_MAX",
    "PAPER_QUAD_MIN",
    "PAPER_QUAD_NODES",
    "PAPER_REG_LAMBDA",
    "PAPER_WEIGHT_LAMBDA",
    "PredictorConfig",
    "REX_1_0_1_GUIDED_MOTION",
    "REX_2_0_0",
    "REX_3_0_0",
    "REX_V1",
    "RexGoalPacket",
    "RexGoalPacketProjector",
    "RexGoalProjection",
    "RexGoalSource",
    "SIGReg",
    "SIGRegConfig",
    "TrainingConfig",
    "ViTTinyEncoder",
    "__version__",
    "confidence_from_goal_property_outputs",
    "load_packet_tensor_bundle",
    "packet_from_tensor_bundle",
    "lewm_loss",
    "packet_to_properties_map",
    "packet_from_goal_property_outputs",
    "prediction_loss",
]
