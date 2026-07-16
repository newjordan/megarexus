"""megarexus one-brain public surface.

Implementation lives in synthesis_lab.orchestration.rdl.brain — this package
is the single import path for application code.
"""

from synthesis_lab.orchestration.rdl.brain import (  # noqa: F401
    MegaMind,
    CentralCortex,
    CentralConfig,
    HemisphereBundle,
    InternalHemisphere,
    ExternalHemisphere,
    OscillationTrainer,
    OscillationConfig,
    Phase,
    BrainState,
    KnownMapView,
    OperatorResult,
    TruthFlags,
)

__all__ = [
    "MegaMind",
    "CentralCortex",
    "CentralConfig",
    "HemisphereBundle",
    "InternalHemisphere",
    "ExternalHemisphere",
    "OscillationTrainer",
    "OscillationConfig",
    "Phase",
    "BrainState",
    "KnownMapView",
    "OperatorResult",
    "TruthFlags",
]
