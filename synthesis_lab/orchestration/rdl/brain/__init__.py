"""Dual-hemisphere RDL brain: Internal (dream) + External (judge) + Central LoRA.

See DESIGN.md. Scaffold + WP-C MegaMind facade — env-graded headlines still
require Arcade.
"""

from .central import CentralConfig, CentralCortex
from .hemispheres import ExternalHemisphere, HemisphereBundle, InternalHemisphere
from .megamind import MegaMind
from .oscillate import OscillationConfig, OscillationTrainer, Phase
from .brain_types import BrainState, KnownMapView, OperatorResult, TruthFlags

__all__ = [
    "BrainState",
    "CentralConfig",
    "CentralCortex",
    "ExternalHemisphere",
    "HemisphereBundle",
    "InternalHemisphere",
    "KnownMapView",
    "MegaMind",
    "OperatorResult",
    "OscillationConfig",
    "OscillationTrainer",
    "Phase",
    "TruthFlags",
]
