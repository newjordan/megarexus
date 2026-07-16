"""legal_search — mine every megarexus system into legal consumable algorithms.

See README.md and MINING_MANIFEST.md.
"""

from legal_search.consume.registry import CONSUME_REGISTRY, consume_all, list_systems
from legal_search.contract import LEGAL_RULES, TRUTH_BOUNDARY
from legal_search.loop import LegalAgent, run_legal_episode

__all__ = [
    "CONSUME_REGISTRY",
    "LEGAL_RULES",
    "TRUTH_BOUNDARY",
    "LegalAgent",
    "consume_all",
    "list_systems",
    "run_legal_episode",
]

__version__ = "0.1.0"
