"""Imagination informers — gentle advisors for the PUCT prior.

Each informer is a single file: dataclass + factory function or class. To
add a new informer:

  1. Drop ``src/lewm/informers/your_informer.py``
  2. Export the factory from this ``__init__``
  3. Add a one-liner test in ``tests/lewm/test_imagination.py``
  4. (Optional) wire into ``scripts/imagination_ablation.py`` to measure
     its KL-to-uniform contribution

No MCTS or PriorMixer code changes required.
"""

from lewm.informers.sprite_density import SpriteDensityInformer
from lewm.informers.replay_frequency import ReplayFrequencyInformer
from lewm.informers.affordance import AffordanceHeadInformer
from lewm.informers.wall_aware import WallAwareInformer

__all__ = [
    "SpriteDensityInformer",
    "ReplayFrequencyInformer",
    "AffordanceHeadInformer",
    "WallAwareInformer",
]
