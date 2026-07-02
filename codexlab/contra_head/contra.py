#!/usr/bin/env python3
"""Contra coordination surface.

This module is the package control mechanism. It keeps the public Contra API
stable while the concrete systems live in DRY modules beside it. Vision remains
the mandatory primary sense under :mod:`contra.vision`; control contract helpers
live in :mod:`contra.controls`.
"""

from __future__ import annotations

from .api import *  # noqa: F401,F403
from .action_semantics_ledger import *  # noqa: F401,F403
from .boundary_pass import *  # noqa: F401,F403
from .controls import *  # noqa: F401,F403
from .debug import *  # noqa: F401,F403
from .deltaband import *  # noqa: F401,F403
from .actionmodband import *  # noqa: F401,F403
from .episode_memory import *  # noqa: F401,F403
from .exploreband import *  # noqa: F401,F403
from .geoband import *  # noqa: F401,F403
from .ghost import *  # noqa: F401,F403
from .handoff import *  # noqa: F401,F403
from .oversight import *  # noqa: F401,F403
from .perception import *  # noqa: F401,F403
from .portband import *  # noqa: F401,F403
from .relationship_ladder import *  # noqa: F401,F403
from .statecon import *  # noqa: F401,F403
from .symmetry import *  # noqa: F401,F403
from .tabula_rasa import *  # noqa: F401,F403
from .vision import *  # noqa: F401,F403


CONTRA_MODULE_SOURCE_PATHS = [
    "codexlab/contra_head/contra.py",
    "codexlab/contra_head/api.py",
    "codexlab/contra_head/action_semantics_ledger.py",
    "codexlab/contra_head/controller_head.py",
    "codexlab/contra_head/boundary_pass.py",
    "codexlab/contra_head/controls.py",
    "codexlab/contra_head/debug.py",
    "codexlab/contra_head/handoff.py",
    "codexlab/contra_head/ghost.py",
    "codexlab/contra_head/vision.py",
    "codexlab/contra_head/perception.py",
    "codexlab/contra_head/portband.py",
    "codexlab/contra_head/statecon.py",
    "codexlab/contra_head/symmetry.py",
    "codexlab/contra_head/tabula_rasa.py",
    "codexlab/contra_head/geoband.py",
    "codexlab/contra_head/oversight.py",
    "codexlab/contra_head/relationship_ladder.py",
    "codexlab/contra_head/deltaband.py",
    "codexlab/contra_head/actionmodband.py",
    "codexlab/contra_head/exploreband.py",
    "codexlab/contra_head/episode_memory.py",
    "codexlab/contra_head/view_system.py",
]


def contra_module_source_paths() -> list[str]:
    return list(CONTRA_MODULE_SOURCE_PATHS)
