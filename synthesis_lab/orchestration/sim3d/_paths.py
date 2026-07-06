"""sys.path bootstrap for the sim3d package (same convention as orchestration/*.py)."""
from __future__ import annotations

import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent            # .../orchestration/sim3d
ORCH = PKG.parent                                # .../orchestration
LAB = ORCH.parent                                # .../synthesis_lab
REPO = LAB.parent                                # .../megarexus
FIXTURES = PKG / "fixtures"

for _p in (LAB, ORCH, LAB / "experiments" / "megarexus_cnn", REPO / "codexlab"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
