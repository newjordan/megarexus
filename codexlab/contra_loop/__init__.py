"""CONTRA Loop Harness -- accumulate verified algorithmic solvers via a discovery loop.

Point an agent at codexlab/contra_loop/CONTRA_LOOP.md. Run via the cockpit:
    python codexlab/contra.py loop iterate --target sp80:1 --dry-run   # proves the spine
    python codexlab/contra.py loop status                              # the solver library
The engine is loop.py (9-step GROUND->..->ADMIT, 5-layer gate, self-verifying registry);
solver_registry.py holds the admission primitives loop.py/baseline.py build on.
"""

from __future__ import annotations

SCHEMA = "codexlab.contra_loop.v1"
