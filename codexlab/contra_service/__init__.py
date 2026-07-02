#!/usr/bin/env python3
"""CONTRA informer-DAG service: library-core + thin service-shell.

The reasoning subsystems (Oracle, Ghost, CONTRA) are *informers*; none admits
truth.  Oracle and Ghost inform CONTRA; CONTRA synthesises guidance and is
itself an informer, hooked to a harness that executes in the live game -- the
only truth source.  This package exposes that DAG over a stable, versioned,
serializable contract so the subsystems can be shared and (later) driven by an
LLM, while the live action loop keeps calling the same handlers in-process.

The server shell (``app``/``serve``) is intentionally NOT imported here, so
``import contra_service`` never pulls in ``http.server`` or the optional FastAPI
dependency.  The dependency arrow is one-way: ``contra_service -> contra`` (and
``-> sp80_player_harness`` in handlers); ``contra`` must never import this.
"""

from __future__ import annotations

from . import contracts, frame_codec, handlers, projections
from .client import ContraClient, HttpContraClient, InProcessContraClient
from .contracts import (
    CONTRACTS,
    CONTRACTS_BY_HANDLER,
    CONTRACTS_BY_PATH,
    EndpointContract,
    assert_no_truth_admission,
    contracts_document,
    truth_admission_violations,
)

__all__ = [
    "contracts",
    "frame_codec",
    "handlers",
    "projections",
    "ContraClient",
    "InProcessContraClient",
    "HttpContraClient",
    "CONTRACTS",
    "CONTRACTS_BY_PATH",
    "CONTRACTS_BY_HANDLER",
    "EndpointContract",
    "assert_no_truth_admission",
    "truth_admission_violations",
    "contracts_document",
]
