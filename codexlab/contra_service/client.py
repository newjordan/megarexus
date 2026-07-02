#!/usr/bin/env python3
"""Clients for the CONTRA informer-DAG service.

A single :class:`ContraClient` Protocol with one method per endpoint, plus a
transport-agnostic :meth:`call` dispatched by contract handler-name.  Two
implementations route to the SAME ``handlers`` functions:

  * :class:`InProcessContraClient` -- direct handler calls, no serialization.
    This is what the live action loop and unit tests hold; a numpy ``raw_frame``
    is passed straight through (exact live-loop parity, no network).
  * ``HttpContraClient`` (added in the HTTP stage) -- canonicalises to JSON and
    POSTs over stdlib ``urllib``.  Because both share one handler, a parity test
    can assert their responses are byte-identical: HTTP can never diverge from
    the verified in-process handler.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from . import frame_codec, handlers
from .contracts import CONTRACTS_BY_HANDLER


@runtime_checkable
class ContraClient(Protocol):
    """Stable client surface for the informer DAG (one method per endpoint)."""

    transport: str

    def call(self, handler_name: str, request: dict | None = None) -> dict: ...
    def contra_view(self, request: dict) -> dict: ...
    def oracle_pass(self, request: dict) -> dict: ...
    def ghost_pass(self, request: dict) -> dict: ...
    def ghost_context(self, request: dict) -> dict: ...
    def recommendation(self, request: dict) -> dict: ...
    def feedback_ingest(self, request: dict) -> dict: ...
    def contracts(self) -> dict: ...
    def health(self) -> dict: ...


class InProcessContraClient:
    """Calls the handler functions directly -- the live-loop fast path."""

    transport = "in_process"

    def __init__(self, *, root: Path | None = None) -> None:
        self._root = root

    def call(self, handler_name: str, request: dict | None = None) -> dict:
        if handler_name not in CONTRACTS_BY_HANDLER:
            raise KeyError(f"unknown handler: {handler_name!r}")
        handler = handlers.HANDLERS[handler_name]
        if request is None:
            return handler()
        if self._root is not None:
            return handler(request, root=self._root)
        return handler(request)

    def contra_view(self, request: dict) -> dict:
        return self.call("contra_view", request)

    def oracle_pass(self, request: dict) -> dict:
        return self.call("oracle_pass", request)

    def ghost_pass(self, request: dict) -> dict:
        return self.call("ghost_pass", request)

    def ghost_context(self, request: dict) -> dict:
        return self.call("ghost_context", request)

    def recommendation(self, request: dict) -> dict:
        return self.call("recommendation", request)

    def feedback_ingest(self, request: dict) -> dict:
        return self.call("feedback_ingest", request)

    def contracts(self) -> dict:
        return self.call("contracts", None)

    def health(self) -> dict:
        return self.call("health", None)


def _canonicalize_request(request: dict | None) -> dict:
    """Make a request JSON-safe for the wire: numpy ``raw_frame`` -> nested lists.
    Other fields are expected already JSON-clean (e.g. a prior CONTRA response
    fed back as ``contra_packet``)."""
    if not isinstance(request, dict):
        return {}
    out = dict(request)
    rf = out.get("raw_frame")
    if isinstance(rf, np.ndarray):
        out["raw_frame"] = frame_codec.encode_frame(rf)
    return out


class HttpContraClient:
    """POSTs/GETs the same handlers over stdlib ``urllib`` -- for sharing, parity
    testing, and LLM invocation.  Shares one handler implementation with the
    in-process client, so a parity check can prove HTTP never diverges."""

    transport = "http"

    def __init__(self, base_url: str, *, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)

    def call(self, handler_name: str, request: dict | None = None) -> dict:
        contract = CONTRACTS_BY_HANDLER.get(handler_name)
        if contract is None:
            raise KeyError(f"unknown handler: {handler_name!r}")
        url = self.base_url + contract.path
        if contract.method == "GET":
            req = urllib.request.Request(url, method="GET")
        else:
            data = json.dumps(_canonicalize_request(request), default=frame_codec.json_default).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, method="POST", headers={"Content-Type": "application/json"}
            )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc

    def contra_view(self, request: dict) -> dict:
        return self.call("contra_view", request)

    def oracle_pass(self, request: dict) -> dict:
        return self.call("oracle_pass", request)

    def ghost_pass(self, request: dict) -> dict:
        return self.call("ghost_pass", request)

    def ghost_context(self, request: dict) -> dict:
        return self.call("ghost_context", request)

    def recommendation(self, request: dict) -> dict:
        return self.call("recommendation", request)

    def feedback_ingest(self, request: dict) -> dict:
        return self.call("feedback_ingest", request)

    def contracts(self) -> dict:
        return self.call("contracts", None)

    def health(self) -> dict:
        return self.call("health", None)
