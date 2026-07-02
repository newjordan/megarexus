#!/usr/bin/env python3
"""HTTP service shell for the CONTRA informer DAG.

Thin adapter over ``handlers`` -- the handlers are the single source of truth;
this module only moves JSON across the wire.  The mandatory shell is stdlib
``http.server`` (zero new hard dependencies, matching the repo's ``urllib`` house
style).  An optional FastAPI app is offered behind an import guard for OpenAPI /
Swagger discovery when the dependency is present; its absence never breaks a run,
and the hand-rolled ``GET /v1/contracts`` provides the same discovery contract on
the stdlib path.

IMPORTANT: ``import contra_service`` must NOT import this module (it would pull in
``http.server``/FastAPI); the package ``__init__`` deliberately omits it.  Boot a
server only via :func:`build_server` / :func:`serve` / ``serve.py``.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import handlers
from .contracts import CONTRACTS, CONTRACTS_BY_PATH
from .frame_codec import json_default

ROOT = Path(__file__).resolve().parents[2]


def dumps(payload: Any) -> bytes:
    """Canonical JSON encoding shared by the server and the parity check."""
    return json.dumps(payload, default=json_default).encode("utf-8")


class _ContraRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # -- helpers --
    def _root(self) -> Path:
        return getattr(self.server, "contra_root", ROOT)

    def _send(self, status: int, payload: Any) -> None:
        body = dumps(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _contract_for(self, method: str):
        path = urlsplit(self.path).path
        contract = CONTRACTS_BY_PATH.get(path)
        if contract is None or contract.method != method:
            return None, path
        return contract, path

    # -- verbs --
    def do_GET(self) -> None:  # noqa: N802
        contract, path = self._contract_for("GET")
        if contract is None:
            return self._send(404, {"error": "not_found", "path": path})
        try:
            self._send(200, handlers.HANDLERS[contract.handler]())
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": "handler_error", "detail": str(exc)})

    def do_HEAD(self) -> None:  # noqa: N802
        path = urlsplit(self.path).path
        contract = CONTRACTS_BY_PATH.get(path)
        ok = contract is not None and contract.method == "GET"
        self.send_response(200 if ok else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        contract, path = self._contract_for("POST")
        if contract is None:
            return self._send(404, {"error": "not_found", "path": path})
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            request = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            return self._send(400, {"error": "bad_json", "detail": str(exc)})
        try:
            result = handlers.HANDLERS[contract.handler](request, root=self._root())
            self._send(200, result)
        except ValueError as exc:
            self._send(400, {"error": "bad_request", "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": "handler_error", "detail": str(exc)})

    def log_message(self, *args: Any) -> None:  # silence per-request stderr noise
        return


def build_server(*, host: str = "127.0.0.1", port: int = 0, root: Path | str = ROOT) -> ThreadingHTTPServer:
    """Build (but do not start) a threading stdlib server.  ``port=0`` -> ephemeral."""
    httpd = ThreadingHTTPServer((host, int(port)), _ContraRequestHandler)
    httpd.contra_root = Path(root)  # type: ignore[attr-defined]
    return httpd


def serve(*, host: str = "127.0.0.1", port: int = 8973, root: Path | str = ROOT) -> None:
    httpd = build_server(host=host, port=port, root=root)
    bound = httpd.server_address
    print(f"contra_service stdlib server on http://{bound[0]}:{bound[1]} (root={root})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()


def build_fastapi_app(*, root: Path | str = ROOT):
    """Optional FastAPI app over the SAME handlers (OpenAPI/Swagger discovery).

    Raises ImportError if FastAPI is absent; callers should fall back to the
    stdlib server.  No per-schema pydantic models -- bodies are plain dicts -- so
    the handlers stay the single source of truth and there is no contract drift.
    """
    from fastapi import FastAPI, Request  # local import: never at module top

    api = FastAPI(title="CONTRA informer-DAG service", version="1")
    root_path = Path(root)

    def _make_post(handler_name: str):
        async def _endpoint(request: Request):
            body = await request.json() if (await request.body()) else {}
            return handlers.HANDLERS[handler_name](body, root=root_path)

        return _endpoint

    def _make_get(handler_name: str):
        async def _endpoint():
            return handlers.HANDLERS[handler_name]()

        return _endpoint

    for contract in CONTRACTS:
        if contract.method == "POST":
            api.post(contract.path, summary=contract.summary)(_make_post(contract.handler))
        else:
            api.get(contract.path, summary=contract.summary)(_make_get(contract.handler))
    return api
