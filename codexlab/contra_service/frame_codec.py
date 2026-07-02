#!/usr/bin/env python3
"""Raw-frame transport codec for the CONTRA service boundary.

In-process clients pass the numpy frame straight through.  Over HTTP a frame is
either inlined as nested integer lists (``raw_frame``) or referenced by path +
sha (``raw_frame_ref``) for large frames.  The canonical frame digest is
``contra_head.vision.sha256_raw_grid`` -- the same digest the CONTRA response carries
as ``source_raw_frame_sha256`` -- so a client can round-trip
``frame -> packet.source_raw_frame_sha256 == frame_sha256(frame)``.

NOTE: this is deliberately NOT the harness ``sha256_frame`` used in on-disk
``.npy`` metadata; that is a different algorithm.  A ``raw_frame_ref`` MUST carry
the canonical ``sha256_raw_grid`` digest (the ``sha_algorithm`` tag makes this
explicit), and the anti-spoof guard validates against it -- a loud failure on
mismatch, never a silent frame substitution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from contra_head.vision import sha256_raw_grid

SHA_ALGORITHM = "contra_head.vision.sha256_raw_grid"


def json_default(obj: Any) -> Any:
    """numpy-safe ``json.dumps(default=...)`` shared by the server and clients."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _as_int16(frame: Any) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.int16)
    if arr.ndim != 2:
        raise ValueError(f"raw_frame must be 2-D, got shape {arr.shape!r}")
    return arr


def encode_frame(frame: Any) -> list[list[int]]:
    """numpy/array-like -> nested python int lists (JSON-serializable)."""
    return _as_int16(frame).tolist()


def decode_frame(value: Any) -> np.ndarray:
    """nested lists / array-like -> int16 ndarray (matches the persisted format)."""
    return _as_int16(value)


def frame_sha256(frame: Any) -> str:
    """Canonical raw-frame digest (== CONTRA response ``source_raw_frame_sha256``)."""
    return sha256_raw_grid(_as_int16(frame))


def frame_ref(frame: Any, *, path: str, shape: list[int] | None = None) -> dict[str, Any]:
    """Build a ``raw_frame_ref`` envelope carrying the canonical sha + algorithm tag."""
    arr = _as_int16(frame)
    return {
        "path": str(path),
        "raw_frame_sha256": sha256_raw_grid(arr),
        "sha_algorithm": SHA_ALGORITHM,
        "shape": [int(v) for v in (shape or arr.shape)],
        "dtype": "int16",
    }


def resolve_frame(payload: dict[str, Any], *, root: Path | None = None) -> np.ndarray:
    """Resolve a request's frame from inline ``raw_frame`` or ``raw_frame_ref``.

    For a ref, loads the ``.npy`` and (if a sha is supplied) validates it against
    the canonical digest -- a loud failure on mismatch, never a silent swap.
    """
    if not isinstance(payload, dict):
        raise ValueError("request payload must be a dict")
    inline = payload.get("raw_frame")
    if inline is not None:
        arr = decode_frame(inline)
        claimed = payload.get("raw_frame_sha256")
        if claimed is not None and claimed != sha256_raw_grid(arr):
            raise ValueError(
                "raw_frame_sha256 does not match inline raw_frame (canonical sha256_raw_grid)"
            )
        return arr
    ref = payload.get("raw_frame_ref")
    if isinstance(ref, dict):
        raw_path = Path(str(ref.get("path")))
        base = Path(root) if root is not None else Path.cwd()
        full = raw_path if raw_path.is_absolute() else base / raw_path
        arr = _as_int16(np.load(full))
        claimed = ref.get("raw_frame_sha256")
        if claimed is not None and claimed != sha256_raw_grid(arr):
            raise ValueError(
                f"raw_frame_ref sha mismatch for {full}: expected canonical {SHA_ALGORITHM} digest "
                "(not the harness sha256_frame .npy-metadata digest)"
            )
        return arr
    raise ValueError("request has neither raw_frame nor raw_frame_ref")
