"""MEGAMIND operators: body / ray / field / conceptual plugs."""

from __future__ import annotations

from .base import Operator, OperatorResult

try:
    from .body import BodyOp
except Exception:  # pragma: no cover
    BodyOp = None  # type: ignore

try:
    from .ray_ops import RayOp
except Exception:  # pragma: no cover
    RayOp = None  # type: ignore

try:
    from .field_ops import FieldOp
except Exception:  # pragma: no cover
    FieldOp = None  # type: ignore

try:
    from .conceptual import ConceptualOp
except Exception:  # pragma: no cover
    ConceptualOp = None  # type: ignore

try:
    from .portband_op import PortBandOp
except Exception:  # pragma: no cover
    PortBandOp = None  # type: ignore

__all__ = [
    "Operator",
    "OperatorResult",
    "BodyOp",
    "RayOp",
    "FieldOp",
    "ConceptualOp",
    "PortBandOp",
]
