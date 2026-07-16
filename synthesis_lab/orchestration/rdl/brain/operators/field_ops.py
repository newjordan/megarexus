"""FieldOp — signed multi-variate potential field (pure numpy, megarexus-native).

Geometry for **dynamic multi-variate pattern growth**, not pour/SP80 roles:
attractors toward non-modal rare colors / largest components (per color var),
repel walls (border-touching solids) and frame borders. Works any HxW.
Returns net potential summary + suggested move deltas.
Hypothesis-only. No external project dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from synthesis_lab.orchestration.rdl.brain.types import KnownMapView, OperatorResult, TruthFlags


def _color_hist(frame: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(np.asarray(frame).reshape(-1), return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def _rare_colors(hist: dict[int, int], background: int, *, max_share: float = 0.08) -> set[int]:
    total = max(1, sum(hist.values()))
    rare: set[int] = set()
    for c, n in hist.items():
        if c == background:
            continue
        if (n / total) <= max_share:
            rare.add(int(c))
    return rare


def _is_wall_component(comp: dict[str, Any], h: int, w: int) -> bool:
    """Border-touching large / elongated components act as wall occluders."""
    x0, y0, x1, y1 = [int(v) for v in comp["bbox_xyxy"]]
    on_border = x0 <= 0 or y0 <= 0 or x1 >= w - 1 or y1 >= h - 1
    if not on_border:
        return False
    bw = max(1, x1 - x0 + 1)
    bh = max(1, y1 - y0 + 1)
    area = int(comp.get("area") or comp.get("area_pixels") or 0)
    elongated = max(bw, bh) >= 3 * min(bw, bh) or max(bw, bh) >= min(h, w) // 3
    large = area >= max(12, (h * w) // 80)
    return elongated or large


def _potential_at(
    x: float,
    y: float,
    attractors: list[tuple[float, float, float]],
    repellers: list[tuple[float, float, float]],
    h: int,
    w: int,
    *,
    border_weight: float = 2.0,
) -> float:
    """Signed potential: +attract / −repel / −border."""
    pot = 0.0
    for ax, ay, strength in attractors:
        d2 = (x - ax) ** 2 + (y - ay) ** 2 + 1.0
        pot += strength / d2
    for rx, ry, strength in repellers:
        d2 = (x - rx) ** 2 + (y - ry) ** 2 + 1.0
        pot -= strength / d2
    # border repulsion (distance to nearest edge, stronger near rim)
    dist_border = min(x, y, (w - 1) - x, (h - 1) - y)
    pot -= border_weight / (dist_border + 1.0) ** 2
    return float(pot)


def _gradient_delta(
    x: float,
    y: float,
    attractors: list[tuple[float, float, float]],
    repellers: list[tuple[float, float, float]],
    h: int,
    w: int,
    *,
    step: float = 1.0,
) -> tuple[float, float, float]:
    """Finite-diff gradient; return (dx, dy, pot_at_origin)."""
    p0 = _potential_at(x, y, attractors, repellers, h, w)
    px = _potential_at(x + step, y, attractors, repellers, h, w)
    py = _potential_at(x, y + step, attractors, repellers, h, w)
    gx = px - p0
    gy = py - p0
    norm = float(np.hypot(gx, gy))
    if norm < 1e-12:
        return 0.0, 0.0, p0
    # unit step in gradient direction (higher potential)
    return float(gx / norm), float(gy / norm), p0


class FieldOp:
    """Signed attract/repel field operator (megarexus-native)."""

    name = "field"

    def __init__(
        self,
        *,
        top_k_large: int = 3,
        rare_share: float = 0.08,
        attract_base: float = 4.0,
        repel_base: float = 3.0,
    ) -> None:
        self.top_k_large = int(top_k_large)
        self.rare_share = float(rare_share)
        self.attract_base = float(attract_base)
        self.repel_base = float(repel_base)

    def run(self, map_view: KnownMapView | Any, context: dict | None = None) -> OperatorResult:
        ctx = context or {}
        if isinstance(map_view, np.ndarray):
            frame: Any = map_view
        else:
            frame = getattr(map_view, "frame", None)

        if isinstance(map_view, KnownMapView):
            comps = list(map_view.components or [])
            bg = int(map_view.background)
            frame = map_view.frame
            if frame is None and map_view.solid is not None:
                # synthetic solid-only boards: treat solid as color 8, open as bg 0
                solid = np.asarray(map_view.solid, dtype=bool)
                frame = np.where(solid, 8, 0).astype(np.int16)
                bg = 0
            elif frame is not None:
                frame = np.asarray(frame)
        else:
            comps = list(ctx.get("components") or [])
            bg = int(ctx.get("background", 0))
            if frame is None or getattr(frame, "size", 0) == 0:
                return OperatorResult(
                    name=self.name,
                    status="degraded",
                    summary={"error": "no_frame"},
                    messages=["no_frame"],
                    truth_flags=TruthFlags(),
                )
            frame = np.asarray(frame)

        if frame is None or getattr(frame, "size", 0) == 0:
            return OperatorResult(
                name=self.name,
                status="degraded",
                summary={"error": "no_frame"},
                messages=["no_frame"],
                truth_flags=TruthFlags(),
            )

        if frame.ndim > 2:
            frame = frame[..., 0]
        h, w = int(frame.shape[0]), int(frame.shape[1])
        if not comps:
            try:
                from synthesis_lab.orchestration.rdl.brain.map_trust import extract_components

                comps, bg, _ = extract_components(frame)
            except Exception:
                comps = []
        hist = _color_hist(frame)
        rare = _rare_colors(hist, bg, max_share=self.rare_share)

        # Largest non-bg components (already sorted area-desc if from map_trust)
        sorted_comps = sorted(comps, key=lambda c: -int(c.get("area") or c.get("area_pixels") or 0))
        large_ids = {c.get("component_id") for c in sorted_comps[: self.top_k_large]}

        attractors: list[tuple[float, float, float]] = []
        repellers: list[tuple[float, float, float]] = []
        attractor_rows: list[dict[str, Any]] = []
        wall_rows: list[dict[str, Any]] = []

        for c in comps:
            cx, cy = float(c["center_xy"][0]), float(c["center_xy"][1])
            color = int(c.get("color") or 0)
            area = int(c.get("area") or c.get("area_pixels") or 0)
            area_scale = min(2.0, 0.5 + area / max(1.0, float(h * w) * 0.05))

            if _is_wall_component(c, h, w):
                strength = self.repel_base * area_scale
                repellers.append((cx, cy, strength))
                wall_rows.append(
                    {
                        "component_id": c.get("component_id"),
                        "center_xy": [cx, cy],
                        "color": color,
                        "strength": strength,
                        "role": "wall_repel",
                    }
                )
                continue

            is_rare = color in rare
            is_large = c.get("component_id") in large_ids
            if is_rare or is_large:
                strength = self.attract_base * area_scale
                if is_rare:
                    strength *= 1.25  # prefer rare non-modal colors
                attractors.append((cx, cy, strength))
                attractor_rows.append(
                    {
                        "component_id": c.get("component_id"),
                        "center_xy": [cx, cy],
                        "color": color,
                        "strength": strength,
                        "rare": is_rare,
                        "large": is_large,
                        "role": "attractor",
                    }
                )

        # Probe origin: context origin, else frame center
        ox = float(ctx.get("origin_xy", [w / 2.0, h / 2.0])[0])
        oy = float(ctx.get("origin_xy", [w / 2.0, h / 2.0])[1])
        ox = float(np.clip(ox, 0, w - 1))
        oy = float(np.clip(oy, 0, h - 1))

        dx, dy, pot0 = _gradient_delta(ox, oy, attractors, repellers, h, w)

        # Sample a few grid points for net potential summary
        sample_pts = [
            (ox, oy),
            (w * 0.25, h * 0.25),
            (w * 0.75, h * 0.25),
            (w * 0.25, h * 0.75),
            (w * 0.75, h * 0.75),
            (w * 0.5, h * 0.5),
        ]
        sample_pots = [
            {
                "xy": [float(px), float(py)],
                "potential": _potential_at(float(px), float(py), attractors, repellers, h, w),
            }
            for px, py in sample_pts
        ]
        mean_pot = float(np.mean([s["potential"] for s in sample_pots])) if sample_pots else 0.0
        max_pot = float(max((s["potential"] for s in sample_pots), default=0.0))
        min_pot = float(min((s["potential"] for s in sample_pots), default=0.0))

        # Cardinal move suggestions (unit steps)
        moves = []
        for name, mdx, mdy in (
            ("right", 1.0, 0.0),
            ("left", -1.0, 0.0),
            ("down", 0.0, 1.0),
            ("up", 0.0, -1.0),
        ):
            p = _potential_at(ox + mdx, oy + mdy, attractors, repellers, h, w)
            moves.append({"dir": name, "delta_xy": [mdx, mdy], "potential": p, "d_pot": p - pot0})
        moves.sort(key=lambda r: -float(r["d_pot"]))

        proposals = [
            {
                "kind": "field_move",
                "delta_xy": [dx, dy],
                "origin_xy": [ox, oy],
                "potential": pot0,
                "best_cardinal": moves[0] if moves else None,
                "admitted": False,
                "hypothesis_only": True,
                "evidence_status": "field_hypothesis",
            }
        ]

        scores = {
            "net_potential_at_origin": pot0,
            "mean_sample_potential": mean_pot,
            "max_sample_potential": max_pot,
            "min_sample_potential": min_pot,
            "n_attractors": len(attractors),
            "n_repellers": len(repellers),
            "suggested_dx": dx,
            "suggested_dy": dy,
            "best_cardinal_d_pot": float(moves[0]["d_pot"]) if moves else 0.0,
        }

        summary = {
            "policy": "sum_signed_potentials_then_follow_positive_differential",
            "schema": "megamind.field_ops.signed_potential.v0",
            "attractors": attractor_rows[:16],
            "walls": wall_rows[:16],
            "rare_colors": sorted(rare),
            "sample_potentials": sample_pots,
            "cardinal_moves": moves,
            "suggested_move_delta": [dx, dy],
            "origin_xy": [ox, oy],
            "admitted": False,
            "hypothesis_only": True,
        }

        vectors = {
            "suggested_move_delta": [dx, dy],
            "origin_xy": [ox, oy],
        }
        return OperatorResult(
            name=self.name,
            status="ok",
            scores=scores,
            proposals=proposals,
            summary=summary,
            payload={
                "suggested_move_delta": [dx, dy],
                "origin_xy": [ox, oy],
                "attractors": attractor_rows[:16],
                "walls": wall_rows[:16],
                "cardinal_moves": moves,
            },
            vectors=vectors,
            meta={
                "schema": "megamind.field_ops.signed_potential.v0",
                "n_attractors": len(attractors),
                "n_repellers": len(repellers),
                "hypothesis_only": True,
            },
            messages=["signed_potential field ok"],
            truth_flags=TruthFlags(),
            hypothesis_only=True,
            hypothesis=True,
        )
