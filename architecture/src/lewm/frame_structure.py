"""Frame-structure adapter — unify three sources of geometric ground truth.

The MCTS planner needs to know *what is a wall, what is a floor, where is
the agent*. There are three sources of that knowledge in Loom, in order
of preference:

1. **`GameInstance` metadata (authoritative)** — `loom_layers.gamemaster_procedural`
   generates procedural games and EXPOSES `walls: frozenset[(y,x)]`,
   `player_start`, `goal_cell`, `key_cell`, `door_cell`, `collectibles`.
   For GameMaster-generated games this is ground truth; no inference
   needed. Use :func:`from_game_instance`.

2. **`wall_grid_map_pack` cached npy (frozen calibration)** —
   `experiments/2026-05-14_wall_grid_map_pack_001_v2/wall_grids_npy/`
   contains pre-computed 64×64 wall grids per still, produced by
   `scripts/build_wall_grid_from_wall_pass.py` from approved
   wall-pass masks via BOX downsample at threshold 0.28. Use
   :func:`from_wall_grid_npy` when a cached grid exists for the frame.

3. **Heuristic trajectory parse (last-resort, real ARC-AGI-3)** — for
   external `arc_agi.Arcade` games without GameMaster metadata or
   cached wall grids, fall back to multi-frame heuristics: walls = cells
   constant across the trajectory; player = the smallest moving sprite.
   Use :func:`parse_trajectory`. Per memory
   ``per_game_change_signatures``, ``player_bbox=None`` for puzzle games
   with no agent (cd82, ar25).

Sprite extraction throughout delegates to the canonical
:func:`lewm.perception.sprites.extract_sprites` — do not re-implement
connected components here.

Integrates with the MCTS planner via:
- ``action_validator_fn``: reject ACTION6(x, y) when ``wall_mask[y, x]``
- ``goal_inference_fn``: combine with :mod:`lewm.goal_candidates`
- PriorMixer (via :class:`lewm.informers.wall_aware.WallAwareInformer`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from lewm.perception.sprites import Sprite, extract_sprites

if TYPE_CHECKING:
    from loom_layers.gamemaster_procedural import GameInstance


ARC_GRID_SIZE = 64
ARC_NUM_COLORS = 16


@dataclass(frozen=True)
class BBox:
    """Axis-aligned bounding box, half-open: ``[y0, y1) x [x0, x1)``."""

    y0: int
    x0: int
    y1: int
    x1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.y0 + self.y1 - 1) / 2.0, (self.x0 + self.x1 - 1) / 2.0)

    def contains(self, y: int, x: int) -> bool:
        return self.y0 <= y < self.y1 and self.x0 <= x < self.x1


@dataclass
class SpriteRegion:
    """Adapter dataclass over :class:`lewm.perception.sprites.Sprite`.

    Sprite uses `bbox=(r_min, c_min, r_max, c_max)` inclusive and `size`;
    we expose half-open :class:`BBox` and `pixel_count` so downstream
    consumers (goal_candidates, wall_aware) get a uniform interface.

    Convert via :func:`from_sprite`.
    """

    color: int
    bbox: BBox
    pixel_count: int
    centroid: tuple[float, float]

    @classmethod
    def from_sprite(cls, sp: Sprite) -> "SpriteRegion":
        r0, c0, r1_incl, c1_incl = sp.bbox
        return cls(
            color=sp.color,
            bbox=BBox(y0=int(r0), x0=int(c0), y1=int(r1_incl) + 1, x1=int(c1_incl) + 1),
            pixel_count=int(sp.size),
            centroid=(float(sp.centroid[0]), float(sp.centroid[1])),
        )


@dataclass
class FrameStructure:
    """Geometric/semantic parse of a frame (single-frame fields) and the
    trajectory it came from (multi-frame fields: walls, player, etc).

    Single-frame: ``sprites``, ``background_color``, ``grid_extent``.
    Multi-frame: ``wall_mask``, ``floor_mask``, ``player_bbox``, ``player_size``.
    """

    # Single-frame fields
    background_color: int
    sprites: list[SpriteRegion]
    grid_extent: BBox

    # Multi-frame fields (None if parsed from a single frame; populated by
    # parse_trajectory)
    wall_mask: Optional[torch.Tensor] = None        # [64, 64] bool
    floor_mask: Optional[torch.Tensor] = None       # [64, 64] bool
    player_bbox: Optional[BBox] = None
    player_size: Optional[tuple[int, int]] = None   # (h, w)
    player_color: Optional[int] = None

    # Provenance (for cache invalidation & debug)
    n_frames_used: int = 0
    extras: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# Sprite extraction — delegates to canonical lewm.perception.sprites.
# ─────────────────────────────────────────────────────────────────────


def _extract_sprite_regions(
    frame: np.ndarray, *, background_color: int,
) -> list[SpriteRegion]:
    """Thin adapter: canonical extract_sprites → list[SpriteRegion]."""

    sprites = extract_sprites(
        frame, background_colors=(int(background_color),), connectivity=4, min_size=1,
    )
    return [SpriteRegion.from_sprite(sp) for sp in sprites]


def _detect_background_color(frame: np.ndarray) -> int:
    """Most-frequent color in the frame."""

    flat = np.asarray(frame, dtype=np.int64).flatten()
    counts = np.bincount(flat, minlength=ARC_NUM_COLORS)
    return int(np.argmax(counts))


def _grid_extent(frame: np.ndarray, background_color: int) -> BBox:
    """Bbox of all non-background pixels; falls back to full frame."""

    mask = np.asarray(frame) != background_color
    if not mask.any():
        return BBox(0, 0, ARC_GRID_SIZE, ARC_GRID_SIZE)
    ys, xs = np.where(mask)
    return BBox(int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)


def parse_frame(frame: np.ndarray) -> FrameStructure:
    """Single-frame parse: sprites + background + grid extent only.

    Multi-frame fields (walls, floors, player) are left as ``None``.
    Use :func:`parse_trajectory` (heuristic) or :func:`from_game_instance`
    (authoritative) to populate them.
    """

    frame = np.asarray(frame, dtype=np.int64)
    if frame.shape != (ARC_GRID_SIZE, ARC_GRID_SIZE):
        raise ValueError(f"expected (64, 64) frame, got {frame.shape}")
    bg = _detect_background_color(frame)
    extent = _grid_extent(frame, bg)
    sprites = _extract_sprite_regions(frame, background_color=bg)
    return FrameStructure(
        background_color=bg, sprites=sprites, grid_extent=extent,
    )


# ─────────────────────────────────────────────────────────────────────
# Multi-frame: walls, floors, player
# ─────────────────────────────────────────────────────────────────────


def _wall_floor_masks(
    frames: list[np.ndarray], background_color: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Walls = cells whose color is constant across the trajectory and
    not background. Floors = complement within the grid extent."""

    if not frames:
        empty = torch.zeros(ARC_GRID_SIZE, ARC_GRID_SIZE, dtype=torch.bool)
        return empty, empty.clone()
    stack = np.stack([np.asarray(f, dtype=np.int64) for f in frames])  # [N, 64, 64]
    first = stack[0]
    # constant_mask[y, x] = True if all frames have the same color at (y, x)
    constant_mask = (stack == first[None]).all(axis=0)
    # walls = constant AND not-background
    wall = constant_mask & (first != background_color)
    floor = ~wall
    return (
        torch.from_numpy(wall).bool(),
        torch.from_numpy(floor).bool(),
    )


def _detect_player(
    frames: list[np.ndarray], background_color: int,
) -> tuple[Optional[BBox], Optional[tuple[int, int]], Optional[int]]:
    """Most-moving sprite color across consecutive frames is the player.

    Heuristic fallback for external ARC-AGI-3 frames; GameMaster
    procedural games should use :func:`from_game_instance` instead,
    which has authoritative `player_start`.

    For each non-background color, compute the total centroid displacement
    of its largest sprite across consecutive frame pairs. The color with
    the highest cumulative displacement (above a small threshold) is the
    player candidate. Returns (bbox, size, color) from the last frame;
    (None, None, None) when no color moves enough (puzzle games).
    """

    if len(frames) < 2:
        return None, None, None

    movements: dict[int, float] = {}
    last_largest: dict[int, SpriteRegion] = {}
    prev_centroids: dict[int, tuple[float, float]] = {}

    for f in frames:
        arr = np.asarray(f, dtype=np.int64)
        comps = _extract_sprite_regions(arr, background_color=background_color)
        by_color: dict[int, SpriteRegion] = {}
        for sp in comps:
            cur = by_color.get(sp.color)
            if cur is None or sp.pixel_count > cur.pixel_count:
                by_color[sp.color] = sp
        for color, largest in by_color.items():
            last_largest[color] = largest
            cy, cx = largest.centroid
            if color in prev_centroids:
                pcy, pcx = prev_centroids[color]
                movements[color] = movements.get(color, 0.0) + float(np.hypot(cy - pcy, cx - pcx))
            prev_centroids[color] = (cy, cx)

    if not movements:
        return None, None, None
    movements = {c: d for c, d in movements.items() if d >= 5.0}
    if not movements:
        return None, None, None

    best_color = sorted(
        movements.items(),
        key=lambda kv: (-kv[1], last_largest[kv[0]].pixel_count),
    )[0][0]
    bbox = last_largest[best_color].bbox
    return bbox, (bbox.height, bbox.width), int(best_color)


def parse_trajectory(
    frames: list[np.ndarray],
    *,
    background_color: Optional[int] = None,
) -> FrameStructure:
    """Multi-frame parse: walls, floors, player, plus single-frame fields
    derived from the LAST frame in ``frames``.

    Caller provides a list of (64, 64) frames in temporal order. If
    ``background_color`` is omitted, it is inferred from the frame
    distribution across the trajectory.
    """

    if not frames:
        raise ValueError("frames must be non-empty")
    arr_stack = np.stack([np.asarray(f, dtype=np.int64) for f in frames])
    if background_color is None:
        flat = arr_stack.flatten()
        background_color = int(np.bincount(flat, minlength=ARC_NUM_COLORS).argmax())

    last = arr_stack[-1]
    extent = _grid_extent(last, background_color)
    sprites = _extract_sprite_regions(last, background_color=background_color)
    wall, floor = _wall_floor_masks(frames, background_color)
    player_bbox, player_size, player_color = _detect_player(frames, background_color)

    return FrameStructure(
        background_color=background_color,
        sprites=sprites,
        grid_extent=extent,
        wall_mask=wall,
        floor_mask=floor,
        player_bbox=player_bbox,
        player_size=player_size,
        player_color=player_color,
        n_frames_used=len(frames),
    )


def from_game_instance(
    instance: "GameInstance",
    *,
    pad_to_arc: bool = True,
) -> FrameStructure:
    """Authoritative parse for a GameMaster-generated game.

    Uses `instance.walls`, `instance.player_start`, `instance.goal_cell`
    directly — no inference needed. The returned structure includes a
    wall_mask derived from the frozen `walls` set and a player_bbox at
    the player_start cell. Goal location is stashed in `extras['goal_cell']`
    for downstream goal-candidate construction.

    `instance.glyph_grid` is used as the seed frame for sprite extraction;
    sprite identities then reflect the procedural game's color palette.
    """

    frame = np.asarray(instance.glyph_grid, dtype=np.int64)
    if frame.shape != (ARC_GRID_SIZE, ARC_GRID_SIZE):
        if pad_to_arc:
            from loom_layers.staged_player.engine import pad_to_arc_frame
            frame = np.asarray(pad_to_arc_frame(frame), dtype=np.int64)
        else:
            raise ValueError(
                f"glyph_grid shape {frame.shape} != ({ARC_GRID_SIZE}, {ARC_GRID_SIZE}); "
                f"pass pad_to_arc=True to use the staged_player ARC pad helper"
            )

    bg = _detect_background_color(frame)
    extent = _grid_extent(frame, bg)
    sprites = _extract_sprite_regions(frame, background_color=bg)

    wall = torch.zeros(ARC_GRID_SIZE, ARC_GRID_SIZE, dtype=torch.bool)
    for (wy, wx) in instance.walls:
        if 0 <= wy < ARC_GRID_SIZE and 0 <= wx < ARC_GRID_SIZE:
            wall[wy, wx] = True
    floor = ~wall

    py, px = instance.player_start
    player_bbox = BBox(y0=int(py), x0=int(px), y1=int(py) + 1, x1=int(px) + 1)

    extras = {
        "goal_cell": tuple(int(v) for v in instance.goal_cell),
        "source": "GameInstance",
        "schema_id": instance.schema.schema_id,
        "seed": int(instance.seed),
    }
    if instance.key_cell is not None:
        extras["key_cell"] = tuple(int(v) for v in instance.key_cell)
    if instance.door_cell is not None:
        extras["door_cell"] = tuple(int(v) for v in instance.door_cell)
    if instance.collectibles:
        extras["collectibles"] = [tuple(int(v) for v in c) for c in instance.collectibles]

    return FrameStructure(
        background_color=bg,
        sprites=sprites,
        grid_extent=extent,
        wall_mask=wall,
        floor_mask=floor,
        player_bbox=player_bbox,
        player_size=(1, 1),
        player_color=None,  # GameInstance doesn't expose the player palette index
        n_frames_used=1,
        extras=extras,
    )


def from_wall_grid_npy(
    frame: np.ndarray,
    wall_grid_path: str | Path,
) -> FrameStructure:
    """Single-frame parse augmented with a pre-computed wall grid from
    `experiments/2026-05-14_wall_grid_map_pack_001_v2/wall_grids_npy/`.

    The cached npy is the canonical 64×64 wall mask projection produced
    by `scripts/build_wall_grid_from_wall_pass.py` from approved
    wall-pass masks. Use this when GameMaster ground truth isn't
    available but a frozen wall grid for this still exists.
    """

    base = parse_frame(frame)
    grid = np.load(Path(wall_grid_path))
    if grid.shape != (ARC_GRID_SIZE, ARC_GRID_SIZE):
        raise ValueError(
            f"wall_grid_npy at {wall_grid_path} has shape {grid.shape}; "
            f"expected ({ARC_GRID_SIZE}, {ARC_GRID_SIZE})"
        )
    wall = torch.from_numpy(grid.astype(bool))
    base.wall_mask = wall
    base.floor_mask = ~wall
    base.extras = {"source": "wall_grid_map_pack_npy", "path": str(wall_grid_path)}
    base.n_frames_used = 1
    return base


__all__ = [
    "BBox", "SpriteRegion", "FrameStructure",
    "parse_frame", "parse_trajectory",
    "from_game_instance", "from_wall_grid_npy",
    "ARC_GRID_SIZE", "ARC_NUM_COLORS",
]
