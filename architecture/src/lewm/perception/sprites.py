"""Color-connected-components sprite extraction for ARC-AGI-3 frames.

ARC-AGI-3 frames are exactly ``64×64`` with cell values in ``[0, 15]`` (16
colors). Each visually distinct sprite — agent, switch, key, target, color
swatch — is by construction a contiguous region of one palette index. So
the cleanest, dependency-free perception primitive is **per-color
connected-component labeling**:

- group adjacent cells of the same color into a connected region,
- discard any color treated as background (default ``{0}``),
- emit one :class:`Sprite` per region with bbox, centroid, color, and size.

This is the v1 (object detection) layer of the iterative-lab plan in
``docs/iterative_world_model_lab.md``. It feeds downstream goal-inference
layers by exposing stable sprite candidates that can be scored or elevated
as ``z_goal`` outside the core perception primitive.

Implementation: pure-numpy iterative BFS — no scipy. ARC frames are tiny
(``≤ 4096`` cells), so a Python BFS over labeled visited bits is microseconds
and avoids the dep.

The :func:`filter_sprites` function provides the heuristic priors the doc
mentioned for HUD strip / color-key swatch suppression: exclude
full-width or full-height bands, exclude tiny noise, exclude sprites of
extreme aspect ratio. None of these are mandatory; they're configurable
post-extraction filters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Sprite dataclass                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class Sprite:
    """One connected sprite candidate.

    Attributes:

    - ``color``: int in ``[0, 15]`` — palette index.
    - ``cells``: list of ``(row, col)`` cells in the sprite, in extraction
      order (BFS).
    - ``bbox``: ``(row_min, col_min, row_max, col_max)`` inclusive.
    - ``centroid``: ``(row_mean, col_mean)`` floating point.
    - ``size``: number of cells (``len(cells)``).

    Derived properties:

    - ``height``, ``width``: bbox extents.
    - ``aspect_ratio``: ``width / height``.
    - ``density``: ``size / (height * width)`` — 1.0 means the bbox is fully
      filled by sprite cells, lower means there are holes.
    - ``mask``: ``(height, width)`` bool array within the bbox; True at sprite
      cells.
    """

    color: int
    cells: list[tuple[int, int]] = field(repr=False)
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    size: int

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def aspect_ratio(self) -> float:
        return self.width / max(self.height, 1)

    @property
    def density(self) -> float:
        return self.size / max(self.height * self.width, 1)

    def mask(self) -> np.ndarray:
        """Bool ``(height, width)`` array within the bbox; True at sprite cells."""

        m = np.zeros((self.height, self.width), dtype=bool)
        r0, c0, _, _ = self.bbox
        for r, c in self.cells:
            m[r - r0, c - c0] = True
        return m


# --------------------------------------------------------------------------- #
# extraction                                                                  #
# --------------------------------------------------------------------------- #


_NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGHBORS_8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def _validate_frame(frame: np.ndarray) -> None:
    if frame.ndim != 2:
        raise ValueError(f"frame must be 2D, got {frame.ndim}D")
    if frame.size == 0:
        raise ValueError("frame is empty")


def extract_sprites(
    frame: np.ndarray,
    *,
    background_colors: tuple[int, ...] = (0,),
    connectivity: int = 4,
    min_size: int = 1,
) -> list[Sprite]:
    """Extract sprite candidates from a 2D color-index frame.

    A sprite is a connected component of cells with the same color value.
    Cells with color in ``background_colors`` are excluded entirely.

    Args:
        frame: ``(H, W)`` integer array, values are color indices.
        background_colors: tuple of color indices to ignore. Default is
            ``(0,)`` matching ARC-AGI-3's typical bg-as-color-0 convention,
            but several public games use color 5 as the field background;
            override per game when needed.
        connectivity: ``4`` (orthogonal only, default) or ``8`` (diagonal too).
        min_size: minimum cell count for a sprite to be returned.

    Returns:
        List of :class:`Sprite`, **sorted by descending size** (then by
        smallest top-left bbox corner for stable ties).
    """

    _validate_frame(frame)
    if connectivity == 4:
        neighbors = _NEIGHBORS_4
    elif connectivity == 8:
        neighbors = _NEIGHBORS_8
    else:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size}")

    h, w = frame.shape
    visited = np.zeros((h, w), dtype=bool)
    bg_set = frozenset(int(c) for c in background_colors)
    sprites: list[Sprite] = []

    # BFS with explicit stack; deterministic scan order (top-left → bottom-right).
    for r in range(h):
        for c in range(w):
            if visited[r, c]:
                continue
            color = int(frame[r, c])
            if color in bg_set:
                visited[r, c] = True
                continue
            stack = [(r, c)]
            cells: list[tuple[int, int]] = []
            r_min = r_max = r
            c_min = c_max = c
            r_sum = 0
            c_sum = 0
            while stack:
                cr, cc = stack.pop()
                if visited[cr, cc]:
                    continue
                if int(frame[cr, cc]) != color:
                    continue
                visited[cr, cc] = True
                cells.append((cr, cc))
                r_sum += cr
                c_sum += cc
                if cr < r_min:
                    r_min = cr
                if cr > r_max:
                    r_max = cr
                if cc < c_min:
                    c_min = cc
                if cc > c_max:
                    c_max = cc
                for dr, dc in neighbors:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        if int(frame[nr, nc]) == color:
                            stack.append((nr, nc))
            if len(cells) < min_size:
                continue
            n = len(cells)
            sprites.append(Sprite(
                color=color,
                cells=cells,
                bbox=(r_min, c_min, r_max, c_max),
                centroid=(r_sum / n, c_sum / n),
                size=n,
            ))

    sprites.sort(key=lambda s: (-s.size, s.bbox[0], s.bbox[1]))
    return sprites


# --------------------------------------------------------------------------- #
# filtering                                                                   #
# --------------------------------------------------------------------------- #


def filter_sprites(
    sprites: list[Sprite],
    *,
    min_size: int | None = None,
    max_size: int | None = None,
    min_aspect_ratio: float | None = None,
    max_aspect_ratio: float | None = None,
    min_density: float | None = None,
    max_density: float | None = None,
    exclude_full_width: bool = False,
    exclude_full_height: bool = False,
    exclude_colors: tuple[int, ...] = (),
    keep_colors: tuple[int, ...] | None = None,
    frame_shape: tuple[int, int] | None = None,
    full_dim_tolerance: int = 1,
) -> list[Sprite]:
    """Heuristic post-filter for sprite candidates.

    All criteria are independently applied; a sprite is kept iff it passes
    every active criterion.

    HUD-strip suppression — set ``exclude_full_width=True`` (and pass
    ``frame_shape``) to drop sprites whose width is within
    ``full_dim_tolerance`` of the frame width. The Loom_view analysis flagged
    HUD strips as a common ~10/25-game cross-game pattern that would
    contaminate any "interactable sprite candidate" list if not filtered.

    Args:
        sprites: list from :func:`extract_sprites`.
        min_size, max_size: cell-count bounds (inclusive).
        min_aspect_ratio, max_aspect_ratio: ``width / height`` bounds.
        min_density, max_density: ``size / (height * width)`` bounds in
            ``[0, 1]``.
        exclude_full_width: drop sprites whose width is ≥
            ``frame_shape[1] - full_dim_tolerance``. Requires ``frame_shape``.
        exclude_full_height: same for height.
        exclude_colors: drop sprites whose color is in this set.
        keep_colors: if set, ONLY keep sprites whose color is in this set.
        frame_shape: ``(H, W)`` of the source frame; required for
            ``exclude_full_*`` criteria.
        full_dim_tolerance: how close to full-width/height counts as
            full-spanning (default 1; e.g. 63 of 64 still counts as full).
    """

    if (exclude_full_width or exclude_full_height) and frame_shape is None:
        raise ValueError(
            "exclude_full_{width,height} require frame_shape to be set"
        )

    keep_set = None if keep_colors is None else frozenset(int(c) for c in keep_colors)
    excl_set = frozenset(int(c) for c in exclude_colors)

    out: list[Sprite] = []
    for s in sprites:
        if min_size is not None and s.size < min_size:
            continue
        if max_size is not None and s.size > max_size:
            continue
        if min_aspect_ratio is not None and s.aspect_ratio < min_aspect_ratio:
            continue
        if max_aspect_ratio is not None and s.aspect_ratio > max_aspect_ratio:
            continue
        if min_density is not None and s.density < min_density:
            continue
        if max_density is not None and s.density > max_density:
            continue
        if exclude_full_width and frame_shape is not None:
            if s.width >= frame_shape[1] - full_dim_tolerance:
                continue
        if exclude_full_height and frame_shape is not None:
            if s.height >= frame_shape[0] - full_dim_tolerance:
                continue
        if s.color in excl_set:
            continue
        if keep_set is not None and s.color not in keep_set:
            continue
        out.append(s)
    return out


# --------------------------------------------------------------------------- #
# diagnostics / visualization                                                 #
# --------------------------------------------------------------------------- #


def render_sprite_overlay(
    frame: np.ndarray,
    sprites: list[Sprite],
    *,
    overlay_color: int = 5,
) -> np.ndarray:
    """Return a copy of ``frame`` with each sprite's bbox drawn in
    ``overlay_color``. Cheap visual sanity check for the extraction.

    Edges are 1-cell thick. Cells already at ``overlay_color`` are preserved
    on the bbox border (visible against any other color).
    """

    if frame.ndim != 2:
        raise ValueError(f"frame must be 2D, got {frame.ndim}D")
    out = frame.copy()
    h, w = out.shape
    for s in sprites:
        r0, c0, r1, c1 = s.bbox
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(h - 1, r1), min(w - 1, c1)
        out[r0, c0:c1 + 1] = overlay_color
        out[r1, c0:c1 + 1] = overlay_color
        out[r0:r1 + 1, c0] = overlay_color
        out[r0:r1 + 1, c1] = overlay_color
    return out


def sprite_stats(sprites: list[Sprite]) -> dict:
    """Quick distributional summary of a sprite list — useful in eval logs."""

    if not sprites:
        return {
            "n_sprites": 0,
            "n_unique_colors": 0,
            "size_min": 0,
            "size_max": 0,
            "size_mean": 0.0,
            "size_p50": 0.0,
        }
    sizes = sorted(s.size for s in sprites)
    return {
        "n_sprites": len(sprites),
        "n_unique_colors": len({s.color for s in sprites}),
        "size_min": sizes[0],
        "size_max": sizes[-1],
        "size_mean": sum(sizes) / len(sizes),
        "size_p50": sizes[len(sizes) // 2],
    }
