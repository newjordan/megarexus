# Dynamic multi-variate pattern growth (geometry expansion)

**Date:** 2026-07-15  
**Doctrine:** Geometry is **not** SP80/pour-bound. Expand all geometric
concepts so the base pad measures **dynamic multi-variate pattern growth** on
arbitrary HxW discrete fields. Training raises useful seeds; unused stay
near-zero. No soft omissions.

## Framing

| Old (narrow) | New (garden) |
|---|---|
| pour mouths / spouts / floors | multi-color variables on any board |
| single-game win geometry | growth / merge / trail / lattice / CA affinity |
| fixed 64×64 SP80 train focus | frame-size-agnostic channels (G39) |
| body = imagine_pour only | pattern growth catalog + expanded geo/field |

## Catalog G1–G40

Planted in `synthesis_lab/orchestration/rdl/brain/operators/pattern_growth.py`  
Wired into `OP_FAMILIES` → 16-D slots → central LoRA pad.

| ID | Axis |
|---|---|
| G1 | multi-channel color variables |
| G2 | density field (local/global) |
| G3 | connectivity / components |
| G4 | expansion frontier |
| G5 | accretion vs erosion (Δ) |
| G6 | branching topology |
| G7 | merge / split dynamics |
| G8 | multi-scale pyramid |
| G9 | anisotropy / orientation |
| G10 | periodicity / lattice |
| G11 | multi-axis symmetry |
| G12 | multi-variate potential field |
| G13 | reaction–diffusion interface |
| G14 | cellular rewrite affinity |
| G15 | trail growth / persistence |
| G16 | packing / compaction |
| G17 | void cavities |
| G18 | topological holes (genus proxy) |
| G19 | aspect-ratio spectrum |
| G20 | color adjacency graph |
| G21 | local rewrite rule diversity |
| G22 | joint multi-variate state vector |
| G23 | growth rate (Δ fill) |
| G24 | front curvature / roughness |
| G25 | multi-blob competition |
| G26 | hierarchy / nesting |
| G27 | lattice vs organic |
| G28 | mass conservation (per color) |
| G29 | irreversible growth |
| G30 | pattern phase (early/mid/late) |
| G31 | variable coupling |
| G32 | spatial frequency |
| G33 | geodesic / pairwise distance field |
| G34 | support scaffold |
| G35 | transform invariance probe |
| G36 | generative grammar fit |
| G37 | multi-objective conflict |
| G38 | **not SP80-bound** (doctrine) |
| G39 | frame-size agnostic |
| G40 | catalog coverage meta |

## Also expanded

- `GeoOp` — multi-color / aspect multivariate scores + `not_sp80_bound`
- `FieldOp` docstring — multi-variate pattern growth (any HxW)
- `collect_real` default — growing multi-color pattern transitions (not Arcade SP80)

## Smoke

```bash
CUDA_VISIBLE_DEVICES= python3 -B \
  synthesis_lab/orchestration/rdl/brain/smoke_all_ideas.py
```

Expect: PatternGrowth **40/40 ok**, pad = `OP_SLOT * len(OP_FAMILIES)`.
