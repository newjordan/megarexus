#!/usr/bin/env python3
"""Tiny WP-A smoke: BodyOp + RayOp on a synthetic 64x64 board.

Run:
  python3 -B \\
    synthesis_lab/orchestration/rdl/brain/operators/smoke_physics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synthesis_lab.orchestration.rdl.brain.types import KnownMapView, TruthFlags
from synthesis_lab.orchestration.rdl.brain.operators.body import BodyOp
from synthesis_lab.orchestration.rdl.brain.operators.ray_ops import RayOp


def _synthetic_board(n: int = 64) -> KnownMapView:
    """Open field with walls, floor strip, spout, U-mouth target."""
    solid = np.zeros((n, n), dtype=bool)
    solid[:, 0] = True
    solid[:, -1] = True
    floor_rows = (n - 2, n - 1)
    for r in floor_rows:
        solid[r, :] = True
    solid[n // 2, n // 4 : 3 * n // 4] = True
    spout = (2, n // 2)
    mouth_y, mouth_x = n // 2 - 1, n // 2
    solid[mouth_y, mouth_x] = False
    mouths = (((mouth_y, mouth_x),),)
    frame = np.full((n, n), 12, dtype=np.int8)
    frame[solid] = 8
    for r in floor_rows:
        frame[r, :] = 1
    frame[spout] = 6
    frame[max(spout[0] - 1, 0), spout[1]] = 4
    frame[mouth_y, mouth_x] = 12
    frame[mouth_y, mouth_x - 1] = 11
    frame[mouth_y, mouth_x + 1] = 11
    frame[mouth_y + 1, mouth_x] = 11
    solid[mouth_y, mouth_x - 1] = True
    solid[mouth_y, mouth_x + 1] = True
    solid[mouth_y + 1, mouth_x] = True

    return KnownMapView(
        solid=solid,
        solid_mask=solid,
        spouts=(spout,),
        mouths=mouths,
        floor_rows=floor_rows,
        glyph_cells=(((mouth_y, mouth_x - 1), (mouth_y, mouth_x + 1), (mouth_y + 1, mouth_x)),),
        capacities=(1.0,),
        emit_volume=8.0,
        frame=frame,
        trusted=False,
        status="degraded",
        meta={"source": "synthetic_64"},
        provenance={"source": "synthetic_64", "trusted": False},
    )


def main() -> int:
    mv = _synthetic_board(64)
    body = BodyOp(lane="stream")
    ray = RayOp()

    b = body.run(mv, {})
    r = ray.run(mv, {})

    print(f"body status={b.status} scores={b.scores} backend={b.summary.get('backend')}")
    print(f"ray  status={r.status} scores={r.scores} backend={r.summary.get('backend')}")
    print(f"body hypothesis_only={b.hypothesis_only} truth={b.truth_flags.as_dict()}")
    print(f"ray  hypothesis_only={r.hypothesis_only} truth={r.truth_flags.as_dict()}")

    ok = b.status in ("ok", "degraded") and r.status in ("ok", "degraded")
    ok = ok and "margin" in b.scores and "n_hits" in r.scores
    ok = ok and b.hypothesis_only and r.hypothesis_only
    ok = ok and not b.truth_flags.any_admitted and not r.truth_flags.any_admitted
    # default TruthFlags skeleton sanity
    tf = TruthFlags()
    ok = ok and not any(tf.as_dict().values())
    print("PASS" if ok else "FAIL")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
