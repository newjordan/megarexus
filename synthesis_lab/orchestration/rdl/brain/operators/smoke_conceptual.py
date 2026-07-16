#!/usr/bin/env python3
"""Smoke: map_trust + FieldOp + ConceptualOp on a synthetic frame.

Run:
  /home/frosty40/miniconda3/bin/python3 -B \\
    synthesis_lab/orchestration/rdl/brain/operators/smoke_conceptual.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from synthesis_lab.orchestration.rdl.brain.map_trust import frame_to_map  # noqa: E402
from synthesis_lab.orchestration.rdl.brain.operators.conceptual import ConceptualOp  # noqa: E402
from synthesis_lab.orchestration.rdl.brain.operators.field_ops import FieldOp  # noqa: E402


def synthetic_frame(seed: int = 0) -> np.ndarray:
    """64x64 color-index frame with bg + blobs (no game source)."""
    rng = np.random.default_rng(seed)
    g = np.full((64, 64), 12, dtype=np.int16)  # modal bg
    # border / wall strip
    g[0, :] = 14
    g[:, 0] = 14
    g[63, :] = 14
    g[:, 63] = 14
    # large component
    g[20:32, 18:30] = 8
    # rare color blobs
    g[40:46, 40:48] = 3
    g[10:14, 45:50] = 5
    # medium blob
    g[48:54, 10:22] = 11
    # tiny noise (should not become components if min_area=3 — make 3x3)
    g[2:5, 2:5] = 6
    # sprinkle
    g[55:58, 55:58] = int(rng.integers(1, 4))
    return g


def main() -> int:
    frame = synthetic_frame(1)
    print(f"frame_shape={frame.shape} unique={sorted(int(x) for x in np.unique(frame))}")

    mv = frame_to_map(frame)
    print(
        f"map_trust: trusted={mv.trusted} status={mv.status} "
        f"n_components={len(mv.components)} bg={mv.background} "
        f"failures={mv.gate_failures} "
        f"comp_src={mv.provenance.get('component_source')}"
    )
    for c in mv.components[:8]:
        print(
            f"  comp {c.get('component_id')} color={c.get('color')} "
            f"area={c.get('area')} center={c.get('center_xy')}"
        )

    if len(mv.components) < 1:
        print("FAIL: expected components from synthetic frame")
        return 1

    # Trust must be fail-closed on non-SP80 synthetic (no source/target geometry)
    if mv.trusted:
        print("WARN: synthetic frame unexpectedly trusted (ok if SP80-like)")
    else:
        print("map_trust: fail-closed OK (trusted=False)")

    field = FieldOp().run(mv, context={"origin_xy": [32.0, 32.0]})
    pot = field.scores.get("net_potential_at_origin")
    pot_s = f"{float(pot):.6f}" if isinstance(pot, (int, float)) else str(pot)
    print(
        f"field: status={field.status} n_attr={field.scores.get('n_attractors')} "
        f"n_rep={field.scores.get('n_repellers')} "
        f"delta={field.summary.get('suggested_move_delta')} "
        f"pot={pot_s}"
    )
    if field.truth_flags.solve_admitted or field.truth_flags.goal_admitted:
        print("FAIL: field truth flags must be false")
        return 1
    if "suggested_move_delta" not in field.summary:
        print("FAIL: field missing suggested_move_delta")
        return 1
    if field.status not in ("ok", "degraded"):
        print(f"FAIL: bad field status {field.status}")
        return 1

    conceptual = ConceptualOp().run(mv)
    print(
        f"conceptual: status={conceptual.status} "
        f"n_ports={conceptual.scores.get('n_ports')} "
        f"n_align={conceptual.scores.get('n_alignments')} "
        f"best_align={conceptual.scores.get('best_alignment_score')} "
        f"sym={conceptual.scores.get('symmetry_score')} "
        f"n_proposals={len(conceptual.proposals)} "
        f"ports_src={conceptual.summary.get('ports_source')}"
    )
    if not conceptual.hypothesis_only:
        print("FAIL: conceptual must be hypothesis_only")
        return 1
    tf = conceptual.truth_flags.as_dict()
    if any(tf.values()):
        print(f"FAIL: conceptual truth_flags not all false: {tf}")
        return 1
    if any(p.get("admitted") for p in conceptual.proposals):
        print("FAIL: a conceptual proposal admitted=True")
        return 1
    if not any(p.get("kind") == "component_centroid" for p in conceptual.proposals):
        print("FAIL: expected centroid proposals")
        return 1
    if conceptual.status not in ("ok", "degraded"):
        print(f"FAIL: bad conceptual status {conceptual.status}")
        return 1

    print("SMOKE_CONCEPTUAL PASS")
    return 0


if __name__ == "__main__":
    code = main()
    try:
        out = Path(__file__).resolve().parent / "_smoke_conceptual_exit.txt"
        out.write_text(f"{code}\n", encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(code)
