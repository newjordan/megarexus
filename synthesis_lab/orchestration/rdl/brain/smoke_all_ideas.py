#!/usr/bin/env python3
"""Base model must contain ALL idea seeds. No silent omissions.

Checks:
  - every OP_FAMILIES channel present in dream_summary
  - every LOOM_IDEA_IDS present
  - every REXUS_IDEA_IDS present
  - every PRIMITIVE_IDEA_IDS present (basic game design atoms)
  - every PATTERN_GROWTH_IDEA_IDS present (multi-variate geometry)
  - every WECO_RSI_IDEA_IDS present (AIDE² bi-level RSI)
  - EXTRA_FEAT_DIM matches pad
  - train auto-fuse + ada + full contra still green
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _check_catalog(name: str, ids, ds: dict, fail: list) -> tuple[int, int]:
    missing = [f for f in ids if f not in ds]
    if missing:
        fail.append(f"missing_{name}:{missing[:12]}{'...' if len(missing) > 12 else ''}")
        print(f"MISSING {name.upper()}", missing)

    ok = deg = 0
    for cid in ids:
        pkt = ds.get(cid) or {}
        status = pkt.get("status")
        if status == "ok":
            ok += 1
        else:
            deg += 1
            print(f"  DEGRADED {cid}: {status} {pkt.get('messages')}")
    print(f"{name} ideas ok={ok} degraded={deg} total={len(ids)}")

    for cid in ids:
        pkt = ds.get(cid) or {}
        if cid not in ds:
            fail.append(f"absent:{cid}")
        elif pkt.get("status") not in ("ok", "degraded"):
            fail.append(f"bad_status:{cid}:{pkt.get('status')}")
        elif pkt.get("status") == "degraded" and not (pkt.get("scores") or pkt.get("messages")):
            fail.append(f"empty_seed:{cid}")

    # Prefer all ok; fail hard if more than 2 degraded
    if ok < len(ids) - 2:
        fail.append(f"too_many_degraded_{name}:{deg}")
    return ok, deg


def main() -> int:
    from synthesis_lab.orchestration.rdl.brain.collect_real import collect_transitions
    from synthesis_lab.orchestration.rdl.brain.hemisphere_i import EXTRA_FEAT_DIM, OP_FAMILIES
    from synthesis_lab.orchestration.rdl.brain.megamind import MegaMind
    from synthesis_lab.orchestration.rdl.brain.operators.loom_imagination import LOOM_IDEA_IDS
    from synthesis_lab.orchestration.rdl.brain.operators.pattern_growth import (
        PATTERN_GROWTH_IDEA_IDS,
    )
    from synthesis_lab.orchestration.rdl.brain.operators.primitive_game_designs import (
        PRIMITIVE_IDEA_IDS,
    )
    from synthesis_lab.orchestration.rdl.brain.operators.rexus_concepts import REXUS_IDEA_IDS
    from synthesis_lab.orchestration.rdl.brain.operators.weco_rsi import WECO_RSI_IDEA_IDS

    out = collect_transitions(n_steps=8, seed=3)  # multi-variate pattern growth default
    mm = MegaMind(device="cpu")
    mm.reset_episode(game_id=out.get("game_id") or "pattern-growth-mv")
    frames, actions, fn = out["frames"], out["actions"], out["frames_next"]
    for i in range(min(4, len(frames))):
        mm.I.remember_transition(frames[i].numpy(), int(actions[i].item()), fn[i].numpy())
    _ = mm.observe(frames[2].numpy())
    st = mm.observe(frames[min(4, len(frames) - 1)].numpy())
    ds = st.dream_summary or {}

    fail = []
    print("n_OP_FAMILIES", len(OP_FAMILIES), "EXTRA_FEAT_DIM", EXTRA_FEAT_DIM)
    print(
        "n_LOOM_IDEAS",
        len(LOOM_IDEA_IDS),
        "n_REXUS_IDEAS",
        len(REXUS_IDEA_IDS),
        "n_PRIMITIVE_IDEAS",
        len(PRIMITIVE_IDEA_IDS),
        "n_PATTERN_GROWTH",
        len(PATTERN_GROWTH_IDEA_IDS),
        "n_WECO_RSI",
        len(WECO_RSI_IDEA_IDS),
    )
    print("source", out.get("source"), "note", out.get("note"))
    print("extra shape", None if st.extra_feats is None else tuple(st.extra_feats.shape))

    missing_fam = [f for f in OP_FAMILIES if f not in ds]
    if missing_fam:
        fail.append(f"missing_families:{missing_fam[:8]}...")
        print("MISSING FAMILIES", missing_fam)

    _check_catalog("Loom", LOOM_IDEA_IDS, ds, fail)
    _check_catalog("Rexus", REXUS_IDEA_IDS, ds, fail)
    _check_catalog("Primitive", PRIMITIVE_IDEA_IDS, ds, fail)
    _check_catalog("PatternGrowth", PATTERN_GROWTH_IDEA_IDS, ds, fail)
    _check_catalog("WecoRSI", WECO_RSI_IDEA_IDS, ds, fail)

    if st.extra_feats is None or int(st.extra_feats.shape[-1]) != EXTRA_FEAT_DIM:
        fail.append(
            f"pad_mismatch:{None if st.extra_feats is None else st.extra_feats.shape[-1]}!={EXTRA_FEAT_DIM}"
        )

    logs = mm.train_transition(frames[:3], actions[:3], fn[:3], game_id=out.get("game_id"))
    cov = next((x for x in logs if x.get("phase") == "coverage"), {})
    print("coverage n_ops_ok", cov.get("n_ops_ok"), "/", cov.get("n_ops"))
    min_ok = max(50, len(OP_FAMILIES) - 25)
    if int(cov.get("n_ops_ok") or 0) < min_ok:
        fail.append(f"ops_ok_low:{cov.get('n_ops_ok')}<{min_ok}")

    if fail:
        print("ALL IDEAS FAIL", fail)
        return 1
    print(
        "ALL IDEAS IN BASE MODEL — garden complete "
        "(Loom + Rexus + Primitives + PatternGrowth + WecoRSI)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
