#!/usr/bin/env python3
"""Coverage audit: creativity must appear on the live MegaMind path.

Exit 0 only if:
  - real SP80 observe trusted
  - body/ray/field/conceptual/portband/macros all ok
  - train_transition auto-fuses nonzero extra
  - world model plug ok (or explicit degraded with message)
  - ada_online phase present
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    from synthesis_lab.orchestration.rdl.brain.collect_real import collect_transitions
    from synthesis_lab.orchestration.rdl.brain.megamind import MegaMind

    out = collect_transitions(n_steps=6, seed=99)
    assert out.get("status") == "ok", out
    frames, actions, fn = out["frames"], out["actions"], out["frames_next"]

    mm = MegaMind(device="cpu")
    st = mm.observe(frames[0].numpy())
    print("trusted", st.trusted)
    ds = st.dream_summary or {}
    required = (
        "body",
        "ray",
        "field",
        "conceptual",
        "portband",
        "macros",
        "legal392",
        "symmetry",
        "geo",
        "saliency",
    )
    fail = []
    for k in required:
        pkt = ds.get(k) or {}
        status = pkt.get("status") if isinstance(pkt, dict) else None
        print(f"  {k}: status={status} scores={list((pkt.get('scores') or {}) if isinstance(pkt, dict) else [])[:6]}")
        if status not in ("ok",):
            fail.append(f"{k}:{status}")

    if not st.trusted:
        fail.append("not_trusted")

    # extra feats
    extra = st.extra_feats
    if extra is None:
        fail.append("no_extra_feats")
    else:
        nz = float((extra.abs() > 0).float().mean())
        print("extra_feats nonzero_frac", nz, "dim", tuple(extra.shape))
        if nz < 0.1:
            fail.append(f"extra_too_sparse:{nz}")

    # judge + train
    jp = mm.judge_plan(frames[0].numpy(), int(actions[0].item()))
    print("wm", (jp.world_summary or {}).get("wm_backend") or (jp.world_summary or {}).get("wm_plug"))
    print("judge", jp.judge_I, "disagree", jp.disagreement)

    logs = mm.train_transition(frames[:4], actions[:4], fn[:4], game_id=out.get("game_id"))
    phases = [x.get("phase") for x in logs]
    print("phases", phases)
    if "ada_online" not in phases:
        fail.append("missing_ada_online")
    cov = next((x for x in logs if x.get("phase") == "coverage"), {})
    print("coverage", cov)
    if not cov.get("auto_extra"):
        fail.append("auto_extra_false")
    if float(cov.get("extra_nonzero") or 0) < 0.05:
        fail.append("train_extra_sparse")

    wm = cov.get("wm") or {}
    if wm.get("status") != "ok":
        # soft warn but fail audit — creativity gap #2
        fail.append(f"wm_not_ok:{wm}")

    if fail:
        print("FAIL", fail)
        return 1
    print("CREATIVITY COVERAGE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
