#!/usr/bin/env python3
"""Thick-load smoke: many seeds on the path; count how many grow (status=ok)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


EXPECTED = [
    "body",
    "ray",
    "field",
    "conceptual",
    "portband",
    "macros",
    "symmetry",
    "geo",
    "saliency",
    "affordances",
    "statecon",
    "delta",
    "ghost",
    "crucible",
    "explore",
    "legal392",
    "perception_trust",
]


def main() -> int:
    from synthesis_lab.orchestration.rdl.brain.collect_real import collect_transitions
    from synthesis_lab.orchestration.rdl.brain.megamind import MegaMind

    out = collect_transitions(n_steps=8, seed=7)
    mm = MegaMind(device="cpu")
    mm.reset_episode(game_id=out.get("game_id") or "sp80")

    # warm episode memory so ghost/delta/explore can fire
    frames, actions, fn = out["frames"], out["actions"], out["frames_next"]
    for i in range(min(3, len(frames))):
        mm.I.remember_transition(
            frames[i].numpy(), int(actions[i].item()), fn[i].numpy()
        )

    st = mm.observe(frames[3].numpy() if len(frames) > 3 else frames[-1].numpy())
    ds = st.dream_summary or {}
    print("trusted", st.trusted, "extra", None if st.extra_feats is None else tuple(st.extra_feats.shape))
    ok = []
    deg = []
    for name in EXPECTED:
        pkt = ds.get(name) or {}
        status = pkt.get("status") if isinstance(pkt, dict) else "missing"
        scores = list((pkt.get("scores") or {}).keys())[:5] if isinstance(pkt, dict) else []
        print(f"  {name:18s} {status:10s} scores={scores}")
        if status == "ok":
            ok.append(name)
        else:
            deg.append(f"{name}:{status}")

    logs = mm.train_transition(frames[:4], actions[:4], fn[:4], game_id=out.get("game_id"))
    cov = next((x for x in logs if x.get("phase") == "coverage"), {})
    print("coverage", {k: cov[k] for k in cov if k != "op_status"})
    print(f"seeds ok={len(ok)}/{len(EXPECTED)} degraded={deg}")

    # bar: load thick — at least 12/17 ok, extra dim 272, auto fuse, wm ok
    fail = []
    if len(ok) < 12:
        fail.append(f"too_few_ok:{len(ok)}")
    if st.extra_feats is None or st.extra_feats.numel() < 200:
        fail.append("extra_dim_low")
    if float((st.extra_feats.abs() > 0).float().mean()) < 0.15:
        fail.append("extra_sparse")
    if (cov.get("wm") or {}).get("status") != "ok":
        fail.append("wm")
    if "ada_online" not in [x.get("phase") for x in logs]:
        fail.append("no_ada")
    if fail:
        print("FAIL", fail)
        return 1
    print("THICK LOAD PASS — garden is planted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
