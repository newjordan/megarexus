#!/usr/bin/env python3
"""FULL stack smoke — no 'still not' bar. Every major creative surface must be ok.

Requires real SP80:
  - trusted map
  - physics body/ray/field ok
  - legal392 ok with feature_dim 392
  - crucible mine_crucible backend (not entropy proxy)
  - FULL run_contra_view fanout layers ok
  - train auto-fuse + ada + wide wm
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CONTRA_LAYERS = [
    "contra_view",
    "contra_vision",
    "contra_control",
    "contra_surfaces",
    "contra_symmetry",
    "contra_actor",
    "contra_target",
    "contra_support",
    "contra_statecon",
    "contra_geoband",
    "contra_boundary",
    "contra_ladder",
    "contra_delta_gate",
    "contra_explore",
    "contra_oracle",
    "contra_ghost",
    "contra_episode",
    "contra_semantics",
    "contra_admission",
    "contra_portband",
]

CORE = [
    "body",
    "ray",
    "field",
    "conceptual",
    "portband",
    "macros",
    "symmetry",
    "geo",
    "saliency",
    "legal392",
    "crucible",
    "ghost",
    "delta",
    "perception_trust",
]


def main() -> int:
    from synthesis_lab.orchestration.rdl.brain.collect_real import collect_transitions
    from synthesis_lab.orchestration.rdl.brain.hemisphere_i import EXTRA_FEAT_DIM, OP_FAMILIES
    from synthesis_lab.orchestration.rdl.brain.megamind import MegaMind

    out = collect_transitions(n_steps=10, seed=11)
    mm = MegaMind(device="cpu")
    mm.reset_episode(game_id=out.get("game_id") or "sp80-589a99af")
    frames, actions, fn = out["frames"], out["actions"], out["frames_next"]
    for i in range(min(5, len(frames))):
        mm.I.remember_transition(
            frames[i].numpy(), int(actions[i].item()), fn[i].numpy()
        )

    # two observes so DeltaBand gets before/after GeoBands from full CONTRA
    _ = mm.observe(frames[min(4, len(frames) - 1)].numpy())
    st = mm.observe(frames[min(5, len(frames) - 1)].numpy())
    ds = st.dream_summary or {}
    fail = []

    print("trusted", st.trusted, "EXTRA_FEAT_DIM", EXTRA_FEAT_DIM, "n_families", len(OP_FAMILIES))
    if not st.trusted:
        fail.append("not_trusted")
    if st.extra_feats is None or int(st.extra_feats.shape[-1]) < EXTRA_FEAT_DIM:
        fail.append(f"extra_dim:{None if st.extra_feats is None else st.extra_feats.shape}")

    ok_c = 0
    for name in CONTRA_LAYERS:
        pkt = ds.get(name) or {}
        status = pkt.get("status") if isinstance(pkt, dict) else "missing"
        print(f"  CONTRA {name:22s} {status}")
        if status == "ok":
            ok_c += 1
        else:
            fail.append(f"contra:{name}:{status}")
    print(f"contra layers ok {ok_c}/{len(CONTRA_LAYERS)}")

    for name in CORE:
        pkt = ds.get(name) or {}
        status = pkt.get("status") if isinstance(pkt, dict) else "missing"
        backend = ""
        if isinstance(pkt, dict):
            msgs = pkt.get("messages") or []
            backend = ",".join(str(m) for m in msgs[:2])
        print(f"  CORE   {name:22s} {status:10s} {backend}")
        if status != "ok":
            fail.append(f"core:{name}:{status}")

    # crucible must be full mine not proxy
    cru = ds.get("crucible") or {}
    msgs = " ".join(str(m) for m in (cru.get("messages") or []))
    scores = cru.get("scores") or {}
    if "crucible_full_mine" not in msgs and float(scores.get("n_rules", -1)) < 0:
        # allow ok if n_rules key exists from mine_crucible
        if "n_discrepancies" not in scores:
            fail.append("crucible_not_full_mine")
    print("crucible scores", {k: scores.get(k) for k in list(scores)[:8]})

    # legal392 must report 392
    leg = (ds.get("legal392") or {}).get("scores") or {}
    if float(leg.get("feature_dim") or 0) != 392:
        fail.append(f"legal_dim:{leg.get('feature_dim')}")

    # contra_view backend
    cv = ds.get("contra_view") or {}
    if cv.get("status") != "ok":
        fail.append("contra_view_dead")
    print("contra_view messages", cv.get("messages"))

    logs = mm.train_transition(frames[:4], actions[:4], fn[:4], game_id=out.get("game_id"))
    phases = [x.get("phase") for x in logs]
    cov = next((x for x in logs if x.get("phase") == "coverage"), {})
    print("phases", phases)
    print("coverage ops", cov.get("n_ops_ok"), "/", cov.get("n_ops"), "wm", cov.get("wm"))
    if "ada_online" not in phases:
        fail.append("no_ada")
    if (cov.get("wm") or {}).get("status") != "ok":
        fail.append("wm")
    if float(cov.get("extra_nonzero") or 0) < 0.1:
        fail.append("sparse_extra")

    if fail:
        print("FULL STACK FAIL", fail)
        return 1
    print("FULL STACK PASS — creativity is on the wire")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
