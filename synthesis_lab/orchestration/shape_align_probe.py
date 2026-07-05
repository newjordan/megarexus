"""Decisive test (NIGHT FINDING 24): does Contra's portband port-ALIGNMENT (a SHAPE
signal) predict the SP80 pour win on L0 AND give a non-degenerate gradient on L1, where
the pooled v47 latent saturated (corr 0.04)? If yes, shape-alignment is the level-
invariant substrate the user's oracle/portband/ghost design targets.

Uses the REAL Contra pipeline (build_contra_perception_pipeline + oracle builders) to get
per actor x target statecons; the alignment scalar per arrangement = the BEST
target_open_axis alignment across statecons (does a pour stream from the source enter a
target through its OPENING). Correlates against env_win (L0) and sustained_fill (both).
Legal: portband is a shape INFORMER; env grades. No answers, no routes.

    python3 synthesis_lab/orchestration/shape_align_probe.py \
        --eval <eval_shape_l0l1.pt> [--diag]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch

_LAB = Path(__file__).resolve().parents[1]; _REPO = _LAB.parent
for p in (_LAB, _LAB / "orchestration", _LAB / "experiments" / "megarexus_cnn",
          _REPO / "codexlab"):
    sys.path.insert(0, str(p))

from contra_head.perception import build_contra_perception_pipeline  # noqa: E402
from contra_head.controller_affordances import (  # noqa: E402
    oracle_actor_affordance_hypotheses, oracle_target_affordance_hypotheses,
)
try:
    from contra_head.portband import _alignment_static_score  # noqa: E402
except Exception:
    _alignment_static_score = None


def perceive(frame):
    return build_contra_perception_pipeline(
        frame=np.asarray(frame, dtype=int),
        level_index=0,
        source_raw_frame_sha256="frame0",
        raw_color_inventory={},
        calibration=None,
        actor_builder=oracle_actor_affordance_hypotheses,
        target_builder=oracle_target_affordance_hypotheses,
    )


def align_features(frame):
    """Return a dict of shape-alignment scalars aggregated over statecons."""
    p = perceive(frame)
    scs = p.get("state_condition_hypotheses", []) or []
    open_axis, ray_hit, axis_al, static = [], [], [], []
    for sc in scs:
        f = sc.get("features", {}) or {}
        open_axis.append(float(f.get("target_open_axis_match", 0) or 0))
        ray_hit.append(float(f.get("ray_intersects_target", 0) or 0))
        axis_al.append(float(f.get("axis_alignment", 0) or 0))
        if _alignment_static_score is not None:
            for a in (sc.get("portband_alignment_hypotheses", []) or []):
                try:
                    static.append(float(_alignment_static_score(a)))
                except Exception:
                    pass
    def agg(xs):
        return (max(xs) if xs else 0.0, float(np.mean(xs)) if xs else 0.0)
    oa_max, oa_mean = agg(open_axis)
    rh_max, rh_mean = agg(ray_hit)
    aa_max, aa_mean = agg(axis_al)
    st_max, st_mean = agg(static)
    return {"n_statecons": len(scs), "n_targets": len(p.get("target_affordance_hypotheses", []) or []),
            "n_actors": len(p.get("actor_affordance_hypotheses", []) or []),
            "open_axis_max": oa_max, "open_axis_mean": oa_mean,
            "ray_hit_max": rh_max, "ray_hit_mean": rh_mean,
            "axis_al_max": aa_max, "axis_al_mean": aa_mean,
            "static_max": st_max, "static_mean": st_mean}


def auc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels).astype(bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    recs = torch.load(a.eval, weights_only=False)

    if a.diag:
        # show features on a win vs a lose L0 arrangement
        l0 = [r for r in recs if r["level"] == 0]
        win = next(r for r in l0 if r["env_win"])
        lose = next(r for r in l0 if not r["env_win"])
        print("WIN  features:", align_features(win["arrangement"]))
        print("LOSE features:", align_features(lose["arrangement"]))
        return

    keys = ["open_axis_max", "open_axis_mean", "ray_hit_max", "ray_hit_mean",
            "axis_al_max", "axis_al_mean", "static_max", "static_mean"]
    for lv in [0, 1]:
        sub = [r for r in recs if r["level"] == lv]
        if a.limit:
            sub = sub[:a.limit]
        feats = {k: [] for k in keys}
        wins, sfs = [], []
        for i, r in enumerate(sub):
            fe = align_features(r["arrangement"])
            for k in keys:
                feats[k].append(fe[k])
            wins.append(int(r["env_win"])); sfs.append(float(r["sustained_fill"]))
            if (i + 1) % 500 == 0:
                print(f"  L{lv} {i+1}/{len(sub)}", flush=True)
        wins = np.array(wins); sfs = np.array(sfs)
        print(f"\n=== L{lv}: n={len(sub)} env_wins={int(wins.sum())} ===", flush=True)
        for k in keys:
            x = np.array(feats[k])
            a_win = auc(x, wins) if wins.sum() > 0 else float("nan")
            c_sf = float(np.corrcoef(x, sfs)[0, 1]) if x.std() > 0 and sfs.std() > 0 else float("nan")
            print(f"  {k:16s} AUC_vs_win={a_win:.3f}  corr_vs_sustained_fill={c_sf:+.3f}  "
                  f"[mean={x.mean():.2f} std={x.std():.2f}]", flush=True)


if __name__ == "__main__":
    main()
