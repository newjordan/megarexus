"""Debug/eyeball renders for sim3d rollouts.

    python3 viz.py --fixture t0_arrangements.npz --index 0 --lane stream --out /tmp/winner.png

Left panel: the pre-pour arrangement frame. Middle: sim occupancy over the scene
(canonical). Right: the final RENDERED frame the scorer actually reads. For Lane P
add --spheres to scatter the individual sphere rest positions (the literal
"pour spheres out of the bucket" picture).
"""
from __future__ import annotations

import argparse

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES

from scene import M
from pour3d import imagine_pour
from metrics import render_frames

# SP80 palette (indexes match frame colors; approximate, for eyeballing only)
PAL = np.array([
    [0, 0, 0], [64, 64, 72], [128, 0, 128], [0, 128, 128], [230, 180, 40],
    [200, 200, 200], [90, 160, 255], [255, 255, 255], [160, 110, 60],
    [240, 140, 40], [120, 200, 120], [60, 60, 200], [24, 24, 28],
    [80, 220, 100], [220, 60, 60], [255, 0, 255],
], dtype=np.uint8)


def frame_rgb(f):
    return PAL[np.clip(np.asarray(f, dtype=np.int16), 0, 15)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="t0_arrangements.npz")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--lane", default="stream")
    ap.add_argument("--spheres", action="store_true")
    ap.add_argument("--out", default="/tmp/sim3d_viz.png")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(FIXTURES / args.fixture)
    pre = z["pre"][args.index] if z["pre"].ndim == 3 else z["pre"]
    name = str(z["names"][args.index]) if "names" in z else f"#{args.index}"
    out = imagine_pour(pre, lane=args.lane)
    sc, res = out["scene"], out["result"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    axes[0].imshow(frame_rgb(pre)); axes[0].set_title(f"pre-pour ({name})")
    occ = np.asarray(res["occupancy"], dtype=float)
    base = np.where(sc.solid, -1.0, 0.0)
    axes[1].imshow(base, cmap="gray", vmin=-1, vmax=1)
    axes[1].imshow(np.ma.masked_where(occ <= 0.01, occ), cmap="Blues", alpha=0.9)
    if args.spheres and "x" in res:
        x = res["x"]
        axes[1].scatter(x[:, 1] - 0.5, x[:, 0] - 0.5, s=14, c="orange",
                        edgecolors="k", linewidths=0.3)
    axes[1].set_title(f"{args.lane} rollout (canonical)  margin={out['margin']:+.2f}")
    axes[1].set_xlim(-0.5, M - 0.5); axes[1].set_ylim(M - 0.5, -0.5)
    frames = render_frames(sc, res["snapshots"],
                           per_cell=res["params"].get("per_cell",
                                                      res["params"].get("cell_cap", 1.0)))
    axes[2].imshow(frame_rgb(frames[-1] if len(frames) else pre))
    axes[2].set_title(f"rendered settle (scored)  winner_ok={out['winner_ok']}")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
