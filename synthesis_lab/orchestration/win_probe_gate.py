"""Stage B gate — FRESH-arrangement ranking AUC of the latent win-probe.

The roadmap's Stage-B success metric: on FRESH (not-in-training) L0 arrangements,
rank by `win_probe(settle_predictor(encode(arrangement)))` in ONE pass, real-pour
each to get the env-graded outcome, and measure ranking AUC + whether the real
winner lands in the top-K the two-tier planner would pixel/geometry-check. Must beat
the saturated pour-outcome head (0.500 on this data) and the pixel-rollout ranker
(≈random discrimination). The env pour is the ground-truth label — legal, honest.

    python3 synthesis_lab/orchestration/win_probe_gate.py \
        --ckpt <v46>/best.pt --levels 0 --seeds 4 --random-cands 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_LAB = Path(__file__).resolve().parents[1]
_REPO = _LAB.parent
for p in (_LAB, _LAB / "orchestration", _LAB / "experiments" / "megarexus_cnn"):
    sys.path.insert(0, str(p))

from imagine_search_gate import (  # noqa: E402
    _present, _candidates, _apply_real, _grid,
    GAME_ID, ACTION_OF_ID, game_live,
)
from eval_ghost import encode  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402


def auc(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


class Probe:
    def __init__(self, ckpt, dev, mode="winprobe", lam=1.0):
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        self.stack = int(ck["config"].get("stack_frames", 2))
        self.m = MegarexusCNN(MegarexusCNNConfig(n_input_frames=self.stack)).to(dev).eval()
        self.m.load_state_dict(ck["model"])
        self.dev = dev
        self.mode = mode      # "winprobe" (v47 win_logit) or "outcome" (v48 fill-lam*lava)
        self.lam = lam

    @torch.no_grad()
    def score(self, arr):
        oh = encode(arr)
        t = torch.cat([oh, oh], 0) if self.stack == 2 else oh
        x = t.unsqueeze(0).to(self.dev)
        if self.mode == "outcome":
            o = self.m.predict_outcome(x)[0]
            return float(torch.sigmoid(o[0]) - self.lam * torch.sigmoid(o[1]))
        return float(torch.sigmoid(self.m.win_logit(x))[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--levels", default="0")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--random-cands", type=int, default=60)
    ap.add_argument("--max-comps", type=int, default=6)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--mode", default="winprobe", choices=["winprobe", "outcome"])
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mags = (1, 2, 3)
    pr = Probe(a.ckpt, dev, mode=a.mode, lam=a.lam)
    arcade = Arcade(arc_api_key="anonymous",
                    environments_dir=str((_REPO / "environment_files").resolve()),
                    operation_mode=OperationMode.OFFLINE)
    print(f"[wpg] ckpt={a.ckpt} dev={dev} stack={pr.stack}", flush=True)
    for level in [int(x) for x in a.levels.split(",") if x != ""]:
        scores, labels = [], []
        winner_ranks = []  # rank (0=best) of each real winner among that seed's candidates
        seed_min_rank = []  # per seed: best (min) rank of a real winner, or None
        for si in range(a.seeds):
            s = a.base_seed + int(level) * 1000 + si
            env = arcade.make(GAME_ID, seed=s)
            _, cur0 = _present(env, level)
            rng = np.random.default_rng(s)
            cands = _candidates(cur0, a.max_comps, mags, a.random_cands, rng)
            seed_scores, seed_labels = [], []
            for arrangement in cands:
                env_i = arcade.make(GAME_ID, seed=s)
                fdi, curi = _present(env_i, level)
                fdi, arr, _ = _apply_real(env_i, fdi, curi, arrangement)
                if not game_live(fdi):
                    continue
                sc = pr.score(arr)
                # real pour -> env-graded win
                env_r = arcade.make(GAME_ID, seed=s)
                fdr, curr = _present(env_r, level)
                base_lv = int(getattr(fdr, "levels_completed", 0) or 0)
                fdr, curr, _ = _apply_real(env_r, fdr, curr, arrangement)
                rw = False
                if game_live(fdr):
                    for _ in range(50):
                        fdr = env_r.step(ACTION_OF_ID[5])
                        if fdr is None:
                            break
                        if int(getattr(fdr, "levels_completed", 0) or 0) > base_lv:
                            rw = True; break
                        if not game_live(fdr):
                            break
                seed_scores.append(sc); seed_labels.append(int(rw))
            scores += seed_scores; labels += seed_labels
            # winner rank within this seed
            order = np.argsort(-np.array(seed_scores))
            ranked_labels = np.array(seed_labels)[order]
            win_ranks_this_seed = [r for r, lab in enumerate(ranked_labels) if lab == 1]
            winner_ranks += win_ranks_this_seed
            seed_min_rank.append(min(win_ranks_this_seed) if win_ranks_this_seed else None)
        A = auc(scores, labels)
        labels = np.array(labels); scores = np.array(scores)
        nwin = int(labels.sum()); n = len(labels)
        order = np.argsort(-scores)
        topk = order[:a.topk]
        in_topk = int(labels[topk].sum())
        # END-TO-END honest solve: rank each seed's proposals by imagined win, pour
        # the top-K for REAL, env grades. solved@K = seeds where a real winner is
        # ranked within the first K pours (the 4-pour budget is the real game).
        solved = {K: sum(1 for mr in seed_min_rank if mr is not None and mr < K)
                  for K in (1, 2, 4)}
        n_solvable = sum(1 for mr in seed_min_rank if mr is not None)
        print(f"[wpg] level {level}: candidates={n} real_wins={nwin} "
              f"| FRESH ranking AUC={A:.3f} | wins in top-{a.topk}={in_topk}/{nwin} "
              f"| winner_ranks={sorted(winner_ranks)}", flush=True)
        print(f"[wpg] level {level}: END-TO-END solved@1={solved[1]}/{a.seeds} "
              f"solved@2={solved[2]}/{a.seeds} solved@4={solved[4]}/{a.seeds} "
              f"(seeds whose proposals contained a real win: {n_solvable}/{a.seeds})",
              flush=True)
        print(f"[wpg] level {level}: read => probe {'BEATS 0.79 bar' if A > 0.79 else 'below 0.79'}; "
              f"rank+pour-top-4-real {'SOLVES L0 CNN-alone' if solved[4] > 0 else 'does not solve'} "
              f"in {solved[4]}/{a.seeds} seeds", flush=True)


if __name__ == "__main__":
    main()
