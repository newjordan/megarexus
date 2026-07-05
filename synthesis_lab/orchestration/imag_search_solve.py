"""Imagination-guided in-budget solve (NIGHT FINDING 33+): hill-climb arrangement-space
scoring by v48's PREDICTED fill - lambda*lava (free, no env pours), multi-restart to avoid
the CEM's premature collapse, then pour only the top-K distinct candidates for REAL (<=4 =
the game budget). Solved if a real win is in those K pours. Tests whether v48's imagination
is accurate enough to GUIDE a search to a real win — the missing proposer piece.

Legal: the search is pure imagination (v48, no pours); the env grades only the final <=4
confirmations; no answer/route injection.

    python3 synthesis_lab/orchestration/imag_search_solve.py --ckpt <v48>/best.pt --levels 0,1,2
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch

_LAB = Path(__file__).resolve().parents[1]; _REPO = _LAB.parent
for p in (_LAB, _LAB / "orchestration", _LAB / "experiments" / "megarexus_cnn"):
    sys.path.insert(0, str(p))
from imagine_search_gate import (_present, _apply_real, _grid, GAME_ID, ACTION_OF_ID, game_live)  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from eval_ghost import encode  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

MOVES = (1, 2, 3, 4)


class V48:
    def __init__(self, ckpt, dev, lam):
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        self.stack = int(ck["config"].get("stack_frames", 2))
        self.m = MegarexusCNN(MegarexusCNNConfig(n_input_frames=self.stack)).to(dev).eval()
        self.m.load_state_dict(ck["model"]); self.dev = dev; self.lam = lam

    @torch.no_grad()
    def score_frame(self, g):
        e = encode(g)
        x = (torch.cat([e, e], 0) if self.stack == 2 else e).unsqueeze(0).to(self.dev)
        o = self.m.predict_outcome(x)[0]
        return float(torch.sigmoid(o[0]) - self.lam * torch.sigmoid(o[1]))


def actions(genome, centers):
    a = []
    for (ci, d, mag) in genome:
        if ci < len(centers):
            cx, cy = centers[ci]; a.append((6, cx, cy)); a += [(int(d), 0, 0)] * int(mag)
    return a


def frame_of(arc, seed, level, genome, centers):
    env = arc.make(GAME_ID, seed=seed); fd, cur = _present(env, level)
    fd, cur, _ = _apply_real(env, fd, cur, actions(genome, centers))
    return (cur.astype(np.int8) if game_live(fd) else None)


def real_win(arc, seed, level, genome, centers):
    env = arc.make(GAME_ID, seed=seed); fd, cur = _present(env, level)
    base = int(getattr(fd, "levels_completed", 0) or 0)
    fd, cur, _ = _apply_real(env, fd, cur, actions(genome, centers))
    if not game_live(fd):
        return False
    try:
        fd = env.step(ACTION_OF_ID[5])
    except Exception:
        return False
    return fd is not None and int(getattr(fd, "levels_completed", 0) or 0) > base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--levels", default="0,1,2")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--steps", type=int, default=80)      # hill-climb steps per restart (imagined)
    ap.add_argument("--topk", type=int, default=4)        # real confirmation pours (budget)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--max-genes", type=int, default=8)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    v = V48(a.ckpt, dev, a.lam)
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((_REPO / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    print(f"[imag] ckpt={a.ckpt}", flush=True)
    for level in [int(x) for x in a.levels.split(",") if x != ""]:
        env0 = arc.make(GAME_ID, seed=a.seed); _, c0 = _present(env0, level)
        centers = [(int(c["center_xy"][0]), int(c["center_xy"][1])) for c in frame_components(c0)[:8]]
        rng = np.random.default_rng(a.seed + level)

        def rand_g():
            k = int(rng.integers(1, a.max_genes + 1))
            return [(int(rng.integers(len(centers))), int(rng.choice(MOVES)), int(rng.integers(1, 4))) for _ in range(k)]

        def mutate(g):
            g = [list(x) for x in g]; op = rng.integers(4)
            if op == 0 and g: i = rng.integers(len(g)); g[i][1] = int(rng.choice(MOVES))
            elif op == 1 and g: i = rng.integers(len(g)); g[i][2] = int(rng.integers(1, 4))
            elif op == 2 and len(g) < a.max_genes: g.append([int(rng.integers(len(centers))), int(rng.choice(MOVES)), int(rng.integers(1, 4))])
            elif g and len(g) > 1: del g[int(rng.integers(len(g)))]
            return [tuple(x) for x in g]

        found = []  # (imagined_score, genome) best per restart
        for r in range(a.restarts):
            cur = rand_g()
            cf = frame_of(arc, a.seed, level, cur, centers)
            cs = v.score_frame(cf) if cf is not None else -1e9
            for _ in range(a.steps):
                cand = mutate(cur)
                cff = frame_of(arc, a.seed, level, cand, centers)
                if cff is None:
                    continue
                s = v.score_frame(cff)
                if s > cs:
                    cur, cs = cand, s
            found.append((cs, cur))
        # distinct top-K by imagined score, pour for real
        found.sort(key=lambda x: -x[0])
        seen, topk = set(), []
        for s, g in found:
            key = tuple(sorted(g))
            if key in seen:
                continue
            seen.add(key); topk.append((s, g))
            if len(topk) >= a.topk:
                break
        solved_at = None
        for rank, (s, g) in enumerate(topk):
            if real_win(arc, a.seed, level, g, centers):
                solved_at = rank + 1; break
        print(f"[imag] L{level}: top-{a.topk} imagined-scores={[round(s,3) for s,_ in topk]} | "
              f"{'SOLVED@'+str(solved_at) if solved_at else 'NOT solved'} in-budget "
              f"(<= {a.topk} real pours; searched {a.restarts}x{a.steps} imagined)", flush=True)


if __name__ == "__main__":
    main()
