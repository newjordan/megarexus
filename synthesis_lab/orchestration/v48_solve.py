"""In-budget CNN solve using the generalising imagination (NIGHT FINDING 31).

v48 predicts (fill, lava) from an arrangement in ONE pass (no env pour). So SEARCH
arrangement-space scoring by v48's imagined win = fill_hat - lambda*lava_hat (free), then
pour only the top-K (<=4, the real game budget) with the env. Solved if a real win is in
the top-K. This is the honest payoff: imagination proposes/ranks, the env decides within
budget. Runs on the REAL levels (held out of v48 training) AND fresh variants.

    python3 synthesis_lab/orchestration/v48_solve.py --ckpt <v48>/best.pt --levels 0,1,2 --topk 4
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
    def __init__(self, ckpt, dev):
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        self.stack = int(ck["config"].get("stack_frames", 2))
        self.m = MegarexusCNN(MegarexusCNNConfig(n_input_frames=self.stack)).to(dev).eval()
        self.m.load_state_dict(ck["model"]); self.dev = dev

    @torch.no_grad()
    def score(self, frames_i8, lam):
        # frames_i8: list of [64,64] int8 arrangements -> imagined (fill - lam*lava)
        enc = []
        for g in frames_i8:
            e = encode(g)
            enc.append(torch.cat([e, e], 0) if self.stack == 2 else e)
        x = torch.stack(enc).to(self.dev)
        out = self.m.predict_outcome(x)
        fill = torch.sigmoid(out[:, 0]); lava = torch.sigmoid(out[:, 1])
        return (fill - lam * lava).cpu().numpy()


def genome_actions(genome, centers):
    acts = []
    for (ci, d, mag) in genome:
        if ci < len(centers):
            cx, cy = centers[ci]; acts.append((6, cx, cy)); acts += [(int(d), 0, 0)] * int(mag)
    return acts


def build_frame(arc, seed, level, genome, centers):
    env = arc.make(GAME_ID, seed=seed)
    fd, cur = _present(env, level)
    fd, cur, _ = _apply_real(env, fd, cur, genome_actions(genome, centers))
    if not game_live(fd):
        return None, None
    return env, cur.astype(np.int8)


def real_pour_win(arc, seed, level, genome, centers):
    env = arc.make(GAME_ID, seed=seed)
    fd, cur = _present(env, level)
    base = int(getattr(fd, "levels_completed", 0) or 0)
    fd, cur, _ = _apply_real(env, fd, cur, genome_actions(genome, centers))
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
    ap.add_argument("--pop", type=int, default=800)      # imagined candidates (free)
    ap.add_argument("--iters", type=int, default=6)      # CEM-ish refine rounds
    ap.add_argument("--topk", type=int, default=4)       # real pours (budget)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--max-genes", type=int, default=8)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    v = V48(a.ckpt, dev)
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((_REPO / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    print(f"[solve] ckpt={a.ckpt}", flush=True)
    for level in [int(x) for x in a.levels.split(",") if x != ""]:
        env0 = arc.make(GAME_ID, seed=a.seed); _, c0 = _present(env0, level)
        comps = frame_components(c0)[:8]
        centers = [(int(c["center_xy"][0]), int(c["center_xy"][1])) for c in comps]
        rng = np.random.default_rng(a.seed + level)

        def rand_genome():
            k = int(rng.integers(1, a.max_genes + 1))
            return [(int(rng.integers(len(centers))), int(rng.choice(MOVES)), int(rng.integers(1, 4))) for _ in range(k)]

        def mutate(g):
            g = [list(x) for x in g]; op = rng.integers(4)
            if op == 0 and g: i = rng.integers(len(g)); g[i][1] = int(rng.choice(MOVES))
            elif op == 1 and g: i = rng.integers(len(g)); g[i][2] = int(rng.integers(1, 4))
            elif op == 2 and len(g) < a.max_genes: g.append([int(rng.integers(len(centers))), int(rng.choice(MOVES)), int(rng.integers(1, 4))])
            elif g and len(g) > 1: del g[int(rng.integers(len(g)))]
            return [tuple(x) for x in g]

        # CEM-ish: keep an elite set, score by v48 (imagined, free), refine
        pop = [rand_genome() for _ in range(a.pop)]
        elite = []
        for it in range(a.iters):
            frames, valid = [], []
            for g in pop:
                env, fr = build_frame(arc, a.seed, level, g, centers)
                if fr is not None:
                    frames.append(fr); valid.append(g)
            if not frames:
                break
            sc = v.score(frames, a.lam)
            order = np.argsort(-sc)
            elite = [valid[i] for i in order[:max(a.topk * 4, 32)]]
            # next pop = elites + mutations of elites + fresh randoms
            pop = list(elite)
            while len(pop) < a.pop:
                base = elite[int(rng.integers(len(elite)))]
                pop.append(mutate(base) if rng.random() < 0.8 else rand_genome())
        # final: score elite, pour top-K for real
        frames, valid = [], []
        for g in elite:
            env, fr = build_frame(arc, a.seed, level, g, centers)
            if fr is not None:
                frames.append(fr); valid.append(g)
        sc = v.score(frames, a.lam)
        order = np.argsort(-sc)
        solved_at = None
        for rank, idx in enumerate(order[:a.topk]):
            if real_pour_win(arc, a.seed, level, valid[idx], centers):
                solved_at = rank + 1; break
        print(f"[solve] L{level}: imagined {len(valid)} elites, top-{a.topk} v48-scores="
              f"{[round(float(sc[i]),3) for i in order[:a.topk]]} | "
              f"{'SOLVED@'+str(solved_at) if solved_at else 'NOT solved'} in-budget (<= {a.topk} real pours)", flush=True)


if __name__ == "__main__":
    main()
