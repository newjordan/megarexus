"""Hard-negative mining for the pour imagination (NIGHT FINDING 38).

The imagination's ARGMAX is imprecise: a v52-guided search finds arrangements it scores as
confident wins (~0.88) that actually LOSE. Those false positives are exactly what breaks
search-to-solve. Mine them: run the v52-guided search on real levels AND fresh multi-base
variants, real-pour the top-K (highest imagined score), and RECORD them with real
(fill, lava, env_win) labels. Adding these high-imagined REAL-LOSSES back to training as
negatives teaches the imagination to reject its own false peaks -> sharper argmax. Legal:
env-graded outcomes, search is the agent's own imagination, no answer injection.

    python3 synthesis_lab/orchestration/harvest_hardneg.py --ckpt <v52>/best.pt \
        --levels 0,1,2 --variant-bases 0,1,2,3 --n-variants 300 --out <data>/hardneg.pt
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
from sp80_variants import make_env_variant  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

MOVES = (1, 2, 3, 4)


class V:
    def __init__(self, ckpt, dev, lam):
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        self.stack = int(ck["config"].get("stack_frames", 2))
        self.m = MegarexusCNN(MegarexusCNNConfig(n_input_frames=self.stack)).to(dev).eval()
        self.m.load_state_dict(ck["model"]); self.dev = dev; self.lam = lam

    @torch.no_grad()
    def score(self, g):
        e = encode(g); x = (torch.cat([e, e], 0) if self.stack == 2 else e).unsqueeze(0).to(self.dev)
        o = self.m.predict_outcome(x)[0]
        return float(torch.sigmoid(o[0]) - self.lam * torch.sigmoid(o[1]))


def sustained_fill(frames, ntarg, K=3):
    n13 = [int((x == 13).sum()) for x in frames]
    b = 0
    for i in range(len(n13) - K + 1):
        b = max(b, min(n13[i:i + K]))
    return b / max(ntarg, 1)


def search_topk(v, present_fn, centers, rng, restarts, steps, topk, max_genes=8):
    """Return the top-K distinct arrangements by imagined score. present_fn(genome)->frame."""
    def rg():
        k = int(rng.integers(1, max_genes + 1))
        return [(int(rng.integers(len(centers))), int(rng.choice(MOVES)), int(rng.integers(1, 4))) for _ in range(k)]

    def mut(g):
        g = [list(x) for x in g]; op = rng.integers(4)
        if op == 0 and g: i = rng.integers(len(g)); g[i][1] = int(rng.choice(MOVES))
        elif op == 1 and g: i = rng.integers(len(g)); g[i][2] = int(rng.integers(1, 4))
        elif op == 2 and len(g) < max_genes: g.append([int(rng.integers(len(centers))), int(rng.choice(MOVES)), int(rng.integers(1, 4))])
        elif g and len(g) > 1: del g[int(rng.integers(len(g)))]
        return [tuple(x) for x in g]

    found = []
    for _ in range(restarts):
        cur = rg(); cf = present_fn(cur); cs = v.score(cf) if cf is not None else -1e9
        for _ in range(steps):
            cand = mut(cur); cff = present_fn(cand)
            if cff is None:
                continue
            s = v.score(cff)
            if s > cs:
                cur, cs = cand, s
        found.append((cs, cur))
    found.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for s, g in found:
        k = tuple(sorted(g))
        if k in seen:
            continue
        seen.add(k); out.append((s, g))
        if len(out) >= topk:
            break
    return out


def acts(gen, centers):
    a = []
    for ci, d, mag in gen:
        if ci < len(centers):
            cx, cy = centers[ci]; a.append((6, cx, cy)); a += [(d, 0, 0)] * mag
    return a


def real_pour(env, cur, base_wall, ntarg, base_lv):
    try:
        fd = env.step(ACTION_OF_ID[5])
    except Exception:
        return None
    if fd is None:
        return None
    won = int(getattr(fd, "levels_completed", 0) or 0) > base_lv
    f = fd.frame
    frames = [np.asarray(x, dtype=np.int8) for x in f] if isinstance(f, list) else [np.asarray(f, dtype=np.int8)]
    lava = max(int((x == 14).sum()) for x in frames) - base_wall
    return sustained_fill(frames, ntarg), int(lava), bool(won)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", default="0,1,2")
    ap.add_argument("--variant-bases", default="0,1,2,3")
    ap.add_argument("--n-variants", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--restarts", type=int, default=20)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--base-seed", type=int, default=70000)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    v = V(a.ckpt, dev, a.lam)
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((_REPO / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    recs, nfp = [], 0

    def mine(present_fn, centers, ntarg, base_wall, base_lv, poolseed, tag):
        nonlocal nfp
        rng = np.random.default_rng(poolseed)
        top = search_topk(v, present_fn, centers, rng, a.restarts, a.steps, a.topk)
        for s, g in top:
            env2, cur2 = present_fn(g, return_env=True)   # fresh env + the arrangement frame
            if env2 is None or cur2 is None:
                continue
            out = real_pour(env2, cur2, base_wall, ntarg, base_lv)
            if out is None:
                continue
            fill, lava, won = out
            recs.append({"arrangement": cur2, "fill": float(fill), "lava": int(lava),
                         "env_win": bool(won), "n_targets": ntarg, "variant": tag})
            nfp += int(s > 0.3 and not won)   # high-imagined real-loss = mined false positive

    # real levels
    for level in [int(x) for x in a.levels.split(",") if x != ""]:
        for si in range(a.seeds):
            seed = a.base_seed + level * 1000 + si
            env0 = arc.make(GAME_ID, seed=seed); fd0, c0 = _present(env0, level)
            centers = [(int(c["center_xy"][0]), int(c["center_xy"][1])) for c in frame_components(c0)[:8]]
            ntarg = int((c0 == 11).sum()); base_wall = int((c0 == 14).sum())
            base_lv = int(getattr(fd0, "levels_completed", 0) or 0)

            def present_fn(g, return_env=False, _seed=seed, _level=level, _c=centers):
                env = arc.make(GAME_ID, seed=_seed); fd, cur = _present(env, _level)
                fd, cur, _ = _apply_real(env, fd, cur, acts(g, _c))
                if not game_live(fd):
                    return (None, None) if return_env else None
                return (env, cur.astype(np.int8)) if return_env else cur.astype(np.int8)
            mine(present_fn, centers, ntarg, base_wall, base_lv, seed, 100000 + level)
        print(f"[hn] level {level}: {len(recs)} recs, {nfp} mined false-positives so far", flush=True)

    # fresh multi-base variants
    bases = [int(x) for x in a.variant_bases.split(",") if x != ""]
    for vi in range(a.n_variants):
        base = bases[vi % len(bases)]
        vs = a.base_seed + 500000 + vi
        env0, g0 = make_env_variant(arc, seed=vs, rng=np.random.default_rng(vs), base_level=base)
        if env0 is None:
            continue
        centers = [(int(c["center_xy"][0]), int(c["center_xy"][1])) for c in frame_components(g0)[:8]]
        ntarg = int((g0 == 11).sum()); base_wall = int((g0 == 14).sum())

        def present_fn(g, return_env=False, _vs=vs, _base=base, _c=centers):
            env, cur = make_env_variant(arc, seed=_vs, rng=np.random.default_rng(_vs), base_level=_base)
            if env is None:
                return (None, None) if return_env else None
            fd = env.step(ACTION_OF_ID[1]); cur = _grid(fd)
            try:
                fd, cur, _ = _apply_real(env, fd, cur, acts(g, _c))
            except Exception:
                return (None, None) if return_env else None
            if not game_live(fd):
                return (None, None) if return_env else None
            return (env, cur.astype(np.int8)) if return_env else cur.astype(np.int8)
        try:
            mine(present_fn, centers, ntarg, base_wall, 0, vs, 200000 + vi)
        except Exception:
            continue
        if (vi + 1) % 50 == 0:
            print(f"[hn] variant {vi+1}/{a.n_variants}: {len(recs)} recs, {nfp} false-positives", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(recs, a.out)
    wins = sum(r["env_win"] for r in recs)
    print(f"[hn] done. {len(recs)} mined pours, {wins} wins, {nfp} high-imagined REAL-LOSSES "
          f"(hard negatives) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
