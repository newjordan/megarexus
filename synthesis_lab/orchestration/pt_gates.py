"""T3 GATES for the per-target head (STAGE_PT_SPEC §5.T3). Pre-registered, in order.

G1 (THE gate): on real L1, the verified winner's arr_score (min-aggregated per-target)
    is POSITIVE and STRICTLY ABOVE all four T0 false positives. (v52 aggregate baseline:
    winner +0.51 BELOW FPs at ~0.65-0.75 — the point is to FLIP this ordering.)
G2 (no ranker regression): REAL-L0/L1 win-AUC of arr_score >= (v52 aggregate AUC - 0.02),
    both computed on the SAME freshly-harvested real eval arrangements (held out, eval only).
G3 (headline, only if G1 AND G2 pass): in-budget solve on real L1 — hill-climb with
    fitness = arr_score, pour top-4 DIVERSE, env grades. PASS = levels_completed 0->1 in <=4.

Legal: search is pure imagination (the head, no pours); env grades only the final <=4
confirmations; real levels are EVAL-only, never trained on. Env is the only decider.

    python3 synthesis_lab/orchestration/pt_gates.py --ckpt <v55>/best.pt \
        --v52 <v52>/best.pt --t0 <t0.json> --gates 1,2,3 --out <run>/gates.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch

_LAB = Path(__file__).resolve().parents[1]; _REPO = _LAB.parent
for p in (_LAB, _LAB / "orchestration", _LAB / "experiments" / "megarexus_cnn"):
    sys.path.insert(0, str(p))
from imagine_search_gate import (_present, _candidates, _apply_real, _grid,
                                 GAME_ID, ACTION_OF_ID, game_live)  # noqa: E402
from imag_search_solve import actions, frame_of, real_win, MOVES  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from pertarget_labels import _dilate_bbox, TARGET_COLOR  # noqa: E402
from eval_ghost import encode  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402


def auc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels).astype(bool)
    p, n = scores[labels], scores[~labels]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def load_model(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    stack = int(ck["config"].get("stack_frames", ck["config"].get("n_input_frames", 2)))
    m = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, stack


def _masks(frame):
    comps = [c for c in frame_components(frame) if int(c["color"]) == TARGET_COLOR]
    tms, rms = [], []
    for c in comps:
        tm = np.asarray(c["mask"], np.float32)
        ring = np.clip(_dilate_bbox(c["bbox_xyxy"], 3).astype(np.float32) - tm, 0, 1)
        tms.append(tm); rms.append(ring)
    return tms, rms


class PTScorer:
    """arr_score(frame) = min_t(sigmoid(fill_t) - sigmoid(lava_t)) from the per-target head."""
    def __init__(self, ckpt, dev):
        self.m, self.stack = load_model(ckpt, dev); self.dev = dev

    @torch.no_grad()
    def score(self, frame):
        tms, rms = _masks(frame)
        if not tms:
            return -1e9
        e = encode(frame.astype(np.int8))
        x = (torch.cat([e, e], 0) if self.stack == 2 else e).unsqueeze(0).to(self.dev)
        tm = torch.from_numpy(np.stack(tms)).unsqueeze(0).to(self.dev)   # [1,T,64,64]
        rm = torch.from_numpy(np.stack(rms)).unsqueeze(0).to(self.dev)
        out = self.m.predict_target_outcome(x, tm, rm)[0]                # [T,2]
        s = torch.sigmoid(out[:, 0]) - torch.sigmoid(out[:, 1])
        return float(s.min())


class V52Scorer:
    """v52 aggregate score = sigmoid(fill) - sigmoid(lava) (lam=1) for baseline comparison."""
    def __init__(self, ckpt, dev):
        self.m, self.stack = load_model(ckpt, dev); self.dev = dev

    @torch.no_grad()
    def score(self, frame):
        e = encode(frame.astype(np.int8))
        x = (torch.cat([e, e], 0) if self.stack == 2 else e).unsqueeze(0).to(self.dev)
        o = self.m.predict_outcome(x)[0]
        return float(torch.sigmoid(o[0]) - torch.sigmoid(o[1]))


def centers_for(arc, seed, level):
    env0 = arc.make(GAME_ID, seed=seed); _, c0 = _present(env0, level)
    return [(int(c["center_xy"][0]), int(c["center_xy"][1])) for c in frame_components(c0)[:8]]


def g1(arc, pt, t0, seed=1234, level=1):
    """Winner arr_score POSITIVE and strictly above all four T0 FP arr_scores."""
    d = json.load(open(t0))
    centers = centers_for(arc, seed, level)
    win_g = [tuple(x) for x in d["winner"]["genome"]]
    wf = frame_of(arc, seed, level, win_g, centers)
    ws = pt.score(wf) if wf is not None else -1e9
    fps = []
    for fp in d["false_positives"]:
        if fp.get("skipped"):
            continue
        g = [tuple(x) for x in fp["genome"]]
        f = frame_of(arc, seed, level, g, centers)
        fps.append({"rank": fp["rank"], "v52_score": fp.get("score"),
                    "arr_score": (pt.score(f) if f is not None else -1e9)})
    above = all(ws > fp["arr_score"] for fp in fps)
    passed = bool(ws > 0 and above)
    return {"gate": "G1", "passed": passed, "winner_arr_score": ws,
            "fp_arr_scores": fps, "winner_positive": bool(ws > 0),
            "winner_above_all_fps": bool(above)}


def harvest_real_eval(arc, level, seeds, rand_cands, base_seed=90000):
    """Fresh real-level eval arrangements (held out, EVAL ONLY): (arr_frame, env_win)."""
    out = []
    for si in range(seeds):
        s = base_seed + level * 1000 + si
        env = arc.make(GAME_ID, seed=s); fd0, cur0 = _present(env, level)
        rng = np.random.default_rng(s)
        cands = _candidates(cur0, 6, (1, 2, 3), rand_cands, rng)
        for arr_seq in cands:
            env2 = arc.make(GAME_ID, seed=s); fd, cur = _present(env2, level)
            base = int(getattr(fd, "levels_completed", 0) or 0)
            try:
                fd, cur, _ = _apply_real(env2, fd, cur, arr_seq)
                if not game_live(fd):
                    continue
                arr = cur.astype(np.int8)
                fd = env2.step(ACTION_OF_ID[5])
            except (ValueError, AttributeError, RuntimeError):
                continue
            if fd is None:
                continue
            won = int(getattr(fd, "levels_completed", 0) or 0) > base
            out.append((arr, bool(won)))
    return out


def g2(arc, pt, v52, levels=(0, 1), seeds=30, rand_cands=30):
    res = {"gate": "G2", "levels": {}}
    ok = True
    for lv in levels:
        data = harvest_real_eval(arc, lv, seeds, rand_cands)
        wins = [w for _, w in data]
        pt_s = [pt.score(a) for a, _ in data]
        v52_s = [v52.score(a) for a, _ in data]
        pt_auc = auc(pt_s, wins); v52_auc = auc(v52_s, wins)
        regressed = not (pt_auc >= v52_auc - 0.02) if not np.isnan(pt_auc) and not np.isnan(v52_auc) else True
        ok = ok and not regressed
        res["levels"][f"L{lv}"] = {"n": len(data), "wins": int(sum(wins)),
                                   "pt_arr_auc": pt_auc, "v52_agg_auc": v52_auc,
                                   "pass": bool(not regressed)}
    res["passed"] = bool(ok)
    return res


def g3(arc, pt, seed=1234, level=1, restarts=40, steps=80, max_genes=8, topk=4):
    """In-budget solve on real L1 with fitness = arr_score; pour top-4 DIVERSE, env grades."""
    centers = centers_for(arc, seed, level)
    rng = np.random.default_rng(seed + 777 + level)

    def rand_g():
        k = int(rng.integers(1, max_genes + 1))
        return [(int(rng.integers(len(centers))), int(rng.choice(MOVES)), int(rng.integers(1, 4))) for _ in range(k)]

    def mutate(g):
        g = [list(x) for x in g]; op = rng.integers(4)
        if op == 0 and g: i = rng.integers(len(g)); g[i][1] = int(rng.choice(MOVES))
        elif op == 1 and g: i = rng.integers(len(g)); g[i][2] = int(rng.integers(1, 4))
        elif op == 2 and len(g) < max_genes: g.append([int(rng.integers(len(centers))), int(rng.choice(MOVES)), int(rng.integers(1, 4))])
        elif g and len(g) > 1: del g[int(rng.integers(len(g)))]
        return [tuple(x) for x in g]

    found = []
    for r in range(restarts):
        cur = rand_g(); cf = frame_of(arc, seed, level, cur, centers)
        cs = pt.score(cf) if cf is not None else -1e9
        for _ in range(steps):
            cand = mutate(cur); cff = frame_of(arc, seed, level, cand, centers)
            if cff is None:
                continue
            s = pt.score(cff)
            if s > cs:
                cur, cs = cand, s
        found.append((cs, cur, r))
    found.sort(key=lambda x: -x[0])
    seen_r, topk_g = set(), []
    for s, g, r in found:            # DIVERSE: best-per-restart
        if r in seen_r:
            continue
        seen_r.add(r); topk_g.append((float(s), g))
        if len(topk_g) >= topk:
            break
    solved_at = None
    for rank, (s, g) in enumerate(topk_g):
        if real_win(arc, seed, level, g, centers):
            solved_at = rank + 1; break
    return {"gate": "G3", "passed": bool(solved_at is not None), "solved_at": solved_at,
            "topk_scores": [s for s, _ in topk_g], "topk_genomes": [g for _, g in topk_g]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--v52", required=True)
    ap.add_argument("--t0", required=True)
    ap.add_argument("--gates", default="1,2,3")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    want = set(a.gates.split(","))
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((_REPO / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    pt = PTScorer(a.ckpt, dev); v52 = V52Scorer(a.v52, dev)
    print(f"[gate] ckpt={a.ckpt} dev={dev}", flush=True)
    result = {"ckpt": a.ckpt}

    if "1" in want:
        r1 = g1(arc, pt, a.t0)
        result["G1"] = r1
        print(f"[gate] G1 passed={r1['passed']} winner_arr={r1['winner_arr_score']:.4f} "
              f"fp_arr={[round(f['arr_score'],4) for f in r1['fp_arr_scores']]}", flush=True)

    if "2" in want:
        r2 = g2(arc, pt, v52, seeds=a.seeds)
        result["G2"] = r2
        for lv, d in r2["levels"].items():
            print(f"[gate] G2 {lv}: pt_arr_AUC={d['pt_arr_auc']:.3f} v52_agg_AUC={d['v52_agg_auc']:.3f} "
                  f"n={d['n']} wins={d['wins']} pass={d['pass']}", flush=True)
        print(f"[gate] G2 passed={r2['passed']}", flush=True)

    g1ok = result.get("G1", {}).get("passed", False)
    g2ok = result.get("G2", {}).get("passed", False)
    if "3" in want:
        if g1ok and g2ok:
            r3 = g3(arc, pt)
            result["G3"] = r3
            print(f"[gate] G3 passed={r3['passed']} solved_at={r3['solved_at']} "
                  f"topk_scores={[round(s,4) for s in r3['topk_scores']]}", flush=True)
        else:
            result["G3"] = {"gate": "G3", "skipped": True, "reason": "G1 and/or G2 did not pass"}
            print("[gate] G3 SKIPPED (G1/G2 not both passed)", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, default=float))
    verdict = "G1=%s G2=%s G3=%s" % (result.get("G1", {}).get("passed"),
                                     result.get("G2", {}).get("passed"),
                                     result.get("G3", {}).get("passed", result.get("G3", {}).get("skipped")))
    print(f"[gate] wrote {a.out}", flush=True)
    print(f"[gate] VERDICT {verdict}", flush=True)


if __name__ == "__main__":
    main()
