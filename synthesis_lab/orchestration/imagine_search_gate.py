"""Imagination-search gate — the honest deliberation path, and the fidelity test.

SP80's pour is a DETERMINISTIC cellular automaton fully determined by the pre-pour
arrangement (sp80.py: in "spill" mode player input is ignored; win = every U-target
recolored 11->13 AND no wall recolored ->14). So the honest planner does NOT need
learned L1+ "positives": it proposes arrangements, IMAGINES the spill with the
CNN's forward_dynamics (autoregressive: one ACTION5 then ACTION0 ticks to settle),
GEOMETRICALLY checks the imagined settle (all targets filled, no wall hit), and
pours for real (env grades). The CA rules are level-invariant, so accurate
imagination generalizes to L1+ with zero L1+ training labels.

This gate doubles as the fidelity diagnostic: for each candidate arrangement it
records BOTH the imagined win-score and the REAL env outcome (reset + apply same
arrangement + real pour). It then reports (a) whether imagination PREDICTS reality
(AUC / accuracy) and (b) whether the candidate set even CONTAINS a winning
arrangement per level (real-win>0) — telling apart "search too small" from
"imagination too weak". Legal: real arrangements via clean env steps, imagined
pours are the model's own world-model rollout, env grades the final real pour.

    python3 synthesis_lab/orchestration/imagine_search_gate.py --ckpt <run>/best.pt \
        --levels 0 --seeds 4 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_LAB = Path(__file__).resolve().parents[1]
_REPO = _LAB.parent
for p in (_LAB, _LAB / "orchestration", _LAB / "experiments" / "megarexus_cnn", _REPO / "codexlab"):
    sys.path.insert(0, str(p))

from collect_catch_play import GAME_ID, ACTION_OF_ID, game_live  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from eval_ghost import encode  # noqa: E402
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

MOVES = (3, 4)          # horizontal shift (blocks redirect streams sideways)
TARGET_COLOR = 11       # unfilled U-target
FILLED_COLOR = 13       # satisfied U-target
WALL_HIT_COLOR = 14     # a stream hit a wall -> FAIL


def _grid(fd):
    f = fd.frame
    return np.asarray(f[-1] if isinstance(f, list) else f, dtype=np.int8)


def _present(env, level):
    env.reset(); env._game.set_level(int(level))
    fd = env.step(ACTION_OF_ID[1])
    return fd, _grid(fd)


ALL_MOVES = (1, 2, 3, 4)  # up/down/left/right — winning arrangements use all axes


def _candidates(frame, max_comps, per_comp_moves, n_random, rng):
    """Arrangement set that actually CONTAINS wins: structured horizontal shifts
    (interpretable) PLUS random arrangements drawn from the same distribution the
    collector used to find real L0 wins (~3%): click a component, then 1-3 moves in
    random directions, 0-4 such clicks. Plus the no-op (pour from start)."""
    cands = [[]]  # no-op
    comps = frame_components(frame)[:max_comps]
    for c in comps:
        cx, cy = int(c["center_xy"][0]), int(c["center_xy"][1])
        for mv in MOVES:
            for mag in per_comp_moves:
                cands.append([(6, cx, cy)] + [(mv, 0, 0)] * mag)
    for _ in range(n_random):
        if not comps:
            break
        seq = []
        for _ in range(int(rng.integers(0, 5))):
            c = comps[int(rng.integers(len(comps)))]
            cx, cy = int(c["center_xy"][0]), int(c["center_xy"][1])
            seq.append((6, cx, cy))
            for _ in range(int(rng.integers(1, 4))):
                seq.append((int(rng.choice(ALL_MOVES)), 0, 0))
        cands.append(seq)
    return cands


def _apply_real(env, fd, cur, arrangement):
    """Apply an arrangement with real env steps. Returns (fd, cur, prev)."""
    prev = cur
    for (aid, x, y) in arrangement:
        if not game_live(fd):
            break
        data = {"x": int(x), "y": int(y)} if aid == 6 else None
        fd = env.step(ACTION_OF_ID[aid], data=data)
        prev = cur; cur = _grid(fd)
    return fd, cur, prev


class Imaginer:
    def __init__(self, ckpt, dev, ca=False, oneshot=False):
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
        self.stack = int(ck["config"].get("stack_frames", 2))
        self.m = MegarexusCNN(MegarexusCNNConfig(n_input_frames=self.stack)).to(dev).eval()
        self.m.load_state_dict(ck["model"])
        self.dev = dev
        self.ca = ca            # roadmap A2: roll out with the local Neural-CA head
        self.oneshot = oneshot  # A3/B: one-shot settled prediction, no rollout

    def _inp(self, cur, prev):
        t = torch.cat([encode(cur), encode(prev)], 0) if self.stack == 2 else encode(cur)
        return t.unsqueeze(0).to(self.dev)

    @torch.no_grad()
    def _step(self, cur, prev, aid):
        logits = self.m.forward_dynamics(self._inp(cur, prev),
                                         torch.tensor([aid], device=self.dev),
                                         torch.tensor([0], device=self.dev),
                                         torch.tensor([0], device=self.dev))
        return logits.argmax(1)[0].cpu().numpy().astype(np.int8)

    @torch.no_grad()
    def _ca_step(self, cur):
        g = torch.from_numpy(cur.astype(np.int64)).clamp(0, 15).unsqueeze(0).to(self.dev)
        oh = torch.nn.functional.one_hot(g, 16).permute(0, 3, 1, 2).float()
        return self.m.forward_spill_ca(oh).argmax(1)[0].cpu().numpy().astype(np.int8)

    def _active_mask(self, cur):
        """Cells the spill can touch THIS tick = dilation (r2) of paint(6)+source(4).
        The CA only updates these; the rest is frozen — this is the roadmap's
        'hard-masked to cells near active paint' and it bounds the rollout to the
        real stream instead of flooding the grid."""
        seed = torch.from_numpy(((cur == 6) | (cur == 4)).astype(np.float32))[None, None]
        return (torch.nn.functional.max_pool2d(seed, 5, 1, 2) > 0)[0, 0].numpy()

    @torch.no_grad()
    def imagine_pour(self, arr, arr_prev, max_ticks=40):
        """Autoregressive spill until settle (identical consecutive frames) or cap.
        CA mode (A2): iterate the local Neural-CA head from the arrangement, updating
        ONLY the active front (paint/source neighbourhood) and copying the rest —
        source emission is a learned local rule, no action conditioning needed.
        Legacy mode: forward_dynamics with ACTION5 then ACTION0 ticks."""
        if self.oneshot:
            # train/test parity: train_settle_oneshot stacks the arrangement with
            # ITSELF (no pre-pour motion), so the gate must too — passing the real
            # arr_prev here feeds motion context the one-shot decoder never saw.
            logits = self.m.forward_dynamics(self._inp(arr, arr),
                                             torch.tensor([5], device=self.dev),
                                             torch.tensor([0], device=self.dev),
                                             torch.tensor([0], device=self.dev))
            return logits.argmax(1)[0].cpu().numpy().astype(np.int8)
        # "Track the win IN the imagination" (user 2026-07-05): the SP80 win is not
        # visible on the static board — you must imagine the paint pour and watch for
        # the moment every target fills (11->13). That moment is TRANSIENT in an
        # imperfect rollout (the CA over-paints and drifts AFTER it), so returning the
        # settled/fixed-point frame MISSES it. Instead return the peak-fill frame:
        # the tick that maximised fill progress (#13 - #11) over the whole rollout.
        def _fill(f):
            return int((f == FILLED_COLOR).sum()) - int((f == TARGET_COLOR).sum())
        if self.ca:
            cur = arr
            best_f, best_frame = _fill(cur), cur
            for _ in range(max_ticks):
                pred = self._ca_step(cur)
                active = self._active_mask(cur)
                nxt = np.where(active, pred, cur).astype(np.int8)
                if _fill(nxt) > best_f:
                    best_f, best_frame = _fill(nxt), nxt
                if np.array_equal(nxt, cur):
                    break
                cur = nxt
            return best_frame
        prev, cur = arr_prev, arr
        cur = self._step(cur, prev, 5); prev = arr
        best_f, best_frame = _fill(cur), cur
        for _ in range(max_ticks):
            nxt = self._step(cur, prev, 0)
            if _fill(nxt) > best_f:
                best_f, best_frame = _fill(nxt), nxt
            if np.array_equal(nxt, cur):
                break
            prev, cur = cur, nxt
        return best_frame


def win_score(initial, settled):
    """Geometric SP80 win-check on an (imagined or real) settled frame.

    Win = every target satisfied: some cells became FILLED(13) and NO unfilled
    TARGET(11) remains. Color 14 is STRUCTURAL border geometry — present in the
    arrangement (~62 cells) and flickering during the win-flash (up to ~303) — NOT
    a wall-hit/fail marker. Keying win on the absence of 14 false-rejected every
    real win (NIGHT 2026-07-05 root cause); it is reported for diagnostics only."""
    n_target = int((initial == TARGET_COLOR).sum())
    n_filled = int((settled == FILLED_COLOR).sum())
    n_left = int((settled == TARGET_COLOR).sum())
    fill_frac = n_filled / max(n_target, 1)
    return {"fill_frac": fill_frac, "targets_left": n_left,
            "border14": int((settled == WALL_HIT_COLOR).sum()),
            "win": bool(n_filled > 0 and n_left == 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--levels", default="0")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--max-comps", type=int, default=3)
    ap.add_argument("--per-comp-moves", default="2,4")
    ap.add_argument("--random-cands", type=int, default=40)
    ap.add_argument("--base-seed", type=int, default=61200001)
    ap.add_argument("--device", default=None)
    ap.add_argument("--ca", action="store_true", help="roll out with the A2 Neural-CA spill head")
    ap.add_argument("--oneshot", action="store_true", help="A3/B one-shot settled prediction (no rollout)")
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mags = [int(x) for x in a.per_comp_moves.split(",") if x != ""]
    im = Imaginer(a.ckpt, dev, ca=a.ca, oneshot=a.oneshot)
    arcade = Arcade(arc_api_key="anonymous",
                    environments_dir=str((_REPO / "environment_files").resolve()),
                    operation_mode=OperationMode.OFFLINE)
    levels = [int(x) for x in a.levels.split(",") if x != ""]
    print(f"[isg] ckpt={a.ckpt} dev={dev} stack={im.stack}", flush=True)
    for level in levels:
        pairs = []           # (imagined_win_bool, real_win_bool)
        pm_im, pm_re = [], []  # imagined / real settled paint mass (Stage-A1 secondary metric)
        real_solvable = 0    # seeds where the candidate set contained a real win
        for si in range(a.seeds):
            s = a.base_seed + int(level) * 1000 + si
            env = arcade.make(GAME_ID, seed=s)
            fd0, cur0 = _present(env, level)
            initial = cur0.copy()
            rng = np.random.default_rng(s)
            cands = _candidates(cur0, a.max_comps, mags, a.random_cands, rng)
            any_real = False
            for arrangement in cands:
                # imagined outcome
                env_i = arcade.make(GAME_ID, seed=s)
                fdi, curi = _present(env_i, level)
                fdi, arr, arr_prev = _apply_real(env_i, fdi, curi, arrangement)
                if not game_live(fdi):
                    continue
                imagined = im.imagine_pour(arr, arr_prev)
                iw = win_score(initial, imagined)["win"]
                # real outcome: reset, same arrangement, real pour
                env_r = arcade.make(GAME_ID, seed=s)
                fdr, curr = _present(env_r, level)
                base_lv = int(getattr(fdr, "levels_completed", 0) or 0)
                fdr, curr, _ = _apply_real(env_r, fdr, curr, arrangement)
                rw = False
                real_settled = curr
                if game_live(fdr):
                    for _ in range(50):  # let the real spill run to settle
                        fdr = env_r.step(ACTION_OF_ID[5])
                        if fdr is None:
                            break
                        real_settled = _grid(fdr)
                        if int(getattr(fdr, "levels_completed", 0) or 0) > base_lv:
                            rw = True; break
                        if not game_live(fdr):
                            break
                pairs.append((bool(iw), bool(rw)))
                # A1 secondary: does imagined settled paint mass match real (~96 cells)?
                pm_im.append(int((imagined != initial).sum()))
                pm_re.append(int((real_settled != initial).sum()))
                any_real = any_real or rw
            real_solvable += int(any_real)
        ip = np.array([p[0] for p in pairs], dtype=bool)
        rp = np.array([p[1] for p in pairs], dtype=bool)
        n = len(pairs)
        acc = float((ip == rp).mean()) if n else float("nan")
        tp = int((ip & rp).sum()); fp = int((ip & ~rp).sum())
        fn = int((~ip & rp).sum()); tn = int((~ip & ~rp).sum())
        print(f"[isg] level {level}: candidates_scored={n} real_wins={int(rp.sum())} "
              f"imagined_wins={int(ip.sum())} | agree_acc={acc:.3f} "
              f"tp={tp} fp={fp} fn={fn} tn={tn} | seeds_with_a_real_win={real_solvable}/{a.seeds}",
              flush=True)
        mim = float(np.mean(pm_im)) if pm_im else float("nan")
        mre = float(np.mean(pm_re)) if pm_re else float("nan")
        ratio = (mim / mre) if mre else float("nan")
        print(f"[isg] level {level}: paint_mass imagined={mim:.1f} real={mre:.1f} "
              f"ratio={ratio:.2f} (A1 secondary: want 0.8-1.2)", flush=True)
        print(f"[isg] level {level}: read => "
              f"{'imagination MISSES real wins (fn>0)' if fn else 'no missed real wins'}; "
              f"{'search set has NO winning arrangement' if rp.sum()==0 else 'winning arrangement exists in set'}",
              flush=True)


if __name__ == "__main__":
    main()
