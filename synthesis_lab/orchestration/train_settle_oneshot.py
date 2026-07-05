"""Stage A3/B — ONE-SHOT settle predictor (no rollout).

The autoregressive rollout (forward_dynamics ticks OR the spill_ca) is too unstable:
small per-tick errors compound and drift the geometry so the exact win-transition
(11->13) never triggers in imagination (NIGHT FINDING 11-15). Fix: don't roll out.
Predict the SETTLED frame DIRECTLY from the pre-pour arrangement in a single forward
pass. The settled state depends on the WHOLE arrangement, so the global FiLM decoder
(which only failed at rollout) is the right tool for a one-shot global prediction —
and one pass cannot compound. Trained on (arrangement, settled=last animation frame)
pairs; full-frame CE (the whole settled frame matters, incl. 13/14). Encoder frozen
(C3). Legal: forward-physics supervision from observed pours; no answers/routes.

    python3 synthesis_lab/orchestration/train_settle_oneshot.py \
        --init-ckpt <v31>/best.pt --shards <pour_v2_l0.pt> \
        --out <run>/best.pt --epochs 40 --lr 3e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_LAB = Path(__file__).resolve().parents[1]
_REPO = _LAB.parent
for p in (_LAB, _LAB / "orchestration", _LAB / "experiments" / "megarexus_cnn"):
    sys.path.insert(0, str(p))

from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402
from eval_ghost import encode  # noqa: E402
from no_imitation_guard import assert_no_imitation_lineage  # noqa: E402
from pour_settle import settle_frame, pour_arrangement, is_win  # noqa: E402

G = 64
# forward_dynamics decoder path (encoder/trunk frozen — C3)
DECODER_PREFIXES = ("dynamics", "up1", "up2", "skip_gate", "head_next_frame",
                    "action_embed", "click_plane_proj")


def extract_settle_pairs(shards):
    """(arrangement_frame, settled_frame) — one per pour: the ACTION5 pre-pour frame
    and the SETTLED pour-outcome frame. The settled frame is the paint-plateau frame
    (pour_settle.settle_frame), NOT steps[-1] which is post-reset garbage — that bug
    starved every prior imaginer of win transitions (NIGHT 2026-07-05)."""
    arrs, setts, nwin = [], [], 0
    for path in shards:
        for traj in torch.load(path, weights_only=False):
            steps = traj["steps"]
            if not any(int(s.get("action_id", -1)) == 5 for s in steps):
                continue
            arr = pour_arrangement(steps)
            settled = settle_frame(steps)
            if arr.shape == (G, G) and settled.shape == (G, G):
                arrs.append(arr); setts.append(settled); nwin += int(is_win(settled))
    print(f"[os] extracted {len(arrs)} pairs, {nwin} complete-win settled labels", flush=True)
    return np.stack(arrs), np.stack(setts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    assert_no_imitation_lineage(list(a.shards), purpose="settle-oneshot shards")
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(a.init_ckpt, map_location=dev, weights_only=False)
    stack = int(ck["config"].get("stack_frames", ck["config"].get("n_input_frames", 2)))
    m = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)).to(dev)
    m.load_state_dict(ck["model"])
    for n, p in m.named_parameters():
        p.requires_grad_(n.startswith(DECODER_PREFIXES))   # C3: decoder only
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=a.lr)

    arr, sett = extract_settle_pairs(a.shards)
    n = len(arr)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    nval = int(n * a.val_frac)
    va, tr = perm[:nval], perm[nval:]
    print(f"[os] settle pairs: {n} (train {len(tr)} / val {len(va)})", flush=True)
    arr_t = torch.from_numpy(arr.astype(np.int64))
    sett_t = torch.from_numpy(sett.astype(np.int64))

    def inp(a_i64):
        # 2-frame stack of the arrangement with itself (no prior frame pre-pour)
        oh = [torch.cat([encode(g.numpy().astype(np.int8)), encode(g.numpy().astype(np.int8))], 0)
              if stack == 2 else encode(g.numpy().astype(np.int8)) for g in a_i64]
        return torch.stack(oh).to(dev)

    def run(idx, train):
        ai = arr_t[idx]; si = sett_t[idx].to(dev)
        B = len(idx)
        logits = m.forward_dynamics(inp(ai),
                                    torch.full((B,), 5, device=dev),
                                    torch.zeros(B, dtype=torch.long, device=dev),
                                    torch.zeros(B, dtype=torch.long, device=dev))
        loss = F.cross_entropy(logits, si)
        acc = (logits.argmax(1) == si).float().mean()
        t13 = (si == 13)
        f13 = ((logits.argmax(1) == 13) & t13).sum() / t13.sum().clamp(min=1)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        return float(loss), float(acc), float(f13)

    best = 1e9
    for ep in range(a.epochs):
        m.train(); rng.shuffle(tr); tl = []
        for i in range(0, len(tr), a.batch_size):
            l, _, _ = run(tr[i:i + a.batch_size], True); tl.append(l)
        m.eval()
        with torch.no_grad():
            vl, vacc, vf = [], [], []
            for i in range(0, len(va), a.batch_size):
                l, ac, f = run(va[i:i + a.batch_size], False)
                vl.append(l); vacc.append(ac); vf.append(f)
        vloss = float(np.mean(vl))
        print(f"[os] epoch {ep:02d} train_loss={np.mean(tl):.4f} val_loss={vloss:.4f} "
              f"val_acc={np.mean(vacc):.4f} val_fill13_recall={np.mean(vf):.4f}", flush=True)
        if vloss < best:
            best = vloss
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": m.state_dict(), "config": ck["config"],
                        "stage": "A3_settle_oneshot", "val_loss": vloss}, a.out)
    print(f"[os] done. best val_loss={best:.4f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
