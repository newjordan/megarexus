"""Stage A2 trainer — the local Neural-CA spill head (roadmap §4 A2).

The pour is a deterministic LOCAL cellular automaton. v38 (global-FiLM decoder
retrain) diverges on the 40-tick rollout (paint mass 6.66x). This trains ONLY the
new `spill_ca` head (encoder + all other heads frozen; C3) to predict one spill
tick from the 16-colour one-hot grid, with the loss MASKED to cells at/near the
active front (changed cells dilated radius 2) — so it learns the exact local rule
(fall / split-on-block / deflect-on-redirect / fill-on-U / fail-on-wall) and nothing
else. A correctly-learned local rule is LEVEL-INVARIANT: it generalises to L1+ with
no L1+ positives. Trained on spill-tick pairs only (pour start ACTION5 + animation
ACTION0), extracted from the harvested pour shards. Legal: pure forward-physics
supervision from observed env transitions; no answers, no route lineage.

    python3 synthesis_lab/orchestration/train_spill_ca.py \
        --init-ckpt <v31>/best.pt --shards <pour_tr.pt> <pour_v2_l0.pt> \
        --out <run>/best.pt --epochs 30 --lr 3e-4
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
from no_imitation_guard import assert_no_imitation_lineage  # noqa: E402
from pour_settle import settle_index  # noqa: E402

G, C = 64, 16


def extract_pairs(shards):
    """(cur_grid, next_grid) int8 spill-tick pairs, restricted to the CLEAN pour
    window [pour_idx, settle_idx] — the monotonic spill growth plus the 11->13
    fill-on-contact. Frames AFTER settle are the win-flash + level RESET (13->11,
    paint cleared); including them (the old bug) taught the CA to UN-satisfy targets
    and to flood forever. One explicit fixed-point pair (settled->settled) per pour
    teaches the CA to STOP at the plateau. NIGHT 2026-07-05 fix."""
    curs, nxts, nwin = [], [], 0
    for path in shards:
        for traj in torch.load(path, weights_only=False):
            steps = traj["steps"]
            acts = [int(s.get("action_id", -1)) for s in steps]
            if 5 not in acts:
                continue
            p = max(i for i, a in enumerate(acts) if a == 5)
            s_idx = settle_index(steps)
            if s_idx <= p:
                continue
            for i in range(p, s_idx):
                ca = np.asarray(steps[i]["frame"], dtype=np.int8)
                cb = np.asarray(steps[i + 1]["frame"], dtype=np.int8)
                if ca.shape == (G, G) and cb.shape == (G, G):
                    curs.append(ca); nxts.append(cb)
            sf = np.asarray(steps[s_idx]["frame"], dtype=np.int8)  # fixed point: hold
            if sf.shape == (G, G):
                curs.append(sf); nxts.append(sf)
                nwin += int((sf == 13).any())
    print(f"[ca] extracted {len(curs)} clean spill pairs; {nwin} pours reach a 13-settle", flush=True)
    return np.stack(curs), np.stack(nxts)


def onehot(batch_i64):
    return F.one_hot(batch_i64.clamp(0, C - 1), C).permute(0, 3, 1, 2).float()


PAINT, SOURCE = 6, 4


def active_mask(cur_i64, nxt_i64):
    """Supervise the active front = dilation(r2) of paint(6) + source(4) + any
    changed cell. Including the paint region even when nothing changes is what
    teaches the CA to HOLD a settled spill (reproduce paint, emit none) — the
    missing 'stop' behaviour. Matches the gate's rollout mask exactly."""
    seed = ((cur_i64 == PAINT) | (cur_i64 == SOURCE) | (cur_i64 != nxt_i64)).float().unsqueeze(1)
    return (F.max_pool2d(seed, 5, stride=1, padding=2) > 0).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    assert_no_imitation_lineage(list(a.shards), purpose="spill_ca train shards")
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(a.init_ckpt, map_location=dev, weights_only=False)
    stack = int(ck["config"].get("stack_frames", ck["config"].get("n_input_frames", 2)))
    m = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)).to(dev)
    m.load_state_dict(ck["model"])
    for n, p in m.named_parameters():
        p.requires_grad_(n.startswith("spill_ca."))   # C3: only the new head trains
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=a.lr)

    cur, nxt = extract_pairs(a.shards)
    n = len(cur)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    nval = int(n * a.val_frac)
    va, tr = perm[:nval], perm[nval:]
    print(f"[ca] spill-tick pairs: {n} (train {len(tr)} / val {len(va)}) from {len(a.shards)} shards", flush=True)
    cur_t = torch.from_numpy(cur.astype(np.int64))
    nxt_t = torch.from_numpy(nxt.astype(np.int64))

    def run_batch(idx, train):
        ci = cur_t[idx].to(dev); ni = nxt_t[idx].to(dev)
        logits = m.forward_spill_ca(onehot(ci))                 # [B,16,64,64]
        mask = active_mask(ci, ni)                              # [B,64,64]
        ce = F.cross_entropy(logits, ni, reduction="none")      # [B,64,64]
        # upweight the RARE level-determining transitions the win-check keys on:
        # a cell becoming a satisfied target (13) or a hit wall (14). These occur
        # only in winning/failing pours (~3% of L0) so plain CE ignores them.
        w = mask * (1.0 + 30.0 * ((ni == 13) | (ni == 14)).float())
        loss = (ce * w).sum() / w.sum().clamp(min=1)
        # front accuracy: argmax correct on the masked (active) cells
        acc = ((logits.argmax(1) == ni).float() * mask).sum() / mask.sum().clamp(min=1)
        # recall of the win-transition (did we predict the 13 cells that exist?)
        t13 = (ni == 13)
        f13 = ((logits.argmax(1) == 13) & t13).sum() / t13.sum().clamp(min=1)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        return float(loss), float(acc), float(f13)

    best = 1e9
    for ep in range(a.epochs):
        m.train()
        rng.shuffle(tr)
        tl = []
        for i in range(0, len(tr), a.batch_size):
            l, _, _ = run_batch(tr[i:i + a.batch_size], True); tl.append(l)
        m.eval()
        with torch.no_grad():
            vl, vacc, vf13 = [], [], []
            for i in range(0, len(va), a.batch_size):
                l, ac, f13 = run_batch(va[i:i + a.batch_size], False)
                vl.append(l); vacc.append(ac); vf13.append(f13)
        vloss = float(np.mean(vl)); vfront = float(np.mean(vacc))
        print(f"[ca] epoch {ep:02d} train_loss={np.mean(tl):.4f} val_loss={vloss:.4f} "
              f"val_front_acc={vfront:.4f} val_fill13_recall={float(np.mean(vf13)):.4f}", flush=True)
        if vloss < best:
            best = vloss
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": m.state_dict(), "config": ck["config"],
                        "stage": "A2_spill_ca", "val_loss": vloss, "val_front_acc": vfront}, a.out)
    print(f"[ca] done. best val_loss={best:.4f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
