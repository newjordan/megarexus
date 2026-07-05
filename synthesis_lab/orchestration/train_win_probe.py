"""Stage B (roadmap §Stage-B) — latent settle-predictor + win-probe (the JEPA layer).

Both pixel imaginers fail on L0 with CORRECT labels (NIGHT FINDING 16-17): the global
one-shot over-floods and can't discriminate (v44: fp=21, tp=0), the local CA rollout
drifts and never triggers the win (v45: imagined_wins=0). And the existing pour-outcome
head (forward_settle level_logit) is SATURATED on this data (AUC 0.500, mean prob 1.0
for both classes). So: don't predict pixels — predict the SETTLED-frame LATENT in one
pass and classify it. Latents RANK; the geometric pixel check stays the decider.

  settle_predictor: pooled trunk latent of the pre-pour arrangement -> predicted
    pooled latent of the real settled frame. Self-supervised on EVERY pour (win or
    lose), JEPA-style: stop-grad on the target embedding, cosine loss. No positives.
  win_probe: small MLP on the PREDICTED settled latent -> P(win). Trained on the
    env-graded outcome (is_win of the settle frame = all targets 11->13). L0 positives
    + all negatives.

Encoder/trunk frozen (C3). Legal: forward-physics + env-graded outcomes from observed
pours; no answers, no route lineage. Reports deployable AUC (probe on predicted latent)
AND oracle AUC (probe on the TRUE settled latent) to locate the bottleneck.

    python3 synthesis_lab/orchestration/train_win_probe.py \
        --init-ckpt <v31>/best.pt --shards <pour_v2_l0.pt> \
        --out <run>/best.pt --epochs 60 --lr 3e-4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_LAB = Path(__file__).resolve().parents[1]
for p in (_LAB, _LAB / "orchestration", _LAB / "experiments" / "megarexus_cnn"):
    sys.path.insert(0, str(p))

from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402
from eval_ghost import encode  # noqa: E402
from no_imitation_guard import assert_no_imitation_lineage  # noqa: E402
from pour_settle import settle_frame, pour_arrangement, is_win  # noqa: E402

G = 64


def auc(scores, labels):
    """Rank AUC (Mann-Whitney), no sklearn. scores/labels are 1-D np arrays."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    assert_no_imitation_lineage(list(a.shards), purpose="win-probe shards")
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(a.init_ckpt, map_location=dev, weights_only=False)
    stack = int(ck["config"].get("stack_frames", ck["config"].get("n_input_frames", 2)))
    m = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()  # trunk frozen; only settle_predictor + win_probe train
    for n, p in m.named_parameters():
        p.requires_grad_(n.startswith(("settle_predictor.", "win_probe.")))

    def inp(frame_i8):
        oh = encode(frame_i8)
        t = torch.cat([oh, oh], 0) if stack == 2 else oh
        return t.unsqueeze(0).to(dev)

    # --- precompute frozen-trunk latents once (arr latent, settled latent, label) ---
    # Accept BOTH shard formats: trajectory shards ({"steps":[...]}) where the
    # settle frame is extracted, and flat ON-DISTRIBUTION shards from
    # harvest_pour_shard ({"arrangement","settled","env_win"}) already env-graded.
    arr_z, set_z, y = [], [], []
    with torch.no_grad():
        for path in a.shards:
            for rec in torch.load(path, weights_only=False):
                if "steps" in rec:
                    steps = rec["steps"]
                    if not any(int(s.get("action_id", -1)) == 5 for s in steps):
                        continue
                    arr = pour_arrangement(steps)
                    sett = settle_frame(steps)
                    label = int(is_win(sett))
                else:  # flat on-distribution record
                    arr = np.asarray(rec["arrangement"], dtype=np.int8)
                    sett = np.asarray(rec["settled"], dtype=np.int8)
                    label = int(bool(rec["env_win"]))
                if arr.shape != (G, G) or sett.shape != (G, G):
                    continue
                arr_z.append(m.encode_latent(inp(arr))[0].cpu())
                set_z.append(m.encode_latent(inp(sett))[0].cpu())
                y.append(label)
    arr_z = torch.stack(arr_z).to(dev)
    set_z = torch.stack(set_z).to(dev)
    y = torch.tensor(y, dtype=torch.float32, device=dev)
    n = len(y)
    pos = int(y.sum())
    print(f"[wp] {n} pours, {pos} wins ({100*pos/n:.1f}%); latent dim {arr_z.shape[1]}", flush=True)

    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    nval = int(n * a.val_frac)
    va = torch.tensor(perm[:nval], device=dev)
    tr = perm[nval:]
    pos_weight = torch.tensor([(n - pos) / max(pos, 1)], device=dev)

    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=a.lr)

    def batch_scores(idx):
        za = arr_z[idx]
        zpred = m.settle_predictor(za)
        wl_pred = m.win_probe(zpred).squeeze(-1)
        wl_oracle = m.win_probe(set_z[idx]).squeeze(-1)  # probe on TRUE settled latent
        cos = F.cosine_similarity(zpred, set_z[idx].detach(), dim=1)
        return zpred, wl_pred, wl_oracle, cos

    best = -1.0
    for ep in range(a.epochs):
        m.settle_predictor.train(); m.win_probe.train()
        rng.shuffle(tr)
        for i in range(0, len(tr), a.batch_size):
            idx = torch.tensor(tr[i:i + a.batch_size], device=dev)
            zpred, wl_pred, wl_oracle, cos = batch_scores(idx)
            settle_loss = (1.0 - cos).mean()
            # win loss on BOTH predicted and true-settled latent so the probe is
            # a good classifier of the latent AND the predictor targets a
            # win-separable point. Deployment uses win_probe(predicted).
            win_loss = F.binary_cross_entropy_with_logits(wl_pred, y[idx], pos_weight=pos_weight) \
                + F.binary_cross_entropy_with_logits(wl_oracle, y[idx], pos_weight=pos_weight)
            loss = settle_loss + win_loss
            opt.zero_grad(); loss.backward(); opt.step()
        # --- val ---
        m.settle_predictor.eval(); m.win_probe.eval()
        with torch.no_grad():
            _, wlp, wlo, cos = batch_scores(va)
            yv = y[va].cpu().numpy()
            auc_pred = auc(torch.sigmoid(wlp).cpu().numpy(), yv)
            auc_oracle = auc(torch.sigmoid(wlo).cpu().numpy(), yv)
            vcos = float(cos.mean())
        print(f"[wp] epoch {ep:02d} settle_cos={vcos:.3f} "
              f"val_AUC_pred={auc_pred:.3f} val_AUC_oracle={auc_oracle:.3f}", flush=True)
        if auc_pred > best:
            best = auc_pred
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": m.state_dict(), "config": ck["config"],
                        "stage": "B_win_probe", "val_auc_pred": auc_pred,
                        "val_auc_oracle": auc_oracle}, a.out)
    print(f"[wp] done. best val_AUC_pred={best:.3f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
