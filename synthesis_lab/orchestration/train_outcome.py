"""Train the SHAPE-based, LAVA-aware pour-outcome imagination on RANDOMISED variant boards
(NIGHT FINDING 29-30, user's "move the blocks to force skill" idea).

Predicts (fill, lava) from an arrangement — the level-invariant objective: fill = fraction
of targets that settle satisfied (POSITIVE), lava = wall/lava contact (NEGATIVE, "floor is
lava"). Conv-then-flatten head keeps spatial layout; encoder frozen (C3). Trained on 600
random layouts, HELD-OUT by variant so validation layouts are unseen. Gate = win-AUC on
held-out variants AND on the REAL unmodified L0/L1/L2 levels (does routing skill transfer?).
Legal: varied problems, env-graded outcomes, real levels never trained on.

    python3 synthesis_lab/orchestration/train_outcome.py \
        --init-ckpt <v31>/best.pt --variants <pour_variants.pt> \
        --real <pour_filllava_l012.pt> --out <run>/best.pt --epochs 60
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F

_LAB = Path(__file__).resolve().parents[1]
for p in (_LAB, _LAB / "orchestration", _LAB / "experiments" / "megarexus_cnn"):
    sys.path.insert(0, str(p))
from model import MegarexusCNN, MegarexusCNNConfig  # noqa: E402
from eval_ghost import encode  # noqa: E402

G = 64
LAVA_SCALE = 256.0


def auc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels).astype(bool)
    p, n = scores[labels], scores[~labels]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--real", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-variant-frac", type=float, default=0.2)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.init_ckpt, map_location=dev, weights_only=False)
    stack = int(ck["config"].get("stack_frames", ck["config"].get("n_input_frames", 2)))
    m = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    for n, p in m.named_parameters():
        p.requires_grad_(n.startswith("outcome_head."))
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=a.lr)

    def load(path):
        arrs, fill, lava, win, var = [], [], [], [], []
        for r in torch.load(path, weights_only=False):
            g = np.asarray(r["arrangement"], dtype=np.int8)
            if g.shape != (G, G):
                continue
            arrs.append(g); fill.append(float(r.get("fill", r.get("sustained_fill", 0.0))))
            lava.append(float(r["lava"])); win.append(int(r["env_win"]))
            var.append(int(r.get("variant", r.get("level", -1))))
        return (np.stack(arrs), np.array(fill, np.float32), np.array(lava, np.float32),
                np.array(win), np.array(var))

    A, FILL, LAVA, WIN, VAR = load(a.variants)
    print(f"[out] {len(A)} variant pours, {int(WIN.sum())} wins, {len(set(VAR.tolist()))} variants", flush=True)
    uniq = sorted(set(VAR.tolist()))
    rng = np.random.default_rng(0); rng.shuffle(uniq)
    nval = int(len(uniq) * a.val_variant_frac)
    val_vars = set(uniq[:nval])
    is_val = np.array([v in val_vars for v in VAR])
    tr_idx = np.nonzero(~is_val)[0]; va_idx = np.nonzero(is_val)[0]
    print(f"[out] train {len(tr_idx)} pours / val {len(va_idx)} pours (held-out variants {nval})", flush=True)

    fill_t = torch.from_numpy(FILL).to(dev)
    lava_t = torch.from_numpy(np.clip(LAVA, 0, LAVA_SCALE) / LAVA_SCALE).to(dev)

    def precompute_enc(arr_np):
        """[N,64,64] int8 -> [N,22,64,64] float16 CPU (single-frame encode incl. Contra
        planes), computed ONCE (float16 to fit RAM; ~2.8GB for all). Epochs then avoid
        recomputing connected-components per sample."""
        encs = []
        for i in range(len(arr_np)):
            encs.append(encode(arr_np[i].astype(np.int8)).half())
            if (i + 1) % 2000 == 0:
                print(f"[out] precompute enc {i+1}/{len(arr_np)}", flush=True)
        return torch.stack(encs)  # CPU float16

    ENC = precompute_enc(A)
    print(f"[out] precomputed encodings {tuple(ENC.shape)} {ENC.dtype}", flush=True)

    def inp(idx, enc=None):
        e = (enc if enc is not None else ENC)[idx]
        return (torch.cat([e, e], 1) if stack == 2 else e).to(dev).float()

    def run_scores(idx):
        out = []
        for i in range(0, len(idx), a.batch_size):
            b = idx[i:i + a.batch_size]
            out.append(m.predict_outcome(inp(b)).detach())
        return torch.cat(out) if out else torch.zeros(0, 2, device=dev)

    # real-level eval data (precompute its encodings once too)
    real = None
    if a.real:
        RA, RFILL, RLAVA, RWIN, RLV = load(a.real)
        RENC = precompute_enc(RA)
        real = (RENC, RWIN, RLV)

    best = -1.0
    for ep in range(a.epochs):
        m.outcome_head.train()
        rng.shuffle(tr_idx)
        for i in range(0, len(tr_idx), a.batch_size):
            b = tr_idx[i:i + a.batch_size]
            logits = m.predict_outcome(inp(b))
            loss = F.binary_cross_entropy_with_logits(logits[:, 0], fill_t[torch.from_numpy(b).to(dev)]) \
                + F.binary_cross_entropy_with_logits(logits[:, 1], lava_t[torch.from_numpy(b).to(dev)])
            opt.zero_grad(); loss.backward(); opt.step()
        m.outcome_head.eval()
        with torch.no_grad():
            vs = run_scores(va_idx).cpu().numpy()
            fh, lh = 1 / (1 + np.exp(-vs[:, 0])), 1 / (1 + np.exp(-vs[:, 1]))
            score = fh - a.lam * lh
            vwin = WIN[va_idx]
            va_auc = auc(score, vwin)
            va_lava_auc = auc(-lh, vwin)
        msg = f"[out] epoch {ep:02d} HELDOUT-VARIANT win_AUC={va_auc:.3f} (lava-only {va_lava_auc:.3f})"
        if real is not None and (ep % 5 == 0 or ep == a.epochs - 1):
            RENC, RWIN, RLV = real
            with torch.no_grad():
                ri = np.arange(len(RENC))
                rs = []
                for i in range(0, len(ri), a.batch_size):
                    rs.append(m.predict_outcome(inp(ri[i:i + a.batch_size], enc=RENC)).cpu())
                rs = torch.cat(rs).numpy()
                rfh, rlh = 1 / (1 + np.exp(-rs[:, 0])), 1 / (1 + np.exp(-rs[:, 1]))
                rscore = rfh - a.lam * rlh
            for lv in sorted(set(RLV.tolist())):
                mk = RLV == lv
                if RWIN[mk].sum() > 0:
                    msg += f" | REAL-L{lv} win_AUC={auc(rscore[mk], RWIN[mk]):.3f}"
                else:
                    msg += f" | REAL-L{lv} (0wins, lava_hat mean={rlh[mk].mean():.2f})"
        print(msg, flush=True)
        if va_auc > best:
            best = va_auc
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": m.state_dict(), "config": ck["config"],
                        "stage": "outcome_lava_variants", "val_win_auc": va_auc}, a.out)
    print(f"[out] done. best HELDOUT-VARIANT win_AUC={best:.3f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
