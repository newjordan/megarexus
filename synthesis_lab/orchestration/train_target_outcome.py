"""Train the PER-TARGET pour-outcome head (STAGE_PT_SPEC §4, T2). Clones
train_outcome.py's shard/loader/eval conventions; predicts (fill_t, lava_t) for ONE
target U at a time and ranks arrangements by min_t(sig(fill_t) - sig(lava_t)) so a loss
is rejected for the SPECIFIC target it misses (the aggregate scalar cannot — F38-40).

Two pre-registered variants (§4):
  (A) frozen   : init from v52 best.pt, train head_target_outcome ONLY. Frozen trunk ->
                 precompute h16 maps float16 ONCE (fast). Default.
  (B) unfrozen : --unfreeze-encoder (enc-lr 3e-5), NEW ckpt dir. Recomputes trunk/h16
                 each batch. Only if (A) fails G1.

Labels come from a §3 per-target shard (harvest_variants --per-target): each record has
`arrangement` (int8 pre-pour frame, for masks) and `pt` = [[fill_t, lavan_t], ...] in
frame_components order. Real levels NEVER trained on (held out; G1/G3 eval only).

    python3 synthesis_lab/orchestration/train_target_outcome.py \
        --init-ckpt <v52>/best.pt --variants <pt_shard.pt> \
        --out <run>/best.pt --epochs 60           # variant A
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
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from pertarget_labels import _dilate_bbox, TARGET_COLOR  # noqa: E402

G = 64


def auc(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels).astype(bool)
    p, n = scores[labels], scores[~labels]
    if len(p) == 0 or len(n) == 0:
        return float("nan")
    return float((p[:, None] > n[None, :]).mean() + 0.5 * (p[:, None] == n[None, :]).mean())


def target_masks_16(arr):
    """[64,64] int8 arrangement -> (tmask16 [T,16,16], rmask16 [T,16,16]) in
    frame_components(color==11) order (matches the shard's `pt` label order)."""
    comps = [c for c in frame_components(arr) if int(c["color"]) == TARGET_COLOR]
    tms, rms = [], []
    for c in comps:
        tm = np.asarray(c["mask"], dtype=np.float32)                 # [64,64]
        ring = _dilate_bbox(c["bbox_xyxy"], 3).astype(np.float32)    # bbox dilated R=3
        ring = np.clip(ring - tm, 0, 1)                              # minus target (§4)
        tm16 = F.max_pool2d(torch.from_numpy(tm)[None, None], 4)[0, 0]
        rm16 = F.max_pool2d(torch.from_numpy(ring)[None, None], 4)[0, 0]
        tms.append(tm16); rms.append(rm16)
    if not tms:
        return None, None
    return torch.stack(tms), torch.stack(rms)      # [T,16,16] each


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", required=True)
    ap.add_argument("--variants", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)   # target examples per batch
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-variant-frac", type=float, default=0.2)
    ap.add_argument("--unfreeze-encoder", action="store_true")
    ap.add_argument("--encoder-lr", type=float, default=3e-5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    ck = torch.load(a.init_ckpt, map_location=dev, weights_only=False)
    stack = int(ck["config"].get("stack_frames", ck["config"].get("n_input_frames", 2)))
    m = MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack)).to(dev)
    m.load_state_dict(ck["model"]); m.eval()
    trainable = ("head_target_outcome.", "trunk.") if a.unfreeze_encoder else ("head_target_outcome.",)
    for n, p in m.named_parameters():
        p.requires_grad_(n.startswith(trainable))
    if a.unfreeze_encoder:
        head_p = [p for n, p in m.named_parameters() if p.requires_grad and n.startswith("head_target_outcome.")]
        enc_p = [p for n, p in m.named_parameters() if p.requires_grad and n.startswith("trunk.")]
        opt = torch.optim.Adam([{"params": head_p, "lr": a.lr}, {"params": enc_p, "lr": a.encoder_lr}])
    else:
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=a.lr)

    # ---- load per-target shard(s) -> per-pour records --------------------------
    pours = []   # each: dict(arr, tm16 [T,16,16], rm16, fill [T], lavan [T], win, var)
    off = 0
    for path in a.variants:
        recs = torch.load(path, weights_only=False)
        vmax = -1
        for r in recs:
            arr = np.asarray(r["arrangement"], dtype=np.int8)
            pt = r.get("pt")
            if arr.shape != (G, G) or not pt:
                continue
            tm16, rm16 = target_masks_16(arr)
            if tm16 is None or tm16.shape[0] != len(pt):
                continue   # mask/label count mismatch -> skip (keeps mask<->label aligned)
            fill = np.array([p[0] for p in pt], np.float32)
            lavan = np.array([p[1] for p in pt], np.float32)
            var = int(r.get("variant", -1)) + off
            vmax = max(vmax, int(r.get("variant", -1)))
            pours.append({"arr": arr, "tm16": tm16, "rm16": rm16, "fill": fill,
                          "lavan": lavan, "win": int(r["env_win"]), "var": var})
        off += vmax + 1
    print(f"[pt] {len(pours)} pours, {sum(p['win'] for p in pours)} wins, "
          f"{len(set(p['var'] for p in pours))} variants, {sum(len(p['fill']) for p in pours)} targets", flush=True)

    # held-out split by variant (val layouts unseen)
    rng = np.random.default_rng(a.seed)
    uniq = sorted(set(p["var"] for p in pours)); rng.shuffle(uniq)
    val_vars = set(uniq[:int(len(uniq) * a.val_variant_frac)])
    tr = [i for i, p in enumerate(pours) if p["var"] not in val_vars]
    va = [i for i, p in enumerate(pours) if p["var"] in val_vars]
    print(f"[pt] train {len(tr)} pours / val {len(va)} pours (held-out variants {len(val_vars)})", flush=True)

    # ---- precompute per-pour input encoding (and h16 if frozen) ---------------
    def enc_pour(arr):
        e = encode(arr.astype(np.int8))
        return (torch.cat([e, e], 0) if stack == 2 else e)      # [C,64,64]

    print(f"[pt] encoding {len(pours)} pours (CPU)...", flush=True)
    e0 = enc_pour(pours[0]["arr"]).half()
    ENC = torch.empty((len(pours),) + tuple(e0.shape), dtype=torch.float16); ENC[0] = e0
    for i in range(1, len(pours)):
        ENC[i] = enc_pour(pours[i]["arr"]).half()
        if (i + 1) % 2000 == 0: print(f"[pt]   enc {i+1}/{len(pours)}", flush=True)
    H16 = None
    if not a.unfreeze_encoder:
        # frozen trunk -> batch-forward h16 ONCE on GPU, then drop encodings (saves RAM)
        with torch.no_grad():
            hs = []
            for i in range(0, len(pours), 256):
                hs.append(m.trunk(ENC[i:i + 256].float().to(dev))[0].half().cpu())
            H16 = torch.cat(hs)
        ENC = None
    print(f"[pt] precompute done ({'h16 '+str(tuple(H16.shape)) if H16 is not None else 'enc '+str(tuple(ENC.shape))})", flush=True)

    # flattened target-example index: (pour_idx, target_idx)
    def examples(pidx):
        out = []
        for i in pidx:
            for t in range(len(pours[i]["fill"])):
                out.append((i, t))
        return out
    tr_ex = examples(tr)

    def h16_for(pidx_batch):
        """[B, lat, 16, 16] for a batch of pour indices (recompute if unfrozen)."""
        if a.unfreeze_encoder:
            x = ENC[pidx_batch].float().to(dev)
            return m.trunk(x)[0]
        return H16[pidx_batch].float().to(dev)

    def forward_examples(exs, train):
        """exs: list of (pour_idx, target_idx). Returns logits [B,2], fill [B], lavan [B]."""
        pidx = [e[0] for e in exs]; tidx = [e[1] for e in exs]
        h16 = h16_for(pidx)                                  # [B,lat,16,16]
        tm = torch.stack([pours[i]["tm16"][t] for i, t in exs]).unsqueeze(1).to(dev)  # [B,1,16,16]
        rm = torch.stack([pours[i]["rm16"][t] for i, t in exs]).unsqueeze(1).to(dev)
        logits = m.target_outcome_from_h16(h16, tm, rm)[:, 0]  # T=1 -> [B,2]
        fill = torch.tensor([pours[i]["fill"][t] for i, t in exs], device=dev)
        lavan = torch.tensor([pours[i]["lavan"][t] for i, t in exs], device=dev)
        return logits, fill, lavan

    @torch.no_grad()
    def eval_val():
        """Held-out arr_score win-AUC (min-aggregated per pour) + per-target fill/lava AUC."""
        m.eval()
        pour_score, pour_win = [], []
        f_pred, f_lab, l_pred, l_lab = [], [], [], []
        for i in va:
            exs = [(i, t) for t in range(len(pours[i]["fill"]))]
            logits, fill, lavan = forward_examples(exs, False)
            fh = torch.sigmoid(logits[:, 0]); lh = torch.sigmoid(logits[:, 1])
            pour_score.append(float((fh - lh).min()))       # arr_score = min_t
            pour_win.append(pours[i]["win"])
            f_pred += fh.cpu().tolist(); f_lab += fill.cpu().tolist()
            l_pred += lh.cpu().tolist(); l_lab += lavan.cpu().tolist()
        win_auc = auc(pour_score, pour_win)
        fill_auc = auc(f_pred, np.array(f_lab) >= 0.95)
        lava_auc = auc(np.array(l_pred), np.array(l_lab) >= 0.5)
        return win_auc, fill_auc, lava_auc

    best = -1.0
    for ep in range(a.epochs):
        m.train() if a.unfreeze_encoder else m.head_target_outcome.train()
        rng.shuffle(tr_ex)
        for i in range(0, len(tr_ex), a.batch_size):
            exs = tr_ex[i:i + a.batch_size]
            logits, fill, lavan = forward_examples(exs, True)
            loss = F.binary_cross_entropy_with_logits(logits[:, 0], fill) \
                + F.binary_cross_entropy_with_logits(logits[:, 1], lavan)
            opt.zero_grad(); loss.backward(); opt.step()
        win_auc, fill_auc, lava_auc = eval_val()
        print(f"[pt] epoch {ep:02d} HELDOUT win_AUC(arr_score)={win_auc:.3f} "
              f"fill_AUC={fill_auc:.3f} lava_AUC={lava_auc:.3f}", flush=True)
        if win_auc > best:
            best = win_auc
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": m.state_dict(), "config": ck["config"],
                        "stage": "target_outcome_pertarget", "val_win_auc": win_auc}, a.out)
    print(f"[pt] done. best HELDOUT win_AUC(arr_score)={best:.3f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
