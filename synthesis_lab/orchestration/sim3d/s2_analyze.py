"""Merge S2 sweep fixtures and print the honest generalization verdict.

Reports the pre-registered gate (win-sign, AUC, S0) AND the metrics that actually
carry weight under the heavy class imbalance the sweep exhibits (win rate <1%):
precision/recall of the sim's win-call, and per-board argmax on the boards where
discrimination is genuinely required (a true winner coexisting with overshoot-FPs).

    python3 s2_analyze.py fixtures/s2_variants_20260705.npz fixtures/s2_variants_20260705_b.npz
"""
from __future__ import annotations

import sys
from collections import defaultdict

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES


def auc_score(scores, labels):
    scores = np.asarray(scores, float); labels = np.asarray(labels, bool)
    n_pos = int(labels.sum()); n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort"); s = scores[order]
    ranks = np.empty(len(scores)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main():
    paths = sys.argv[1:] or [str(FIXTURES / "s2_variants_20260705.npz")]
    sm, win, base, var, rl = [], [], [], [], []
    board_off = 0
    for p in paths:
        z = np.load(p, allow_pickle=True)
        sm.append(z["sim_margin"]); win.append(z["env_win"]); base.append(z["base"])
        # offset variant ids per batch so boards from different seed-ranges stay distinct
        var.append(z["variant"].astype(np.int64) + board_off)
        rl.append(z["real_lava"]); board_off += 100000
    sm = np.concatenate(sm); win = np.concatenate(win).astype(bool)
    base = np.concatenate(base); var = np.concatenate(var); rl = np.concatenate(rl)

    n = len(sm); nw = int(win.sum()); pred = sm > 0
    TP = int(np.sum(pred & win)); FP = int(np.sum(pred & ~win))
    FN = int(np.sum(~pred & win)); TN = int(np.sum(~pred & ~win))
    win_sign = float(np.mean(pred == win)); auc = auc_score(sm, win)

    boards = defaultdict(list)
    for i in range(n):
        boards[int(var[i])].append((sm[i], bool(win[i])))
    winnable = [v for v, r in boards.items() if any(w for _, w in r)]
    # boards where argmax is NON-trivial: a true win AND >=1 FP present
    nontrivial = [v for v in winnable
                  if any((s > 0 and not w) for s, w in boards[v])]
    argmax_hits = sum(int(max(boards[v], key=lambda r: r[0])[1]) for v in winnable)
    nontrivial_hits = sum(int(max(boards[v], key=lambda r: r[0])[1]) for v in nontrivial)

    print(f"=== S2 MERGED ({len(paths)} batch(es), {n} pours) ===")
    print(f"win rate            : {nw}/{n} = {nw/n:.4f}")
    print(f"S0 always-loss base : {1-nw/n:.4f}  (win-sign only meaningful ABOVE this)")
    print(f"GATE win-sign       : {win_sign:.4f}   (>=0.95 {'PASS' if win_sign>=0.95 else 'FAIL'})")
    print(f"GATE margin AUC     : {auc:.4f}   (>=0.98 {'PASS' if auc>=0.98 else 'FAIL'})")
    print(f"confusion           : TP={TP} FP={FP} FN={FN} TN={TN}")
    print(f"precision(sim-win)  : {TP/max(TP+FP,1):.3f}   <- the honest use-case metric")
    print(f"recall(sim-win)     : {TP/max(TP+FN,1):.3f}   <- never discards a true winner if 1.0")
    print(f"winnable boards     : {len(winnable)}   argmax@1 = {argmax_hits}/{len(winnable)}")
    print(f"NON-TRIVIAL boards  : {len(nontrivial)} (win coexists w/ overshoot-FP) "
          f"argmax@1 = {nontrivial_hits}/{len(nontrivial)}  <- the v52-wall test")
    print(f"nontrivial board ids: {sorted(v%100000 for v in nontrivial)} "
          f"(mod 100000 = original variant id per batch)")
    for b in sorted(set(base.tolist())):
        m = base == b
        print(f"  base{b}: n={int(m.sum())} wins={int(win[m].sum())} "
              f"win-sign={np.mean(pred[m]==win[m]):.4f} auc={auc_score(sm[m],win[m]):.4f}")


if __name__ == "__main__":
    main()
