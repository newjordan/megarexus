"""DW Phase-3 round 2 — joint L0+L1 selection, fresh L2 zero-shot exam.

Protocol amendment (recorded in DW_LAB_LOG): round-1 selection on L0 alone was
under-determined (configs tied); the L1 corpus discriminates (its single win
pour separates mouth/entry + cell_cap). Round 2 therefore SELECTS on the L0+L1
train splits, gates on their holdouts, and takes the ZERO-SHOT exam on L2 —
a level never seen by any selection step (multi-source, all-loss corpus, so
the exam metrics are lava agreement + first-div + IoU; outcome baseline is
reported alongside).

    python3 joint_fit_round2.py
Exit 0 if holdout gates AND the L2 zero-shot gates pass, 2 otherwise.
"""
from __future__ import annotations

import itertools
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES

from dynamics_fit import load_pours, eval_config, GATE_OUTCOME, GATE_FIRSTDIV

FIXES = ["calib_l0_pours.npz", "calib_l1_pours_20260706.npz"]
EXAM = "calib_l2_pours_20260706.npz"


def main():
    train, hold = [], []
    for fx in FIXES:
        pours, skipped = load_pours(FIXTURES / fx)
        for p in pours:
            (hold if p["i"] % 4 == 3 else train).append(p)
        print(f"{fx}: {len(pours)} pours ({len(skipped)} gate-skipped), "
              f"{sum(p['obs_win'] for p in pours)} wins")
    print(f"joint: train {len(train)} / holdout {len(hold)}")

    table = []
    for deflect, spread, mouth, edge, cap in itertools.product(
            ["none", "dl", "dr", "both"], ["none", "lateral"],
            ["fill", "entry"], ["dispose", "halt"], [1.0, 2.0, 3.0]):
        cfg = {"deflect": deflect, "spread": spread, "mouth": mouth,
               "edge": edge, "cell_cap": cap}
        table.append({"cfg": cfg, "train": eval_config(cfg, train)})
    table.sort(key=lambda e: (e["train"]["outcome_acc"] + e["train"]["lava_acc"],
                              e["train"]["mean_iou"]), reverse=True)
    best = table[0]
    print(f"SELECTED (train): {best['cfg']} -> "
          f"{ {k: round(v, 3) for k, v in best['train'].items()} }")

    hr = eval_config(best["cfg"], hold)
    hold_wins = [p for p in hold if p["obs_win"]]
    hw = eval_config(best["cfg"], hold_wins) if hold_wins else None
    print(f"HOLDOUT: { {k: round(v, 3) for k, v in hr.items()} }"
          + (f" | win-side ({len(hold_wins)}): {hw['outcome_acc']:.2f}" if hw else ""))
    gates_h = {"outcome": hr["outcome_acc"] >= GATE_OUTCOME,
               "firstdiv": hr["median_first_div"] >= GATE_FIRSTDIV}

    exam, ex_skip = load_pours(FIXTURES / EXAM)
    er = eval_config(best["cfg"], exam)
    n_win = sum(p["obs_win"] for p in exam)
    base = max(n_win, len(exam) - n_win) / max(len(exam), 1)
    print(f"L2 ZERO-SHOT ({len(exam)} pours, {len(ex_skip)} gate-skipped, "
          f"baseline {base:.3f}): { {k: round(v, 3) for k, v in er.items()} }")
    gates_e = {"lava": er["lava_acc"] >= 0.90,
               "firstdiv": er["median_first_div"] >= GATE_FIRSTDIV,
               "outcome_vs_baseline": er["outcome_acc"] >= base}
    passed = all(gates_h.values()) and all(gates_e.values())
    print(f"gates holdout {gates_h} exam {gates_e} -> {'PASS' if passed else 'FAIL'}")

    (FIXTURES / "dynamics_round2_20260706.json").write_text(json.dumps({
        "protocol": "round-2: select on L0+L1 train, gate on L0+L1 holdout, "
                    "zero-shot exam on L2 (never used in selection)",
        "selected": best, "holdout": hr, "holdout_win_side": hw,
        "l2_exam": er, "l2_baseline": base, "l2_gate_skipped": ex_skip,
        "gates": {"holdout": gates_h, "exam": gates_e}, "pass": bool(passed),
        "table_top10": table[:10]}, indent=1, default=str))
    print(f"-> {FIXTURES / 'dynamics_round2_20260706.json'}")
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
