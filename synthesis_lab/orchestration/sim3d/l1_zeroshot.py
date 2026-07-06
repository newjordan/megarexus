"""DW Phase-3 — L1 ZERO-SHOT exam of the L0-selected dynamics config.

The pre-registered generalization test (plan §2 H2 caveat lane, v48→v52 lesson):
run the config SELECTED ON L0 OBSERVATIONS, unchanged, against the freshly
harvested L1 observed-pour corpus (rotated geometry). No re-selection. The full
family table on L1 is also reported for under-determination diagnostics only.

    python3 l1_zeroshot.py
Exit 0 if zero-shot gates pass (outcome >= 0.90, median first-div >= 10), 2 otherwise.
"""
from __future__ import annotations

import itertools
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES

from dynamics_fit import load_pours, eval_config, GATE_OUTCOME, GATE_FIRSTDIV

L0_SELECTED = {"deflect": "none", "spread": "lateral", "mouth": "fill",
               "edge": "dispose", "cell_cap": 1.0}


def main():
    pours, skipped = load_pours(FIXTURES / "calib_l1_pours_20260706.npz")
    n_win = sum(p["obs_win"] for p in pours)
    base = max(n_win, len(pours) - n_win) / len(pours)
    print(f"L1 pours: {len(pours)} usable ({len(skipped)} gate-skipped), wins {n_win} "
          f"(majority baseline {base:.3f})")

    r = eval_config(L0_SELECTED, pours)
    win_pours = [p for p in pours if p["obs_win"]]
    rw = eval_config(L0_SELECTED, win_pours) if win_pours else None
    gates = {"outcome_pass": r["outcome_acc"] >= GATE_OUTCOME,
             "firstdiv_pass": r["median_first_div"] >= GATE_FIRSTDIV,
             "win_pour_correct": (rw["outcome_acc"] == 1.0) if rw else None}
    passed = gates["outcome_pass"] and gates["firstdiv_pass"]
    print(f"ZERO-SHOT {L0_SELECTED}\n  -> {r}")
    if rw:
        print(f"  win-side pours ({len(win_pours)}): outcome_acc {rw['outcome_acc']:.2f}, "
              f"lava_acc {rw['lava_acc']:.2f}")
    print(f"  gates {gates} -> {'PASS' if passed else 'FAIL'}")

    # under-determination diagnostics: full family on L1 (NOT a selection pass)
    table = []
    for deflect, spread, mouth, edge, cap in itertools.product(
            ["none", "dl", "dr", "both"], ["none", "lateral"],
            ["fill", "entry"], ["dispose", "halt"], [1.0, 2.0, 3.0]):
        cfg = {"deflect": deflect, "spread": spread, "mouth": mouth,
               "edge": edge, "cell_cap": cap}
        table.append({"cfg": cfg, "l1": eval_config(cfg, pours)})
    table.sort(key=lambda e: (e["l1"]["outcome_acc"] + e["l1"]["lava_acc"],
                              e["l1"]["mean_iou"]), reverse=True)
    n_tied_top = sum(1 for e in table
                     if (e["l1"]["outcome_acc"], e["l1"]["lava_acc"]) ==
                        (table[0]["l1"]["outcome_acc"], table[0]["l1"]["lava_acc"])
                     and abs(e["l1"]["mean_iou"] - table[0]["l1"]["mean_iou"]) < 1e-6)
    print(f"  family diagnostics: best-L1 IoU {table[0]['l1']['mean_iou']:.3f} "
          f"({n_tied_top} configs tied at top)")

    (FIXTURES / "l1_zeroshot_20260706.json").write_text(json.dumps({
        "protocol": "zero-shot: config selected on L0 only, evaluated unchanged on L1",
        "l0_selected": L0_SELECTED, "zero_shot": r, "win_side": rw,
        "majority_baseline": base, "n_wins": int(n_win),
        "gate_skipped": skipped, "gates": gates, "pass": bool(passed),
        "family_table_l1": table}, indent=1, default=str))
    print(f"-> {FIXTURES / 'l1_zeroshot_20260706.json'}")
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
