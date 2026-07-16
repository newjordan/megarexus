# SIM3D STAGE S1 (iteration 1) RESULT — the overshoot-FP flood — 2026-07-05

Execution-plan §5.S1 fidelity closure, driven by the S2 corpus
(`SIM3D_S2_RESULT_20260705.md`): the 42 false positives where the sim scores +1
(fills, no lava) but the env floods the floor (`real_lava≈254`). Decisive board:
**336** (1 true winner tied at margin +1 with 8 overshoot-FPs).

```
SIM3D STAGE S1 RESULT — 2026-07-05  (iteration 1: DIAGNOSED, fix attempt REVERTED)
honesty_audit: exit 0, no new flags       t0-five untouched: YES
S0 decisive after (reverted) change: PASS  (physics restored exactly)
Gate criterion: 55/55 win-sign AND per-target MAE <= 0.10 AND S0 PASS  ->  NOT MET (S1 still open)
Corpus: S2 FP set; tick-matched board 336 (seed 30336) arr-by-arr vs real animation
New mechanics facts: the overshoot-FP flood path + why a blanket fix regresses (below)
Next per ladder: S1 iteration 2 — a WIN-vs-FLOOD timing race, validated on the FULL S2 corpus
```

Tooling added: `sim3d/s1_tickmatch.py` (reproduces a variant's S2-sampled arrangements
EXACTLY — same per-variant rng as `s2_generalize` — pours each for real, lines the
animation up against the sim, sweeps emit_volume).

## Diagnosis (verified, keep these facts)
Tick-matched board 336's winner (arr 4) and its 8 overshoot-FPs against the real
animations. Ground truth: the real WINNER emits only **17 macro-cells** of paint, fills
its one mouth, floods nothing (n14=19); every overshoot-FP emits **41–50 macro-cells**
and flashes the whole floor strip (n14=254). The emitted VOLUME is arrangement-dependent
(the pour self-stops at the clean fill, runs long when it overflows).

**First divergence (visual, arr 0):** real paint sheds off the source-bar pool and
cascades **block-to-block down the left side into the open leftmost column → the floor →
flood**. The sim pools the same paint 1-cell-deep ON the blocks and **self-settles at ~23
emitted cells, deepest paint row 13, never reaching the floor (row 15)**. Instrumented:
`maxsettled=1.0`, `spout_settled=0` (not backpressure), emitted flat at 23 regardless of
emit_volume (37→220 identical) — the paint reaches a resting state and stops the pour.

**Root cause:** the lateral shed (`raycast.py` step 3, solid-base branch) moves only
`total − cap` — the excess ABOVE cap. A 1-cell-deep pool on a bar has `total == cap` →
zero excess → it never dribbles over the bar's open lip. So bar-top pools rest forever
and paint never cascades to the floor. (Mouth stacks are correctly bounded by solid
prongs, which yield no lateral outs — that part is right.)

## Fix attempt: `ledge_spill` (a bar-lip pool dribbles over an open ledge at rest) — REVERTED
Implemented behind a default-off `HydroParams.ledge_spill` flag; when a solid-based
column has an open-drop ledge among its lateral outs, drain the column over the lip
(`avail = min(flow, total)`) instead of only the over-cap excess.

Result on board 336: **fixed 5 of 8 overshoot-FPs** (+1 → 0, below the winner) and
**kept S0 PASS**. Encouraging — but the honest test is the FULL S2 corpus, and there it
**REGRESSED** (same 500 boards, seed 30000):

| metric | ledge OFF (validated) | ledge ON | verdict |
|---|---|---|---|
| precision | 0.559 | **0.447** | worse (FP 30 → 47) |
| recall | 1.000 | 1.000 | unchanged |
| win-sign | 0.9949 | 0.9921 | worse |
| margin AUC | 0.9974 | 0.9960 | worse |
| argmax@1 (winnable) | 8/9 | **6/9** | worse |

The blanket rule added 17 NEW false positives on other boards (recall was never the
problem — FN stayed 0). Draining paint off ledges that reality holds shifts fill/lava
patterns elsewhere and manufactures spurious sim-wins. **Net negative → reverted;
`raycast.py` physics restored exactly (S0 re-verified PASS, board 336 back to original).**

## The lesson + the correct next direction
- **Tuning to one board (336) is misleading.** ledge_spill looked good on 336 and passed
  S0, yet regressed the corpus. Every S1 rule change MUST be scored on the full S2 sweep
  (`s2_generalize.py` + `s2_analyze.py`), not just the decisive board.
- The 3 stubborn FPs on 336 (arr 1/5/8 — the heaviest emitters, real peak6=50) exposed
  the real structure: a **WIN-vs-FLOOD race**. The sim fills the mouth and declares the
  win BEFORE its (now-cascading) overflow reaches the floor; reality floods first (so the
  env scores a loss). The discriminator is not "does paint reach the floor eventually"
  but "does the overflow reach the floor BEFORE all mouths fill" — a RELATIVE-RATE /
  ordering property. A blanket spill gets the destination right but the timing wrong.
- **Iteration 2 direction:** model the race explicitly — emission continues until win OR
  floor-contact (stop emitting AT the win, not stop_on_win ticks later), with a cascade
  rate calibrated so a filling winner wins before overflow but an overshoot-FP floods
  first. Calibrate the rate on the 55-pour L0 corpus + S2 boards (legal), gate on the
  FULL S2 sweep (precision up, recall stays 1.0) AND S0 PASS. Do NOT touch the t0 five.

## Iteration 1 status
Converted "unknown weakness" into a precisely-characterized, reproducible one. Fix path
handed to iteration 2 below.

---

# SIM3D STAGE S1 — iteration 2 RESULT — floor-gated ledge spill — 2026-07-06 — ADOPTED

Iteration 1's lesson made the fix: the blanket ledge_spill regressed because it accelerated
BOTH floor-cascade AND mouth-feed. So gate the at-rest spill on DESTINATION — a bar-lip pool
dribbles over an open ledge only when the straight drop below reaches the FLOOR strip, never
a mouth (`HydroParams.ledge_spill_floor`, using a `drops_to_floor` ray check; also added
`backpressure_cap` param + `win_tick`/`lava_tick` diagnostics).

**Full S2, 11,810 pours, BOTH seed batches (30000 & 31000):**
| metric | baseline (iter1 default) | iter1 blanket ledge | **iter2 floor-gated (ADOPTED)** |
|---|---|---|---|
| precision | 0.736 | 0.447 ❌ | **0.807** ✅ |
| recall | 0.992 | 0.992 | **0.992** (unchanged) |
| argmax@1 | 26/29 | — | **28/29** |
| **nontrivial argmax** | **0/2** | — | **1/1** (both discrimination boards now handled) |
| S0 decisive | PASS | PASS | **PASS** |

Precision +0.071 (14 FPs killed), recall untouched, and the two boards that actually
required winner-vs-overshoot discrimination (336, 269) are now BOTH resolved (one became
trivially-correct, the other nontrivial-correct). Generalizes across independent seeds →
adopted as the `HydroParams` default (`ledge_spill_floor=True`). Reversible via the flag.

## Why it works where iter1 failed
The `drops_to_floor` gate is the whole trick: iter1 drained paint over any lip (including
lips feeding mouths → +17 spurious mouth-fill FPs, precision DOWN); iter2 drains only over
lips whose straight drop lands on the floor strip → routes overshoot to the floor (correct
flood) without touching the fill path. It is a DESTINATION-aware rule, not a rate hack.

## Status
S1 **materially advanced** (not fully closed — precision 0.807, not 1.0; ~28 overshoot-FPs
remain, and the winner can still TIE an FP at margin +1 on rare boards). But the argmax wall
that blocked in-budget solves is broken on the S2 corpus, and the fix is validated + adopted.
Remaining FPs are the iter-3 corpus (same method: tick-match, gate on full S2, keep recall).
Next: re-attempt L1 (re-confirm) and L2 (was blocked by exactly this FP) in-budget solves.
