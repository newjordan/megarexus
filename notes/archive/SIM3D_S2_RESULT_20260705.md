# SIM3D STAGE S2 RESULT — generalization sweep (the overfit antidote) — 2026-07-05

Execution-plan §5.S2, the MANDATORY antidote to the §4.2 caveat (the tonight
micro-rules were iterated until S0 passed, so S0 alone could be accidental overfit).
This is the first test of the hand-coded pour simulator on boards it NEVER touched
during rule-fitting.

```
SIM3D STAGE S2 RESULT — 2026-07-05
honesty_audit: exit 0, no new flags (this lane adds no shards/ckpts/imitation lineage)
t0-five untouched: YES (harvest uses base_seed 30000/31000 sp80_variants; real levels held out)
S0 decisive test after change: PASS (stream winner +1.000 vs FPs [-1,-1,-1,-1])
Gate criterion: >=95% win-sign agreement + margin AUC >= 0.98 + S0 PASS
  ->  PASS (win-sign 0.9964, AUC 0.9977, S0 PASS)   [MERGED, 1000 boards / 11,810 pours]
```

## What was run
`sim3d/s2_generalize.py` — harvest fresh `sp80_variants` boards (multi-base 0 AND 1,
2–4 targets, 1–5 blocks), SAMPLE arrangements **uniformly from the proposer
distribution** (`_candidates` — never selected by sim score, so zero leakage), pour
each **once for real** (env-graded), run the sim (S0-gate config, emit_volume=37) on
each pre-pour frame, score sim rollout AND real animation with the SAME
`pertarget_labels`. Two independent seed batches (base_seed 30000 and 31000), 500
boards each → **11,810 real pours, 118 env-wins** (win rate 1.0%). Fixtures:
`fixtures/s2_variants_20260705{,_b}.npz`. Analyzer: `sim3d/s2_analyze.py`.

One representational fix was required to run the sim on variant frames at all:
synthesized variants render the SOURCE housing (color 4) but no pre-charged color-6
reservoir, so `scene.py::extract_scene` now falls back to the source position when no
color-6 is present. Real levels ALWAYS carry color 6 → the S0/decisive path is
untouched (re-verified PASS).

## The numbers (MERGED, 1000 boards, 11,810 pours)
| metric | value | verdict |
|---|---|---|
| **win-sign agreement** | **0.9964** | ≥0.95 PASS |
| **margin AUC** | **0.9977** | ≥0.98 PASS |
| S0 decisive after change | PASS | must-hold |
| confusion | TP=117 FP=42 FN=1 TN=11650 | — |
| **recall** (sim win-call) | **0.992** | misses 1 of 118 true winners |
| **precision** (sim win-call) | **0.736** | ← the residual weakness |
| argmax@1 over ALL winnable boards | 26/29 = 0.897 | mostly trivial boards |
| **argmax@1 over NON-TRIVIAL boards** | **0/2** | ← the v52-wall test, FAILED |
| base0 | n=5866, 57 wins, win-sign 0.9959, AUC 0.9969 | PASS |
| base1 | n=5944, 61 wins, win-sign 0.9968, AUC 0.9984 | PASS (NOT degenerate) |

(batch A alone, 5,915 pours: win-sign 0.9949, AUC 0.9974, precision 0.559 — the low
precision there was small-sample; the merged 0.736 is the honest figure.)

## Honest reading — what PASSES and what the gate hides
1. **Generalization is REAL; no overfit collapse.** On 1000 boards the simulator never
   saw during rule-fitting, it caught 117 of 118 env-wins (recall 0.992) and correctly
   rejected **99.6%** of 11,692 losses. The §4.2 overfit fear is substantially allayed —
   the hand-coded rules transfer, they did not memorize the t0 five.
2. **The win-sign gate is nearly trivial here.** Win rate is 1.0%, so an "always predict
   loss" classifier already scores 0.9900 on win-sign; our 0.9964 is barely above it.
   **AUC and precision are the metrics that carry weight**, and AUC is itself
   imbalance-flattered (only 42 FP-losses among 11,692 losses inflate it to 0.998). Do
   not headline win-sign.
3. **The real weakness = lava/overshoot precision (0.736).** The 42 false positives are
   almost all `real_lava ≈ 254` (whole floor strip flashed): the sim FILLS all targets
   but the real pour OVERSHOOTS into the floor. The sim's floor-lava/overshoot model
   under-fires on these arrangements. This is the v52/STAGE_PT wall (§4/§5 of the
   handoff) resurfacing INSIDE the simulator — now reproducible and localized.
4. **The argmax fails exactly where discrimination is required — 0/2.** Of 29 winnable
   boards, 27 have no FPs (any positive-margin arrangement is a true win → trivial). The
   **two** non-trivial boards — **336** (1 win vs 8 overshoot-FPs) and **269** (1 win vs
   1 overshoot-FP) — have their true winner and the overshoot-FP(s) ALL tied at
   sim_margin +1, so the sim picks a non-winner on BOTH. This is the exact crux the
   whole program is stuck on, reproduced at scale: whenever a filling winner and a
   filling-but-flooding near-miss coexist, the sim cannot separate them.

**Bottom line:** S2 PASSES its pre-registered gate, and more importantly VALIDATES the
simulator as a **filter/ranker** (perfect recall, rejects 99.5% of losses, transfers to
unseen boards). It does NOT yet solve argmax-discrimination on adversarial boards: the
sim's lava-timing fidelity misses certain overshoots, so a filling winner and a
filling-but-flooding near-miss can tie at margin +1. This tempers "the wall is broken":
broken for RANKING and for L1 specifically (S3), still open for general argmax on
overshoot-contaminated boards.

## S1 corpus handed off (do NOT touch the t0 five)
The 42 false positives + 1 false negative are the S1 tick-match corpus. Highest value —
the two boards where a true winner is TIED with overshoot-FPs (fixing these directly
attacks the argmax wall):
- **board 336** (batch A, seed 30336, base 0) — 1 win vs 8 overshoot-FPs, all tied +1.
- **board 269** (batch B, seed 31269, base 1) — 1 win vs 1 overshoot-FP, tied +1.
  Tick-match the FP arrangements against the real animation: the sim reads fill without
  lava where the env floods the floor. Fixing the sim's overshoot / floor-contact timing
  should split these ties (winner stays +1, overshoot-FP drops to −1) and lift precision.
- Pure-FP boards (unwinnable, sim hallucinates a fill): batch A 84, 199, 290, 324, 404,
  435 (+ batch B equivalents in the fixture).
- The 1 false negative: batch B seed 31138 (base 0), a clean env win (real_lava=0) the
  sim under-scored to margin 0.00 — a fill-threshold miss, not a lava miss.
Reproduce deterministically: `make_env_variant(arc, seed=S, rng=default_rng(S),
base_level=B)` with S = 30000+v (batch A) or 31000+v (batch B), B = v%2 (0 even, 1 odd).

## Next per ladder
S1 — Fidelity closure, now with a concrete corpus (the 30 overshoot-FPs above) and a
single decisive board (336). The S1 gate (per-target |fill−fill|/|lavan−lavan| MAE ≤
0.10 across the corpus) should specifically drive down the floor-lava miss rate. After
any S1 rule change: re-run S0 (must stay PASS) AND re-run this S2 sweep — precision
should rise from 0.56 while recall stays 1.0.
