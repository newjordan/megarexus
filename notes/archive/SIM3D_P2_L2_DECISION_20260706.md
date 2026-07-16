# REGIME L2-FRONTIER P2 RESULT — 2026-07-06 — DECISION-BRANCH NEGATIVE (escalate)

**The pre-registered P2 negative branch fired: the faithful post-P0 sim + constructive
enumeration find NO clean all-3-fill L2 arrangement, and the ENV floods on 100% of reachable
pours. Strong evidence L2 is NOT solvable by arrange-and-pour. Per `TRAINING_REGIME_20260706_L2_FRONTIER.md`
§2 P2, I STOP and escalate the intended-L2-mechanic question.** Artifacts:
`fixtures/p2_l2_construct_result.json`, `fixtures/p2_proposer_l2_postP0_20260706.json`.

```
honesty_audit: exit 0 (no new flags)   t0-five untouched: YES   S0: PASS
Gate criterion (§2 P2): env-confirmed L2 win <=4 pours -> headline 3 levels
  OR pre-registered NEGATIVE: faithful sim + constructive sweep finds NO margin>0 -> escalate
Result: NEGATIVE — 0 margin>0 arrangements across 14k+ evals; 200/200 env pours flood. Escalated.
```

## The evidence (converging, four independent lines)
1. **ENV ground truth — 200 real env-graded L2 arrangements: 100% FLOOD, 0 clean.** Not one
   clean (no-floor-touch) pour, not even a clean 1-fill. 5 arrangements filled ALL 3 mouths
   yet STILL flooded (lost). (`p0_l2_diag_latch.npz`.)
2. **Constructive sweep (`p2_construct.py`, 4,033 imagined evals):** all single-block + all
   two-block combinations + a 3-block hill-climb, scored by the POST-P0 fill-faithful sim.
   **n_margin_positive = 0; n_clean_multifill = 0.** Best achievable is EITHER clean+fill-1
   (`[1,0,0]`, lava 0) OR flood+fill-2 (`[1,1,*]`, lava 1) — never clean+fill-2-or-3.
3. **Mutation search (`proposer_search.py`, 10,304 evals, post-P0 sim):** max fitness rose from
   the old buggy-sim 0.05 to **1.52** (P0 works — the sim now rewards multi-target fills), but
   margin never exceeded **0.0**; NOT solved.
4. **Structural capacity argument (quantified):** real L2 emits ~**54** macro cells of paint;
   safe holding = 3 one-cell mouths + ~21 movable-bar top cells ≈ **24**; so ≈ **30 cells must
   overflow to the floor** → flood is unavoidable. The win needs all 3 mouths entered AND zero
   floor touch (game L730/L815) — incompatible with a 54-vs-24 paint-vs-capacity budget.

The four agree: the L2 pour emits far more paint than the mouths+bars can hold, so every
reachable arrangement floods. Filling more targets requires routing more paint to the bottom,
which increases the overflow. **Clean-and-fill-all is not in the reachable arrangement space.**

## What P0/P1 bought (this is NOT a sim-fidelity failure)
The post-P0 sim faithfully REPRODUCES the env's verdict here: it, too, finds every multi-fill
arrangement floods (best margin 0.0). Pre-P0 the sim reported 0 fills (Correction B) and this
question could not even be asked. The negative is now a statement about the GAME, not the sim:
recall on the S2-MS gate is 1.0 (the sim never hides a real winner), so "sim finds no winner"
is meaningful evidence, not a sim blind spot.

## Decision escalated (per §2 P2, do not grind)
L2 appears unwinnable by arrange-and-pour. The intended L2 mechanic is unclear. Open questions
for the user (SP80 design knowledge / direction):
- Is L2 intended to be won by pouring all 3 sources at once at all? (Every such pour floods.)
- Is there a paint-reduction mechanic I'm missing — e.g. capping a source with a block, or a
  deep multi-bar container outside the sampled arrangement space? (Capacity math says even an
  optimal container from 21 bar-cells < 54 emitted.)
- L2 has NO redirect (`tuvkdkhdokr`) — unlike L4/L5. Should the frontier move to L3
  (connected mouths + sidewalls, may hold more) or L4/L5 (redirects — a genuinely different
  routing tool), rather than forcing L2?
- Or: accept the honest boundary — **2 levels remain the headline** (L0 CNN-alone, L1
  sim-guided SOLVED@1) — and pivot the program (e.g. P3 training on the P0/P1 corpus, which is
  no-imitation-legal and independent of the L2 unlock).

## Headline (unchanged)
**2 levels in-budget — L0 CNN-alone, L1 sim-guided SOLVED@1, robust.** New tooling (agent-
generated, no imitation): `p2_construct.py` (constructive route sweep). The L2 negative is an
honest, quantified boundary, not a wall hit by insufficient search.
