# HANDOFF — The Pour-Simulator Problem (2026-07-05)

Written at a session boundary for the next push: **the user has an idea for the pour
simulator.** This note frames the goal, the wall, and the full failure map so that idea
lands with complete context. Companion docs: `STAGE_CA_SPEC_20260705.md` (a drafted
Neural-CA lane — may be superseded by the new idea), `NIGHT_REGIME_20260705.md` F1–47,
`ROADMAP_WORLD_MODEL_20260705.md` (binding plan), memory `megarexus-cnn-deliberation-gap`.

---

## 1. THE ONE GOAL
A CNN, **fresh-from-0, no imitation**, that COMPLETES SP80 levels in-budget and env-graded:
per level, ≤4 pours, no human/route/answer data. Current honest headline = **1 level
CNN-alone** (L0, by ranking). L1 has been solved once by env-feedback *search* (lava_search,
981 evals) but NOT by a CNN choosing an arrangement in ≤4 pours. **The prize: move the headline
to 2 levels** by making the CNN pick a winning L1 arrangement within budget.

## 2. THE MECHANISM (how an in-budget solve must work)
The pour is a **deterministic, LOCAL cellular automaton**, fully determined by the pre-pour
arrangement (in "spill" mode `step()` ignores input). So the loop is:
```
   propose arrangement → IMAGINE the pour → geometric win-check → POUR for real (env grades)
```
WIN = every U-target recolored 11→13 AND no wall hit (14-growth = "lava"). You cannot see the
win on the STATIC board — you must **imagine the pour and watch the win happen** (user's
validated framing). Everything hinges on the "IMAGINE the pour" box being faithful.

## 3. WHAT IS PROVEN (assets — build on these, don't re-litigate)
- **Perception is CNN-able.** Frozen encoder: click 0.6px, dynamics ~1.0 on held-out frames.
- **v52 is a genuine cross-level RANKER** (`runs/outcome_v52_multibase_unfrozen_apollo_20260705/
  best.pt`): predicts aggregate (fill, lava) from an arrangement; REAL-L0 AUC 0.98, L1 0.99;
  correctly scores the verified L1 winner. It RANKS well but its ARGMAX is imprecise (see §5).
- **The per-target GROUND TRUTH separates winner from false positives (STAGE_PT T0 PASS).**
  On real L1 the verified winner reads fill=1.0 / lava=0.0 on all 3 targets; every v52 false
  positive is lava-saturated (lava=1.0) and/or misses fill on a specific target. **The
  discriminating SIGNAL exists in the pour outcome** — see `runs/t0_pertarget_20260705.json`.
- **The verified L1 winner** (env-confirmed 240/240, lava≈1), genome (ci,dir,mag) applied on L1
  seed 1234 with `centers = frame_components(present(L1))[:8]`:
  `[(1,1,2),(6,4,2),(4,1,3),(1,1,3),(3,4,3),(5,1,1),(7,2,3)]`. Its 4 v52 false-positive genomes
  (all env-losses) are in the same t0 JSON — the canonical winner-vs-near-miss test set.
- **Tooling:** `pertarget_labels.py` (per-target fill/lava from any frame sequence — real OR
  simulated), `pt_gates.py` (G1/G2/G3 harness), `imagine_search_gate.py` (`--ca` Neural-CA
  rollout + `forward_dynamics` rollout), `train_spill_ca.py` (Neural-CA trainer, encoder-frozen,
  active-front-masked, clean-window via `pour_settle`), `sp80_variants.py` (randomized boards),
  `harvest_variants.py --per-target`, `pour_settle.py`. GB10 cuda works and is fast.
- **DEVIATION-1 (critical label fact):** one ACTION5 press returns the WHOLE animation as a LIST;
  on a WIN the all-filled state holds only **~2 frames** before the level RESETS (13→11, paint
  cleared, wall-14 win-flash 27→64). So read fill (2-frame window) + lava over the PRE-RESET
  outcome frames only (`pertarget_labels` does this; `runs/T0_DEVIATION_label_window.md`). The
  literal 3-frame window scored the confirmed 240/240 winner as fill=0 — a trap.

## 4. THE WALL (state it precisely — this is what the new idea must beat)
**The winner and its near-miss false positives have near-identical STATIC pre-pour arrangements.
What separates them is how the paint STREAMS ROUTE during the pour — whether they fill cleanly or
OVERSHOOT into a wall (lava).** That signal is DYNAMIC, not static. Concretely, the STAGE_PT
per-target head predicted **near-identical (fill≈0.16, lava≈0.95) for the winner AND an FP**
(their static features around each target look the same) — it is structurally blind to the
routing difference. So: **no static model can pick the winner. You must SIMULATE the pour** —
and specifically simulate WHERE the streams hit walls (the lava/overshoot), which is the hardest
part. This is the crux the simulator must nail.

## 5. THE FAILURE MAP (everything tried; why each died — do NOT repeat)
| Approach | Result | Why it failed |
|---|---|---|
| Reactive CNN policy / BC of Contra / DAgger | 0/18 levels | deliberation gap; a reactive net can't do multi-step arrange-then-pour |
| **Pixel-CA rollout** (v38–42, `forward_dynamics` / `spill_ca`) | drifts, over-paints ~6× paint mass | wrong bias + (once) a post-reset LABEL bug (F16); the 40-tick rollout compounds error. **THE RECURRING WALL.** |
| Latent settle-predictor + win-probe (Stage B) | AUC-good, argmax-imprecise | ranks well but the score's high region is false positives |
| Aggregate (fill,lava) head = **v52** | great RANKER, bad ARGMAX | search maximizing v52's score lands on FPs (0.6–0.8) that outscore the true winner (0.51) |
| **Per-target (fill_t,lava_t) head** (STAGE_PT v55/v56) | **G1 FAILS both** | flips the ordering (winner now #1) but score stays NEGATIVE by a ~0.003 noise margin; head blind to arrangement (§4). G2 regressed (L0 0.756 vs v52 0.916). **DEAD END — do not retry a static per-target predictor.** |
| CEM over arrangements (v48_solve) | collapses to FPs | flat/false-positive fitness landscape |
| Diversity search (best-per-restart) | still FPs (L1 top-4 [0.79,0.67,0.67,0.58]) | ranker can't put the winner at the top |
| Hard-negative mining (v53/v54) | HURTS | FPs look ~identical to winners to a static net |
| rotation via `set_data`, `steps[-1]` settle | banned | F32 / F16 bugs |

**Two distinct unsolved walls remain, don't conflate them:**
- **(a) FIDELITY / RANKER:** imagine the pour precisely enough to tell winner from overshoot-FP.
  ← the simulator problem; every attempt drifts. *This is where the new idea goes.*
- **(b) PROPOSER:** even with a perfect ranker, random/CEM search finds ~0 L1 winners in
  thousands of arrangements (the winner is a precise, rare arrangement). Roadmap Stage C
  (portband-seeded CEM) is the plan for (b). Not the current focus.

## 6. WHY A SIMULATOR IS THE CRUX (and why it keeps failing)
The pour is a *known* deterministic local CA — in principle a small shared-weight local rule
(fall / split-on-block / deflect-on-redirect / fill-on-U / hit-wall) learned EXACTLY is
**level-invariant by construction** and would generalize to L1+ with zero L1+ positives. The
roadmap (Q1) bets on exactly this (`spill_ca`, active-front mask, correct settle data). BUT every
pixel rollout so far DRIFTS over ~40 ticks — the model smears, over-paints, and loses the sharp
"stops at the wall / fills the U" behavior. The open technical question the new idea must answer:
**how to roll the pour out (esp. the wall-hit / overshoot / stop conditions) without compounding
drift** — e.g. multi-step/scheduled-sampling rollout loss, a discrete/argmax-in-the-loop CA, an
explicit stream/flow representation, hard geometric constraints, or something else entirely.

## 7. THE DRAFTED LANE (context for the new idea; may be superseded)
`STAGE_CA_SPEC_20260705.md` pre-registers a Neural-CA simulator lane (CA0 baseline → CA1 tick
pairs → CA2 train with a k-step anti-drift loss → CG1 winner-vs-FP separation from the SIMULATED
rollout → CG2 rollout fidelity → CG3 in-budget solve). CA0 (no-training: measure how far the
EXISTING `spill_ca`/`forward_dynamics` rollouts drift on the winner + 4 FPs) was **NOT yet run** —
it's the cheap first shot and the baseline any new simulator must beat. **The decisive test for
ANY simulator idea:** roll out the winner + the 4 FP arrangements, read per-target (fill,lava)
via `pertarget_labels` on the rollout, and check the winner's `min_t(fill_t − lava_t)` comes out
POSITIVE and ABOVE all 4 FPs (real labels: winner ≈ +1, FPs ≈ 0). That one test, on the canonical
5 arrangements, tells you in minutes whether a simulator idea has legs — before any training.

## 8. FRESH-SESSION FACTS / GOTCHAS
- Honesty first: `python3 synthesis_lab/orchestration/honesty_audit.py` (exit 2 → stop). Guard
  every shard/ckpt; real levels EVAL-ONLY; env is the only grader. Banned: `_invalid/`,
  `sp80_spill_*`, `sp80_player_harness`.
- L1 work uses **seed 1234**; `centers = frame_components(present(L1,1234))[:8]`; genome format
  `(component_index, direction 1-4, magnitude)`; apply via `imag_search_solve.actions`.
- GB10 cuda works locally (fast); v52 is local. Apollo optional (detached-launch gotchas in
  STAGE_PT §7). `nvidia-smi` before GPU work (shared rig).
- Current branch `night-shape-lava-imagination-20260705`; STAGE_PT committed `f1f3de6` (code +
  notes only, no .pt). Working tree has pre-existing unrelated edits (not mine) — leave them.
- Reusable to score ANY simulator: `pertarget_labels.pertarget_labels(pre_frame, rollout_frames)`
  → per-target (fill,lava); the winner/FP genomes + real labels in `runs/t0_pertarget_20260705.json`.

## 9. OPEN QUESTION (where the new idea plugs in)
Everything upstream (perception, ranking, per-target ground truth, the win predicate, the env
loop) is in place and proven. The ONE missing capability is a **faithful pour simulator** —
specifically one that predicts the overshoot/wall-hit (lava) that distinguishes a true winner
from a fill-but-slam near-miss, without the ~40-tick pixel drift that has beaten every attempt.
**Bring the crazy idea here.** Fastest way to test it: the §7 5-arrangement decisive check.

---

## 10. RESOLVED 2026-07-05 ~23:00 — the crazy idea WORKED (same night)
The user's idea: extract a 3D representation of the board objects and SIMULATE the pour
(spheres out of the bucket, ray/pathing queries over the extruded 2D shapes) instead of
learning the rollout. Built as `synthesis_lab/orchestration/sim3d/` (hand-coded, zero
training, deterministic, rotation-canonicalized). Results, same session:
- **§7 decisive test: PASS** — stream lane scores the verified winner +1.000, all 4 FPs
  −1.000 (`sim3d/fixtures/s0_result_stream.json`). First imaginer ever to pass.
- 55-pour L0 corpus: **100 % win-sign agreement** with the env.
- **STAGE S3: L1 SOLVED IN-BUDGET, SOLVED@1** — 4,848 imagined evals, 0 search pours,
  first real pour wins, NEW winning arrangement, independently re-confirmed
  (`sim3d/fixtures/s3_l1_result.json`). Headline: **2 levels in-budget**
  (L0 CNN-alone, L1 sim-guided).
- The §6 open question ("roll out the pour without compounding drift") is answered by
  not learning the dynamics at all: the pour CA's micro-rules were reverse-engineered
  from legal L0 animations (key rule: COLUMN-BASE SHEDDING — see execution plan §3).
- §5's wall row "Pixel-CA rollout ... THE RECURRING WALL" → beaten by construction.
Work order for continuing agents: `SIM3D_EXECUTION_PLAN_20260705.md` (stages S1–S6,
gates, honesty rules — S2 generalization sweep is MANDATORY before further claims).
`STAGE_CA_SPEC_20260705.md` (neural-CA lane) is SUPERSEDED unless S2/S4 uncover
geometry the hydraulic CA can't model.
