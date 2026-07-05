# ROADMAP — World-Model Base for Megarexus (SP80)

**Date:** 2026-07-05 · **Author:** Fable 5 synthesis session · **Status:** BINDING
plan-of-record until superseded by an env-graded result.

**Audience:** any agent (of any capability) adding to this system. This document
is the integration contract. If you are a smaller agent: you do not need to
re-derive the design — execute the stage the Scoreboard (§4) says is next,
respect the Frozen Contracts (§3), and report using the template (§7). If your
result contradicts a gate criterion here, the env is right and this doc gets
amended — never the other way around.

Companion docs: `REPORT_CNN_DELIBERATION_20260705.md` (the wall, the evidence),
`NIGHT_REGIME_20260705.md` (findings 1–9), `HANDOFF.md` §0 (live state).

---

## 0. TL;DR

The program has already converged, independently, on LeCun's H-JEPA shape:
**world model + MPC over plans + hard win-check + optional test-time
adaptation**. That frame is correct — keep it. What was missing is (a) the
right inductive bias for the pour CA, (b) a latent layer used for the two
things latents are actually good at here (amortized scoring + adaptation), and
(c) a proposer that can put a winning arrangement into the candidate set for
L1+ without any solved example. The Contra subsystems map cleanly onto the
world-model components (§2) — Contra doesn't get replaced, it gets *absorbed as
priors and probes*.

**One-line strategy:** pixels decide, latents rank, portband proposes, the env
grades, and adaptation (AdaJEPA-style) happens only inside an episode with a
control arm.

---

## 1. Verdicts on the four open questions (REPORT §9)

### Q1 — Pixel vs latent world model for a deterministic CA
**Verdict: the CA substrate should stay pixel/discrete — but with a LOCAL,
ITERATED inductive bias. Latents are a complement, not a replacement.**

The pour is a deterministic, *local*, discrete cellular automaton. The ground
truth is literally a small-neighborhood update rule applied repeatedly. The
current dynamics head (global FiLM trunk, 64→16 channels, full receptive
field) has the wrong bias for that: it can smear global context into a local
rule and has no reason to learn the exact per-cell transition. Compounding
pixel error over ~40 ticks is the signature of *wrong bias + under-exposure*,
not of "pixels are wrong."

- v38 (retrain decoder on dense spill ticks) is the correct next test.
- If v38 still misses (gate `fn > 0` on L0), the fix is **A2: a dedicated
  Neural-CA spill head** — a small shared-weight conv stack (receptive field
  3×3 or 5×5, ~2–4 layers, 16-way per-cell softmax) applied *iteratively*,
  trained ONLY on spill-tick pairs, hard-masked to cells near active paint.
  A local rule learned exactly is **level-invariant by construction** — this is
  the single highest-leverage architectural change available.
- A JEPA latent predictor does NOT model a discrete CA more faithfully than a
  correct local model — latents buy robustness to *irrelevant* variation, but
  in a deterministic CA every cell is relevant. Use latents for §Q4's two-tier
  scoring, not as the CA simulator.

### Q2 — Is TTA worth it in a 4-attempt budget?
**Verdict: yes — because the budget is 4 POURS, not 4 data points.** Each
failed pour emits ~40 spill-tick transitions of *this level's geometry under
the true rule*. That is ~8× AdaJEPA's entire 5-transition buffer, from one
attempt. AdaJEPA's published recipe fits almost unchanged (§5). The existing
honesty fence (per-seed weight reset + no-adapt control arm on identical
seeds/candidates) is sound; add one thing: log the adapted-parameter delta
norm per step, so a suspicious jump is visible in the run artifact.

### Q3 — The L1+ proposer (the crux)
**Verdict: staged — CEM over arrangements with a RELAXED geometric fitness,
population seeded by portband; a learned proposer only later, as amortization.**

The key unlock: the win-check doesn't have to be binary. Relax it into a
fitness: `score = (#targets recolored 11→13) − λ·(wall hit) ` evaluated on the
*imagined* settled frame. That turns "winning arrangements are rare" into a
gradient a Cross-Entropy-Method loop can climb — no positives, no teacher, no
answer injection. Portband's alignments/route-flow hypotheses (pixel-derived,
truth-boundaried) seed the initial population — Contra as *informer prior*,
exactly its sanctioned role. A learned proposer distilled from CEM's imagined
winners is honesty-safe (headline is still env-graded) but bootstraps on model
error — it is gated behind imagination fidelity (Stage C before Stage E).

### Q4 — Architecture sanity
**Verdict: the frame is right, with three amendments and one rejection.**
1. **Two-tier scoring:** latent settle-predictor + win-probe as a fast RANKER
   over hundreds/thousands of candidates; pixel rollout + geometric check as
   the DECIDER over the top-K. The hard, non-learned check keeps its veto.
2. **Relaxed fitness** (Q3) as the search signal.
3. **Neural-CA bias** (Q1) if v38 alone doesn't close fidelity.
4. **REJECTED: AdaJEPA's planning objective** `min Σ d(ẑ, z_goal)`. It needs a
   goal *observation*. No solved L1+ frame exists and honestly cannot — so
   goal-conditioned latent distance is unusable as the decider. (Partial
   loophole, optional: a *synthetic* goal image built by recoloring targets
   11→13 in the current frame — derived from the win predicate, not from any
   answer, so it's honest — usable as an auxiliary ranking signal only.)
   We adopt AdaJEPA's **adaptation**, and JEPA's **latent prediction**; we keep
   our own **win predicate** as the objective. Our predicate is *stronger* than
   their goal-distance — don't trade down.

---

## 2. Target architecture — Contra absorbed, not replaced

```
             ┌────────────────────────────────────────────────┐
             │ L5 DELIBERATION LOOP (per level, ≤4 pours)     │
             │  propose → rank → imagine → check → POUR(env)  │
             │       ↑                                │       │
             │       └── TTA adapt on observed spill ─┘       │
             └────────────────────────────────────────────────┘
   L4 PROPOSER      portband-seeded CEM  → later: learned proposer head
   L3 WIN-CHECK     geometric 11→13 / →14 (DECIDER)  + latent win-probe (RANKER)
   L2 WORLD MODEL   tick dynamics (pixel, local CA bias) + settle-predictor (latent jump)
   L1 PERCEPTION    frozen encoder (proven: click 0.6px) + 6 Contra informer planes
   L0 ENV TRUTH     SP80 offline Arcade via arc_play_harness (only grader)
```

| Contra subsystem | Neural realization | Status |
|---|---|---|
| **oracle** (goal inventory) | `head_goal` + the geometric win predicate over colors 11/13/14 | exists |
| **portband** (flow routing) | (i) learned spill dynamics = L2; (ii) proposer prior seeding CEM = L4; (iii) trainable levers stay as symbolic-side knobs | (i) in flight, (ii) Stage C |
| **ghost** (relation memory of satisfied end-states) | settle-predictor latent + win-probe (what "done" looks like), and the per-episode TTA buffer (relations observed *this* episode) | Stage B/D |

The model **grows additively**: `MegarexusCNN.load_state_dict` already
tolerates missing heads (`model.py:334`). New capability ⇒ new head, encoder
frozen. This is the mechanism that lets many small agents add to one growing
model without regressing what's proven.

---

## 3. FROZEN CONTRACTS (violating any of these invalidates the work)

**C1 — Env is the only grader.** A capability claim requires an
env-graded, fresh-from-0 (`random init` per `megarexus-fromzero-training-ruling`)
result through `arc_play_harness.run_episode` / the gates. Imagined wins are
diagnostics, never headlines.

**C2 — Honesty guards run first, always.**
- Every new shard/ckpt path through `no_imitation_guard.assert_no_imitation_lineage`.
- Every report leads with `honesty_audit.py` output (exit 2 = stop).
- NEVER import from `_invalid/`, anything named in `sp80_spill_model.QUARANTINED.md`,
  or the suspect files `sp80_spill_planner_backend.py`,
  `sp80_frame_spill_perception.py`, `sp80_player_harness.py`.
- No `on_set_level` state farming for training data. (`set_level` for
  *presenting* a level to attempt fresh is the sanctioned fan-out mechanism —
  see `contra_fanout_trainer.py:56`.)

**C3 — Encoder freeze discipline.** The perception encoder is proven capital.
Stages A–E train decoders/heads only. Unfreezing the encoder is a HUMAN
decision (write `NEEDS_HUMAN_*.json`, per the intelligent-loop convention).

**C4 — The scoreboard is `imagine_search_gate` + `honesty_audit`.** The gate's
per-level counters are the program's only progress metrics:
- `real_wins` — a winning arrangement existed in the candidate set (proposer quality)
- `imagined_wins`, `tp`, `fn` — imagination found/ranked it (fidelity)
- headline: **levels completed, CNN-alone, fresh-from-0** (currently 1).
Any experiment must state, in advance, which counter it moves.

**C5 — Interfaces.** Don't change these signatures; extend by addition:
- `policy(obs, history) -> (action_id, x, y)` (`arc_play_harness.py:117`)
- `forward_dynamics(frame, action_id, x, y) -> [B,16,64,64]` (`model.py:366`)
- `Imaginer.imagine_pour(arr, arr_prev, max_ticks) -> settled` (`imagine_search_gate.py:123`)
- `win_score(initial, settled)` geometric check (`imagine_search_gate.py:137`)
- Run naming: data/gate stages `stageNN_<desc>_<date>`, trained outcomes
  `outcome_vNN_<desc>_<date>`, under `experiments/megarexus_cnn/runs/`.
- TTA fence: per-seed weight reset + NO-ADAPT control arm on identical
  seeds/candidates, both reported.

---

## 4. STAGED ROADMAP (execute top-down; each stage has a gate and a fallback)

### Stage A — Pour-imagination fidelity (IN FLIGHT: v37 → v38)
- **A1 (v38, queued):** retrain dynamics decoder (encoder frozen) on dense
  spill-tick transitions from the 146k-step pour harvest.
  **Gate:** `imagine_search_gate` L0, 3 seeds: `fn = 0` and `tp ≥ 1`
  (v31 baseline: `fn = 2`, `tp = 0`). Secondary: imagined settled paint mass
  within ±20% of real (~96 cells; v31 imagined ~16).
- **A2 (fallback if A1 fails):** add `head_spill_ca` — shared-weight local conv
  (k=3 or 5, 2–4 layers, per-cell 16-way softmax), applied iteratively;
  trained only on spill ticks; loss masked to cells within radius 2 of paint
  or paint-adjacent change. Wire into `Imaginer.imagine_pour` behind a flag.
  Same gate as A1.
- **A3 (fallback if A2 fails):** escalate to Stage B with the latent path as
  ranker AND re-examine tick alignment/label construction in `dataset.py`
  before concluding the CA is unlearnable (it isn't — it's deterministic).

### Stage B — Latent settle-predictor + win-probe (the JEPA layer)
Amortizes the 40-tick rollout into one forward pass → enables 10–100× larger
candidate sets for Stage C. Independent of A2's outcome; buildable in parallel
if compute allows (Apollo is sequential — then B follows A).
- **B1:** new heads (additive): `settle_predictor` — from pre-pour frame
  encoding, predict the *encoder embedding of the real settled frame*
  (JEPA-style: latent target, stop-grad on target embedding, cosine or L2).
  Training data: EVERY harvested pour, win or lose — abundant, no positives
  needed, self-supervised. `win_probe` — small MLP on predicted settled latent
  → P(win), trained on env-graded pour outcomes (L0 positives + all negatives).
- **B2:** two-tier planner in `imagine_search_gate`: rank ALL candidates with
  `win_probe(settle_predictor(f))`, pixel-roll + geometric-check only top-K
  (K≈16). The geometric check REMAINS the decider (C4/Q4).
- **Gate:** fresh-arrangement ranking AUC on L0 > pour-outcome head's 0.79 AND
  > pixel-rollout ranking; wall-clock per 1000 candidates < 60 s.
- **Honesty note:** the win-probe generalizing to L1+ is unproven (trained on
  L0 positives only) — that is WHY it only ranks and never decides.

### Stage C — Proposer: portband-seeded CEM over arrangements
- **C1:** define the arrangement genome: per movable block, its placed
  position (the pre-pour arrangement fully determines the pour, REPORT §2).
  Sampler = current structured-shifts + random baseline.
- **C2:** CEM loop: population N≈200–1000 (sized by Stage B throughput) →
  fitness = relaxed geometric score on imagined settle
  (`#targets filled − λ·wall_hit`, λ≈#targets) → refit distribution on top
  quantile → iterate ~5–10 gens. Seed generation 0 with arrangements decoded
  from `build_portband_hypotheses` alignments (portband as prior).
- **Gate (the L1 unlock):** on L1, `real_wins ≥ 1` — i.e. the gate's replay
  arm confirms the final population CONTAINS a real winning arrangement
  (this isolates proposer quality from imagination fidelity, per REPORT §6).
  Then combined: env-graded L1 completion within 4 pours ⇒ **headline moves
  to 2 levels CNN-alone** (first movement since the wall).
- **Fallback:** if CEM stalls, the fitness landscape is too flat → increase
  fidelity weight on target-column cells in A2's loss, or add portband's
  lineup-matrix rank as a fitness prior term (still pixel-derived).

### Stage D — Test-time adaptation in the 4-pour loop (AdaJEPA proper)
- **D1:** in `tta_search_gate` (machinery already validated): after each real
  failed pour, push its spill-tick transitions into a small buffer; take
  **1 gradient step per replan** on the dynamics (and settle-predictor if B
  landed) — **final layers only**, latent/pixel prediction loss with stop-grad
  targets, lr small (start 1e-4); re-rank remaining candidates; next pour.
- **Fence (C5):** per-seed weight reset; NO-ADAPT control arm, identical
  seeds/candidates; log adapted-Δ-norm per step.
- **Gate:** on L1+ seeds where the no-adapt arm fails, TTA arm completes
  strictly more episodes across ≥10 paired seeds. If no separation: park TTA
  (AdaJEPA says frozen-near-optimal models don't benefit — that's a fine
  outcome, record it).

### Stage E — Amortization: learned proposer + distilled outcome head
Only after C's gate passes (never bootstrap on unproven imagination).
- `head_proposer` (additive): trained to output CEM's *final population
  distribution* from the initial frame — amortized search, not imitation of
  any human/teacher (the "teacher" is our own honest search; headline claims
  remain env-graded, C1).
- The existing `settle`/`level` pour-outcome heads get retrained continuously
  on the growing env-graded pour ledger — they become the fast pre-filter;
  they can never be the decider for L1+ (no supervision exists; REPORT §7).
- **Gate:** proposer-seeded CEM reaches `real_wins ≥ 1` in ≤ half the
  generations/population of Stage C on a held-out level.

### Stage F — The growing-model flywheel (steady state)
Every level completed honestly emits: new spill ticks (→ A/B data), new
env-graded pour outcomes (→ win-probe/outcome heads), new confirmed relations
(→ ghost). Loop: harvest → guard (C2) → retrain heads (encoder frozen) →
re-run full gate stack → `honesty_audit` → update Scoreboard in `HANDOFF.md`.
This is the "always be training, master the field" ruling, mechanized.

**Decision tree for the next agent, compressed:**
```
v38 gate: fn=0 on L0? ──no──► do A2 (neural-CA head) ──still no──► A3 + B as ranker-decider study
        └─yes──► Stage B (latent ranker) ──► Stage C (CEM, L1 unlock) ──►
                 C gate real_wins≥1? ──no──► C fallback (fitness shaping)
                                   └─yes──► env-graded L1 attempt ──► Stage D (TTA) ──► E ──► F
```

---

## 5. AdaJEPA — exactly what we adopt / reject

| AdaJEPA element | Their spec | Our use |
|---|---|---|
| Latent predictor `f_θ(z, E_a(a))` | predict next latent | Stage B settle-predictor (predict *settled* latent in one jump — stronger than tick-wise for our CA) |
| Adaptation loss | latent pred. error, **stop-grad targets** | Stage B training AND Stage D TTA, verbatim |
| Adaptation scope | **final layers of encoder+predictor only** | same (also satisfies our C3 freeze discipline) |
| Budget | **buffer=5 transitions, 1 grad step/replan** | 1 step/replan; buffer = last pour's spill ticks (~40) |
| Planning objective | `min Σ α_k d(ẑ_{t+k}, z_goal)` | **REJECTED as decider** — needs a goal observation; none exists for L1+ honestly. Optional aux ranker via synthetic goal image (recolor 11→13). Our geometric predicate stays the objective. |
| When TTA helps | distribution shift, low-data | our shift = unseen level geometry; signal = observed spill on THIS level |

---

## 6. Anti-patterns (each has already burned this project once)

1. Headlines from imagined wins, replayed routes, or human-mined lineage —
   `IMITATION_ASSISTED` at best, `INVALID_HEADLINE` if gated (audit enforces).
2. A learned win-decider for L1+ — there is nothing legal to train it on;
   only the geometric predicate decides (probes rank).
3. Retraining the encoder to fix a decoder problem — destroys proven
   perception; C3.
4. TTA without the control arm + per-seed reset — indistinguishable from
   level memorization; the fence IS the result's validity.
5. "It works on the training seed" — every trainer here commits only on
   held-out seed/level (lever-trainer convention); keep that bar.
6. Importing quarantined spill code because it "already solves the CA" — it
   read privileged sprite state; that's the reward-hack lineage, C2.

---

## 7. Report template (use for every stage result)

```
STAGE <A1|A2|B|C|D|E> RESULT — <date> — run <outcome_vNN_...>
honesty_audit: <exit code / classification>   guard: PASS
Counter targeted (C4): <fn|tp|real_wins|headline>   before → after
Gate criterion: <verbatim from roadmap>   PASS/FAIL
Env-graded evidence: <gate JSON path(s)>
Control arm (if TTA): <paired numbers>
Next per decision tree: <stage>
```

---

## ADDENDUM 2026-07-05 ~13:45 — SHAPE-BASED CONTRA-INFORMED IMAGINATION (user design, binding)
Four user steers tonight converge on ONE architecture, correcting the pooled-latent
Stage-B path:
1. The SP80 win is NOT visible on the static board — it must be IMAGINED (pour forward).
2. The win is a SEQUENCE, not a frame: fill AND settle WITHOUT overshoot (calibration /
   "don't overpour"). Proven: peak-frame all-filled = 0.34 precision vs env; terminal
   frame = post-reset. The only valid DECIDER is the full sequence (env levels_completed
   or a sequence-faithful simulator). Single-frame geometric win_score is RETIRED as a
   decider.
3+4. The imagination must be SHAPE-BASED, informed by oracle + portband + ghost:
   - oracle  -> GOAL shape: which U-targets, what "filled" looks like
     (codexlab/oracle.py, contra_head/view_system.build_oracle_pass).
   - portband -> FLOW shape: actor bbox vs each target's open_box + OPEN_DIRECTION —
     does the pour stream ALIGN to enter the U through its opening (and is it calibrated
     to fill, not overshoot). codexlab/contra_head/portband.build_portband_hypotheses.
   - ghost   -> END-STATE RELATION shape: what a satisfied "done" board's relations look
     like (contra_head/episode_memory, ghost.py).

WHY (the level-invariance argument): the pooled win-probe (v47) collapsed the board to a
128-vector, discarding SHAPE -> solved L0 by memorising its win-region as a scalar but is
useless on L1 (saturated, corr 0.04 w/ progress). Shape RELATIONS ("a stream aligned to a
U's opening, calibrated to fill without overshoot") are IDENTICAL across levels. So a
shape-based, portband-alignment-informed imagination is the level-invariant unlock a
pooled scalar can never be. This IS "the CNN maximises Contra concepts."

REVISED PLAN:
 - DECISIVE TEST FIRST: does portband port-ALIGNMENT (actor->target open_direction) +
   oracle goal predict pour WIN on L0 AND generalise to L1 (where v47 failed)? If a
   shape-alignment score separates env-wins on both levels, the shape path is the unlock.
 - Then: SHAPE features (per-target: is a stream aligned to its opening; predicted
   fill-vs-overshoot) become the imagination's substrate — spatial, not pooled — feeding
   a per-target fill-and-settle predictor (sequence-aware). Fitness for the L1 proposer =
   this shape-based imagined win.
 - DECIDER stays the env sequence (4-pour budget). RANKER/PROPOSER = shape-based
   Contra-informed imagination.
Honesty: portband/oracle/ghost are INFORMERS (shape priors), truth-boundaried; the env
grades. No answer injection, no route lineage. [[megarexus-contra-architecture]]
