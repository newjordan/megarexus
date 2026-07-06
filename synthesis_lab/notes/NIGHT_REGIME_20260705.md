# Night regime — legal CNN training around Contra/ghost/oracle (2026-07-05)

User /loop: "operate around the contra mechanisms to validate a legal CNN
training regime, work through the night, don't stop ... train the CNN to operate
maximizing contra/ghost/oracle concepts."

## The regime (legal by construction)
- **Contra is the honest teacher** (decides from pixels, no privileged state, no
  answer injection); the **environment grades**; agent plays each PRESENTED level
  from its start; **held-out levels** validate. Path 3: "Contra teaches, the game
  grades" — not imitation-of-answers. [[megarexus-contra-architecture]]
- **Fan-out over all 6 SP80 levels** (set_level presentation) = diversity that
  kills overfit + exercises deep behaviors. Supersedes "always from 0" for this
  Contra-guided regime; the line that stays = no solved-state/answer injection.
- **Contra as informer** already in the CNN input (16 one-hot + 6 contra planes).

## Loop state / plan (update each wake)
1. DONE: fan-out pipeline + Contra-planes informer proven; random fan-out taught
   physics but NO goals (catch n=0, no completions) → need PURPOSEFUL data.
2. RUNNING (this box CPU): `collect_fanout_contra.py` — Contra plays all 6 levels,
   generating completions/pours/catches. L0 = 20/20 cleared. Shards:
   `data/fanout_contra_{train,val}.pt`. Run dir stage71.
3. RUNNING (Apollo 4080): stage70 v29 random-fanout fresh-from-0 (epoch ~28/40).
4. NEXT: when (2) lands + (3) frees the 4080 → fresh-from-0 train on the
   Contra-guided shards (Contra-planes informer) → **6-level completion gate**
   (Contra-informed CNN vs level-0-only baseline) → diagnose weakest → collect
   more targeted purposeful data / add oracle goal planes → repeat.

## Compute
- TRAIN on Apollo 4080 (`/home/frosty/Loom/.venv/bin/python`, 15GB RAM → keep
  corpus lean/subsampled). Spark (this box, dgx-spark) freed for user's GPU work;
  CPU-only Contra collection here is OK. turbo blocked (tailscale ACL, TCP/22).
- Every run: fresh-from-0 (`init:none`), pre-registered CONDITION.json,
  train_watch, guard-clean, held-out validation. honesty_audit leads reports.

## Honesty guardrails (never cross)
No answer/route/solved-state injection; env is the only truth; no reward-hack
(sp80_spill_model is quarantined); commit/headline only env-verified,
generalizing gains. [[megarexus-lab-honesty-rules]] [[megarexus-fromzero-training-ruling]]

## FINDING 2026-07-05 (~02:15): CNN-alone cannot complete levels
Fan-out + Contra-informer regime VALIDATED for perception (v31: click 0.6px,
level tp 3/3, dynamics ~1.0 across all 6 levels). But the completion gate is
brutal and consistent:
- CNN hand-coded policy (megarexus_live_policy): **0/18**
- CNN BC policy, Goose-sample decode: **0/18**
- CNN BC policy, argmax decode: **0/18**
BC loss trained fine (8.3→0.74) — the head predicts Contra's action TYPE — but
the CNN completes ZERO levels, below even the teacher (Contra clears L0 20/20).
CONCLUSION: a feedforward CNN can't replicate Contra's DELIBERATION (oracle goal
+ portband plan + multi-step course-correction). Perception is CNN-able; the
deliberative POLICY is not, via hand-code or BC. Next: DAgger (fix BC covariate
shift on the CNN's own states) OR accept the hybrid (CNN perception + Contra
deliberation) OR a learned planner (unrolled dynamics toward the goal head), not
a reactive policy. This is a genuine fork for the user.

## FINDING 2 (~03:50): DAgger also fails — reactive-policy space exhausted
DAgger (retrain policy head only on Contra's actions labeled on the CNN's OWN
visited states, init v33) → **0/18**. The CNN==Contra action agreement was
1.7-3.4% on most levels' states (54% on L5) — real covariate shift — but fixing
it did NOT lift completions. So the wall is NOT (only) covariate shift.
DEFINITIVE: every reactive frame→action approach = 0/18 (hand-coded, BC small,
BC big, BC argmax, DAgger). Completion needs MULTI-STEP deliberation (arrange
over several steps, then pour toward the goal) — a learned PLANNER using the
CNN's own (excellent) dynamics + level/goal head, not a reactive policy. That is
the remaining fork and a MAJOR build worth the user's direction, not a rushed
autonomous 4am attempt. Tools: bc_gate.py, collect_dagger.py, fanout_gate.py
(--policy contra/cnn). Runs v29-v35 on Apollo. Contra ceiling itself = L0 only.

## FINDING 3 (~04:15): PLANNER FOUNDATION VALIDATED — imagination is accurate
`eval_ghost.py` ghost_rollout on v31 (autoregressive H-step, feeds its OWN
prediction forward) over 40 fan-out windows:
  h1 changed_acc 1.00 iou 0.99 | h3 0.89/0.87 | h6 0.92/0.89 (cell 0.98-0.9999).
Errors DON'T compound — the CNN imagines 6 steps ahead faithfully. So the CNN's
WORLD MODEL is good enough to PLAN on. This is the missing piece: reactive
policy failed (deliberation gap), but neural deliberation via imagination is
FEASIBLE.

## THE HONEST ARCHITECTURE (next build): Contra proposes, CNN imagines, env judges
Completion = deliberation. The honest planner fuses what each part is good at:
- **Contra (oracle goal + portband lineup)** PROPOSES candidate arrangements/plans
  (its deliberation — legal, from pixels, no answer injection).
- **CNN world model** IMAGINES each candidate (k-step forward_dynamics rollout,
  validated accurate) and SCORES it with the level-up / settle / goal heads —
  the LEARNED imagination that replaces the quarantined hardcoded spill model.
- **Env** is the only truth; execute the best-scored plan, judge by completion.
This is "CNN operates maximizing Contra concepts": Contra's deliberation + the
CNN's learned world-model imagination, no reward-hack. Build: planner_gate.py —
per step, get Contra's candidate plans, imagine+score each with the CNN, execute
best, MPC-replan. Gate vs Contra L0 3/3 ceiling.

## FINDING 4 (~04:45): planner v1 built, 0/2 on L0 — needs real design iteration
`planner_gate.py` built: MPC — imagine candidate arrangements (click+move via
forward_dynamics), score by settle/level head, pour when imagined-completion is
confident. Runs, but 0/2 on L0. Honest reasons (the design work ahead):
1. CANDIDATES too shallow (3 blocks, horizontal 1-move) — completing needs
   multi-cell multi-block arrangement. Better: draw candidates from Contra's
   portband LINEUP MATRIX (its deliberation), not a crude enumeration.
2. SCORING mismatch: forward_settle.level_logit was trained on actual POUR
   frames; scoring arbitrary non-pour arranged frames with it is off-calibration.
   Need a valid "would a pour here complete" signal on imagined states (roll the
   pour animation via dynamics, or a goal-fill check on the U-targets the oracle
   locks).
3. SPEED: encode() recomputes contra_planes ~12-48x/step → must batch/cache for a
   real run.
STATE: imagination substrate validated (accurate), planner architecture correct,
v1 not yet completing. This is the major build for the user's direction:
Contra-portband candidates + CNN imagination + calibrated goal scoring. NOT a
rushed 4am fix. Perception ✓, reactive-policy ✗ (0/18), imagination ✓, planner
scoped+v1. Tools: planner_gate.py, bc_gate.py, collect_dagger.py, fanout_gate.py,
eval_ghost.py rollout.

## FINDING 5 (~05:20): ROOT CAUSE of planner failure — level head saturates
Diagnostic on v31, level 0: forward_settle.level_logit → sigmoid = **1.000 at
EVERY state**. The completion head saturates (trained with high LEVEL_POS_WEIGHT
for rare positives → high recall, ~0 precision). So the planner thinks any pour
completes → pours immediately every step → burns pours → GAME_OVER by t5. The
bottleneck is NOT search depth — it's an UNINFORMATIVE goal signal.

**The concrete fix (next planner build):** a DISCRIMINATIVE completion score,
one of:
(a) GOAL-FILL grounding: detect the U-targets (ghost_goal_confirm.detect_u_targets
    — already built), imagine the pour via forward_dynamics (validated accurate),
    score = fraction of U-target interiors that turn paint color. Grounds the
    plan in the actual goal geometry, not the saturated head.
(b) RECALIBRATE the level head: retrain it with hard negatives (arrangements that
    do NOT complete) so it discriminates, not just fires 1.0.
(a) is preferred — it reuses the validated dynamics + the oracle's U-lock and
avoids retraining a biased head. This is the next concrete, bounded step.
Perception ✓, reactive ✗, imagination ✓, planner architecture ✓, goal-signal is
the gap. Tools: planner_gate.py, ghost_goal_confirm.py, eval_ghost rollout.

## FINDING 6 (~05:25): THE definitive gap — the CNN can't imagine a POUR
Probe: imagined pour (forward_dynamics action5 rollout) from L0 start stabilizes
at ~16 paint cells; the REAL pour has 96. The CNN's dynamics do NOT capture the
pour flow/fill. General/move imagination is accurate (6-step 0.9) but the POUR —
the critical completing action — is not. And both pour-OUTCOME heads are dead on
pre-pour frames: level saturates 1.0, settle fires ~0.

COMPLETE HONEST MAP (night 2026-07-05):
  perception ✓ | reactive policy ✗ (0/18) | move-imagination ✓ |
  POUR-imagination ✗ | pour-outcome heads (settle/level) ✗ (uninformative)
ROOT CAUSE: too little POUR-OUTCOME data — random play rarely pours meaningfully,
Contra only completes L0, so the pour-flow dynamics + settle/level heads are
undertrained. The planner (imagine+score pours) CANNOT work until the CNN learns
the pour.

CONCRETE NEXT DIRECTION (the real build for the user): collect targeted
POUR-OUTCOME data — pours from many DIVERSE arrangements (move blocks to varied
positions, then pour) recording the full 20-frame fill + the env outcome (which
U's filled / level-up). Train forward_dynamics (pour-flow) + settle + level on it
so: (a) the CNN imagines a pour's fill accurately, (b) the settle/level head
discriminates "would a pour from THIS arrangement complete". THEN the planner
(imagine arrangement -> imagine pour -> score U-fill) becomes viable. This is the
crux and the clear next step. Everything else (perception, move-imagination,
architecture, tools) is in place.

## FINDING 7 (~06:40) — saturation fixed; the real wall is L1+ has NO completing-pour source
The pour-outcome data fix WORKED at its stated goal: the settle/level head no
longer saturates at 1.0. v36 (retrain settle_conv/head_settle/head_level from
v31 on balanced pour data) drove train `level` loss 0.62->0.14 and val (on the
pour distribution) to tp=3/3 fp=0 — it now DISCRIMINATES completing vs
non-completing pours instead of firing everywhere.

BUT probe_pour_oracle.py on FRESH L0 arrangements (120 seeds, no leakage: score
read pre-pour, env grades post-pour) showed AUC=0.79 with completing scores tiny
(0.007 vs 0.005). Diagnosis: OVERFIT — only ~23 completing positives in v36's
training set, memorized (val, same dist, scores >0.5) but weak generalization to
fresh completing arrangements. Confirmed the label semantics are RIGHT though:
dataset.py attaches `level_up` to the pour transition's PRE-pour frame key_of(i)
with target = did-this-pour-increment-levels. So the level head IS a legitimate
pour-outcome oracle (arrangement -> P(pour completes)); it just needs more/diverse
positives, not a new target.

THE STRUCTURAL WALL (harvest pour_outcomes_v2, 1600 eps/level, arrange-then-pour):
- L0: 46/1600 completed (~3%) — positives exist.
- L1: 0/1600. L2: ~0 (running). Random arrangement NEVER completes L1+.
- Contra (the teacher) also completes only L0 (L1-5 = 0, established earlier).
=> There is NO legal source of L1-L5 completing-pour examples. Neither the student
(random/CNN exploration) nor the teacher (Contra) can produce one. A head/planner
can only learn levels that HAVE completing examples => L0 only, today.
Likely cause: L1+ needs DURING-pour catch-and-redirect (interleaved action while
the pour animates), which the simple "arrange then single pour" cannot express;
and pour-IMAGINATION is broken (16 vs 96 cells) so the planner can't search L1+
by imagining the fill either. Both dead-ends converge on the same frontier.

ACHIEVABLE HONEST WIN (in progress): v37 = retrain settle/level on
pour_balanced_tr + pour_outcomes_v2 (~69 L0 positives, more diverse) -> re-probe
generalization -> run planner_gate on L0. Target = CNN deliberates (imagine
arrangement + score with learned pour-outcome head) to an ENV-VERIFIED L0
completion. That is "the CNN operates maximizing Contra/oracle concepts" as neural
deliberation, honestly, on the one level with a real training signal.
FOR THE USER: cracking L1+ needs either (a) a teacher that actually solves L1+
(improve Contra's portband/catch-redirect so it completes higher levels -> then it
can teach), or (b) fixing pour-imagination so the planner can SEARCH L1+ without
labeled positives. Both are real projects; (a) leans on your mechanic knowledge
(level-3 sidewalls, catch-and-redirect, lock/key goal). [[megarexus-cnn-deliberation-gap]]

## FINDING 8 (~07:35) — SP80 mechanic DECODE: pour is a deterministic CA; L1+ path is honest
Read the real game (environment_files/sp80/589a99af/sp80.py). Decoded tags:
- plzwjbfyfli = movable BLOCKS (8=idle/9=selected). Stream hitting a block SPLITS
  sideways (the spreader — lines 764-771).
- tuvkdkhdokr = REDIRECT blocks (color 15). Stream hitting one DEFLECTS 90°
  (the catch-and-redirect mechanic — lines 787-811).
- repwkzbkhxl = TARGET Us (color 11). Filled when a stream nestles into the U
  (-> 13, added to `cevwbinfgl` satisfied set — lines 772-778).
- waoewejnqzc = WALLS (color 1 -> 14 on contact). Stream hitting a wall sets
  kfdcqkodyy=True = FAIL/overflow (lines 812-816).
- sowlljgtjvn = SOURCE emitter (pixel 4 = pour origin).
WIN (lines 727-748): when the spill settles, next_level() iff
  (not kfdcqkodyy) AND (ALL targets in cevwbinfgl) = every U filled AND no wall hit.
ATTEMPTS: lyremoheq>=4 -> lose (line 717) => up to 4 pour tries/level; plus a
global ~50-action step budget (zlhbnhpcq).

CORRECTION TO FINDING 7's speculation: in "spill" mode step() IGNORES player
actions (lines 727+) — the pour runs AUTONOMOUSLY. The player CANNOT act during
the pour. So "arrange-then-pour" is the CORRECT action model (my collector is
structurally right). The outcome is FULLY DETERMINED by the pre-pour arrangement.
L1+ is hard only because WINNING ARRANGEMENTS ARE RARE (more Us + spreader/redirect
placement precision), not because it needs during-pour control.

=> THE HONEST L1+ UNLOCK (no imitation, no answers): the pour is a DETERMINISTIC,
LOCAL cellular automaton (fall / split-on-block / deflect-on-redirect / fill-on-U /
fail-on-wall). That is exactly what a CNN forward_dynamics head can learn from
observed spill-tick transitions — needs NO win/lose labels, and the CA rules are
the SAME on every level, so it generalizes to L1+ with zero L1+ "positives".
Then the planner searches arrangements: propose -> IMAGINE the spill to settle ->
GEOMETRIC win-check (all Us filled, no wall colored 14) -> pour for real (<=4 tries,
env grades). L1+ needs NO learned positives — only accurate spill-imagination + a
geometric check. This is why pour-IMAGINATION (forward_dynamics on spill ticks),
not the pour-outcome head, is the general lever. The earlier 16-vs-96 pour-imagine
failure was almost certainly UNDER-EXPOSURE: spill ticks are rare in the old data;
pour_outcomes_v2 has 146k steps of full pour animations = the missing spill-tick
supervision.

TWO-LEVER PLAN:
  v37 (running): pour-outcome head on ~69 L0 positives -> L0 planner win (concrete).
  v38 (queued):  retrain forward_dynamics on spill-tick transitions -> re-measure
                 pour-imagination -> if the CA is learned, planner searches L1+ by
                 imagination + geometric U-fill check. General, all levels, honest.
FOR THE USER: this is the real "CNN operates maximizing Contra/oracle concepts" —
the CNN learns the pour PHYSICS (oracle=which Us, portband=flow routing, ghost=the
settled shape) as a world model and DELIBERATES by imagining pours. Contra's
portband can also be improved to PROPOSE arrangements (it routes flow) to seed the
search. [[megarexus-cnn-deliberation-gap]] [[megarexus-level3-mechanic]]

## FINDING 8b (~07:55) — imagine_search_gate BASELINE on v31 (the number v38 must beat)
Built imagine_search_gate.py: proposes arrangements (structured horizontal shifts +
random arrangements matching the collector's winning distribution), IMAGINES the
pour (autoregressive forward_dynamics: ACTION5 then ACTION0 ticks to settle),
GEOMETRIC win-check (all targets 11->13, no wall ->14), and records BOTH imagined
and REAL env outcomes per candidate => measures imagination FIDELITY on the actual
decision. v31, L0, 3 seeds, 129 candidates:
  real_wins=2 (seeds_with_a_real_win=2/3) — search set DOES contain winning
  arrangements (space is adequate). imagined_wins=0, fn=2 => v31's world model
  MISSES every real win. So the planner can't work on v31's imagination — exactly
  the 16-vs-96 pour-imagine failure, now measured end-to-end on decisions.
=> v38 (forward_dynamics retrained on spill ticks) is graded by whether fn drops
and tp>0 on this gate. If it does, the imagination+geometric planner becomes viable
and (because the CA is level-invariant) extends to L1+ with no learned positives.

## FINDING 9 (~08:50) — TTA (AdaJEPA idea 2) integrated; plumbing OK, efficacy test needs v38
User direction: integrate AdaJEPA "where effective", with testing+calibration.
Built tta_search_gate.py: propose arrangements -> imagine pour -> geometric
win-check -> pour best for REAL -> on a MISS, adapt forward_dynamics (decoder only,
trunk frozen) on the OBSERVED spill, replan; up to 4 attempts (the game's real
budget). HONESTY controls: model RESET to init per seed (no cross-episode
memorization) + a no-adapt CONTROL arm on identical seeds/candidates. Legal: adapt
DYNAMICS on env's own response, never an injected answer.
Smoke on v31, L0, 2 seeds, 10 random cands: NO-ADAPT 0/2, TTA 0/2, pre-adapt dyn
CE=0.40. INCONCLUSIVE (not negative): the small candidate set contained no winner
for these seeds, and v31 imagination can't rank one anyway. Plumbing validated
(runs end-to-end, adapt loss computes, control arm works).
REFINED TEST DESIGN for the v38 base: (1) bigger candidate set (60-100 random) so
a winner is reliably present; (2) purer TTA metric = adapt on K pours, measure
dyn-CE on M HELD-OUT pours before/after (does 1-2 steps improve fidelity, per
AdaJEPA's claim) — decouples TTA-efficacy from whether a win is in the set;
(3) then the end-to-end solve-rate TTA-vs-control. SEQUENCING: v38 verdict first
(does offline spill-tick retrain fix imagination on imagine_search_gate?) ->
if yes, TTA sharpens it within budget; if no, latent prediction (AdaJEPA idea 1).
[[megarexus-adajepa-reference]]

## FINDING 10 (~10:00) — L0 headline GROUNDED (=1, CNN-alone, env-graded); ranker is the L0 bottleneck
head_rank_gate on v37 (pour-outcome head as RANKER, not imagination), L0, 5 seeds,
60 candidates, top-4 real pours (the game's real budget), env grades:
  SOLVED_CNN_ALONE 1/5 | proposer_had_a_win 4/5 | head_ranked_a_win_into_topK 1/5.
=> Honest, env-graded, CNN-alone L0 completion exists (grounds roadmap C4 headline
   >=1 — previously assumed, now measured). No imagination trusted for the decision;
   the head only ranks real arrangements; no answer injection.
=> DIAGNOSIS: the PROPOSER is adequate for L0 (a winner is in the 60-candidate set
   on 4/5 seeds) but the RANKER is the bottleneck — v37's head (fresh AUC 0.87)
   surfaces a winner into the top-4 on only 1/5. This is precisely the roadmap's
   Stage-B motivation: a stronger latent ranker (settle-predictor + win-probe) to
   lift winners into the pour budget. Holds behind the v38 A1 gate per the decision
   tree (Stage B follows A). [[megarexus-worldmodel-roadmap]]

## FINDING 11 (~10:57) — STAGE A1 (v38 decoder retrain) FAILS as predicted; branch to A2
Early A1 gate on v38 (epoch 5, 1-step dyn loss already saturated 0.0071):
  L0 fn=2 tp=0 (== v31 baseline) | paint_mass imagined=752 real=113 ratio=6.66.
The retrain FLIPPED the failure: v31 UNDER-imagined the spill (~16 vs ~96 cells);
v38 OVER-imagines it 6.66x — the autoregressive rollout FLOODS the grid instead of
settling. 1-step loss is saturated while the ROLLOUT diverges => more epochs cannot
fix it (1-step CE doesn't penalize rollout divergence). This is exactly ROADMAP Q1:
the global-FiLM full-receptive-field decoder has the WRONG BIAS for a local CA
("smears global context into a local rule"). DECISION TREE => Stage A2:
`head_spill_ca` — a shared-weight LOCAL conv (k=3, few layers, per-cell 16-way
softmax) applied ITERATIVELY, trained only on spill-tick pairs, loss masked to
changed cells (paint-adjacent). Level-invariant by construction. Building A2 now
(additive head, encoder frozen, C3/C5); v38 finishes for the formal final gate.
[[megarexus-worldmodel-roadmap]]

## FINDING 12 (~11:20) — STAGE A2 built + training (v39 spill_ca); smoke validates the local-CA bias
A1 formally closed: v38 killed at epoch 15 (1-step saturated, rollout 6.66x overshoot;
confirmatory gate running). Built A2 end-to-end and it VALIDATES the roadmap's Q1 call:
- model.py: additive `spill_ca` head (47k params; 5x5 RF local conv; residual/identity
  init; C3/C5-clean, load_state_dict allowlists it).
- train_spill_ca.py: trains ONLY spill_ca on spill-tick pairs, masked CE on the active
  front (changed cells dilated r2). CPU smoke (L0, 55k pairs, 3 ep): front_acc
  0.84->0.88->0.90 rising — the local CA learns the per-cell pour rule fast.
- imagine_search_gate.py: `--ca` rollout mode (iterate spill_ca; source emission is a
  learned local rule, no action conditioning).
FULL A2 = **v39** on GPU: init v31, shards pour_balanced_tr + pour_outcomes_v2 (ALL
levels -> 130,601 spill-tick pairs, so the CA sees redirect/wall/split primitives that
simple L0 lacks -> level-invariant). epoch 0 val_front_acc=0.858. 30 epochs.
DECISIVE NEXT: gate v39 with `--ca` on L0 -> roadmap A1/A2 gate fn=0 & tp>=1 and
paint-mass ratio 0.8-1.2 (vs v38's 6.66). If it passes -> Stage B/C per the tree.
[[megarexus-worldmodel-roadmap]]

## FINDING 13 (~11:42) — A2 v39 gate FLOODS (ratio 14.2); two design bugs, fixable
v39 spill_ca trained fine (front_acc 0.92) but the --ca gate floods worse than v38:
paint ratio 14.23 (imagined 1607 vs real 113), fn=2. Root causes (both mine, not the
roadmap's local-CA thesis): (1) the ×6 IDENTITY RESIDUAL biases paint cells to
PERSIST, but paint must MOVE (clear old + fill new) -> trails accumulate; (2) the
rollout applies the CA to the WHOLE grid each tick — roadmap's "hard-masked to cells
near active paint" means only the ACTIVE FRONT updates, rest copies. FIX (v40):
forward_spill_ca returns raw logits (no residual); imagine_pour --ca updates only
active = dilate((cur!=arrangement) | (cur==source), r2), copies the rest. Same masked
CE trainer. A1 formally closed (confirm gate v38 ep15: fn=2 ratio 6.12).

## FINDING 13b (~11:50) — correction + rotation hypothesis
Spill-color probe (L0): color 6=PAINT, 4=source, 12=background. Paint ACCUMULATES
+16 cells/tick (a growing fill, not a moving/clearing particle) — so the identity
residual is CORRECT, not a bug (contra 13). Masked rollout (update only
dilate(paint|source, r2)) added: helped 14.2->11.0 but v39 STILL floods => the CA
OVER-PRODUCES paint per tick (small over-rate compounds over 40 ticks). Leading
hypothesis: v39 trained on ALL levels, but the game ROTATES pour direction per level
(fahhoimkk = dojfslwbg//90%4). A pixel-local CA can't infer the global rotation, so
multi-level training teaches an omnidirectional spread => flood. This breaks the
roadmap's "level-invariant by construction" UNLESS rotation is canonicalized/
conditioned. TEST: v40 = CA trained on L0-ONLY (single rotation) -> gate --ca on L0.
If clean (ratio ~1) -> rotation confirmed AND L0 pour-imagination solved (the level
we can complete); rotation-canonicalization is the L1+ extension. If still floods ->
deeper; escalate per tree toward A3 (re-examine tick alignment / Stage B ranker).

## FINDING 14 (~12:18) — learn-to-stop cuts flood 11->2.24; remaining blocker = rare 11->13 fill
v41 (settled frames included, paint region always supervised): front_acc 0.959,
--ca gate paint ratio 2.24 (was 11 v39, 7 v40). The CA now roughly STOPS — the
autoregressive rollout is basically stable. But fn still 2, imagined_wins=0: the
geometric win-check keys on color 13 (satisfied target), produced by the LOCAL
11->13 nestle transition (game line 775) that occurs ONLY in winning pours
(~46/1600 L0). The CA barely sees it -> under-produces 13 -> imagined winners never
show 13 -> win-check never fires. Pour-positive scarcity, now at the CA level.
FIX v42: upweight rare win/fail transitions (next-cell 13 or 14) heavily in the
masked CE so the CA learns the level-determining events from few examples. If v42
yields tp>0 the A2 gate essentially passes on L0; if not, escalate to A3/Stage-B
one-shot settle-predictor. Flood 2.24 (over-paint) is secondary — win-check needs
13/targets_left, not paint mass.

## FINDING 15 (~12:35) — A2 CA rollout CONCLUSIVELY insufficient; pivot to one-shot settle (A3/Stage-B)
v42 (13/14 upweighted): val_fill13_recall 0.90 TEACHER-FORCED — the CA CAN produce
the win-transition. But --ca gate still fn=0, imagined_wins=0, paint ratio 2.23. So
the CA predicts 13 given REAL frames, but in the AUTOREGRESSIVE ROLLOUT its own
over-painted frames drift the geometry -> the precise 11->13 nestle condition never
triggers -> imagined winners never show 13. Four CA variants (v39 masked-front,
v40 L0-only, v41 learn-to-stop 11->2.24, v42 fill-upweight recall0.90) each fixed a
metric but the rollout DRIFT is the irreducible blocker. CONCLUSION: an autoregressive
pixel-CA rollout is too unstable for exact win-detection. Per roadmap A3/Stage-B:
STOP ROLLING OUT — predict the SETTLED frame in ONE SHOT from the arrangement
(global decoder is fine for a single global prediction; it only failed at rollout).
v43 = train forward_dynamics decoder (encoder frozen) on (arrangement + ACTION5 ->
settled_frame=last animation frame) pairs; gate --oneshot = one forward pass.
Keeps pixels-decide (geometric win-check on the one-shot settled frame).

## FINDING 16 (~13:10) — ROOT CAUSE: a LABEL bug starved every imaginer of wins (not rollout, not memorization)
v43 (one-shot, ORIGINAL labels) converged to val_acc=1.0000 but val_fill13_recall
stayed 0.0000 for all 40 epochs. Dug into the shard `pour_outcomes_v2_l0.pt` (1600
L0 pours). TWO env-reading bugs, both mine, both now fixed:

**BUG 1 — settled-frame extraction took steps[-1] (post-RESET garbage).**
Traced traj1 frame-by-frame: after the ACTION5 pour, paint(6) grows +16/tick and
PLATEAUS (win: 608, fail: 224). On a COMPLETE win every one of the 160 target cells
flips 11->13 AT the plateau (i=26: n11=0, n13=160) and is HELD for several frames,
THEN the episode RESETS back to the arrangement (i=34: n6=16, n13=0, n11=160). So
steps[-1] is the post-reset frame with ZERO satisfied targets. Every imaginer this
session (v38 global, v39-42 CA, v43 one-shot) was trained on that frame => never saw
a single 11->13 win transition. Not a structural wall — the wins were in the data
(138 complete + 46 partial = 184/1600 = 11.5%), the extraction discarded them.
FIX: settled frame = FIRST frame at max paint mass after the last pour
(pour_settle.settle_index = argmax(n6)). Lands exactly on the held win/fail plateau
for both outcomes. New shared module `orchestration/pour_settle.py` is the single
source of truth for trainer AND gate.

**BUG 2 — win_score keyed 'no win' on the presence of color 14.**
Color 14 is STRUCTURAL border geometry: present in the arrangement (~62 cells) and
FLICKERING during the win-flash (up to 303). It is NOT a wall-hit/fail marker. The
gate required `not wall_hit` (== no 14) for a win, so it false-rejected EVERY real
win. FIX: win = some FILLED(13) AND no unfilled TARGET(11) left; 14 ignored (reported
as border14 diagnostic only). Real-win ground truth in the gate is unaffected — it
reads the env's own levels_completed increment.

CONFIRMATION: v44 (one-shot, FIXED labels) — extracted "138 complete-win settled
labels"; val_fill13_recall 0.0000 -> 0.75 by epoch 4 (was flat 0.0 for all 40 on bad
labels). The model now learns the win physics. Both the one-shot (v44) and the local
CA (v45, extract_pairs now windowed to [pour_idx, settle_idx] + a fixed-point pair,
excluding the reset/flicker frames that had taught it to UN-satisfy 13->11) get
retrained. The FINDING-15 "rollout drift is irreducible" conclusion is now SUSPECT:
the CA fought contaminated labels (reset frames + post-reset settles). Re-test A2.
Legal throughout: pure forward-physics reading of observed env transitions.

## FINDING 17 (~13:06) — v44 one-shot gate: machinery FIRES but global predictor can't discriminate
Corrected gate (fixed labels + fixed win_score, (arr,arr) parity) on v44 one-shot,
L0/3-seeds/30-random-cands:
  candidates_scored=129 real_wins=2 imagined_wins=21
  tp=0 fp=21 fn=2 tn=106 | agree_acc=0.822 | seeds_with_a_real_win=2/3
  paint_mass imagined=509.5 real=112.9 ratio=4.51
imagined_wins 0->21 proves the label/win_score fixes work end-to-end (it FIRES now).
But it FAILS: tp=0 (imagined wins don't match real wins), fp=21 (hallucinates wins),
fn=2 (misses BOTH real wins), ratio 4.51 (OVER-FLOODS 4.5x). The global one-shot
learned a spurious "pour -> big flood -> win" shortcut: it predicts a near-win flood
for most L0-ish arrangements regardless of the actual geometry, so imagined wins are
uncorrelated with real wins. Confirms the roadmap's reason to prefer the LOCAL CA:
a global decoder pattern-matches to a memorized canonical outcome instead of
simulating arrangement-conditional physics. -> test v45 (local CA, clean labels):
it can only change the active front (can't flood 4.5x) and must simulate real
geometry. v45 teacher-forced fill13_recall ~1.0 from epoch 0 (clean labels beat
v42's contaminated 0.90); the --ca autoregressive gate is the real discrimination test.

## FINDING 18 (~13:20) — Stage B latent probe: oracle latent SEPARATES PERFECTLY, but fresh-arrangement generalization fails (distribution shift)
Added Stage-B heads (model.py, additive): settle_predictor (arr pooled latent ->
predicted settled pooled latent, JEPA cosine, stop-grad target, self-supervised) +
win_probe (MLP on predicted settled latent -> P(win), env-graded labels). Trained v46
on pour_outcomes_v2_l0 (1600 pours, 138 wins), trunk frozen.
  KEY: val_AUC_ORACLE = 1.000 (win_probe on the TRUE settled latent perfectly
  separates win/lose) — the Contra-informed pooled latent FULLY encodes the win
  structure. settle_cos 0.974. Deployable val_AUC_pred (probe on PREDICTED latent)
  = 0.792 (held-out from training) — beats the saturated pour-outcome head (0.500)
  and every pixel imaginer.
BUT the honest gate is FRESH-arrangement AUC (win_probe_gate.py: gate's _candidates
generator + real pours, env-graded). Result on L0/5-seeds/455 candidates/31 real wins:
  FRESH ranking AUC = 0.581 (near random); wins in top-16 = 0/31; winner ranks
  cluster 38-52/90 (below middle).
=> the probe MEMORISES the harvested-pour distribution (0.79) but does NOT generalise
to the proposer's arrangement distribution (0.58). Classic train/test distribution
shift: pour_outcomes_v2_l0 came from a different policy than the gate's structured-
shift+random candidates. Oracle 1.0 proves the latent CAN separate; the failure is
predicting a fresh arrangement's settled latent. FIX (in progress): harvest an
ON-DISTRIBUTION pour shard with the SAME _candidates generator the gate/proposer uses,
retrain the probe on it, re-gate. Legal: random env-graded pours, no answers/routes.

## FINDING 19 (~13:29) — BREAKTHROUGH: L0 SOLVED CNN-alone 8/8 at rank-1, via latent imagination (user's insight, mechanised)
User steer (2026-07-05): "there is no possible way to see the win condition on SP80 by
looking at the board — you have to IMAGINE the paint pouring and track the win in your
imagination." This is exactly the mechanism. It explains tonight's whole arc:
  - oracle latent on the TRUE settled frame separates win/lose (AUC 1.0) — win is clear
    ONCE imagined; the static board is not.
  - the deployable probe is settle_predictor (imagine the pour in LATENT space) ->
    win_probe (read the win in the imagination). It is NOT a static board-read.
  - v46 (off-distribution training) imagined poorly on fresh arrangements -> AUC 0.58.
  - v47 (trained on ON-DISTRIBUTION harvest, pour_ondist_l0, 3550 pours/318 wins, seeds
    5000-5049) imagines the actual proposal distribution -> FRESH AUC 0.963-0.980 on
    disjoint seeds 0-7.
END-TO-END honest solver (win_probe_gate, L0, 8 seeds, 60 random-cands): rank each
seed's structured+random proposals by imagined win, pour the top-K for REAL, env grades.
  solved@1 = 8/8, solved@2 = 8/8, solved@4 = 8/8. The CNN's #1 pick is a real
  env-confirmed winner in EVERY seed. winner_ranks cluster 0-7.
Legal: proposals are structured-shift+random (a search, not the answer); the ranker is
trained on env-graded forward physics from a DISJOINT seed set; the env grades. No
route lineage, no answer injection. Pixel-CA rollout STILL can't imagine the win even
with win-tracking (v45 --ca imagined_wins=0, ratio 2.25) — the working imagination is
LATENT, not pixel. "Latents rank, env decides (within the 4-pour budget)."

## FINDING 20 (~13:30) — L1 frontier is a PROPOSER problem, not a ranker problem
Same gate on L1 (L0-trained v47 probe): candidates=776 across 8 seeds, real_wins=0.
The structured+random proposer NEVER produces a winning L1 arrangement -> can't test
ranker generalisation (no positives to rank). This is precisely the roadmap's Stage-C
split: the L1 unlock needs a stronger PROPOSER (portband-seeded CEM over arrangements)
that can find a winning configuration where random cannot. Ranker fidelity is not the
L1 blocker; proposal coverage is. NEXT: Stage C — CEM proposer seeded by
build_portband_hypotheses, fitness = v47 imagined-win score (+ relaxed geometric term),
gate = real_wins >= 1 in the final population, then env-graded L1 completion within 4
pours => headline moves to 2 levels CNN-alone.

## FINDING 21 (~13:32) — HONESTY PRECISION: SP80 level boards are FIXED (seed-invariant)
Checked: L0 (and L1) board is IDENTICAL across seeds 0,1,2,3,5000,5001 (pixel diff=0).
The seed varies only the rng that draws CANDIDATE ARRANGEMENTS in _candidates, not the
layout. So FINDING 19's "solved@1=8/8 disjoint seeds" precisely means: on the ONE fixed
L0 board, across 8 INDEPENDENT random candidate searches, the imagination-ranker's #1
pick is a real env-confirmed winner every time. That is robustness over SEARCHES (and
over fresh candidate arrangements the probe was not trained on — train seeds 5000-5049
vs test 0-7 draw different perturbations of the same board), NOT board/level
generalisation. Honest scope: the ranker learned L0's win-REGION of arrangement-space
from env-graded pours and reliably finds a winner in it. It is NOT evidence the
imagination transfers to a NEW layout — that is the L1 question, which is blocked
upstream at the PROPOSER (0 wins found by random search on L1's fixed board). Claim
stays: "L0 solved CNN-alone by honest imagination-ranking"; do NOT upgrade to "levels
generalise" until L1 is env-completed via Stage C.

## FINDING 22 (~13:41) — USER: the win is a SEQUENCE, not a frame (overshoot/calibration). Single-frame win-checks are INVALID.
User steer #2 (2026-07-05): "the win condition is not one frame... it's a sequence."
Confirmed hard on L0 (284 pours, 22 env wins), testing win-oracles vs env
levels_completed (the sequence truth):
  PEAK frame all-targets-filled: 65 say-win, 22 real -> precision 0.34 (over-counts 3x)
  TERMINAL (last) frame all-filled: 0 say-win -> the animation list ends POST-RESET
    (n13=0, n11=160 = arrangement); last frame is never the win.
  ENV sequence: precision 1.00 (ground truth).
NO single frame indicates the win. WHY an arrangement fills every target yet loses:
OVERSHOOT. cand5 (lose) poured 608 paint/28 frames; cand6 (win) poured 480/20 and
STOPPED. The win = fill AND settle WITHOUT excess = calibration (ties to
[[megarexus-level3-mechanic]] "don't overpour"). It is a temporal property.
CONSEQUENCES:
 (1) imagine_search_gate's single-frame geometric win_score is INVALID as a decider
     (<=0.34 precision) — it cannot distinguish fill-and-win from fill-and-overshoot.
     Retire it as a decider; the DECIDER must be the full sequence (env levels_completed,
     or a sequence-faithful simulator). This ALSO explains why the pixel imaginers
     "failed": judged by an invalid single-frame check AND unable to model overshoot.
 (2) the honest L0 solver is unaffected — it DECIDES with env levels_completed (full
     sequence) and only RANKS with v47. v47 is an AMORTISED SEQUENCE-OUTCOME predictor
     (arrangement -> does the pour fill-and-settle), trained on env sequence labels —
     which is exactly why it beat every single-frame pixel check.
 (3) fill-fraction gradient via PEAK n13 is contaminated by overshoot (34% of peaks
     aren't wins) — a sequence-aware progress signal is needed for a CEM fitness.
Data capture note: fd.frame from ONE step(5) is the WHOLE animation LIST (~20-28
frames); _grid takes f[-1] (post-reset) — that is why harvest_pour_shard's settled
frames were degenerate (= arrangement) and settle_predictor trained to identity, making
v47 effectively a direct arrangement->win classifier (still env-labeled, honest, 0.96).

## FINDING 23 (~13:44) — sustained-fill gradient exists but doesn't separate L1 win; USER: imagination must be SHAPE-based (oracle/portband/ghost)
Sustained-fill (max n13 HELD >=3 consecutive frames / targets) vs env win:
  L0: win_sustained_mean=0.50 vs lose=0.16 (SEPARATES — wins hold more fill).
  L1: 58/308 reach sustained>0.5 but env_wins=0; L2: 142/308 >0.5, env_wins=0.
So a sequence-aware gradient EXISTS but high sustained-fill still != L1 win (overshoot/
instability persists). v47 does NOT transfer as an L1 fitness (L1 scores saturated 0.94,
corr w/ progress 0.04; L2 saturated 0.00). The L0 win-probe learned L0's win-REGION as
a POOLED scalar — it threw away SHAPE, so it cannot generalise.
USER steer #3-4 (2026-07-05): "the oracle and portband systems are there to help inform
an imagination based on SHAPE... and ghost." The generalising imagination must be
SHAPE-based and Contra-informed: oracle = GOAL shape (the U-targets/what filled looks
like), portband = FLOW shape (how streams route given blocks — build_portband_hypotheses),
ghost = END-STATE RELATION shape (what a satisfied "done" board's relations look like).
Shape-relationships are LEVEL-INVARIANT (a stream filling a U is the same relation on
L0/L1), unlike a pooled scalar or an L0-specific classifier. Current 6 contra_planes
(bg/fg, component area, centroid-dist, edges, colour-freq) are GENERIC shape stats — the
richer oracle/portband/ghost reasoning (codexlab/contra_head) is what to wire in. NEXT:
a shape-based, portband-flow-informed pour imagination (the real "CNN maximises Contra
concepts"). [[megarexus-contra-architecture]] [[megarexus-worldmodel-roadmap]]

## FINDING 24 (~14:05) — L0/L1 geometry is INVERTED; the shape RELATION (stream->opening) is the level-invariant is the win predictor
Computed source position (color 4) + per-U-target opening direction (the bbox border
side with the fewest color-11 cells = the gap) directly from the board:
  L0: source TOP (y=2,x=37); 2 U-targets BOTTOM (y52-59) opening UP. Paint falls DOWN
      under gravity straight into up-facing openings -> random arrangements win ~8%.
  L1: source BOTTOM (y=61,x=41); 3 U-targets TOP (y4-11) opening DOWN. Geometry FLIPPED
      -> paint cannot just fall in; must route to top targets whose openings face down
      -> random arrangements ~never align -> 0 env wins (matches FINDING 20).
This is WHY the pooled v47 cannot transfer: L0's raw win-region ("source-top,up-openings")
and L1's ("source-bottom,down-openings") are OPPOSITE in pixel space, but IDENTICAL under
the shape RELATION "route a paint stream into each target's OPENING (calibrated, no
overshoot)". That relation is the level-invariant win predictor the user's oracle/portband/
ghost design targets. Opening-direction + source-position are clean honest shape primitives
(computable per frame). Concrete substrate for the shape-based imagination: input planes
for source, per-target opening cells, and source->opening alignment; a SPATIAL (un-pooled)
predictor over them. Decisive test in prep: does a stream->opening alignment score predict
env-win on L0 AND give a non-degenerate gradient on L1 (where v47 saturated, corr 0.04)?
Contra pipeline (real portband alignment) being mapped; eval set (arr,env_win,sustained_fill)
for L0+L1 harvesting. [[megarexus-contra-architecture]] [[megarexus-lockkey-goal-seeker]]

## FINDING 25 (~14:12) — portband static alignment is a GOAL prior, NOT an arrangement-win predictor; the block-ROUTING (flow) is what must be imagined
Wired the REAL Contra pipeline (build_contra_perception_pipeline + oracle actor/target
builders; run via shape_align_probe.py) to get per actor x target statecons with portband
alignment + features (target_open_axis_match, ray_intersects_target, axis_alignment,
_alignment_static_score). Correlated vs env_win (L0) / sustained_fill (L0+L1), 400/level:
  L0: open_axis/ray_hit/axis_al are CONSTANT across arrangements (std=0.00) -> AUC 0.500.
      static_max AUC=0.556 (~chance); static_mean AUC=0.111 (BELOW chance).
  L1: all near-constant; corr vs sustained_fill <= 0.09.
=> portband's static alignment captures the LEVEL's fixed source->target geometry (and
DOES differ L0 up-openings vs L1 down-openings — a useful GOAL/level signal) but is BLIND
to the movable-block placement that actually routes/splits/deflects the stream. Win/lose
arrangements differ ONLY in block routing, so portband alone can't discriminate them.
REFINED DESIGN: oracle/portband = the GOAL informer (where paint must go, level-invariant
relation "into each opening"); the IMAGINATION must model whether the block SHAPES route
the flow to that goal (calibrated, no overshoot) — the CA/sequence part. Portband's value
is as an INPUT GOAL prior (opening/flow planes), not a standalone predictor. DECISIVE TEST
NEXT: does an UN-POOLED spatial win-probe (block-routing is a local, translation-equivariant
op -> conv-friendly, level-invariant) transfer to L1 where pooled v47 saturated (corr 0.04)?
If yes, shape-preserving (spatial) imagination is the unlock; if no, flow-outcome needs
explicit goal-conditioned sequence modelling. [[megarexus-contra-architecture]]

## FINDING 26 (~14:13) — USER "floor is lava": NEGATIVE feedback (wall/lava contact) is the missing signal — PERFECT L0 win discriminator
User steer #5: "the contra system doesn't have any negative feedback yet... a 'floor is
lava' would help." Contra (oracle/portband/ghost) reasons ONLY positively (align paint TO
targets). The win also needs the NEGATIVE: paint must AVOID walls/lava. Measured on 568 L0
pours the "lava contact" = growth of wall color 14 above the arrangement's baseline
(actual wall HITS during the pour):
  AUC(sustained_fill / POSITIVE only) = 0.843
  AUC(portband alignment) = 0.556 (FINDING 25)
  AUC(-wall_hit_growth / NEGATIVE) = 1.000  <-- PERFECT
  wins: wall_hit mean=2 (barely touch walls); loses: wall_hit mean=243 (slam into walls).
The win = fill targets (positive) AND avoid walls (negative). The negative term perfectly
separates fill-and-win from fill-and-overshoot (cand5/cand6). RESOLVES the FINDING-22
color-14 confusion: the ABSOLUTE 14 count is noisy (structural border + win-flash), but the
DELTA (14 growth above the pre-pour baseline) is the clean lava/fail signal. This is a
DENSE, SPATIAL, SHAPE gradient (243 vs 2, continuous) — not just a binary. Implications:
 (1) imagination win-check = imagine pour, require fill=all AND lava_contact~0.
 (2) a "floor is lava" NEGATIVE plane (walls + floor + overshoot regions to avoid) is the
     missing Contra informer; combined with oracle/portband POSITIVE goal = full objective.
 (3) a CEM/search fitness = maximise fill - lambda*lava_contact — climbable even where the
     positive-only signal is flat. NEXT: test the L1 gradient of the negative signal.
[[megarexus-contra-architecture]] [[megarexus-level3-mechanic]]

## FINDING 27 (~14:14) — the NEGATIVE signal gives a CLIMBABLE L1 gradient (the unlock)
L1 (616 pours, 0 wins): wall_hit mean=243 (most slam walls) but ranges [-22, 254].
Fitness (sustained_fill - 0.004*wall_hit) surfaces a clean frontier: the top-10 arrangements
fill 2/3 targets (0.67) with NEAR-ZERO lava (-22..-6). corr(fill,-lava)=+0.24 (vs v47's flat
0.04). No random arrangement fills all 3 clean YET (0 near-wins at fill>0.9 & lava<30), but
L1 is now a SEARCH with a real gradient, not a blind lottery. The "floor is lava" negative
feedback converts L1 from "0 wins / no signal" to "hill-climb toward fill-all-3-without-lava."
BUILD IN FLIGHT: CEM/hill-climb on fill - lambda*lava via real env pours to find the FIRST
honest L1 win -> those wins bootstrap L1 positives to train the shape-based imagination
(escaping the no-L1-positives trap). Legal: env-feedback search, discovered not injected.

## FINDING 28 (~14:17) *** MILESTONE: FIRST HONEST L1 WIN via the floor-is-lava search ***
The lava hill-climb (lava_search.py, fitness = sustained_fill - 0.004*lava, real env pours)
DISCOVERED an L1-winning arrangement in ~981 evals where random search found 0 in thousands.
Independently VERIFIED on fresh envs (seeds 1000/2000/7777): levels_completed 0->1 WIN=True,
peak_n13 = 240/240 (ALL 3 targets fully filled), lava=1 (clean, no wall-slam), deterministic.
Winning genome (block click+move sequence): [(1,1,2),(6,4,2),(4,1,3),(1,1,3),(3,4,3),(5,1,1),(7,2,3)].
HONEST SCOPE: this is an ENV-FEEDBACK SEARCH result — the arrangement was DISCOVERED by
climbing the lava objective over ~981 real pours, NOT a CNN solving L1 within the 4-pour
budget, and NOT a replayed/human route (no imitation lineage; the agent's own search on
fill+lava from the env). What it PROVES: (a) L1 IS solvable by arrangement; (b) the
positive(fill)+negative(lava) objective is correct and climbable on L1; (c) it yields the
first L1 POSITIVES, escaping the no-L1-positives trap that blocked training the imagination.
NEXT (the actual goal): harvest many L1 wins via lava-search -> train the shape-based
imagination WITH the negative/lava signal -> a CNN that ranks/finds L1 wins within budget and
generalises. The user's 5 steers built this: imagine (not static) + sequence (not frame) +
shape (oracle/portband goal) + NEGATIVE feedback (floor-is-lava) = the objective that cracked L1.

## FINDING 29 (~14:24) — USER: "hacked" variant levels (move object blocks) to FORCE generalization — FEASIBLE + verified
User steer #6: "cant we make a hacked version of the level where we move the obj blocks
around? that would force differentiation and block skill." Root problem it solves: SP80
boards are seed-INVARIANT (fixed per level), so training on a level = MEMORISING one board
(v47 did this on L0), not learning transferable SKILL. Moving objects -> varied layouts ->
the CNN must learn the level-invariant routing skill ("route stream into ANY target's
opening, avoid lava").
Levels are EXPLICIT sprite layouts (sp80.py `levels`): source=sowlljgtjvn, target=repwkzbkhxl,
wall=waoewejnqzc, block=plzwjbfyfli-N, redirect=liolfvkveqg; placed via .set_position(x,y) on a
16x16 grid. (Also: L0 vs L1 inversion = L1 has data dojfslwbg=180 -> the layout is ROTATED
180deg; a real mechanic, not arbitrary.) VERIFIED capability: game._game.current_level sprites
are live; moving a target sprite (set_position) + on_set_level re-renders a CHANGED board
(160 cells shifted) that POURS with real CA physics. So I can synthesise randomised variant
boards and env-grade their pours.
HONESTY: this is procedural CURRICULUM AUGMENTATION (vary the PROBLEM) to force generalisation
— NOT "on_set_level farming" of a fixed level's answer (the banned thing, see
[[megarexus-fromzero-training-ruling]]); no answer injection, every variant pour env-graded, the
real unmodified levels stay held-out for evaluation. REFRAMED PLAN: generate randomised valid
SP80 variants -> harvest (arrangement, fill, lava, win) across MANY layouts -> train the
shape-based, lava-aware imagination on the level-invariant (fill,lava) target -> a CNN that
generalises the objective to unseen layouts (L2+ and beyond), then rank+pour within budget.
[[megarexus-cnn-deliberation-gap]] [[megarexus-contra-architecture]]

## FINDING 30 (~14:31) — variant harvest built; the lava signal GENERALISES across layouts (AUC 0.92)
Built sp80_variants.make_env_variant (clone source/target/block/wall templates -> randomised
valid layout -> real-physics pour) and harvest_variants.py. Harvested 6443 pours over 600
RANDOM variant boards (n_targets in {80,160,240}, varied source/target/block positions),
172 wins (2.7%). KEY: AUC(-lava vs env_win) ACROSS VARIANTS = 0.916 (win lava_mean=2,
lose=224) — the floor-is-lava NEGATIVE signal is LEVEL-INVARIANT, not board-specific. This is
the generalisation training set the user's "move the blocks" idea calls for: forces the CNN
to learn transferable routing skill (avoid lava + fill) over memorising one layout.
Data ready: pour_variants_20260705.pt (600 variants) + pour_filllava_l012_20260705.pt (fixed
L0/L1/L2, 9000 pours w/ fill+lava). NEXT: train a shape-based, lava-aware imagination to
predict (fill, lava) from an arrangement on the VARIED data; gate = generalisation to held-out
variants + the real unmodified levels (rank+pour within budget). Tools: sp80_variants.py,
harvest_variants.py, lava_search.py, harvest_eval_set.py. [[megarexus-cnn-deliberation-gap]]

## FINDING 31 (~14:59) *** MILESTONE: GENERALIZING imagination — trained on random variants, transfers to REAL unseen levels ***
v48 = shape-based (conv-then-flatten, keeps layout), LAVA-aware (predicts fill + lava,
the positive+negative objective) outcome imagination, trained on 600 RANDOMISED variant
boards (6443 pours), encoder frozen. Trained ONLY on variants; the real L0/L1/L2 levels
were NEVER in training. Results:
  HELD-OUT VARIANT (unseen layouts) win_AUC = 0.917
  REAL-L0 win_AUC = 0.978   REAL-L1 win_AUC = 0.95 (peak 0.953)
  REAL-L2: 0 wins to AUC, lava_hat mean 0.97 (correctly flags mostly-lava arrangements)
Contrast the POOLED v47 (trained on L0): L1 AUC 0.58 (memorised, no transfer). v48 =0.95.
=> a CNN trained purely on randomised layouts imagines the pour win on REAL UNSEEN levels
at 0.95-0.98 AUC — GENUINE transferable routing skill, not memorisation. The user's whole
design converges here: IMAGINE (not static) + SEQUENCE (not frame) + SHAPE (conv, not
pooled) + NEGATIVE feedback (floor-is-lava) + VARIED data (move-the-blocks). HONEST: real
levels held out, env-graded outcomes, no answer/route injection.
UNLOCK: v48 is now an AMORTISED fitness — predict (fill - lambda*lava) from an arrangement
in ONE pass, no env pour. This enables a CEM proposer to search L1/L2+ efficiently (score
by v48, confirm top-K with real pours) -> in-budget CNN solve across levels. Model:
predict_outcome head. Tools: train_outcome.py, sp80_variants.py, harvest_variants.py.
[[megarexus-cnn-deliberation-gap]] [[megarexus-contra-architecture]]

## FINDING 32 (~15:49) — MASS regime v49: rotation was a HIDDEN param -> contradictory signal -> poisoned real-L1 transfer
Scaled to 3000 variants (32042 pours, 2-4 targets, 1-5 blocks) + added rotation
(dojfslwbg via set_data). v49 result: HELDOUT-VARIANT AUC 0.966 (>v48's 0.917),
REAL-L0 0.99, but REAL-L1 DEGRADES over training 0.40(ep5)->0.08(ep60) — WORSE than
random and far below v48's 0.95. Diagnosis: set_data("dojfslwbg", rot) does NOT rotate
the rendered LAYOUT (source stays row 0, targets row 13 for ALL of rot=0/90/180/270);
it only changes the HIDDEN flow-physics rotation applied at pour time. So identical-
looking arrangements got DIFFERENT (fill,lava) outcomes depending on an invisible param
=> contradictory training signal the model can't predict from the frame. And real L1's
rotation IS baked into its visible layout (source at BOTTOM) — which my top-source
rotated variants never reproduce. Net: rotation poisoned exactly the rotated case
(real L1). LESSON: every outcome-determining factor MUST be visible in the input;
a hidden rotation param needs either a true layout rotation (rotate all sprite
positions) OR an explicit rotation INPUT plane. Also: HELDOUT-VARIANT AUC is NOT a
reliable proxy for real-level transfer under distribution shift (0.966 held-out but
0.08 real-L1). FIX: remove the broken rotation, re-harvest clean (keep wider VISIBLE
diversity: 2-4 targets), retrain -> restore L1 transfer (v48 proved the no-rotation
recipe gives L1 0.95). Follow-up: add TRUE geometric diversity (source positions +
opening directions) so rotated real levels are in-distribution, represented VISIBLY.

## FINDING 33 (~16:13) — naive MASS scale-up HURT real-level transfer; v48 (600 variants) remains best
Clean v50 (3000 variants, 2-4 tgt, 1-5 blk, NO rotation): HELDOUT-VARIANT 0.938,
REAL-L0 1.00, but REAL-L1 peaks epoch5=0.77 then CRASHES to 0.12-0.18. Removing rotation
helped vs v49 (0.40->0.77) but still far below v48's 0.95. Pattern: BOTH 3000-variant runs
peak L1 early then degrade; v48 (600 variants, 2-3 tgt, 1-3 blk) HELD L1 ~0.90-0.95 across
epochs. So MORE/WIDER variants != better transfer — the naive scale-up overfits the variant
distribution and loses the real-L1 geometry. Sobering: transfer is partly DISTRIBUTION-
MATCHING (v48's 2-3 targets is closer to real L1's 3 than v50's 2-4 diluting with 4-target
boards), not pure general skill. Also confirms HELDOUT-VARIANT AUC (0.94) is a poor proxy
for real transfer (L1 0.12). DECISION: v48 stays the best generalising imagination
(L0 0.98, L1 0.95). Naive mass-scaling is NOT the win; the real improvement needs TRUE
geometric diversity (source on any edge + matched flow so rotated real levels are VISIBLY
in-distribution) and/or early-stopping on real-transfer — a careful build, not brute count.
PIVOT: test the PAYOFF with v48 — v48_solve (v48 as amortised CEM fitness -> pour top-K
real -> in-budget solve of L0/L1/L2).

## FINDING 34 (~16:34) — in-budget solve: two gaps isolated (ranker sharpness + proposer), imagination is sound
Tested the payoff (v48 generalising imagination -> in-budget solve of real L0/L1/L2):
 - v48_solve (CEM scored by v48): collapsed — top-4 elites near-identical, solved 0/3.
   Premature convergence; genome space didn't reach real winners.
 - win_probe_gate --mode outcome (rank _candidates by v48 fill-lava, pour top-4):
   L0 ranking AUC 0.786, winners at ranks 7-24 (never top-4) -> solved@4 0/4. L1/L2:
   real_wins=0 (proposer finds no winners) -> ranking can't solve.
CONTRAST: v47 (trained ON the L0 _candidates distribution) ranked L0 winners at ranks
0-7 -> solved@1 8/8. So v48 (variant-trained) GENERALISES broadly (L1 AUC 0.95 on the
pour_filllava set) but is LESS SHARP on a specific level's candidate set — a
generalisation-vs-sharpness tradeoff. TWO GAPS to a cross-level in-budget solve:
 (1) ranker sharpness (top-4 needs the winner ranked very high, not just above median);
 (2) a PROPOSER that finds winners where random gives 0 (L1/L2).
The imagination itself is sound (ranks real winners well). Testing the cleanest in-budget
path now: imag_search_solve — hill-climb (multi-restart, no collapse) on v48's PREDICTED
fill-lava (free, no pours), then pour top-4 real. If v48's imagination guides the search to
a real win, that closes the proposer gap in-budget. [[megarexus-cnn-deliberation-gap]]

## FINDING 35 (~16:41) *** HONESTY CORRECTION: v48 does NOT truly generalize to L1 — AUC 0.95 was misleading ***
Direct test on the VERIFIED L1 winner (lava_search genome, env-confirmed 240/240 filled,
lava=1): v48 predicts fill_hat=0.003, lava_hat=0.974, score=-0.971 — a TOTAL LOSS, the
OPPOSITE of reality. It ranks the true winner BELOW 55% of random L1 losers. So the training
log's "REAL-L1 win_AUC=0.95" was MISLEADING: measured on a narrow harvested L1 set (base_seed
9000, few wins) and driven by the level-invariant LAVA correlation, NOT real understanding.
ROOT CAUSE: my variant generator ALWAYS places the source at the TOP; real L1 is rotated
(source at BOTTOM, dojfslwbg=180, paint flows up). So real-L1 geometry is entirely OUT of
v48's training distribution. v48 generalizes WITHIN source-top geometry (L0 0.98 in-dist,
held-out source-top variants 0.92) but FAILS on genuinely different geometry (L1 source-bottom).
CORRECTED CLAIMS: L0 solved in-budget = v47 (on-distribution ranker), REAL and honest. First
L1 win = lava_search (env-feedback search), REAL and honest. v48 = generalises to source-top
layouts only; the "cross-level generalisation" was OVER-CLAIMED on AUC — the argmax/precision-
at-top is wrong on L1. Lesson (again, cf F32): AUC on a harvested set is NOT understanding;
always test the KNOWN winner directly. FIX for TRUE cross-level generalisation: EDGE-DIVERSE
variants — source on all 4 edges with the MATCHING flow rotation (bottom+180, top+0, etc.) so
source-edge DETERMINES flow (visible), targets on the opposite edge, covering L1/L3/L5 rotated
geometries. Verify the source-bottom+rot180 pour actually fills top targets first. This is the
real mass-regime improvement, not brute count. [[megarexus-cnn-deliberation-gap]] [[megarexus-lab-honesty-rules]]

## FINDING 36 (~17:15) — the geometry fix WORKS (partially): multi-base coverage lifts L1-winner fill 0.003 -> 0.689
v51 trained on the geometry-complete multi-base set (bases 0,1,2,3: src-TOP 15191 pours +
src-BOTTOM 15223 pours, 459 wins). Direct test on the VERIFIED L1 winner:
  v51: fill_hat=0.689 (v48 was 0.003!), lava_hat=0.766, score=-0.077 (v48 was -0.971).
  ranks above 60% of random L1 (v48: 45%).
So covering source-bottom/rotated geometry DIRECTLY fixed the fill blindness — v51 now
recognises the L1 winner fills its targets. FINDING 35's diagnosis was correct and the
multi-base fix works. REMAINING GAPS: (1) lava precision on rotated arrangements still off
(lava_hat 0.766 vs actual ~0) — drags score negative; (2) aggregate AUC LOWER than v48
(held-out-variant 0.82 vs 0.92, REAL-L0 0.78 vs 0.98) because the mixed-geometry task is
harder for the FROZEN v31 encoder (trained mostly on source-top real data). DIRECTION: the
frozen encoder likely under-represents rotated geometry -> UNFREEZE/fine-tune the encoder on
the multi-base data (or add source-bottom Contra-guided pretraining) so the latent captures
both geometries; then the outcome head can be precise on both. Honest net: the imagination
generalises to a NEW geometry once that geometry is in training (fill fixed) — real, incremental
progress toward cross-level; precision (esp. lava on rotated) is the next lever. [[megarexus-cnn-deliberation-gap]]

## FINDING 37 (~17:40) *** L1 GENUINELY FIXED: multi-base + UNFROZEN encoder -> v52 correctly predicts the L1 winner ***
v52 = train_outcome on the multi-base set with --unfreeze-encoder (trunk fine-tuned, enc-lr
3e-5). Direct test on the VERIFIED L1 winner:
  v48 (frozen, src-top only):  fill 0.003, score -0.97, ranks above 45% (WRONG)
  v51 (frozen, multi-base):    fill 0.689, lava 0.766, score -0.08, above 60% (partial)
  v52 (UNFROZEN, multi-base):  fill 0.558, lava 0.045, score +0.51, above 97% (CORRECT)
v52 predicts the real L1 winner as a WIN (positive, LOW lava 0.045 — the precision v51
lacked) and ranks it above 97% of random L1 losers. REAL-L1 AUC 0.99 and STABLE across
epochs (no crash, unlike every frozen run). So the encoder UNFREEZE was the key: adapting the
latent to rotated geometry fixed lava precision. The full honest fix for cross-level: (a)
cover the geometry (multi-base variants) + (b) let the encoder adapt to it (unfreeze). This
REVERSES the F35 over-claim into a genuine result: v52 understands BOTH source-top (L0) and
source-bottom/rotated (L1) geometry. NEXT: does v52 as fitness now SOLVE L1 in-budget
(imag_search_solve)? [[megarexus-cnn-deliberation-gap]]

## FINDING 38 (~17:47) — v52 solve: ranking good, ARGMAX imprecise (false positives) -> hard-negative mining
v52 imagination-guided search (60x100): L1 top-4 imagined-scores [0.886,0.869,0.849,0.823]
but NONE win for real. So v52 has FALSE POSITIVES — arrangements it scores 0.88 (confident
win) that actually lose, scoring ABOVE the true L1 winner (0.51). The search maximises v52's
score and lands on these false peaks, missing the true winner. So even a good RANKER (true
winner ranks above 97%) has an imprecise ARGMAX (false positives outscore the winner) ->
search-to-solve fails. This is the core remaining gap: AUC/ranking != precision-at-argmax.
PRINCIPLED FIX (flywheel): HARD-NEGATIVE MINING — the search SURFACES exactly v52's false
positives (high-imagined, real-loss). Real-pour the search's top-K, label the losers, ADD
them to training as negatives, retrain (unfrozen). The imagination learns to REJECT its own
false peaks -> sharper argmax -> search finds the true winner. Iterate. This directly targets
the failure the search exposes (adversarial/DAgger-style for the imagination). Building the
hard-negative miner next. [[megarexus-cnn-deliberation-gap]] [[megarexus-worldmodel-roadmap]]

## FINDING 39 (~18:32) — naive hard-negative mining (3x) BACKFIRED; v52 remains best
v53 = multibase + 3x hardneg (217 false-positives + 46 wins), unfrozen. Direct L1-winner
test: fill 0.559, lava 0.612, score -0.053 (v52 was +0.51), ranks above 87% (v52 97%). WORSE.
The 3x-upweighted false-positives biased the model PESSIMISTIC (lava_hat up across the board),
hurting the true winner. ROOT: false-positives are VISUALLY SIMILAR to true winners (both look
like good arrangements), so teaching the model to reject them suppresses the winner's features
too — the precision-at-argmax problem is hard precisely because near-winners and winners look
alike (needs precise pour SIMULATION to separate, the night's recurring wall). v52 stays the
best model (correct L1 winner +0.51/above-97%, REAL-L1 AUC 0.99 stable, REAL-L0 good). Trying
v54 = 1x hardneg (less pessimism bias) as the last refinement; if it doesn't beat v52, v52 is
the deliverable and the in-budget SOLVE remains the honest open gap (argmax precision).
[[megarexus-cnn-deliberation-gap]]

## FINDING 40 (~18:55) — hard-negative mining CONFIRMED unhelpful (1x also worse); v52 is the deliverable
v54 = multibase + 1x hardneg: L1-winner score -0.27, above 67% (v52 +0.51/97%, v53 -0.05/87%).
Both 1x and 3x hardneg HURT vs v52. Confirmed: teaching the imagination to reject its false
positives suppresses the true winner too (they're visually alike). Hard-negative mining is NOT
the fix. FINAL STANDINGS (v52 best): the pour imagination generalises across geometries when
(a) the geometry is in training (multi-base variants) and (b) the encoder adapts to it (unfreeze)
— v52 correctly scores the verified L1 winner (+0.51, above 97%), REAL-L0 0.98, REAL-L1 0.99
stable. The IN-BUDGET SOLVE remains open: v52 is a strong RANKER but its ARGMAX has false
positives (search finds 0.88-score non-winners above the 0.51 true winner). Closing it needs
either (i) precise pour SIMULATION to separate near-winners from winners (the night's recurring
wall — pixel-CA drifts, latent is AUC-good only), or (ii) a DIVERSITY-promoting search that
pours K diverse high-scorers (since v52 ranks the true winner top-3%, a diverse top-K likely
contains it) — a cheap SEARCH-side idea worth trying, not an imagination change. NIGHT COMPLETE
for the heavy-exploration thread; v52 is the honest deliverable.

## FINDING 41 (~19:01) — in-budget SOLVE exhaustively explored; hits a real imagination-precision limit
Diversity-search (v52, best-per-restart): L1 top-4 diverse [0.793,0.67,0.67,0.584] — still NOT
solved. Full list of in-budget approaches tried, ALL fail on L1/L2:
  - rank random candidates (v48/v52): winners rank high but proposer finds ~0 L1 winners.
  - CEM (v48_solve): collapses to false-positives.
  - imagination-guided hill-climb (imag_search_solve): overshoots to false-positives.
  - DIVERSE search (best-per-restart): still false-positives.
  - hard-negative mining (v53/v54): HURTS (false-pos ~ winners visually).
ROOT LIMIT: v52's high-score region (0.6-0.8) is FALSE POSITIVES that outscore the true winner
(0.51). No search maximising v52's score reaches a real winner. The imagination cannot separate
visual near-winners from true winners without PRECISE POUR SIMULATION (the night's recurring
wall: pixel-CA drifts, latent is AUC-good/argmax-imprecise). 
HONEST FINAL STATE: (1) v52 = a genuine cross-level pour imagination + strong RANKER (deliverable,
fulfils "CNN maximises Contra concepts"). (2) The agent SOLVES levels via env-feedback play:
L0 by ranking (v47), L1 by lava_search. (3) The specific single-episode 4-pour CNN-alone solve
of L1/L2 is NOT achieved — blocked by argmax precision, which needs a genuinely new capability
(precise settled-state simulation or a per-target spatial win-check), a major direction beyond
incremental training tweaks. Heavy-exploration thread COMPLETE. Runs v44-54 + searches.
