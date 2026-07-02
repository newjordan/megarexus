# Research log

## 2026-07-01 — Lab initialized

Created `synthesis_lab/` as the explicit research workspace for synthesis of:

- LeWM/Rex action-side world model / runner stack,
- Contra thinking/control packet engine,
- Stochastic Goose stochastic runtime,
- ARC scorecard/environment evidence layer.

Next research step: define a minimal adapter contract between Contra packets, Goose action proposals, and LeWM/Rex action owner/orchestrator.

## 2026-07-01 — MegarexusCNN SCOUT_EXPERIMENT (v0 + v1 flow regime)

Built and trained the first synthesis CNN (`experiments/megarexus_cnn/`):
conv world model (LeWM/Rex role, FiLM action conditioning), Contra
perception planes as input channels, Goose-exact 4101-logit head + change
objective + stochastic collection. Full results and honesty provisions:
`notes/megarexus_cnn_scout.md`, `manifests/megarexus_cnn_manifest.md`.

Key regime finding for catch-and-redirect: ACTION5 pours return ~20-22
animation frames per step (previous regimes discarded all but the last);
paint velocity needs a 2-frame input stack. With both fixed, the ghost
overlay pass confirms the paint-flow mechanic on fresh rollouts (h=1
changed-cell acc 1.00, IoU 0.98; beats persistence through h=5). Human
pour segments remain under-modeled (concurrent bar moves under-covered by
random data). Lock/key asymmetry seeker fired 0/80 on sp80+snake held-out
frames — `component_asymmetry` yields no confident facings there; needs
per-game recalibration before it can serve as the goal provider.

## 2026-07-01 — Catch-play regime (v2/v3): the CNN learns to call catches

Portband framing operationalized: gloves = moveable objects found by
controlled interaction probing, edges from Portband's `_object_ports`,
catch labels hindsight-from-pixels. v2's global-pool catch head failed
(collapsed to base rate); v3's glove-masked spatial pooling reached
**catch prediction 1.000 vs 0.528 base on 106 unseen-seed pours** while
the world model improved to 0.987 flow changed-cell acc (persistence-
beating overall cell acc 0.991 vs 0.971) and inverse dynamics hit 0.990.
Multi-object roster (SP80 L1 has 2 gloves) and edge-adjoin trials
(70 trials, 0.26 base) collected; adjoin head + behavioral catch probe
are the queued next steps. Details: `notes/megarexus_cnn_scout.md`.

## 2026-07-01 — Behavioral catch probe: the system plays catch

Ghost-imagined pour column (world model, no env access) + Portband glove
moved under it → **77/80 strict catches vs 30/80 random** on 20 paired
unseen seeds (strict = paint on the glove's top port edge within 8
animation frames). Negative result kept: hill-climbing the saturated
catch-head logits underperforms random; imagination-based placement is
the right controller. Probe hygiene fixes: HUD strips no longer accepted
as gloves (band suppression + direction-consistent displacement);
curriculum re-collection queued for v4.

## 2026-07-01 — Level probe: where the system stands against SP80

Environment-field evidence (local offline, SCOUT): model-guided catch play
completes **level 1 in 10/20 episodes vs 5/20 random** on paired unseen
seeds; the human corpus reaches level 4+ every episode with one full WIN.
Correction logged: flow/catch rollouts did complete levels (earlier note
said 0). Measured blockers: (1) level-1 move budget — 4/5 catches can
still GAME_OVER without completing; (2) level-2 glove-probe miss — the
movable color-9 bar exists but 6-candidate shuffled probing misses it
~25% of the time, matching the 3/10 stalls; (3) old curriculum's level
completions rode the strip-glove bug — v4 re-collection queued.
Details: `notes/megarexus_cnn_scout.md`; report
`runs/catch_v3/level_probe.json`.

## 2026-07-01 — Shape/map audit: the CNN trains on ~none of it; backlog built

Full inventory of contra_head + lewm shape/map recognition (floor/wall
scoring incl. the "centroid solver" terms, checkerboard, contact graph,
symmetry passes, lock-key sensor, geoband, portband ledger/matrix, LeWM
wall-constancy masks, centroid player tracker): training currently
consumes two low-level component functions + `_object_ports` labels;
zero role semantics. Gas budget probed separately: drain physics and
death frames are learned perfectly by the world model but unused by the
controller. Audit surfaced `sp80_spill_model.py` (hand-coded SP80 mechanics + win
condition). USER RULING, same day: that file is an answer key and is
banned from training in every form (labels, informer, agreement metric,
verifier) — "we are training systems, not the answers." Tier 1 of the
backlog rewritten to learn outcome semantics purely from experience:
behavioral role labels from pixels, level-up/death heads from
environment fields. A 5-tier training backlog (one element per run,
ablated like the catch head) is in `notes/megarexus_cnn_scout.md`.

## 2026-07-01 — outcome_v4: first backlog run (level-up, control, settle)

Three evidence-labeled heads trained on the v3 recipe; all v3 metrics
held. **Controllability: 1.000 vs 0.528 base on 1,848 unseen-seed
samples** — behaviorally, it ranks level 2's movable bar first (+4.31 vs
≤−14) with zero probe actions, replacing the probe-luck stalls.
**Level-up: 3/3 true positives, 0 false positives** on val pours — the
win condition as a learned signal (thin positives; grows with
collection). **Settle: label sent back** — SP80 liquid is transient, so
the persistent-diff mask (median 2 cells) mostly captures gauge drain;
redesign queued (multi-pour integration / fill delta). Details:
`notes/megarexus_cnn_scout.md` → outcome_v4.

## 2026-07-01 — Level probe v2: level 1 solved 20/20 with the learned finder

Wiring the controllability head into glove finding (rank + click, 1
action, vs up to 18 for interaction probing) took the paired level probe
from 10/20 to **20/20 level-1 completions** on the same unseen seeds
(random baseline reproduced exactly at 5/20). Zero level-2 stalls — every
episode now finds the level-2 bar instantly and dies on level 2 by gas.
The frontier is now level-2 gameplay: budget-aware placement + a
redesigned fill signal. Report:
`runs/outcome_v4/level_probe_learned.json`.
(Correction, same day: "dies by gas" was wrong — see the next entry.)

## 2026-07-01 — Level 2 decoded and completed (arrange-then-pour)

User guidance ("you can't miss" is critical; level 2 = blocking overflow
with side-walls, orthogonal catch toward the flow) drove level probe v3
(miss-gated pours + imagined-gauge budget check + side-block logging) and
a diagnostic chain that overturned the level-2 story: the gauge is row 63
there and is NOT the killer — level 2 grants ~4 pours and the 5th call is
instant GAME_OVER; the flow is an upward jet that splits on the bar and
overflows both flanks toward three funnel mouths; bar position alone
changes nothing. Mining the human corpus showed the real mechanic: both
8-platforms are ALSO movable, and humans steer bar+platforms to route the
risers. Offline, flow settles within the pour step, so the solve is
arrange-then-pour: the human WIN configuration + one pour →
**levels_completed=2, first level-2 completion in the lab** (level 3:
three spouts/funnels). Recorded as a demonstration-derived scripted
rollout, NOT model capability. Fill/settle label declared dead (level 2
has no persistent fill pixels); outcome semantics = level-up head. Next:
plumbing-play curriculum (randomized multi-object arrangements, pour
last), outcome_v5 with real level-up positives + 4-axis multi-object
control samples, then imagination search over arrangements.
Details: `notes/megarexus_cnn_scout.md`; probe report
`runs/outcome_v4/level_probe_careful.json`.

## 2026-07-02 — outcome_v5/v6/v7: plumbing curriculum, a label-leakage
## discovery, and the planner's first honest failures

Plumbing-play curriculum landed (`orchestration/collect_plumbing_play.py`):
reach the plumbing level with the trained catch policy (60/60), then
randomized 4-axis click+move arrangements, pour LAST; every 5th episode
replays the human demo arrangement (tagged demo_replay, never model
capability). New regime fact found while debugging the replay:
**click-selection RECOLORS movables** (selected=color 9, deselected=8) —
color-keyed tracking follows the wrong object on multi-movable levels;
the collector re-identifies by click-point and position+area.

outcome_v5 (v4 recipe + plumbing shards) surfaced two infrastructure
defects, both caught by the loop's gates, both fixed:

1. **Checkpoint policy**: best.pt selects on changed-cell acc alone and
   froze at epoch 12 while every head kept improving to epoch 40.
   train.py now also saves last.pt; release choice belongs to the
   external no-loss gate (same-valset eval vs the previous release).
2. **Label leakage via prev-frame selection**: `levels_completed` is
   stored post-action, but the loader compared it across rows as if
   frame-level; exactly on level-completing pours the prev frame was
   dropped and the encoder self-stacked [cur, cur] — the label leaked
   into the input structure. The level head "passed" val 23/23 by
   recognizing ONE pixel-identical demo frame in self-stacked form
   (+9.2) while scoring the same pixels honestly stacked at −14.7.
   outcome_v6 over-corrected (masked the completing-pour dynamics pair —
   but its true next frame is the first ANIMATION frame on the old
   layout, a valid target): pour-at-completion imagination went
   unsupervised, extrapolated a 1221-cell scene swap, and the level
   probe collapsed 20/20 → 0/20 on decline:gauge (imagined drain
   102 > 64). outcome_v7 keeps v4/v5 pair semantics and fixes ONLY the
   prev selection on level-up rows.

**outcome_v7 is the release**: same-valset gate holds vs v4 (catch
105/106 with one borderline +0.23-logit miss, bc 0.996 > 0.976, inv
0.996, control 1.0, level 3/3 fp 0), level probe reproduces **20/20 vs
5/20**, and the honest-encoding diagnostic flipped: real demo
arrangement +12.3 (was −14.7), imagined demo plan +9.5, imagined-vs-real
end-frame agreement 0.9993 with exact component positions (the platform
drift in imagination is gone).

Arrange-probe (`orchestration/arrange_probe.py`, imagination search +
own level head as scorer, paired random baseline): v5 0/10 vs 1/10
random (scorer dead by leakage); v7 **0/10 — a new, crisp failure**: the
head scores the PRISTINE unarranged pre-pour frame +9.81 (false
positive), so the null plan wins every episode. Cause is a collector
coverage hole: no episode ever poured without arranging, so that region
has zero negative supervision. Next element: plumbing_play_v2
(pour-first null-arrangement negatives + a 500-episode random pool for
diverse self-collected completion positives), then outcome_v8.

## 2026-07-02 — outcome_v8 + arrange planner: first model-guided
## plumbing completions (10/10 vs 1/10 random)

Data for v8 closed the null-plan hole and the memorization hole:
plumbing_play_v2 (500/60 eps, `--pour-first-prob` null-arrangement
negatives) and plumbing_play_v3 (300/40 eps, `--demo-jitter-prob`
replays with ±1-move noise → 13 completions spanning 12 DISTINCT
arrangements, vs one pixel-identical frame before). Trained as
outcome_v8 (v7 recipe + curated v23 shards). Gates: same-valset vs v7
holds head-by-head (catch 104/106 with one borderline miss, control
1.0, level 3/3 fp 0, inv 0.9968, bc 1.000); level probe reproduces
20/20 vs 5/20. best.pt failed the gate (missed all 3 old level
positives) — **last.pt is the release**, external-gate policy again
doing its job. On the new combined val: level head 109/109 tp, 0 fp
(diverse positives at last), control 0.999 (n=5,682), settle IoU 0.656
— the settle label declared dead on thin v4 data came back to life
with coverage.

The arrange probe then exposed three buried assumptions in the PROBE
itself, each fixed and verified in isolation:

1. **Recorder prev-frame convention at plumbing entry**: the probe fed
   the level head the pre-pour frame as prev; the training convention
   is frames[-2] of the completing pour animation. Null false positive
   +12.8 → honest −21.4 after the fix.
2. **Controllability is selection-conditional**: "would it move NOW"
   is truthfully no for unselected components (platform2 scored −84).
   Fix composes two learned systems — imagine the click with the
   dynamics head, then query the control head on the imagined
   post-click frame (`rank_movables_if_clicked`). Roster becomes exact:
   bar +114, platform2 +76, platform1 +29, cliff to everything else.
3. **The last mover's selection recolor is part of the scored state**:
   identical offsets score +19.95 with bar moved last vs −40.9 with
   bar first (first-two order irrelevant). The search must enumerate
   execution order, not just offsets: `exhaustive_x_plans` covers
   (dx0,dx1,dx2) ∈ {−3..3}³ × choice of last mover (883 plans incl.
   null), all scored purely in imagination by the model's own level
   head.

**Verdict run (paired seeds 20280001–10): model 10/10 plumbing
completions, random baseline 1/10.** Every episode the search picked
the same winner — platform2 +2, platform1 +3, bar +1, bar last —
score +19.95 with runner-up +3.09 (a decisive basin, not a
tie-break), and `imagined_real_agreement=1.0`: the world model's
rollout of the chosen plan matched the executed frames exactly. These
are the first plumbing completions where arrangement choice, scoring,
and execution order all came from learned heads with no hand-coded
goal.

Honest scope: level-2 layout is deterministic across seeds, so this is
one planning problem solved reliably 10×, not 10 distinct puzzles; and
the discovered plan sits in the demo basin because that is where the
level head's positives (jittered demo replays + self-collected random
completions) live. Demonstrations are legal experience — but
generalization to novel layouts (level 3: three spouts, wider
platforms) is untested and is the real next test of the machinery.
Next element per the loop: pure object-control curriculum
(user-endorsed) — broad click/select/move coverage to strengthen the
existing control + dynamics heads, whose selection-conditional
composition the planner now provably depends on.
