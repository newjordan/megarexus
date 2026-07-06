# Course advisory — Stage23/v13 resume (written 2026-07-03, post-wrap review)

This is the pick-up document for the next agent session. It supersedes the
resume section of `WRAP_20260703_STAGE23_V13_READY_NOT_RUN.md` by adding the
whole-program assessment, an authorized relaunch procedure, and a
pre-committed decision tree. Read `HANDOFF.md` for standing rules; this file
tells you what to do next and why.

## 0. Resume execution update (2026-07-03 23:04 CDT)

Stage23/v13 was relaunched twice after clean preflight checks, then stopped
twice before checkpoint because unrelated GB10 jobs appeared after preflight.
This is a runtime/provenance update only, not a training result.

- Preflight at `2026-07-03T23:03:25-05:00`: no active Megarexus `train.py`,
  GB10 had no compute process and 0% GPU util, evidence server
  `http://127.0.0.1:8765/` returned 200, Stage23 `CONDITION.json`, v12 init
  checkpoint, all 11 train shards, and all 6 val shards hash-matched.
- First relaunch wrapper PID `2643869`; actual `train.py` PID `2643873`.
- First relaunch stop at `2026-07-03T23:05:51-05:00`: `nvidia-smi` showed unrelated
  `llama-dspark-cache-probe` PID `2643087` as the only compute process,
  holding GB10 memory/utilization. The Stage23 wrapper process group was
  terminated (`2643869`, `2643873`, `2643874`); no Megarexus `train.py`
  remained after stop.
- Second preflight at `2026-07-03T23:07:08-05:00`: no active Megarexus
  `train.py`, GB10 had no compute process and 0% GPU util, evidence server
  returned 200, Stage23 condition, v12 init checkpoint, all 11 train shards,
  and all 6 val shards hash-matched.
- Second relaunch wrapper PID: `2647364`; actual `train.py` PID: `2647368`.
- Second relaunch stop at `2026-07-03T23:09:21-05:00`: `nvidia-smi` showed
  unrelated DeepSpec Leanstral export PID `2647359` as the only compute
  process, holding GB10 memory/utilization. The Stage23 wrapper process
  group was terminated (`2647364`, `2647368`, `2647369`); no Megarexus
  `train.py` remained after stop.
- Stage23 `CONDITION.json` status is now
  `blocked_resource_conflict_before_checkpoint`.
- Current Stage23 condition sha256 after recording the second stop:
  `45f68d8c2ad979e6ae9a77379b506e4a911220d229bf151a5eabc5690e39a718`.
- Log path:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v13_level3_human_mined_20260704_train.log`.
- Boundary: the log is 60 bytes and only has three `[train] device=cuda`
  startup lines. No v13 output dir, checkpoint, or `history.json` exists.
  Do not claim any v13 model result. Next valid relaunch requires a fresh
  clean GB10 preflight after the Leanstral/DeepSpec jobs are gone.
- Readiness recheck at `2026-07-03T23:17:53-05:00`: GB10 was clear again
  (`nvidia-smi` showed no compute process) and no Megarexus `train.py` was
  running. However, the user clarified the real target: fix the missed
  level-0 goal-lock/target-inventory signal. Do not treat a plain v13
  relaunch as that fix. v13 remains a recorded model-learning experiment;
  the goal-lock fix requires Contra training first: extract/validate the level-0
  completion goal schema, wire it as a goal provider, and only then train a
  goal head under its own condition.
- Strict launch-gate recheck at `2026-07-03T23:18:39-05:00`: no Megarexus
  `train.py`, but `nvidia-smi` again showed a low-memory unrelated
  `llama-cli` compute process on the GB10. Under the recorded Stage23
  preflight rule ("no compute process"), do not relaunch yet.
- Apollo relaunch (user-approved host switch) at `2026-07-03T23:26:18-05:00`:
  created Apollo-specific condition
  `synthesis_lab/experiments/megarexus_cnn/runs/stage23_v13_level3_human_mined_apollo_20260704/CONDITION.json`.
  It records the host/GPU/path change explicitly: Apollo RTX 4080, workspace
  `/home/frosty/megarexus_train_20260703`, CUDA-capable Python
  `/home/frosty/Loom/.venv/bin/python` (torch 2.5.1+cu121), same source
  condition/data/checkpoint/settings otherwise. Condition sha256 after final
  launch provenance:
  `691c4148849781f13d7b1b5d7823b8569da2fddb20bdda1d41144aab8a4b7ab3`.
- Apollo launch details: first Apollo attempt with base
  `/home/frosty/miniconda3/bin/python3` was stopped immediately because it
  logged `[train] device=cpu` (CPU-only torch, no checkpoint/history/output
  dir). Second attempt with `/home/frosty/ARC3-solution/.venv/bin/python`
  logged `[train] device=cuda` but was stopped before output because torch
  2.8's `torch.load` default rejected the trusted numpy-containing `.pt`
  shards. Final active launch uses `/home/frosty/Loom/.venv/bin/python`;
  wrapper PID `622267`, launcher PID `622272`, train PID `622279`, log
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v13_level3_human_mined_apollo_torch25_20260704_train.log`.
  Boundary at last check: startup/data-load only, no checkpoint/history yet.
- Apollo status at `2026-07-03T23:53:38-05:00`: v13 is still active on
  Apollo train PID `622279`, RTX 4080 allocation about 2964 MiB, and epochs
  1-5 have checkpointed. Recent fixed-val lines still show level false
  positives (`fp=8-9`), so do not claim v13 success or key/lock/target
  mastery before the full gate chain.
- Stage24 Contra training static review completed while Apollo trained:
  condition
  `synthesis_lab/experiments/megarexus_cnn/runs/stage24_goal_schema_extraction_20260704/CONDITION.json`,
  sha256
  `ca60e73ad8cb873792d2bcee1418104672ddde52ba565c23f38409e9da97dcdd`.
  Review page verified at `http://127.0.0.1:8766/index.html` (`200 OK`);
  raw packet
  `synthesis_lab/experiments/megarexus_cnn/runs/stage24_goal_schema_extraction_20260704/goal_schema_hypotheses.json`,
  sha256
  `e98bc2fee347098de7e077e2611a37e4e2754da409b48dba9ea9e3222c98cb63`.
  The extractor isolated the reset-free level-0 scoring frames in both
  Stage20 and Stage14 (`call0005`, 19 scoring frames, reset at frame 19).
  Both windows agree on two late-filled target candidates
  (`c0006_v11_00`, `c0007_v11_01`, both fill to raw value 13), one
  early-changing key candidate opposite the target centroid
  (`c0005_v09_00`), and one expanding paint-source candidate
  (`c0004_v06_00`). Boundary: this is static hindsight review, not control
  truth, not a wired goal provider, not solve evidence, and not Contra
  training mastery.
  Next valid Contra training step is to turn these completion-window hypotheses into
  a recorded hindsight goal-label dataset across all completed levels, then
  train/evaluate a goal head under its own condition before runtime use.

## 0b. Contra training update (2026-07-04 00:31 CDT)

Stage23/v13 on Apollo is complete and **not promoted**. Local sync:
`synthesis_lab/experiments/megarexus_cnn/runs/outcome_v13_level3_human_mined_apollo_torch25_20260704`;
Apollo condition
`synthesis_lab/experiments/megarexus_cnn/runs/stage23_v13_level3_human_mined_apollo_20260704/CONDITION.json`
sha256
`001520da91c0d9565dcea97c66e2d8149b95039a26d040e0dd7dda57804d92a5`.
Final epoch 8 fixed-val changed-cell accuracy was `0.9959374822280301`
with level `141/141` true positives but `7` false positives. Boundary:
v13 does not solve the key/lock/target requirement and should not replace
the v12/graph-adaptive live lane.

Contra training has now closed the missing level-0 target-designation loop up to
recorded-frame evidence:

- **Stage25 goal-label dataset:** built from recorded completion windows,
  not a simulator. Dataset:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage25_goal_label_dataset_20260704/goal_label_dataset.pt`,
  sha256
  `2108bf61fc6bbe0b59ae46f9fee768660470efd74e99a5537d265ed051e3ae17`.
  Summary: `881` samples, `735` train, `146` val, `62` unique label
  groups, levels `0->1:707`, `1->2:146`, `2->3:22`, `3->4:6`.
  Review page verified at `http://127.0.0.1:8767/index.html`.
- **Stage26 goal head:** additive Contra training head trained from v12 parent
  `outcome_v14_goal_head_v12parent_20260704/best.pt`, sha256
  `cae6d8aa3d5ffb3a8aa17281090320b2e9eb2ddc8bfc32f3b1cc00484f104a92`.
  Best label-val metrics: mean IoU `0.9989463885625204`, target
  `0.9999203681945801`, key `0.9994872808456421`, source
  `0.9974315166473389`; unseen-group mean IoU
  `0.9781931638717651`. Review page verified at
  `http://127.0.0.1:8768/index.html`.
- **Stage27 post-level-0 diagnostic:** initial comparison against the later
  1→2 pre-completion key location failed by design: target IoU
  `0.97165991902834`, source IoU `1.0`, but key IoU `0.0`. The key mask
  had `100%` drift between the first stable post-level-0 frame and the
  later pre-completion frame. Condition sha256
  `d59c6d63a7f20bff118c891074e173cb0e9d6935aae3dc6666e094dade41e098`.
  This was the right failure: it proved the current-frame key object cannot
  be scored against its future moved position.
- **Stage28b current-frame designation probe:** reran the same recorded live
  holdout frames with the corrected label space: target locks stay spatial,
  while key/source raw values are mapped back to their current components in
  the post-level-0 frame. CPU fallback was used after Stage28 CUDA OOM before
  result. Condition sha256
  `5e4a46005e7c9360b0bc768829de057be0d73dc44afc8ee746cd32c2191486a6`.
  Review page verified at `http://127.0.0.1:8769/index.html`; result
  sha256
  `9e675412b9dd3db0461a091e087482605d970ffc73192bdcaa826aa2fa4a0c58`.
  Result: `2/2` live post-level-0 cases passed; mean IoU
  `0.9772199730094466`, target-lock mean IoU `0.97165991902834`,
  current-key mean IoU `0.96`, paint-source mean IoU `1.0`.

Boundary after Stage28b: Contra training can designate the current-frame target
locks, current key block, and source immediately after level 0 on the two
recorded live holdout post-level-0 frames. This is **not** live solve
evidence and **not** runtime wiring evidence. Next valid work is to wire
the goal-head/schema provider into `megarexus_live_policy.py` so the action
system consumes the target inventory before arranging/pouring, then run the
same-path adapter gate. Do not call the project complete until that runtime
provider is verified.

Stage29/29b then added and smoke-tested the runtime provider path. Code
changes:

- `megarexus_live_policy.py` now compatible-loads old checkpoints after the
  additive goal head and exposes opt-in `MEGAREXUS_GOAL_PROVIDER=1`.
- With the provider enabled, each level >= 1 frame emits a current-frame
  schema: target-lock components, current key component, paint source, mask
  pixels, and threshold.
- `MEGAREXUS_GOAL_POUR_BUDGET=1` derives the pour cap from the current
  target count and records `goal_target_count`, `pour_cap`, and `goal_schema`
  on arrange-pour ledger rows.
- Generic arrange planning seeds the movable roster with the detected
  current key component when the graph-template path is not taking over.
  In the current graph-template path, the schema is still attached to plan
  summaries and pour rows; the graph roster already includes the key-like
  components in the replayed level-1 scene.

Stage29 replay showed the wiring worked but the first target-count summary
included small prediction islands. It is retained as a diagnostic:
condition sha256
`65d1a0b5774c29adbd4d413e3e579a6c4d180e975b4cfa3adff794bbd1cdf231`.
Stage29b tightened target component counting to mask islands >= 32 pixels
and passed the saved-frame replay wiring smoke. Condition sha256
`fbbc5f5911a2dc6f820f5e77057f6f3b5c774a748849d4b8c66fa737ad309c7a`;
review page verified at `http://127.0.0.1:8770/index.html`; raw report
sha256
`f332a9364fd58f9eb950454b8076eafbe3c6d68d70f42adb16935596521af082`.
Result across both live frame archives: `13` provider updates per episode,
first level-1 target count `3`, latest target count `3`, plan summaries
carry `goal_schema`, and the level-1 arrange pour records
`goal_target_count=3` and `pour_cap=3`.

Boundary after Stage29b: runtime wiring is smoke-verified on saved frames,
not environment-gated. Next valid step is a same-path offline adapter gate
with `MEGAREXUS_GOAL_PROVIDER=1`, `MEGAREXUS_GOAL_POUR_BUDGET=1`, v14
checkpoint, paired random baseline, and review page before any live scout.

## 1. Program state and trajectory

Live evidence is genuinely monotonic across the program's three live scouts,
all with zero policy errors:

| Live run | Checkpoint + policy | Score | Levels |
|---|---|---|---|
| v9retry first live (2026-07-02) | v9retry last, plain adapter | 4.7619 | 1 |
| Stage14 (2026-07-03) | v12 best + graph-adaptive | 14.2857 | 2 |
| Stage20 (2026-07-03, current boundary) | v12 best + graph-adaptive + Stage17 L2 template | 28.5714 | 3 |

Current basin: **level-3 control**. Stage20 reached level 3, built 8 plans
and poured 6 times at L3, then exhausted budget (179 plan-budget no-ops +
209 pour-budget no-ops in the trace).

The honest structural read of the run history:

- **The CNN has been frozen at v8 quality since 2026-07-02.**
  `outcome_v8/last.pt` was the last checkpoint that measurably improved the
  model (settle IoU 0.29 → 0.656, level 109/109 tp / 0 fp, first
  model-guided level-2 plumbing 10/10 vs 1/10 random).
- Every later attempt to improve the CNN either failed the fixed-v8 no-loss
  gate (v9, v10 fresh/ft/sel, v11 — changed-cell drift + level
  false-positives) or passed it by changing nothing measurable (v10dynonly =
  dynamics-only prefixes; v12 = 8,321-param level-head recalibration; both
  metrically identical to v8 on the fixed val).
- The live level-2 and level-3 gains came from **action-side wiring** on top
  of the frozen model: the graph-adaptive arrange policy (perturb gate 9/10)
  and the Stage17 raw action-ID template family (36/1458 candidates reached
  level >= 3). This boundary is already stated in the notes; keep stating it.
- The heldout-perturb basin (seeds 20290103/04/05, levels=1, actions=43) was
  never solved by any CNN change — only by the adaptive graph policy. Seed
  20290106 remains a real miss (9/10).

**What Stage23/v13 actually tests:** whether the Stage22b human-mined
level-3→4 transition shard (8 trajectories; 6/12 controlled offline replays
reached level 4) can teach the CNN new dynamics. It unfreezes far more than
any run since v11 (dynamics, next-frame decoder, settle, level, control
heads) at lr 5e-5 (5x v12's lr). This is the first real model-learning
attempt since v11. **Fixed-val failure risk is high by precedent** — v11
failed with less unfrozen — and the new shard is tiny relative to the base
shards. That is acceptable: it is a properly recorded, gated experiment. The
decision tree in section 6 exists so a failure becomes a clean controlled
negative, not churn.

## 2. Level-3 mechanic — user-provided direction (2026-07-03)

The user described the level-3 mechanic after the wrap. Use this as research
direction under the honesty rules; it is NOT evidence and must never become
a hand-coded label, reward, or simulator.

- Two top objects split the flow; unblocked, the level produces four
  streams. The sidewalls can block: **placing the top two blocks against the
  sidewalls converts four streams into two.**
- The two remaining streams are distributed to two catch gloves, which score
  the bucket.
- The more important lesson: **calculate how many pours are actually needed,
  then do not overpour.** Pours are a scarce, fatal resource (level-2
  finding: the 5th pour call is an instant GAME_OVER). The Stage20 stall is
  consistent with this reading — 6 pours at level 3 exhausted the MAX_POURS
  budget with no level-up, followed by 209 pour-budget no-ops: a retry loop
  where a calculated pour count was needed.
- Framing: an interrelational calibration problem. The system needs a robust
  enough imagination (dynamics/settle heads) to sequence and navigate paint
  between objects and predict per-pour outcomes, so that pour count becomes
  a planned quantity rather than a retry loop.

Legal uses (research direction): shape Lane B's candidate family
(wall-adjacent placements of the two top movables; pour count as an explicit
plan variable scored by imagined outcome); design data-collection or probe
conditions; after v13, evaluate specifically whether imagined rollouts
predict level-3 paint routing well enough to count needed pours before
pouring. Illegal uses (answer-key ban): hand-coding the stream-splitting
mechanics into training labels, rewards, or a spill model. The 8 kept
Stage22b trajectories are legal experience and may already embody the
wall-adjacent arrangement — verify from the captured frames, do not assume.

### 2b. Lock/key goal linkage — user follow-up + status audit (2026-07-03)

The user sharpened the direction: **pours must be directly linked to the
lock-and-key mechanism.** The lock/key signal defines pour amounts — defines
everything — because it sets the goal. Specifically: the **singular color
block is the key; the U-shaped targets are ALWAYS opposite it.**

Status audit (verified against code and logs, 2026-07-03):

- **Not wired:** `megarexus_live_policy.py` has zero references to the
  lock/key seeker or any goal signal. Pour count is the hard-coded
  `MAX_POURS_PER_LEVEL = 6` cap — the Stage20 stall is the direct symptom.
- **Seeker parked on a stale negative:** the goal seeker exists
  (`codexlab/contra_head/vision.py::antisymmetric_relationship_hypotheses`,
  informer-only, truth flags False) but its only controlled test was
  2026-07-01: 0/80 confident facings on sp80+snake held-out frames. It has
  never seen the Stage14/Stage20 live frames and was never re-run with the
  now-much-more-specific hypothesis (singular color block vs U-shaped
  targets, opposite placement).
- **No trained head:** the CNN has no goal head. Nothing recognizes the
  key-to-opposite-target relation.

**The level-0 goal lock (user, same day — this is the centerpiece):** the
goal is VISIBLE at level-0 completion. The environment shows the objective
when it scores. Verified in the artifacts: Stage20
`policy_frames/policy_frames_meta.json` record 5 is the levels 0→1
transition with the full 20-frame scoring animation captured (records 15
and 26 hold the 1→2 and 2→3 completions, 22 and 25 frames). After that
window the system has everything needed to define the goal schema — and
the schema is the LOCK INVENTORY, not the pour history:

- **Primary (the solving target): how many lock/bucket/targets there are
  and WHERE they are.** This is "where do we shoot." Target count and
  positions define the objective.
- **Secondary (supply): there is (Y) paint — where is it coming from.**
  Source count and origins.
- **Derived: required pours/streams come FROM the lock count** — match
  supply routing to the lock inventory. Never copied from how many pours a
  past level happened to take, and never a hard-coded cap.

The ghost/imagination should LOCK IN that goal schema and the paint routing
from the level-0 observation. There is no excuse for the policy not knowing
the objective — how many locks there are and where — after level 0. This
also means
the goal-head training set already exists: every completed level in the
corpus (human demos to level 4+, both live runs' transition records, the 8
Stage22b trajectories) is a free hindsight goal label. Nobody has extracted
it yet.

Contra training (goal-signal linkage), in order of cost:

1. **Re-probe with ground truth (read-only, can run DURING v13 training —
   no GPU/resource conflict):** run `antisymmetric_relationship_hypotheses`
   over the captured live frames (Stage20 `policy_frames.npz`, plus
   Stage14's), stratified by level. The level-0 completion window gives
   GROUND TRUTH: the scored target is known from the animation, so the
   seeker's hypotheses can be validated against it, not eyeballed. If the
   generic `component_asymmetry` detector misses the relation, a more
   specific informer in the same family (singular-color-block vs U-shape,
   opposite-side scoring) is a legal Contra-role addition — informer-only,
   truth flags False.
2. **Runtime level-0 goal lock (wire as goal provider):** after the first
   level-up in an episode, run hindsight extraction over the completion
   window (the captured n_frames animation of the scoring pour) and emit
   the goal schema: the lock/target inventory (count + positions — where do
   we shoot), the key block, and the paint sources ((Y) count + origins —
   where it comes from). Lock that schema. At each subsequent level,
   re-instantiate it: count and locate the locks FIRST, identify the paint
   sources, derive the required streams/pours from the lock count, plan the
   arrangement so each stream is routed to a lock, pour exactly that many.
   The ghost confirms before committing: imagine the pour and check the
   predicted paint reaches the identified lock. This is Contra informing
   action selection plus LeWM/Rex verification — the same boundary as the
   graph-adaptive policy.
3. **Train the Contra goal head (hindsight-labeled, a proper future condition):**
   extract the goal labels from every completed level already in the corpus
   (env fields + pixels in hindsight — legal label source; zero new
   collection needed) and train a head to recognize the key-to-opposite-
   target relation from the pre-completion frame. This is the "trained head
   that recognizes the key" — record as its own CONDITION.json; full gate
   stack applies.

Role ownership: goal seeker = Contra/informer; pour-outcome prediction =
LeWM/Rex imagination; pour count = plan variable derived from both. Never a
hand-coded key-target map in training.

## 3. Preflight checklist (mandatory before relaunch)

Run all of these; any failure blocks relaunch until resolved.

1. No active Megarexus training process: `pgrep -af train.py` is empty.
2. GPU free: `nvidia-smi` shows no compute process. (Verified free at the
   post-wrap review on 2026-07-03; the unrelated Leanstral export PID
   2517434 had finished.)
3. Condition integrity: `sha256sum
   synthesis_lab/experiments/megarexus_cnn/runs/stage23_v13_level3_human_mined_20260704/CONDITION.json`
   equals
   `b09ca513f43ce59301c1c236aca81b70e443ccd346bfa86a2df4dddf27790950`
   (post-erratum authoritative hash; see the erratum in the wrap note).
4. Init checkpoint present:
   `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v12_headlevel_perturb_recovery_20260703/best.pt`
   hashes to
   `211e1eca2543f581b871f4a8f26dee583d8a0205eef5ba92ba1073be61939999`.
5. All 11 train + 6 val shards listed in the condition exist and hash-match
   (all verified correct at the post-wrap review, including the corrected
   `perturb_recovery_mining_v11_train.pt` =
   `8e8d850d098e4a33abdfa028109b8ee890d6116c6933b9e340adb21ad18949d5`).
6. Evidence server up: `http://127.0.0.1:8765/` returns 200.
7. Expected quirk: the 20-byte startup log
   `runs/outcome_v13_level3_human_mined_20260704_train.log` from the
   interrupted launch will be overwritten/appended. That is fine. Do not
   treat its current content as a result.

## 4. Relaunch procedure (authorized)

The user has authorized auto-relaunch once preflight passes — do not wait
for a further go-ahead.

- Run the exact `command[]` recorded in the Stage23 `CONDITION.json`,
  unmodified. In particular keep `--num-workers 0` (the memsafe condition
  from v12) and `--seed 0`.
- Background it and tee output to the recorded log path.
- Update the condition `status` field from
  `launch_interrupted_by_user_before_checkpoint` to a relaunched status with
  a timestamp (and note the new condition hash in the session log if you
  edit the file).
- Expected wallclock: roughly 1–1.5 h for 8 epochs on the GB10 (v12's
  8-epoch run fit that window). `--save-every 1` means a checkpoint per
  epoch.
- Babysit per the session-handoff protocol: match sleep timers to epoch
  cadence, check the log for epoch/checkpoint lines, stop only for runtime
  failure or resource conflict. Do not claim any result until at least one
  checkpoint plus `history.json` exists.

## 5. Gate chain after training (pre-committed)

Strictly ordered; never skip forward. This instantiates `gate_after` from
the condition.

1. **Fixed-v8 no-loss gate:** `gate_eval_many.py` over all v13 epoch
   checkpoints + best + last, against `outcome_v8/last.pt` and the existing
   v12 rows, all heads on the fixed v8 val shards.
2. **Level-3 behavioral gate (only if 1 passes):** same-path
   `adapter_gate.py --target-levels 4` — level-3 completion means reaching
   level 4 — 10 paired seeds vs random, with the standard discipline checks
   (zero policy errors, no frame fallback, pour budgets respected).
3. **One live scout (only if 1 and 2 pass):** target is level 4 / score
   above 28.5714. One scorecard, then wrap per protocol.
4. **Review pages before raw JSON:** build the review surface for every gate
   before reporting numbers.

No live scorecard from training metrics alone, ever.

## 6. Decision tree

- **Fixed-val fails (v11-style drift):** record the controlled negative in
  RESEARCH_LOG with the failing rows. Do NOT iterate lr/epochs blind. Two
  pre-approved next conditions, one change each (pick one, record a new
  CONDITION.json):
  - **v13b:** shrink trainable prefixes to level + control heads only, lr
    back to 1e-5, same shards. Tests whether the shard helps at all without
    touching dynamics.
  - **Lane B (action-side, no CNN change):** mine a level-3 plan/template
    family from the 8 kept Stage22b trajectories, Stage17-style; wire it
    into the graph-adaptive policy; gate with
    `adapter_gate.py --target-levels 4`. This is the proven-cheaper path —
    it is exactly how levels 2 and 3 fell. Per section 2, bias the candidate
    family toward wall-adjacent placements of the two top movables, and make
    pour count an explicit plan variable (plan = arrangement + N pours,
    scored by imagined settle/level outcome) instead of pouring to the
    budget cap.
- **Fixed-val passes but the level-3 gate fails:** v13 did not learn the
  transition usefully. Keep v12 best as the live checkpoint. Diff v13's
  per-head `history.json` against v12 to see what actually moved, then go to
  Lane B.
- **Both gates pass:** run the single live scout, then write the wrap note
  and HANDOFF §0 update per the session-handoff protocol.

## 7. Standing boundaries (restated, non-negotiable)

- Answer-key ban: `sp80_spill_model.py` and any hand-authored mechanic model
  never touch training in any form.
- Stage22b is controlled offline replay data — not model-selected, not live,
  not solve evidence.
- Action-side wiring (graph policy, mined templates) is never claimed as
  CNN-only learning.
- The closed SDK payload in `result.json` is the controlling live evidence;
  public scorecard URLs have returned 404.
- Stage numbers are local bookkeeping labels, not a countdown.
- Paired random baselines always; one backlog element per training run;
  negatives get documented, not buried.

## 8. Open items

- The arXiv fit check for `https://arxiv.org/abs/2607.01224` has no valid
  verdict. Do not cite one until a fresh read-only review reads the paper
  and any official code.
- Perturb seed 20290106 remains a real 9/10 miss for the graph-adaptive
  policy — a candidate future condition, lower priority than the level-3
  basin.
- The Stage20 level-3 stall trace (8 plans built, 6 pours, 179 plan-budget +
  209 pour-budget no-ops) is the raw material for Lane B analysis if v13
  fails — reread it through the section-2 lens (overpour retry loop vs
  calculated pour count).
- Pour-count calibration as its own future condition: use the imagination
  heads to predict per-pour outcome and stop pouring when the predicted
  score is achieved; gate on wasted-pour count, not only levels reached.
  Per section 2b, the pour count should ultimately be set by the lock/key
  goal signal, not by a heuristic stopping rule.
- **Contra training provider-enabled offline adapter gate is now closed:** Stage24/25/26/28b
  closed the static extraction, dataset, trained goal head, and post-level-0
  current-object designation probes; Stage29b smoke-verified runtime wiring
  on saved frames; Stage30 smoke-verified the provider-enabled same-path
  adapter gate; Stage31 passed the full 10-seed provider-enabled same-path
  adapter gate. This is local offline evidence, not a live scorecard. The
  next missing piece is a one-scorecard live scout with the provider enabled,
  only after a fresh release/live condition is recorded.
- Verify from the Stage22b captured frames whether the 8 kept human-mined
  trajectories use the wall-adjacent arrangement described in section 2.

## 9. Stage30/31 update — Contra training after level 0 (2026-07-04)

The missed critical element has been addressed in the current lane: level 0
is treated as the target-solution reveal that defines the goal schema for
subsequent levels. The trained additive goal head designates three channels
from the current frame after level 0: `target_lock`, `key_block`, and
`paint_source`. Runtime wiring exposes that schema through
`megarexus_live_policy.py` with `MEGAREXUS_GOAL_PROVIDER=1`, and the adapter
gate now records provider updates, target counts, frame provenance, and
pour-budget checks.

Stage30 smoke:

- Run dir:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage30_goal_provider_adapter_smoke_20260704`
- Review page: `http://127.0.0.1:8771/index.html` (verified `200 OK`,
  browser screenshot saved as `review_screenshot.png`)
- Result: `PASS`, model `1/1`, paired random `0/1`, target levels `2`
- Model episode: seed `20280001`, levels_completed `3`, actions `120`,
  provider enabled, pour budget enabled, provider updates `115`, latest
  target count `3`, all adapter/provider checks true
- Hashes: `adapter_gate.json`
  `d5e4bc09eeac74eecd8c6ed25d7d18c3e170d3f182fc43c72189f9479f02d708`;
  `CONDITION.json`
  `c8656210dc35daf1934f65f9e771f834024fd82917ad84105d5c97c870a3476c`;
  `index.html`
  `cb2dc2185a4fb017597d755b1c9bb22118fc535baaac84a0a930abd48804a3b4`

Stage31 full provider gate:

- Run dir:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage31_goal_provider_adapter_gate_20260704`
- Review page: `http://127.0.0.1:8772/index.html` (verified `200 OK`,
  browser screenshot saved as `review_screenshot.png`)
- Condition: local offline adapter gate through the same `run_game` path used
  by the live scout; target `levels_completed >= 2`; provider enabled;
  provider pour budget enabled; threshold `0.5`; `CUDA_VISIBLE_DEVICES=`;
  paired random baseline; seeds `20280001..20280010`; `max_actions=120`
- Result: `PASS`, model `10/10`, paired random `0/10`
- Model evidence: every model episode reached levels_completed `3`; every
  episode reported provider enabled, pour budget enabled, provider updates
  `115`, latest target count `3`; every episode passed
  `levels_ge_2`, `levels_ge_target`, `zero_policy_errors`,
  `no_frames_fallback`, `no_pour_loop`, `pour_budget_respected`, and
  `all_pours_confident`
- Random baseline levels_completed were `[0, 0, 1, 0, 0, 0, 0, 0, 0, 1]`
- Hashes: `adapter_gate.json`
  `4c68d7f7a9349a75b4f853647362bb1ea33d2b62354eb06a4a2dc7377daf9fac`;
  `CONDITION.json`
  `5560178f4c55064fd00b4c3bfb8d58d335f1cc119477486f21359ef30f57753f`;
  `index.html`
  `56f3f5fab57268706661bfab0d6cc5730f96cfddf42b0b6b8d6f1a2309f7bf96`;
  review screenshot
  `40e2cae5686fe5c9cafda14e197d5c61c90457f17dcfd2b63b8e54237db23167`

Boundary: Stage31 proves the Contra training provider can be used by the same-path
offline adapter gate after level 0 to designate target/key/source information
and constrain pours. It does not prove live ARCPrize scorecard performance,
does not prove level-4 completion, and is not a release claim.

Stage32 live scorecard is now complete from that recorded condition:
`synthesis_lab/experiments/megarexus_cnn/runs/stage32_live_scout_v14_goal_provider_20260704/CONDITION.json`,
sha256
`53df6a85985e98eef50fc59641c20d7fb76f4ce10511529469e3baf1609e0fec`.
Run dir:
`codexlab/runs/arc_live_sp80_contra_training_v14_goal_provider_20260704T060248Z`.
Review page verified `200 OK`: `http://127.0.0.1:8773/index.html`.
Closed SDK scorecard `666f929b-d875-49a9-9034-f83030411c7d`, score
`28.57142857142857`, levels_completed `3`, actions `497`, resets `3`,
policy_error_count `0`. Contra training provider enabled, provider updates
`492` across levels `1,2,3`, latest target count `3`, frame fallback `0`,
captured calls `497`, and all goal-budget ledger rows use
`goal_target_count=3` / `pour_cap=3`.

## 10. Minimal completion audit (2026-07-04)

- Contra training is trained and offline-verified for key/lock/target after
  level 0: Stage26 best checkpoint, Stage28b recorded-frame probe, and
  Stage31 provider-enabled same-path adapter gate are the controlling
  evidence.
- Stage32 supplies the missing live scorecard evidence: the Contra training
  provider was enabled and designated target/key/source after level 0 during
  a controlled live run that reached levels_completed `3` with zero policy
  errors.
- The user goal for key/lock/target designation after level 0 is complete.
  Boundary: this is not a level-4 completion claim and not a broader release
  claim.
- Keep this lane named **Contra training** in handoff/reporting.

## 11. Paint-output correction (2026-07-04)

Stage32 is still a legal closed-SDK live scorecard (`28.57142857142857`,
levels_completed `3`), but it is not public/ranked ARCPrize submission
evidence. Its frame audit exposed the missed critical element: the old
provider treated channel 2 as one `paint_source`, while level 3 has two
paint-color output blocks. From here, Contra training uses
`paint_output`: every small block of the paint color is a paint output.

- Stage33 rebuilt the goal-label dataset with channels
  `target_lock`, `key_block`, `paint_output`. Dataset:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage33_paint_output_label_dataset_20260704/goal_label_dataset.pt`,
  sha256
  `20937e2e57eab0dbf1d39917952b5d9a74c4bcff4398b9731b150a15c6da0c79`.
  Coverage: `917` samples; paint-output counts `{1: 877, 2: 6, 3: 34}`.
- Stage34 trained the replacement goal head from the v12 parent. Best
  checkpoint:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v15_paint_output_goal_head_v12parent_20260704/best.pt`,
  sha256
  `756b37cadba53e2798da7bfc631da16652d579c48abe185f69f82c0b0140f064`.
  Label-val mean IoU `0.8665396571159363`; paint-output IoU
  `0.6003924012184143`, recall `1.0`, precision `0.6003924012184143`.
- Stage35 focused probe:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage35_paint_output_multi_spout_probe_20260704/index.html`.
  The raw mask counted the 3-output validation cases correctly, but on the
  actual Stage32 level-3 two-output frames it only seeded one visible raw-4
  paint block. `megarexus_live_policy.py` now expands a detected
  paint-output seed to all small same-color paint blocks. Provider probe
  `provider_expansion_probe.json` sha256
  `4ff12914120d432c5752b0c8612704c6d2283394b809c195f3d598d0ce955e2b`
  reports both raw-4 outputs on Stage32 calls `26` through `31`.

Boundary: Stage35 is local training/provider evidence, not a new live
scorecard. The next live condition must use the v15 paint-output checkpoint
and the same-color provider expansion.

Stage36 human solution hint: the user supplied an annotated level-3 image:
`synthesis_lab/experiments/megarexus_cnn/runs/stage36_human_solution_verticals_review_20260704/QDTOJK0.png`,
sha256
`c00282a527013616c6aa7b45a651ad4ef8b4094cd6d8b91e750938b505ed1a74`.
Read it as a human-supplied hypothesis, not controlled action evidence:
level 3 is about controlling verticals; the mechanism is triple catch plus
double paint. Next Contra training target is vertical catch-routing/control.
