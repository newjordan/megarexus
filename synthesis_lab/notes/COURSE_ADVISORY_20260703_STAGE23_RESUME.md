# Course advisory — Stage23/v13 resume (written 2026-07-03, post-wrap review)

This is the pick-up document for the next agent session. It supersedes the
resume section of `WRAP_20260703_STAGE23_V13_READY_NOT_RUN.md` by adding the
whole-program assessment, an authorized relaunch procedure, and a
pre-committed decision tree. Read `HANDOFF.md` for standing rules; this file
tells you what to do next and why.

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
- Verify from the Stage22b captured frames whether the 8 kept human-mined
  trajectories use the wall-adjacent arrangement described in section 2.
