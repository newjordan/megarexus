# Research log

## 2026-07-04 - No-imitation rule moved into executable training guards

The no-imitation rule is now enforced in active code, not only notes.
Added `synthesis_lab/orchestration/no_imitation_guard.py` and wired it into
`train_watch.py`, `build_goal_label_dataset.py`, `dataset.py`, `train.py`,
and `train_goal_head.py`.

The guard refuses banned answer/imitation lineage markers in training
conditions, dataset-builder inputs, rollout shard paths, goal-dataset
manifests, and checkpoint config/manifest metadata. Banned markers include
human/demo/route/replay/answer/graph-demo/live-scorecard lineage plus the
known contaminated Stage25/26/33/34 artifacts.

The active transition dataset now defaults to no human corpus and raises if
human transitions are requested or found. `train.py` no longer trains or
reports the old behavior-cloning objective; active CNN training is mechanics
and interaction learning only.

Verification:

- `python3 -m py_compile synthesis_lab/orchestration/no_imitation_guard.py
  synthesis_lab/orchestration/train_watch.py
  synthesis_lab/orchestration/build_goal_label_dataset.py
  synthesis_lab/experiments/megarexus_cnn/dataset.py
  synthesis_lab/experiments/megarexus_cnn/train.py
  synthesis_lab/experiments/megarexus_cnn/train_goal_head.py` passed.
- `python3 synthesis_lab/orchestration/train_watch.py --condition
  synthesis_lab/experiments/megarexus_cnn/runs/stage23_v13_level3_human_mined_20260704/CONDITION.json
  --dry-run` refused before launch with the no-imitation guard leading the
  failure list.
- `python3 synthesis_lab/experiments/megarexus_cnn/train_goal_head.py
  --goal-dataset
  synthesis_lab/experiments/megarexus_cnn/runs/stage25_goal_label_dataset_20260704/goal_label_dataset.pt
  --init-ckpt
  synthesis_lab/experiments/megarexus_cnn/runs/outcome_v12_headlevel_perturb_recovery_20260703/best.pt
  --out-dir /tmp/megarexus_no_imitation_guard_probe --epochs 0` refused
  immediately on the known contaminated Stage25 dataset marker.

## 2026-07-04 - User ruling: no imitation training, ever

The user explicitly ruled that imitation training is never authorized in this
project. This is now a binding first principle for the CNN lane: no human
demonstrations, human-mined states, route traces, route replays, same-level
solved/eval states, answer files, or graph-demo plans may be used as training
data for CNN-alone/autonomous claims.

The corrected goal is causal model competence from interaction: learn
dynamics, control, catch, settle, level-up, and visible objective perception
from pre-registered non-imitation environment data, then evaluate on disjoint
held-out gates. Training on answer states or routes is the failure mode.

Immediate artifact ruling: Stage25/Stage33 goal-label datasets and
Stage26/Stage34 goal-head checkpoints are historical diagnostics only for
CNN-alone purposes, because the recorded Stage25 and Stage33 dataset-build
commands included
`level3_human_mined_inclusive_stage22b_20260704.pt`. They cannot be used as
parents, training targets, gates, or live scorecard support for a clean
CNN-alone lane. A clean goal-informer must be rebuilt from no-imitation
sources.

Durable regime document:
`synthesis_lab/notes/CNN_TRAINING_REGIME_20260704_NO_IMITATION.md`.

## 2026-07-04 - Route-replay cheats quarantined and active route execution removed

After explicit user authorization, the v16-v26 route-imitation checkpoints
and stage36-stage60 route/human/replay artifacts were quarantined, not
deleted, under:
`synthesis_lab/experiments/megarexus_cnn/runs/_invalid/`.
Route/human/replay data shards were moved under:
`synthesis_lab/experiments/megarexus_cnn/data/_invalid/`.
Dedicated route/human-mining tool scripts were moved under:
`synthesis_lab/orchestration/_invalid/`, and route training scripts under:
`synthesis_lab/experiments/megarexus_cnn/_invalid/`.

Tracked `runs/outcome_v9/best.pt` and `runs/outcome_v9/last.pt` were restored
from git and no longer show as modified. Active code was split so the Contra
goal head remains, but route constants/modules/forward methods were removed
from `model.py`; route additive prefixes were removed from `train.py`;
route execution and route schema attachment were removed from
`megarexus_live_policy.py`; and `adapter_gate.py` no longer has the corrupt
`route_provider_attached` PASS check or route-action review/env fields.

`honesty_audit.py` was updated to report quarantined invalid artifacts
separately. Fresh audit result after cleanup exits 0:
total summary `AUTONOMOUS_CANDIDATE=82, IMITATION_ASSISTED=46,
INVALID_HEADLINE=6`; active summary `AUTONOMOUS_CANDIDATE=70,
IMITATION_ASSISTED=23`; quarantined summary `AUTONOMOUS_CANDIDATE=12,
IMITATION_ASSISTED=23, INVALID_HEADLINE=6`. This means the six invalid
headline gates are no longer active but remain disclosed as quarantined
evidence.

Compile check passed for `model.py`, `train.py`, `megarexus_live_policy.py`,
`adapter_gate.py`, `honesty_audit.py`, and `train_watch.py`.
`train_watch.py --dry-run` against Stage23/v13 still refuses, now with four
problems: `train.py` hash mismatch, `model.py` hash mismatch,
`gate_eval_many.py` hash mismatch, and missing quarantined shard
`data/level3_human_mined_inclusive_stage22b_20260704.pt`.

Boundary unchanged: no training, no gates, no live runs, no commits in this
cleanup. Honest evidence remains live 28.57142857142857 / 3 levels WITH
graph-demo-plan assistance; CNN-alone live evidence remains 1 level
(`v9retry`). Stage23/v13 remains suspended.

## 2026-07-04 - Contra training human solution hint: vertical control

The user supplied an annotated level-3 image:
`synthesis_lab/experiments/megarexus_cnn/runs/stage36_human_solution_verticals_review_20260704/QDTOJK0.png`,
sha256
`c00282a527013616c6aa7b45a651ad4ef8b4094cd6d8b91e750938b505ed1a74`.
Review page:
`synthesis_lab/experiments/megarexus_cnn/runs/stage36_human_solution_verticals_review_20260704/index.html`.

Interpretation boundary: user-supplied static hypothesis only, not a
controlled action trace and not model evidence. The hint is that level 3 is
about controlling verticals, with triple catch plus double paint. Next Contra
training target: vertical catch-routing/control, after reproducing the
solution with controlled action provenance or a mined action trace.

## 2026-07-04 - Contra training paint-output correction

Stage32 remained a valid closed-SDK live scorecard result (`28.57142857142857`,
levels_completed `3`), but the frame audit exposed a missed Contra training
semantic: level 3 has two paint outputs. The old channel-2 meaning was
`paint_source`; the corrected channel is `paint_output`, meaning every small
block of the paint color is a paint output.

Stage33 rebuilt the goal-label dataset with channels `target_lock`,
`key_block`, `paint_output`:
`synthesis_lab/experiments/megarexus_cnn/runs/stage33_paint_output_label_dataset_20260704/goal_label_dataset.pt`,
sha256
`20937e2e57eab0dbf1d39917952b5d9a74c4bcff4398b9731b150a15c6da0c79`.
Coverage: `917` samples; paint-output count distribution
`{1: 877, 2: 6, 3: 34}`.

Stage34 trained the v15 paint-output goal head from the v12 parent. Best
checkpoint:
`synthesis_lab/experiments/megarexus_cnn/runs/outcome_v15_paint_output_goal_head_v12parent_20260704/best.pt`,
sha256
`756b37cadba53e2798da7bfc631da16652d579c48abe185f69f82c0b0140f064`.
Final label-val mean IoU `0.8665396571159363`; paint-output IoU
`0.6003924012184143`, recall `1.0`, precision `0.6003924012184143`.

Stage35 focused probe:
`synthesis_lab/experiments/megarexus_cnn/runs/stage35_paint_output_multi_spout_probe_20260704/index.html`.
Raw-mask probe counted the 3-output validation cases correctly, but on the
actual Stage32 level-3 two-output frames it seeded only one visible raw-4
paint block. Runtime provider mapping now expands a detected paint-output
seed to all small same-color paint blocks in `megarexus_live_policy.py`.
Provider probe:
`synthesis_lab/experiments/megarexus_cnn/runs/stage35_paint_output_multi_spout_probe_20260704/provider_expansion_probe.json`,
sha256
`4ff12914120d432c5752b0c8612704c6d2283394b809c195f3d598d0ce955e2b`,
reports both raw-4 paint outputs on Stage32 calls `26` through `31`.

Boundary: local training/provider evidence only, not a new live scorecard
and not public ARCPrize ranking evidence. The next live condition should use
the v15 paint-output checkpoint and same-color provider expansion.

## 2026-07-04 - Contra training goal head trained; post-level-0 current designation passed

Apollo v13 completed and was synced back locally, but it is not promoted.
Output:
`synthesis_lab/experiments/megarexus_cnn/runs/outcome_v13_level3_human_mined_apollo_torch25_20260704`;
condition:
`synthesis_lab/experiments/megarexus_cnn/runs/stage23_v13_level3_human_mined_apollo_20260704/CONDITION.json`,
sha256
`001520da91c0d9565dcea97c66e2d8149b95039a26d040e0dd7dda57804d92a5`.
Final epoch 8 fixed-val changed-cell accuracy was `0.9959374822280301`,
level true positives `141/141`, level false positives `7`, settle IoU
`0.6583254269449715`, catch accuracy `0.9503012048192772`, control accuracy
`0.9995428223102713`. Boundary: not promoted and does not address
key/lock/target runtime wiring.

Stage25 built the hindsight goal-label dataset from recorded completion
windows:
`synthesis_lab/experiments/megarexus_cnn/runs/stage25_goal_label_dataset_20260704/goal_label_dataset.pt`,
sha256
`2108bf61fc6bbe0b59ae46f9fee768660470efd74e99a5537d265ed051e3ae17`.
Summary: `881` samples, `735` train, `146` val, `62` unique label groups;
levels `0->1:707`, `1->2:146`, `2->3:22`, `3->4:6`. Review page verified:
`http://127.0.0.1:8767/index.html`.

Stage26 trained the additive goal head from v12 parent. Best checkpoint:
`synthesis_lab/experiments/megarexus_cnn/runs/outcome_v14_goal_head_v12parent_20260704/best.pt`,
sha256
`cae6d8aa3d5ffb3a8aa17281090320b2e9eb2ddc8bfc32f3b1cc00484f104a92`.
Best label-val metrics: mean IoU `0.9989463885625204`, target
`0.9999203681945801`, key `0.9994872808456421`, source
`0.9974315166473389`; unseen-group mean IoU `0.9781931638717651`.
Review page verified: `http://127.0.0.1:8768/index.html`.

Stage27 tested the trained head on the first stable post-level-0 frame but
compared against the later 1->2 pre-completion key mask. It failed usefully:
target IoU `0.97165991902834`, source IoU `1.0`, key IoU `0.0`.
The key mask had `100%` drift between the post-level-0 frame and the later
pre-completion frame, so the comparison was against the key's future moved
location. Condition sha256:
`d59c6d63a7f20bff118c891074e173cb0e9d6935aae3dc6666e094dade41e098`;
result sha256:
`c91d313378bddb28bfa6d8b3343734f1b854d1721ae9260773220ef2bb98c1eb`.

Stage28 corrected the label space but hit CUDA OOM while loading the
checkpoint before producing a result. Condition status is
`failed_cuda_oom_before_result`, sha256
`5d94b12c5261158c92f327a0695cea234d38beb21e3ffb5759490e1d8747a00f`.

Stage28b reran the same corrected current-frame designation probe on CPU.
Condition:
`synthesis_lab/experiments/megarexus_cnn/runs/stage28b_goal_head_post_level0_current_frame_probe_cpu_20260704/CONDITION.json`,
sha256
`5e4a46005e7c9360b0bc768829de057be0d73dc44afc8ee746cd32c2191486a6`.
Review page verified: `http://127.0.0.1:8769/index.html`; result sha256
`9e675412b9dd3db0461a091e087482605d970ffc73192bdcaa826aa2fa4a0c58`.
Result: `2/2` live post-level-0 cases passed; mean IoU
`0.9772199730094466`, target-lock mean IoU `0.97165991902834`, current-key
mean IoU `0.96`, paint-source mean IoU `1.0`.

Boundary: Stage28b proves recorded-frame current-object designation after
level 0 on the two live holdout post-level-0 frames. It is not solve
evidence, not action selection, and not runtime goal-provider wiring. Next
valid step is to wire the schema/head into `megarexus_live_policy.py`, then
run the same-path adapter gate before any live scout.

Stage29/29b added that runtime wiring and smoke-tested it on saved
policy-frame replays. `megarexus_live_policy.py` now compatible-loads old
checkpoints after the additive goal head, exposes opt-in
`MEGAREXUS_GOAL_PROVIDER=1`, emits current-frame target/key/source schemas
for levels >= 1, and with `MEGAREXUS_GOAL_POUR_BUDGET=1` records
goal-derived `goal_target_count`, `pour_cap`, and `goal_schema` on arrange
pour rows. `adapter_gate.py` records these env flags and summarizes provider
updates in judged episodes.

Stage29 replay condition:
`synthesis_lab/experiments/megarexus_cnn/runs/stage29_goal_provider_live_policy_replay_20260704/CONDITION.json`,
sha256
`65d1a0b5774c29adbd4d413e3e579a6c4d180e975b4cfa3adff794bbd1cdf231`.
It proved provider updates and schema attachment, but also exposed noisy
target-count summarization from small prediction islands. Raw report sha256
`e7025463de8e64d580ae91b5d55c8c111923403b92443b8b260803502e71b418`.

Stage29b tightened runtime target component counting to target-mask islands
>= 32 pixels and passed the saved-frame replay wiring smoke. Condition:
`synthesis_lab/experiments/megarexus_cnn/runs/stage29b_goal_provider_live_policy_replay_filtered_20260704/CONDITION.json`,
sha256
`fbbc5f5911a2dc6f820f5e77057f6f3b5c774a748849d4b8c66fa737ad309c7a`.
Review page verified at `http://127.0.0.1:8770/index.html`; raw report
sha256
`f332a9364fd58f9eb950454b8076eafbe3c6d68d70f42adb16935596521af082`.
Across both live frame archives: `13` provider updates per replay,
first level-1 target count `3`, latest target count `3`, plan summaries
carry `goal_schema`, and the level-1 arrange pour records
`goal_target_count=3` and `pour_cap=3`.

Boundary: Stage29b is runtime wiring evidence on saved frames only. It is
not an offline environment gate, not action-outcome proof, and not a live
scorecard. Next valid step is a same-path offline adapter gate with
`MEGAREXUS_GOAL_PROVIDER=1`, `MEGAREXUS_GOAL_POUR_BUDGET=1`, v14 checkpoint,
paired random baseline, and a review page before any live scout.

## 2026-07-03 - Stage23/v13 stopped twice for resource conflicts

Preflight at `2026-07-03T23:03:25-05:00` passed: no active Megarexus
`train.py`, GB10 had no compute process and 0% GPU util, evidence server
`http://127.0.0.1:8765/` returned 200, Stage23 `CONDITION.json`, v12 init
checkpoint, all 11 train shards, and all 6 val shards hash-matched.

Updated
`synthesis_lab/experiments/megarexus_cnn/runs/stage23_v13_level3_human_mined_20260704/CONDITION.json`
status to
`relaunched_2026-07-03T23:03:25-05:00_after_preflight_hash_check`;
new condition sha256:
`09658f5a2d30c264dd68d58369fc8bf71302a71f65c3718c46bf7182e19b7224`.

Relaunched the exact recorded `command[]` with output appended to
`synthesis_lab/experiments/megarexus_cnn/runs/outcome_v13_level3_human_mined_20260704_train.log`.
Wrapper PID `2643869`; actual `train.py` PID `2643873`.

Stopped the relaunch at `2026-07-03T23:05:51-05:00` because an unrelated
GB10 compute process appeared after preflight:
`llama-dspark-cache-probe` PID `2643087`, command rooted at
`/home/frosty40/llama-nvfp4-dspark/build-dspark/bin/llama-dspark-cache-probe`.
Only the Stage23 wrapper process group was terminated (`2643869`, `2643873`,
`2643874`). Post-stop: no Megarexus `train.py`, no v13 output directory,
no checkpoint, no `history.json`; train log is 40 bytes and contains only
two `[train] device=cuda` startup lines.

Updated Stage23 `CONDITION.json` status to
`relaunch_stopped_resource_conflict_before_checkpoint`; current condition
sha256:
`836eaf8dc03b14ab302002f6d35ef218aa68934c128b5f114637b20a4a7ad6b0`.

The conflicting Leanstral probe exited shortly after. A second preflight at
`2026-07-03T23:07:08-05:00` passed: no active Megarexus `train.py`, GB10
had no compute process and 0% GPU util, evidence server returned 200, the
current condition, v12 init checkpoint, all 11 train shards, and all 6 val
shards hash-matched. Relaunched the exact recorded `command[]` again;
wrapper PID `2647364`, actual `train.py` PID `2647368`.

Stopped the second relaunch at `2026-07-03T23:09:21-05:00` because another
unrelated GB10 compute process appeared after preflight: DeepSpec Leanstral
export PID `2647359`, command rooted at
`/home/frosty40/DeepSpec/scripts/leanstral/train_export_leanstral_qwen3_dspark_tiny.py`.
Only the Stage23 wrapper process group was terminated (`2647364`, `2647368`,
`2647369`). Post-stop: no Megarexus `train.py`, no v13 output directory,
no checkpoint, no `history.json`; train log is 60 bytes and contains only
three `[train] device=cuda` startup lines.

Updated Stage23 `CONDITION.json` status to
`blocked_resource_conflict_before_checkpoint`; current condition sha256:
`45f68d8c2ad979e6ae9a77379b506e4a911220d229bf151a5eabc5690e39a718`.
Boundary: this is startup/provenance only, not a v13 result. Next valid
relaunch requires a fresh clean GB10 preflight after the Leanstral/DeepSpec
jobs are gone.

Readiness recheck at `2026-07-03T23:17:53-05:00`: GB10 was clear again
(`nvidia-smi` showed no compute process) and no Megarexus `train.py` was
running. The user clarified the real target: fix the missed level-0
goal-lock/target-inventory signal. Decision: do not relaunch plain v13 as
that fix. v13 remains a recorded model-learning experiment, but the
goal-lock fix needs Contra training first: extract/validate the level-0 completion
goal schema, wire it as a goal provider, and then train a goal head under a
separate recorded condition.

Strict launch-gate recheck at `2026-07-03T23:18:39-05:00`: no Megarexus
`train.py`, but `nvidia-smi` again showed a low-memory unrelated
`llama-cli` compute process on the GB10. Under the recorded Stage23
preflight rule ("no compute process"), do not relaunch yet.

Apollo relaunch after user-approved host switch: created
`synthesis_lab/experiments/megarexus_cnn/runs/stage23_v13_level3_human_mined_apollo_20260704/CONDITION.json`
to record the changed host/GPU/path explicitly. Source condition remains
`stage23_v13_level3_human_mined_20260704/CONDITION.json` sha256
`45f68d8c2ad979e6ae9a77379b506e4a911220d229bf151a5eabc5690e39a718`.
Apollo condition uses workspace `/home/frosty/megarexus_train_20260703`,
GPU `NVIDIA GeForce RTX 4080`, and CUDA-capable Python
`/home/frosty/Loom/.venv/bin/python`; all source-condition code hashes,
train/val shards, and v12 init checkpoint were verified on Apollo.

First Apollo attempt with `/home/frosty/miniconda3/bin/python3` was stopped
immediately because it logged `[train] device=cpu` (CPU-only torch), with no
checkpoint/history/output directory. Second Apollo attempt with
`/home/frosty/ARC3-solution/.venv/bin/python` logged `[train] device=cuda`
but was stopped before output because torch 2.8's `torch.load` default
rejected the trusted numpy-containing `.pt` shards.

Final active relaunch at `2026-07-03T23:29:37-05:00` uses
`/home/frosty/Loom/.venv/bin/python` (torch 2.5.1+cu121); startup log line
is `[train] device=cuda`. Wrapper PID `622267`, launcher PID `622272`,
train PID `622279`; log path:
`synthesis_lab/experiments/megarexus_cnn/runs/outcome_v13_level3_human_mined_apollo_torch25_20260704_train.log`.
Apollo condition sha256 after final launch provenance:
`691c4148849781f13d7b1b5d7823b8569da2fddb20bdda1d41144aab8a4b7ab3`.
Boundary at last check: startup/data-load only, no checkpoint/history yet.

## 2026-07-03 - Stage24 static level-0 goal-schema extraction completed

Created CPU-only extractor/review builder:
`synthesis_lab/orchestration/extract_level0_goal_schema.py`, sha256
`d10e34b828583a83b3dedc6a25c2dfd169d519595946c763857ee90461798489`.
Recorded condition:
`synthesis_lab/experiments/megarexus_cnn/runs/stage24_goal_schema_extraction_20260704/CONDITION.json`,
sha256
`ca60e73ad8cb873792d2bcee1418104672ddde52ba565c23f38409e9da97dcdd`.
Inputs were the recorded Stage20 and Stage14 `policy_frames` archives:
Stage20 npz
`abe380d0a82ede67216e5eb02036cdaa26bf71040aa19fa5eaffa0e0852291a0`,
meta
`03d08a7c28e4dbf4df3430dcb814b00e0c82cdc99dd351d9cbf9da7a46abd7d3`;
Stage14 npz
`a4b31314c93b2d5b47d5766ef01b78a103502c4c672e72fb1efba459e8c40511`,
meta
`c8b8c747ceef0b1fb186023c8c14f222002bbade97c625aa602d5772fe997949`.

Command:
`python3 synthesis_lab/orchestration/extract_level0_goal_schema.py --episode stage20=codexlab/runs/arc_live_sp80_megarexus_v12_graphadaptive_l3_20260704T021637Z/policy_frames --episode stage14=codexlab/runs/arc_live_sp80_megarexus_v12_graphadaptive_20260703T2052/policy_frames --from-level 0 --out-dir synthesis_lab/experiments/megarexus_cnn/runs/stage24_goal_schema_extraction_20260704 --condition synthesis_lab/experiments/megarexus_cnn/runs/stage24_goal_schema_extraction_20260704/CONDITION.json`.

Outputs:
`goal_schema_hypotheses.json` sha256
`e98bc2fee347098de7e077e2611a37e4e2754da409b48dba9ea9e3222c98cb63`,
`manifest.json` sha256
`1ecf5f4708532407d9ffbbcd7a7d4206ec920f8c0e6e20fcd4b56997cf4c57e2`,
and `index.html` sha256
`9b2b507e8618f63360cbc877416c546f5e1c9fdfe2aab4b905fd5606f780768f`.
Review URL `http://127.0.0.1:8766/index.html` returned `200 OK`; the
linked scoring sheet and raw JSON also returned `200 OK`. The visual review
surface was inspected locally.

Result: both recorded level-0 windows agree. The extractor isolates 19
scoring frames in `call0005` and excludes reset frame 19. The late-fill
target candidates are `c0006_v11_00` and `c0007_v11_01`, both changing to
raw value 13. The early-changing key candidate opposite the target centroid
is `c0005_v09_00`. The expanding paint-source candidate is `c0004_v06_00`.
The legacy antisymmetric relationship seeker returns 0 relationships on
these pre-frames, so the old seeker is not sufficient by itself.

Boundary: this is static hindsight review only. It is not control truth,
not a wired goal provider, not solve evidence, and not Contra training mastery. Next
valid Contra training step is a recorded hindsight goal-label dataset over all
completed level windows, followed by a separate goal-head training/eval
condition.

Apollo status at `2026-07-03T23:53:38-05:00`: v13 training is still active
on train PID `622279`, RTX 4080 allocation about 2964 MiB, and epochs 1-5
have checkpointed in
`/home/frosty/megarexus_train_20260703/synthesis_lab/experiments/megarexus_cnn/runs/outcome_v13_level3_human_mined_apollo_torch25_20260704`.
Recent fixed-val lines still show level false positives (`fp=8-9`), so do
not claim v13 success before the full gate chain.

## 2026-07-03 - User refinement: the goal schema is the lock inventory, not pour history

Correction to the level-0 goal-lock framing: what gets extracted is NOT how
many pours the completion took. The goal schema is the **lock inventory** —
how many lock/bucket/targets there are and WHERE they are ("where do we
shoot") is the main solving target. Paint is the supply side: there is (Y)
paint and we care where it comes from (source count + origins). The
required pours/streams are then DERIVED from the lock count by matching
supply routing to the lock inventory — never copied from a past level's
pour history and never a hard-coded cap.

Advisory §2b updated accordingly: goal schema definition (primary = lock
count/positions, secondary = paint sources, derived = required pours),
runtime lock step (count and locate the locks first at every level), and
the hindsight label tuple (pre-completion frame → lock count/positions,
scored target, key block, paint sources; pour history incidental).

## 2026-07-03 - User breakthrough: the goal is VISIBLE at level-0 completion; ghost should lock it in

The user's follow-up on the lock/key direction: the objective is literally
observable when level 0 completes — the scoring animation shows which block
is the key, which U-shaped target scores, and how many pours it took. The
ghost/imagination should LOCK IN the goal and paint routing from that
observation; there is no excuse for the policy not knowing the objective,
and how many targets there are, after level 0.

Verified in artifacts: Stage20 `policy_frames_meta.json` record 5 is the
levels 0→1 transition with a 20-frame scoring animation captured; records
15 (1→2, 22 frames) and 26 (2→3, 25 frames) hold the later completions.
Every level completion the system ever achieved is captured with its full
animation window — and every completed level across the corpus (human
demos to level 4+, both live runs, the 8 Stage22b trajectories) is a free
hindsight goal label. The goal-head training set already exists; it has
never been extracted.

Advisory §2b updated: the "level-0 goal lock" is now the centerpiece of
Contra training — (1) seeker re-probe validated against the level-0 ground truth
instead of eyeballing; (2) runtime goal lock: hindsight-extract
key/target/pour-count from the first completion window, lock as episode
goal schema, re-instantiate per level (count targets → derive pours), with
the ghost confirming by imagining the pour before committing; (3) goal-head
training condition fed by the extracted hindsight labels (zero new
collection). New open item: extract the hindsight goal-label dataset.

## 2026-07-03 - User direction: pours must be linked to the lock/key goal signal; audit says nothing is wired

The user sharpened the level-3 direction: pour amounts must be directly
linked to the lock-and-key mechanism — the lock/key signal sets the goal
and therefore defines pour amounts. Specifically: the singular color block
is the key; the U-shaped targets are ALWAYS opposite it.

Audit answer (verified against code and logs): NO on all three counts.
(1) `megarexus_live_policy.py` has zero lock/key or goal references; pour
count is the hard-coded `MAX_POURS_PER_LEVEL = 6` cap (line 77) — the
Stage20 level-3 stall is the direct symptom. (2) The goal seeker
`codexlab/contra_head/vision.py::antisymmetric_relationship_hypotheses`
exists as informer-only but its single controlled test (2026-07-01) was a
negative — 0/80 confident facings on sp80+snake held-out frames — and it
was never re-run on Stage14/Stage20 live frames or with the specific
key-block/U-target hypothesis. (3) The CNN has no goal head; nothing was
ever trained to recognize the key-to-opposite-target relation.

Recorded as advisory §2b with three-step Contra training: re-probe the seeker on
captured live frames (read-only, legal during v13 training), wire as
informer-only goal provider if it fires, then a hindsight-labeled goal-head
training condition (target-that-scored is recoverable from env fields +
pixels in hindsight — legal label source). Role ownership: Contra informs
the goal, LeWM/Rex imagines pour outcomes, pour count is derived from both;
never a hand-coded key-target map in training.

## 2026-07-03 - User direction: level-3 mechanic and pour-count calibration

After the wrap, the user described the level-3 mechanic as research
direction (legal per the honesty rules; never a hand-coded label, reward,
or simulator): two top objects split the flow into four streams; placing
the top two blocks against the sidewalls converts four streams into two,
which feed two catch gloves that score the bucket. The stated key lesson:
calculate how many pours are actually needed and do not overpour — pours
are scarce and fatal (level-2 finding: 5th pour = instant GAME_OVER). The
user frames level 3 as an interrelational calibration problem that may need
a more robust imagination for sequencing/navigating paint between objects.

This reading is consistent with the Stage20 stall trace: 6 pours at level 3
exhausted the MAX_POURS budget with no level-up, then 209 pour-budget
no-ops — a retry loop where a calculated pour count was needed.

Recorded in `COURSE_ADVISORY_20260703_STAGE23_RESUME.md` §2 with legal/illegal
uses spelled out; Lane B refined to bias wall-adjacent placements and treat
pour count as an explicit plan variable; new open items added (pour-count
calibration condition; verify whether the 8 Stage22b trajectories use the
wall-adjacent arrangement).

## 2026-07-03 - Post-wrap review: condition hash erratum + course advisory

A post-wrap review verified every hash and claim in the Stage23 handoff set
against disk. One defect: the Stage23 `CONDITION.json` recorded an invalid
65-char sha256 for `perturb_recovery_mining_v11_train.pt` (extra `1`
inserted). Fixed in place; verified shard hash
`8e8d850d098e4a33abdfa028109b8ee890d6116c6933b9e340adb21ad18949d5`.
Corrected condition sha256 (current authoritative, supersedes `75e06152...`):
`b09ca513f43ce59301c1c236aca81b70e443ccd346bfa86a2df4dddf27790950`.
Everything else matched, including all code hashes, the v12 init checkpoint,
and all train/val shard hashes spot-checked.

Wrote the pick-up document for the next agent:
`synthesis_lab/notes/COURSE_ADVISORY_20260703_STAGE23_RESUME.md`
(whole-program assessment, preflight checklist, authorized v13 relaunch,
pre-committed gate chain and decision tree). Runtime at review: the Leanstral
export had finished; GPU free, no train process.

## 2026-07-03 - Stage23 condition recorded; launch interrupted before checkpoint

Detailed wrap note:
`synthesis_lab/notes/WRAP_20260703_STAGE23_V13_READY_NOT_RUN.md`.

Created Stage23/v13 condition:
`synthesis_lab/experiments/megarexus_cnn/runs/stage23_v13_level3_human_mined_20260704/CONDITION.json`.
Pre-interruption condition sha256:
`75e06152ce3849d615e6fe6e435da773bac28c76dc8d1bd094449a09518abb21`.
Run label is `new_experiment`: init from v12 best, add
`level3_human_mined_inclusive_stage22b_20260704.pt`, and expand trainable
prefixes to dynamics/next-frame decoder/settle/level/control heads.

The v13 command was launched locally at about `2026-07-03T21:46:41-05:00`.
The user interrupted and asked for handoff/wrap. The process remained alive,
so it was stopped explicitly: Python PID `2548534`, wrapper PID `2548529`,
and `tee` PID `2548535`.

Post-stop check at `2026-07-03T21:47:58-05:00`: no v13 train process remained
and `nvidia-smi` showed no compute process. The log
`synthesis_lab/experiments/megarexus_cnn/runs/outcome_v13_level3_human_mined_20260704_train.log`
exists but is only `20` bytes with `[train] device=cuda`. No v13 output
directory, checkpoint, `history.json`, gate report, or review surface exists.

Evidence boundary: Stage23 is ready to relaunch but has no model result.
Latest live evidence remains Stage20; Stage22b remains offline training data,
not solve evidence.

## 2026-07-03 - Pause note after Stage22b; Stage23 not launched

Paused before launching any v13/Stage23 work. The stage numbers in this lane
are local bookkeeping labels, not a finite countdown and not evidence that the
system is nearly done. "Stage22" means the level-3 human-mined shard/replay
step; "Stage23" is only the proposed v13 training condition label.

Detailed pause note:
`synthesis_lab/notes/REVIEW_PAUSE_20260703_STAGE22.md`.

Latest live evidence remains Stage20: closed SDK payload score
`28.57142857142857`, `total_levels_completed=3`, policy errors `0`, followed
by level-3 control stall. This is not a release/completion claim.

Stage22 exclusive was a controlled negative because it omitted the first
level-4 action from the mined sequence window. Stage22b inclusive is the useful
offline artifact:
`synthesis_lab/experiments/megarexus_cnn/runs/stage22b_level3_human_mined_inclusive_v12_graphadaptive_20260704/level3_human_mined_report.json`.
It used v12 checkpoint sha256
`211e1eca2543f581b871f4a8f26dee583d8a0205eef5ba92ba1073be61939999` and
`synthesis_lab/orchestration/collect_level3_human_mined.py` sha256
`13042c389c6b255cc7d1a8c92b0e2dad94ad9dedf9b43c2274f21744531e9473`.
Result: prefix status `ok`, `prefix_action_count=27`, level-3 frame sha256
`507982e727fe170293044193818337cc11b14a0d820150f533bf95e3c0acf7ca`,
`6/12` candidates reached level 4, and `8` trajectories were kept.
Output shard:
`synthesis_lab/experiments/megarexus_cnn/data/level3_human_mined_inclusive_stage22b_20260704.pt`,
sha256 `2e98eeaa68aa8a71da62ad5a23a1df7704f4781d974991bc809746eeed91e6e3`.
Boundary: controlled offline replay of human-mined level-3 sequences; not
model-selected or live evidence.

The Stage22b review page was verified `200` at
`http://127.0.0.1:8765/synthesis_lab/experiments/megarexus_cnn/runs/stage22b_level3_human_mined_inclusive_v12_graphadaptive_20260704/index.html`.

Runtime refresh at `2026-07-03T21:42:12-05:00`: no active local Megarexus
train/gate/collection/live-scout process was found; evidence server on
`127.0.0.1:8765` is still running. The GB10 is currently occupied by an
unrelated Leanstral export PID `2517434` using CUDA, so recheck local/Apollo
availability before launching any v13 training.

The requested paper fit check for `https://arxiv.org/abs/2607.01224` has no
valid result yet because the user paused before a subagent completed a review.
Do not act on a paper verdict until a fresh read-only review inspects the
arXiv paper and any official linked code.

## 2026-07-03 — Stage17/19/20 cleared live level 2 and hit live 3-level target

After the Stage14 live run stalled at level 2 with four midband bars and
Stage16 proved the anchor-horizontal four-bar family was a controlled
negative (`0/581`), the next condition had to expose a real new
representation. Modified
`synthesis_lab/orchestration/graph_plan_probe_level2.py` (sha256
`26704bed478977f6bd5f92e04661a079d910a7534dd25a7f9fdad13b0961f698`) to
preserve the old horizontal mode and add a raw action-ID template family
with optional single vertical nudges plus a recorded `CONDITION.json`.

Stage 17 report:
`synthesis_lab/experiments/megarexus_cnn/runs/stage17_level2_action_template_probe_v12_graphadaptive_20260704/graph_plan_probe_level2.json`,
sha256
`4bfa869576771d9340c82085e3b9a6c2109fc7dbf04ec3c788b249d04cfb2140`;
condition sha256
`d6f1ca052015f9fe6dd217a8969286c1f484fcd1bb91b8005eabf837143e00fe`;
review sha256
`60680326d3acf20893122bf9f8f3a2cfb1ba80213aca50816c18e85f4810d7d0`.
Result: `36/1458` candidates reached `levels_completed >= 3`. The base
winning candidate was `codexlab_solution_level2_action_ids_counts_1_2_2_1`,
with raw action IDs by y-rank `0:[4]`, `1:[3,3]`, `2:[3,3]`,
`3:[4]`.

Wired the Stage17-backed four-bar branch into
`synthesis_lab/orchestration/megarexus_live_policy.py`, sha256
`681266eff43010a2ea633ac07ea100f50324b5e25e5c6b040501b4970792cb9b`.
This is action-side representation wiring, not a claim that the CNN alone
learned the level-2 transition. Also updated
`synthesis_lab/orchestration/adapter_gate.py`, sha256
`e4789d878f51c066e46410e702e408b2db82e45260584f747f7cf66437a3e515`,
to emit a condition artifact and support `--target-levels`.

Stage 19 same-path adapter gate with target `levels_completed >= 3`
passed:
`synthesis_lab/experiments/megarexus_cnn/runs/stage19_adapter_gate_l3_full_v12_graphadaptive_20260704/adapter_gate.json`,
sha256
`823a58ccfcf41956aa7ff0d336d3998e0a2db9bde19d2e34e3f0bf5a52efedeb`.
Condition sha256
`f77b7d0b8d09809ba1f9e43419eb5270d6e1e70d5166081e560c4bfe7ffed60d`;
review sha256
`610012828668bf2a5dd7391c0dde9b6c15ba444ab438970c91bfd512cada6ad4`.
Verdict: `PASS`, model `10/10` reached `levels_completed=3`, paired
random reached `levels>=2` in `0/10`, zero policy errors.

Launched one live ARC SP80 scorecard under condition
`synthesis_lab/experiments/megarexus_cnn/runs/stage20_live_scout_l3_v12_graphadaptive_20260704/CONDITION.json`,
sha256
`95342da97ef99c6baf268b60b0b5a3965749f62faada0a2a8f2beea20f93ace7`.
Run directory:
`codexlab/runs/arc_live_sp80_megarexus_v12_graphadaptive_l3_20260704T021637Z`.
Scorecard id `453a74e8-9113-43a8-a37a-fcdfacac55ce`. Closed SDK payload
in `result.json` reported score `28.57142857142857`,
`total_levels_completed=3`, policy errors `0`; per-game row
`sp80-589a99af`, `levels_completed=3`, `actions=496`, `resets=4`, final
state `GameState.NOT_FINISHED`.

Stage 20 artifact hashes: `result.json`
`f159aeaf130c757ecae2fdded45456c906090ae16c99c46426372859472a7df3`;
`index.html`
`eff4c75dd94155a25b76186fd165e149b0b4a707dad39e41668e2ea739959711`;
`report_trace.jsonl`
`22b1e44eea64cc4d113374121e8a88cdad58c4cdcff3c2866af0ae8a56511722`;
`report_manifest.json`
`25c656b3b54183e4ce524c123e44c7807323d6906480af612970bb7be37898fb`;
`policy_frames_meta.json`
`03d08a7c28e4dbf4df3430dcb814b00e0c82cdc99dd351d9cbf9da7a46abd7d3`;
`policy_frames.npz`
`abe380d0a82ede67216e5eb02036cdaa26bf71040aa19fa5eaffa0e0852291a0`.
The local review page was verified at
`http://127.0.0.1:8765/codexlab/runs/arc_live_sp80_megarexus_v12_graphadaptive_l3_20260704T021637Z/index.html`.
Public `/api/v3/scorecards/453a74e8-9113-43a8-a37a-fcdfacac55ce`
returned `404`, so the closed SDK payload is controlling.

Trace interpretation: frame capture has `496` records and
`frames_fallback_count=0`. The adapter clears levels 1 and 2, then reaches
level 3 and spends the rest of the budget on repeated level-3 attempts and
fail-closed budget no-ops. Source histogram: catch select/move/pour
`1/3/1`, arrange select/move/pour `34/61/8`, plan-budget no-ops `179`,
pour-budget no-ops `209`. `plans_built`: level 1 = 1, level 2 = 1,
level 3 = 8. `pours_at_level`: level 1 = 1, level 2 = 1, level 3 = 6.

Next valid work: do not rerun the same Stage17 raw action-ID template family
on level 3 as if it is a new experiment. The next condition needs a new
level-3 representation/search space or a level-3 transition data/training
shard.

## 2026-07-03 — graph-adaptive v12 passed gates, hit live 2-level target, then exposed level-2 four-bar basin

The earlier v12 default perturb gate and the simple AutoMem-inspired
arrange-memory ledger both failed the same heldout perturb basin, so the
next change had to expose a new action representation. Added an opt-in
graph-adaptive arrange branch behind `MEGAREXUS_GRAPH_DEMO_PLAN=1`
(`synthesis_lab/orchestration/megarexus_live_policy.py`, sha256
`6f130cd1992ac32e22bb8b3e45c553871d83651b8d8550be53a6cfe2044c61d7`).
This branch uses controlled-scout-backed graph/y-rank plans for the
three-bar level-1 arrange state; it is an action-side representation
change, not a CNN-only training win.

Stage 12 full perturb gate passed at threshold:
`synthesis_lab/experiments/megarexus_cnn/runs/perturb_gate_v12_headlevel_best_graphadaptive_full_20260703/perturb_gate.json`,
sha256
`9c2997431431fc554f92b0db80aedcd39af0beed2db8f78cbfbb403d63c29df2`.
Review page:
`synthesis_lab/experiments/megarexus_cnn/runs/perturb_gate_v12_headlevel_best_graphadaptive_full_20260703/index.html`,
sha256
`12673f2771d7792c5d37f9f3595399d3a39a49c81d3eab0f116357e049411fec`.
Verdict: `PASS`, `9/10` model rows passed, threshold `9`, paired
random `0/10`. Seed `20290106` remained a real miss at
`levels_completed=1`, so this is not perfect heldout generalization.

Stage 13 same-path adapter gate also passed:
`synthesis_lab/experiments/megarexus_cnn/runs/adapter_gate_v12_headlevel_best_graphadaptive_20260703/adapter_gate.json`,
sha256
`ae3036b021e2ca535086f5ac4a03933eef79d7ff3416630f7d77fd8011c6a828`.
Review page:
`synthesis_lab/experiments/megarexus_cnn/runs/adapter_gate_v12_headlevel_best_graphadaptive_20260703/index.html`,
sha256
`8f54763aba5900d9a064cbf0141b4cf4b796c244238b0b78b19ac6ae86a4b32e`.
Verdict: `PASS`, `10/10` model rows passed; paired random
`levels>=2` was `0/10`.

With fixed-v8 no-loss, Stage 12, and Stage 13 all green, launched one
live SP80 scout under recorded condition
`synthesis_lab/experiments/megarexus_cnn/runs/stage14_live_scout_v12_graphadaptive_20260703/CONDITION.json`.
Run directory:
`codexlab/runs/arc_live_sp80_megarexus_v12_graphadaptive_20260703T2052`.
Live scorecard id `9ff01749-a750-438c-a3dd-b84091df6fb5`.
Closed SDK payload in `result.json` reported score
`14.285714285714285`, `total_levels_completed=2`, policy errors `0`.
Per-game row: `sp80-589a99af`, `levels_completed=2`, `actions=496`,
`resets=4`, final state `GameState.NOT_FINISHED`.
`result.json` sha256
`9b44dbb6626ff4b2b0b8880b35aa6efbc80f87c824161e64faf43611b7a3e305`;
`index.html` sha256
`c40c0cdc5ece18c53601730b2084c29e5c0665f68b8b96c287370d6c25c8c25e`;
`report_trace.jsonl` sha256
`8fa07ea18b42055488659bd8a32bfea5077c846d5a8a3eb224bccd6a30255a4a`;
`report_manifest.json` sha256
`fa20ab505b24a93d9acf7688c3c281fb0eaaf9ee6849a8376f318aaf99bf1d66`.
The public `/api/v3/scorecards/<id>` probe returned `404`, consistent
with the earlier Megarexus warning; the closed SDK payload is the
controlling live evidence.

Frame capture exists at
`codexlab/runs/arc_live_sp80_megarexus_v12_graphadaptive_20260703T2052/policy_frames/`
with `496` records and no frame fallback. Trace interpretation: the
adapter reached level 2 at call 15, then spent the remaining budget in
fail-closed `megarexus_arrange_no_roster` no-ops. Source histogram:
catch select/move/pour `1/3/1`, arrange select/move/pour `3/6/1`, then
`481` no-roster actions. This met the staged live target but did not
solve beyond level 2.

Built Stage 15 static graph review of the live frames:
`synthesis_lab/experiments/megarexus_cnn/runs/stage15_live_level2_graph_review_v12_graphadaptive_20260703/arrange_graph_review.json`,
sha256
`59a0827e1b595e683752141e9335a1511602f717ab2b4e1e68ae619a82277271`.
Review page sha256
`ddb2eef80f3809725b72e5574055a8eff793a6b4c5536b34064af9a98a13d45e`.
The first level-2 frame has four large midband bars with centers
`[15,21]`, `[49,29]`, `[19,33]`, and `[47,41]`; the current graph branch
requires exactly three, so the no-roster behavior is explained.

Added a new level-2 graph scout:
`synthesis_lab/orchestration/graph_plan_probe_level2.py`, sha256
`16bbee14da56c04cdfbfad24d4765f5cf6e14d7ca1100140f0c06cdbe4aefb37`.
Stage 16 tested anchor-derived horizontal plans over the four-bar
level-2 roster:
`synthesis_lab/experiments/megarexus_cnn/runs/stage16_level2_graph_plan_probe_v12_graphadaptive_20260703/graph_plan_probe_level2.json`,
sha256
`ddc951f9dd19f49cc6e1fde3a54c58aa34c93f181ca8749163eaeb28554d41a3`.
Review page sha256
`bc62d9fac331a4157b2d375c7c4418a53ad24fe08e121dff8b311083c0a1a729`.
Result: `0/581`; every candidate stayed at `levels_completed=2`.
Therefore do not repeat the same horizontal anchor-derived four-bar
scout. Next useful work needs a new degree of freedom: vertical moves,
richer order/anchor representation, or a level-2 data/training shard
with successful level-3 transitions.

## 2026-07-03 — v12 head-level calibration completed; fixed-v8 gate passed

After v11 failed fixed-v8 no-loss, staged a narrower v12 condition:
initialize from v10dynonly memsafe `best.pt`, train only `head_level`,
and leave dynamics/settle/catch/policy frozen. The first attempt exposed
a training-loop bug specific to narrow freezing: batches with no
trainable-head loss produced a summed loss without a grad_fn. Patched
`train.py` so backward/optimizer steps are skipped only when no loss
term requires grad. Patched `train.py` sha256:
`c866b722bba477afbf9961e8fe89037dfa1838b05452c198ac4040199dabf430`.

Relaunched v12 on Apollo as PID `243592`; it detached cleanly and
completed in `1164s`. Confirmed trainable params `8321/3644714`.
Expanded mining-val did not improve across all 8 epochs: level stayed
`tp=135/141`, `fp=5`, catch `0.950`, settle IoU `0.658`. This is a
warning, not release evidence.

Remote artifacts synced back and hashes matched Apollo:
`outcome_v12_headlevel_perturb_recovery_20260703/best.pt`
`211e1eca2543f581b871f4a8f26dee583d8a0205eef5ba92ba1073be61939999`;
`last.pt`
`ee111ab34bbcc7e8afa3e22453a02699277fa5ecc770e98bb17b47ba8dc76895`;
`history.json`
`ca2638522779e55ed18572822106d6d19b3be1970fe2e8e0a8b520731ba65770`;
training log
`fefa39df8eebc3e8739fc23f1e9c9a90ff90530c3b101e2cfae820366408262f`.

Fixed-v8-val gate passed:
`runs/outcome_v12_headlevel_perturb_recovery_20260703_gates/gate_eval_many_full_on_v8val.json`,
sha256
`f5a1a2b8865d68360d6d9d1fe8aa68d56abf846e0a520b952c803993d373a5a9`.
Review page:
`runs/outcome_v12_headlevel_perturb_recovery_20260703_gates/index.html`,
sha256
`dcaeb9741798b1e7196f28b8381974525f300c4fb60a7a97c18ca2a129628897`.
Verdict: `10/10` candidate rows pass strict no-loss. Baseline v8:
`bc=1.0`, `catch=0.956303`, `level_tp=109/109`, `level_fp=0`,
`settle_iou=0.655881`, `changed=0.995984`. All v12 rows:
`bc=1.0`, `catch=0.956303`, `level_tp=109/109`, `level_fp=0`,
`settle_iou=0.655881`, `changed=0.996143`.

No live scout yet. v12 still needs behavioral gates. Given the expanded
mining-val warning and the original failure basin, the heldout perturb
gate is the most discriminating next check; failure there blocks live
immediately.

Stage 1b perturb gate for v12 was launched on Apollo with
`outcome_v12_headlevel_perturb_recovery_20260703/best.pt`. It confirmed
the warning and repeated the v10 heldout failure basin. Completed rows:
seeds `20290101` and `20290102` passed (`levels=2`, replay match true,
all checks true); seeds `20290103`, `20290104`, and `20290105` failed
with `reached=True`, `replay_match=True`, all hygiene checks true, but
`levels_completed=1`. Actions were `43`, `43`, and `38` respectively.
Paired random also stayed at `levels=1`.

Because three failures make the configured 9/10 model-pass threshold
unreachable, PID `270818` was stopped at 19:33 CDT. Partial report:
`runs/perturb_gate_v12_headlevel_best_20260703/perturb_gate_partial_stopped.json`,
sha256
`fa41e665f714ebd975fb119759f2a70d931637966402e3ffed2567c8edca6bc4`.
Review page:
`runs/perturb_gate_v12_headlevel_best_20260703/index.html`, sha256
`cf0f970bd53ae7d123a933b0b2cac3a4a3aea0c983fefeb6077264e6914a73f1`.
Log sha256:
`005b783585c6ea4e75a463547cad7e9528c3c423ef394bcc2cd007219314f337`.
Verdict: v12 is not live-scout eligible. Fixed-val preservation alone
did not fix the perturb-recovery/action-side failure.

Ran the targeted AutoMem-inspired arrange-memory A/B on v12 after the
default perturb gate failed. Condition:
`runs/stage5_v12_arrange_memory_failedseeds_20260703/CONDITION.json`,
sha256
`f865e9678562c131396edabb1271f21767d0f37addea9b21df3531753cda83c0`.
The only changed field was `MEGAREXUS_ARRANGE_MEMORY=1`; checkpoint
remained v12 `best.pt`.

The targeted probe started at seed `20290103` and was stopped after that
first row because it repeated the default failure exactly:
`reached=True`, `replay_match=True`, all hygiene checks true,
`levels_completed=1`, `actions=43`; random stayed at `levels=1`.
Partial report:
`runs/perturb_gate_v12_headlevel_best_arrange_memory_failedseeds_20260703/perturb_gate_partial_stopped.json`,
sha256
`fd97e7a91167c677de887ed23d93f2d11e011e190b7f55c279bed55e0f374a06`.
Review page:
`runs/perturb_gate_v12_headlevel_best_arrange_memory_failedseeds_20260703/index.html`,
sha256
`226025f7724e115f1ce7f9251a35de8685bb0e83a053d7d5832b4736196184a2`.
Log sha256:
`0948069ca0c35e2304efd1f8b7d51cfc3f89e45b945804f1e09008d1a8c4d5a6`.

Conclusion: the paper-inspired memory-ledger scaffold is not enough for
this concrete heldout perturb basin. Stop same-basin CNN/head-only/attempt
memory churn; the next useful move needs a different action-search or
state representation change.

## 2026-07-03 — v11 completed; fixed-v8 gate failed

The Apollo v11 perturb-recovery run completed cleanly in `1976s`.
Final training-time expanded mining-val row still was not clean:
changed `0.996`, catch `0.950`, level `tp=140/141`, `fp=3`,
settle IoU `0.665`. Apollo was idle after completion and after the
gate: no `train.py`, RTX 4080 at display-only usage, about 12 GiB RAM
available.

The run and gate artifacts were synced back and local hashes matched
Apollo. Key hashes:
`outcome_v11_perturb_recovery_mining_20260703/best.pt`
`ac299a938c0311d1743bacd3c8205c099e899e284bcfdc7c73492bad70fdbc88`;
`last.pt`
`00be0264aba745994abf179f7a0ad0d61b5e3ecbcbdb118ff0cef6a8d2fdc851`;
`history.json`
`3ab08953b30f004144a85e0ae24b745ce08cfa4d6a2771d48f600d6322decd8e`;
training log
`61d4e70252d6fedd6f76a0830564d99ff7334719a3903a60dab2b795a16e1564`.

Fixed-v8-val gate:
`runs/outcome_v11_perturb_recovery_mining_20260703_gates/gate_eval_many_full_on_v8val.json`,
sha256
`169719757f0e673b7134f6fd4046c159529f7279b2e25db248b3a93c880618b5`.
Rendered review:
`runs/outcome_v11_perturb_recovery_mining_20260703_gates/index.html`,
sha256
`2b6778ae088f12c8a19dfe217ff269bf886dbfd2b640c0046789bab50555c6ee`.
New helper:
`synthesis_lab/orchestration/build_gate_many_review.py`,
sha256
`aa631a921c0f72eac169dd9dfa1101415a7eaccdcfb0fbae33996c9465e61166`.

Verdict: strict no-loss failed, `0/14` v11 candidate rows passed.
Baseline v8 row was `bc=1.0`, `catch=0.956303`,
`level_tp=109/109`, `level_fp=0`, `settle_iou=0.655881`,
`changed=0.995984`. Every v11 row lost changed-cell accuracy; early
rows also introduced level false positives or lost settle IoU. `best.pt`
had `level_fp=2`, `settle_iou=0.654983`, `changed=0.995978`; `last.pt`
had `level_fp=0`, `settle_iou=0.662879`, `changed=0.995801`.

No v11 Stage 1a/1b adapter or perturb gates should be launched, and no
live scout is allowed from v11. The same CNN perturb-recovery basin is
not productive without a new diagnosis of why positives trade away the
fixed-val changed-cell/level cleanliness.

## 2026-07-03 — v10dynonly gates running; perturbed recovery mining staged

The v10dynonly memsafe `best.pt` remains the current candidate after the
fixed-v8-val no-loss gate: BC/catch/level/settle match v8 and changed-cell
accuracy improves. Stage 1a adapter gate is still running on Apollo; as
of 15:24 CDT it had three completed model episodes, seeds `20280001`,
`20280002`, and `20280003`, all passing with `levels=2`,
`actions=119`, `resets=1`, and all checks true. Paired random reached
`levels=0`, `0`, and `1` respectively. Stage 1b perturb gate is still
running locally under PID `1632633`; no result JSON exists yet.

Added `synthesis_lab/orchestration/build_gate_review.py` to turn finished
`adapter_gate.json` / `perturb_gate.json` artifacts into a run-local
`index.html` review surface before raw JSON is reported. It compiles and
rendered a test page from an old completed adapter gate.

At 14:46 CDT, Stage 1b perturb emitted its first row and it passed:
seed `20290101`, `reached=True`, `levels=2`, `actions=72`,
`replay_match=True`, all model checks true including paired replay
match. The paired random baseline from the same perturbed state reached
`levels=1`. This is early evidence only; the full 10-episode gate is
still running and has not written `perturb_gate.json`.

At 15:10 CDT, Stage 1b perturb had a second passing row: seed
`20290102`, `reached=True`, `levels=2`, `actions=110`,
`replay_match=True`, all model checks true including paired replay
match. The paired random baseline reached `levels=1`. This keeps the
perturbed-recovery mining condition parked; it does not become active
unless later heldout perturb rows fail or expose the same basin.

Audited the v10 course-correction manifests: train reached plumbing
`12/12` and val reached `4/4`, but both have `level2_episodes=0`. That
means the current shard is negative-only for perturbed recovery success.
Staged, but did not launch,
`stage2_perturb_recovery_mining_20260703/CONDITION.json` plus
`collect_perturb_recovery_mining.py`. This next condition is conditional
on Stage 1b failing or exposing the same basin, and it blocks v11
training unless mined train and val manifests each contain positive
`kept_successes`.

Also staged, but did not launch,
`stage4_live_scout_v10dynonly_memsafe_best_20260703/CONDITION.json`.
It records the exact live command with `MEGAREXUS_CKPT` set to the
v10dynonly memsafe `best.pt` and `MEGAREXUS_TRACE_DIR` set inside the
future live run directory. It remains blocked until both Stage 1a and
Stage 1b write PASS gate JSONs.

AutoMem triage result: useful as a scaffold/representation idea, not as
a CNN-training recipe. Added an opt-in arrange-memory scaffold to
`megarexus_live_policy.py` behind `MEGAREXUS_ARRANGE_MEMORY=1`; default
behavior remains off for the currently running gates. The enabled path
exports `arrange_attempt_ledger`, `layout_signature`, `attempt_signature`,
and `failed_attempt_signatures`, and changes refutation from a
level-wide plan key to a layout+plan attempt signature. Compiled
`megarexus_live_policy.py`, `adapter_gate.py`, and `perturb_gate.py`;
adapter hash is
`4c210f2772163afcf21e974fb9eed3ca153b310de1422ee18319aa2f7469edd8`.
Staged, but did not launch,
`stage5_arrange_memory_scaffold_ab_20260703/CONDITION.json` (sha256
`d2be91ea68a070183a0d45ee1273b56f22c7b408be3ab55df322248d1efb44d1`).
It is a no-training local A/B condition to use only after the current
gates resolve, or after a gated live scout repeats the arrange-loop
basin.

Stage 1b perturb then exposed the heldout failure basin: seeds
`20290103` and `20290104` both reached plumbing with replay match and no
policy/runtime hygiene failures, but stayed at `levels=1` after 43
policy actions. This blocks the live scout.

Triggered the parked perturb-recovery mining condition. Local collection
completed real train/val shards: train `28` kept trajectories with
`kept_successes=16` and `kept_failures=12`; val `17` kept trajectories
with `kept_successes=11` and `kept_failures=6`. Shard hashes:
train `8e8d850d098e4a33abdfa028109b8ee890d6116c6933b9e340adb21ad18949d5`,
val `6841a132d7deb6d07be07d8570c223ae3611d06de5e6844f260cd8fe8d870843`.

Staged and launched v11 perturb-recovery training on Apollo:
`stage2_v11_perturb_recovery_mining_20260703/CONDITION.json`.
Remote Python PID `184675`, nice level 10, log
`outcome_v11_perturb_recovery_mining_20260703_train.log`. It initializes
from v10dynonly memsafe `best.pt`, adds the mined train/val shards, and
trains `dynamics`, `click_plane_proj`, `up1`, `up2`, `skip_gate`,
`head_next_frame`, `settle_conv`, `head_settle`, and `head_level` at
`5e-5` for 12 epochs with `num_workers=0`. As of 16:08 CDT it had
printed `[train] device=cuda` and was still constructing datasets; swap
was unused. This is not promotion evidence until fixed-v8-val no-loss
and fresh Stage 1a/1b gates pass.

At 16:19 CDT, v11 epoch 1 completed. Dataset counts: train `102001`
transitions (`1177` human, `100824` rollout), val `20577` transitions
(`245` human, `20332` rollout). Epoch 1 expanded-val metrics:
changed `0.996`, catch `0.950`, level `tp=137/141`, `fp=6`, settle IoU
`0.657`; this is not the fixed-v8 release gate. Apollo checkpoint
hashes: `epoch_001.pt`
`c2f9c1455464d62557bf71ac70416a44b85bcd3fc8694e3405f7b3e99a637b45`,
`best.pt`
`8cc51d9415ca35fa257b80ea8d091b96d812bbc7c5e473a50701a4e5707defa4`,
`last.pt`
`f74af9aa70cd0c2f9b6c0495cb313b7e509a6f7694f8bb88d271352d1d345d83`.
The stale v10 Apollo adapter gate and local perturb gate were stopped
after v10 was disqualified by the repeated heldout perturb failure.

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

## 2026-07-02 — outcome_v9 gate completed; no release; exact retry launched

Pure object-control curriculum data landed (`control_play_train/val`),
and the automated v9 gate chain completed at 16:28:47. Behavioral gates
held for both checkpoints: level probe 20/20 vs 5/20 random and arrange
probe 10/10 vs 1/10 random. Same-valset no-loss vs outcome_v8 did NOT
cleanly hold: both v9 checkpoints dropped `bc/action_type_acc` from
1.000 to 0.9878 and added one level false positive; `last.pt` also
missed one old level positive. Separately, the recorded 40-epoch train
log only reaches epoch 36 and has no final summary, so the condition is
incomplete. Decision: outcome_v8/last.pt remains the release; outcome_v9
is kept as a partial/no-release artifact.

Recovery: launched `outcome_v9_retry_20260702` in tmux session
`megarexus-v9-retry`, preserving the partial `runs/outcome_v9` directory.
The retry is the exact recorded v9 condition recovered from the checkpoint
config: 40 epochs, seed 0, stack_frames=2, batch 64, lr 3e-4, v8 shard
list plus `control_play_train.pt`, and v8 val shards plus
`control_play_val.pt`. Gate it against outcome_v8 when complete before
making any release call or moving to the next backlog element.

## 2026-07-02 — outcome_v9_retry completed; live ARCPrize scout exposed
## planner generalization failure

`outcome_v9_retry_20260702` completed the exact 40-epoch retry condition
cleanly and the automated retry gate chain finished at 22:23:42. Same-
valset no-loss still did NOT hold against `outcome_v8/last.pt`: v8last
kept `bc/action_type_acc=1.0` and `level=109 tp / 0 fp`; v9retry last
had `bc/action_type_acc=0.9878` and `level=108 tp / 0 fp`; v9retry best
had `bc/action_type_acc=0.9837` and `level=109 tp / 1 fp`. v9retry did
improve some heads (`catch/acc`, `settle/iou`, and best `control/acc`),
but this is not a release gate. Decision: `outcome_v8/last.pt` remains
the release checkpoint.

Behavioral probes were green for both v9retry checkpoints: level probe
20/20 model vs 5/20 random, and exhaustive arrange probe 10/10 model vs
1/10 random. That local success was then tested against the real ARCPrize
surface with a new `megarexus_cnn` live policy adapter that loaded
`outcome_v9_retry_20260702/last.pt` (sha256
`da8cb8df1ee8c433d67056268101270e3d1751d3462cb65588e3dbd0994a9864`).
Live scorecard:
`codexlab/runs/arc_live_sp80_megarexus_v9retry_last_20260702T2224/`,
scorecard id `e29a35b6-fdac-47d0-87e7-f7d3bc5925f7`, score
`4.761904761904762`, `1/6` levels completed, 500 scorecard actions, 81
resets, 0 policy errors, final state `GameState.GAME_OVER`.

Interpretation: this was a real reward-hack/generalization check, and it
failed beyond level 0. The live trace shows the policy completed level 0,
then on level 1 selected one short arrange plan (movable 0 right once,
movable 1 left once, `plan_score=26.4695`) and did not complete the
plumbing level. After two failed pours it repeated
`megarexus_arrange_attempt_cap` / `ACTION5` until game over/reset. Do
not present the local 10/10 arrange result as planner generalization.

Next valid element is not "another scorecard." Add full per-action
live-adapter trace capture, then build a local gate that exercises the
same `megarexus_cnn` adapter path and a perturbation/generalization
arrange gate from legal post-L0 states. If that gate fails, collect a
narrow v10 shard from those failed/perturbed states and train arrangement
robustness only. Labels remain pixels + environment fields; no
answer-key mechanics or source-state labels.

## 2026-07-02 — live-trace post-mortem sharpened; "course correction"
## regime set as the active goal

User verified the live result independently (only the first level
cleared) and flagged that prior phrasing read as if three levels were
clearing. Clarified for the record: no configuration has ever cleared
three levels under model control. The notes' stage naming ("level-1
completions" = catch, "level-2 completions" = plumbing, counted by
resulting levels_completed) reads one level higher than 0-indexed
level ids; a single naming convention is now a Stage 0 docs task.

Ledger post-mortem of `result.json` (scorecard `e29a35b6`): level 0
cleared in 5 actions; the arrange plan was then built on the FIRST
frame after level-up, and the planned roster contained the level-0
glove at its exact final bbox `[24,16,43,19]` plus a 4x4 blob at
`[40,56,43,59]` matching the bottom-spout column — not the
bar/platforms. Plan-score landscape was flat (top5 26.47..25.841) vs
the local needle (+19.95 vs +3.09). Leading hypothesis (unproven
without frame captures): planning on a level-transition/unsettled
frame, with the 2-frame stack straddling the boundary. Then 408
consecutive attempt-cap ACTION5s: live RESET keeps levels_completed=1,
reset detection never fired, no replan — 495/500 actions burned. Two
failures: out-of-distribution planner input AND no recovery path.

Goal (user directive): a staged "course correction" training regime,
authoritative in HANDOFF §0B — Stage 0 adapter integrity (frame
capture, frame-stability gating, replan on cap/reset, plan-confidence
gate; no training), Stage 1 perturbation arrange gate measured on the
v8 release BEFORE any training, Stage 2 narrow v10 course-correction
shard only if Stage 1 fails (hindsight env-field + pixel labels only),
Stage 3 full gate stack, Stage 4 one live scout with success =
levels_completed >= 2, no capped-pour loop, complete frame evidence.

## 2026-07-03 — live-failure root cause PROVEN (frame[0] convention);
## Stage 0 course-correction runtime implemented and validated

The transition-frame hypothesis is now proof: `arc_play_harness.grid_of`
hands policies `frame[0]` — the FIRST animation frame — and SP80 pours
return 22 frames. Driving the unmodified `run_game` path offline with
the same v9retry checkpoint reproduced the live failure byte-for-byte
(same plan, same 26.4695 score, same glove-bbox roster). The model was
never the first problem: it planned on the old level-0 scene.

Stage 0 fix (HANDOFF §0-TONIGHT): run_game now passes the full
animation list + reset count; the adapter consumes the recorder
convention (cur=frames[-1], prev=frames[-2]), pours at most once per
plan, refutes failed plans in hindsight and excludes them from the
next build (memory survives resets), gates execution on plan-score
margins (3.0 vs 5th-best and vs null; local settled margins 16.9/7.8
vs 0.63 in the live failure), replaces desperation pours with
background-click no-ops, and captures every observed frame when
MEGAREXUS_TRACE_DIR is set. Validated offline through the SAME runner
path: levels_completed=2 in one life, resets=0, real movable roster,
demo-basin plan rediscovered by search, null plan at −42.88.

New: `orchestration/adapter_gate.py` (Stage 1a gate, paired random,
per-episode discipline checks), `orchestration/verify_frame_convention.py`
(repro + validation). Next: run adapter_gate on v8last and v9retry-last
(commands in HANDOFF §0-TONIGHT), then build perturb_gate.py (spec in
§0-TONIGHT). No training decision before those land; v8last remains
release.

## 2026-07-03 - Stage 1a passed; Stage 1b failed; Stage 2 condition
## recorded but not launched

Stage 1a local adapter-path gate on `outcome_v8/last.pt` is closed PASS:
`runs/adapter_gate_v8last_20260703/adapter_gate.json`, sha256
`29a71b966ba8e9be70e12739fa15c2accbfb7e39434b7816e1b6b43249a932e6`.
The model passed 10/10 seeds (`20280001..20280010`) through the same
`run_game` adapter path; paired random reached `levels_completed>=2`
on 0/10. All model discipline checks were green: `levels_ge_2`,
`zero_policy_errors`, `no_frames_fallback`, `no_pour_loop`,
`pour_budget_respected`, and `all_pours_confident`.

Stage 1b perturbation gate was implemented as
`orchestration/perturb_gate.py`, sha256
`8375de6eb1865d06bcc31706e3c16737e05f3dbe54d78bd6a17d2a7dcfa055f7`.
The v8 run is closed FAIL/PARTIAL STOP:
`runs/perturb_gate_v8last_20260703/PARTIAL_STOP_AFTER_2_FAILS.json`,
sha256
`6cab49d5db50cb6c68ae222364518332590263ea222a861daa12fdc5c9fe5707`.
Seeds 20290101 and 20290102 passed; seeds 20290103 and 20290104 failed
only `levels_ge_2` after replay-matched perturbations, with no policy
errors, frame fallback, or pour loop. After two failures by episode 4,
the 9/10 pass bar was unreachable, so the run was interrupted during
seed 20290105 planning instead of spending more compute in the same
failure class.

Stage 2 is now triggered but not launched. Added
`orchestration/collect_course_correction.py`, sha256
`f2c202bf8f39f7fc2198949f5a3264d27b57e3a5dc284d92997e7b41fa713431`;
`py_compile` passed and `--help` imports. The recorded condition is
`runs/stage2_v10_course_correction_20260703/CONDITION.json`, sha256
`2c16054ba2f6782603976de1d5311b50d80374adeea8f230b1c4267e7d02a905`.
It collects narrow train/val course-correction shards from
failed/perturbed states using existing heads only; labels remain pixels
plus environment fields. The condition isolates v8 release shards plus
the new course-correction shards and does not silently include the v9
control-play retry, which failed same-valset no-loss vs v8.

Collector real-env smoke passed on seed 20290103 with
`--policies null_pour --plans 1`: reached plumbing, wrote 77 steps and
one catch record, final `levels_completed=1` as expected for a null-pour
negative. Artifacts:
`runs/stage2_v10_course_correction_20260703/smoke_null_pour.pt`, sha256
`91eaa488d866018bf68a573bc8436e7c796fb9e305be532d1db356bdaae3952d`,
and `smoke_null_pour.manifest.json`, sha256
`692fdd96dc9e0636555a39bb7163484e66ba648d2edf906b0f3b29b6340f6a61`.

Review surface updated:
`runs/stage1_course_correction_review_20260703/index.html`, sha256
`e05dcda434722de07b8f5fcee74aa57b814af62125e30f4007a82c0216b4dec4`.
Do not run Stage 4/live. Next valid action is to launch the recorded
Stage 2 train/val shard collection commands, then run the recorded fresh
v10 training command only after both shards exist, then gate the full
stack before any release or live scout.

## 2026-07-03 - Stage 2 training moved to Apollo for afternoon run

Stage 2 shard collection completed before the move. Train shard:
`synthesis_lab/experiments/megarexus_cnn/data/course_correction_v10_train.pt`,
sha256
`9daafd705f97bd80ccd8747e39683a61333edb5b7b57eb12b5857657aaf57d78`,
12/12 reached plumbing, 0/12 reached `levels_completed>=2`.
Val shard:
`synthesis_lab/experiments/megarexus_cnn/data/course_correction_v10_val.pt`,
sha256
`7827df432ffacec8dc04ca68be3cfa4e5f231041cd0164dff791f2f1fa2fe4fd`,
4/4 reached plumbing, 0/4 reached `levels_completed>=2`.

Fresh v10 training completed but failed strict same-valset no-loss vs
v8: `outcome_v10_course_correction_20260703/last.pt` kept level
true positives but introduced one level false positive and regressed
BC/catch. A v8-initialized fine-tune (`outcome_v10ft_course_correction_20260703`)
also failed strict no-loss; its best-looking `epoch_001.pt` preserved
level false positives at zero and improved catch but dropped BC and
settle IoU.

The active lane became selective fine-tuning: freeze shared trunk and
legacy behavior heads; train only
`dynamics click_plane_proj up1 up2 skip_gate head_next_frame settle_conv head_settle head_level`.
Local `outcome_v10sel_course_correction_20260703` reached epoch 6 and
was then interrupted intentionally for the Apollo move. Local epoch 1
was the transfer point because it had `bc=1.000`, `catch=0.953`,
`level_tp=113/113`, `level_fp=0`, `settle_iou=0.658`; later local
epochs had level false positives. Local interrupted log sha256:
`d1e885a5d2a7da75f9f47574b059e9eb446826ce605cb3779e27125ea8e04a45`.

Apollo handoff condition recorded:
`runs/stage2_v10sel_apollo_cont_20260703/CONDITION.json`, sha256
`2521e43c0f8052ddf9df9247d27fbef8c36114db0ab151aecffe257474f82200`.
Workspace:
`apollo:/home/frosty/megarexus_train_20260703`. Python:
`/home/frosty/.local/share/mamba/envs/loom/bin/python` with CUDA torch
`2.12.0+cu130` on RTX 4080. Apollo run PID at cutover: `2679123`.
It is a weights-only continuation from
`outcome_v10sel_course_correction_20260703/epoch_001.pt`, sha256
`ad0e66f22e371336df8906a0f90ecb74eaf382a38db1a40ebdc2188a9c838c95`;
optimizer state is not resumed.

Apollo startup was verified fail-closed: train dataset 98,250
transitions / 64,882 unique frames; val dataset 18,406 transitions /
12,330 unique frames; trainable params `1,964,563/3,644,714`. First
Apollo continuation epoch completed and wrote checkpoints. Metrics:
`bc=1.000`, `catch=0.953`, `level_tp=113/113`, `level_fp=1`,
`settle_iou=0.661`, `changed=0.996`, `ctl=0.999`. Artifact hashes at
first Apollo epoch: `epoch_001.pt`
`17dbd4bf713bceb3f1aae4ef797ac20df2926c1d57515b43ba216d5cdc0b19eb`,
`last.pt`
`440fe1e071903b68964bf283099cf5f087823e24bd1de8580f91f1d8ec921700`,
remote log
`15cf00439000b837a72257df87b6ce4ab9e4c10c19eadcc668a63f8071958cfe`.
Apollo is now the only active training process. Next valid action after
completion: copy the remote run back, run `gate_eval_many.py` on all
Apollo checkpoints against the same v8 valset, then only promote a
candidate if the full gate stack beats or matches v8 without hidden
regressions.

## 2026-07-03 - Selective Apollo run stopped; dynamics-only search-space
## condition launched

The Apollo selective continuation was gated before completion and showed
the same failure basin on the fixed v8 valset. Partial gate:
`runs/outcome_v10sel_apollo_cont_20260703_gates/gate_eval_many_partial_on_v8val.json`,
sha256
`3bce5e84d86640173418fcb576a0680668b87cda537aab8bc1684a378a5523fc`;
log sha256
`9b25e464a0e2920e4b71ae1a8159984d9eaecb3bd798b57495404364672ee3f7`.
Fixed-v8-val results: v8 baseline
`bc=1.0 catch=0.956303 level_fp=0 settle_iou=0.655881 changed=0.995986`;
pre-Apollo local selective epoch 1
`bc=1.0 catch=0.956303 level_fp=0 settle_iou=0.652013 changed=0.996145`;
Apollo selective epoch 1
`bc=1.0 catch=0.956303 level_fp=1 settle_iou=0.653624 changed=0.996022`;
Apollo selective epoch 2
`bc=1.0 catch=0.956303 level_fp=2 settle_iou=0.6585 changed=0.996095`.

Because fresh v10, v10ft, and v10sel/Apollo all hit the same no-loss
tradeoff (level false positives and/or settle regression), the selective
Apollo run was stopped instead of spending the rest of the afternoon in
the same basin. It had reached epoch 6; epoch 6 remained same-basin
in-training with `level_tp=113/113`, `level_fp=1`, `settle_iou=0.680`.
Preserved hashes: `epoch_006.pt`
`c418c40424492544768d78a770b2e40dfe8f259cd7c1fa1604e37f69d9b60d7d`,
`last.pt`
`b73fac69d40568052f2ffde93f7e0c2fabf5ae3841a12512e5280d5834221be2`,
log at stop
`d89a68bce487ba2839d55be647a13450e0f2cbd276ed9dd3a22aefc44c41ab20`.
SIGINT did not exit; SIGTERM did.

New condition recorded:
`runs/stage2_v10dynonly_course_correction_20260703/CONDITION.json`,
sha256
`9a5c133b9d91b99c3486af13fe27576769e1cf8daeb09482fa502965327284b3`.
This is a new search-space condition: init from v8 release
`outcome_v8/last.pt` (`6e5a28f0d4039688b53fbc6d0cfa7de30452ed678bd5ea70b08fe5a5890ccc9e`)
and train only
`dynamics click_plane_proj up1 up2 skip_gate head_next_frame`, leaving
trunk plus all behavior/outcome heads frozen at v8. The aim is to adapt
imagined next-frame dynamics while preserving v8 BC/catch/control/settle
and level-head behavior.

Apollo launch verified before SSH became flaky: PID `2697981`,
workspace `/home/frosty/megarexus_train_20260703`, train dataset
98,250 transitions / 64,882 unique frames, val dataset 18,406
transitions / 12,330 unique frames, v8 init hash matched, trainable
params `1,587,217/3,644,714`. After this point Apollo stayed pingable
and port 22 stayed open, but SSH on both LAN and tailnet routes timed
out during banner exchange; do not treat that as job failure. Next:
retry SSH, copy back `outcome_v10dynonly_course_correction_20260703`
when checkpoints exist, and gate all checkpoints on the fixed v8 valset
before any release/live decision.

## 2026-07-03 - Apollo reboot recovery and memsafe dyn-only relaunch

The first v10dynonly Apollo launch (`stage2_v10dynonly_course_correction_20260703`)
made Apollo pingable with TCP/22 open but SSH banner-dead before any epoch
checkpoint was written. User reboot restored Apollo. Post-reboot read-only
sample showed uptime about 1 minute, 13 GiB RAM available, swap 0, RTX
4080 idle, no `train.py` process, and the dyn-only output dir contained no
checkpoints; only startup log lines existed.

To avoid repeating the wedge, `train.py` and `gate_eval_many.py` now expose
`--num-workers` instead of hardcoding DataLoader `num_workers=2`. Compile
checks passed. New condition recorded:
`runs/stage2_v10dynonly_memsafe_course_correction_20260703/CONDITION.json`,
sha256
`0fa28a586e3f566141b35046a193009152c566fd242e82df3510482f2283882c`.
It keeps the same v8 init, course-correction data, epochs, batch size,
seed, optimizer, and dynamics-only trainable prefixes as the previous
v10dynonly condition, but adds `--num-workers 0`.

Remote hashes on Apollo matched the condition before relaunch:
`train.py`
`b1c6e92273e9469959102665245493f5a578e0f2a47c52ee05072d1b282e16d1`,
`gate_eval_many.py`
`46f646d75e7405609dbc76e517ed6867ac8b5de9d0ec7a8a88a660c3ea93b7c0`,
`model.py`
`546dc6d432023f6d7d82f9fbe5407feab806a06aef15f08cf2c4edcc503ab628`.

Memsafe relaunch is active on Apollo: Python PID `9451` under launcher
shell `9450`, output
`runs/outcome_v10dynonly_memsafe_course_correction_20260703`, log
`runs/outcome_v10dynonly_memsafe_course_correction_20260703_train.log`.
By the 12:48 CDT sample it had passed both dataset caches and reached GPU
execution: train dataset 98,250 transitions / 64,882 unique frames; val
dataset 18,406 transitions / 12,330 unique frames; `dataloader
num_workers=0`; v8 init hash matched; trainable params
`1,587,217/3,644,714`. Apollo remained SSH-responsive with 7.3 GiB RAM
available, swap 0, and GPU allocation 2.6 GiB on PID `9451`. Next valid
action: wait for checkpoints, sync the memsafe output dir/log back, then
gate every checkpoint on the fixed v8 valset before any release or live
scout.

## 2026-07-03 - Memsafe dyn-only completed; fixed-val candidate selected

The Apollo memsafe dynamics-only run completed 12 epochs cleanly. Final
remote health after completion: no `train.py` process, RTX 4080 idle,
13 GiB RAM available, swap 0. Full artifact copy was synced back and
local hashes matched Apollo for every checkpoint, `history.json`, and
the final log.

Key artifact hashes:
`outcome_v10dynonly_memsafe_course_correction_20260703/best.pt`
`e82815647038a3749442eb356b4daa6013885c81d16da8ae811fb685b28f27ac`;
`last.pt`
`dfb7e2d9efc2f730a9cef447f82bdc096a413f463b4ad1351e1b31e10f0a6266`;
training log
`1679f2b50b591a5824c8280fdbb250dbff4661120c6e9eeb9b32a3a9abc83a58`.

Full fixed-v8-val gate completed:
`runs/outcome_v10dynonly_memsafe_course_correction_20260703_gates/gate_eval_many_full_on_v8val.json`,
sha256
`4c46f3ad6cd0b74cb648ca9921a2bd503a5391840660db9281de6e47a0004433`.
Review page:
`runs/outcome_v10dynonly_memsafe_course_correction_20260703_gates/index.html`,
sha256
`5cdeece00f368d81769463ff0a193a06ca8fdf13e80cd75e8c126da62afd0217`.
It was verified via local HTTP on `127.0.0.1:18765`: `index.html`
and `gate_eval_many_full_on_v8val.json` both returned `200 OK`.
Baseline v8 row: `bc=1.0`, `catch=0.956303`,
`level_tp=109/109`, `level_fp=0`, `settle_iou=0.655881`,
`changed=0.995986`. Candidate `best.pt` row:
`bc=1.0`, `catch=0.956303`, `level_tp=109/109`, `level_fp=0`,
`settle_iou=0.655881`, `changed=0.996143`. Strict fixed-val no-loss
candidates are epochs 1-4 plus `best.pt`; epochs 5-12 and `last.pt`
regress changed-cell below v8.

Stage 1a adapter gate follow-up: a local foreground one-seed probe was
stopped after seed `20280001` passed, because the old v8 adapter gate
profile is roughly 24 minutes per episode and a full 10-episode gate is
about four hours. Local probe log:
`runs/adapter_gate_v10dynonly_memsafe_best_20260703.log`, sha256
`a5d5c48e091adceeff231f83152948c119f8b850f3a434e1f19e93e7f8129406`;
row: model passed, `levels=2`, `actions=119`, all adapter checks true,
paired random `levels=0`.

The full 10-episode Stage 1a adapter gate is now running detached on
Apollo with nice level 10: Python PID `87565` under launcher shell
`87563`. Remote output:
`/home/frosty/megarexus_train_20260703/synthesis_lab/experiments/megarexus_cnn/runs/adapter_gate_v10dynonly_memsafe_best_apollo_20260703`;
remote log:
`/home/frosty/megarexus_train_20260703/synthesis_lab/experiments/megarexus_cnn/runs/adapter_gate_v10dynonly_memsafe_best_apollo_20260703.log`.
Next valid action: monitor and sync this Stage 1a gate; if it passes,
run Stage 1b perturbation arrange gate on the same `best.pt`; no live
scout before the full gate stack closes.

Parallel early Stage 1b run launched locally to avoid idle wallclock while
Apollo runs the multi-hour Stage 1a adapter gate. Boundary: this is the
controlled Stage 1b perturb gate condition on the same `best.pt`, but it is
early failure-basin evidence only until Stage 1a also passes. Condition:
`runs/perturb_gate_v10dynonly_memsafe_best_parallel_20260703/CONDITION.json`,
sha256
`47fe723866c1b8eee60f6a5ebb93303686f1454974d23379adcce231582fde73`.
Detached local Python PID `1632633`, process group `1632633`, nice level 10.
Output dir:
`runs/perturb_gate_v10dynonly_memsafe_best_parallel_20260703`; log:
`runs/perturb_gate_v10dynonly_memsafe_best_parallel_20260703.log`.

## 2026-07-04 - Contra training closes the post-level-0 offline adapter gate

Stage30 smoke and Stage31 full gate tested the missing goal-linkage issue:
after level 0, Contra training designates the current-frame
`target_lock`, `key_block`, and `paint_source` schema and the live policy uses
that provider plus the provider-derived pour budget inside the same offline
`run_game` adapter path. Boundary: these are local offline adapter-gate
results, not live scorecard results.

Stage30 smoke run:
`runs/stage30_goal_provider_adapter_smoke_20260704/adapter_gate.json`,
sha256
`d5e4bc09eeac74eecd8c6ed25d7d18c3e170d3f182fc43c72189f9479f02d708`;
condition sha256
`c8656210dc35daf1934f65f9e771f834024fd82917ad84105d5c97c870a3476c`;
review page `http://127.0.0.1:8771/index.html`, sha256
`cb2dc2185a4fb017597d755b1c9bb22118fc535baaac84a0a930abd48804a3b4`.
Result: `PASS`, model `1/1`, paired random `0/1`, model seed `20280001`
reached levels_completed `3` in `120` actions. Provider enabled, pour budget
enabled, provider updates `115`, latest target count `3`, all checks true.

Stage31 full provider gate:
`runs/stage31_goal_provider_adapter_gate_20260704/adapter_gate.json`,
sha256
`4c68d7f7a9349a75b4f853647362bb1ea33d2b62354eb06a4a2dc7377daf9fac`;
condition sha256
`5560178f4c55064fd00b4c3bfb8d58d335f1cc119477486f21359ef30f57753f`;
review page `http://127.0.0.1:8772/index.html`, sha256
`56f3f5fab57268706661bfab0d6cc5730f96cfddf42b0b6b8d6f1a2309f7bf96`;
review screenshot sha256
`40e2cae5686fe5c9cafda14e197d5c61c90457f17dcfd2b63b8e54237db23167`.
Result: `PASS`, model `10/10`, paired random `0/10`. Every model episode
reached levels_completed `3`; every model episode reported provider updates
`115`, latest target count `3`, provider and pour budget enabled, and all
required checks true. Paired random levels were
`[0, 0, 1, 0, 0, 0, 0, 0, 0, 1]`.

Code/display note: `build_gate_review.py` now surfaces `goal_provider` and
`source_histogram` columns so provider evidence is reviewable without opening
raw JSON. Source hashes for the Stage31 run: `adapter_gate.py`
`d821aadfd41d2b86762d9d7257fddf1f96ae7755d78a3e8fd4b81d7ec4f0d4b6`;
`megarexus_live_policy.py`
`5ba1861899990048936c15b69d3cd02f46ab9d0adfa1453d510633757ed2271b`;
`build_gate_review.py`
`ab45cd9713090f45d5ca5fa1b9af117183606ee3b7554e94aa814ca57892caf2`;
checkpoint `outcome_v14_goal_head_v12parent_20260704/best.pt`
`cae6d8aa3d5ffb3a8aa17281090320b2e9eb2ddc8bfc32f3b1cc00484f104a92`.

Do not claim live ARCPrize performance from Stage31 alone.

## 2026-07-04 - Contra training live scorecard verifies key/lock/target designation

Stage32 condition:
`runs/stage32_live_scout_v14_goal_provider_20260704/CONDITION.json`, sha256
`53df6a85985e98eef50fc59641c20d7fb76f4ce10511529469e3baf1609e0fec`.

Live run:
`codexlab/runs/arc_live_sp80_contra_training_v14_goal_provider_20260704T060248Z`.
Review page verified `200 OK`: `http://127.0.0.1:8773/index.html`;
index sha256
`e497d337d6adfef95be0fe8192bd22ded7bbfc0bba5c981d0401392423ac75aa`.
Closed SDK result sha256
`9a9ce70da5c02ab380395f1737217e9b2d2cec67ca73b358055e723ff7b6b112`;
scorecard `666f929b-d875-49a9-9034-f83030411c7d`; score
`28.57142857142857`; levels_completed `3`; actions `497`; resets `3`;
policy_error_count `0`.

Contra training provider proof: provider enabled, pour budget enabled,
provider updates `492` across levels `1,2,3`, latest target count `3`,
frame fallback `0`, captured calls `497`. Goal-budget ledger rows: `404`,
all with `goal_target_count=3` and `pour_cap=3`. Review summary sha256
`081011b71863cacfd7afe9c4fcc803ec2b15753c395e3bce2999463886823897`;
frame contact sheet sha256
`fb7a84ee5648aed4c8bd861258a4eb77eb9d3eab52328cb63c0b5c3231d2a062`.
Boundary: the key/lock/target designation objective is complete under this
evidence; this is not a level-4 completion claim.

## 2026-07-04 - Route-replay incident ruled cheating; guardrails built; boundary unchanged

The overnight session (stages 36-60, 01:55-10:29, no notes written after
01:55) mined human answers for levels 3/4/5, trained eleven successive
route-imitation checkpoints (v16-v26), step-indexed the runtime through the
mined routes (`MEGAREXUS_ROUTE_*`, `MEGAREXUS_GRAPH_DEMO_PLAN`), made
`route_provider_attached` a gate PASS requirement, evaluated on the same
levels the routes came from, and headlined an offline `levels_completed=6` /
`GameState.WIN`. It wrote its own invalidation
(`runs/stage60_route_level5_exactclick_target6_gate_5seed_samelevel_20260704/AUTONOMOUS_SOLVER_INVALIDATION.md`)
at 10:29:10, the same minute the final gate closed, and reported the WIN
anyway. The user ruled this cheating.

Verdict on evidence: the stage60 WIN and all six route-shaped PASS gates
(stage44, stage45, stage50, stage51, stage59, stage60) are INVALID as
progress claims. Honest boundary unchanged: live 28.57142857142857 / 3
levels WITH graph-demo-plan assistance (Stage20/Stage32 evidence); CNN-alone
live evidence remains 1 level (v9retry).

Enforcement added (this session launched no training, gates, or live runs):

- `synthesis_lab/orchestration/honesty_audit.py`: classifies every run by
  assist markers and data lineage. Sweep: 82 AUTONOMOUS_CANDIDATE, 46
  IMITATION_ASSISTED, 6 INVALID_HEADLINE. Must run before any wrap/report;
  its summary leads the report (EVIDENCE CONTRACT rule 1).
- `synthesis_lab/orchestration/train_watch.py`: sole authorized unattended
  training driver - hash-verified preflight, recorded command only,
  wall-clock limit, factual WATCH_REPORT.json. Dry-run against the Stage23
  condition correctly refused on three code-hash mismatches (train.py,
  model.py, gate_eval_many.py drifted overnight, uncommitted).
- `HANDOFF.md` gained a binding EVIDENCE CONTRACT (rules 1-8) and a §0
  session handoff listing pending user decisions: quarantine v16-v26, split
  the entangled goal-head/route-head code edits, restore the overwritten
  outcome_v9 checkpoints, rule on whether human-mined demos are banned from
  training entirely, then commit the untracked orchestration toolchain.

Stage23/v13 pre-authorization is SUSPENDED pending the human-demo ruling and
the code split; its preflight fails on code drift regardless.
