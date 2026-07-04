# Research log

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
