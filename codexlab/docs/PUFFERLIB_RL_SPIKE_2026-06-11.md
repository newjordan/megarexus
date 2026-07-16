# PufferLib RL Spike - 2026-06-11

Branch: `spike/pufferlib-rl-contra`

## Condition

Run label: `scout`

Goal: test whether PufferLib can be used behind the existing SP80 primitive brain boundary without changing CONTRA action authority.

Boundary: this is not a trained policy result, not control truth, not solve evidence, and not evidence that PufferLib can currently train CONTRA on this host.

## What Was Added

- `codexlab/sp80_puffer_policy.py`: optional `puffer_policy` brain adapter. It chooses one primitive only and keeps motor execution in the existing SP80 controller.
- `codexlab/sp80_pufferlib_probe.py`: compatibility probe that writes `result.json`, `index.html`, and a policy JSON under a run directory.
- `codexlab/sp80_player_harness.py`: new optional `--backend puffer_policy` with `--puffer-policy` and `--require-pufferlib`.

## Upstream PufferLib Runtime Result

Source checked:

- URL: `https://github.com/newjordan/PufferLib.git`
- Branch: `4.0`
- Commit: `9836f0d2e78889c1aaf189c04d161b6fc61a9386`

Runtime import result:

- Active repo Python: `pufferlib` is not installed.
- PyPI package tested: `pufferlib==3.0.0` in `/tmp/rexus-pufferlib-venv`.
- PyPI install failed while building the wheel on this `aarch64` host because the build linked `raylib-5.5_linux_amd64/lib/libraylib.a`; linker error: `file in wrong format`.
- Source checkout `4.0` imports top-level `pufferlib` from `PYTHONPATH`, but `pufferlib.pufferl` fails because the native `_C` backend is not built.
- Follow-up patch proved the root build issue is fixable in the 4.0 source checkout by detecting `aarch64`, building raylib 5.5 from source, stripping x86 SIMD flags, and using GCC/`gomp` when clang/`omp5` are absent.

Probe artifact for the 4.0 source failure:

- `codexlab/runs/pufferlib_sp80_probe_4_0_source/result.json`
- `codexlab/runs/pufferlib_sp80_probe_4_0_source/index.html`

Patch artifact:

- `codexlab/docs/pufferlib_4_0_arm64_build.patch`

Patched local checkout result:

- Checkout path: `/tmp/pufferlib-4.0-newjordan`
- Build command: `bash ./build.sh minimal --cpu`
- Built native backend: `pufferlib/_C.cpython-313-aarch64-linux-gnu.so`
- Additional active-env dependency installed for import validation: `rich_argparse`
- Hard import passed: `PYTHONPATH=/tmp/pufferlib-4.0-newjordan python -B -c "import pufferlib; import pufferlib.pufferl"`
- Probe artifact: `codexlab/runs/pufferlib_sp80_probe_4_0_local_arm64_build_v2/result.json`
- Probe review: `codexlab/runs/pufferlib_sp80_probe_4_0_local_arm64_build_v2/index.html`

## Local Adapter Result

The `puffer_policy` adapter path was tested with a manual scout policy, not a trained PufferLib export.

Artifact:

- `codexlab/runs/pufferlib_sp80_probe_harness_adapter_v3/result.json`
- `codexlab/runs/pufferlib_sp80_probe_harness_adapter_v3/index.html`
- `codexlab/runs/pufferlib_sp80_probe_harness_adapter_v3/harness_action_set/packets.jsonl`
- `codexlab/runs/pufferlib_sp80_probe_harness_adapter_v3/harness_action_set/brain_calls.jsonl`

Result:

- Probe status: `passed`
- Harness smoke: `passed`
- Harness failure count: `0`
- Scope: proves the branch can route a Puffer-style exported primitive policy through the existing SP80 action-set harness.

## Paintflow Groundhog Scout

User goal: master paintflow as a mechanic, starting with catching paint and then redirecting paint to any requested bucket.

Condition artifact:

- `codexlab/conditions/puffer_paintflow_groundhog_v0.json`

Runner:

- `codexlab/sp80_puffer_paintflow_run.py`

Important PufferLib 4.0 boundary:

- The patched 4.0 checkout imports `pufferlib.pufferl`.
- The checked-out 4.0 package does not expose the older Python `pufferlib.PufferEnv`, `pufferlib.vector`, or `pufferlib.emulation` APIs used by its example files.
- This scout therefore gates on the native PufferLib runtime import but executes SP80 through the existing Python game/controller API.
- A true PufferLib training run for SP80 requires either a native PufferLib SP80 environment or a restored Python env/vector wrapper compatible with 4.0.

Run artifact:

- `codexlab/runs/puffer_paintflow_groundhog_v0_20260611T_scout/result.json`
- `codexlab/runs/puffer_paintflow_groundhog_v0_20260611T_scout/index.html`
- `codexlab/runs/puffer_paintflow_groundhog_v0_20260611T_scout/frames/`

Run condition:

- Run label: `scout`
- Seed: `20260611`
- Levels: `0, 1, 2, 3, 4, 5`
- Episodes: `24`
- Max actions per episode: `30`
- Policy: `random_valid`
- Metrics: `target_catch_rate`, `any_catch_rate`, `all_bucket_catch_rate`, `overflow_rate`, `level_completion_rate`, `mean_episode_reward`

Result:

- `episode_count`: `24`
- `target_catch_rate`: `0.0`
- `any_catch_rate`: `0.0`
- `all_bucket_catch_rate`: `0.0`
- `overflow_rate`: `0.0`
- `level_completion_rate`: `0.0`
- `mean_episode_reward`: `-0.05458333333333334`

Interpretation:

- This is a useful negative baseline: random valid primitive actions under the bounded budget did not discover catching or redirection.
- This is not a same-basin stale loop yet; it is the first recorded negative scout for this representation.
- Next representation step should add a mechanics-aware action generator or build the native PufferLib SP80 environment, not just run more random scouts.

## Paintflow Groundhog Level-1 Pose Bandit

Condition artifacts:

- `codexlab/conditions/puffer_paintflow_groundhog_l1_pose_bandit_v0.json`
- `codexlab/conditions/puffer_paintflow_groundhog_l1_learned_clean_pose_v0.json`

Runner:

- `codexlab/sp80_puffer_paintflow_run.py`

Representation change:

- Added pre-reset paintflow event hooks around SP80 flow cleanup and level advance so `cevwbinfgl`/overflow state is captured before the game clears it.
- Added `level1_pose_bandit_v0`, a tabular pose-bandit over legal level-1 bar poses. Each episode selects one pose, executes bounded primitive actions, activates flow, and updates pose value from target clean catch, all-bucket clean catch, overflow, and no-catch outcome.
- Added `level1_learned_clean_pose_v0`, a deterministic evaluation policy derived from the learned clean poses: target bucket `0` uses bar pose `x=6,y=6`; target bucket `1` uses bar pose `x=6,y=11`.

Bandit training run:

- Run artifact: `codexlab/runs/puffer_paintflow_groundhog_l1_pose_bandit_v0_20260611T_groundhog_v2/index.html`
- Result JSON: `codexlab/runs/puffer_paintflow_groundhog_l1_pose_bandit_v0_20260611T_groundhog_v2/result.json`
- Run label: `new_experiment`
- Episodes: `96`
- Policy: `level1_pose_bandit_v0`
- PufferLib gate: `pufferlib.available=true`, `pufferlib.pufferl` imported from `/tmp/pufferlib-4.0-newjordan`

Bandit result:

- `target_clean_catch_rate`: `0.6770833333333334`
- `all_bucket_clean_catch_rate`: `0.6770833333333334`
- `level_completion_rate`: `0.6770833333333334`
- `overflow_rate`: `0.3229166666666667`
- Learned positive clean poses included `target_0:pose_x06_y04`, `target_0:pose_x06_y06`, and `target_1:pose_x06_y11` with `q=10.0`.

Deterministic learned-policy eval:

- Review page: `codexlab/runs/puffer_paintflow_groundhog_l1_learned_clean_pose_v0_20260611T_eval_v2/index.html`
- Result JSON: `codexlab/runs/puffer_paintflow_groundhog_l1_learned_clean_pose_v0_20260611T_eval_v2/result.json`
- Packets: `codexlab/runs/puffer_paintflow_groundhog_l1_learned_clean_pose_v0_20260611T_eval_v2/packets.jsonl`
- Brain calls: `codexlab/runs/puffer_paintflow_groundhog_l1_learned_clean_pose_v0_20260611T_eval_v2/brain_calls.jsonl`
- Manifest: `codexlab/runs/puffer_paintflow_groundhog_l1_learned_clean_pose_v0_20260611T_eval_v2/report_manifest.json`
- Trace: `codexlab/runs/puffer_paintflow_groundhog_l1_learned_clean_pose_v0_20260611T_eval_v2/report_trace.jsonl`
- Run label: `controlled_run`
- Episodes: `32`
- Policy: `level1_learned_clean_pose_v0`
- Action packets: `304`
- Brain calls: `32`
- Oracle sidecars: `0`
- Review page local image check: `80` image refs, `0` missing

Deterministic eval result:

- `episode_count`: `32`
- `target_catch_rate`: `1.0`
- `target_clean_catch_rate`: `1.0`
- `all_bucket_catch_rate`: `1.0`
- `all_bucket_clean_catch_rate`: `1.0`
- `overflow_rate`: `0.0`
- `level_completion_rate`: `1.0`
- `mean_episode_reward`: `10.905000000000001`

Boundary:

- This is evidence that the current branch can groundhog SP80 level-1 paintflow into a primitive-action algorithm that catches cleanly and redirects/catches the requested target bucket under the recorded target schedule.
- This is still not a native PufferLib trainer result. PufferLib is an import-gated runtime dependency here; SP80 execution remains through the repo's Python game/controller API.
- This is not a full Angel/CONTRA gameplay report and does not satisfy `angel_report_verify.py --strict-png-ledger --require-gameplay-visual-contract`.

## Controlled CONTRA/Oracle/Ghost Level-1 Run

Condition artifact:

- `codexlab/conditions/puffer_contra_l1_learned_paintflow_v0.json`

Code changes:

- `codexlab/sp80_puffer_policy.py` now exposes `PufferPaintflowLearnedBrain`, a primitive-only learned paintflow policy with a `puffer_paintflow_learned_policy` recommendation overlay.
- `codexlab/sp80_player_harness.py` now accepts backend `puffer_paintflow_learned` and lets a backend augment the CONTRA recommendation packet before the brain consumes it.
- `codexlab/sp80_run_contra_levels.py` now accepts the Puffer backends and passes `--require-pufferlib` / `--target-bucket-index` through to the controlled harness.

Final full-report run:

- Run dir: `codexlab/runs/puffer_contra_l1_learned_paintflow_v0_20260611T_full_report_v2`
- Level result: `codexlab/runs/puffer_contra_l1_learned_paintflow_v0_20260611T_full_report_v2/level_00/result.json`
- Packets: `codexlab/runs/puffer_contra_l1_learned_paintflow_v0_20260611T_full_report_v2/level_00/packets.jsonl`
- Brain calls: `codexlab/runs/puffer_contra_l1_learned_paintflow_v0_20260611T_full_report_v2/level_00/brain_calls.jsonl`
- Ghost: `codexlab/runs/puffer_contra_l1_learned_paintflow_v0_20260611T_full_report_v2/level_00/ghost.json`
- Angel report: `codexlab/runs/puffer_contra_l1_learned_paintflow_v0_20260611T_full_report_v2/angel_report`
- Angel viewer URL: `/?tool=contra-oracle&stage=portband&data=/data/contra_sp80_episode_levels_20260611T155111Z/contraview_report.json`
- Contraview report JSON: `codexlab/runs/puffer_contra_l1_learned_paintflow_v0_20260611T_full_report_v2/angel_report/contraview_report.json`

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab:/tmp/pufferlib-4.0-newjordan python3 -B codexlab/sp80_run_contra_levels.py --levels 0 --backend puffer_paintflow_learned --require-pufferlib --target-bucket-index 0 --max-actions 20 --scale 8 --out-dir codexlab/runs/puffer_contra_l1_learned_paintflow_v0_20260611T_full_report_v2
```

Result:

- Summary status: `reported`
- Level row status: `nominal`
- Episode status: `completed`
- Action count: `6`
- `levels_completed_delta`: `1`
- Ghost status: `captured_from_controlled_completion`
- Episode verifier return code: `0`
- Angel report verifier: `Angel report verifier passed`
- Angel assets: `243` PNGs
- First brain call consumed a `contra_recommendation_packet` row with candidate kind `puffer_paintflow_learned_policy`.

Boundary:

- This is a full controlled SP80/CONTRA/Oracle/Ghost level-1 run through the existing episode harness and strict Angel report verifier.
- The learned row is an action recommendation input, not standalone solve truth. The recorded action sequence and `levels_completed_delta=1` are the controlled-run result.
- This still is not native PufferLib training. PufferLib is required and imported from the patched local 4.0 checkout, but the actual environment execution is the repo SP80 Python harness.

## Verification

Passed:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/dev_check.py
```

Passed:

```bash
PYTHON=python3 bash codexlab/sp80_integration_smoke.sh
```

Observed integration smoke summary:

- PNG frames produced: `38`
- Run status: `passed`
- Failure count: `0`
- SP80 verifier: `passed`

## Next Valid Step

Do not claim native PufferLib training yet. The patched checkout proves the native CPU backend can build and import on this host, and the level-1 pose-bandit/eval proves the SP80 paintflow mechanic can be driven through primitive actions in the repo harness. The next valid step is to either build a native PufferLib SP80 environment for actual Puffer training or promote the learned level-1 pose policy into the full controlled SP80/CONTRA gameplay harness with the strict Angel visual contract.
