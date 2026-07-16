# CODEX HANDOFF - 2026-07-04 - Mythos boundary correction

Written by Codex after the user clarified that Fable's handoff/mythos
boundary remains authoritative: the pre-v8 line is admitted work, v8 is the
last solid release checkpoint, and the corruption/drift starts after v8.

## Audit First

Command run before writing this note:

```bash
python3 synthesis_lab/orchestration/honesty_audit.py
```

Result:

- `summary: AUTONOMOUS_CANDIDATE=84, IMITATION_ASSISTED=46, INVALID_HEADLINE=6`
- `active_summary: AUTONOMOUS_CANDIDATE=72, IMITATION_ASSISTED=23`
- `quarantined_summary: AUTONOMOUS_CANDIDATE=12, IMITATION_ASSISTED=23, INVALID_HEADLINE=6`

Interpretation: invalid headline artifacts remain quarantined and disclosed.
There are no active invalid-headline artifacts according to this audit.

## Corrected Boundary

Codex overcorrected by treating all pre-v8 lineage as unusable under the new
no-imitation wording. That is not the user's intended Mythos boundary.

Use this corrected boundary:

- `outcome_v8/last.pt` is the admitted Mythos release checkpoint.
- Everything required to build v8 is admitted unless the user explicitly
  reopens that boundary.
- `outcome_v9` and `outcome_v9_retry_20260702` are not releases. They are
  scout/diagnostic artifacts that drifted after v8.
- The major route-imitation corruption is v16-v26 plus stage36-stage60 route,
  human-mined, runtime-replay, and route-provider gate artifacts.
- Stage25/Stage33 goal-label datasets and Stage26/Stage34 goal-head
  checkpoints remain historical diagnostics only for CNN-alone claims because
  their recorded lineage includes human-mined level-3 data.

Do not reset the project to a from-scratch clean-room baseline unless the user
explicitly asks for that. The main training lane should preserve v8 and cut off
the corrupted post-v8 additions.

## Current Release / Comparator State

Admitted release checkpoint:

- `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v8/last.pt`
- sha256: `6e5a28f0d4039688b53fbc6d0cfa7de30452ed678bd5ea70b08fe5a5890ccc9e`

Other v8 artifact:

- `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v8/best.pt`
- sha256: `1a66cb646c3cb7fbf79de5da5e699e884aedc07fab08ff83b10c2ac659c5ae52`

Recorded v8 metrics from checkpoint metadata:

- `level/true_pos=109`, `level/false_pos=0`
- `settle/iou=0.6558808423215203`
- `control/acc=0.9994720168954594`
- `catch/acc=0.9563025210084034`
- `inv/click_cell_acc=0.4816017316017316`
- `bc/action_type_acc=1.0`

Use v8 as the baseline/comparator for any admitted continuation.

## v9 / v9retry Status

Do not promote v9 or v9retry.

Memory + handoff boundary says:

- v9retry was the recovered v9 condition: 40 epochs, seed 0,
  `stack_frames=2`, batch 64, lr `3e-4`, v8 shards plus
  `control_play_train.pt`, v8 val shards plus `control_play_val.pt`.
- Local behavioral gates were green for v9retry, but this was scout evidence,
  not release proof.
- v9retry failed same-valset no-loss against v8:
  - v8last: `bc/action_type_acc=1.0`, `level=109 tp / 0 fp`
  - v9retry last: `bc/action_type_acc=0.9878`, `level=108 tp / 0 fp`
  - v9retry best: `bc/action_type_acc=0.9837`, `level=109 tp / 1 fp`
- Live ARCPrize scout for v9retry got only `1/6` levels, score
  `4.761904761904762`, 500 actions, 81 resets, 0 policy errors.
- Concrete live failure basin: clear level 0, choose one short level-1 arrange
  plan, then repeat `megarexus_arrange_attempt_cap` / `ACTION5` until game
  over.

## Codex-Created Clean-Room Diagnostic

Codex created and launched a clean-room smoke after initially overapplying the
no-imitation rule. Treat it as diagnostic only, not the main Mythos lane.

Condition:

- `synthesis_lab/experiments/megarexus_cnn/runs/stage61_noimit_mechanics_smoke_20260704/CONDITION.json`
- committed in `993d07d Register clean CNN mechanics smoke condition`

Output:

- `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v27_noimit_mechanics_smoke_20260704/`
- review page: `http://100.124.153.1:8766/synthesis_lab/experiments/megarexus_cnn/runs/outcome_v27_noimit_mechanics_smoke_20260704/index.html`
- watcher report: `WATCH_REPORT.json`
- watcher exit `0`, timeout `false`

Checkpoint hashes:

- `best.pt`: `12fa722ba7c7a7f6846a8070258f41cc84d8561ea287cf5f97101bb12ce96651`
- `epoch_002.pt`: `a59a76d666dcf01bfb1fe0c0f0b8b802e72acc0be6ddaea74882f76efb32185b`
- `epoch_004.pt`: `649501959f9e92eb32db880d6afe1355b0c780887ae9f8b1d8d7a15fed532eae`
- `last.pt`: `7f95afb259fe9d262f58fe763b6cc7abc7641e8a84fbcc5e8c730f46b97e51bd`

What it showed:

- Clean training path works.
- It learned basic dynamics/control/catch.
- It does not replace v8.
- It is weak on the important v8 skills:
  - level positives: `0/3 true positives`
  - click localization: `0.0432` accuracy, mean distance about `25 px`

## Current Commits

Current HEAD when this note was written:

- `993d07d22f5448cf7d7c21bb5506b280c4c1d529`

Recent commits:

- `993d07d Register clean CNN mechanics smoke condition`
- `981b93b Enforce no-imitation CNN training guards`
- `01f42a9 Record no-imitation CNN training regime`
- `07717dc Quarantine route replay artifacts`
- `9dc393e Goal schema is the lock inventory, not pour history`

## Dirty / Untracked State To Preserve

Do not clean this tree casually. There are many untracked artifacts and two
modified files not resolved by this note:

- `environment_files/sp80/589a99af/metadata.json`
- `synthesis_lab/notes/COURSE_ADVISORY_20260703_STAGE23_RESUME.md`
- untracked `codexlab/runs/`
- quarantined `_invalid/` artifacts under runs/data/orchestration
- untracked historical run outputs and logs
- untracked v27 smoke outputs and `stage61` launcher log

This note itself is a new handoff note. Commit it only if the user asks or
after confirming the desired scope.

## Next Valid Work

Correct mainline:

1. Use `outcome_v8/last.pt` as the admitted parent/comparator.
2. Create a pre-registered `train_watch.py` condition that declares the
   parent as `admitted_legacy_v8_parent`.
3. Add only post-v8 data that passes the corrected boundary:
   no v9 control retry unless explicitly re-audited and admitted, no route
   heads, no human-mined level traces, no runtime replay, no answer labels.
4. Target the real missing skills without copying answers:
   level-positive/key-lock/objective recognition, click localization, and
   robust recovery from failed arrange states.
5. Gate against v8 first. Promotion requires no-loss or an explicitly named
   tradeoff the user accepts.

Invalid direction:

- Do not continue from v16-v26.
- Do not treat route-provider PASS gates as model evidence.
- Do not call the clean-room v27 smoke the new baseline for Mythos.
- Do not launch live scorecards or additional training without a recorded
  condition and dry-run preflight.

