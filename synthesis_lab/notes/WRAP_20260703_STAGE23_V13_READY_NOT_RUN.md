# Wrap note - 2026-07-03 21:48 CDT

This is the clean handoff after the user interrupted the v13 launch and asked
to wrap.

## Bottom line

Stage23/v13 is recorded but not completed.

- Condition:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage23_v13_level3_human_mined_20260704/CONDITION.json`
- Condition sha256 before interruption:
  `75e06152ce3849d615e6fe6e435da773bac28c76dc8d1bd094449a09518abb21`
- Run label: `new_experiment`
- Purpose: train from v12 best plus the Stage22b level-3 human-mined shard.
- Status after wrap:
  `launch_interrupted_by_user_before_checkpoint`

## What happened

The exact v13 command from the condition was launched locally at about
`2026-07-03T21:46:41-05:00`. The user interrupted and asked for handoff/wrap.
The launch kept running in the background, so it was explicitly stopped.

Stopped process family:

- Python train process PID `2548534`
- Shell wrapper PID `2548529`
- `tee` PID `2548535`

Post-stop checks at `2026-07-03T21:47:58-05:00`:

- No v13 `train.py` process remained.
- `nvidia-smi` showed no compute process and 0% GPU util.
- Log exists but only contains `[train] device=cuda`.
- Log path:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v13_level3_human_mined_20260704_train.log`
- Log size: `20` bytes.
- No v13 output directory exists:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v13_level3_human_mined_20260704`
- No v13 checkpoint, `history.json`, gate report, or review surface exists.

## Current real evidence boundary

- Stage20 remains the latest live evidence: closed SDK score
  `28.57142857142857`, `total_levels_completed=3`, policy errors `0`, then
  level-3 control stall.
- Stage22b remains a controlled offline training-data artifact, not solve
  evidence: `6/12` candidates reached level 4 and `8` trajectories were kept.
- Stage23 condition is ready to relaunch, but the run itself has produced no
  usable model artifact.

## Erratum (2026-07-03 post-wrap review)

The wrapped `CONDITION.json` recorded an invalid 65-character sha256 for
`synthesis_lab/experiments/megarexus_cnn/data/perturb_recovery_mining_v11_train.pt`
(an extra `1` was inserted mid-string). The shard file itself is fine; its
verified hash is
`8e8d850d098e4a33abdfa028109b8ee890d6116c6933b9e340adb21ad18949d5`.
The condition file was corrected in place on 2026-07-03 during the post-wrap
review. The corrected condition file's sha256 — the current authoritative
hash, superseding the pre-interruption `75e06152...` — is
`b09ca513f43ce59301c1c236aca81b70e443ccd346bfa86a2df4dddf27790950`.
All other hashes and claims in this note were verified against disk during
the same review and matched.

Course advisory for the next agent:
`synthesis_lab/notes/COURSE_ADVISORY_20260703_STAGE23_RESUME.md`.

## Resume point

Before relaunch, recheck:

1. No active Megarexus `train.py` process.
2. `nvidia-smi` or Apollo availability.
3. Current condition hash, because this file was updated after the stopped
   launch.

Then relaunch the command recorded in the Stage23 condition and let it reach at
least checkpoint output before gating. Do not claim v13 results from the
20-byte startup log.
