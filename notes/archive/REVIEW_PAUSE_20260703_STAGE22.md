# Review pause - 2026-07-03 21:42 CDT

This is a pause/context-refresh note for the Megarexus CNN/SP80 lane.

## Stage numbering boundary

The stage numbers are local bookkeeping labels, not a finite countdown and not
evidence that the work is nearly done. In particular, "Stage22" does not mean
"stage 22 of 23".

- Stage20 was the latest live ARC SP80 scout: it reached
  `total_levels_completed=3` with closed SDK score `28.57142857142857`, then
  stalled in level-3 control.
- Stage21 was static review of the first live level-3 frame.
- Stage22/Stage22b are offline level-3 transition-shard collection/replay
  artifacts.
- Stage23 is only the next proposed v13 training condition label. It has not
  been launched.

## Current controlled evidence

Stage22 exclusive was a controlled negative: it omitted the first level-4
action from mined human sequences and did not reach level 4.

Stage22b inclusive is the current useful artifact:

- Review URL verified `200`:
  `http://127.0.0.1:8765/synthesis_lab/experiments/megarexus_cnn/runs/stage22b_level3_human_mined_inclusive_v12_graphadaptive_20260704/index.html`
- Report:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage22b_level3_human_mined_inclusive_v12_graphadaptive_20260704/level3_human_mined_report.json`
- Condition sha256:
  `d2e5a3feb35b3a2a055abe20a1e8847699d178163e8960d3009bd35bfa8f5d4a`
- Script:
  `synthesis_lab/orchestration/collect_level3_human_mined.py`
- Script sha256:
  `13042c389c6b255cc7d1a8c92b0e2dad94ad9dedf9b43c2274f21744531e9473`
- v12 checkpoint sha256:
  `211e1eca2543f581b871f4a8f26dee583d8a0205eef5ba92ba1073be61939999`
- Prefix status: `ok`, `prefix_action_count=27`
- Level-3 frame sha256:
  `507982e727fe170293044193818337cc11b14a0d820150f533bf95e3c0acf7ca`
- Result: `6/12` candidates reached level 4; `8` trajectories were kept.
- Output shard:
  `synthesis_lab/experiments/megarexus_cnn/data/level3_human_mined_inclusive_stage22b_20260704.pt`
- Output shard sha256:
  `2e98eeaa68aa8a71da62ad5a23a1df7704f4781d974991bc809746eeed91e6e3`
- Boundary from the report:
  controlled offline replay of human-mined level-3 sequences; not model-selected
  or live evidence.

## Not done

- v13/Stage23 training has not been launched.
- No v13 gate has been run.
- No new live scorecard should be launched from Stage22b alone.
- The arXiv paper fit check for `https://arxiv.org/abs/2607.01224` was paused
  before a subagent produced a result. Do not cite or act on a paper verdict
  until a fresh review is completed from the arXiv PDF and any official code.

## Runtime context at pause

Checked at `2026-07-03T21:42:12-05:00`:

- Megarexus has no active training, gate, collection, or live scout process in
  the local process scan.
- Evidence server is still running at `127.0.0.1:8765`.
- GB10 is not free for new Megarexus training at this exact pause: unrelated
  Leanstral export PID `2517434` is using CUDA with about `3555MiB`.
- Memory is healthy: `121Gi` total, `103Gi` available, `4.0Gi` swap used.
- Disk is acceptable: `/` has `930G` free; `/srv/models-hdd` has `15T` free.

## Next valid resume point

Resume by rechecking runtime first, then either:

1. Complete the arXiv paper fit review as a read-only research task, or
2. Create a recorded Stage23/v13 condition that consumes the Stage22b shard and
   labels itself as a new experiment, not a reproduction.

Do not interpret the stage label as a claim that the remaining work is small.
