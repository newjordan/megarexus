---
name: run-synthesis-lab
description: Run, probe, and screenshot the megarexus synthesis_lab SP80 stack (MegarexusCNN + offline arc_agi Arcade). Use when asked to run the model, verify the stack with a pre-flight, run a paired level probe, render or screenshot an SP80 frame, or sanity-check checkpoints, corpus, and data shards.
---

The synthesis_lab stack is not a server or GUI — it is a trained CNN
(`MegarexusCNN`, 3.64M params) plus a local OFFLINE ARC-AGI-3 game
environment (SP80 "liquid pour", `sp80-589a99af`). The agent handle is
`.claude/skills/run-synthesis-lab/driver.py`: `preflight` verifies the
whole stack, `probe` plays real paired episodes, `render` screenshots a
frame. All paths below are relative to the repo root
(`/home/frosty40/megarexus`).

Everything here is SCOUT_EXPERIMENT, offline only. **Before making any
claim or training anything, read `synthesis_lab/notes/HANDOFF.md`** —
§2 has non-negotiable honesty rules (answer-key ban: hand-authored game
models like `codexlab/contra_head/sp80_spill_model.py` must never touch
training; scripted completions are never model capability).

## Prerequisites

Everything is preinstalled on this rig — no install step was needed.
The stack runs on the miniconda interpreter (NOT `python3` from PATH
unless it resolves there):

- `/home/frosty40/miniconda3/bin/python3` — Python 3.13.12 with
  torch 2.11.0+cu130 (CUDA on NVIDIA GB10), numpy 2.4.3, PIL, and
  `arc_agi` (site-packages).
- Game definition: `environment_files/sp80/589a99af/sp80.py` (loaded
  by the offline Arcade; nothing to install).

On a genuinely clean machine you would need those packages; no install
commands were verified, so none are listed. Verify readiness with
`preflight` below instead.

## Setup / Build

None. No build step, no env vars, no services.

## Run (agent path)

### 1. Pre-flight — is the stack healthy?

```bash
/home/frosty40/miniconda3/bin/python3 -B .claude/skills/run-synthesis-lab/driver.py preflight
```

~30 s. Boots the offline Arcade, runs the Portband adapter on the real
reset frame, loads `runs/outcome_v4/best.pt`, executes
`forward_settle`/`forward_control`, and dumps stored val metrics,
level-probe reports, corpus stats (1245 frames, 6 episodes), and data
shard counts. Ends with `ALL SECTIONS PASSED` and exit 0, or a
`FAILED sections: [...]` list and exit 1.

### 2. Probe — actually play the game, model vs random

```bash
/home/frosty40/miniconda3/bin/python3 -B .claude/skills/run-synthesis-lab/driver.py probe --episodes 2
```

~10 s for 2 episodes. Runs full paired episodes through
`synthesis_lab/orchestration/level_probe.py` against a scratch run-dir
(symlinked checkpoint) so the canonical reports in
`runs/outcome_v4/` are never overwritten. Verified output:

```
  model   2/2 episodes completed a level (pours=14, strict_catches=12)
  random  0/2 episodes completed a level (pours=7, strict_catches=2)
```

Canonical 20-episode result (stored in
`synthesis_lab/experiments/megarexus_cnn/runs/outcome_v4/level_probe_learned.json`):
model 20/20 level-1 completions, random 5/20, paired seeds
20280001–20280020. Full JSON report path is printed at the end.

### 3. Render — screenshot a frame

```bash
/home/frosty40/miniconda3/bin/python3 -B .claude/skills/run-synthesis-lab/driver.py render --out /tmp/sp80_reset.png
```

Boots the env, resets, saves a 512×512 PNG (project palette from
`dataset.py`). Level 0 looks like: green gauge strip on row 0, spout at
top center (dark + magenta), blue movable bar, two yellow funnels,
gray floor. `--seed` and `--scale` are optional.

| driver command | what it does |
|---|---|
| `preflight [--run-dir D] [--seed N]` | full stack sanity pass, exit 0/1 |
| `probe [--episodes N] [--seed N] [--finder learned\|probe] [--run-dir D]` | paired model-vs-random episodes, clobber-safe |
| `render [--seed N] [--scale K] [--out PATH]` | reset-frame PNG |

## Going deeper (not driver-wrapped)

The project's own orchestration lives in `synthesis_lab/orchestration/`
(`level_probe.py`, `behavioral_catch_probe.py`, `collect_catch_play.py`)
and training in `synthesis_lab/experiments/megarexus_cnn/train.py`
(`--run-name <name>`, run from the experiment dir). Training and
collection were NOT run in the session that authored this skill —
consult `synthesis_lab/notes/HANDOFF.md` §3 before using them.

## Test

There is no separate test suite; `preflight` is the health check. The
stored val metrics it prints (catch 1.0 vs 0.528 base, control 1.0 vs
0.528, level head 3/3 tp 0 fp) are the expected checkpoint numbers.

## Gotchas

- **`level_probe.py` writes its report INTO `--run-dir`** (default name
  `level_probe.json`, via `--out-name`). Pointing it at
  `runs/outcome_v4` can clobber the canonical
  `level_probe_learned.json`/`level_probe_careful.json` evidence. The
  driver's `probe` avoids this with a scratch dir + `best.pt` symlink.
- **Scorecard INFO lines are local artifacts.** OFFLINE Arcade prints
  `Initialized ScorecardManager…` / `Created new scorecard: <uuid>` on
  every boot. No live scorecard is opened.
- **`glove_ports(bbox_xyxy, component_id)` takes a single bbox**, not
  the list returned by `frame_components()`. Passing the component list
  raises `TypeError: int() argument must be … not 'dict'`.
- **Corpus manifest grouping key is `episode_id`**, not `episode`
  (`KeyError: 'episode'` otherwise). Manifest convention: row t's
  action produces frame t+1.
- **`torch.load` needs `weights_only=False`** for both checkpoints and
  the data shards (they contain plain Python objects).
- **Short random episodes are normal, not crashes.** Each level grants
  ~4 pours; the next pour call is instant GAME_OVER. Random-policy rows
  like `pours=3 … GameState.GAME_OVER` are the pour-supply mechanic.
- **Don't confuse this with `codexlab/pufferlib_env_check.py`.** That
  is a different subproject's gate (PufferLib RL) and fails on this rig
  (`ModuleNotFoundError: No module named 'pufferlib'`). This stack does
  not use PufferLib.
- **`RequestsDependencyWarning` on every run is benign** (urllib3 /
  chardet version mismatch in the conda env).

## Troubleshooting

- **`ModuleNotFoundError: No module named 'arc_agi'` (or torch)**:
  wrong interpreter — use `/home/frosty40/miniconda3/bin/python3`.
- **`TypeError: int() argument must be a string … not 'dict'` in
  `glove_ports`**: you passed `frame_components()` output; pass
  `comp["bbox_xyxy"]` and `comp["component_id"]`.
- **`KeyError: 'episode'` reading the corpus manifest**: the key is
  `episode_id`.
