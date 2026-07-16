# Level-group multi-game training (robust patterns)

**Date:** 2026-07-15  
**Goal:** general solving capability via **ROBUST patterns**, not single-game grind.

## Regime

Train **levels in groups**, on **all games at once**:

```text
Cohort L0–L1:
  for every game in suite(24):
      play honestly from reset
      keep transitions while levels_completed ∈ {0,1}
  interleave (round-robin) → one mixed stream
  MegaMind.train_transition on mixed mini-batches
  shared weights must work across all prefixes
```

Then advance the group (L2–L3, …) the same way.

## Why

| Failure mode | This regime |
|---|---|
| Fit SP80 / one prefix only | 24 games every cohort |
| Memorize level sequences | level **bands**, mixed games |
| Fragile color/layout hacks | multi-variate pattern growth pad (G1–G40) |
| Soft omit hard games | suite discovery includes all local envs |

## Suite (24)

Under `synthesis_lab/experiments/other_games/environment_files/`:

ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0  
r11l re86 s5i5 sb26 sc25 sk48 su15 tn36 tr87 tu93 vc33 wa30

## Entry points

| File | Job |
|---|---|
| `brain/suite_games.py` | discover suite, round-robin |
| `brain/collect_level_group.py` | multi-game level-band collector |
| `brain/train_level_groups.py` | train + metrics + checkpoint |

```bash
python -B synthesis_lab/orchestration/rdl/brain/train_level_groups.py \
  --groups 0-1 --steps-per-game 16 --epochs 3 --device cpu
```

## Honesty

- No `on_set_level` injection  
- No spill / win-bank train labels  
- Env `levels_completed` is the only level signal  
- Arcade failure → per-game pattern-growth synthetic (cohort stays multi-game)
