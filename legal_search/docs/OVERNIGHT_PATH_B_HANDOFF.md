# Overnight Path-B Handoff — level-shaped legal LoRA

**Date:** 2026-07-09  
**Schema:** `legal_search.overnight_level.v1`  
**Status:** **FINISHED** — Path B complete; micro-env floor held; Arcade level not locked on best ckpt  
**Lane:** legal_search only (gated after JOB 1 mine)

## One-line outcome

Path-B overnight loop ran **39 cycles / ~8h**, best **synth level rate 0.975** (39/40), retained at `runs/overnight_level/best/best.pt`. **Arcade probe on that best ckpt still `any_arcade_level=false`** (SP80 thrash-die; ft09 no-op).

## Mission arc (how we got here)

| Phase | What | Result |
|---|---|---|
| JOB 1 mine | Full legal_search stack → 25 systems, v2_fullmine **392-D** | Gate: `MINING_COMPLETE.json` |
| Gated train | Legal labels only (did_change, weak action, levels_up windows) | Snack → huge regimes; change_acc ≠ levels |
| Honest level probe | Arcade SP80 / ft09 | **0 levels** on change-acc models |
| Path B | Level-shaped collect + train + `level_eval` | Micro-env floor; Arcade still hard |
| Overnight autopilot | `train/overnight_level_loop.sh` 8h | Best synth 0.975; Arcade not retained |

**Banned lineage (still banned):** spill_model, dropletsim, human_mined_routes, fixture_win_banks_as_labels.

## Final run state

```json
{
  "status": "stopped",
  "started": "20260709T060604Z",
  "cycle": 39,
  "best_synth_rate": 0.975,
  "best_ckpt": "…/cycles/c036_level_h_s619/best.pt"
}
```

- Live mirror: `legal_search/runs/overnight_level/state.json`
- Master log: `legal_search/runs/overnight_level/logs/overnight_20260709T060604Z.log`
- Milestones: `legal_search/runs/overnight_level/MILESTONES.log`

## Best checkpoint (promote target)

| Item | Value |
|---|---|
| Path | `legal_search/runs/overnight_level/best/best.pt` |
| Source cycle | `c036_level_h_s619` (seed 619, heavy) |
| Features | `legal_search.features.v2_fullmine`, dim **392** |
| Legal / gated | true |
| Level-shaped | true (`policy_coef=0.9`, `levels_up_boost=10.0`) |
| Train size (that cycle) | n_train 8914 / n_val 990; n_levels_up_train 463; n_success_window 5558 |

### Best metrics

| | val loss | change_acc | base rate |
|---|---:|---:|---:|
| base | 3.63 | 0.419 | 0.859 |
| **best** | **0.531** | **0.995** | 0.859 |

### Best level_eval

| Env | Result |
|---|---|
| synthetic_blob | **39/40** level-ups, rate **0.975**, mean steps ~65 |
| arcade sp80-589a99af | 3 eps → GAME_OVER ~7 steps, change_rate 1.0, **0 levels** |
| arcade ft09-0d8bbf25 | 3 eps → 150 steps, change_rate 0.0 (no-ops), **0 levels** |
| `any_synth_level` | **true** |
| `any_arcade_level` | **false** |
| `floor_bar` | true (synth floor held) |

## What worked

1. **JOB1 mine depth** — full consume stack, legal feature encoder, gate discipline after premature-complete fix.
2. **Level metric over change_acc** — evaluation correctly became `levels_completed` / level_up rate, not val change_acc alone.
3. **Path-B success-window / thrash weights** — synthetic micro-env went from “can it level?” to **near-ceiling (0.975)**.
4. **Overnight loop** — collect → train → level_eval → promote on synth or any arcade level-up; 39 cycles without gate breach.
5. **Collect-time Arcade pings** — `MILESTONES.log` recorded arcade level-ups during **cycles 9, 11, 18, 31** (collection / intermediate, **not** retained on final best eval).

## What failed / open gap

1. **Best-ckpt Arcade = still zero levels.** Promote rule kept the best **synth** model; that model dies (SP80 thrash) or freezes (ft09 no-op).
2. **Collect milestones ≠ reliable eval.** Level-ups during rollouts did not transfer into a durable Arcade-winning checkpoint under the retained best.
3. **Last cycles regressed synth** (e.g. c039 rate 0.125) — loop did not overwrite best; noise / regime variance remains.
4. **Failure modes are opposite extremes:**
   - SP80: over-active → early GAME_OVER
   - ft09: under-active → pure no-ops to step cap

## Artifacts map

```
legal_search/runs/overnight_level/
  state.json                 # finished, cycle 39, best_synth 0.975
  best/
    best.pt                  # PROMOTE / probe this first
    level_eval.json
    metrics.json
    config.json
  cycles/c001_…_c039_…/      # full per-cycle data.pt, train logs, evals
  logs/overnight_20260709T060604Z.log
  MILESTONES.log             # arcade level-up cycles 9/11/18/31
  README.md                  # loop ops

legal_search/train/
  overnight_level_loop.sh
  collect.py                 # annotate_level_weights, level_shaped, multi-game
  train_lora.py              # regimes incl. level
  level_eval.py              # synth + arcade probe
```

Related earlier runs (context only):

- `runs/blend_lora_gated_v1` — snack gated train  
- `runs/gated_v2_regime` — huge regime  
- `runs/level_v1` — first level-shaped (~0.95 synth)

## Re-probe (do not retrain to “check”)

```bash
python3 -B legal_search/train/level_eval.py \
  --ckpt legal_search/runs/overnight_level/best/best.pt \
  --synth-eps 40 --arcade-eps 3
```

## Suggested next work (Path C — not started)

Do **not** re-run the same 8h Path-B loop expecting Arcade. Open Path C:

1. **Promote on Arcade, not only synth** — keep a separate `best_arcade.pt` if any level-up; do not discard mid-cycle Arcade wins for pure synth rate.
2. **Anti-thrash + anti-no-op shaping** — SP80 needs survival / deliberate change; ft09 needs forced exploration when change_rate≈0.
3. **Denser real Arcade success windows** — mine trajectories around the collect-time level-up events (cycles 9/11/18/31) into level-shaped weights; inspect those cycle `data.meta.json` / logs for seeds and game_ids.
4. **Eval budget** — more Arcade eps + longer SP80 survival budget before GAME_OVER counts as policy fail.
5. **Hold legal gate** — still no spill / human routes / fixture win banks as labels.

## Docs chain

| Path | Role |
|---|---|
| `legal_search/MINING_THEN_TRAIN.md` | mine-then-train gate |
| `legal_search/MINING_COMPLETE.json` | JOB1 gate (must stay complete) |
| `legal_search/docs/MINING_HANDOFF.md` | systems → features |
| `legal_search/docs/TRAIN_PLAN.md` | train architecture / labels |
| `legal_search/docs/FULL_MINE_TO_LORA.md` | 392-D fullmine |
| `legal_search/docs/ADAJEPA_PORT.md` | online adapt ideas |
| `legal_search/docs/USED_VS_UNUSED.md` | mine inventory |
| **`legal_search/docs/OVERNIGHT_PATH_B_HANDOFF.md`** | **this file** |
| `PRIMARY_LANE.md` | specialist vs legal split |

## Claim language (honest)

- **Can claim:** legal fullmine LoRA, gated train, level-shaped Path B, **synthetic level floor ~97.5%**, overnight autopilot finished cleanly.  
- **Cannot claim:** solved SP80/ft09, portfolio ARC levels, or “Arcade level from best ckpt.”  
- Collect-time Arcade milestones are **signals for Path C**, not a solved scorecard.

---

Handoff complete. Thanks for the work window — Path B delivered the micro-env floor and a clean legal best ckpt; Arcade is the next job.
