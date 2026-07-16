# Primary lane (2026-07-08)

## Order of operations (hard)

**Read first:** [`legal_search/MINING_THEN_TRAIN.md`](legal_search/MINING_THEN_TRAIN.md)

1. **JOB 1 — Mining** (complete checklist → write `legal_search/MINING_COMPLETE.json`)
2. **JOB 2 — Training** (blocked until gate file says `complete: true`)

`train_lora.py` and train-oriented `collect.py` **refuse to run** while mining is incomplete.

## Two different primaries — do not mix claims

### A) Honest specialist evidence (what already scored)

| Game | Policy | Status |
|---|---|---|
| sp80 | `megarexus_dw_directgrid` | Live + ranked 6/6 (specialist; fixture/search DW) |
| ft09 | `ft09_glyph` | Live 6/6 (specialist) |

Runners: `codexlab/arc_suite_runner.py`, `codexlab/arc_live_suite_runner.py`.

### B) Legal general agent (mined algorithms + blend LoRA)

Package: **`legal_search/`**

- Consume every system (CONTRA, PortBand, Oracle, Ghost, Loom/LeWM, DW search, Goose, …)
- Strip game answers / fixture win banks / quarantine lineages
- Loop: explore → induce → search → act → env judges
- **Blend LoRA** trains on legal experience (`legal_search/train/`)

```bash
python3 -B legal_search/smoke.py
python3 -B legal_search/train/train_lora.py --out-dir legal_search/runs/blend_lora_v0
```

Docs: `legal_search/README.md`, `legal_search/MINING_MANIFEST.md`,  
`legal_search/docs/TRAIN_PLAN.md`, `legal_search/docs/MINING_HANDOFF.md`.  
Trained full-mine: `legal_search/runs/blend_lora_v1_fullmine/`  
**Scale train:** `legal_search/runs/blend_lora_v2_train/` (20 ep, 1858 train rows — **use this ckpt**).  
Feature schema `v2_fullmine` (392-D, all 25 systems → LoRA).  
Deploy online adapt: `legal_search.train.ada_adapt.AdaBlendPolicy`.

## Not primary

- `codexlab/sp80_player_harness.py` (control lab)
- CONTRA cockpit as solver
- Quarantined spill/droplet/human routes
- Frozen SP80-only CNN as “general AGI agent”

## Rule of thumb

- **Specialist scorecard claim** → name the specialist policy.
- **General learning claim** → must go through `legal_search` with empty cross-game memory and no banned priors.
