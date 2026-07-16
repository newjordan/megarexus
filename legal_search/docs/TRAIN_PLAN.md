# Legal Blend LoRA — Training Plan

**Date:** 2026-07-08  
**Status:** BLOCKED until mining complete — see `../MINING_THEN_TRAIN.md`  
**Package:** `legal_search/train/`  
**Gate:** `../MINING_COMPLETE.json` must exist with `complete: true`

## Goal

Blend **every mined legal_search consume system** into features that condition a
**LoRA-adapted** action/change model. Train for real on GPU with **legal labels only**.

## Non-goals

- Not a Qwen text-LoRA over SP80 control intents (`sp80_lora_dry_run`).
- Not BC on human-mined routes / `_invalid/` shards.
- Not shipping r4/r5 fixture win banks as targets.
- Not claiming portfolio ARC solve from v0.

## Architecture

```text
frame one-hot [B,16,64,64]
  + legal feature vector [B,F] from consume_all (pooled)
      → Conv trunk (trainable or warm-start)
      → LoRA on late Linear/Conv projections
      → heads:
           change_logits [B,4101]   # Goose: did this action change frame?
           policy_logits [B,4101]   # weak policy / ranking bias
```

## Labels (legal)

| Head | Label |
|---|---|
| change | BCE: taken action changed frame? |
| policy | CE: taken action index (from legal explorer / search), weak |
| optional aux | levels_up window weight on policy |

## Data sources (priority)

1. Offline Arcade rollouts with `LegalAgent` (preferred).
2. Synthetic grid physics micro-envs (movable blob) if Arcade unavailable —
   still teaches change+click structure without SP80 answer keys.
3. Never: `_invalid/` human routes, spill/droplet oracles.

## Mining job (ongoing)

- `legal_search/MINING_MANIFEST.md` — system keep/strip table
- `legal_search/docs/MINING_HANDOFF.md` — how adapters feed features
- Every new system must register in `consume/registry.py` + feature encoder dims

## Train command

```bash
python3 -B legal_search/train/train_lora.py \
  --out-dir legal_search/runs/blend_lora_v0 \
  --epochs 8 --batch-size 64
```

## Success criteria (v0)

1. Smoke: forward + one optimizer step on GPU.
2. Train loss decreases vs step 0.
3. Val change-head better than always-predict-base-rate (or random).
4. Artifacts: `best.pt`, `metrics.json`, `config.json`, `train_log.jsonl`.
5. Receipt: `legal=true`, no banned lineage paths in config.

## AdaJEPA beneficials (added pre scale-up train)

See `ADAJEPA_PORT.md`. At **deploy / collect** time:

- online buffer of recent transitions  
- **one gradient step** on **LoRA only** after each step  
- SSL change (+ light latent) loss — no reward  
- horizon-1 plan from change/policy blend, then replan  

Offline `train_lora.py` still builds the base; Ada loop keeps learning in closed loop.

## Claim language

> "Trained LoRA blend on legal_search experience (change + weak policy).  
> Specialist SP80 DW 6/6 remains separate research evidence.  
> Deploy may run AdaJEPA-style online LoRA adapt (plan-act-adapt-replan)."
