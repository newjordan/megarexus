# Full mine → LoRA (v2)

**Goal:** Every registered system is mined into a fixed slot and fused into the
trainable `LegalBlendLoRA`. No system is “documented only.”

## Pipeline

```text
frame + transitions
  → consume_all(SYSTEM_ORDER)     # 25 adapters
  → encode_system_slot × 25       # SYS_SLOT=12 each
  → global hist/edge/sym/comps
  → meta presence flags
  → FEATURE_DIM vector (v2)
  → feat_mlp (LoRA) + conv trunk
  → change_logits + policy_logits
  → train on did_change / weak policy
```

## Dimensionality (v2_fullmine)

| Block | Dim |
|---|---:|
| Global (hist/edge/sym/comps) | 59 |
| 25 systems × 12 | 300 |
| Meta (25 presence + 8 guards) | 33 |
| **FEATURE_DIM** | **392** |

Schema: `legal_search.features.v2_fullmine`

## Always-on mining

- Collect + train default: `run_consume=True` every step
- Specialist / quarantine slots encode **boundaries** (not solutions)
- Empty slot if adapter fails — still present flag 0

## Train

```bash
python3 -B legal_search/train/train_lora.py \
  --out-dir legal_search/runs/blend_lora_v1_fullmine \
  --epochs 10 --batch-size 32 --collect-episodes 20
```
