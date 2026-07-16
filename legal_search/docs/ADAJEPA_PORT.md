# AdaJEPA → legal blend LoRA (what we take / skip)

**Source:** [AdaJEPA: An Adaptive Latent World Model](https://agenticlearning.ai/adajepa/)  
(Wang, Bounou, LeCun, Ren — arXiv:2606.32026)

## TL;DR from AdaJEPA

```text
plan → act → observe (o_t, a_t, o_{t+1}) → adapt (self-supervised) → replan
```

- World model is **not frozen** at test time  
- **One gradient step** per replan is enough  
- Small **online buffer** (~5 transitions)  
- Update only **late layers** (we use **LoRA only**)  
- Loss is **latent / prediction error**, not reward or demos  

This is **highly compatible** with our legal Tufa+++ story and does **not** require a 30B LM.

## Map onto megarexus legal_search

| AdaJEPA | Our port | Status |
|---|---|---|
| Sensory encoder E^s | Conv trunk + legal feature fuse | offline train + online |
| Action encoder E^a | Structured action → goose index | online adapt uses taken a |
| Latent predictor f | change/policy heads (+ fused latent) | change head = SSL signal |
| MPC plan in latent | Horizon-1: argmax change/policy under avail | light MPC |
| Online buffer B | `OnlineReplayBuffer` (default 8) | **added** |
| L_ada one step | `ada_adapt_step` on **LoRA params only** | **added** |
| Plan-act-adapt-replan | `AdaBlendPolicy.observe_and_adapt` | **added** |
| Goal latent z_g | Optional later; levels_up as sparse env signal only | not required for v0 |
| Full CEM multi-horizon | Later; program_search already exists | deferred |

## Legal alignment

| AdaJEPA property | Legal? |
|---|---|
| Learn from own transitions only | **Yes** |
| No expert demos required | **Yes** |
| No reward hacking / win hardcode | **Yes** — we use frame-change SSL |
| Test-time adapt per game | **Yes** — wipe buffer on new game_id |
| Frozen specialist answers | **No** — we don't add those |

## What we deliberately skip (for now)

1. Full JEPA pretrain stack rewrite  
2. Multi-horizon CEM with goal images  
3. Decoder visualization path  
4. Any need for 30B base model  

## How to use before / during training

1. **Offline train** still builds the base WM+heads (`train_lora.py`).  
2. **Deploy / collect** with `AdaBlendPolicy(adapt=True)` so each step does 1 LoRA update.  
3. Optional: more offline data *after* online-adapted rollouts (flywheel).

```python
from legal_search.train.ada_adapt import AdaBlendPolicy
pol = AdaBlendPolicy("legal_search/runs/blend_lora_v1_fullmine/best.pt", adapt=True)
a = pol.decide(frame, ...)
# after env step:
pol.observe_and_adapt(frame_before, action, frame_after, features_before, did_change=...)
```

## Citation

Wang et al., AdaJEPA: An Adaptive Latent World Model, 2026. https://arxiv.org/abs/2606.32026
