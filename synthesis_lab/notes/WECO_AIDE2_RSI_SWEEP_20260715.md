# Weco AIDE² recursive self-improvement — garden synthesis

**Source:** [AIDE²: First Evidence of Recursive Self-Improvement](https://www.weco.ai/blog/first-evidence-of-recursive-self-improvement) (Weco, 2026-07-14)  
**Companion:** [4 Levels of RSI](https://www.weco.ai/blog/4-levels-of-recursive-self-improvement)

**Doctrine:** plant every transferable idea as base-model seeds (W1–W32).  
Training + multi-game level groups raise what works. **We do not claim megarexus is Level-1 RSI** — W5 scores `claimed_for_megamind=0`.

## Core architecture (planted)

| AIDE² idea | Garden mapping |
|---|---|
| **Bi-level** outer optimizes inner harness | Outer: Central LoRA + Ada + level-group raise schedule; Inner: I + operator garden + dream |
| **Keep only if better** (~9/10 reject) | Fail-closed gates; private env metric beats public loss |
| **Public / private scores** | Public: train loss / propose scores; Private: `levels_completed`, E-judge, env truth |
| **Fixed cost budget** | Fixed steps/epochs per cohort; efficiency > brute force |
| **Task heterogeneity** | **24-game L0–L1 interleaved** (already running) |
| **1st-order gen** | Hold on held-out within task / private split |
| **2nd-order gen** | Hold across games never tuned singly |
| **3rd-order / ignition** | Improved inner becomes better outer (test defined; not claimed) |

## RSI ladder (W4–W7)

| Level | Name | Meaning |
|---|---|---|
| 0 | Delegation | Autonomous loop, may be slower than human R&D |
| 1 | Net positive | Beats fair human baseline under fixed budget + multi-step trend + generalization |
| 2 | Ignition | Improves its *ability to improve* (inner → better outer) |
| 3 | Inflection | Acceleration at fixed budget |

AIDE2 reports **Level 1**. Ignition test: weak / not claimed.

## Search ideas discovered (W23–W28)

- Diverse drafts → bandit arms on draft lineages  
- Greedy selection **inside** lineage  
- **Fork-on-stall** of global best under fresh strategy  
- **Minimal per-operator context** (~16× compression) → reinvest as more steps  
- MCTS / island GA / majority vote / fancy judges mostly **rejected** under fixed budget  

## Anti-reward-hacking (W19–W22)

Three layers (emergent under private selection):

1. Anti-overfit instructions every stage  
2. Hard-coded guards (maps to TruthFlags / spill-winbank fences)  
3. Statistical peer rejection of extreme “successes”  

Aligned with CONTRA fail-closed and env-only grading.

## Catalog

`synthesis_lab/orchestration/rdl/brain/operators/weco_rsi.py` → **W1–W32**  
Wired into `OP_FAMILIES` → 16-D slots → central pad.

## How this ties to multi-game raise

```text
Outer (raise schedule):
  level group L0–L1 on all 24 games interleaved
  keep checkpoint only if pattern_robust_frac / multi-game private holds

Inner (per batch):
  MegaMind observe + train_transition
  operator garden + pattern growth + Weco RSI seeds on pad
```

This is the megarexus instantiation of **task heterogeneity + fixed budget + private selection**.
