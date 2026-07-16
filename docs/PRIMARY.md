# PRIMARY — one brain

**Date:** 2026-07-16  
**Single brain:** **MegaMind** dual-hemisphere (I dream · E judge · C commits)

## Open this first

| What | Path |
|---|---|
| **Brain package** | `brain/` (public) → `synthesis_lab/orchestration/rdl/brain/` |
| **Status / pad** | `synthesis_lab/orchestration/rdl/brain/STATUS.md` |
| **Train flywheel** | `python -u -B synthesis_lab/orchestration/rdl/brain/train_flywheel.py` |
| **Smoke all seeds** | `python -B synthesis_lab/orchestration/rdl/brain/smoke_all_ideas.py` |
| **Math of world capture** | [`docs/MATHEMATICS_OF_WORLD_CAPTURE.md`](MATHEMATICS_OF_WORLD_CAPTURE.md) (for mathematicians) |

## Doctrine

1. **One brain** — MegaMind. Not five primaries.  
2. **Plant everything useful** on the operator pad; training raises what works.  
3. **Multi-game level groups** (24 envs) — robust patterns, not SP80-only.  
4. **Keep only if better** on private multi-game score (Weco-aligned).  
5. **Fail-closed truth** — env grades; no spill / win-bank / human-route labels.  
6. **Illegal stays out** — `_invalid/`, spill models, answer-key solvers.

## What is *not* the brain

| Lane | Status |
|---|---|
| Specialist SP80/ft09 scorecard policies | Evidence / suite tools only (`codexlab/`) |
| `megarexus_cnn` | Legacy experiment; not the primary train path |
| `legal_search` algorithms | **Integrated** as operator backends (keep source; no run bank) |
| Flat `orchestration/*.py` collectors | Lab tools; not entrypoints |
| Dated HANDOFF notes | Archive only |

## Runtime deps the brain still uses

- `legal_search/algorithms/*` + features (geometry, crucible, …)  
- `codexlab/contra_head` (full CONTRA view)  
- `synthesis_lab/experiments/other_games/environment_files` (24-game suite)  
- Optional: sim3d pour/body plugs (degrade if absent)

## Cleanup rule

Delete only if:

- **illegal / quarantined**, or  
- **already integrated** into the MegaMind pad/operators to the needed optimum, or  
- **pure run artifact** (`.pt`, `runs/`, logs, cycles).
