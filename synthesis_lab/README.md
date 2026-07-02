# Megarexus Synthesis Lab

Research lab workspace for synthesizing the imported ARCPrize3 engines:

- **LeWM/Rex**: action-side world model and action loop owner.
- **Contra**: thinking/control/informer engine around perception, packets, affordances, and solver loops.
- **Stochastic Goose**: stochastic ARC agent/runtime and fleet runner source.
- **ARC scorecard / environment fields**: goal-function evidence source (`levels_completed`, `score`, `level_scores`, `level_actions`).

## Lab role labels

Use these labels in notes, experiment manifests, and code comments:

- `LEWM_REX_ACTION_OWNER` — owns action-side world model / policy-planner behavior.
- `CONTRA_THINKING_ENGINE` — produces structured perception, affordance, and controller packets.
- `STOCHASTIC_GOOSE_RUNTIME` — stochastic action/runtime/fleet engine source.
- `GAMEOPS_SIGNAL_TRANSPORT` — shared transport/normalization/instrumentation layer.
- `ARC_SCORECARD_EVIDENCE` — only source for score/level-completion claims.
- `SCOUT_EXPERIMENT` — small/seeded/local probe, not a benchmark claim.
- `LIVE_SCORECARD_RUN` — live ARCPrize3 scorecard run.

## Directory map

```text
synthesis_lab/
  README.md
  manifests/        # explicit source/role/experiment manifests
  engine_adapters/  # glue code between LeWM/Rex, Contra, and Goose
  orchestration/    # runners that coordinate the engines
  experiments/      # new research experiments only
  notes/            # lab notes, hypotheses, result summaries
```

## Non-negotiable ownership language

- Say: **LeWM/Rex beat a level; Contra/Goose informed or proposed actions as configured; ARC scorecard measured evidence.**
- Do not say: Lewis/Contra/GameOps beat a level unless they directly owned the policy/action loop in that experiment.
- Always separate:
  - action owner,
  - thinking/informer engine,
  - runtime/orchestration,
  - evidence source.
```
