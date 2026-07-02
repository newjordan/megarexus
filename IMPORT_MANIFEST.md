# megarexus engine import manifest

Imported for ARCPrize3 thinking/running engines. Historical run outputs, caches,
old tests, smoke/verifier/regression files were intentionally excluded.

## Sources

- `/home/frosty40/rexus/codexlab` -> `codexlab/`
  - ARCPrize3 runners/harnesses
  - Contra thinking/control engine (`contra_head/`, `contra_loop/`, `contra_service/`)
  - LeWM/Rex bridge/player/env code
- `/home/frosty40/rexus/architecture` -> `architecture/`
  - LeWM/LOOM architecture notes/source/upstream bridge
- `/home/frosty40/Loom-LeWM-Only/lewmtrain` -> `lewmtrain/`
  - LeWM training/trajectory build scripts only; data and run logs excluded
- `/home/frosty40/arc-proof/stochastic-goose` -> `stochastic_goose/`
  - Stochastic Goose agent/runtime files; run logs excluded
- `/home/frosty40/rexus/corpus` -> `corpus/`
- `/home/frosty40/rexus/environment_files` -> `environment_files/`

## Role boundaries

- LeWM/Rex owns action-side world-model / policy-planner operation.
- Contra/Stochastic Goose are thinking/running engines around action selection.
- ARC scorecard/live environment fields remain the goal-function evidence.
