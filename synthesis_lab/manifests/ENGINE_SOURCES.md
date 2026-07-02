# Engine source manifest

## Imported engine roots

- `../codexlab/contra.py`
- `../codexlab/contra_head/`
- `../codexlab/contra_loop/`
- `../codexlab/contra_service/`
- `../codexlab/lewm_bridge/`
- `../codexlab/players/`
- `../architecture/`
- `../lewmtrain/`
- `../stochastic_goose/`

## Synthesis target

Create a research-only integration layer where:

1. Contra can emit perception/affordance/controller packets.
2. Stochastic Goose can propose stochastic actions or policy candidates.
3. LeWM/Rex remains the action-side world model / policy-planner owner when used as owner.
4. ARC scorecard/local environment fields remain the evidence layer.

## Old files policy

Do not copy old tests, historical run dumps, pod returns, cache files, or stale smoke/regression scripts into this lab. New experiments go under `../synthesis_lab/experiments/`.
