# Mining debt (honest, living)

**Gate:** open for JOB 2 after inventory sign-off (`MINING_COMPLETE.json`). Residual debt remains for future digs; see `USED_VS_UNUSED.md`.

## Progress

| Source | Legal mine | Status |
|---|---|---|
| PortBand | `ports.py` | **Deep** — ports, alignments, edge-lock, lineup, **fleet combos**, **event surfaces** |
| Vision | `shape_traces.py` + components | **Deep** — traces, contact graph, **color groups** (no roles) |
| StateCon | `statecon_graph.py` | **Deep** — working set, **branch graph**, **support groups** |
| GeoBand / Symmetry / Affordances / Delta / Explore | geometry_rel, local_symmetry, affordances, … | **Partial+** |
| Crucible | `crucible_mine.py` | **Deep** — v0, discrepancies, anomalies, rules, **falsify loop** |
| DW / sim3d | `dynamics.py` + `perception_trust.py` | **Deep** — DynamicsHypothesis families, **perception trust gate**, no fixtures |
| Loom / LeWM | `worldmodel.py` + `cem_plan.py` | **Deep** — affordance, consistency, **discrete CEM** (no weight req) |
| Match calibration | `calibrate.py` | **Deep** — reweight + **stage0 click rank** + **stage1 delta obs** |
| Ghost | `ghost_relations.py` | **Deep** — confirmed relations + lineup summary |
| Goose | `explore.py` | **Partial** — pure rebuild tabular rates |
| Quarantine / specialists / CNN-as-general / Puffer | boundaries | **Correct fences** |

## Still owed (residual, not gate-ready alone)

1. PortBand full Monte-Carlo fleet boundary sampling parity with 222k source (legal subset present)
2. Crucible full grammar atom offsets/effects exhaustiveness + corpus falsify package
3. DW online multi-family dynamics fit (gravity/flow) without SP80 paint roles
4. Optional real LeWM encoder weight load path (generic/online only) behind runtime lock
5. Match calibration transfer_prior / macroband legal subset (stage0/1 present)
6. Goose optional CNN online adapter when external runtime present

## Verification

```bash
python3 -B legal_search/smoke.py
python3 -B legal_search/mine_probe.py --arcade
python3 -B legal_search/tests/test_shaft_mines.py
```

`ready_for_gate=False` until residual debt is paid and A–E signed in `MINING_THEN_TRAIN.md`.
