# Controller Head API Map

Date: 2026-05-31
Scope: `codexlab/contra/controller_head.py`

## Active Callers

- `codexlab/contra/api.py` is the stable public API caller importing symbols from `controller_head.py`.
- Focused regression/dev-check scripts also import `controller_head.py` symbols to lock packet shapes.
- `codexlab/contra/contra.py` references `codexlab/contra/controller_head.py` only as source-path metadata.

## Symbols Imported By `contra/api.py`

- `CONTRA_CONTROLLER_REQUEST_SCHEMA`
- `CONTRA_CONTROLLER_RESPONSE_SCHEMA`
- `RAW_COLOR_COMPONENT_PACKET_SCHEMA`
- `ACTOR_AFFORDANCE_HYPOTHESIS_SCHEMA`
- `TARGET_AFFORDANCE_HYPOTHESIS_SCHEMA`
- `CONTRA_LAYER_OUTPUT_SCHEMA`
- `CONTRA_LAYER_OUTPUT_SET_SCHEMA`
- `oracle_actor_affordance_hypotheses`
- `oracle_target_affordance_hypotheses`
- `build_raw_color_component_packet`
- `calibration_actor_affordance_hypotheses`
- `calibration_target_affordance_hypotheses`
- `empty_actor_affordance_hypotheses`
- `empty_target_affordance_hypotheses`
- `run_contra_view`

## Regression-Only Imports

- `ContraControllerRequest`
- `CONTRA_RECOVERED_ERROR_PACKET_SCHEMA`
- helper modules: `controller_affordances`, `controller_error_packets`, `controller_geometry`, `controller_layer_outputs`

## Extraction Completed

Created `codexlab/contra/controller_geometry.py` for pure controller helpers:

- raw grid conversion
- component bbox math
- direction inference
- contact edge lookup
- open-region bbox search
- short deterministic digest helper

Created `codexlab/contra/controller_affordances.py` for:

- oracle affordance builders
- calibration affordance builders
- empty affordance builders
- affordance schema constants

Created `codexlab/contra/controller_error_packets.py` for:

- empty raw-color packet factory
- empty scene-surface packet factory
- empty symmetry packet factory
- recovered-error packet factory
- recovered/raw packet schema constants

Created `codexlab/contra/controller_layer_outputs.py` for:

- layer-output packet schema constants
- layer-output row construction
- layer-output set construction

`controller_head.py` imports and re-exports the same symbols used by `contra/api.py`, so the stable API surface remains unchanged.

## Remaining Split Candidates

- Remaining controller-head reductions should stay limited to orchestration helpers unless a new packet regression covers the moved behavior.
- `vision.py` and `portband.py` are now larger reduction targets than `controller_head.py`.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python -B codexlab/dev_check.py
```

Result: syntax parsing, both focused packet regressions, and import smoke passed.
