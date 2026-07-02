# CONTRA / Oracle Handoff - 2026-05-27

## Current State

Active repo: `/home/frosty40/rexus`

Current pushed commit at handoff: `a2b2bf3`

Latest meaningful commit:

```text
a2b2bf3 Add CONTRA AR25 target smoke artifact
```

This commit preserved the first cross-game smoke artifact where CONTRA/Oracle
seeded an actor/target relation and GeoBand rendered a target relationship on
AR25 level 5.

Network viewer from the last checked run:

```text
http://192.168.1.176:8766/codexlab/runs/contra_single_game_test_viewer/ar25_level_05/index.html
```

Primary evidence artifacts:

```text
codexlab/runs/contra_api_single_cross_game_ar25_latest/manifest.json
codexlab/runs/contra_api_single_cross_game_ar25_latest/packets/cross_game_no_calibration_ar25_level_05.json
codexlab/runs/contra_single_game_test_viewer/ar25_level_05/index.html
codexlab/runs/contra_single_game_test_viewer/ar25_level_05/assets/statecon_geoband_overlay.png
```

Last verification before commit:

```text
python3 codexlab/contra_api_verify.py codexlab/runs/contra_api_single_cross_game_ar25_latest --require-probes
```

Result:

```text
CONTRA API verifier passed
```

## Non-Negotiable Boundaries

CONTRA is the unified game assessment/control/communication system. It is
called once through `contra.run_contra_view(...)`, and the whole network should
fire together:

1. raw vision packet
2. control isolation packet
3. scene surfaces
4. actors, targets, supports
5. StateCon
6. GeoBand
7. boundary queues
8. DeltaBand gate
9. debug summary
10. layer handoffs
11. fail-closed admission result

Oracle is not CONTRA. Oracle is a sidecar assumption head. It may supply one
small missing-element hypothesis at a time so downstream review output is not
empty, but it must not promote assumptions into control truth.

Static hypotheses are never:

- control truth
- goal truth
- action choice
- solve condition
- gameplay completion evidence

GeoBand is a relationship informer, not a predictor. It records static geometry
and relationship bandwidth. It does not decide what action will work.

DeltaBand only becomes valid after a controlled before/action/after boundary.
For cross-game static frames, DeltaBand should normally skip fail-closed.

## Current Implementation Anchors

Core CONTRA modules:

```text
codexlab/contra/controller_head.py
codexlab/contra/perception.py
codexlab/contra/vision.py
codexlab/contra/boundary_pass.py
codexlab/contra/deltaband.py
codexlab/contra/debug.py
codexlab/contra/api.py
```

Oracle:

```text
codexlab/oracle.py
```

API verifier and matrix runner:

```text
codexlab/contra_api_matrix.py
codexlab/contra_api_verify.py
codexlab/contra_api_smoke.py
```

Human review viewer:

```text
codexlab/contra_cross_game_smoke_calibration.py
```

Paint game calibration and viewers:

```text
codexlab/sp80_surface_calibration.json
codexlab/sp80_color_board_viewer.py
codexlab/sp80_verify_color_board_surfaces.py
codexlab/sp80_manual_calibration_table.py
codexlab/sp80_verify_manual_calibration_table.py
codexlab/sp80_deltaband_controlled_run.py
codexlab/sp80_verify_deltaband_run.py
```

## What Worked

CONTRA v0.01 criteria are now represented by the API matrix/verifier:

- one call fires the whole network
- normal bad input does not hard-crash the run
- inactive layers explain why they are inactive
- outputs are inspectable as JSON plus viewer artifacts
- static hypotheses stay `not_admitted`
- graceful fail-closed behavior is visible in `layer_output_packets`

On AR25 level 5, the fresh single-game run produced:

```text
actors: 1
targets: 1
supports: 1
statecons: 1
geobands: 1
deltaband_gate: skipped fail-closed
admission_status: not_admitted
```

This is important because it means the current Oracle bridge can seed enough
actor/target structure for StateCon and GeoBand to render a target relationship
outside the paint game. It is still a hypothesis artifact, not truth.

## Known Issues

The AR25 viewer target text label clips at the right edge. The packet is
present; this is a viewer label-placement problem.

Some cross-game samples still produce raw vision and surfaces but no
actor/target. That can be correct fail-closed behavior, but each case should be
visually reviewed before calling it a system gap.

The latest single-game matrix runner still includes SP80 paint packets and
probes because `contra_api_matrix.py --cross-game-limit 1` runs the standard
matrix shape. This is acceptable for evidence, but a narrower "one packet only"
runner may be worth adding later.

Oracle can currently guess actor/goal from shape, asymmetry, location, color,
and lock/key fit signals. These are weak hypotheses. Do not treat them as
calibration.

GeoBand currently activates when actor, target, and support structures are
available. Missing support/action attachment remains a common reason for
inactive or partial networks.

## Next Session Focus

Return to SP80 paint first.

The next work should be subtle additions, not a broad rewrite:

1. Improve viewer label placement.
   Keep labels inside image bounds so target/actor/support text does not clip
   on right or bottom edges.

2. Strengthen Oracle as a sidecar, one assumption at a time.
   Add only one small rule per missing element. Keep each rule review-only and
   fail-closed.

3. Build paint-level calibration pressure back into CONTRA.
   Use the six SP80 paint levels as the main correctness target before claiming
   generalization.

4. Refine actor/target/support admission order.
   Preserve the ladder:
   vision isolation -> component admission -> actor/target/support hypotheses
   -> StateCon -> GeoBand -> DeltaBand -> verifier/admission result.

5. Keep GeoBand network-shaped.
   Goal/target, actor, support/action objects, and actor-to-actor relationships
   all need relational edges where applicable.

6. Add controlled DeltaBand only when a real event boundary exists.
   For SP80 this means using `sp80_deltaband_controlled_run.py`, not static
   frame guesses.

## Suggested First Commands

Check repo state:

```bash
git status -sb
git log --oneline -5
```

Reverify current matrix:

```bash
python3 codexlab/contra_api_verify.py codexlab/runs/contra_api_v001_matrix --require-probes
```

Reopen latest AR25 viewer:

```text
http://192.168.1.176:8766/codexlab/runs/contra_single_game_test_viewer/ar25_level_05/index.html
```

Regenerate full v0.01 matrix only after source changes:

```bash
python3 codexlab/contra_api_matrix.py --out-dir codexlab/runs/contra_api_v001_matrix
python3 codexlab/contra_api_verify.py codexlab/runs/contra_api_v001_matrix --require-probes
```

For paint-focused DeltaBand:

```bash
python3 codexlab/sp80_deltaband_controlled_run.py --levels all --out-dir codexlab/runs/contra_v001_sp80_deltaband_controlled_run --scale 12
python3 codexlab/sp80_verify_deltaband_run.py codexlab/runs/contra_v001_sp80_deltaband_controlled_run
```

## Important Language

Use these words carefully:

- `hypothesis`: allowed for static CONTRA/Oracle output.
- `assumption`: Oracle-only review placeholder.
- `not_admitted`: required until a verifier/admission gate accepts it.
- `fail_closed`: layer skipped or withheld output for a named reason.
- `controlled observation`: only after a real before/action/after run.

Avoid these words unless the exact condition has been run and recorded:

- solved
- learned
- understands
- truth
- action choice
- prediction
- gameplay completion

## Clean Handoff Summary

We have a coherent CONTRA v0.01 control path and a separate Oracle sidecar.
The system can now run one API call, expose inactive layers instead of
silently failing, and render a cross-game target relationship as review-only
GeoBand evidence. The next session should return to SP80 paint, add small
Oracle/CONTRA refinements, regenerate artifacts, and verify that the full
network still fires gracefully.
