# CONTRA Frame Debug Template Handoff

Date: 2026-05-29

## Why this exists

The current failure mode is presentation, not just computation. CONTRA/SP80/DeltaBand review pages have drifted into table-first summaries. That is unusable for debugging visual state systems.

The standard template must be frames-first and system-layer-first.

## Non-negotiable rule

For CONTRA, SP80, Oracle, GeoBand, DeltaBand, symmetry, StateCon, and paint-fill work:

1. Put large frame evidence before all metrics.
2. Show one visual output per system/layer being discussed.
3. Do not make the user infer a system result from raw JSON or small thumbnails.
4. Put numeric tables after the visual deck.
5. If a requested system/layer has no frame output, the page must show an explicit missing-frame slot for that system.

## Required Page Order

1. Title, exact condition, command, run directory.
2. Boundary statement: controlled/static/proxy and what is not admitted.
3. Frame debug deck.
4. Metrics and per-level calculation details.
5. Raw artifacts and provenance.

## Required Frame Debug Deck

For controlled before/action/after work, every level needs:

- Raw before frame.
- Raw after frame.
- Final action frame.
- Full action rollout contact sheet.
- Oracle before overlay when Oracle is in scope.
- Oracle after overlay when Oracle is in scope.
- GeoBand before overlay.
- GeoBand after overlay.
- GeoBand primary field before/after.
- GeoBand boundary queue before/after.
- DeltaBand relationship delta sheet.
- DeltaBand pixel delta sheet.

If symmetry is in scope:

- Symmetry group overlay.
- Reverse/polar/mirror subgroup overlay.
- Lock/key sensor overlay.

If paint fill is in scope:

- Paint source/actor overlay.
- Fill frontier overlay.
- Target/goal overlay.
- Before/after fill delta.

## Image Size Standard

- No thumbnails for primary debug frames.
- Use two columns maximum for full-board images on desktop.
- Each primary board image should be inspectable at normal browser zoom.
- Avoid raster text covering objects. Prefer HTML titles above images.
- Any raster labels must be large and placed in unused margins or non-critical areas.

## Current Corrected Runner

`codexlab/sp80_deltaband_controlled_run.py` now writes:

- `before_geoband_artifacts`
- `after_geoband_artifacts`
- `deltaband_relationship_delta_sheet`
- `deltaband_pixel_delta_overlay`
- `action_sequence_sheet`
- Oracle before/after overlays

Latest generated corrected run:

`codexlab/runs/contra_sp80_deltaband_rollout_00_03_geoband_deltaband_frames_20260529T155514Z/`

## Known Template Gap

This is still embedded in one runner. Claude should extract a reusable standardized frame-debug template/generator so all future CONTRA/SP80/paint pages use the same ordering and required frame slots.

Suggested target:

`codexlab/contra_frame_debug_review_template.py`

The generator should accept a manifest/run directory and fail closed if required frame slots for the requested systems are missing.
