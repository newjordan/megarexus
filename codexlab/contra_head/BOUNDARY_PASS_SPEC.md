# CONTRA Boundary-Pass Spec

CONTRA boundary passes are fail-closed relationship tests. A boundary pass is
not a natural-language question and does not admit control truth.

## Contract

1. The first GeoBand pass includes all visible relationship objects available to
   the current StateCon.
2. Later passes refine that inclusive field into subgroups.
3. A refinement pass may exclude an object from a subgroup, but it may not erase
   the object from the primary GeoBand field.
4. Every excluded primary object must create an unresolved boundary row with a
   reason code.
5. Every `Y` object is a known object with an unresolved attachment. It must
   record `base_shape_family`, `unknown_attachment_basis`, and
   `unknown_attachment_marker: Y`.
6. Boundary rows remain unresolved until reviewed or tested through a controlled
   DeltaBand action boundary.
7. No boundary pass, boundary queue, StateCon, or GeoBand row is control truth,
   goal truth, action choice, or solve truth.

## First-Class Pass Rows

Each GeoBand pass row uses `codexlab.geoband_boundary_pass.v1` and records:

- `pass_id`
- `pass_order`
- `policy`
- `input_object_ids`
- `output_group_ids`
- `unresolved_boundary_ids`
- `reason_codes`
- `admission_status`

The current required pass order is:

1. `geoband_primary_relationship_group`
2. `geoband_subgroup_refinement`

## Boundary Queue

Each StateCon carries a `geoband_boundary_queue` with unresolved boundary rows.
The current boundary test types are:

- `shape_equivalence_unknown`
- `shape_plus_y_attachment`
- `group_asymmetry_unresolved`
- `independent_vs_compound_unknown`
- `actor_target_support_role_unknown`
- `geometry_relationship_unresolved`

These are boundary tests, not prompts. They define what remains unresolved in
the current visual relationship field.

## Human Oversight Boundary

The static viewer can surface candidate objects, passes, subgroups, and
unresolved boundaries. It cannot complete the manual SP80 calibration table or
the DeltaBand controlled event run without human review and action-boundary
evidence.

The viewer must therefore carry two fail-closed oversight packets:

- `codexlab.sp80_manual_calibration_review.v1`: per-level review table with
  seed candidates and empty `human_reviewed_expectations` until a human reviews
  actor ids, target ids, support/action ids, connector ids, known-Y ids,
  unresolved ids, and the image hash.
- `codexlab.deltaband_controlled_event_requirement.v1`: per-level controlled
  event requirement with all before/action/after artifacts marked missing until
  an actual controlled action boundary is recorded.
