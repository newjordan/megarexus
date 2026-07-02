# CONTRA Ghost Relational Memory Architecture

Date: 2026-06-02
Status: architecture plan, not implementation evidence.
Scope: CONTRA relationship memory, with SP80 used as the current controlled test condition for ghost references, shared-offset relation scoring, and controlled-run verification.

## Executive Summary

CONTRA is missing a grounded memory layer between controlled action evidence and the next action recommendation. The current system can observe useful PortBand, DeltaBand, ActionModBand, edge, angle, and flow signals, but it often treats them as packet-local facts. That forces later rounds to rediscover relationships instead of carrying forward confirmed structure.

The proposed architecture adds `Ghost`: a past-evidence memory system built from controlled completed levels. Ghost is the opposite of Oracle. Oracle is speculative and future-facing. Ghost is historical and grounded: it stores what was actually true at a verified prior completion boundary, then provides relationship hints for later levels.

The key design principle is:

```text
past controlled win -> confirmed relationship memory -> current-frame relational weighting -> ranked primitive
```

Not:

```text
past controlled win -> replay route / copy coordinates / hardcode solution
```

For SP80 level 2, the intended enhancement is not to encode the known answer. The intended enhancement is to let CONTRA recognize that a source/nozzle object, flow axis, edge alignment, and shared offset can jointly satisfy multiple lower objects or goals. The scoring should prefer transforms that maximize a coherent relationship graph.

## Problem Statement

Recent controlled runs show that CONTRA can emit positive relationship signals and the harness can execute recommendation-backed primitives. However, failures remain in three places:

1. **Confirmed relationships are not promoted.**
   A prior action can confirm an edge, angle, flow direction, or paint-response relationship, but the next recommendation packet does not carry that confirmation as durable evidence.

2. **Single-row ranking underweights confluence.**
   The scorer can rank an individual PortBand row, but it does not adequately reward one transform that improves several relationships at once.

3. **Prior wins are not consulted as grounded relation references.**
   A completed level contains a final frame and final relationship graph that should be usable as historical evidence. Today it is not converted into a reusable memory artifact.

The system therefore fails to exploit the most important evidence class: relationships and angles that were already confirmed by controlled gameplay.

## Architectural Goals

- Capture controlled win states as grounded relationship memory.
- Promote confirmed action-response relationships into reusable memory rows.
- Compare current relationship graphs to prior winning relationship graphs.
- Reward shared transforms and shared offsets that satisfy multiple object/goal relationships.
- Preserve fail-closed truth boundaries: no ghost row is solve truth, control truth, or action authority.
- Keep the harness connected to CONTRA through ranked primitives from `CONTROL_PRIMITIVES`.
- Make every use of ghost evidence visible in packets, logs, reports, and verifiers.

## Non-Goals

- Do not hardcode level-specific coordinates, object indexes, or action counts.
- Do not replay known routes.
- Do not treat known solution files as training labels for action choice.
- Do not use Ghost as Oracle or future truth.
- Do not claim level 2 is solved until a controlled run records `levels_completed_delta > 0`.
- Do not make SP80-only paintflow the generic CONTRA model. SP80 paintflow is a specialization of relationship memory.

## Core Concept: Ghost

`Ghost` is a verified, historical reference built from a controlled completion boundary.

It stores:

- the final raw frame of a completed level
- the final CONTRA packet
- selected action memory leading to completion
- confirmed relation rows from before/action/after evidence
- completion proof from engine state
- source run provenance and verifier status

It does not store:

- an action script to replay
- coordinates to copy into a later level
- a solved-plan assertion
- future goal truth

Ghost answers this question:

```text
Which relationships were true at a prior controlled win, and which current
candidate transforms make the current board more like that relationship graph?
```

## Core Concept: Confirmed Relationship Memory

Confirmed relationship memory is a promotion layer over controlled before/action/after evidence.

The promotion source is:

- `before_contra_packet`
- selected recommendation row
- executed primitive
- `after_contra_packet`
- `DeltaBand`
- `ActionModBand`
- episode memory
- engine completion delta

The promoted memory row should record:

```json
{
  "schema": "codexlab.contra_confirmed_relation_memory.v1",
  "relation_memory_id": "<stable-id>",
  "source_run_id": "<run-id>",
  "source_packet_ids": ["..."],
  "game_id": "sp80-589a99af",
  "level_index": 0,
  "round_index": 13,
  "relation_family": "edge_alignment|flow_axis|shared_offset|support_bridge|activation_survived",
  "object_role_refs": {
    "source": ["..."],
    "targets": ["..."],
    "supports": ["..."]
  },
  "geometry": {
    "flow_axis": "vertical|horizontal|unknown",
    "flow_direction": "up|down|left|right|unknown",
    "edge_alignment_kind": "centered_dual_port_projection|single_port_projection|...",
    "shared_transform_xy": [0, 0],
    "angle_error": 0.0,
    "offset_error": 0.0
  },
  "confirmation": {
    "confirmed_by": ["deltaband", "actionmodband", "levels_completed_delta"],
    "levels_completed_delta": 1,
    "survived_activation": true,
    "confidence_score": 1.0,
    "confidence_is_probability": false
  },
  "truth_boundary": {
    "control_truth_admitted": false,
    "goal_truth_admitted": false,
    "solve_truth_admitted": false,
    "action_choice_admitted": false
  }
}
```

## Core Concept: Shared Offset Relation

A shared offset is a candidate transform that improves more than one relationship at the same time.

For SP80, this often means:

- a source/nozzle flow relation
- an edge alignment relation
- a lower-object support relation
- a multi-goal paint-path relation

The generic abstraction is:

```text
one transform -> many relationship improvements
```

The scorer should prefer candidates where a single transform creates confluence:

- source edge aligns with one or more target/support edges
- flow axis stays orthogonal and coherent
- multiple target objects become compatible under the same offset
- activation would preserve or extend the confirmed relationship graph

The emitted row should look like:

```json
{
  "schema": "codexlab.contra_shared_offset_relation_candidate.v1",
  "candidate_id": "<stable-id>",
  "source": "current_relationship_graph",
  "evidence_sources": [
    "portband.lineup_matrix.rows",
    "action_interaction_matrix",
    "confirmed_relation_memory",
    "ghost_pass"
  ],
  "object_refs": {
    "source_object_ids": ["..."],
    "target_object_ids": ["..."],
    "support_object_ids": ["..."]
  },
  "shared_transform_xy": [0, 0],
  "flow": {
    "orthogonal_flow_lock": true,
    "flow_axis": "vertical|horizontal",
    "flow_source_role": "nozzle|source|actor_origin|unknown"
  },
  "relationship_scores": {
    "edge_alignment_count": 0,
    "shared_offset_match_count": 0,
    "multi_target_confluence_score": 0.0,
    "ghost_match_score": 0.0,
    "confirmed_relation_preservation_score": 0.0,
    "contradiction_penalty": 0.0,
    "value_score": 0.0
  },
  "truth_boundary": "relationship_candidate_not_action_truth"
}
```

## System Architecture

### Stage 1: Capture

When a controlled episode records `levels_completed_delta > 0`, the harness writes a ghost artifact:

```text
codexlab/runs/<run_id>/ghost.json
```

Required fields:

- schema
- created time
- game id
- level index
- source run dir
- verifier status
- final raw frame path/hash
- final CONTRA packet hash/path or embedded packet ref
- completion proof
- action memory summary
- confirmed relationship rows
- truth boundaries

Capture happens only after a controlled run. Static packets cannot create Ghost.

### Stage 2: Extract

`ghost_extract_relations(...)` converts final packets and action memory into stable relationship rows.

Extractors:

- flow-axis extractor
- edge-alignment extractor
- shared-offset extractor
- activation-survival extractor
- object-role/shape-role extractor
- contradiction extractor

Extraction output is relationship memory, not route memory.

### Stage 3: Load

At the start of a later level or action round, the harness loads compatible ghosts.

Compatibility checks:

- same game id
- controlled completion proof present
- verifier passed
- compatible object families or shape roles
- compatible flow/source family when known
- no stale or invalid artifact status

Compatibility does not require same level index, same object count, or same coordinates.

### Stage 4: Compare

`ghost_pass` compares current relationship packets against ghost relations.

Inputs:

- current `run_contra_view(...)` packet
- current PortBand/action-state rows
- current action memory
- loaded Ghost rows

Outputs:

- `ghost_edge_alignment_match`
- `ghost_flow_axis_match`
- `ghost_shared_offset_match`
- `ghost_missing_relation`
- `ghost_relation_delta`
- `ghost_candidate_transform_xy`

Every row must include:

- current evidence refs
- ghost evidence refs
- similarity score
- contradiction score
- boundary flags

### Stage 5: Score

The recommendation packet consumes ghost and shared-offset signals as value modifiers.

High value:

- current transform preserves confirmed flow axis
- current transform improves multiple edge/offset relations
- current transform matches a prior winning relationship shape
- current transform extends the prior relationship graph to more targets

Low value:

- transform breaks a confirmed relation
- transform repeats a known failed relation
- transform optimizes a single noisy PortBand row while harming confluence
- transform depends on coordinate replay

The scorer should compute:

```text
value =
  current_portband_score
  + confirmed_relation_preservation
  + shared_offset_confluence
  + ghost_relationship_match
  - contradiction_penalty
  - stale_loop_penalty
```

The result remains a ranked primitive recommendation, not action truth.

## Integration Points

### `codexlab/sp80_player_harness.py`

Responsibilities:

- write `ghost.json` after controlled completion
- load compatible ghosts before building recommendation packets
- include ghost refs in `brain_calls.jsonl` and `packets.jsonl`
- never replay ghost actions

### `contra.run_contra_view(...)`

Responsibilities:

- optionally accept past relationship context
- expose `ghost_pass` rows in the packet
- preserve `config_pass`, `oracle_pass`, `contra_pass`, and `view_system_passes`

### PortBand / Action Matrix

Responsibilities:

- emit shared-offset candidates
- attach source/target/support object refs
- score multi-target confluence
- identify contradictions against confirmed memory

### Episode Verifier

Responsibilities:

- verify ghost load source is a controlled completed run
- verify no replay/coordinate-copy fields are used for action choice
- verify ghost evidence appears in recommendation rows when it affects ranking
- verify all boundaries remain false for control/goal/solve/action truth

## Rollout Plan

### Phase 0: Design Lock

Deliverable:

```text
codexlab/docs/CONTRA_GHOST_RELATIONAL_MEMORY_ARCHITECTURE_2026-06-02.md
```

Gate:

- accepted boundary: Ghost is historical evidence, not Oracle and not replay
- accepted target: relational weighting and optimals, not known-solution coding

### Phase 1: Ghost Capture

Implement:

- `ghost.json` writer on completed controlled episodes
- relation-memory extraction scaffold
- verifier for ghost artifact provenance

Gate:

- run level 0
- complete level
- verify ghost exists
- verify ghost has completion proof
- verify no action replay fields are admitted

Success:

```text
controlled completion -> verified ghost.json
```

### Phase 2: Ghost Loader

Implement:

- load latest compatible ghost
- attach ghost summary to recommendation request
- log loaded/not-loaded reason

Gate:

- run next level with ghost loading enabled
- report loaded ghost count
- verifier confirms ghost source was controlled completion

Success:

```text
verified prior win ghost appears in later level packet context
```

### Phase 3: Ghost Pass

Implement:

- relationship graph comparison
- ghost match/missing/contradiction rows
- review-page section for ghost evidence

Gate:

- static current frame plus loaded ghost produces `ghost_pass`
- pass rows cite both current and ghost evidence
- no solve/control/action truth admitted

Success:

```text
current frame -> ghost relationship comparison rows
```

### Phase 4: Shared Offset Candidate Rows

Implement:

- detect transforms that improve multiple edge/flow/target relations
- emit `shared_offset_relation_candidate` rows
- integrate with PortBand/action interaction matrix

Gate:

- level 2 emits shared-offset candidates from current relationships
- candidates cite object refs and transform hypotheses
- no known route/action-count/coordinate solution is encoded

Success:

```text
one transform -> multiple relationship improvements -> ranked candidate row
```

### Phase 5: Recommendation Scoring Integration

Implement:

- boost candidates preserving confirmed relationships
- boost candidates matching ghost relationship structure
- boost shared-offset confluence
- penalize contradiction and stale loops

Gate:

- recommendation packet shows score terms
- selected primitive is still from `CONTROL_PRIMITIVES`
- verifier confirms ghost evidence is a modifier only

Success:

```text
relationship memory changes ranking without becoming action authority
```

### Phase 6: Controlled Runs

Run:

- level 0 controlled run to create ghost
- level 1 controlled run with ghost enabled
- level 2 controlled run with ghost enabled

Report:

- review pages
- raw artifacts
- verifier output
- ghost loaded/not-loaded status
- shared-offset candidates
- selected primitive timeline
- levels completed delta

Success for this phase is not necessarily solve. First success is:

```text
ghost created -> ghost loaded -> relation rows emitted -> ranking affected -> verifier passed
```

## Verification Matrix

| Requirement | Verification |
| --- | --- |
| Ghost only from controlled completion | `ghost_verify.py` checks `levels_completed_delta > 0` and source verifier status |
| No route replay | verifier rejects action-list/admitted coordinate fields in ghost scoring path |
| No Oracle conflation | packet separates `oracle_pass` and `ghost_pass` |
| Current evidence still primary | recommendation rows include current PortBand/action-state refs |
| Ghost used as modifier only | `action_choice_admitted=False`, `control_truth_admitted=False` |
| Shared offset is relational | rows cite multiple object refs and shared transform, not action sequence |
| Reports inspectable | run has `index.html`, `packets.jsonl`, `brain_calls.jsonl`, verifier output |

## Failure Modes And Controls

### Failure: Known Solution Leakage

Risk:

- algorithm silently encodes level-specific action counts or coordinates.

Control:

- verifiers reject coordinate-copy and route-replay fields.
- docs and packet boundaries label known-solution artifacts as calibration/historical only.

### Failure: Ghost Becomes Oracle

Risk:

- ghost rows are treated as future truth.

Control:

- `ghost_pass` rows carry `past_evidence_not_future_truth`.
- Oracle and Ghost are separate packet sections.

### Failure: Single Noisy Relationship Dominates

Risk:

- a single PortBand row overpowers a multi-object graph.

Control:

- score confluence and contradiction explicitly.
- require shared-offset rows to cite multiple relationship improvements.

### Failure: Stale Loop With Ghost Enabled

Risk:

- ghost repeatedly boosts a wrong current candidate.

Control:

- ActionModBand feedback demotes ghost matches that fail to change the expected relation.
- after two same-basin failures, report `stale_loop` and require representation change.

### Failure: Overfitting SP80 Paintflow

Risk:

- Ghost becomes a paint-only trick.

Control:

- keep schema generic: source, target, support, transform, edge, flow, relation.
- SP80-specific flow terms stay optional specializations.

## Acceptance Criteria

### Architecture Acceptance

- This document is present and names the Ghost boundary.
- It distinguishes Ghost from Oracle.
- It defines relationship-memory and shared-offset contracts.
- It defines verification and non-goals.

### Phase 1 Acceptance

- A controlled level completion writes `ghost.json`.
- The ghost verifier passes.
- The artifact contains final frame, final packet refs, relation rows, and completion proof.

### Phase 2 Acceptance

- A later run loads the ghost.
- Packets record why the ghost was compatible.
- No action replay fields are consumed.

### Phase 3 Acceptance

- `ghost_pass` emits relation comparison rows.
- Review surface shows current-vs-ghost relationship evidence.

### Phase 4 Acceptance

- Shared-offset candidates appear on level 2 from current relationship geometry.
- Rows cite multiple object/relationship refs.

### Phase 5 Acceptance

- Recommendation scores change because of confirmed relation memory and shared-offset confluence.
- Verifier confirms action choice remains recommendation-backed and bounded.

### Phase 6 Acceptance

- Controlled run artifacts exist.
- Verifiers pass.
- Report clearly states whether levels completed increased.

## First Implementation Slice

Do not begin with level 2 route behavior. Begin with the memory substrate.

1. Add `codexlab/contra/ghost.py`.
2. Add ghost schema constants and extraction helpers.
3. Add ghost write on controlled episode completion.
4. Add `codexlab/contra_verify_ghost.py`.
5. Run level 0 to create a ghost.
6. Verify the ghost.
7. Only then add ghost loading and scoring.

The first implementation milestone is:

```text
verified controlled win ghost exists and can be inspected
```

not:

```text
level 2 solved
```

## Final Boundary

Ghost is grounded memory. It is allowed to say:

```text
This current relationship resembles a relationship that survived a prior
controlled win.
```

It is not allowed to say:

```text
This action solves the current level.
```

Only the engine and controlled verifier can say that.
