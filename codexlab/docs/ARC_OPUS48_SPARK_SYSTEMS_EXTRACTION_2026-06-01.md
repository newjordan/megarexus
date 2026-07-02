# ARC Opus 4.8 Spark Systems Extraction

Date: 2026-06-01
Status: systems/process/guideline extraction from public recordings and Spark
subagent mining.

## Boundary

This document is hypothesis and design guidance only. It is not local CONTRA
control truth, goal truth, action choice, prediction truth, or solve evidence.

Inputs:

```text
codexlab/runs/arc_public_recordings_anthropic-opus-4-8-high_20260601T222454Z/download_manifest.json
codexlab/runs/arc_public_recordings_anthropic-opus-4-8-high_20260601T222454Z/raw_recordings/
codexlab/runs/arc_opus48_playthrough_mining_20260601T221911Z/summary.json
codexlab/runs/arc_opus48_playthrough_mining_20260601T221911Z/*__reasoning.jsonl
reference/TRAINING_BIBLE_MACRO.reference_only.md
reference/arc_game_mechanic_decode_2026-05-19/README.md
```

Spark subagent slices:

- Partial-progress recordings: `lp85`, `ar25`, `ft09`, `dc22`, `vc33`,
  `sp80`, `s5i5`.
- Zero-completion/high-action recordings: `bp35`, `cd82`, `g50t`, `ka59`,
  `lf52`, `r11l`, `sc25`, `sk48`, `tn36`, `tr87`.
- Cross-source process extraction: public scorecard mining, human strategy
  corpus, and source-code mechanic decode.

Downloaded corpus summary:

- Scorecard: `anthropic-opus-4-8-high`
- Recordings: `25`
- Total bytes: `252372365`
- Local size: about `241M`
- Largest recordings by bytes: `g50t`, `sb26`, `sp80`, `cd82`, `ar25`

## Core Finding

The useful extraction is not "Claude knows ARC." The useful extraction is a
process failure/success boundary:

1. Partial progress happens when the run first stabilizes a control surface,
   then binds that surface to a game family, then commits small objective
   transitions.
2. Failure loops happen when the run keeps exploring after the action map is
   unstable, repeats low-novelty probes, burns reset/action budget, or treats a
   timer/counter as a goal oracle.
3. The human strategy corpus is behavioral. The source-code decode is
   API-structural. Public playthroughs show what a frontier model actually does
   under budget. CONTRA needs to keep those three layers separate.

## Required Process Model

### 1. Surface First

Before scoring candidate actions, identify:

- available action ids
- click-vs-D-pad-vs-sprite-move-vs-place mechanics
- controllable objects
- static/invariant objects
- counters, timers, borders, and candidate goal regions

CONTRA opportunity:

- put source/action assumptions into `config_pass`
- keep Oracle/human-derived semantics as review-only rows
- require raw-frame component and control-isolation packets before planner
  routing

Guideline:

```text
No family planner before control-surface packet exists.
```

### 2. Family Binding

Use source-code mechanic decode as the first hard prior:

- click-only
- D-pad navigation
- click plus sprite movement
- click plus place/sprite movement
- click plus agent movement

Then use observed deltas to override or narrow the family.

CONTRA opportunity:

- add a family-routing packet that records source-decode family, observed
  action responses, and contradictions
- route to click planner, directional planner, sprite-move planner, or
  placement planner only after a bounded probe ledger exists

Guideline:

```text
Behavioral family labels are priors. Source action signatures and controlled
deltas are stronger.
```

### 3. Action Semantics Probe Ledger

Partial-progress recordings repeatedly made progress after discovering a small
stable action set:

- `ACTION6` dominated click-family progress in `ft09`, `vc33`, `s5i5`, `lp85`.
- Direction/action mapping mattered in `ar25`, `dc22`, and `sp80`.
- `sp80` public run inferred `ACTION1=up`, `ACTION2=down`, `ACTION3=left`,
  `ACTION4=right`, and `ACTION5=simulation/run`, with transitions on
  `ACTION5`.

CONTRA opportunity:

Record per-action before/after deltas with keys:

```text
game_id
level_index
action_id
selected_object_or_click_cell
before_signature
after_signature
delta_signature
novelty_score
irreversible_risk
```

Guideline:

```text
No committed plan until each proposed action class has a recorded delta or a
documented missing-delta reason.
```

### 4. Predict, Observe, Revise

Public successful fragments repeatedly used a predict/observe/update loop.
Failure fragments often continued after contradiction without clearing the
representation.

CONTRA opportunity:

- add expected-delta rows before action execution
- compare observed DeltaBand/ActionModBand response after execution
- when expected and observed deltas disagree twice under the same
  representation class, clear local rank and force a new branch

Guideline:

```text
Two same-basin mismatches create stale_loop. Resume only with a new
representation/search-space branch or explicit approval.
```

### 5. No-Repeat Exploration Memory

Zero-completion recordings show high repeat-risk counts and long action budgets
without progress. This is not just "bad reasoning"; it is missing memory and
branch decay.

CONTRA opportunity:

Dead-probe keys:

```text
game_id
level_index
action_id
object_id_or_click_cell
delta_signature
local_hypothesis_id
```

Decay or block a branch when the same key recurs with no objective delta.

Guideline:

```text
Repeating a low-novelty probe is a bug unless the packet records why the repeat
tests a new hypothesis.
```

### 6. Budget-Aware Control

Failure runs burned actions and resets while still uncertain about objective
state. Timers, counters, and border bars were sometimes misread as goal truth.

CONTRA opportunity:

- track action budget, reset count, level step budget, and novelty gain
- distinguish `progress_counter`, `timer`, `resource`, and `goal_status`
  candidates as separate review-only fields
- switch to constrained diagnostics when budget is burning without objective
  gain

Guideline:

```text
If budget decreases and objective confidence is flat, stop broad probing and
enter diagnostic mode.
```

### 7. Coordinate Vetting For ACTION6

Click-style games progress through `ACTION6`, but failure cases show repeated
coordinate-frame, row/column, or off-by-one mistakes.

CONTRA opportunity:

- add one low-cost click probe to establish display/raw coordinate mapping
- record click cell, target object, nearest component, and observed local delta
- forbid bulk click plans until coordinate mapping is confirmed

Guideline:

```text
ACTION6 needs coordinate-map verification before it can drive a planner.
```

### 8. Irreversible-State Quarantine

Failure logs show risky state perturbations before control confidence:
dead-ended objects, drained resources, unreachable regions, and timer burn.

CONTRA opportunity:

- mark actions with suspected irreversible deltas
- require checkpoint/reset branch before continuing after a worsening delta
- do not let Oracle or static geometry override a recorded worsening signal

Guideline:

```text
Suspected irreversible worsening moves the branch to quarantine, not to normal
exploration.
```

## Review-Only Versus Authority

Review-only:

- transcript/human strategy labels
- Oracle rows
- static goal candidates
- PortBand route-flow hints
- action-state tags before controlled action
- action interaction graph/matrix hints before controlled action
- family-routing priors
- counters/timers/goal-status candidates

Potential action/control authority only after controlled local execution:

- before/action/after frame packet
- DeltaBand action-surface result event
- ActionModBand response packet
- episode-memory observation tied to a recorded action
- level-completion delta from engine state

Hard rule:

```text
Static review can rank what to test. It cannot choose, certify, or claim.
```

## Concrete Systems To Add

### A. `ActionSemanticsLedger`

Purpose: store per-action response observations before planning.

Outputs:

- action id
- selected object/cell
- before/after hashes
- changed component ids
- movement vector or no-op
- coordinate-map confidence
- irreversible-risk flag

### B. `FamilyRoutePrior`

Purpose: bind source-code action-family priors to observed deltas.

Inputs:

- source decode family
- available actions
- baseline action count pattern
- observed action deltas
- contradiction count

Outputs:

- planner lane recommendation
- confidence
- missing probes
- fail-closed reason

### C. `DeadProbeMemory`

Purpose: prevent repeated low-novelty exploration loops.

Outputs:

- probe key
- novelty score
- repeated count
- objective delta
- branch decay recommendation

### D. `BudgetGovernor`

Purpose: prevent action/reset burn without information gain.

Outputs:

- actions spent
- resets spent
- novelty gained
- objective confidence delta
- diagnostic-mode trigger

### E. `CoordinateVetting`

Purpose: make click games safe to route.

Outputs:

- raw/display coordinate transform status
- clicked component id
- nearest candidate object
- local delta
- row/column ambiguity flags

### F. `ContradictionReplanGate`

Purpose: force representation changes after repeated mismatch.

Outputs:

- expected delta
- observed delta
- mismatch count
- same-basin id
- stale_loop status
- required representation change

### G. `GoalSpecCandidatePacket`

Purpose: isolate target/timer/resource/goal hypotheses without granting truth.

Outputs:

- candidate type: goal, timer, resource, progress counter, decoration, unknown
- source: raw frame, source decode, transcript prior, public-play pattern
- admission status
- verifier requirement

## Implementation Order

1. Add `ActionSemanticsLedger` to controlled harness artifacts.
2. Add `CoordinateVetting` for `ACTION6`.
3. Add `DeadProbeMemory` and `BudgetGovernor`.
4. Add `ContradictionReplanGate`.
5. Add `FamilyRoutePrior`.
6. Add `GoalSpecCandidatePacket`.

Reason for this order: action deltas and coordinate vetting create the hard
local substrate. Family routing and goal candidates should consume that
substrate, not replace it.

## SP80-Specific Extraction

Public Opus 4.8 `sp80` completed two levels and transitioned on `ACTION5`.
The mined hypothesis is:

- directional movement must be proven through a local action ledger
- `ACTION5` should be treated as a commit/simulation/run action candidate
- flow-routing/PortBand hints remain review-only until a controlled before and
  after action sequence records the consequence

SP80 guideline:

```text
For SP80, no route-flow or PortBand claim may bypass the action semantics
ledger plus ACTION5 commit-result packet.
```

## Conflicts To Preserve

Do not smooth these conflicts away:

- Human strategy families are behavioral; source decode families are
  API-structural.
- Public scorecard progress is partial and often ends in `GAME_OVER` or
  `NOT_FINISHED`.
- Generic "probe ACTION5" advice is not universal; source action signatures and
  observed deltas decide whether ACTION5 matters.
- Counters can be progress, timer, resource, or decoration. They are not goal
  truth without engine-state confirmation.
- Public playthrough reasoning is useful for process mining, not proof of what
  CONTRA can do.

## Reporting Standard

Any future report that uses this extraction must name:

- public scorecard id
- recording guid or local raw JSONL path
- whether the claim is source decode, public playthrough hypothesis, human
  strategy prior, or local controlled evidence
- exact local controlled runner if a CONTRA claim is made
- review page path before raw JSON

Minimum boundary sentence:

```text
Public ARC recording mining. Hypothesis/design material only. Not local CONTRA
control truth, goal truth, action choice, prediction truth, or solve evidence.
```
