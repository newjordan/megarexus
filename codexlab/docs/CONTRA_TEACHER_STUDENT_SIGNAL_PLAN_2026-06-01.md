# CONTRA Teacher-Student Signal Plan

Date: 2026-06-01
Status: active implementation plan.

## Boundary

Teacher data from public Claude/Opus ARC playthroughs is process supervision,
not local CONTRA evidence. It may teach CONTRA to produce better intermediate
signals, but it cannot certify control truth, goal truth, action choice,
prediction truth, or solve evidence.

Minimum claim shape:

```text
Teacher trace -> extracted signal target -> CONTRA packet target -> local
controlled verifier later.
```

Never use:

```text
Teacher reasoning -> solve claim
```

## Objective

Train and evaluate CONTRA as a signal student. The first milestone is not
gameplay score. The first milestone is cleaner packet signals:

- control-surface classification
- action-family routing
- action semantics hypotheses
- coordinate ambiguity detection
- dead-probe/repeat detection
- budget burn detection
- timer/resource/progress/goal separation
- contradiction/stale-loop gating

## Phases

### Phase 0: Teacher Corpus

Inputs already downloaded:

```text
codexlab/runs/arc_public_recordings_anthropic-opus-4-8-high_20260601T222454Z/download_manifest.json
codexlab/runs/arc_public_recordings_anthropic-opus-4-8-high_20260601T222454Z/raw_recordings/
codexlab/runs/arc_opus48_playthrough_mining_20260601T221911Z/summary.json
codexlab/runs/arc_opus48_playthrough_mining_20260601T221911Z/*__reasoning.jsonl
```

Output:

```text
codexlab/datasets/arc_opus48_scorecard_reasoning_corpus_v0/
  all_reasoning.jsonl
  train.jsonl
  heldout.jsonl
  manifest.json
codexlab/datasets/arc_opus48_teacher_signals_v0/
  train.jsonl
  heldout.jsonl
  manifest.json
```

Gate:

- no raw frames required in the dataset rows
- full reasoning text is preserved in the reasoning corpus for algorithm extraction
- every row cites source scorecard, run guid, reasoning file, and manifest
- reasoning corpus rows have `truth_status: hypothesis_teacher_reasoning`
- every row has `truth_status: hypothesis_teacher_signal`

### Phase 1: Student Signal Rows

Row families:

- `control_surface_signal`
- `family_route_prior_signal`
- `action_semantics_signal`
- `coordinate_vetting_signal`
- `dead_probe_memory_signal`
- `budget_governor_signal`
- `goal_spec_candidate_signal`
- `contradiction_replan_signal`

Targets are packet recommendations, not actions.

### Phase 2: CONTRA Packet Hooks

Add packet producers in this order:

1. Action semantics ledger. `implemented: codexlab/contra/action_semantics_ledger.py`
2. Coordinate vetting.
3. Dead-probe memory.
4. Budget governor.
5. Contradiction replan gate.
6. Family route prior.
7. Goal-spec candidate packet.

Gate:

- static packets can rank what to test
- only controlled before/action/after packets can support action/control claims

### Phase 3: Evaluation Before Training

Build offline scorecards before any model training:

- Does the student signal identify click vs D-pad vs sprite-move families?
- Does it flag repeated low-novelty probes?
- Does it flag coordinate ambiguity on `ACTION6` runs?
- Does it separate timer/resource/progress/goal candidates?
- Does it trigger stale-loop earlier than the public Claude run?

Gate:

- evaluation rows must be distinct from train rows by game id where possible
- no gameplay score claims

First scorer:

```text
codexlab/arc_score_action_semantics_ledger.py
codexlab/runs/action_semantics_ledger_offline_score_20260602T002300Z/manifest.json
codexlab/runs/action_semantics_ledger_offline_score_20260602T002300Z/index.html
```

Status: `historical_static_proxy`, `offline_hypothesis_score`.
Boundary: public recording static frames and teacher-signal comparison only;
not local CONTRA gameplay evidence.

### Phase 4: Small Model Distillation

Only after the signal rows and packet evaluators exist:

- format rows as chat or structured JSON targets
- train/dry-run a small model on signal packet emission
- compare packet-quality metrics against rule baselines

Gate:

- local controlled verifier remains the only source of control/action truth

## First Implementation Step

Create `codexlab/arc_export_teacher_signal_dataset.py`.

The exporter should:

1. read `summary.json`
2. read `download_manifest.json`
3. read `*__reasoning.jsonl`
4. emit signal rows for:
   - level-completion transitions
   - high repeat-risk zero-completion runs
   - coordinate/action-family click-heavy runs
   - contradiction-heavy rows
   - budget-burn rows
5. write `manifest.json` with hashes and boundaries

## Reporting Rule

Any report must include:

```text
Status: hypothesis_teacher_signal
Boundary: not local CONTRA evidence
Source: public scorecard id + recording guid + local row path
Next verifier: local controlled CONTRA/SP80 packet gate
```
