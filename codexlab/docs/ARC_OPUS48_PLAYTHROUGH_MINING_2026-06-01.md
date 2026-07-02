# ARC Opus 4.8 Playthrough Mining

Date: 2026-06-01
Status: hypothesis queue, not run evidence.

## Source Boundary

Public ARC Prize scorecard reviewed:

```text
https://arcprize.org/scorecards/model/anthropic-opus-4-8-high
https://arcprize.org/api/models/scorecard/anthropic-opus-4-8-high
```

Live scorecard fields at review time:

- Model: `claude-opus-4-8`
- Runner: `benchmark_agent`
- Published: `2026-05-31T20:49:31.346372Z`
- Last update: `2026-05-31T17:58:15.296369Z`
- Score: `4.7917398946851`
- Levels completed: `29 / 183`
- Environments completed: `0 / 25`
- Total actions: `7839`

Mining artifact:

```text
codexlab/runs/arc_opus48_playthrough_mining_20260601T221911Z/index.html
codexlab/runs/arc_opus48_playthrough_mining_20260601T221911Z/index.md
codexlab/runs/arc_opus48_playthrough_mining_20260601T221911Z/summary.json
```

The mining script strips raw frame arrays from local event logs and keeps public
recording provenance, action ids, level-completion transitions, bounded
reasoning excerpts, usage/cost metadata, and keyword pattern counts.

This material is historical public scorecard review only. It is not local
CONTRA control truth, goal truth, action choice, prediction, or solve evidence.

## Highest-Value Recordings

Start mining these first because they contain actual level-completion
transitions and enough later action budget to expose where the model stalls:

| Game | Levels | Actions | Main transition actions |
| --- | ---: | ---: | --- |
| `lp85-305b61c3` | 4 | 212 | `ACTION6` at completions 1-4 |
| `ar25-0c556536` | 4 | 1133 | `ACTION2`, `ACTION2`, `ACTION1`, `ACTION2` |
| `ft09-0d8bbf25` | 3 | 167 | `ACTION6` at completions 1-3 |
| `dc22-fdcac232` | 3 | 866 | `ACTION4`, `ACTION4`, `ACTION2` |
| `vc33-5430563c` | 3 | 388 | `ACTION6` at completions 1-3 |
| `sp80-589a99af` | 2 | 178 | `ACTION5`, `ACTION5` |
| `s5i5-18d95033` | 2 | 309 | `ACTION6`, `ACTION6` |

Zero-completion recordings with large action budgets are still useful as
negative mining surfaces for repeated probes and bad family routing:

```text
bp35, cd82, g50t, ka59, lf52, r11l, sc25, sk48, tn36, tr87
```

## Candidate System Additions

### 1. Action Semantics Probe Ledger

Source signal: repeated control-surface and predict/observe/update language
across the public Claude recordings.

Implementation target: before planning, record per-action before/after deltas
for `ACTION1` through `ACTION6`, grouped by game family and selected object or
click cell.

Boundary: this should create probe evidence for review and routing only. It is
not action authority or solve truth.

### 2. No-Repeat Exploration Memory

Source signal: zero-completion runs spent 100+ actions in the same public
recording without level completion.

Implementation target: add dead-probe keys over:

```text
game_id, level_index, action_id, selected_object_or_click_cell, delta_signature
```

Use those keys to decay repeated branches and force a representation change
after two same-basin failures.

Boundary: reweights exploration only.

### 3. Goal-Spec Candidate Extractor

Source signal: goal-inference terms appear often while recordings still spend
many actions without completing later levels.

Implementation target: promote bordered regions, invariant objects, target-like
regions, and resource counters into a review-only goal-spec packet with explicit
admission flags.

Boundary: goal candidates remain hypotheses until engine/verifier completion
semantics confirms them.

### 4. Resource-Aware Probe Budget

Source signal: several partial-progress recordings finish early levels and then
stall for hundreds of actions or resets.

Implementation target: track actions, resets, steps remaining, and budget decay
inside CONTRA exploration packets. Use it to stop low-information loops and
preserve budget for family-specific probes.

Boundary: budget controls search allocation only.

### 5. Family-First Planner Routing

Source signal: progress clusters around games where early control/mechanic
family becomes legible, especially click-driven `ACTION6` games and known
family-specific control surfaces.

Implementation target: combine `reference/arc_game_mechanic_decode_2026-05-19/`
family structure with observed action-response deltas to choose click-only,
D-pad, sprite-move, or placement planner lanes.

Boundary: family routing is a prior; controlled deltas must override it.

### 6. Contradiction-Triggered Replan

Source signal: public reasoning contains explicit wrong/unexpected/revise
language, and partial-progress runs often continue after the mismatch.

Implementation target: when expected delta and observed delta disagree twice
under the same representation class, clear local hypothesis rank and force a
new representation branch.

Boundary: stale-loop prevention only; not evidence of correctness.

## Implementation Queue

1. Add the Action Semantics Probe Ledger as a controlled-harness artifact.
2. Feed ledger rows into action-state tags and `action_interaction_graph`, not
   PortBand route-flow authority.
3. Add the No-Repeat Exploration Memory decay rule.
4. Add family routing as a review prior using source decode families plus
   observed deltas.
5. Add goal-spec candidate packets after the first three controls exist.

Do not promote any mined public recording into local evidence. Every proposed
addition needs a controlled local runner, review surface, and verifier before
it can support a CONTRA claim.
