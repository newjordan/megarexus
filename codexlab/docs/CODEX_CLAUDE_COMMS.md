# Codex / Claude CONTRA Communication Protocol

> Purpose: keep Codex and Claude aligned while CONTRA is split across an active
> implementation lane and a Claude review/planning lane. This document is a
> coordination contract, not evidence and not a promotion gate.

## 1. Active Path Declaration

Every handoff or request between agents must start by naming the active path:

```text
active_implementation_path: codexlab/
review_or_charter_path: claudelab/
repo_root: $REPO
git_commit: <short hash>
```

Current boundary as of 2026-05-28:

- `codexlab/` is the active committed implementation path.
- `claudelab/` is a review/planning mirror unless the human explicitly promotes
  it to active implementation.
- Do not claim a `claudelab/` result is current implementation evidence unless
  the command was actually run from that path and the output path is named.

## 2. Truth Boundary

Both agents must preserve the CONTRA charter boundary:

- Static hypotheses are not control truth.
- Oracle assumptions are not CONTRA truth.
- GeoBand and symmetry are static relational context, not prediction.
- Match calibration and transfer priors reweight roles only; they do not admit
  actor truth, target truth, control truth, action choice, or solve truth.
- WIN or solved status can only come from engine state, never from a review
  page, packet hypothesis, or agent summary.

If an artifact lacks the requested condition, mark it `not_run`, `stale`,
`proxy`, or `invalid`. Do not patch wording around the gap.

## 3. Message Header

Use this compact header for every cross-agent handoff:

```text
CONTRA_COMMS_V1
from: <codex|claude>
to: <codex|claude|human>
repo_root: $REPO
active_path: <codexlab|claudelab>
commit: <short hash or uncommitted>
condition: <static_review|controlled_run|charter_review|implementation>
truth_status: <not_admitted|controlled|not_run|invalid|stale>
requested_next_action: <one concrete action>
```

Follow the header with:

```text
changed_files:
- <path>

artifacts:
- <path or URL>

commands_run:
- <exact command>

known_boundaries:
- <what this does not prove>

open_questions:
- <question>
```

## 4. Ownership Rules

- Do not edit the other agent's active file set while it is mid-task unless the
  human redirects the work.
- Do not revert files you did not change.
- Do not stage broad paths like `claudelab/` or `codexlab/` when only a small
  document or packet is intended.
- If both agents need the same file, one owns the edit and the other reviews.
- If work crosses `codexlab/` and `claudelab/`, the handoff must name which path
  is source of truth for that turn.

## 5. Evidence Surface Rules

When either agent presents packet or run results to the human, provide a review
surface before raw JSON:

- `index.html` or equivalent readable page
- raw packet links
- exact command provenance
- verifier status where available
- live URL check when a URL is sent
- visual screenshot or Playwright check when page readability matters

For CONTRA/SP80 reports, include:

- level/game id
- admission status
- transfer-prior status
- selected Stage 0 candidate and click policy
- actor/target/support/StateCon/GeoBand counts
- Oracle sidecar status if present
- boundary sentence: static review only unless controlled action evidence exists

## 6. Calibration Vocabulary

Use these terms consistently:

- `answer_key_calibration`: human-reviewed expectations used to score derived
  abstractions. This is not a detector output.
- `match_runtime_calibration`: initialization clicks, init-choice probes,
  DeltaBand controlled boundaries, and action-result transfer priors gathered
  during a match.
- `action_result_transfer_prior`: temporal context from observed action/result.
  It has function-specific confidence math and fast miss decay. It reweights
  current-frame roles only.
- `primary_actor_transfer_prior_candidate`: a current-frame candidate boosted by
  recorded actor-origin observations.
- `secondary_actor_like_*`: actor-looking geometry preserved as relational,
  key/lock, symmetry, or affected-object context when a primary actor prior is
  already high confidence.

## 7. Transfer Prior Guardrail

For the level 03 paint lesson:

If levels 01 and 02 record controlled actor-origin success for the purple /
magenta paint head, then level 03 may carry a high-confidence
`actor_origin` transfer prior for the matching purple / magenta component.

That does exactly two things:

1. Reweights the purple / magenta component toward primary actor origin.
2. Reweights other actor-like components toward secondary relational roles.

It does not:

- delete the secondary object
- admit actor truth
- admit action choice
- infer the target
- claim a solve

## 8. Good Handoff Example

```text
CONTRA_COMMS_V1
from: codex
to: claude
repo_root: $REPO
active_path: codexlab
commit: f2ea50f
condition: static_review
truth_status: not_admitted
requested_next_action: review transfer-prior wording before DeltaBand changes

changed_files:
- codexlab/contra/match_calibration/transfer_prior.py
- codexlab/contra/match_calibration/stage0.py

artifacts:
- codexlab/runs/contra_sp80_levels_00_03_transfer_prior_report_20260528T164437/index.html
- codexlab/runs/contra_sp80_levels_00_03_transfer_prior_report_20260528T164437/report.json

commands_run:
- python3 codexlab/contra_api_verify.py codexlab/runs/.../level_03/result.json

known_boundaries:
- no prior observations were injected in the static 0-3 report
- transfer prior stayed no_prior_observations_supplied

open_questions:
- where should controlled action-result observations be recorded for match memory?
```

## 9. Stop Conditions

Stop and ask the human before proceeding if:

- the active path is ambiguous
- an artifact is stale relative to source changes
- a controlled-run claim lacks before/action/after provenance
- a charter rule conflicts with current implementation work
- a proposed shortcut would promote hypothesis into truth

