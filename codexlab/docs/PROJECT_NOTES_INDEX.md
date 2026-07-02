# CONTRA Project Notes Index

Date: 2026-05-31
Scope: house cleanup before CONTRA code reduction.

## Canonical Notes

- `codexlab/README.md` - entry point for the current codexlab lane.
- `codexlab/docs/DEV_FACTORY_FLOOR_2026-05-31.md` - dev readiness check, cleanup boundaries, and next reduction targets.
- `codexlab/docs/RUN_ARTIFACT_CLEANUP_2026-05-31.md` - artifact deletion ledger and protected evidence boundary.
- `codexlab/docs/CONTRA_CHARTER.md` - promoted unique Claude-side charter note.
- `codexlab/docs/CODEX_CLAUDE_COMMS.md` - promoted cross-agent communication note.
- `codexlab/docs/CONTRA_FRAME_DEBUG_TEMPLATE_HANDOFF.md` - promoted frame-debug handoff/template note.
- `codexlab/docs/CONTRA_GAMEPLAY_REPORTING_SOP.md` - promoted Angel gameplay reporting SOP.
- `.codex/skills/codexlab-dev/SKILL.md` - project skill for dev gates, integration smoke, and cleanup boundaries.
- `codexlab/docs/CONTRA_ORACLE_HANDOFF_2026-05-27.md` - Oracle handoff context.
- `codexlab/docs/CONTRA_PASS_REVIEW_SURFACE_SPO.md` - pass/review surface guidance.
- `codexlab/docs/CONTRA_TINY_MODEL_MIXING_POSSIBILITIES.md` - model-mixing ideas; keep as hypothesis notes, not result evidence.
- `codexlab/docs/ARC_OPUS48_PLAYTHROUGH_MINING_2026-06-01.md` - public Claude Opus 4.8 ARC scorecard/playthrough mining; hypothesis queue only, not local run evidence.
- `codexlab/docs/ARC_OPUS48_SPARK_SYSTEMS_EXTRACTION_2026-06-01.md` - Spark subagent synthesis from downloaded public Opus 4.8 playthroughs; process/guideline extraction only.
- `codexlab/docs/CONTRA_TEACHER_STUDENT_SIGNAL_PLAN_2026-06-01.md` - active teacher/student execution plan for converting public Opus traces into CONTRA signal-supervision rows.
- `codexlab/datasets/arc_opus48_scorecard_reasoning_corpus_v0/manifest.json` - full public Opus 4.8 scorecard reasoning corpus for algorithm extraction; hypothesis/process supervision only.
- `codexlab/datasets/arc_opus48_teacher_signals_v0/manifest.json` - exported teacher-signal dataset; hypothesis/process supervision only.
- `codexlab/contra/action_semantics_ledger.py` - first CONTRA student packet producer; static probe ledger only, not action/control truth.
- `codexlab/runs/action_semantics_ledger_offline_score_20260602T002300Z/manifest.json` - offline historical/static proxy score for ActionSemanticsLedger against teacher action-semantics rows.
- `codexlab/contra/BOUNDARY_PASS_SPEC.md` - boundary pass specification.
- `codexlab/contra/BOUNDARY_PASS_COMPLETION_AUDIT.md` - boundary pass audit record.

## Non-Canonical Or Scratch Notes

- `codexlab/docs/CLAUDE_OVERNIGHT_LOOP_PROMPT.md` - prompt artifact from overnight/loop work. Keep for provenance, but do not treat as an active run plan without review.
- `claudelab/*.md` - Claude-side copies and communications. Keep as scratch/reference until we intentionally merge or archive them.

## Evidence Boundary

Historical run notes and generated review pages are retained as historical artifacts only. They are not fresh evidence unless the exact command, input condition, and output path are re-run or explicitly verified in the current session.

## Cleanup Rule

Before the code-reduction pass, keep notes in one of three states:

- `canonical`: referenced by current CONTRA code or current SP80 review workflow.
- `scratch`: useful context, but not a dependency and not evidence.
- `archive/delete candidate`: stale prompt, duplicate note, failed-loop material, or note that only describes a discarded representation.
