# Loom Documentation Map

Status: active map as of 2026-05-24.

Prime directive: build a **LeWM + model system that can compete in ARCPrize3
legally**.

Canonical architecture entrypoint: [`README.md`](README.md).

## Read Order For Any Agent

1. [`../../AGENTS.md`](../../AGENTS.md) - folder-level prime directive and
   first-read guardrails.
2. [`README.md`](README.md) - current architecture canon and change-history
   index.
3. [`../../lab/comm/latest/LEWM_FORWARD_SURFACE_LOCK_2026-05-24.md`](../../lab/comm/latest/LEWM_FORWARD_SURFACE_LOCK_2026-05-24.md)
   - current forward surface: Rexus/LeWM `O_pred` before Q6.
4. [`../../lab/comm/latest/NOT_FORWARD_LEWM_QUARANTINE_2026-05-24.md`](../../lab/comm/latest/NOT_FORWARD_LEWM_QUARANTINE_2026-05-24.md)
   - sequestered non-forward/Q6 surfaces.
5. [`../TRAINING_DATASET_BIBLE.md`](../TRAINING_DATASET_BIBLE.md) - dataset
   row contract and quarantine rules.
6. [`../REXUS_CORE_ACTION_LOOP.md`](../REXUS_CORE_ACTION_LOOP.md) - Rexus
   action-loop trace contract.
7. [`../REXUS_GRAM_PLUS_PRETRAIN_REQUIREMENTS_2026-05-24.md`](../REXUS_GRAM_PLUS_PRETRAIN_REQUIREMENTS_2026-05-24.md)
   - GRAM++ architecture requirements that block new Rexus training until
   operating picture, path fan, lattice, LPRM feature contract, memory, and
   leakage gates are explicit.
8. [`../../experiments/2026-05-24_rexus_gram_plus_path_probe_v0/REPORT.md`](../../experiments/2026-05-24_rexus_gram_plus_path_probe_v0/REPORT.md)
   - First RexusGramPlusPathProbe proxy scout; confirms current blockers for
   complete candidate-bank prediction and explicit recursive path fans.
9. [`../../experiments/2026-05-24_rexus_candidate_bank_predictor_probe_v0/REPORT.md`](../../experiments/2026-05-24_rexus_candidate_bank_predictor_probe_v0/REPORT.md)
   - First bounded candidate-bank `O_pred` probe; removes candidate-bank
   blocker for the V0 condition, leaves recursive path fan blocked.
10. [`../../experiments/2026-05-24_rexus_gram_plus_path_probe_v1_pathfan/REPORT.md`](../../experiments/2026-05-24_rexus_gram_plus_path_probe_v1_pathfan/REPORT.md)
   - First bounded recursive path-fan scout; stricter GRAM++ row-shape
   checklist passes for 4 states with independent recursive metadata, depth >0
   coverage, and unique path IDs, proxy only.
11. [`../../lab/architecture_sweeps/2026-05-24_rexus_foundation_idea_sweep/reports/REXUS_ADMISSION_MATRIX_2026-05-24.md`](../../lab/architecture_sweeps/2026-05-24_rexus_foundation_idea_sweep/reports/REXUS_ADMISSION_MATRIX_2026-05-24.md)
   - Current decision matrix for what can enter Rexus versus proxy/quarantine.
12. [`../../lab/architecture_sweeps/2026-05-24_rexus_foundation_idea_sweep/reports/RUNTIME_GRADUATION_QUEUE_2026-05-24.md`](../../lab/architecture_sweeps/2026-05-24_rexus_foundation_idea_sweep/reports/RUNTIME_GRADUATION_QUEUE_2026-05-24.md)
   - Current follow-up order and runtime gates for TIMR, registry/gym replay,
   active inference, no-backtrack, meta-horizon, LPRM, Q6, ACTION6, and ARC
   ablations.
13. [`../../experiments/2026-05-24_rexus_scorer_feature_graduation_manifest_v0/manifest.json`](../../experiments/2026-05-24_rexus_scorer_feature_graduation_manifest_v0/manifest.json)
   - Fail-closed next-artifact manifest for active inference, no-backtrack,
   and meta-horizon.
14. [`../../experiments/2026-05-24_rexus_lprm_q6_downstream_manifest_v0/manifest.json`](../../experiments/2026-05-24_rexus_lprm_q6_downstream_manifest_v0/manifest.json)
   - Fail-closed downstream manifest for LPRM and Q6; scorer interface only,
   no runtime/training/live gate.
15. [`../../experiments/2026-05-24_rexus_training_authorization_language_audit_v0/audit.json`](../../experiments/2026-05-24_rexus_training_authorization_language_audit_v0/audit.json)
   - Documentation language guard forbidding training-ready-true snippets in
   human decision docs; bounded probes must use row-shape language instead.
16. [`../../experiments/2026-05-24_rexus_admission_surface_consistency_audit_v0/audit.json`](../../experiments/2026-05-24_rexus_admission_surface_consistency_audit_v0/audit.json)
   - Cross-surface guard ensuring the admission matrix, runtime queue, canon
   docs, scorer manifest, LPRM/Q6 manifest, needs-runner backlog, and
   training-language audit agree that broad GRAM++ is not unlocked and
   runtime/training admission remains zero.
17. [`../../experiments/2026-05-24_rexus_needs_runner_conversion_backlog_v0/manifest.json`](../../experiments/2026-05-24_rexus_needs_runner_conversion_backlog_v0/manifest.json)
   - Fail-closed conversion backlog for the 8 Round-1 unclear-review rows that
   need runners or Rexus row conversion before admission.

## Active Canon Docs

| Doc | Role |
| --- | --- |
| [`README.md`](README.md) | Current canon architecture and change-history index. |
| [`LEWM_PAPER_PASSGATE.md`](LEWM_PAPER_PASSGATE.md) | Passgate stamp for this canon doc set. |
| [`../REXUS_CORE_ACTION_LOOP.md`](../REXUS_CORE_ACTION_LOOP.md) | Canonical 12-step Rexus action-side interface. |
| [`../REXUS_GRAM_PLUS_PRETRAIN_REQUIREMENTS_2026-05-24.md`](../REXUS_GRAM_PLUS_PRETRAIN_REQUIREMENTS_2026-05-24.md) | Active pretraining architecture gate for GRAM++/LPRM/path-fan requirements. |
| [`../../experiments/2026-05-24_rexus_gram_plus_path_probe_v0/REPORT.md`](../../experiments/2026-05-24_rexus_gram_plus_path_probe_v0/REPORT.md) | Active proxy scout artifact for RexusGramPlusPathProbe; not training or scorecard evidence. |
| [`../../experiments/2026-05-24_rexus_candidate_bank_predictor_probe_v0/REPORT.md`](../../experiments/2026-05-24_rexus_candidate_bank_predictor_probe_v0/REPORT.md) | Active bounded candidate-bank `O_pred` probe; not counterfactual truth or scorecard evidence. |
| [`../../experiments/2026-05-24_rexus_gram_plus_path_probe_v1_pathfan/REPORT.md`](../../experiments/2026-05-24_rexus_gram_plus_path_probe_v1_pathfan/REPORT.md) | Active bounded recursive path-fan probe; proxy row-shape evidence only. |
| [`../../lab/architecture_sweeps/2026-05-24_rexus_foundation_idea_sweep/reports/REXUS_ADMISSION_MATRIX_2026-05-24.md`](../../lab/architecture_sweeps/2026-05-24_rexus_foundation_idea_sweep/reports/REXUS_ADMISSION_MATRIX_2026-05-24.md) | Active admission decision surface for loose Rexus ideas. |
| [`../../lab/architecture_sweeps/2026-05-24_rexus_foundation_idea_sweep/reports/RUNTIME_GRADUATION_QUEUE_2026-05-24.md`](../../lab/architecture_sweeps/2026-05-24_rexus_foundation_idea_sweep/reports/RUNTIME_GRADUATION_QUEUE_2026-05-24.md) | Active runtime-graduation queue; no conditional lane enters runtime without its row-level gate. |
| [`../../experiments/2026-05-24_rexus_scorer_feature_graduation_manifest_v0/manifest.json`](../../experiments/2026-05-24_rexus_scorer_feature_graduation_manifest_v0/manifest.json) | Active scorer-feature guard; active inference, no-backtrack, and meta-horizon remain blocked until next controlled artifacts. |
| [`../../experiments/2026-05-24_rexus_lprm_q6_downstream_manifest_v0/manifest.json`](../../experiments/2026-05-24_rexus_lprm_q6_downstream_manifest_v0/manifest.json) | Active LPRM/Q6 guard; scorer interface only, runtime/training/live gate closed. |
| [`../../experiments/2026-05-24_rexus_training_authorization_language_audit_v0/audit.json`](../../experiments/2026-05-24_rexus_training_authorization_language_audit_v0/audit.json) | Active language guard; bounded row-shape passes must not be written as training-ready true. |
| [`../../experiments/2026-05-24_rexus_admission_surface_consistency_audit_v0/audit.json`](../../experiments/2026-05-24_rexus_admission_surface_consistency_audit_v0/audit.json) | Active consistency guard; no surface may imply runtime/training admission while the machine stoplights remain closed. |
| [`../../experiments/2026-05-24_rexus_needs_runner_conversion_backlog_v0/manifest.json`](../../experiments/2026-05-24_rexus_needs_runner_conversion_backlog_v0/manifest.json) | Active needs-runner backlog; remaining conversion rows are planning work only. |
| [`../TRAINING_DATASET_BIBLE.md`](../TRAINING_DATASET_BIBLE.md) | Dataset families, row identity, provenance, split, and no-leakage rules. |
| [`../../lab/foundations/NOW.md`](../../lab/foundations/NOW.md) | Current short-form work direction. |
| [`../../lab/foundations/STATUS.md`](../../lab/foundations/STATUS.md) | Current foundation status and deployment queue. |
| [`../../lab/foundations/NEEDS_RESEARCH.md`](../../lab/foundations/NEEDS_RESEARCH.md) | Research queue tied to missing Rexus/LeWM proof surfaces. |

## Historical Or Superseded Docs

These files remain useful only when read through the canon. If a line conflicts
with the canon architecture, the canon wins.

| Doc | Current status | Use |
| --- | --- | --- |
| [`../loom_direction.md`](../loom_direction.md) | historical foundation direction | Origin of "validate LeWM Layer 0 before extra layers". |
| [`../PROJECT_STATE.md`](../PROJECT_STATE.md) | historical 2026-05-12 state map | Subsystem history, not current authority. |
| [`../rex.md`](../rex.md) | historical Rex lineage | Legacy checkpoint lineage only; active system name is Rexus. |
| [`../FOUNDATION_SWEEP_DESIGN.md`](../FOUNDATION_SWEEP_DESIGN.md) | superseded Qwen sweep proposal | Historical sweep ideas only; not an active training queue. |
| [`../LOOM_REASONING_STACK_DESIGN.md`](../LOOM_REASONING_STACK_DESIGN.md) | superseded architecture proposal | Historical Qwen-backbone LRS proposal; not current foundation path. |
| [`../GRAM_LPRM_DESIGN.md`](../GRAM_LPRM_DESIGN.md) | superseded draft | Historical GRAM/LPRM plan; use `GRAM_LEWM_INTERNAL_PATH_PROCESSING` for current boundary. |
| [`../GRAM_LEWM_INTERNAL_PATH_PROCESSING_2026-05-22.md`](../GRAM_LEWM_INTERNAL_PATH_PROCESSING_2026-05-22.md) | active research boundary | Current GRAM/LDT translation, still proxy until runnable path probe exists. |
| [`../REXUS_TRAINING_STRATEGY_2026-05-22.md`](../REXUS_TRAINING_STRATEGY_2026-05-22.md) | active but gated | Training stack order; current first step remains Rexus/LeWM `O_pred`. |
| [`../../lab/comm/latest/REXUS_INFRASTRUCTURE_FINAL_REPORT.md`](../../lab/comm/latest/REXUS_INFRASTRUCTURE_FINAL_REPORT.md) | historical handoff | Does not authorize training or promotion. |
| [`../../lab/comm/latest/REXUS_SOVEREIGN_CLOSEOUT.md`](../../lab/comm/latest/REXUS_SOVEREIGN_CLOSEOUT.md) | historical closeout | Read only as context unless refreshed under current canon. |
| [`../../lab/comm/latest/REXUS_BIBLE_BIRTH_REPORT.md`](../../lab/comm/latest/REXUS_BIBLE_BIRTH_REPORT.md) | historical architecture framing | Use role language only if consistent with current Rexus ownership. |
| [`../../lab/comm/latest/MESI_HYPOTHESIS_AND_ABLATION_MATRIX.md`](../../lab/comm/latest/MESI_HYPOTHESIS_AND_ABLATION_MATRIX.md) | historical hypothesis matrix | MESI ideas are downstream; not a substitute for LeWM/Rexus prediction. |
| [`../../lab/comm/latest/QWEN_LEWM_TARGET_FIT_MATRIX_2026-05-23.md`](../../lab/comm/latest/QWEN_LEWM_TARGET_FIT_MATRIX_2026-05-23.md) | Q6 diagnostic boundary | Q6 re-enters only after real Rexus `O_pred` rows exist. |

## Maintenance Rule

When a doc is created or changed:

1. Link it from this map if it affects architecture, data, training, eval, or
   model roles.
2. Mark it `active`, `active but gated`, `historical`, `superseded`, or
   `quarantined`.
3. Point stale or conflicting docs back to [`README.md`](README.md).
4. Do not erase history; add a dated supersession note and preserve the old
   evidence trail.
