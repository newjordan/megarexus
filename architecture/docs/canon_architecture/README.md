# Loom Canon Architecture

Status: active canon as of 2026-05-24.

LeWM passgate: [`LEWM_PAPER_PASSGATE.md`](LEWM_PAPER_PASSGATE.md). Documentation
map: [`DOC_MAP.md`](DOC_MAP.md). This canon records the `adapted-paper-grounded`
ARCPrize3 architecture surface.

Prime directive: build a **LeWM + model system that can compete in ARCPrize3
legally**.

This document is the architecture entrypoint for agents working in Loom. It
links the current canon to the historical notes that changed it. If another
document conflicts with this one, follow this document and the linked forward
surface lock first.

## Current Canon

Loom is a LeWorldModel-rooted action system. The paper/root architecture is
LeWM. The active action-side system name is **Rexus**.

```text
observed frame/state S_t
  -> LeWM/Rexus encoder and memory state
  -> Rexus candidate actions A_1...A_k
  -> LeWM/Rexus predicted outcomes O_pred_1...O_pred_k
  -> State Prober / verifier delta against O_true when offline truth is legal
  -> optional scorer/ranker over grounded candidates
  -> Rexus action selection
  -> ARC engine step
  -> measured O_true and scorecard fields
```

The immediate foundation target is the smallest runnable predictor surface:

```text
recorded S_t + candidate A_k -> Rexus/LeWM O_pred -> State Prober delta vs recorded O_true
```

Q6/Qwen does not come before this. Q6 can only re-enter as a scorer over real
Rexus candidate transitions with candidate ids and non-null `O_pred`.

The next pretraining architecture gate is GRAM++ readiness:
[`../REXUS_GRAM_PLUS_PRETRAIN_REQUIREMENTS_2026-05-24.md`](../REXUS_GRAM_PLUS_PRETRAIN_REQUIREMENTS_2026-05-24.md).
It blocks new Rexus training claims until operating-picture packets, complete
candidate banks, candidate-linked `O_pred`, explicit latent path fans, lattice
projection verdicts, legal State Prober labels, LPRM feature allowlists,
goal-score vectors, episode memory, and leakage audits are recorded.

The installed car surface is
[`../../src/loom_layers/rexus_architecture_car.py`](../../src/loom_layers/rexus_architecture_car.py).
It registers every current Rexus lane as a disabled fail-closed component that
can attach metadata to core-loop candidate scores. This is the architecture
chassis, not runtime admission: runtime, training, and promotion flags remain
false until condition-specific evidence unlocks them.

The first runnable proxy scout for that gate is
`RexusGramPlusPathProbe`:

- code: [`../../src/loom_layers/rexus_gram_plus_path_probe.py`](../../src/loom_layers/rexus_gram_plus_path_probe.py)
- runner: [`../../lab/scripts/run_rexus_gram_plus_path_probe.py`](../../lab/scripts/run_rexus_gram_plus_path_probe.py)
- artifact:
  [`../../experiments/2026-05-24_rexus_gram_plus_path_probe_v0/`](../../experiments/2026-05-24_rexus_gram_plus_path_probe_v0/)

Its first run is a proxy scout and is training-blocked: current Phase 0 rows
have selected-action `O_pred`, but not complete candidate-bank prediction or
independent recursive GRAM path fans.

The first bounded candidate-bank repair exists at
[`../../experiments/2026-05-24_rexus_candidate_bank_predictor_probe_v0/`](../../experiments/2026-05-24_rexus_candidate_bank_predictor_probe_v0/).
It emits frozen Phase 0 `O_pred` for decode-derived candidate banks. The
downstream GRAM probe over those rows passes `candidate_bank_complete` and
`candidate_opred_complete`, but still fails `gram_path_fan_recorded`; recursive
path rollout remains the active blocker.

The first bounded recursive path-fan scout exists at
[`../../experiments/2026-05-24_rexus_gram_plus_path_probe_v1_pathfan/`](../../experiments/2026-05-24_rexus_gram_plus_path_probe_v1_pathfan/).
It records the GRAM++ row-shape checklist fields for 4 held-out states at width
2/depth 2, including independent recursive metadata, depth >0 coverage, and
unique path IDs.

Current loose-lane admission decisions live in
[`../../lab/architecture_sweeps/2026-05-24_rexus_foundation_idea_sweep/reports/REXUS_ADMISSION_MATRIX_2026-05-24.md`](../../lab/architecture_sweeps/2026-05-24_rexus_foundation_idea_sweep/reports/REXUS_ADMISSION_MATRIX_2026-05-24.md).
Use that matrix before letting TIMR, meta-horizon, active inference,
no-backtrack, LPRM, Q6, ACTION6 tensorizer, registry/gym replay, or ARC
ablation work enter Rexus runtime.

The follow-up order and runtime-specific gates live in
[`../../lab/architecture_sweeps/2026-05-24_rexus_foundation_idea_sweep/reports/RUNTIME_GRADUATION_QUEUE_2026-05-24.md`](../../lab/architecture_sweeps/2026-05-24_rexus_foundation_idea_sweep/reports/RUNTIME_GRADUATION_QUEUE_2026-05-24.md).
That queue keeps all conditional lanes closed until they produce Rexus
`S_t -> candidate A -> O_pred -> score/prune/select -> O_true` evidence,
State Prober labels where legal, and scorecard evidence only for gameplay
claims.

Current guard manifests keep the most tempting downstream lanes explicit and
fail-closed:

- admission consistency:
  [`../../experiments/2026-05-24_rexus_admission_surface_consistency_audit_v0/audit.json`](../../experiments/2026-05-24_rexus_admission_surface_consistency_audit_v0/audit.json)
  verifies that the admission matrix, runtime queue, canon docs, scorer
  manifest, LPRM/Q6 manifest, needs-runner backlog, and training-language audit
  agree that broad GRAM++ is not unlocked and runtime/training admission
  remains zero;
- needs-runner conversion:
  [`../../experiments/2026-05-24_rexus_needs_runner_conversion_backlog_v0/manifest.json`](../../experiments/2026-05-24_rexus_needs_runner_conversion_backlog_v0/manifest.json)
  keeps the 8 Round-1 unclear-review rows that need runners/conversion as
  planning work only, with no runtime or training admission;
- scorer features:
  [`../../experiments/2026-05-24_rexus_scorer_feature_graduation_manifest_v0/manifest.json`](../../experiments/2026-05-24_rexus_scorer_feature_graduation_manifest_v0/manifest.json)
  blocks active inference, no-backtrack, and meta-horizon from runtime until
  their next controlled artifacts exist;
- LPRM/Q6:
  [`../../experiments/2026-05-24_rexus_lprm_q6_downstream_manifest_v0/manifest.json`](../../experiments/2026-05-24_rexus_lprm_q6_downstream_manifest_v0/manifest.json)
  allows only Q6 scorer-interface re-entry and keeps LPRM/Q6 runtime,
  training, and live gated action closed;
- training language:
  [`../../experiments/2026-05-24_rexus_training_authorization_language_audit_v0/audit.json`](../../experiments/2026-05-24_rexus_training_authorization_language_audit_v0/audit.json)
  verifies that bounded row-shape passes are not written as
  training-ready true in human decision docs.

## Ownership

| Component | Canon role | Cannot do |
| --- | --- | --- |
| LeWM | Paper/root world-model architecture: encoder, predictor, SIGReg lineage, action-conditioned transition prediction. | Claim ARC progress without Rexus traces and scorecard evidence. |
| Rexus | Active action-side system: candidate generation, transition prediction surface, memory, pruning, selection, action loop. | Use future `O_true` during online action selection. |
| State Prober | Offline/recorded-condition verifier for `O_pred/O_true` deltas and S/A/O consistency. | Become an online oracle or choose actions with future truth. |
| Q6/Qwen | Optional reasoning/scoring module inside Rexus after candidate transitions exist. | Own actions, substitute for LeWM prediction, or prove target fit from chat metrics. |
| Lewis | Goal/context informer and semantic packet source. | Choose actions, shape reward directly, or become the goal function. |
| GameOps | Signal transport, normalization, hooks, tags, codepoints, and scorecard plumbing. | Become the source of mechanics truth unless a condition-specific scout proves it. |
| ARC scorecard | Goal-function evidence for ARC claims: `levels_completed`, `score`, `level_scores`, `level_actions`. | Be replaced by proxy loss, Lewis confidence, Q6 confidence, or text accuracy. |

## Legal ARCPrize3 Boundary

Legal competition-facing work must preserve these rules:

- **no pre-programmed win conditions**: the agent must discover all moves dynamically via active inference and epistemic exploration over predicted latent states (variational free energy minimization, information gain, and novelty search); the external scorecard remains a sparse environmental feedback, never hardcoded inside the agent's scoring, lattice, or policy transition logic;
- no hardcoded answers or route scripts as the model policy;
- no leaked held-out truth in online action selection;
- no recorded future `O_true` in live scorer inputs;
- no proxy result reported as scorecard evidence;
- no Q6/Qwen chat metric reported as Rexus target fit;
- no promotion without same-condition comparator, complete S/A/O traces, and
  scorecard fields for ARC claims.

## Canon Data Surfaces

| Surface | Canon use |
| --- | --- |
| `experiments/remote_bad_runs/` | Raw negative input only. Convert into registered S/A/O rows with source hashes, frame/action hashes, outcomes, and split policy. |
| `experiments/2026-05-23_timr_micro_mcts_labeled_v0/` | Corpus scaffold and invalid-action label/balance audit. v0 has null `O_pred`. |
| `experiments/2026-05-24_rexus_testing_debt_deep_v0/` | Testing-debt and blocker audit. Use its replay-fidelity blocker when choosing repairs. |
| `docs/TRAINING_DATASET_BIBLE.md` | Dataset row contract: provenance, S/A/O identity, split policy, quarantine rules. |
| `lab/foundations/NEEDS_RESEARCH.md` | Research queue for mobility, recovery, ARC adapter, and motor probes. |

## Sequestered Surfaces

These are retained for history or diagnostics only:

| Surface | Reason |
| --- | --- |
| `experiments/2026-05-23_q6_rejection_dataset_v1/` | Q6 rejection/chat/scorer material. It is not the LeWM/Rexus predictor surface. |
| `experiments/2026-05-23_rexus_q6_shadow_scout_v0/` | Label-reference scorer only; no actual Q6 runtime, no `O_pred`, no scorecard. |
| `experiments/2026-05-24_q6_adapter_fixed_candidate_baseline_v0/` | Actual adapter diagnostic failed clean contract and rejected all candidates. |
| stale `lab/comm/shared/*qwen*` notes | Historical unless refreshed to point at the current forward surface lock. |

## Promotion Gates

The next architecture milestone is not another Q6 run. It is a controlled
Rexus/LeWM predictor run with:

- `game_id`, `episode_id`, `step_id`, and row-scoped `candidate_key`;
- bare `candidate_id` only with `source_opred_row_id` or equivalent row scope;
- state encoding or frame reference;
- action payload;
- non-null `O_pred`;
- `O_true` only under legal offline policy;
- source path and hash;
- split and truth status;
- State Prober delta report.

After that exists, Q6 may be evaluated as a scorer over the same candidate
rows. A live/gated action loop comes only after scorer evidence is clean and a
legal scorecard condition exists.

## Change History Index

Read these in order when reconstructing why the canon changed:

| Date | Link | What changed |
| --- | --- | --- |
| 2026-05-09 | [`docs/loom_direction.md`](../loom_direction.md) | Original layered LeWM direction: validate Layer 0 before adding extra layers. |
| 2026-05-22 | [`docs/TRAINING_DATASET_BIBLE.md`](../TRAINING_DATASET_BIBLE.md) | Dataset truth contract: S/A/O rows, provenance, splits, quarantine. |
| 2026-05-22 | [`docs/REXUS_TRAINING_STRATEGY_2026-05-22.md`](../REXUS_TRAINING_STRATEGY_2026-05-22.md) | Training stack order: LeWM/Rexus dynamics before helper scorers and policy. |
| 2026-05-22 | [`docs/GRAM_LEWM_INTERNAL_PATH_PROCESSING_2026-05-22.md`](../GRAM_LEWM_INTERNAL_PATH_PROCESSING_2026-05-22.md) | GRAM/LPRM/LDT translated into offline path labels and online no-future-truth rules. |
| 2026-05-23 | [`docs/REXUS_CORE_ACTION_LOOP.md`](../REXUS_CORE_ACTION_LOOP.md) | Canonical 12-step Rexus action-loop interface and required trace fields. |
| 2026-05-23 | [`lab/comm/latest/REXUS_CANONICAL_NAME_LOCK_2026-05-23.md`](../../lab/comm/latest/REXUS_CANONICAL_NAME_LOCK_2026-05-23.md) | Naming lock: active system is Rexus; Q6 is a component, not action owner. |
| 2026-05-23 | [`lab/comm/latest/QWEN_LEWM_TARGET_LOCK_DO_NOT_LOSE.md`](../../lab/comm/latest/QWEN_LEWM_TARGET_LOCK_DO_NOT_LOSE.md) | Q6 boundary: only scorer over Rexus transitions, never standalone target proof. |
| 2026-05-23 | [`lab/comm/latest/QWEN_LEWM_TARGET_FIT_MATRIX_2026-05-23.md`](../../lab/comm/latest/QWEN_LEWM_TARGET_FIT_MATRIX_2026-05-23.md) | Target-fit matrix, now diagnostic boundary rather than foundation roadmap. |
| 2026-05-24 | [`lab/comm/latest/LEWM_FORWARD_SURFACE_LOCK_2026-05-24.md`](../../lab/comm/latest/LEWM_FORWARD_SURFACE_LOCK_2026-05-24.md) | Current forward lock: Rexus/LeWM `O_pred` before Q6. |
| 2026-05-24 | [`lab/comm/latest/NOT_FORWARD_LEWM_QUARANTINE_2026-05-24.md`](../../lab/comm/latest/NOT_FORWARD_LEWM_QUARANTINE_2026-05-24.md) | Quarantine index for Q6-centered and non-forward surfaces. |
| 2026-05-24 | [`docs/REXUS_GRAM_PLUS_PRETRAIN_REQUIREMENTS_2026-05-24.md`](../REXUS_GRAM_PLUS_PRETRAIN_REQUIREMENTS_2026-05-24.md) | GRAM++ pretraining architecture gate: operating picture, path fan, lattice, LPRM feature contract, memory, and leakage audit before training. |
| 2026-05-24 | [`experiments/2026-05-24_rexus_gram_plus_path_probe_v0/REPORT.md`](../../experiments/2026-05-24_rexus_gram_plus_path_probe_v0/REPORT.md) | First RexusGramPlusPathProbe proxy scout; scaffold works, training blocked on complete candidate bank and explicit recursive path fan. |
| 2026-05-24 | [`experiments/2026-05-24_rexus_candidate_bank_predictor_probe_v0/REPORT.md`](../../experiments/2026-05-24_rexus_candidate_bank_predictor_probe_v0/REPORT.md) | CandidateBankPredictorProbe V0; complete candidate-bank `O_pred` for bounded held-out states, recursive path fan still blocked. |
| 2026-05-24 | [`experiments/2026-05-24_rexus_gram_plus_path_probe_v1_pathfan/REPORT.md`](../../experiments/2026-05-24_rexus_gram_plus_path_probe_v1_pathfan/REPORT.md) | Bounded recursive path-fan scout; GRAM++ row-shape checklist passes for 4 states, still proxy only. |
| 2026-05-24 | [`lab/architecture_sweeps/2026-05-24_rexus_foundation_idea_sweep/reports/REXUS_ADMISSION_MATRIX_2026-05-24.md`](../../lab/architecture_sweeps/2026-05-24_rexus_foundation_idea_sweep/reports/REXUS_ADMISSION_MATRIX_2026-05-24.md) | Admission matrix for which loose ideas can enter Rexus versus remain proxy/quarantine. |
| 2026-05-24 | [`experiments/2026-05-24_rexus_scorer_feature_graduation_manifest_v0/manifest.json`](../../experiments/2026-05-24_rexus_scorer_feature_graduation_manifest_v0/manifest.json) | Scorer-feature graduation manifest; active inference, no-backtrack, and meta-horizon remain runtime-blocked until next controlled artifacts. |
| 2026-05-24 | [`experiments/2026-05-24_rexus_lprm_q6_downstream_manifest_v0/manifest.json`](../../experiments/2026-05-24_rexus_lprm_q6_downstream_manifest_v0/manifest.json) | LPRM/Q6 downstream manifest; Q6 scorer interface is allowed, but LPRM/Q6 runtime, training, and live gated action remain closed. |
| 2026-05-24 | [`experiments/2026-05-24_rexus_training_authorization_language_audit_v0/audit.json`](../../experiments/2026-05-24_rexus_training_authorization_language_audit_v0/audit.json) | Training-language guard audit; row-shape passes must not be written as training-ready true. |
| 2026-05-24 | [`experiments/2026-05-24_rexus_needs_runner_conversion_backlog_v0/manifest.json`](../../experiments/2026-05-24_rexus_needs_runner_conversion_backlog_v0/manifest.json) | Needs-runner conversion backlog; 8 remaining unclear-review rows have next artifacts and remain runtime/training closed. |
| 2026-05-24 | [`experiments/2026-05-24_rexus_admission_surface_consistency_audit_v0/audit.json`](../../experiments/2026-05-24_rexus_admission_surface_consistency_audit_v0/audit.json) | Cross-surface consistency audit; current docs and manifests agree that runtime/training admission is zero and broad GRAM++ is not unlocked. |
| 2026-05-24 | [`docs/canon_architecture/DOC_MAP.md`](DOC_MAP.md) | Doc graph connecting active canon, historical context, and superseded notes. |
| 2026-05-24 | [`docs/canon_architecture/README.md`](README.md) | Enforced discovery-driven active inference boundary: agent must discover all moves without programmed win conditions. |

## Update Rule

When the architecture changes:

1. Update this file first.
2. Add a dated row to the change history index.
3. Link the condition artifact, report, or code path that proves the change.
4. State whether the change is paper-faithful, adapted-paper-grounded, proxy,
   or invalid.
5. Do not delete old history; quarantine or supersede it with a dated reason.
