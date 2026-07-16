# LeWM Paper Passgate

Status: `adapted-paper-grounded`
Surface: `other`
Task: Loom canon architecture entrypoint and documentation map

Primary paper: Maes et al., `LeWorldModel`, arXiv:2603.19312v2.

## Required Checks

| check | status | note |
|---|---|---|
| `primary_paper_read` | `pass` | Used LeWM paper anchor from passgate skill |
| `provenance_artifacts_recorded` | `pass` | Canon doc, doc map, and linked history saved |
| `surface_classified` | `pass` | Architecture doc for ARC adaptation, not PushT parity |

## All Checks

| check | status | note |
|---|---|---|
| `primary_paper_read` | `pass` | Used LeWM paper anchor from passgate skill |
| `surface_classified` | `pass` | Architecture doc for ARC adaptation, not PushT parity |
| `offline_obs_action_trajectories` | `unchecked` |  |
| `behavior_policy_recorded` | `unchecked` |  |
| `frame_contract_recorded` | `unchecked` |  |
| `frame_skip_or_deviation_recorded` | `unchecked` |  |
| `no_reward_or_task_spec_in_worldmodel_training` | `unchecked` |  |
| `lewm_loss_matches` | `unchecked` |  |
| `decoder_not_core_training` | `unchecked` |  |
| `goal_observation_and_terminal_latent_cost` | `unchecked` |  |
| `cem_mpc_setup_recorded` | `unchecked` |  |
| `eval_budget_goal_sampling_recorded` | `unchecked` |  |
| `push_t_condition_recorded_for_comparison` | `unchecked` |  |
| `arc_deviation_labeled` | `unchecked` |  |
| `provenance_artifacts_recorded` | `pass` | Canon doc, doc map, and linked history saved |

## Evidence

- `external/loom/docs/canon_architecture/README.md` exists=True sha256=443409c64217d35a9fcaaea28e865a260ea0b7479969bc026066cd12f5c9b33a
- `external/loom/docs/canon_architecture/DOC_MAP.md` exists=True sha256=79c582cd39e5b507f0728a01c24723d66e84ae5fda9e650d4dc8d9741696d9ab
- `external/loom/docs/REXUS_CORE_ACTION_LOOP.md` exists=True sha256=36b1e054b9c95281fa74e9569c941f3f38fc0fcaf042f0517c8675388e845b06
- `external/loom/lab/comm/latest/LEWM_FORWARD_SURFACE_LOCK_2026-05-24.md` exists=True sha256=806518adabfc559646aba831e6ee848b668a1ba01ce14b6c09e13956c422ca66

## Deviations

- ARCPrize3 discrete-action architecture, not PushT parity
- Document set is architecture canon, not a training/eval result

## Note

Current canon requires Rexus/LeWM candidate O_pred before Q6 scorer work.
