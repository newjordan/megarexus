# SP80 Live Scorecard Report - 2026-07-07

## Scope

This report covers only `sp80-589a99af` under the
`megarexus_dw_directgrid` policy. It does not cover r11l, ft09, LS20, the
portfolio runner, or any broad ARCPrize3 generalization claim.

## Bottom Line

`sp80-589a99af` has a recorded live ARC SDK scorecard close:

- Scorecard ID: `8820f639-fe4d-438f-9129-83abe0646843`
- Run directory:
  `codexlab/runs/arc_live_sp80_megarexus_dw_directgrid_20260707T125653Z/`
- Result: `6/6` levels, score `100.0`, `157` actions, `0` resets,
  `0` policy errors, final state `GameState.WIN`
- Evidence category: `live_scorecard`
- ARC mode in game row: `live_scorecard`, `OperationMode.NORMAL`

This is real live SP80 score evidence. It is not a broad ARCPrize3 claim, not a
multi-game leaderboard result, and not proof of generalization beyond SP80.

## Live Condition

Command recorded in `result.json`:

```bash
codexlab/arc_live_suite_runner.py --allow-live --policy megarexus_dw_directgrid --games sp80-589a99af --max-actions 220 --run-id arc_live_sp80_megarexus_dw_directgrid_20260707T125653Z --out-dir codexlab/runs/arc_live_sp80_megarexus_dw_directgrid_20260707T125653Z
```

Recorded condition:

```text
live ARC scorecard policy=megarexus_dw_directgrid max_actions_per_game=220
```

The SDK returned `source_url=null`, so there is no public source URL to verify
from this artifact. The controlling evidence is the local SDK result payload and
its report artifacts.

## Live Result Details

| Field | Value |
| --- | --- |
| Game | `sp80-589a99af` |
| Policy | `megarexus_dw_directgrid` |
| Score | `100.0` |
| Levels completed | `6/6` |
| Actions | `157` |
| Resets | `0` |
| Final state | `GameState.WIN` |
| Policy errors | `0` |
| First frame hash | `4537a63ec78645d0b682cfc1bc96b1d1dfd54acb52276e0451bf1a2358084198` |
| Last frame hash | `0c75d3a0d4b3a2e6f6624cf5e872dc376e6ee2c3b236222045407d065b7198de` |

Per-level action counts:

| Level | Actions | Baseline actions | Advanced |
| --- | ---: | ---: | --- |
| 0 | 12 | 39 | yes |
| 1 | 8 | 58 | yes |
| 2 | 13 | 25 | yes |
| 3 | 34 | 148 | yes |
| 4 | 48 | 96 | yes |
| 5 | 42 | 152 | yes |

## Pre-Live Gate

The live run was preceded by a current-code local/offline gate:

- Run directory:
  `codexlab/runs/arc_suite_megarexus_dw_directgrid_current_20260707T1308Z/`
- Command:

```bash
codexlab/arc_suite_runner.py --policy megarexus_dw_directgrid --games sp80-589a99af --max-actions 220 --run-id arc_suite_megarexus_dw_directgrid_current_20260707T1308Z --out-dir codexlab/runs/arc_suite_megarexus_dw_directgrid_current_20260707T1308Z
```

Recorded local/offline result:

- `6/6` levels
- score `100.0`
- `157` actions
- `0` resets
- final state `GameState.WIN`
- `0` policy errors

The offline gate is supporting evidence only. The live scorecard above is the
actual SP80 score evidence.

## Policy Character

The live policy was not a Tufa ledger replay. It was also not a broad learned
generalist policy.

The implementation is `megarexus_dw_directgrid_policy()` in
`codexlab/arc_play_harness.py`. Its exported trace says:

```text
L0_L3_current_frame_scored_DW_fixture_genome_priors_then_r4_genome_search;
L4_current_frame_scored_DW_direct_priors_then_r5_direct_grid;
L5_current_frame_scored_DW_direct_grid_priority_then_direct_priors;
offline_report_is_not_live_proof
```

Live policy trace summary:

- `plans_built=6`
- `direct_actions=157`
- `fallback_actions=0`
- no fallback policy initialized
- no policy errors

Planner provenance by level:

| Level | Planner event | Action count | Model/search | Fixture/prior |
| --- | --- | ---: | --- | --- |
| 0 | `plan_built` | 12 | r4 genome prior, kernel fitness | `dw_solve_r4_L0_s101_20260706.json` |
| 1 | `plan_built` | 8 | r4 genome prior, kernel fitness | `dw_solve_r4_L1_s101_20260706.json` |
| 2 | `plan_built` | 13 | r4 genome prior, kernel fitness | `dw_solve_r4_L2_s101_20260706.json` |
| 3 | `plan_built` | 34 | r4 genome prior, kernel fitness | `dw_solve_r4_L3_s555_20260706.json` |
| 4 | `defer_low_confidence_l4_plan` | n/a | r5 direct prior, rejected low confidence | `dw_solve_r5_L4_s101_directgrid_20260706.json` |
| 4 | `plan_built` | 47 | r5 direct prior, kernel fitness | `dw_solve_r5_L4_s101_directgrid_20260706.json` |
| 5 | `plan_built` | 42 | r5 direct grid, geometry priority | none recorded |

Interpretation: the SP80 result is a prepared, game-specific DW solver/planner
that reads the current frame and selects or searches SP80 pour plans. It should
be reported as a successful SP80 solver, not as ARCPrize-wide intelligence.

## Artifact Hashes

Live run:

| Path | SHA256 |
| --- | --- |
| `codexlab/runs/arc_live_sp80_megarexus_dw_directgrid_20260707T125653Z/result.json` | `3d0869afd7781b8efc7232474010e7e9a6d32cbd7bc532a233ad2cf853acbf7e` |
| `codexlab/runs/arc_live_sp80_megarexus_dw_directgrid_20260707T125653Z/index.html` | `2215bc50c180de8f8b67a4285bc226d26763890cfd4c2e08cc6a9c61d0508efb` |
| `codexlab/runs/arc_live_sp80_megarexus_dw_directgrid_20260707T125653Z/report_manifest.json` | `109415c515e36360b469074bca0be5e92cdb4fba668e7f33c8504911633a69b4` |
| `codexlab/runs/arc_live_sp80_megarexus_dw_directgrid_20260707T125653Z/report_trace.jsonl` | `ab887dbca510500c102dc2a5f389349a35294a331bff1a3edd0fe11abfe428d1` |
| `codexlab/runs/arc_live_sp80_megarexus_dw_directgrid_20260707T125653Z/review_screenshot.png` | `7e5ac745168324fb90389e6d5cf6531b164959bc4e396637804f1b7093ec15f0` |

Offline pre-live gate:

| Path | SHA256 |
| --- | --- |
| `codexlab/runs/arc_suite_megarexus_dw_directgrid_current_20260707T1308Z/result.json` | `bb32edbac52b01b8c2ed1f40ca6cbec312fd6372e11c6a9bbe6a71b73feacb76` |
| `codexlab/runs/arc_suite_megarexus_dw_directgrid_current_20260707T1308Z/index.html` | `14495a6798e5da9972a836d05b0b4ba53c1548411ea616e8b24ecfcde0876c30` |

Source files inspected:

| Path | SHA256 |
| --- | --- |
| `codexlab/arc_play_harness.py` | `a50efacd0ac57247652230c5a821cea3ff9a8e5bcd44e68cef534ce516b1c018` |
| `synthesis_lab/orchestration/sim3d/dw_solve.py` | `50bdfd3b6bf59c15d1f46a449586dc6bca5e50c6d51e9ec4357f55aabb359143` |

## Boundary

Use this result as:

- real SP80 live scorecard evidence
- evidence that the `megarexus_dw_directgrid` SP80 policy can close all six
  SP80 levels under the ARC SDK live scorecard path
- evidence that the current offline SP80 gate matched the live result on action
  count and outcome

Do not use this result as:

- a broad ARCPrize3 solved claim
- evidence that r11l, LS20, or any other downloaded game is solved
- a learned generalist result
- a public URL claim, because the SDK artifact records `source_url=null`
