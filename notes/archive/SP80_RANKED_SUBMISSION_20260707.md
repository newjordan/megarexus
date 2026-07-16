# SP80 Ranked Submission - 2026-07-07

## Scope

SP80-only ranked ARC-AGI-3 competition-mode scorecard using the Megarexus DW directgrid policy.

This is not a full 25-game portfolio submission. The scorecard is ranked/competition-mode, but only `sp80-589a99af` was actively played.

## Source Revision

- Commit: `6ee941aabc9dd11725f699fad3998ba271bad27f`
- Source URL attached to scorecard: `https://github.com/newjordan/megarexus/tree/6ee941aabc9dd11725f699fad3998ba271bad27f`
- Commit URL: `https://github.com/newjordan/megarexus/commit/6ee941aabc9dd11725f699fad3998ba271bad27f`

## Preconditions

Syntax and whitespace gates passed before live submission:

```bash
python3 -m py_compile codexlab/arc_live_suite_runner.py codexlab/arc_suite_runner.py codexlab/arc_play_harness.py synthesis_lab/orchestration/sim3d/dw_solve.py synthesis_lab/orchestration/sim3d/virtual_arranger.py synthesis_lab/orchestration/sim3d/imagination_kernel.py synthesis_lab/orchestration/sim3d/arranger_gate.py synthesis_lab/orchestration/sim3d/dynamics_fit.py synthesis_lab/orchestration/sim3d/r4_loader_regress.py
git diff --cached --check
```

Same-policy offline preflight:

```bash
python3 codexlab/arc_suite_runner.py --policy megarexus_dw_directgrid --games sp80-589a99af --max-actions 220 --run-id arc_suite_megarexus_dw_directgrid_preranked_20260707T1515Z
```

Preflight result: score `100.0`, levels `6/6`, actions `157`, resets `0`, policy errors `0`.

## Ranked Submission Command

```bash
python3 codexlab/arc_live_suite_runner.py --allow-live --competition --policy megarexus_dw_directgrid --games sp80-589a99af --max-actions 220 --run-id arc_ranked_sp80_megarexus_dw_directgrid_20260707T1623Z --source-url https://github.com/newjordan/megarexus/tree/6ee941aabc9dd11725f699fad3998ba271bad27f --tag agent --tag megarexus --tag sp80 --tag dw_directgrid --tag ranked --opaque-json '{"commit":"6ee941aabc9dd11725f699fad3998ba271bad27f","scope":"sp80-589a99af-only","preflight":"codexlab/runs/arc_suite_megarexus_dw_directgrid_preranked_20260707T1515Z/result.json"}'
```

## Result

- Scorecard ID: `cb30f3dd-48fd-4a6e-8a07-60e41a69d8e1`
- Public scorecard page: `https://arcprize.org/scorecards/cb30f3dd-48fd-4a6e-8a07-60e41a69d8e1`
- Public API: `https://arcprize.org/api/v3/scorecards/cb30f3dd-48fd-4a6e-8a07-60e41a69d8e1`
- `competition_mode`: `true`
- Competition aggregate score: `4.0`
- Competition total environments: `25`
- Competition environments completed: `1`
- Competition total levels completed: `6`
- Total actions: `158`

SP80 row:

- Game: `sp80-589a99af`
- Game score: `100.0`
- Levels completed: `6/6`
- Actions: `158` in public API, `157` policy-issued actions locally
- Resets: `1` in public API competition accounting, `0` policy-triggered resets locally
- State: `WIN`
- Level actions: `[13, 8, 13, 34, 48, 42]`
- Level scores: `[115, 115, 115, 115, 115, 115]`
- Policy errors: `0`
- Policy trace: `plans_built=6`, `direct_actions=157`, `fallback_actions=0`

The aggregate is `4.0` because competition mode scores over the full 25-environment set while this run only attempted SP80.

## Artifacts

- Local run directory: `codexlab/runs/arc_ranked_sp80_megarexus_dw_directgrid_20260707T1623Z/`
- Review page: `codexlab/runs/arc_ranked_sp80_megarexus_dw_directgrid_20260707T1623Z/index.html`
- Raw result: `codexlab/runs/arc_ranked_sp80_megarexus_dw_directgrid_20260707T1623Z/result.json`
- Manifest: `codexlab/runs/arc_ranked_sp80_megarexus_dw_directgrid_20260707T1623Z/report_manifest.json`
- Trace: `codexlab/runs/arc_ranked_sp80_megarexus_dw_directgrid_20260707T1623Z/report_trace.jsonl`

Artifact hashes:

```text
03b1388e5b4e9018647dbce4f8ea5c13e2b200bf281931f63511f242cea6511f  result.json
639b96ebb684d8fe24de8bf9b6547148355774b5a005875776a8a0b80f8df6bd  index.html
87aa5e75677a33fb813eacce8434d2210f2ca4403d1fdf0f6cd7d69dacfa3cd2  report_manifest.json
ed5aa482f44bfaeeed81c3691fe5107f42f2edc697c2e0aa4fcc5b79ae6442eb  report_trace.jsonl
```

## Link Checks

Verified on 2026-07-07:

- `curl -I https://arcprize.org/scorecards/cb30f3dd-48fd-4a6e-8a07-60e41a69d8e1` returned `HTTP/2 200`.
- `curl -sS https://arcprize.org/api/v3/scorecards/cb30f3dd-48fd-4a6e-8a07-60e41a69d8e1` returned the ranked card JSON with `competition_mode:true`.
- `curl -I http://100.124.153.1:18790/index.html` returned `HTTP/1.0 200 OK`.
- `curl -I http://192.168.1.176:18790/index.html` returned `HTTP/1.0 200 OK`.

Detached review server:

```text
PID 754451: python3 -m http.server -b 0.0.0.0 -d $REPO/codexlab/runs/arc_ranked_sp80_megarexus_dw_directgrid_20260707T1623Z 18790
```

## Boundary

This is real ranked/competition-mode ARC scorecard evidence for SP80. It is not evidence that the current system solves the other 24 public environments, and it should not be reported as a full portfolio solve.
