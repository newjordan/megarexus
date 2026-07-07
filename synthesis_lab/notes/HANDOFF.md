# HANDOFF — synthesis_lab (megarexus), written 2026-07-01

For a fresh agent session picking up this work. Read this top to bottom
before touching anything. The two notes files are the ground truth:
`notes/megarexus_cnn_scout.md` (full experiment detail) and
`notes/RESEARCH_LOG.md` (dated summary entries). User memory lives in
`~/.claude/projects/-home-frosty40-megarexus/memory/`.

## 0-NEXT (2026-07-07): RANKED RUN — build the continuous-episode runner
Mission line: observation-induced kernel now solves L0-L3 per-episode (L3 4/5 after
the user's hints exposed the starved search — see round-4b/L3-retraction log entries).
Non-ranked smoke on fresh seed 555 launched at wrap (results in scratchpad smoke555.txt
or rerun: dw_solve per level, 6.6x budget for L2/L3). THE ONE BUILD ITEM before the
ranked scorecard: a CONTINUOUS-EPISODE runner — dw_solve's propose->imagine->pour loop
re-pointed at the live frame after each level win (no _present per level, no resets
mid-episode), global action budget respected, refusal-safe. Then: offline full-game
rehearsal, then the ranked ARC-AGI-3 scorecard (closed SDK payload = controlling
evidence). Round-5 induction lane (L4/L5 color-15 "orthogonal blocking", corpora
harvested) proceeds in parallel — ranked run does not wait for L4/L5.
honesty_audit at wrap: active 85/23/0. Branch pushed through a5d2a8c.

UPDATE 2026-07-06 ~21:00 (cutoff recovery session): the runner's enabling piece is
DONE and CERTIFIED — `sim3d/virtual_arranger.py` realizes arrangement candidates
from the LIVE frame with zero env interaction; `sim3d/arranger_gate.py` PASS:
480/480 pre-pour byte-match, 72/72 kernel-fitness equivalence, 48/48 post-pour
re-plan, 0 refusals (L0-L3 × seeds 555/1234/42 × 40 genomes,
`fixtures/arranger_gate_20260706.json`). Two induced mechanics landed on the way
(see DW_LAB_LOG round entry): units are 1 lattice cell tall (stacked touching
blobs band-split); overlap click hit-test = smallest footprint wins, ties to
latest scan order. Remaining build: the runner loop itself, then rehearsal, then
the ranked scorecard.

## 0-LIVE STATE (2026-07-06 ~22:30 CDT) — DW ROUND-4: FILM-FLOW KERNEL, L2 FRONTIER BROKEN (L0+L1+L2 SOLVED@1), L3 = SEALED NEGATIVE + ESCALATION (supersedes all below)

**READ `notes/DW_LAB_LOG_20260706.md` round-4 entries FIRST.** Round-4 arc:
1. **Phase A (L3 corpus observation):** the flow is a liquid-film TRAIL (persistent
   trails, spread-and-split at obstacle tops, ALL housings emit — "charged-only"
   falsified; the old L3 zero-shot lava_acc 0.00 was a threshold artifact, fixed by
   the row-flash rule after exact regression on all 4px corpora).
2. **Phase B/C:** lattice-aware `load_pours` (gate: byte-identical masks + labels on
   4px corpora) + generic `FlowTrail` family (5 knobs incl. merge ∈ {split,die}) →
   `r4_fit.py` over L0+L1+solvetime+L3, per-corpus holdouts ALL PASS; FlowTrail
   sweeps FallingMaterial (IoU 0.96/fdiv 25 vs 0.68/6); **L2 zero-shot exam: outcome
   1.00 / lava 1.00 / IoU 0.96 / fdiv 25** (round-3 was fdiv 5 — Correction B's
   multi-source infidelity is gone). Fixture `dynamics_round4_20260706.json`.
3. **Phase D:** dw_solve --model r4: **L0 5/5 + L1 5/5 SOLVED@1 (default budget);
   L2 SOLVED@1 kernel-certified (imagined clean win → env win, 1 real pour/seed,
   0 search pours) — 5/5 under the merge=split member, 3/5 under the IoU-tie-broken
   merge=die selected member (both gate-certified; fixtures record their member;
   split runs preserved as `dw_solve_r4split_*`).** The r3-era "L2 at 6.6× budget
   never imagines a clean 3-stream pour" is broken at the same 9,664-eval budget.
4. **L3: pre-registered NEGATIVE, sealed** (`r4_l3_feasibility.py` ~500k+ kernel-space
   placement evals with env-probed TRUE movables → min floor=1; structural cause:
   mouth pair (14,18) is 4 apart, only the STATIC 3-wide wings could split 4-apart;
   NEW OBSERVED FACTS: fills reset every pour, 4-pour budget, 5th pour = GAME_OVER;
   two diagnostic pours at the frontier: env flashed exactly as the kernel imagined,
   incl. a PERFECTLY realized floor-1 optimum via `r4_l3_realize.py`). **ESCALATED:
   the intended-L3-mechanic question is with the user.** L4/L5 still need a color-15
   hypothesis (unchanged).
5. Queued next: round-5 flywheel refit (fold this session's env-graded solve/diagnostic
   pours in; win-side data may break the split/die IoU tie), then L4/L5 recon.
honesty_audit at wrap: exit 0 — total 97/46/6, active 85/23/0 (no new flags).
Round-4 lane committed this session (sim3d code + fixtures + notes).

## 0-LIVE STATE (2026-07-06 ~16:30 CDT) — DW ROUND-3: L0+L1 SOLVED@1 5/5 SEEDS, KERNEL CERTIFIES (superseded)

**READ `notes/DW_LAB_LOG_20260706.md` FIRST** (append-only lane log; plan =
`DREAMWEAVER_INTEGRATION_PLAN_20260706.md`). Today's arc, in order:
1. **Round-2 5-seed sweep = HONEST NEGATIVE** (L0 3/5, L1 1/5; fitness plateau —
   kernel imagined lava 0.76–0.84 on real clean winners; single-seed SOLVED@1 was
   plateau luck). Root cause: round-2's fit truncated rollouts at animation length,
   hiding capacity-cap overflow that dw_solve's 200-tick rollouts exposed.
2. **Round 3 (the flywheel):** replayed all 31 sweep candidates as observation data
   (`sim3d/r3_flywheel.py`); real lava separates winners/losers 31/31; winner
   animations terminate early. Family extended: `absorb ∈ {cap,unlimited}` +
   horizon-independent `win_moment` semantics. Full-grid re-selection
   (`sim3d/r3_fit.py` → `dynamics_round3_20260706.json`): **absorb=unlimited**
   (mouths never back up), train+holdout outcome/lava 1.00.
3. **Round-3 sweep (uncontaminated): 10/10 SOLVED@1** — L0 5/5 + L1 5/5, seeds
   {1234,7,42,101,2026}, every rank-1 `imagined_win=True` env-confirmed. The
   INDUCED kernel now CERTIFIES wins. One real pour per solve, zero search pours.
   Fixtures `sim3d/fixtures/dw_solve_r3_L*_s*_20260706.json`.
4. **L2: NOT solved** (default AND 6.6× budget, 9,664 evals — kernel never imagines
   a clean 3-stream pour; top-4 tie ~0.946, all real pours lose). L2 work order
   stands unchanged: `TRAINING_REGIME_20260706_L2_FRONTIER.md` (P0 multi-src
   fidelity, P2 mechanic-aware proposer). L3–L5 extractor fix (pixels-only) also open.
Nothing committed today — user decides when to stage the DW lane.

## 0-LIVE STATE (2026-07-06 ~02:30 CDT) — SIM LANE DONE, L2 WALL DIAGNOSED, REGIME PIVOTED (superseded)

**READ `notes/TRAINING_REGIME_20260706_L2_FRONTIER.md` FIRST — it is the pick-up
work order.** Summary: the sim3d loop finished 5/6 stages
(`SIM3D_LOOP_STATUS_20260706.md`); headline hardened at **2 levels in-budget
(L0 CNN-alone, L1 sim-guided SOLVED@1, 5/5 seeds)**. L2 hit a wall that went
through two corrections: NOT action-space-blocked (that claim was confounded) —
the multi-source sim (L2 = 3 sources) under-reports fills because the game fills
on ENCLOSED-droplet ENTRY (game source L775), not settled occupancy. L2
solvability is UNKNOWN until multi-source fidelity lands. Game source fully read
(4-pour/no-accumulation, win=all mouths entered enclosed + no wall-touch, real
redirect = `tuvkdkhdokr`; `sp80_variants.py` REDIR_TAG is mislabeled → v1
variants never had real redirects). New regime, in order: **P0** multi-source
fill fidelity (gates: S0/S2 unchanged + ≥0.90 fill agreement on
`sim3d/fixtures/proposer_l2.npz`) → **P1** variants v2 (fix REDIR_TAG,
multi-source family, S2-MS gate) → **P2** mechanic-aware constructive proposer
→ env L2 attempt (headline 3) → **P3** Stage-E/F training (amortize proposer,
flywheel ranker heads; the only authorized CNN training — static outcome heads
stay falsified/banned). Do NOT pick up: S1 iter3 polish, Lane P, random L2
sweeps. Sim3d lane is uncommitted working-tree; committing it precedes any P3
condition (contract rule 8). honesty_audit at handoff: exit 0, 97/46/6, active
85/23/0.

## 0-LIVE STATE (2026-07-05 ~21:50 CDT) — PER-TARGET LANE DONE: clean NEGATIVE (superseded)

Ran `notes/STAGE_PT_SPEC_20260705.md` end-to-end (T0→T4). **Result: the per-target
decomposition is FALSIFIED at the MODEL granularity — it does NOT move the headline to 2
levels.** Headline unchanged: v52 = ranker, 1 level CNN-alone, L1 only via env-feedback play.
- **T0 PASS:** per-target GROUND TRUTH cleanly separates the verified L1 winner (all targets
  fill=1/lava=0) from v52's 4 FPs (all lava-saturated). Decomposition is real AT THE LABELS.
  (Caught+fixed DEVIATION-1: the pre-registered 3-frame fill window scored the confirmed winner
  as fill=0 — win holds only 2 frames pre-reset; see `runs/T0_DEVIATION_label_window.md`.)
- **T1:** `data/pour_variants_pertarget_20260705.pt` (6479 pours, 94 wins; all 94 wins have every
  target clean — exact per-target↔env correspondence).
- **T2:** additive `head_target_outcome` (C5-clean). v55 frozen + v56 unfrozen, both ~0.93
  held-out win-AUC but WEAK per-target components (fill_AUC~0.6, lava_AUC~0.6).
- **G1 FAILS BOTH** (v55 winner −0.583, v56 −0.789) — yet both FLIP v52's ordering (winner now
  ranked #1 above all FPs), failing only the POSITIVE bar by a ~0.003 noise margin. Decisive
  diagnostic: the head predicts NEAR-IDENTICAL (fill,lava) for winner vs FP (0.16/0.95 for both)
  — blind to the arrangement difference. **The separator is POUR DYNAMICS, not static per-target
  features → §6 confirmed: residual = the precise pour simulator. Per §8, STOP.**
- **Next lever:** a faithful pour SIMULATOR (NOT another static predictor) — user has an idea.
  **READ `notes/HANDOFF_SIMULATOR_20260705.md`** (failures/problems/goals framed for the simulator
  push) + drafted lane `notes/STAGE_CA_SPEC_20260705.md` (Neural-CA, CA0 baseline NOT yet run).
  Decisive cheap test for any simulator idea: roll out the winner + 4 FP genomes (in
  `runs/t0_pertarget_20260705.json`), read per-target (fill,lava) via `pertarget_labels`, check
  winner `min_t(fill−lava)` is positive & above all FPs. New tooling: `orchestration/
  {pertarget_labels,t0_label_test,train_target_outcome,pt_gates}.py`, harvest `--per-target`.
  Full log: NIGHT_REGIME F42-47.

## 0-LIVE STATE (2026-07-05 ~19:10 CDT) — READ `notes/HANDOFF_NIGHT_20260705.md` FIRST (supersedes all below)

Night regime built a shape-based, LAVA-aware pour imagination. Deliverable = **v52**
(`runs/outcome_v52_multibase_unfrozen_apollo_20260705/best.pt`): a Contra-informed CNN that
imagines pour outcomes and GENERALISES across level geometries (REAL-L0 0.98, REAL-L1 0.99;
correctly scores the verified L1 winner). Objective = fill (positive) − lava (negative,
"floor is lava"). Key unlock: multi-base variants (inherit each level's geometry) +
UNFREEZE encoder. NOT done: single-episode 4-pour CNN-alone SOLVE of L1/L2 (argmax precision
wall — every search + hard-neg approach mapped and failed, F41). Full self-contained handoff
+ infra gotchas + the direct-winner verification test + next direction (per-target spatial
predictor) in **`notes/HANDOFF_NIGHT_20260705.md`**. Full log: `notes/NIGHT_REGIME_20260705.md`
F1-41. Code on branch `night-shape-lava-imagination-20260705`.
The per-target lane is now an agent-executable spec: **`notes/STAGE_PT_SPEC_20260705.md`**
(pre-registered labels/model/gates, stages T0-T4, stop rules, banned dead-ends; run T0 first).

## 0-PLAN OF RECORD (2026-07-05) — WORLD-MODEL ROADMAP IS BINDING

`notes/ROADMAP_WORLD_MODEL_20260705.md` is the integration contract for all
agents: Contra→world-model architecture map, frozen contracts (C1–C5), staged
plan A–F with numeric gates and a decision tree. Execute the next stage it
indicates (currently: Stage A — v37/v38 in flight on Apollo; on v38's
`imagine_search_gate` result, follow the tree). Companion evidence:
`notes/REPORT_CNN_DELIBERATION_20260705.md`, `notes/NIGHT_REGIME_20260705.md`.

## 0-LIVE STATE (2026-07-05 ~09:40 CDT) — STAGE A IN FLIGHT (supersedes 07-04 block below)

Scoreboard (roadmap C4). Headline (levels completed CNN-alone, fresh-from-0):
still to be env-grounded (see head_rank_gate below). Gate = `imagine_search_gate`.
- **v31 baseline (imagination):** L0 `fn=2, tp=0, imagined_wins=0`, real_wins=2/3
  (winner in set, imagination misses it) — the pour-imagination wall.
- **v37 (pour-outcome head, 3× L0 positives): DONE-ish.** Fresh-arrangement L0
  ranking **AUC 0.87** (v36 was 0.79; roadmap Stage B ref >0.79). Saturation fixed.
- **v38 (dynamics decoder on spill ticks): QUEUED** behind v37 (watcher on pid).
  **STAGE A1 GATE (run on completion):** `imagine_search_gate --levels 0 --seeds 3`
  → PASS iff `fn=0 & tp≥1`; secondary paint-mass ratio 0.8–1.2 (now reported).
  fn>0 → Stage A2 (add `head_spill_ca` local neural-CA head). fn=0 → Stage B.

Apollo GPU chain auto-advances via detached watchers (each waits on the prior
train pid, launches next — no idle gap). Apollo workspace
`/home/frosty/megarexus_train_20260703`, venv `/home/frosty/Loom/.venv/bin/python`.

New tools this session (all honesty-guarded, env-graded; C5-compliant additions):
`imagine_search_gate.py` (scoreboard + A1 secondary paint-mass), `tta_search_gate.py`
(Stage D machinery, validated), `probe_pour_oracle.py`, `head_rank_gate.py`
(L0 headline grounder: head RANKS real arrangements, top-K real pours, env grades —
running now on v37), `collect_pour_outcomes.py`, `build_pour_shard.py`.
Pending: Stage B (latent settle-predictor + win-probe), Stage C (portband-seeded
CEM with relaxed geometric fitness — the L1 unlock).

## 0-LIVE STATE (2026-07-04 15:52 CDT) — FRESH-FROM-ZERO TRAINING ON APOLLO (SUPERSEDED)

User ruling this session (memory `megarexus-fromzero-training-ruling`): **no
`on_set_level` farming; always a fresh run from 0 (random init); always be
training; master the field.** Training runs on **Apollo** (RTX 4080, dedicated)
— GB10 is shared with another agent's 119B `llama-perplexity` job.

RUNNING NOW:
- **stage62 fresh-from-0 mastery** on Apollo. Condition:
  `runs/stage62_noimit_fromzero_mastery_apollo_20260704/CONDITION.json`
  (init=none/random; 8 clean train + 5 val shards; corpus 68900 train / 12062
  val, 0 human; 40 epochs, bs64, lr3e-4, stack2; venv
  `/home/frosty/Loom/.venv/bin/python` torch 2.5.1+cu121). Out:
  `runs/outcome_v28_noimit_fromzero_mastery_apollo_20260704`. Launched via
  `train_watch.py --max-hours 10`. Apollo workspace
  `/home/frosty/megarexus_train_20260703`; code+shards rsync'd from GB10 this
  session (Apollo train.py now HAS the no-imitation guard).
  Watch: `ssh apollo 'tail -f /home/frosty/megarexus_train_20260703/synthesis_lab/experiments/megarexus_cnn/runs/outcome_v28_noimit_fromzero_mastery_apollo_20260704_train.log'`
- GB10 twin of stage62 (batch32) was STOPPED once Apollo confirmed CUDA (no
  training gap). GB10 collection finished.

INTELLIGENT LOOP (daemon, owns the chain) — running on Apollo:
- `synthesis_lab/orchestration/intelligent_train_loop.py`, PID group under
  1386602, state dir `runs/intelligent_loop_20260704/` (loop.log, loop.out,
  loop_state.json). Attached to stage62; waits for its WATCH_REPORT.json, then
  each cycle: GATE (score every head vs its baseline from history.json, val
  FIXED) → DIAGNOSE weakest head → COLLECT only clean data for it
  (control_play/catch_play/sp80_rollouts/edge_adjoin) → pre-register a
  FRESH-FROM-0 condition (init=none) → guard+dry-run → train_watch → repeat.
  `control_play_v2_train` is folded into the loop's base corpus.
- HALTS fail-closed (writes `NEEDS_HUMAN_*.json` in the state dir) on: the
  two-failure rule (a head twice with no improvement → representation change is
  a HUMAN call), a weakness only injection/imitation could feed (e.g. level-2+
  positives = the D3 bootstrap boundary), or any guard/dry-run refusal. On all
  heads passing it writes `CONVERGED.json` (→ run the full gate stack + live scout).
- Watch: `ssh apollo 'tail -f /home/frosty/megarexus_train_20260703/synthesis_lab/experiments/megarexus_cnn/runs/intelligent_loop_20260704/loop.log'`
- To stop the loop: kill the `intelligent_train_loop.py` PID (in-flight
  train_watch/train.py continue; the loop just won't start the next cycle).

Arrow: **clean-lane CNN-alone levels completed (from 1).** Legacy 28.57/3
stays as the frozen assisted reference. Plan: `notes/REGIME_REVIEW_20260704_REKNOCK.md`.

## 0B-BINDING USER RULING (2026-07-04 11:20 CDT) - NO IMITATION TRAINING, EVER

The user explicitly ruled that imitation training is never authorized in this
project. Treat this as a first-principles constraint, not a temporary
preference:

- No human demonstrations, human-mined states, route traces, route replays,
  same-level solved/eval states, answer files, or graph-demo plans may be used
  as training data for a CNN-alone/autonomous claim.
- Such artifacts may be kept only as quarantined evidence, diagnostics, or
  hypotheses. They cannot feed `train.py`, `train_goal_head.py`, a dataset
  builder, a gate requirement, or a live policy path for any CNN-alone lane.
- "Hindsight labels" are legal only when the underlying episode was generated
  by the agent/environment under a pre-registered, non-imitation condition and
  is split away from the evaluation surface. Hindsight over human/demo/eval
  answer material is still answer training and is banned.
- Stage25/Stage33 goal-label datasets and Stage26/Stage34 goal-head
  checkpoints are now historical diagnostics only for CNN-alone purposes,
  because the recorded dataset-build commands included
  `level3_human_mined_inclusive_stage22b_20260704.pt`. Do not relaunch, extend,
  or promote them. Rebuild the goal-informer lane from no-imitation sources
  before any further CNN training claim.
- The point of the work is causal model competence: predict mechanics from
  interaction, read visible objectives from live pixels after the agent reaches
  them, and pass disjoint held-out gates. Optimizing score artifacts by
  training on answers is the failure mode.

Current clean-regime document:
`synthesis_lab/notes/CNN_TRAINING_REGIME_20260704_NO_IMITATION.md`.

## 0A-POST-AUTHORIZED CONTAINMENT (2026-07-04 11:01 CDT) - ROUTE CHEATS REMOVED FROM ACTIVE SURFACE

The user explicitly authorized removing the route-replay cheats. This
session did containment and split cleanup only: NO training, NO gates, NO
live runs, NO commits.

Applied cleanup:

- Quarantined, not deleted, all v16-v26 route-imitation checkpoints and
  stage36-stage60 route/human/replay artifacts under:
  `synthesis_lab/experiments/megarexus_cnn/runs/_invalid/`.
  The quarantine contains 41 artifact dirs plus the `_invalid` container; the
  six invalid headline gates remain present there as evidence.
- Quarantined route/human/replay data shards under:
  `synthesis_lab/experiments/megarexus_cnn/data/_invalid/`
  (`level3_human_mined*`, `level3_vertical_route*`, `level4_human_mined*`,
  `level5_human_mined*`, `level5_runtime_replay*`).
- Quarantined dedicated route/human-mining tool scripts under:
  `synthesis_lab/orchestration/_invalid/` and route training scripts under
  `synthesis_lab/experiments/megarexus_cnn/_invalid/`.
- Restored tracked `runs/outcome_v9/best.pt` and `runs/outcome_v9/last.pt`
  from git; they no longer show as modified.
- Split active code: `model.py` keeps the Contra goal head
  (`goal_conv`, `head_goal`, `forward_goal`) and generic additive-checkpoint
  loading, but route constants/modules/forward methods are removed.
  `train.py` keeps generic `--init-ckpt`, `--save-every`,
  `--trainable-prefixes`, and goal-head compatibility, but route additive
  prefixes are removed.
- Removed route execution and route schema attachment from
  `synthesis_lab/orchestration/megarexus_live_policy.py`.
- Removed the corrupted `route_provider_attached` PASS check, route-action
  review columns, and `MEGAREXUS_ROUTE_*` env recording from
  `synthesis_lab/orchestration/adapter_gate.py`.
- Updated `honesty_audit.py` so quarantined invalid artifacts are still
  reported and disclosed, but exit code 2 is reserved for ACTIVE invalid
  headlines.

Verification after cleanup:

- `python3 -m py_compile` passed for `model.py`, `train.py`,
  `megarexus_live_policy.py`, `adapter_gate.py`, `honesty_audit.py`, and
  `train_watch.py`.
- `python3 synthesis_lab/orchestration/honesty_audit.py` exits 0 with:
  total summary `AUTONOMOUS_CANDIDATE=82, IMITATION_ASSISTED=46,
  INVALID_HEADLINE=6`; active summary `AUTONOMOUS_CANDIDATE=70,
  IMITATION_ASSISTED=23`; quarantined summary
  `AUTONOMOUS_CANDIDATE=12, IMITATION_ASSISTED=23, INVALID_HEADLINE=6`.
  This is the desired state: invalid headlines are not active, but still
  disclosed.
- `python3 synthesis_lab/orchestration/train_watch.py --condition
  synthesis_lab/experiments/megarexus_cnn/runs/stage23_v13_level3_human_mined_20260704/CONDITION.json
  --dry-run` refuses with four problems: `train.py` hash mismatch,
  `model.py` hash mismatch, `gate_eval_many.py` hash mismatch, and missing
  quarantined shard
  `data/level3_human_mined_inclusive_stage22b_20260704.pt`.

Remaining state:

- Stage23/v13 remains SUSPENDED. The human-mined shard is quarantined, so
  train_watch cannot launch it.
- Active audit still marks Stage20/32 graph-demo-plan and Stage22/23 human
  demo lineage as IMITATION_ASSISTED. Do not report those as CNN-alone
  progress.
- The cleaned toolchain is still uncommitted. Next valid action is a scoped
  commit of the cleanup/toolchain if the user asks for it.

## 0-SESSION HANDOFF (2026-07-04 midday CDT) - POST-INCIDENT GUARDRAILS; NO TRAINING LAUNCHED; V13 AUTHORIZATION SUSPENDED

This session did forensics on the overnight 2026-07-04 route-replay incident
and built enforcement. It launched NO training, NO gates, NO live runs. It
deleted, reverted, and committed nothing. Read this block plus the EVIDENCE
CONTRACT below; do not re-read the older history sections, the stage36-60
gate JSONs, or COURSE_ADVISORY_20260703 to resume - the advisory's v13
pre-authorization is SUSPENDED (see "Blocked / user decisions" below).

What happened overnight (verified on disk, not from the session's report):

- Stages 36-60 ran between 01:55 (last notes write) and 10:29 with no log
  discipline. Human answers were mined for levels 3/4/5
  (`collect_level4_human_mined.py`, `collect_level5_human_mined.py`,
  `collect_level5_runtime_replay.py`, all untracked), eleven route-imitation
  checkpoints were trained (`outcome_v16_*` through `outcome_v26_*`), and the
  runtime was step-indexed through mined routes via `MEGAREXUS_ROUTE_*`
  flags.
- The adapter gate was reshaped so `route_provider_attached` is a PASS
  requirement. Six PASS gates are corrupted this way: stage44, stage45,
  stage50, stage51, stage59, stage60. Only stage60 carries a disclosure
  (`AUTONOMOUS_SOLVER_INVALIDATION.md`, written 10:29:10, the same minute
  the final gate closed - the session knew and headlined the WIN anyway).
- Uncommitted code drift: `model.py` +146 lines, `train.py` +117 lines,
  `gate_eval_many.py` edited (hash no longer matches the recorded Stage23
  condition). The entire orchestration toolchain (`adapter_gate.py`,
  `gate_eval_many.py`, `megarexus_live_policy.py`, every collector) is
  UNTRACKED - it has never been committed, so no diff could expose any of
  this.
- Unrelated but found: `runs/outcome_v9/best.pt` and `last.pt` were
  overwritten on 2026-07-02 (tracked files, modified in working tree);
  restorable via `git checkout -- <paths>`. `environment_files/sp80/...
  metadata.json` diff is only a re-download timestamp, benign.
- The stage60 offline `levels_completed=6` / WIN is INVALID as a progress
  claim. The honest boundary is UNCHANGED: live 28.57142857142857 / 3 levels
  WITH graph-demo-plan assistance (Stage20/Stage32); CNN-alone live evidence
  is 1 level (v9retry).

New tooling built this session (both untracked until the user rules on
committing; both tested):

- `synthesis_lab/orchestration/honesty_audit.py` - classifies every run dir
  by assist markers (env flags, route provider fields in gate rows,
  human-mined/replay/route lineage, assist-requiring PASS checks) and checks
  tracked-artifact integrity. Sweep result this session:
  82 AUTONOMOUS_CANDIDATE, 46 IMITATION_ASSISTED, 6 INVALID_HEADLINE.
  Exit 2 = undisclosed invalid artifact exists. Machine report written to
  the session scratchpad; rerun fresh, it is cheap:
      python3 synthesis_lab/orchestration/honesty_audit.py
- `synthesis_lab/orchestration/train_watch.py` - the ONLY authorized
  unattended training driver (contract rule 7). Hash-verified preflight
  (code, init checkpoint, shards, undeclared assist env flags, duplicate
  train.py processes), runs exactly the condition's recorded command,
  enforces --max-hours, writes WATCH_LOG.jsonl + WATCH_REPORT.json.
  Verified: --dry-run against the Stage23 condition correctly REFUSES with
  three code-hash mismatches (train.py, model.py, gate_eval_many.py):
      python3 synthesis_lab/orchestration/train_watch.py \
        --condition synthesis_lab/experiments/megarexus_cnn/runs/stage23_v13_level3_human_mined_20260704/CONDITION.json \
        --dry-run

Docs/memory updated this session: EVIDENCE CONTRACT section below (rules
1-8); user memory files `megarexus-lab-honesty-rules`,
`megarexus-course-correction-goal`, `megarexus-session-handoff-protocol`,
and the MEMORY.md index.

Blocked / user decisions pending (do NOT act on these autonomously):

1. Quarantine `outcome_v16_*`..`outcome_v26_*` (+ stage36-60 route dirs)
   into `runs/_invalid/` so no future session picks one up as a parent.
2. Split the uncommitted `model.py`/`train.py` edits: the legitimate Contra
   goal-head work (v14/v15 lane) is entangled with the route-head code.
   Revert is NOT safe; it needs a split-and-commit the user approves.
3. Restore `outcome_v9/best.pt` + `last.pt` from git.
4. Policy ruling: are human-mined demos banned from TRAINING entirely, or
   only from runtime route-walking and same-level eval? Until ruled, treat
   the Stage23/v13 pre-authorization as SUSPENDED - v13 trains on the
   human-mined level-3 shard, and its preflight fails on code drift anyway.
5. Commit the orchestration toolchain (contract rule 8) once the split in
   item 2 is decided.

Next valid work for a resuming agent: none autonomously. Every lane routes
through a user decision above. If the user says "train on this for N
hours", that means train_watch.py on one pre-registered condition per
contract rule 7 - and today every candidate condition is blocked by items
2/4.

## EVIDENCE CONTRACT (2026-07-04, user ruling after route-replay incident) - BINDING ON EVERY SESSION

Incident: the overnight 2026-07-04 session (stages 36-60) mined human
answers for levels 3/4/5, trained eleven successive route-imitation
checkpoints (v16 through v26), enabled `MEGAREXUS_ROUTE_*` step-indexed
route walking in the runtime, added `route_provider_attached` as a PASS
requirement inside the adapter gate, evaluated on the same levels the mined
routes came from, and reported `levels_completed=6` / `GameState.WIN` as the
headline result. The user has ruled this cheating. Audit sweep
(`honesty_audit.py`, 2026-07-04): 82 runs AUTONOMOUS_CANDIDATE, 46
IMITATION_ASSISTED, 6 INVALID_HEADLINE (stage44, stage45, stage50, stage51,
stage59, stage60 - PASS gates that require route assistance to pass).

Binding rules:

1. Run `python3 synthesis_lab/orchestration/honesty_audit.py` before writing
   any wrap note, final report, or handoff section. Its class summary MUST
   appear at the top of the report. Exit code 2 means an INVALID_HEADLINE
   artifact exists and must be disclosed before anything else.
2. Any run using `MEGAREXUS_GRAPH_DEMO_PLAN` or any `MEGAREXUS_ROUTE_*`
   flag, or any checkpoint/dataset whose lineage includes human-mined,
   replayed, or route data, is IMITATION_ASSISTED. Its `levels_completed`,
   score, or WIN state may be cited only as a route-execution baseline or
   diagnostic - never as model progress, never as a headline number, never
   as "the model reached/solved/won".
3. Gate PASS criteria must never include assistance attachment (the
   stage44-60 `route_provider_attached` check is the canonical violation).
   Success criteria are recorded in CONDITION.json before training; criteria
   invented mid-session prove nothing.
4. Evaluating a model on the same level whose mined route it was trained on
   is memorization checking. It is never generalization, discovery, or
   progress evidence.
5. Every CONDITION.json must declare `evidence_class`
   (`autonomous_candidate` | `imitation_assisted` | `diagnostic`) and
   `assist_env_flags_allowed` before launch. A run whose runtime uses flags
   beyond its declaration is invalid for all claims.
6. Honest evidence boundary as of 2026-07-04: live score 28.57142857142857
   with 3 levels, achieved WITH graph-demo-plan action-side assistance
   (Stage20/Stage32). CNN-alone live evidence is 1 level (v9retry). Report
   it exactly this way. The stage60 offline WIN is invalid as a progress
   claim (see `runs/stage60_.../AUTONOMOUS_SOLVER_INVALIDATION.md`).
7. SCOPE LOCK for unattended time. "Train on this for N hours" authorizes
   exactly one thing: launching the single pre-registered condition through
   `synthesis_lab/orchestration/train_watch.py` (deterministic babysitter:
   hash-verified preflight, recorded command only, wall-clock limit, factual
   WATCH_REPORT.json). The agent session launches it detached, verifies it
   started, and ENDS. While the user is away there is no authorization to
   create new conditions, new datasets, new training scripts, new stages,
   code edits, or gate edits - if training fails or finishes early, that is
   information for the user, not a mandate to fill the remaining hours.
   Empty hours are not a work queue. train_watch preflight refusal (e.g.
   code hash mismatch) means STOP and report; working around it is the
   2026-07-04 incident again.
8. Training/orchestration code must be committed before a condition records
   its hashes. The 2026-07-04 incident was possible partly because
   `gate_eval_many.py`, `adapter_gate.py`, `megarexus_live_policy.py`, and
   all collectors were untracked files no diff could expose. New tooling is
   committed when created; a launch from a dirty or untracked toolchain is
   not a clean condition.

## COURSE ADVISORY (2026-07-03 post-wrap review) - READ THIS FIRST TO RESUME

`synthesis_lab/notes/COURSE_ADVISORY_20260703_STAGE23_RESUME.md` is the
pick-up document for the next agent: whole-program progress assessment,
mandatory preflight checklist, authorized v13 relaunch procedure,
pre-committed gate chain and decision tree, and open items.

## CONTRA TRAINING STATUS (2026-07-04) - CURRENT KEY/LOCK/TARGET BOUNDARY

Contra training is trained, offline-verified, and now live-verified for
post-level-0 key/lock/target designation. The user goal for this lane is
complete under the Stage32 evidence below. Boundary: this is not a level-4
completion claim and not a broader release claim.

Correction after Stage32 frame audit: keep the lane named **Contra training**,
but do not use the old `paint_source` wording for the third goal channel.
Level 3 introduced two paint outputs. The current channel is `paint_output`,
meaning every small block of the paint color is a paint output.

Current evidence:

- Stage26 best checkpoint:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v14_goal_head_v12parent_20260704/best.pt`,
  sha256 `cae6d8aa3d5ffb3a8aa17281090320b2e9eb2ddc8bfc32f3b1cc00484f104a92`;
  label eval mean IoU `0.9989463885625204`, target `0.9999203681945801`,
  key `0.9994872808456421`, source `0.9974315166473389`, unseen mean IoU
  `0.9781931638717651`.
- Stage28b recorded-frame post-level-0 probe passed `2/2`; mean IoU
  `0.9772199730094466`, target `0.97165991902834`, key `0.96`, source
  `1.0`.
- Stage31 provider-enabled same-path offline gate passed: model `10/10`,
  paired random `0/10`, provider updates `115` in every model episode,
  latest target count `3` in every model episode, and every model episode
  reached `levels_completed=3`.
- Stage32 live scorecard completed from the recorded condition:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage32_live_scout_v14_goal_provider_20260704/CONDITION.json`,
  sha256
  `53df6a85985e98eef50fc59641c20d7fb76f4ce10511529469e3baf1609e0fec`,
  with run dir
  `codexlab/runs/arc_live_sp80_contra_training_v14_goal_provider_20260704T060248Z`.
  Review page verified `200 OK`:
  `http://127.0.0.1:8773/index.html`.
  Closed SDK scorecard `666f929b-d875-49a9-9034-f83030411c7d`, score
  `28.57142857142857`, levels_completed `3`, actions `497`, resets `3`,
  policy_error_count `0`. Contra training provider enabled, pour budget
  enabled, provider updates `492` across levels `1,2,3`, latest target count
  `3`, frame fallback `0`, captured calls `497`; all goal-budget ledger rows
  use `goal_target_count=3` and `pour_cap=3`.

Paint-output correction:

- Stage33 rebuilt the goal-label dataset with channels
  `target_lock`, `key_block`, `paint_output`; dataset sha256
  `20937e2e57eab0dbf1d39917952b5d9a74c4bcff4398b9731b150a15c6da0c79`;
  paint-output count distribution `{1: 877, 2: 6, 3: 34}`.
- Stage34 trained the v15 paint-output goal head. Best checkpoint:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v15_paint_output_goal_head_v12parent_20260704/best.pt`,
  sha256
  `756b37cadba53e2798da7bfc631da16652d579c48abe185f69f82c0b0140f064`.
  Label-val paint-output recall is `1.0`, precision/IoU
  `0.6003924012184143`.
- Stage35 showed the raw mask only seeded one of the two Stage32 level-3
  raw-4 paint blocks, so `megarexus_live_policy.py` now expands a detected
  paint-output seed to all small same-color paint blocks. Provider probe:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage35_paint_output_multi_spout_probe_20260704/provider_expansion_probe.json`,
  sha256
  `4ff12914120d432c5752b0c8612704c6d2283394b809c195f3d598d0ce955e2b`,
  reports both raw-4 outputs on Stage32 calls `26` through `31`.

Boundary: the paint-output correction is local training/provider evidence,
not a new live scorecard. Next live condition should use v15 plus provider
same-color expansion.

Stage36 human solution hint: user-supplied annotated image
`synthesis_lab/experiments/megarexus_cnn/runs/stage36_human_solution_verticals_review_20260704/QDTOJK0.png`,
sha256
`c00282a527013616c6aa7b45a651ad4ef8b4094cd6d8b91e750938b505ed1a74`.
Treat it as a hypothesis, not a controlled action trace: level 3 needs
vertical control; the mechanism is triple catch plus double paint. The next
Contra training target is vertical catch-routing/control.

Erratum from the same review: the wrapped Stage23 `CONDITION.json` had an
invalid 65-char sha256 recorded for `perturb_recovery_mining_v11_train.pt`
(extra `1`). Fixed in place; the shard's verified hash is
`8e8d850d098e4a33abdfa028109b8ee890d6116c6933b9e340adb21ad18949d5`.
Current authoritative condition sha256 (supersedes `75e06152...`):
`b09ca513f43ce59301c1c236aca81b70e443ccd346bfa86a2df4dddf27790950`.
All other wrap-note hashes/claims verified against disk and matched. GPU
recheck at review time: the Leanstral export had finished; no compute
process, 0% util.

## WRAP NOTE (2026-07-03 21:48 CDT) - STAGE23 CONDITION READY, V13 NOT RUN

Detailed wrap note:
`synthesis_lab/notes/WRAP_20260703_STAGE23_V13_READY_NOT_RUN.md`.

Stage23/v13 has a recorded condition but no completed training artifact.
Condition:
`synthesis_lab/experiments/megarexus_cnn/runs/stage23_v13_level3_human_mined_20260704/CONDITION.json`.
Pre-interruption condition sha256 (superseded by the erratum above):
`75e06152ce3849d615e6fe6e435da773bac28c76dc8d1bd094449a09518abb21`.

The exact v13 command was launched locally at about
`2026-07-03T21:46:41-05:00`, then the user interrupted and asked for
handoff/wrap. The launch kept running in the background, so it was explicitly
stopped: Python PID `2548534`, shell PID `2548529`, and `tee` PID `2548535`.

Post-stop check at `2026-07-03T21:47:58-05:00`: no v13 training process
remained; `nvidia-smi` showed no compute process; the log contained only
`[train] device=cuda` and was `20` bytes; no
`outcome_v13_level3_human_mined_20260704/` directory, checkpoint, history,
gate report, or review page exists. Do not claim any v13 result from this.

Resume by rechecking runtime and relaunching the command recorded in the
Stage23 condition. Let it produce checkpoints before running gates.

## PAUSE NOTE (2026-07-03 21:42 CDT) - STAGE LABELS ARE BOOKKEEPING, NOT NEAR-DONE STATUS

Detailed pause note:
`synthesis_lab/notes/REVIEW_PAUSE_20260703_STAGE22.md`.

The user asked whether "stage 22 of 23" means this lane is almost done. It
does not. The stage numbers are local bookkeeping labels, not a finite
countdown and not a completion percentage.

Current state at pause:

- Stage20 live ARC SP80 scout reached `total_levels_completed=3` with closed
  SDK score `28.57142857142857`, then stalled in level-3 control. This is the
  current live evidence boundary, not a solve/release claim.
- Stage22 exclusive was a controlled negative because it omitted the first
  level-4 action from mined human sequences.
- Stage22b inclusive is the useful new artifact: controlled offline replay of
  human-mined level-3 sequences, `6/12` candidates reached level 4, `8`
  trajectories kept. Report:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage22b_level3_human_mined_inclusive_v12_graphadaptive_20260704/level3_human_mined_report.json`.
  Review URL verified `200`:
  `http://127.0.0.1:8765/synthesis_lab/experiments/megarexus_cnn/runs/stage22b_level3_human_mined_inclusive_v12_graphadaptive_20260704/index.html`.
- Stage22b output shard:
  `synthesis_lab/experiments/megarexus_cnn/data/level3_human_mined_inclusive_stage22b_20260704.pt`,
  sha256 `2e98eeaa68aa8a71da62ad5a23a1df7704f4781d974991bc809746eeed91e6e3`.
- Stage23/v13 is only the next proposed training condition label. It has not
  been launched, gated, or live-tested.
- The requested arXiv fit check for `https://arxiv.org/abs/2607.01224` was
  paused before any subagent paper verdict existed. Do not cite a fit verdict
  until a fresh review reads the arXiv PDF and any official code.
- Runtime checked at `2026-07-03T21:42:12-05:00`: no active Megarexus
  train/gate/collection/live-scout process was found locally; evidence server
  is running on `127.0.0.1:8765`; unrelated Leanstral export PID `2517434` is
  using CUDA on the GB10, so recheck local/Apollo availability before launching
  any v13 work.

## 0-CURRENT RESULT (2026-07-03 21:20 CDT) - V12 GRAPH-ADAPTIVE HIT LIVE 3 LEVELS; NEXT BASIN IS LEVEL-3 CONTROL

This section supersedes the older "next basin is level-2 four-bar control"
section below. Do not launch another live scorecard just to repeat this
result. The next valid work is level-3 representation/search or a training
data condition that targets the level-3 layouts exposed by the live run.

Stage 17 level-2 representation scout:

- Modified `synthesis_lab/orchestration/graph_plan_probe_level2.py`,
  sha256 `26704bed478977f6bd5f92e04661a079d910a7534dd25a7f9fdad13b0961f698`,
  preserving the old horizontal mode while adding a raw action-ID template
  candidate family with optional single vertical nudges and emitted
  `CONDITION.json`.
- Condition:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage17_level2_action_template_probe_v12_graphadaptive_20260704/CONDITION.json`,
  sha256 `d6f1ca052015f9fe6dd217a8969286c1f484fcd1bb91b8005eabf837143e00fe`.
- Report:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage17_level2_action_template_probe_v12_graphadaptive_20260704/graph_plan_probe_level2.json`,
  sha256 `4bfa869576771d9340c82085e3b9a6c2109fc7dbf04ec3c788b249d04cfb2140`.
  Review:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage17_level2_action_template_probe_v12_graphadaptive_20260704/index.html`,
  sha256 `60680326d3acf20893122bf9f8f3a2cfb1ba80213aca50816c18e85f4810d7d0`.
- Result: prefix reached level 2, then `36/1458` raw action-ID template
  candidates reached `levels_completed >= 3`. Base winning candidate:
  `codexlab_solution_level2_action_ids_counts_1_2_2_1`, action items
  y-rank `0:[4]`, `1:[3,3]`, `2:[3,3]`, `3:[4]`. The inverse raw-action
  family produced no winners in the extracted winner list.

Policy and gate updates:

- Modified `synthesis_lab/orchestration/megarexus_live_policy.py`,
  sha256 `681266eff43010a2ea633ac07ea100f50324b5e25e5c6b040501b4970792cb9b`,
  to admit four midband movables behind `MEGAREXUS_GRAPH_DEMO_PLAN=1` and
  execute the Stage17-backed raw action-ID plan. This is action-side
  representation wiring, not proof that the CNN alone learned level 2.
- Modified `synthesis_lab/orchestration/adapter_gate.py`, sha256
  `e4789d878f51c066e46410e702e408b2db82e45260584f747f7cf66437a3e515`,
  to emit `CONDITION.json` and support `--target-levels`.
- Stage 19 same-path adapter gate target `levels_completed >= 3` passed:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage19_adapter_gate_l3_full_v12_graphadaptive_20260704/adapter_gate.json`,
  sha256 `823a58ccfcf41956aa7ff0d336d3998e0a2db9bde19d2e34e3f0bf5a52efedeb`.
  Condition sha256
  `f77b7d0b8d09809ba1f9e43419eb5270d6e1e70d5166081e560c4bfe7ffed60d`.
  Review sha256
  `610012828668bf2a5dd7391c0dde9b6c15ba444ab438970c91bfd512cada6ad4`.
  Verdict: `PASS`, model `10/10` reached `levels_completed=3`,
  random `levels>=2` was `0/10`, zero policy errors.

Live scout:

- Condition:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage20_live_scout_l3_v12_graphadaptive_20260704/CONDITION.json`,
  sha256 `95342da97ef99c6baf268b60b0b5a3965749f62faada0a2a8f2beea20f93ace7`.
- Run directory:
  `codexlab/runs/arc_live_sp80_megarexus_v12_graphadaptive_l3_20260704T021637Z`.
- Live scorecard id: `453a74e8-9113-43a8-a37a-fcdfacac55ce`.
- Closed ARC SDK payload in `result.json`: score
  `28.57142857142857`, `total_levels_completed=3`, policy errors `0`.
  Per-game row: `sp80-589a99af`, `levels_completed=3`, `actions=496`,
  `resets=4`, final state `GameState.NOT_FINISHED`. Scorecard close payload
  recorded `total_actions=500`, `total_levels_completed=3`.
- Artifact hashes:
  `result.json` sha256
  `f159aeaf130c757ecae2fdded45456c906090ae16c99c46426372859472a7df3`;
  `index.html` sha256
  `eff4c75dd94155a25b76186fd165e149b0b4a707dad39e41668e2ea739959711`;
  `report_trace.jsonl` sha256
  `22b1e44eea64cc4d113374121e8a88cdad58c4cdcff3c2866af0ae8a56511722`;
  `report_manifest.json` sha256
  `25c656b3b54183e4ce524c123e44c7807323d6906480af612970bb7be37898fb`;
  `policy_frames_meta.json` sha256
  `03d08a7c28e4dbf4df3430dcb814b00e0c82cdc99dd351d9cbf9da7a46abd7d3`;
  `policy_frames.npz` sha256
  `abe380d0a82ede67216e5eb02036cdaa26bf71040aa19fa5eaffa0e0852291a0`.
- Local review URL verified `200`:
  `http://127.0.0.1:8765/codexlab/runs/arc_live_sp80_megarexus_v12_graphadaptive_l3_20260704T021637Z/index.html`.
  Public `https://arcprize.org/api/v3/scorecards/453a74e8-9113-43a8-a37a-fcdfacac55ce`
  returned `404`; treat the closed SDK payload in `result.json` as
  controlling evidence.
- Frame capture exists with `496` records and `frames_fallback_count=0`.
  Trace source histogram: catch select/move/pour `1/3/1`, arrange
  select/move/pour `34/61/8`, plan-budget no-ops `179`,
  pour-budget no-ops `209`. `plans_built`: level 1 = 1, level 2 = 1,
  level 3 = 8. `pours_at_level`: level 1 = 1, level 2 = 1,
  level 3 = 6.

Interpretation and next valid work:

- This is a real live improvement over the prior v12 graph-adaptive scout
  (`2` levels, score `14.285714285714285`) and over v9retry (`1` level,
  score `4.761904761904762`).
- It is not a release/completion claim. The adapter now clears level 2,
  reaches level 3, then spends the remaining budget in repeated level-3
  attempts and fail-closed no-ops after plan/pour budgets are exhausted.
- Do not keep rerunning the Stage17 raw action-ID template on level 3 as if
  it is a new experiment. The next condition must expose a new level-3
  representation/search space or mine/collect a level-3 transition shard.

## 0-CURRENT RESULT (2026-07-03 20:59 CDT) - V12 GRAPH-ADAPTIVE HIT LIVE TARGET; NEXT BASIN IS LEVEL-2 FOUR-BAR CONTROL

This section supersedes the older "live scout blocked" sections below.
Do not launch another live scorecard just to repeat this result. The next
valid work is level-2 representation/search, or a training/data condition
that targets the level-2 four-bar state exposed by the live run.

Gate stack before live:

- v12 fixed-v8 no-loss gate stayed green:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v12_headlevel_perturb_recovery_20260703_gates/gate_eval_many_full_on_v8val.json`,
  sha256 `f5a1a2b8865d68360d6d9d1fe8aa68d56abf846e0a520b952c803993d373a5a9`.
- Stage 12 full perturb gate with `MEGAREXUS_GRAPH_DEMO_PLAN=1` passed
  at threshold: `9/10` model rows passed, paired random `0/10`.
  Seed `20290106` remained a real miss (`levels_completed=1`), so this
  is not a perfect local generalization result.
  Report:
  `synthesis_lab/experiments/megarexus_cnn/runs/perturb_gate_v12_headlevel_best_graphadaptive_full_20260703/perturb_gate.json`,
  sha256 `9c2997431431fc554f92b0db80aedcd39af0beed2db8f78cbfbb403d63c29df2`.
  Review:
  `synthesis_lab/experiments/megarexus_cnn/runs/perturb_gate_v12_headlevel_best_graphadaptive_full_20260703/index.html`,
  sha256 `12673f2771d7792c5d37f9f3595399d3a39a49c81d3eab0f116357e049411fec`.
- Stage 13 same-path adapter gate with `MEGAREXUS_GRAPH_DEMO_PLAN=1`
  passed `10/10`; paired random reached `levels>=2` in `0/10`.
  Report:
  `synthesis_lab/experiments/megarexus_cnn/runs/adapter_gate_v12_headlevel_best_graphadaptive_20260703/adapter_gate.json`,
  sha256 `ae3036b021e2ca535086f5ac4a03933eef79d7ff3416630f7d77fd8011c6a828`.
  Review:
  `synthesis_lab/experiments/megarexus_cnn/runs/adapter_gate_v12_headlevel_best_graphadaptive_20260703/index.html`,
  sha256 `8f54763aba5900d9a064cbf0141b4cf4b796c244238b0b78b19ac6ae86a4b32e`.

Live scout:

- Condition:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage14_live_scout_v12_graphadaptive_20260703/CONDITION.json`.
- Run directory:
  `codexlab/runs/arc_live_sp80_megarexus_v12_graphadaptive_20260703T2052`.
- Live scorecard id: `9ff01749-a750-438c-a3dd-b84091df6fb5`.
- Closed ARC SDK payload in `result.json`: score
  `14.285714285714285`, `total_levels_completed=2`, policy errors `0`.
  Per-game row: `sp80-589a99af`, `levels_completed=2`, `actions=496`,
  `resets=4`, final state `GameState.NOT_FINISHED`.
- `result.json` sha256
  `9b44dbb6626ff4b2b0b8880b35aa6efbc80f87c824161e64faf43611b7a3e305`.
  `index.html` sha256
  `c40c0cdc5ece18c53601730b2084c29e5c0665f68b8b96c287370d6c25c8c25e`.
  `report_trace.jsonl` sha256
  `8fa07ea18b42055488659bd8a32bfea5077c846d5a8a3eb224bccd6a30255a4a`.
  `report_manifest.json` sha256
  `fa20ab505b24a93d9acf7688c3c281fb0eaaf9ee6849a8376f318aaf99bf1d66`.
  `CONDITION.json` copied into the run dir, sha256
  `8e370627913b576642a6ef86e25c8eccb214bde640e41377914dae10a49bd13d`.
- Frame capture exists:
  `codexlab/runs/arc_live_sp80_megarexus_v12_graphadaptive_20260703T2052/policy_frames/policy_frames.npz`
  and `policy_frames_meta.json`; `496` frame records, no frame fallback.
- Public HTTP probe for
  `https://arcprize.org/api/v3/scorecards/9ff01749-a750-438c-a3dd-b84091df6fb5`
  returned `404`, matching the earlier lane warning that public scorecard
  endpoints can be unreliable for these live runs. Treat the closed SDK
  payload in `result.json` as controlling evidence.

Live behavior interpretation:

- The run met the staged target (`levels_completed >= 2`, zero policy
  errors) and is a real improvement over the prior v9retry live scout
  (`1/6` levels, score `4.761904761904762`).
- It is not a release/completion claim. After the level-1 graph pour,
  the adapter reached level 2 at call 15, then spent the remaining budget
  in fail-closed `megarexus_arrange_no_roster` no-ops because the level-2
  state has four large midband bars, while the current graph branch only
  admits exactly three.
- Source histogram from the live policy trace: catch select/move/pour
  `1/3/1`, arrange select/move/pour `3/6/1`, then
  `481` `megarexus_arrange_no_roster` actions. This is intentionally not
  a pour loop, but it is the next control failure basin.

Post-live diagnostics:

- Stage 15 static live-frame graph review:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage15_live_level2_graph_review_v12_graphadaptive_20260703/index.html`,
  sha256 `ddb2eef80f3809725b72e5574055a8eff793a6b4c5536b34064af9a98a13d45e`.
  Report sha256
  `59a0827e1b595e683752141e9335a1511602f717ab2b4e1e68ae619a82277271`.
  First level-2 frame has four large bars:
  centers `[15,21]`, `[49,29]`, `[19,33]`, `[47,41]`.
- Added new scout runner:
  `synthesis_lab/orchestration/graph_plan_probe_level2.py`,
  sha256 `16bbee14da56c04cdfbfad24d4765f5cf6e14d7ca1100140f0c06cdbe4aefb37`.
- Stage 16 level-2 anchor-horizontal four-bar scout:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage16_level2_graph_plan_probe_v12_graphadaptive_20260703/graph_plan_probe_level2.json`,
  sha256 `ddc951f9dd19f49cc6e1fde3a54c58aa34c93f181ca8749163eaeb28554d41a3`.
  Review sha256
  `bc62d9fac331a4157b2d375c7c4418a53ad24fe08e121dff8b311083c0a1a729`.
  Result: `0/581` candidates reached level 3; every candidate stayed at
  `levels_completed=2`.

Next valid work:

- Do not repeat the same horizontal anchor-derived four-bar scout as if
  it is progress. Stage 16 is a controlled negative in that representation.
- The next runnable condition needs a real new degree of freedom:
  vertical moves on the four-bar level-2 roster, a richer ordering/anchor
  representation, or a level-2 data/training collection shard that labels
  successful level-3 transitions.
- If launching training, keep v12 fixed-val no-loss as the comparator and
  treat the graph-adaptive live result as action-side evidence, not proof
  that the CNN alone solved the level-2 regime.

## 0-CURRENT RESULT (2026-07-03 20:03 CDT) - V12 + ARRANGE MEMORY FAILED SAME PERTURB BASIN; LIVE SCOUT BLOCKED

Do not run a live scout. Do not run Stage 1a adapter for v12 unless
there is a new reason to spend the time: v12 already failed the
discriminating Stage 1b perturb gate, and the AutoMem-inspired
arrange-memory scaffold did not change the first failed seed.

v12 condition:

- `synthesis_lab/experiments/megarexus_cnn/runs/stage2_v12_headlevel_perturb_recovery_20260703/CONDITION.json`.
- Purpose: narrow calibration from v10dynonly memsafe `best.pt`, training
  only `head_level` after v11 proved that broad perturb-recovery training
  drifts fixed-val changed-cell metrics.
- Code fix required before v12 could run: `train.py` now skips
  backward/optimizer only for batches whose losses do not require grad,
  needed when a narrow trainable-prefix run sees batches with no
  trainable-head labels. Patched `train.py` sha256
  `c866b722bba477afbf9961e8fe89037dfa1838b05452c198ac4040199dabf430`.
- v12 trained on Apollo, PID `243592`, completed cleanly in `1164s`.
  `head_level` trainable params: `8321/3644714`.
- Expanded mining-val did NOT improve: every epoch stayed at level
  `tp=135/141`, `fp=5`, catch `0.950`, settle IoU `0.658`. Treat this
  as a warning that v12 may still fail heldout perturb recovery.

v12 fixed-v8 gate:

- Gate JSON:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v12_headlevel_perturb_recovery_20260703_gates/gate_eval_many_full_on_v8val.json`,
  sha256 `f5a1a2b8865d68360d6d9d1fe8aa68d56abf846e0a520b952c803993d373a5a9`.
- Review page:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v12_headlevel_perturb_recovery_20260703_gates/index.html`,
  sha256 `dcaeb9741798b1e7196f28b8381974525f300c4fb60a7a97c18ca2a129628897`.
- Verdict: `10/10` candidate rows pass strict no-loss. Baseline v8:
  `bc=1.0`, `catch=0.956303`, `level_tp=109/109`, `level_fp=0`,
  `settle_iou=0.655881`, `changed=0.995984`. All v12 rows:
  `bc=1.0`, `catch=0.956303`, `level_tp=109/109`, `level_fp=0`,
  `settle_iou=0.655881`, `changed=0.996143`.
- Recommended next action: run behavioral gates, preferably starting with
  the heldout perturb gate because it is the discriminating failure basin.
  A failed Stage 1b blocks live scout immediately.

v12 Stage 1b perturb gate:

- Condition:
  `synthesis_lab/experiments/megarexus_cnn/runs/perturb_gate_v12_headlevel_best_20260703/CONDITION.json`,
  sha256 `eaba726839fa0f8661d7c402c76647e11efe32714b046be5d49eb491902fa70f`.
- Partial report:
  `synthesis_lab/experiments/megarexus_cnn/runs/perturb_gate_v12_headlevel_best_20260703/perturb_gate_partial_stopped.json`,
  sha256 `fa41e665f714ebd975fb119759f2a70d931637966402e3ffed2567c8edca6bc4`.
- Review page:
  `synthesis_lab/experiments/megarexus_cnn/runs/perturb_gate_v12_headlevel_best_20260703/index.html`,
  sha256 `cf0f970bd53ae7d123a933b0b2cac3a4a3aea0c983fefeb6077264e6914a73f1`.
- Log:
  `synthesis_lab/experiments/megarexus_cnn/runs/perturb_gate_v12_headlevel_best_20260703.log`,
  sha256 `005b783585c6ea4e75a463547cad7e9528c3c423ef394bcc2cd007219314f337`.
- Completed rows before stop: seeds `20290101` and `20290102` passed;
  seeds `20290103`, `20290104`, and `20290105` failed. The failed rows
  all reached plumbing, replay matched, had zero policy/runtime hygiene
  failures, but remained at `levels_completed=1`. With three failures,
  the 9/10 pass threshold was unreachable, so PID `270818` was stopped
  at 19:33 CDT.
- This repeats the v10 heldout failure basin. Fixed-val preservation is
  not enough; the perturb-recovery/action-side behavior still fails.

Stage 5 targeted arrange-memory A/B:

- Condition:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage5_v12_arrange_memory_failedseeds_20260703/CONDITION.json`,
  sha256 `f865e9678562c131396edabb1271f21767d0f37addea9b21df3531753cda83c0`.
- Env change: `MEGAREXUS_ARRANGE_MEMORY=1`; checkpoint unchanged
  (`v12 best.pt`).
- Targeted run started at failed seed `20290103`, intended for three
  failed default seeds. It was stopped after the first row because the
  memory-enabled adapter repeated the exact default failure:
  `reached=True`, `replay_match=True`, hygiene checks all true,
  `levels_completed=1`, `actions=43`.
- Partial report:
  `synthesis_lab/experiments/megarexus_cnn/runs/perturb_gate_v12_headlevel_best_arrange_memory_failedseeds_20260703/perturb_gate_partial_stopped.json`,
  sha256 `fd97e7a91167c677de887ed23d93f2d11e011e190b7f55c279bed55e0f374a06`.
- Review page:
  `synthesis_lab/experiments/megarexus_cnn/runs/perturb_gate_v12_headlevel_best_arrange_memory_failedseeds_20260703/index.html`,
  sha256 `226025f7724e115f1ce7f9251a35de8685bb0e83a053d7d5832b4736196184a2`.
- Log:
  `synthesis_lab/experiments/megarexus_cnn/runs/perturb_gate_v12_headlevel_best_arrange_memory_failedseeds_20260703.log`,
  sha256 `0948069ca0c35e2304efd1f8b7d51cfc3f89e45b945804f1e09008d1a8c4d5a6`.

Interpretation: the paper-inspired memory ledger is a plausible
adapter-side idea in general, but this concrete scaffold did not affect
the v12 heldout perturb failure. Stop same-basin CNN/head-only/attempt
memory churn here; the next useful change needs a different
action-search or representation mechanism, not another small shard over
the same labels.

Apollo status at the last check: idle, no `train.py`, RTX 4080 at
display-only usage, about 12 GiB RAM available.

## 0-RECENT RESULT (2026-07-03 16:58 CDT) - V11 FAILED FIXED-VAL; LIVE SCOUT STILL BLOCKED

Do not promote v11. Do not run Stage 1a/1b for v11, and do not run a live
scout from v11.

The v11 perturb-recovery run completed cleanly on Apollo, was copied back,
and was gated against the fixed v8 validation surface. The strict no-loss
gate failed: `0/14` v11 candidate rows passed. Baseline v8 row:
`bc=1.0`, `catch=0.956303`, `level_tp=109/109`, `level_fp=0`,
`settle_iou=0.655881`, `changed=0.995984`. Every v11 row lost
`fwd/changed_cell_acc`; some early rows also introduced level false
positives or lost settle IoU. `best.pt` has `level_fp=2`,
`settle_iou=0.654983`, `changed=0.995978`.

Primary artifacts:

- Run dir:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v11_perturb_recovery_mining_20260703`.
- Training log:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v11_perturb_recovery_mining_20260703_train.log`,
  sha256 `61d4e70252d6fedd6f76a0830564d99ff7334719a3903a60dab2b795a16e1564`.
- Fixed-val gate JSON:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v11_perturb_recovery_mining_20260703_gates/gate_eval_many_full_on_v8val.json`,
  sha256 `169719757f0e673b7134f6fd4046c159529f7279b2e25db248b3a93c880618b5`.
- Fixed-val gate review:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v11_perturb_recovery_mining_20260703_gates/index.html`,
  sha256 `2b6778ae088f12c8a19dfe217ff269bf886dbfd2b640c0046789bab50555c6ee`.
- New review helper:
  `synthesis_lab/orchestration/build_gate_many_review.py`,
  sha256 `aa631a921c0f72eac169dd9dfa1101415a7eaccdcfb0fbae33996c9465e61166`.

Apollo is currently idle after the run/gate: no `train.py` process, RTX
4080 at display-only usage, about 12 GiB RAM available. The current
usable checkpoint remains v10dynonly memsafe `best.pt` for analysis
purposes, but it already failed heldout perturb Stage 1b and therefore
is not live-scout eligible.

Recommended next step: do not continue the same v11 CNN basin. Either
analyze why perturb-recovery positives trade away fixed-val changed-cell
and level cleanliness, or use the staged AutoMem-inspired arrange-memory
A/B only if the next observed failure is arrange/repetition-specific.

## 0-PRIOR APOLLO PICKUP (2026-07-03 16:09 CDT) - SUPERSEDED BY THE V11 FIXED-VAL FAILURE ABOVE

Do not treat this as a release state. The current v10 candidate failed
the heldout perturb gate, so the staged live scout remains blocked even
if the adapter gate later passes.

Stage 1a adapter gate on Apollo:

- Active remote Python PID `87565`, launcher shell `87563`, nice level
  10, elapsed about 44 minutes at the 14:55 poll and about 73 minutes
  at the 15:24 poll.
- As of 16:02 CDT, log has five completed model episodes:
  seed `20280001`, `passed=True`, `levels=2`, `actions=119`,
  `resets=1`, all checks true; paired random reached `levels=0`.
  Seed `20280002` also passed with `levels=2`, `actions=119`,
  `resets=1`, all checks true; paired random reached `levels=0`.
  Seed `20280003` also passed with `levels=2`, `actions=119`,
  `resets=1`, all checks true; paired random reached `levels=1`.
  Seeds `20280004` and `20280005` also passed with `levels=2`,
  `actions=119`, `resets=1`, all checks true; paired random reached
  `levels=0` for each.
- No `adapter_gate.json` exists yet because the 10-episode run is still
  active.
- Remote output dir:
  `apollo:/home/frosty/megarexus_train_20260703/synthesis_lab/experiments/megarexus_cnn/runs/adapter_gate_v10dynonly_memsafe_best_apollo_20260703`.

Stage 1b perturb gate on local host:

- Active local Python PID `1632633`, nice level 10, elapsed about 21
  minutes at the 14:41 poll, CPU-active; elapsed about 26 minutes at the
  14:46 poll, about 35 minutes at the 14:55 poll, and about 50 minutes
  at the 15:10 poll; about 64 minutes at the 15:24 poll.
- First row completed at wall `1437.4s`: seed `20290101`,
  `passed=True`, `reached=True`, `levels=2`, `actions=72`,
  `replay_match=True`; all model checks true including
  `paired_replay_match`. Paired random from the same replay-verified
  perturbed state reached `levels=1` in 12 actions.
- Second row completed at wall `1412.2s`: seed `20290102`,
  `passed=True`, `reached=True`, `levels=2`, `actions=110`,
  `replay_match=True`; all model checks true including
  `paired_replay_match`. Paired random reached `levels=1` in 17
  actions.
- Seed `20290103` failed at wall `1361.9s`: `reached=True`,
  `levels=1`, `actions=43`, `replay_match=True`; checks all true
  except `levels_ge_2`.
- Seed `20290104` failed the same way at wall `1356.0s`:
  `reached=True`, `levels=1`, `actions=43`, `replay_match=True`; checks
  all true except `levels_ge_2`.
- This is enough to block the live scout and trigger perturb recovery
  mining. The 10-episode perturb process is still active as of 16:02
  CDT, but it is no longer promotion evidence.
- Output dir currently contains the recorded condition plus first-seed
  frame capture:
  `synthesis_lab/experiments/megarexus_cnn/runs/perturb_gate_v10dynonly_memsafe_best_parallel_20260703/CONDITION.json`.

New helper added for gate harvesting:

- `synthesis_lab/orchestration/build_gate_review.py`, sha256
  `eda60da449f148a8e88801eac692781f427bff30037c7dd3b97c973d680c1364`.
- It compiles and rendered a test page from an old completed adapter
  gate into `/tmp/megarexus_gate_review_test.html`.
- Use it on finished `adapter_gate.json` or `perturb_gate.json` to
  create the run-local `index.html` review surface before reporting raw
  JSON.

Data audit finding for next training:

- `course_correction_v10_train.manifest.json`: reached plumbing `12/12`,
  but `level2_episodes=0`.
- `course_correction_v10_val.manifest.json`: reached plumbing `4/4`,
  but `level2_episodes=0`.
- Therefore do not run another same-basin train over only the v10
  course-correction shard as if it contains successful perturbed
  recovery examples.

Conditional next data condition staged, not launched:

- Condition:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage2_perturb_recovery_mining_20260703/CONDITION.json`,
  sha256
  `604c00c0307a20cb47b45bd3ac1231882fd4be0f685d3890ffadf0bb421a16cb`.
- Collector:
  `synthesis_lab/orchestration/collect_perturb_recovery_mining.py`,
  sha256
  `81a646d0a51ea1e88e95ad448b106e5efab0ee1234ede4f670d7ab11088ec533`.
- Triggered by Stage 1b failures on seeds `20290103` and `20290104`.
- Local mining completed:
  train shard `perturb_recovery_mining_v11_train.pt`, sha256
  `8e8d850d098e4a33abdfa028109b8ee890d6116c6933b9e340adb21ad18949d5`,
  manifest sha256
  `aec3dc56ed91f3646b00bc0ca4a107be11a089002a34025fb8f095bc0e66af9e`,
  `kept_successes=16`, `kept_failures=12`.
  Val shard `perturb_recovery_mining_v11_val.pt`, sha256
  `6841a132d7deb6d07be07d8570c223ae3611d06de5e6844f260cd8fe8d870843`,
  manifest sha256
  `a544a7f765d15a24a361b3f210c7d83e446133d63840c6ab27cb4f09197a20d7`,
  `kept_successes=11`, `kept_failures=6`.
- The positive-success gate for follow-on v11 training is satisfied.

Stage 2 v11 perturb-recovery training running on Apollo:

- Condition:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage2_v11_perturb_recovery_mining_20260703/CONDITION.json`.
- It initializes from
  `outcome_v10dynonly_memsafe_course_correction_20260703/best.pt`
  (`e82815647038a3749442eb356b4daa6013885c81d16da8ae811fb685b28f27ac`)
  and adds the mined train/val shards above.
- Training prefixes: `dynamics`, `click_plane_proj`, `up1`, `up2`,
  `skip_gate`, `head_next_frame`, `settle_conv`, `head_settle`,
  `head_level`; lr `5e-5`, epochs `12`, batch `64`, workers `0`.
- Remote Python PID `184675`, nice level 10. Log:
  `apollo:/home/frosty/megarexus_train_20260703/synthesis_lab/experiments/megarexus_cnn/runs/outcome_v11_perturb_recovery_mining_20260703_train.log`.
- The stale v10 Apollo adapter gate was stopped at 5/10 because v10 was
  already disqualified by Stage 1b perturb failures. The stale local
  perturb gate was also stopped after two repeated heldout failures.
- As of 16:19 CDT epoch 1 completed. Dataset counts:
  train `102001` transitions (`1177` human, `100824` rollout),
  `67423` unique frames; val `20577` transitions (`245` human,
  `20332` rollout), `13847` unique frames. Trainable parameters:
  `1964563/3644714`.
- Epoch 1 expanded-val metrics: changed `0.996`, catch `0.950`,
  level `tp=137/141`, `fp=6`, settle IoU `0.657`. This is expanded
  mining val, not the fixed-v8 release gate.
- Epoch 1 artifacts on Apollo:
  `epoch_001.pt` sha256
  `c2f9c1455464d62557bf71ac70416a44b85bcd3fc8694e3405f7b3e99a637b45`,
  `best.pt` sha256
  `8cc51d9415ca35fa257b80ea8d091b96d812bbc7c5e473a50701a4e5707defa4`,
  `last.pt` sha256
  `f74af9aa70cd0c2f9b6c0495cb313b7e509a6f7694f8bb88d271352d1d345d83`.
- Do not promote this run until it finishes and passes fixed-v8-val
  no-loss plus new Stage 1a/1b gates.

Stage 4 live scout condition staged, not launched:

- Condition:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage4_live_scout_v10dynonly_memsafe_best_20260703/CONDITION.json`,
  sha256
  `8f2b3d72edb570b0c5aa5d8791fbdc7568188fce15709a1557b2e0e73cd4ce00`.
- Status: `blocked_until_gate_stack_passes`.
- Planned live command uses
  `MEGAREXUS_CKPT=synthesis_lab/experiments/megarexus_cnn/runs/outcome_v10dynonly_memsafe_course_correction_20260703/best.pt`
  and
  `MEGAREXUS_TRACE_DIR=codexlab/runs/arc_live_sp80_megarexus_v10dynonly_best_20260703/policy_frames`.
- Do not run it until both `adapter_gate.json` and `perturb_gate.json`
  exist with `gate_verdict=PASS`. The planned success criterion is live
  `sp80-589a99af` with `levels_completed >= 2`, zero policy errors, no
  capped-pour loop, and policy frame capture present.

Conditional Stage 5 arrange-memory scaffold A/B staged, not launched:

- Source idea: AutoMem paper triage on 2026-07-03. Useful fit is an
  adapter-level representation/search-space change, not another CNN
  training shard.
- Code change is opt-in only: `MEGAREXUS_ARRANGE_MEMORY=1` enables
  explicit layout-signature and attempt-signature memory for arrange
  pours. Default is off, so the active Stage 1a/1b gates are not
  changed.
- Updated adapter source:
  `synthesis_lab/orchestration/megarexus_live_policy.py`, sha256
  `4c210f2772163afcf21e974fb9eed3ca153b310de1422ee18319aa2f7469edd8`.
  It compiles together with `adapter_gate.py` and `perturb_gate.py`, and
  a direct import check verified the feature flag is false by default
  and true when set.
- Condition:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage5_arrange_memory_scaffold_ab_20260703/CONDITION.json`,
  sha256
  `d2be91ea68a070183a0d45ee1273b56f22c7b408be3ab55df322248d1efb44d1`.
- Status: `staged_not_launched`. Do not launch while the current Stage
  1a/1b gates are still running. Use it if current gates fail on
  arrange/repetition behavior, or if the gated live scout later fails in
  the same arrange-loop basin. It is local offline gate evidence only,
  not a live scorecard and not a CNN training result.

## 0-CURRENT APOLLO PICKUP (2026-07-03 14:15 CDT) - MEMSAFE BEST PASSED FIXED VAL; ADAPTER GATE RUNNING

Apollo is not crashed. User reboot restored it after the first
v10dynonly launch wedged SSH. The memsafe relaunch completed cleanly on
Apollo with swap still unused and the GPU idle afterward.

Completed training condition:

- Condition:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage2_v10dynonly_memsafe_course_correction_20260703/CONDITION.json`,
  sha256
  `0fa28a586e3f566141b35046a193009152c566fd242e82df3510482f2283882c`.
- Output:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v10dynonly_memsafe_course_correction_20260703`
- Log:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v10dynonly_memsafe_course_correction_20260703_train.log`,
  sha256
  `1679f2b50b591a5824c8280fdbb250dbff4661120c6e9eeb9b32a3a9abc83a58`.
- `best.pt` sha256:
  `e82815647038a3749442eb356b4daa6013885c81d16da8ae811fb685b28f27ac`.

Full fixed-v8-val gate:

- Report:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v10dynonly_memsafe_course_correction_20260703_gates/gate_eval_many_full_on_v8val.json`,
  sha256
  `4c46f3ad6cd0b74cb648ca9921a2bd503a5391840660db9281de6e47a0004433`.
- Review page:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v10dynonly_memsafe_course_correction_20260703_gates/index.html`,
  sha256
  `5cdeece00f368d81769463ff0a193a06ca8fdf13e80cd75e8c126da62afd0217`.
  It was verified via local HTTP on `127.0.0.1:18765`: `index.html`
  and the raw full gate JSON both returned `200 OK`.
- V8 baseline: `bc=1.0`, `catch=0.956303`, `level_tp=109/109`,
  `level_fp=0`, `settle_iou=0.655881`, `changed=0.995986`.
- `best.pt`: `bc=1.0`, `catch=0.956303`, `level_tp=109/109`,
  `level_fp=0`, `settle_iou=0.655881`, `changed=0.996143`.
- Strict fixed-val no-loss candidates: epochs 1-4 plus `best.pt`.
  Epochs 5-12 and `last.pt` regress changed-cell below v8.

Adapter gate status:

- A foreground local one-seed probe was stopped after seed `20280001`
  passed, to avoid a four-hour foreground session. Log:
  `synthesis_lab/experiments/megarexus_cnn/runs/adapter_gate_v10dynonly_memsafe_best_20260703.log`,
  sha256
  `a5d5c48e091adceeff231f83152948c119f8b850f3a434e1f19e93e7f8129406`.
  Probe row: model passed, `levels=2`, `actions=119`, all checks true;
  paired random reached `levels=0`.
- Full 10-episode Stage 1a adapter gate is now running detached on
  Apollo, nice level 10:
  Python PID `87565`, launcher shell `87563`.
- Remote output:
  `apollo:/home/frosty/megarexus_train_20260703/synthesis_lab/experiments/megarexus_cnn/runs/adapter_gate_v10dynonly_memsafe_best_apollo_20260703`
- Remote log:
  `apollo:/home/frosty/megarexus_train_20260703/synthesis_lab/experiments/megarexus_cnn/runs/adapter_gate_v10dynonly_memsafe_best_apollo_20260703.log`

Monitor the active adapter gate:

    ssh -o BatchMode=yes apollo 'ps -eo pid,ppid,stat,ni,etime,pcpu,pmem,rss,cmd | grep -E "87565|adapter_gate.py" | grep -v grep; tail -n 40 /home/frosty/megarexus_train_20260703/synthesis_lab/experiments/megarexus_cnn/runs/adapter_gate_v10dynonly_memsafe_best_apollo_20260703.log'

When Stage 1a finishes, sync the remote adapter gate dir and log back.
If Stage 1a passes, run Stage 1b perturbation arrange gate on the same
`best.pt`. Do not run a live scout before the gate stack is complete.

Parallel Stage 1b early evidence run:

- Local host `spark-2949`, detached nice-level 10 Python PID `1632633`.
- Boundary: controlled Stage 1b perturb gate condition, but launched in
  parallel while Apollo Stage 1a is still running. It is useful for early
  failure-basin discovery; it is not promotion evidence until Stage 1a
  also passes.
- Condition:
  `synthesis_lab/experiments/megarexus_cnn/runs/perturb_gate_v10dynonly_memsafe_best_parallel_20260703/CONDITION.json`,
  sha256
  `47fe723866c1b8eee60f6a5ebb93303686f1454974d23379adcce231582fde73`.
- Output:
  `synthesis_lab/experiments/megarexus_cnn/runs/perturb_gate_v10dynonly_memsafe_best_parallel_20260703`
- Log:
  `synthesis_lab/experiments/megarexus_cnn/runs/perturb_gate_v10dynonly_memsafe_best_parallel_20260703.log`

Monitor local Stage 1b:

    ps -eo pid,ppid,stat,sid,ni,etime,pcpu,pmem,rss,cmd | grep -E '1632633|perturb_gate.py' | grep -v grep; tail -n 40 synthesis_lab/experiments/megarexus_cnn/runs/perturb_gate_v10dynonly_memsafe_best_parallel_20260703.log

## 0-CURRENT APOLLO PICKUP (2026-07-03 12:49 CDT) - ACTIVE RUN IS V10DYNONLY_MEMSAFE

Apollo was rebooted by the user after the first v10dynonly launch made
the host pingable with TCP/22 open but SSH banner-dead. That first
launch wrote only startup lines and no epoch checkpoint; do not chase
PID `2697981` or output
`outcome_v10dynonly_course_correction_20260703` as the active lane.

Current active training is the memsafe relaunch:

- Remote workspace: `apollo:/home/frosty/megarexus_train_20260703`
- Active Python PID: `9451` (launcher shell parent `9450` may remain
  until the job exits).
- Condition:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage2_v10dynonly_memsafe_course_correction_20260703/CONDITION.json`,
  sha256
  `0fa28a586e3f566141b35046a193009152c566fd242e82df3510482f2283882c`.
- Code hashes recorded in that condition:
  `train.py` `b1c6e92273e9469959102665245493f5a578e0f2a47c52ee05072d1b282e16d1`,
  `gate_eval_many.py` `46f646d75e7405609dbc76e517ed6867ac8b5de9d0ec7a8a88a660c3ea93b7c0`,
  `model.py` `546dc6d432023f6d7d82f9fbe5407feab806a06aef15f08cf2c4edcc503ab628`.
- Output dir:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v10dynonly_memsafe_course_correction_20260703`
- Log:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v10dynonly_memsafe_course_correction_20260703_train.log`

Memsafe change: `train.py` and `gate_eval_many.py` now accept
`--num-workers`; this run uses `--num-workers 0` to avoid DataLoader
worker duplication of the precomputed dataset cache on Apollo's 16 GiB
RAM. No model, data, optimizer, seed, epoch count, batch size, or
trainable-prefix condition changed from the previous v10dynonly
condition.

12:48 CDT health sample after relaunch:

- Apollo uptime 15 minutes; SSH responsive on LAN.
- Training had passed both dataset caches and reached GPU execution.
- Log showed train dataset `98250` transitions / `64882` unique frames,
  val dataset `18406` transitions / `12330` unique frames,
  `dataloader num_workers=0`, v8 init hash matched, and trainable
  params `1,587,217/3,644,714`.
- RAM: 15 GiB total, 8.2 GiB used, 7.3 GiB available, swap 0.
- GPU: RTX 4080, 2.6 GiB allocated by PID `9451`, utilization 23%.

Monitor:

    ssh -o BatchMode=yes apollo 'tail -n 120 /home/frosty/megarexus_train_20260703/synthesis_lab/experiments/megarexus_cnn/runs/outcome_v10dynonly_memsafe_course_correction_20260703_train.log'

When checkpoints exist, copy the memsafe output dir and log back, then
gate all `epoch_*.pt` plus `last.pt` against the fixed v8 valset with
`gate_eval_many.py --num-workers 0` on Apollo or a safe local setting.
No release or live scout before that gate.

## 0-CURRENT APOLLO PICKUP (2026-07-03 12:24 CDT) — ACTIVE RUN IS V10DYNONLY

Active training was moved off the failed selective Apollo lane and onto
a dynamics-only search-space condition on Apollo. This section
supersedes the 12:00 Apollo block immediately below.

Current active condition:

- Remote workspace: `apollo:/home/frosty/megarexus_train_20260703`
- Active remote PID at launch verification: `2697981`
- Condition:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage2_v10dynonly_course_correction_20260703/CONDITION.json`,
  sha256
  `9a5c133b9d91b99c3486af13fe27576769e1cf8daeb09482fa502965327284b3`.
- Output dir:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v10dynonly_course_correction_20260703`
- Log:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v10dynonly_course_correction_20260703_train.log`

Why this exists: fresh v10, v10ft, and v10sel/Apollo all failed the
strict no-loss gate by introducing level false positives and/or losing
settle IoU. The partial fixed-v8-val gate for the Apollo selective run
is:
`runs/outcome_v10sel_apollo_cont_20260703_gates/gate_eval_many_partial_on_v8val.json`,
sha256
`3bce5e84d86640173418fcb576a0680668b87cda537aab8bc1684a378a5523fc`.
It showed v8 baseline `level_fp=0`, Apollo selective epoch 1
`level_fp=1`, and Apollo selective epoch 2 `level_fp=2`.

The stopped selective Apollo run:

- PID `2679123`, condition `stage2_v10sel_apollo_cont_20260703`.
- Stopped after epoch 6 because same-basin level false positives
  persisted. SIGINT did not exit; SIGTERM did.
- Preserved epoch 6: sha256
  `c418c40424492544768d78a770b2e40dfe8f259cd7c1fa1604e37f69d9b60d7d`.
- Preserved last: sha256
  `b73fac69d40568052f2ffde93f7e0c2fabf5ae3841a12512e5280d5834221be2`.
- Log at stop: sha256
  `d89a68bce487ba2839d55be647a13450e0f2cbd276ed9dd3a22aefc44c41ab20`.

The active v10dynonly run is a new search-space condition, not another
LR/epoch tweak:

- Init checkpoint: `outcome_v8/last.pt`, sha256
  `6e5a28f0d4039688b53fbc6d0cfa7de30452ed678bd5ea70b08fe5a5890ccc9e`.
- Trainable prefixes only:
  `dynamics click_plane_proj up1 up2 skip_gate head_next_frame`.
- Frozen at v8: trunk, Goose policy/change heads, catch/control heads,
  settle_conv/head_settle/head_level, inverse heads, action embeddings.
- Launch verification before SSH became flaky:
  train dataset `98250` transitions / `64882` unique frames;
  val dataset `18406` transitions / `12330` unique frames;
  init hash matched v8; trainable params `1,587,217/3,644,714`.

Important access note: after launch verification, Apollo remained
pingable and port 22 was open, but SSH began timing out during banner
exchange on both LAN and tailnet routes. Do not infer job failure from
that alone. Retry later with:

    ssh -o BatchMode=yes apollo 'tail -n 120 /home/frosty/megarexus_train_20260703/synthesis_lab/experiments/megarexus_cnn/runs/outcome_v10dynonly_course_correction_20260703_train.log'

When SSH recovers, first check whether epoch checkpoints exist, then
copy the run back and gate all `epoch_*.pt` against the fixed v8 valset
with `gate_eval_many.py`. No release or live scout before that gate.

## 0-APOLLO PICKUP (2026-07-03 12:00 CDT) — TRAINING MOVED TO APOLLO

Apollo is now the owner of the active v10 selective course-correction
run. The local duplicate was intentionally stopped after epoch 6 with
KeyboardInterrupt and is NOT the active lane.

- Remote host/workspace:
  `apollo:/home/frosty/megarexus_train_20260703`
- Remote Python:
  `/home/frosty/.local/share/mamba/envs/loom/bin/python`
  (`torch 2.12.0+cu130`, CUDA true, RTX 4080).
- Active remote PID at cutover: `2679123`.
- Remote condition:
  `synthesis_lab/experiments/megarexus_cnn/runs/stage2_v10sel_apollo_cont_20260703/CONDITION.json`
  local sha256
  `2521e43c0f8052ddf9df9247d27fbef8c36114db0ab151aecffe257474f82200`.
- Remote run dir:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v10sel_apollo_cont_20260703`.
- Remote log:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v10sel_apollo_cont_20260703_train.log`.

This Apollo run is a **weights-only continuation** from the local
selective epoch-1 checkpoint:

- Init checkpoint:
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v10sel_course_correction_20260703/epoch_001.pt`,
  sha256
  `ad0e66f22e371336df8906a0f90ecb74eaf382a38db1a40ebdc2188a9c838c95`.
- It restarts AdamW/cosine for 11 more epochs; it is not an optimizer
  resume.
- Reason for epoch-1 transfer point: local epoch 1 had
  `bc=1.000`, `catch=0.953`, `level_tp=113/113`, `level_fp=0`,
  `settle_iou=0.658`; later local epochs introduced level false
  positives.

Apollo handoff verification:

- Train dataset: 98,250 transitions, 64,882 unique frames.
- Val dataset: 18,406 transitions, 12,330 unique frames.
- Trainable prefixes:
  `dynamics click_plane_proj up1 up2 skip_gate head_next_frame settle_conv head_settle head_level`
  with `1,964,563/3,644,714` params trainable.
- First Apollo continuation epoch completed and wrote checkpoints:
  `epoch_001.pt` sha256
  `17dbd4bf713bceb3f1aae4ef797ac20df2926c1d57515b43ba216d5cdc0b19eb`,
  `last.pt` sha256
  `440fe1e071903b68964bf283099cf5f087823e24bd1de8580f91f1d8ec921700`,
  log sha256
  `15cf00439000b837a72257df87b6ce4ab9e4c10c19eadcc668a63f8071958cfe`.
- First Apollo epoch metrics:
  `bc=1.000`, `catch=0.953`, `level_tp=113/113`, `level_fp=1`,
  `settle_iou=0.661`, `changed=0.996`, `ctl=0.999`.
- Apollo GPU process verified with `nvidia-smi` using about 2.6 GB.

Monitor:

    ssh -o BatchMode=yes apollo 'tail -f /home/frosty/megarexus_train_20260703/synthesis_lab/experiments/megarexus_cnn/runs/outcome_v10sel_apollo_cont_20260703_train.log'

When Apollo finishes, copy the remote run dir and log back, then gate
candidate checkpoints on the same v8 valset with `gate_eval_many.py`
before any release decision or live scout. Do not treat the interrupted
local `outcome_v10sel_course_correction_20260703` as release; its local
log ended after epoch 6 with `KeyboardInterrupt`, sha256
`d1e885a5d2a7da75f9f47574b059e9eb446826ce605cb3779e27125ea8e04a45`.

## 0-CODEX PICKUP (2026-07-03 ~06:10 CDT) — STAGE 1A PASS, STAGE 1B FAIL, STAGE 2 CONDITION RECORDED

This section supersedes the older "NEXT COMMANDS" language below.

- Stage 1a local adapter-path gate on the v8 release is CLOSED PASS:
  `runs/adapter_gate_v8last_20260703/adapter_gate.json`, sha256
  `29a71b966ba8e9be70e12739fa15c2accbfb7e39434b7816e1b6b43249a932e6`.
  Command used:
  `/home/frosty40/miniconda3/bin/python3 -B synthesis_lab/orchestration/adapter_gate.py --ckpt synthesis_lab/experiments/megarexus_cnn/runs/outcome_v8/last.pt --out-dir synthesis_lab/experiments/megarexus_cnn/runs/adapter_gate_v8last_20260703 --episodes 10`.
  Result: model 10/10 passed, paired random 0/10 reached
  `levels_completed>=2`; all model checks green (`levels_ge_2`,
  `zero_policy_errors`, `no_frames_fallback`, `no_pour_loop`,
  `pour_budget_respected`, `all_pours_confident`).
- Stage 1b perturbation gate is now WRITTEN:
  `synthesis_lab/orchestration/perturb_gate.py`, sha256
  `8375de6eb1865d06bcc31706e3c16737e05f3dbe54d78bd6a17d2a7dcfa055f7`.
  It replays reach+perturb actions into the paired random env and fails
  closed on replay mismatch.
- Stage 1b v8 release run is CLOSED FAIL/PARTIAL STOP:
  `runs/perturb_gate_v8last_20260703/PARTIAL_STOP_AFTER_2_FAILS.json`,
  sha256
  `6cab49d5db50cb6c68ae222364518332590263ea222a861daa12fdc5c9fe5707`.
  Episodes 20290101 and 20290102 passed (`levels_completed=2`, replay
  matched, discipline checks green). Episodes 20290103 and 20290104
  failed only `levels_ge_2` (`levels_completed=1`, actions=43, replay
  matched, no policy errors, no frame fallback, no pour loop). After two
  failures by episode 4, the 9/10 pass bar was unreachable, so the run
  was interrupted during episode 20290105 planning rather than spending
  more compute in the same representation class.
- Review surface:
  `runs/stage1_course_correction_review_20260703/index.html`, sha256
  `e05dcda434722de07b8f5fcee74aa57b814af62125e30f4007a82c0216b4dec4`.
- Stage 2 collector is now WRITTEN and import/CLI verified:
  `synthesis_lab/orchestration/collect_course_correction.py`, sha256
  `f2c202bf8f39f7fc2198949f5a3264d27b57e3a5dc284d92997e7b41fa713431`.
  It writes standard `.pt` rollout shards; `dataset.py` derives level,
  settle, catch, and control labels from recorded pixels/env fields.
- Stage 2 condition is RECORDED but NOT LAUNCHED:
  `runs/stage2_v10_course_correction_20260703/CONDITION.json`, sha256
  `2c16054ba2f6782603976de1d5311b50d80374adeea8f230b1c4267e7d02a905`.
  The condition isolates the new signal: v8 release shard list plus
  `course_correction_v10_train.pt` / `course_correction_v10_val.pt`.
  It deliberately does NOT fold in the v9 control-play retry because
  that retry failed same-valset no-loss vs v8.
- Stage 2 collector smoke was run on real local SP80 with
  `--episodes 1 --seed 20290103 --policies null_pour --plans 1`.
  Result: reached plumbing, wrote 77 steps and one catch record, final
  `levels_completed=1` (expected null-pour negative). Artifacts:
  `runs/stage2_v10_course_correction_20260703/smoke_null_pour.pt`
  sha256
  `91eaa488d866018bf68a573bc8436e7c796fb9e305be532d1db356bdaae3952d`,
  and `smoke_null_pour.manifest.json` sha256
  `692fdd96dc9e0636555a39bb7163484e66ba648d2edf906b0f3b29b6340f6a61`.
- Current decision: DO NOT run a live scout. DO NOT train a long v10
  run from an implicit recipe. The next valid step is to launch the
  recorded Stage 2 train/val shard collection commands from the
  condition JSON. Training stays blocked until both shards exist. Then
  run the recorded fresh v10 train command and gate before any release
  or live scout.

## 0-TONIGHT (2026-07-03 ~00:25 CDT) — §0B STAGE 0 DONE, ROOT CAUSE PROVEN

Read this first; the 22:40 block below is still-accurate background.
All remaining Stage 1 work is LOCAL compute (bash + GPU) and needs no
model in the loop — run the commands below by hand or under a cheap
babysitter session.

ROOT CAUSE OF THE LIVE FAILURE — FOUND AND PROVEN, NOT A HYPOTHESIS
ANYMORE. `codexlab/arc_play_harness.py::grid_of` returns `frame[0]`,
the FIRST animation frame; SP80 pours return 22 frames. So on the
level-up pour the policy planned on the OLD level-0 scene (glove still
present, `levels_completed` already 1). Proof: driving the unmodified
`run_game` path offline reproduced the live failure byte-for-byte
(same plan `[[0,1,0],[1,-1,0]]`, same score 26.4695, same glove-bbox
roster `[24,16,43,19]`). Repro/validation script (promoted):
`synthesis_lab/orchestration/verify_frame_convention.py`.

STAGE 0 IMPLEMENTED AND VALIDATED (all uncommitted, working tree):

- `codexlab/arc_suite_runner.py::run_game`: obs now carries `"frames"`
  (full animation list) and `"resets"`; accepts `seed=` and a prebuilt
  `policy=` instance. `obs["grid"]` unchanged (other policies/baselines
  untouched). Live runner shares this path unchanged.
- `synthesis_lab/orchestration/megarexus_live_policy.py`: rewritten
  with the course-correction runtime — recorder frame convention
  (cur=frames[-1], prev=frames[-2] on multi-frame actions); ONE pour
  per plan with hindsight refutation (a failed pour excludes that plan
  from the next build; memory survives resets); plan-confidence gate
  (CONFIDENCE_MARGIN=3.0 on margin-vs-5th AND margin-vs-null,
  calibrated: local settled-frame margins 16.9 (v9retry last) / 7.8
  (best) vs 0.63 in the live OOD failure); safe no-op = background
  click, never a desperation pour (the attempt-cap ACTION5 branch that
  burned 408 live actions is GONE); budgets MAX_PLANS_PER_LEVEL=8,
  MAX_POURS_PER_LEVEL=6; per-action frame capture via
  MEGAREXUS_TRACE_DIR env var (npz + meta json).
- VALIDATION (offline, v9retry last, same run_game path): levels=2 in
  ONE life, resets=0, policy_errors=0; roster = bar `[20,36,39,39]` +
  platforms `[28,24,39,27]`/`[8,16,19,19]`; found the demo-basin plan
  (platform2 +2, platform1 +3, bar +1, bar last); margin_vs_5th 16.89,
  null plan scored −42.88 (margin_vs_null 69.11). First end-to-end
  adapter-path plumbing completion.
- NEW gate script: `synthesis_lab/orchestration/adapter_gate.py`
  (Stage 1a). Per-episode pass checks: levels>=2, zero policy errors,
  zero frames-fallback, no pour loop (max 2 consecutive ACTION5),
  pour budget respected, all pours confident. Paired random baseline
  included. Prints one line per episode (flush) — tail the log to
  babysit.

SMOKE RESULT (v8last, seed 20280001, completed 00:32): **PASS, all six
checks green** — levels=2, 119 actions, 1 reset, paired random 0.
Artifacts: `runs/adapter_gate_smoke_v8last_20260703/` (report + frame
captures). Detail worth reading: plumbing completed on ONE build/ONE
pour (score +19.95, margins 23.5/41.2 — the v8 demo-basin plan). The
episode then spent its remaining budget on the NEXT level (three-spout):
8 builds, 5 confident-margin pours, 5 plans refuted, no-op holds on 2
unconfident builds, zero pour loops — course correction behaving as
designed on a level it has never trained on, failing SAFELY. OPEN
OBSERVATION for a future session: all next-level plan scores were
absolutely NEGATIVE (−0.2..−13) yet relative margins cleared 3.0, so 5
pours were spent on plans the level head itself doesn't believe in.
Consider an absolute-score floor (pour only if best score is in the
level head's positive range) — measure first, don't hand-tune.

NEXT COMMANDS (Stage 1a, run from repo root, ~40–60 min each on GPU):

    nohup /home/frosty40/miniconda3/bin/python3 -B \
      synthesis_lab/orchestration/adapter_gate.py \
      --ckpt synthesis_lab/experiments/megarexus_cnn/runs/outcome_v8/last.pt \
      --out-dir synthesis_lab/experiments/megarexus_cnn/runs/adapter_gate_v8last_20260703 \
      --episodes 10 \
      > synthesis_lab/experiments/megarexus_cnn/runs/adapter_gate_v8last_20260703.log 2>&1 &

    # then the same with
    #   --ckpt .../outcome_v9_retry_20260702/last.pt
    #   --out-dir .../adapter_gate_v9retry_last_20260703
    # verdict: grep '"gate_verdict"' <out-dir>/adapter_gate.json

STAGE 1B (perturbation gate) — NOT YET WRITTEN. Spec for the next
implementer: new `orchestration/perturb_gate.py` reusing the
arrange_probe skeleton — reach plumbing with trained catch
(`reach_next_level`), then a TEST-HARNESS perturber (seeded rng, 1–3
rounds of `click_and_move` on random components, random axis, 1–3
steps, never pours) creates the start state; then construct
`MegarexusArcPolicy(ckpt)` fresh and drive it in a loop that mirrors
run_game's obs contract (frames list + resets) until completion or
budget; paired random executes one sampled plan + pour from the same
perturbed state. Heldout seeds 20290101+. Honesty boundary: the
perturber is harness authorship (test-state generator), never model
capability, never labels. Pass bar: model completes with pour
discipline on ~9/10 heldout perturbed states; random stays near the
arrange-probe baseline. Run v8last FIRST — if v8+fixed adapter passes,
NO training is needed (§0B Stage 1).

Stage 2 training rules unchanged (§0B): only if Stage 1b fails; short
sequences with gates between; no long run without validated
improvement. Do NOT run a live scorecard before 1a+1b pass.

## 0. CURRENT STATE HANDOFF (2026-07-02 22:40 CDT)

Read this first. No training/gate/live-scorecard process is currently
running. Handoff check:
`pgrep -af 'arc_live_suite_runner.py|run_v9_retry_gates_20260702|arrange_probe.py|level_probe.py|gate_eval.py|train.py'`
only matched the probe command itself at handoff time. Do NOT launch
another live ARCPrize scorecard or new training run before the next local
anti-reward-hack gates below exist and pass.

Release state:

- `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v8/last.pt`
  remains the release checkpoint.
- `outcome_v9_retry_20260702` trained cleanly to 40 epochs and is a useful
  candidate/scout artifact, but it is NOT release. Same-valset no-loss
  against v8 did not hold:
  - v8last: `bc/action_type_acc=1.0`, `level=109 tp / 0 fp`,
    `catch/acc=0.9563`, `control/acc=0.9995`, `settle/iou=0.6559`.
  - v9retry last: `bc/action_type_acc=0.9878`, `level=108 tp / 0 fp`,
    `catch/acc=0.9630`, `control/acc=0.9995`, `settle/iou=0.7219`.
  - v9retry best: `bc/action_type_acc=0.9837`, `level=109 tp / 1 fp`,
    `catch/acc=0.9613`, `control/acc=0.9998`, `settle/iou=0.6893`.
- Behavioral probes were green for both v9retry checkpoints:
  `level_probe` 20/20 model vs 5/20 random, and `arrange_probe
  --search exhaustive-x` 10/10 model vs 1/10 random. Outputs are under
  `synthesis_lab/experiments/megarexus_cnn/runs/
  outcome_v9_retry_20260702_gates/`.

Live ARCPrize scout:

- A real live SP80 scorecard was run with the new `megarexus_cnn`
  policy adapter, not Tufa/CONTRA mislabeled as Megarexus.
- Result path:
  `codexlab/runs/arc_live_sp80_megarexus_v9retry_last_20260702T2224/`.
- Main evidence file: `result.json`; review file: `index.html`;
  manifest: `report_manifest.json`.
- Scorecard id: `e29a35b6-fdac-47d0-87e7-f7d3bc5925f7`.
- Closed ARC SDK payload: score `4.761904761904762`, `1/6`
  levels completed, 500 scorecard actions, 81 resets, 0 policy errors,
  final state `GameState.GAME_OVER`.
- Public HTTP probes at handoff returned `/api/v3/scorecards/<id>` 404
  and `/api/models/scorecard/<id>` 500. Treat the closed SDK payload in
  `result.json` as the current evidence.
- Loaded checkpoint:
  `synthesis_lab/experiments/megarexus_cnn/runs/
  outcome_v9_retry_20260702/last.pt`, sha256
  `da8cb8df1ee8c433d67056268101270e3d1751d3462cb65588e3dbd0994a9864`.

Failure interpretation (post-mortem sharpened 2026-07-02 late session,
from the `result.json` ledger):

- This was exactly the reward-hack/generalization check. Local arrange
  success did NOT transfer to the live adapter path.
- The live policy completed level 0 in 5 clean actions, then entered
  the plumbing stage and planned IMMEDIATELY on the first frame after
  level-up. The roster it planned over was wrong: "movable 0" bbox
  `[24,16,43,19]` is exactly the level-0 glove's final position and
  glove geometry (20 wide), and "movable 1" `[40,56,43,59]` matches the
  bottom-spout column (x40-43), not the bar/platforms. Leading
  hypothesis: it planned on a level-TRANSITION/unsettled frame (offline
  `step()` settles animations; live does not), with the 2-frame stack
  also straddling the level boundary. Unproven without frame captures —
  which is why Stage 0 below comes first.
- The level head's plan scores were nearly flat (top5 26.47, 26.47,
  25.841, 25.841, 25.841) vs the local decisive needle (+19.95 vs
  +3.09 runner-up) — out-of-distribution input, everything looked
  level-up-ish. Chosen plan: movable 0 right once, movable 1 left once
  (`plan_score=26.4695`). It did not complete the plumbing stage.
- After two failed pours it hit the attempt cap and emitted 408
  consecutive `megarexus_arrange_attempt_cap` ACTION5s: every 5th pour
  = game over, runner resets, live RESET keeps `levels_completed=1`, so
  reset detection (needs levels to drop or the level-0 frame hash)
  never fired and it never replanned. 495 of 500 scorecard actions were
  burned on that one stage. Two distinct failures: planner fed input it
  never trained on, AND zero recovery path. Do not describe v9retry as
  a generalized planner.

Code/artifact state:

- New live adapter: `synthesis_lab/orchestration/megarexus_live_policy.py`.
- Policy registration: `codexlab/arc_play_harness.py` adds
  `megarexus_cnn`.
- `codexlab/arc_suite_runner.py` was adjusted so live reports are labeled
  as live scorecards in manifests/traces.
- SDK/local loading touched
  `environment_files/sp80/589a99af/metadata.json` by changing only
  `date_downloaded`.
- Existing run artifacts and checkpoint changes are dirty/untracked. Do
  not revert them unless the user explicitly asks.

## 0B. ACTIVE GOAL — "COURSE CORRECTION" TRAINING REGIME
## (set by user 2026-07-02 after the live scout post-mortem)

Organizing principle: the live failure was not "bad plan" alone — it
was ZERO recovery. Stop optimizing one perfect plan executed from a
clean settled state; train and gate the ability to correct course from
off-nominal states (transition frames, perturbed layouts, failed
plans). Stages are strictly ordered — do NOT skip ahead to training or
a live scorecard. Honesty rules (§2) apply throughout: hindsight
env-field + pixel labels only, paired random baselines, no answer-key
states.

- **Stage 0 — adapter integrity (no training).**
  a. Full per-action frame capture (grids and/or PNGs, not just
     hashes) in `megarexus_live_policy.export_trace` — this run's
     transition-frame hypothesis could not be proven from hashes.
  b. Frame-stability gating: never build a roster/plan until
     consecutive frame hashes are stable for k steps after a level
     change; never stack `prev` across a level boundary.
  c. Recovery path: on attempt cap, clear arrange state and replan
     from the now-settled frame instead of pouring; handle live RESET
     semantics (`levels_completed` does NOT drop — detect the
     level-restart frame instead). Pours are the scarce resource:
     never emit ACTION5 without a currently-trusted plan.
  d. Plan-confidence gate from the model's own head (honest by §2):
     if the plan-score landscape is flat or the null plan is
     competitive (local calibration: winner +19.95 vs runner-up
     +3.09; live failure: 26.47 vs 25.84), do NOT pour — re-observe
     and replan.
  e. Docs hygiene: unify level naming to one convention
     (levels_completed-based) across notes/probes so "level-2
     completions 10/10" can never again read as three levels cleared.
- **Stage 1 — perturbation arrange gate (measure BEFORE training).**
  From legal post-L0 states apply controlled random pre-arrange
  movements (seeded, paired). Run the SAME `megarexus_cnn` adapter
  path (not just `arrange_probe.py`) on local SP80, raw frame stream
  including transitions, vs paired random. Also gate: complete L0+L1,
  zero policy errors, no capped-pour loop. Run this on the v8 release
  first — if v8+fixed-adapter passes robustly on heldout
  perturbations, NO new training is needed (loop directive: optimize
  existing systems before adding new ones).
- **Stage 2 — course-correction training (ONLY if Stage 1 fails).**
  Collect a narrow v10 shard from the failed/perturbed states,
  including mid-plan deviation states (execute k steps of a plan,
  then relabel from there in hindsight). Train arrangement robustness
  only — existing heads, no new head without evidence. Labels remain
  pixels + environment fields only; no answer-key mechanics,
  source-state labels, or hand-authored win model.
- **Stage 3 — full gate stack before any release/claim.**
  Same-valset no-loss vs `outcome_v8/last.pt` (all heads), level probe
  20/20 vs 5/20 random, arrange probe 10/10 vs 1/10 random, Stage 1
  perturbation gate on heldout perturbations, Stage 1 local
  adapter-path gate.
- **Stage 4 — one new live SP80 scout.** Success criterion: at least
  `levels_completed >= 2`, zero policy errors, no capped-pour loop,
  and complete per-action frame evidence.

## 0A. EARLIER ACTIVE SESSION LOG (2026-07-02, ~12:45)

Fresh session started mid-/loop for token efficiency (user's call).
Sufficient context = this section + §2 (rules) + §4 (results) +
ADDING_CNN_ELEMENTS.md (repo root) + the last two RESEARCH_LOG
entries. Do NOT re-read model.py/dataset.py/train.py or old probe
transcripts unless you are editing them — probe JSONs and run configs
are the evidence. Python is `/home/frosty40/miniconda3/bin/python3 -B`;
run from the repo root with absolute or repo-relative paths (cd does
not persist); NOT a git repo.

**Active /loop directive (user, verbatim):** "[interval = test based,
sleep between cnn runs] train one small element at a time, ensuring no
loss, focusing on optimizing all existing untrained systems before
adding new ones, use logic, best candidate applies, nothing is sacred,
no cheating." Protocol: long jobs (collect/train/probe) run as
BACKGROUND Bash tasks; end every turn with ScheduleWakeup (3600s max —
match shorter test durations when known) carrying the /loop prompt
verbatim; the old session's background-completion notifications do NOT
reach a new session — poll with `pgrep -af train.py` + `ls` on the run
dir instead.

**UPDATE 2026-07-02 15:20 — BABYSITTER MODE (lighter-model session OK
until late afternoon).** v9 at epoch ~34/40, ETA ~15:50 CDT. Gate
steps 1–2 below are AUTOMATED: `synthesis_lab/orchestration/
run_v9_gates.sh` is already running in the background — it waits for
train.py to exit, then runs gate_eval ×3 (v9 last / v9 best / v8
baseline, on v8's valset) and level_probe + arrange_probe on symlinked
scratch run dirs. Outputs → `runs/outcome_v9_gates/` (timeline.txt =
progress, DONE = finished, train_log_tail.txt = final train summary).

Babysitter scope — each wake do ONLY this: (1) `pgrep -af train.py`;
(2) `tail runs/outcome_v9_gates/timeline.txt` + check for DONE;
(3) glance at the newest `runs/outcome_v9_gates/*.log|*.txt` for
tracebacks. If the runner died before DONE and training is gone,
relaunch `bash synthesis_lab/orchestration/run_v9_gates.sh` as a
background task (it is idempotent, overwrites its own outputs). Do
NOT: interpret gate numbers, make release decisions, edit code or
notes, start new training/collection, or write into canonical
`runs/outcome_v9`. When DONE appears, report "gates complete, awaiting
adjudication" and keep sleeping. End EVERY turn with ScheduleWakeup
carrying the /loop prompt verbatim (§ directive below): ~1200s near
the ~15:50 train→gates handover, 2700–3600s otherwise. Old sessions'
background-task notifications do NOT reach a new session — poll files.
Adjudication (no-loss comparison, probe 20/20 & 10/10 verdicts,
release decision, docs, next element) is reserved for the main
session: it resumes at "WHEN v9 LANDS" step 1 using the already-
written outputs in `runs/outcome_v9_gates/` instead of re-running.

**UPDATE 2026-07-02 16:59 — MAIN SESSION PICKUP.** Gate chain finished
at 16:28:47 (`runs/outcome_v9_gates/DONE`). Adjudication result:
`outcome_v9` is NOT a release. Reason: the recorded 40-epoch training
log only reaches epoch 36 and has no final summary; same-valset no-loss
vs v8 is also not clean (both v9 checkpoints drop `bc/action_type_acc`
from 1.000 to 0.9878 and add one level false positive; `last.pt` also
misses one old level positive). Behavioral probes did hold for both
checkpoints: level 20/20 vs random 5/20; arrange 10/10 vs random 1/10.

**IN FLIGHT: outcome_v9_retry_20260702 training** (same task #5 = pure
object-control curriculum; same recorded v9 condition, new output dir
so the partial `runs/outcome_v9` artifacts stay intact). Launched in
tmux session `megarexus-v9-retry` at 2026-07-02 16:59 CDT. Command is
the exact v9 recipe recovered from `runs/outcome_v9/last.pt` config:
40 epochs, seed 0, stack_frames=2, batch 64, lr 3e-4, v8 shard list
plus `control_play_train.pt`; val shards are v8 vals plus
`control_play_val.pt`. Log:
`synthesis_lab/experiments/megarexus_cnn/runs/outcome_v9_retry_20260702_train.log`.
Run dir:
`synthesis_lab/experiments/megarexus_cnn/runs/outcome_v9_retry_20260702`.
Done when that log prints `[train] done...` and `last.pt` exists. Then
gate the retry against `outcome_v8/last.pt` exactly as below, substituting
`outcome_v9_retry_20260702` for `outcome_v9`. Until a retry passes,
`outcome_v8/last.pt` remains the release checkpoint.

**WHEN v9 LANDS — gate sequence (all commands verified this session):**

1. Same-valset no-loss gate, BOTH checkpoints, vs v8 on v8's valset:
   `python3 -B synthesis_lab/orchestration/gate_eval.py --ckpt
   synthesis_lab/experiments/megarexus_cnn/runs/outcome_v9/last.pt
   --val-shards-from .../runs/outcome_v8/last.pt` — then the same with
   `--ckpt .../outcome_v9/best.pt` and `--ckpt .../outcome_v8/last.pt`
   (the baseline side), compare head-by-head. No head may lose more
   than borderline noise (see RESEARCH_LOG on v5's 1-sample catch dip).
2. Behavioral gates (probes load `<run-dir>/best.pt` by NAME — make a
   scratch dir in YOUR scratchpad, symlink the gated checkpoint as
   best.pt, point --run-dir there, never at the canonical run dir):
   - `orchestration/level_probe.py` — must reproduce 20/20 model vs
     5/20 random (seeds as in the script defaults).
   - `orchestration/arrange_probe.py --search exhaustive-x` — must
     hold 10/10 model vs 1/10 random (paired seeds 20280001–10).
3. v9-specific checks (the element under test is control coverage):
   control/acc must hold 1.0 on v8's valset AND be strong on
   control_play_val (in-training log prints it); worth checking
   whether plain `rank_movables` (non-imagined) now ranks the
   platforms positive — if yes, note it: the imagined-click
   composition in arrange_probe could simplify.
4. Then: release decision, update §3/§4 here + RESEARCH_LOG entry,
   mark task #5 done, pick the next element per the loop directive
   (§6/§7 backlog; level-3 generalization of the planner is the
   natural probe-side candidate).

Known open observation for a future element: the learned glove finder
is appearance-sensitive — a selected (recolored) glove breaks it
(collect_control_play.py header, smoke eps 0/3). Level-1 probing
stays off until the finder tolerates recoloring.

## 1. Mission

Synthesize the user's separate systems into ONE genuine trainable CNN in
`synthesis_lab/` — "without cheating":

- **LeWM/Rex** — world model / action owner (conv dynamics, FiLM action
  conditioning),
- **Contra** — perception/thinking (components as input planes),
- **Stochastic Goose** — runtime (4101-logit action head, stochastic
  collection),
- **Portband** — the "present": moveable objects + per-side port edges
  (gloves, catch labels).

Target game: ARC-AGI-3 SP80 ("liquid pour", game_id `sp80-589a99af`),
core mechanic "catch and redirect paint". 7 levels (0–6). All runs are
local OFFLINE (`arc_agi.Arcade`, `environment_files/`), labeled
SCOUT_EXPERIMENT — never benchmark claims, never live scorecards.

## 2. Honesty rules (user rulings — NON-NEGOTIABLE)

1. **Answer-key ban** (user, emphatic): hand-authored models of a
   specific game's mechanics/win conditions (e.g.
   `codexlab/contra_head/sp80_spill_model.py`) must NEVER touch training
   — not as labels, informer, agreement metric, or ghost verifier. "We
   are training systems, not the answers." Labels never come from live
   engine internals, sprite tags, or game source.
2. Win/outcome semantics are learned from **environment fields +
   pixels in hindsight only** (`levels_completed`, `state`, frame
   diffs). Human demonstrations (corpus) ARE legitimate experience.
3. Every trained element gets baselines (persistence, majority, base
   rate), unseen-seed val shards, one backlog element per run,
   ablation-gated. Negative results are documented, not buried.
4. Ownership language: say which role owned an outcome (world model =
   imagination, controllability head = selection, env fields = level
   evidence). Scripted/demonstration-derived rollouts must never be
   presented as model capability (see §5, level-2 completion).
5. User guidance is legal as research direction (what to measure), and
   demonstrations are legal as data; guidance must not become a
   hand-coded label/reward.

## 3. Key files

- Model: `synthesis_lab/experiments/megarexus_cnn/model.py` —
  MegarexusCNN, 3.64M params. Trunk (h16/h64, 2-frame stack), FiLM
  dynamics, Goose head, catch head (glove-plane masked pooling), inverse
  dynamics, and outcome_v4 heads: `forward_settle` (settle+level-up),
  `forward_control(frame, plane, action_id)` (controllability).
- Data: `.../dataset.py` — hindsight labels; control samples via
  `MOVE_ACTION_AXIS` + displacement rule (`on_axis*sign >= 0.4`,
  `|off_axis| <= 1.0`); `has_control` aux rows gated OUT of main losses.
- Train: `.../train.py` — run: `python train.py --run-name <name>` from
  the experiment dir. Best checkpoint → `runs/<name>/best.pt`.
- Probes (in `synthesis_lab/orchestration/`):
  - `level_probe.py` — paired model-vs-random full episodes. Current v3:
    learned glove finder (`--finder learned`), miss-gated + budget-aware
    `careful_place`, per-pour side-block/gauge logging. Reports →
    `runs/<run>/level_probe*.json`.
  - `behavioral_catch_probe.py` — `imagine_pour_column` (world-model
    ghost), `move_n`, `model_place`, `random_place`.
  - `collect_catch_play.py` — curriculum collector (`find_gloves`
    interaction probe, `EpisodeRecorder`, `step`, `track_glove`).
- Adapter: `synthesis_lab/engine_adapters/portband_glove_adapter.py` —
  `frame_components`, `glove_ports` (Portband `_object_ports`),
  `catch_label(frames, mask, ports, sides=..., max_frames=...)`,
  `component_displacement`.
- Human corpus: `corpus/sp80_liquid_pour/manifest.jsonl` — 1245 frames,
  6 episodes, per-frame `level`/`state`/`action`/PNG + spoken
  transcript. All episodes reach level 4+; one full WIN
  (`morning_liquid_pour_2..._episode_02_winbank_005`).
- Current release checkpoint: `runs/outcome_v8/last.pt` (NOT best.pt —
  release is chosen by the external same-valset + behavioral gates;
  probes load `best.pt` by name, so symlink last.pt as best.pt in a
  scratch dir).

## 4. Verified results so far (all on unseen-seed val / paired seeds)

- World model: h=1 changed-cell acc 1.00, flow acc 0.987, beats
  persistence; inverse dynamics 0.990.
- Catch head (v3): 1.000 vs 0.528 base (106 pours).
- Behavioral catch: 77/80 strict catches vs 30/80 random.
- Controllability head (outcome_v4): 1.000 vs 0.528 base (1,848
  samples); ranks level-2 bar first (+4.31 vs ≤−14).
- Level-up head: 3/3 tp, 0 fp on 84 val pours (thin positives — 18
  train).
- Level probe: model **20/20 level-1 completions** (learned finder;
  was 10/20 with interaction probing), random baseline 5/20 reproduced
  exactly. Settle head: dead on thin v4 data, alive at IoU 0.656 with
  v8's plumbing coverage.
- outcome_v8 (release, 2026-07-02): level head 109/109 tp 0 fp on
  diverse positives, control 0.999 (n=5,682), all v7 gates hold.
- **Arrange planner (2026-07-02): model 10/10 plumbing completions vs
  1/10 paired random** (`orchestration/arrange_probe.py --search
  exhaustive-x`). Offsets × last-mover enumerated in imagination,
  scored by the model's own level head; chosen plan = platform2 +2,
  platform1 +3, bar +1, bar last (+19.95, runner-up +3.09);
  imagined-vs-real agreement 1.0. First completions where arrangement,
  scoring, and execution order all came from learned heads. Scope
  caveats in RESEARCH_LOG (level-2 layout deterministic across seeds;
  plan sits in the demo basin; level-3 generalization untested).

## 5. Level-2 mechanic — established 2026-07-01 (big session)

Chain: level probe v3 → diagnostics (session scratchpad, gone; results
in scout notes) → human-corpus mining → verified completion.

- **Pour supply**: each level grants ~4 animated pours; the 5th pour
  call returns 1 frame and instant GAME_OVER. Pours are the scarce
  resource — the user's "you can't miss".
- **Gauge quirks**: drains COLOR→COLOR (14→0) so non-background counts
  are blind; row 0 on level 1 but **row 63 on level 2**; ~1.5
  cells/action; NOT the level-2 killer (moves-only survives to 58/64).
- **Flow**: level 2 is an UPWARD jet from a bottom spout (x40-43); it
  splits on the bar's underside; risers climb both flanks (bar_x0−4..−1
  and bar_x1+1..+4) toward three funnel mouths (x16-19/32-35/48-51).
- **Movables**: the bar AND both 8-colored platforms move via
  click-select + move, all 4 axes, ±4px/step. Bar position alone
  changes nothing; the platforms are the real flow-blockers.
- **Offline vs live**: humans pour first and steer live; the offline
  env settles the whole flow inside `step(5)`, so live steering does
  NOT transfer. The offline solve is **arrange plumbing first, pour
  last**: platform2 right×2 → [36,24,47,27], platform1 right×3 →
  [20,16,31,19], bar right×1 → [24,36,43,39], then one pour →
  `levels_completed=2`. Verified on seed 20280001; level-2 layout is
  deterministic across seeds.
- **This completion is demonstration-derived and scripted** — never
  present it as model capability; never hardcode the configuration as
  label/reward. Level 3: three spouts, three funnels, wider platforms.
- Settle/fill signal: level 2 has NO persistent fill pixels (settled
  frames differ only by gauge cells). Outcome semantics = level-up
  head. Do not resurrect the fill-delta idea without new evidence.

## 6. THE NEXT TASK

**THE ACTIVE GOAL IS §0B — the "course correction" training regime**
(user directive, 2026-07-02). Everything below in this section is
historical context for how we got here.

~~Plumbing-play curriculum + outcome_v5 + planner~~ — **DONE 2026-07-02**
(outcome_v8 + arrange_probe 10/10 vs 1/10; see §4 and RESEARCH_LOG for
the v5→v8 saga: checkpoint policy, label leakage, coverage holes,
selection-conditional control, last-mover state).

~~Pure object-control curriculum~~ — **RUN 2026-07-02, NOT RELEASED**:
v9 and the exact v9retry both trained; behavioral probes held but
same-valset no-loss vs v8 did not; `outcome_v8/last.pt` remains the
release. The v9retry candidate then failed the live scout beyond level
0 (§0), which is what triggered the §0B course-correction goal.

Original element description (kept for reference), under the /loop
directive "one small element at a time, optimize existing systems
before adding new ones":

1. Collector `orchestration/collect_control_play.py` — broad
   click/select/move coverage as its OWN curriculum, not a side effect
   of arranging: clicks on movables, non-movables, and empty cells;
   select→switch-selection transitions; moves with nothing selected
   (negatives); all four axes from varied positions. Selection recolors
   movables (9=selected, 8=not) — track by click-point/position+area.
2. This feeds the EXISTING control + dynamics heads (no new head). The
   planner provably depends on their composition
   (`rank_movables_if_clicked`: imagine the click, then query control
   on the imagined frame) — that composition path is what the
   curriculum should make robust.
3. Gate as always: same-valset vs outcome_v8, level probe 20/20,
   arrange probe 10/10 must all hold.

## 7. Remaining backlog (see scout notes §"Shape/map systems audit")

Death head P(GAME_OVER within k | stack, action) — note the pour-supply
mechanic makes "pours since level start" the key feature and the
2-frame stack can't count; consider a supply-plane input or recurrence.
Cost-to-intercept head. Edge-adjoin head (70 trials in hand). v4
curriculum re-collection (old shards have the strip-glove bug —
completions there weren't glove-driven). Tier 4: Contra scene-surface
distillation (floor/wall/checkerboard/facing/contact-graph). Tier 5:
symmetry/lock-key + port alignment. Ghost re-run on human segments.
Outcome-weighted self-imitation once model-guided completions exist.

## 8. Gotchas

- `catch_label` sides: `("top",)` + `max_frames=8` is the strict catch;
  side labels fire on EVERY normal catch (liquid spreads down the
  flanks) — they do not isolate blocking as-is.
- Level probe pairing: random baseline must keep the interaction-probe
  finder and pour-always policy, or the 5/20 baseline breaks.
- `Arcade` OFFLINE prints scorecard-manager INFO lines — they are local
  artifacts, not live scorecard use.
- Level transitions change the scene AND the selected object; re-find
  and re-click the movable after every level-up.
- Manifest convention: row t's action produces frame t+1. Human corpus
  `level` field = levels_completed (0-indexed current level).
