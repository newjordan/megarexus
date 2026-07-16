# HANDOFF — synthesis_lab (megarexus), written 2026-07-01

## 0-NOW — 2026-07-15: slow-burn history/action experiment stopped at user request

The user asked to continue training Megarexus under a low-load slow-burn
constraint (one pinned CPU core; GPU disabled; `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`) and to consult SOL/Xhigh when
the next target was unclear. The workday ended with the active trainer stopped
cleanly. No training or collection process remains, and no final checkpoint was
written for this experiment.

### Exact stop-state

- Dataset: `synthesis_lab/experiments/learned_physics/runs/history_action_quartets_s94016_20260714.pt`
  - 192 quartet anchors, 768 history rows, 0 L0→L1 replay-hash mismatches.
- Trainer: `experiments/learned_physics/train_history_action_model.py`
- Intended output (absent): `synthesis_lab/experiments/learned_physics/runs/history_action_model_s94016_20260714.pt`
- The trainer was launched with 500 updates and ran for roughly 43 minutes on
  one CPU core before being terminated at the user's end-of-day request. It
  was CPU-active, not proven hung; because the output is written only after
  training and evaluation, there is no partial model artifact to score.
- All generated datasets and scripts were preserved. The working tree was
  already uncommitted; do not overwrite unrelated changes.

### What led to this target

Earlier small models showed strong teacher-forced pixel accuracy but failed
free rollout or action-ablation gates; the action-counterfactual panel then
recovered identity at 0.8906, macro F1 0.4125, per-action F1
`[0.363, 0.696, 0.285, 0.306]`, head-swap drop 0.048, and pairwise Spearman
0.562. These failed the pre-set promotion gates, so nothing was promoted.
SOL/Xhigh recommended a shared history-conditioned action transition model:
L0 prefix transition + recent SELECT response + up to four preceding MOVE
transitions + current frame/selected mask + candidate ACTION1–4, with a shared
decoder and history/prefix ablations. That recommendation produced the dataset
above and the trainer, but the run did not reach a result artifact.

### Resume procedure

1. Inspect the trainer before relaunching. It has already had three pre-training
   shape/loss fixes: GRU input 27, decoder input 64, and scalar weighted loss.
   Run `python3 -m py_compile experiments/learned_physics/train_history_action_model.py`.
2. Prefer a bounded smoke or a more efficient batched implementation first;
   the 500-update script is too slow for unattended one-core use without
   progress checkpoints. Do not treat an interrupted run as evidence.
3. If relaunching, use the recorded low-load envelope:
   `taskset -c 0 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`
   and the existing dataset path above.
4. Apply the SOL gates before any promotion: test macro F1 ≥ 0.60, every
   action F1 ≥ 0.45, identity ≥ 0.90, action permutation macro drop ≥ 0.15,
   pairwise Spearman ≥ 0.65, and history ablation loss ≥ 0.10 macro F1.
   These gates are not yet measured for s94016. If the model fails, stop
   rendered-frame recurrence work and move to full transient response frames
   from each environment action, per the SOL boundary.

### Shutdown verification

The trainer PID was checked after termination and was gone; no matching
`train_*` or `collect_*` Python process remained. The absent output checkpoint
is the authoritative reason this target is incomplete.

For a fresh agent session picking up this work. Read this top to bottom
before touching anything. The two notes files are the ground truth:
`notes/megarexus_cnn_scout.md` (full experiment detail) and
`notes/RESEARCH_LOG.md` (dated summary entries). User memory lives in
`~/.claude/projects/-home-frosty40-megarexus/memory/`.

## 0-NOW (2026-07-13 ~15:20): THE SP80 IMAGINATION GAP IS CLOSED — THE POUR DYNAMICS WAS ARCHITECTURALLY UNDER-POWERED, AND THE WIDE LANE FIXED IT

**Read first:** `synthesis_lab/notes/WIDE_MODEL_SESSION_20260713.md`. honesty_audit
exit 0 at session start and end (97/46/6, active 85/23/0, unchanged).

User directive mid-session: **"make sure we are doing more than sp80 — need to train
a wide model."** Two lanes ran; the wide lane ended up fixing the home game.

**1. The L1 wall was misdiagnosed.** The §7 work order predicted "merged_v5's L1
positives activate L1 ranking". The win probe NEVER needed them: trained with ZERO L1
positives it ranks all 4 known winners **2/4/3/1 of 19,751** when reading the REAL
settled board (direct AUC 1.0000). Reading the IMAGINED board it ranked them
3/550/2681/113. **The entire L1 gap was imagination error.** New tool
`eval_l1_winner_rank.py` (learned_physics) is the honest metric.

**2. Root cause: `DynamicsUNet` is a 2-level UNet (1.06M params) and a pour is
LONG-RANGE.** Its receptive field could not represent paint propagating across the
board. The wide lane's `WideWorldModel` (3-level, 7.7M params, built for 25 games),
trained FROM SCRATCH on the same SP80 data with the same holdout, gets **~0.99
changed-pixel agreement vs the specialist's 0.83**, and the deployed chain goes:

| chain (probe FIXED, winners' episodes held OUT) | AUC | winner ranks of 19,751 |
|---|---|---|
| merged_v4 dynamics (this morning's deploy) | 0.9578 | 3, 550, 2681, 113 |
| merged_v5 + D4 (orientation fix — real, but superseded) | 0.9899 | 10, 783, 12, 1 |
| **SP80 on the WIDE architecture** | **0.9996** | **38, 3, 2, 1** |
| *direct read (upper bound)* | *1.0000* | *2, 4, 3, 1* |

Winners score 0.71/0.81/0.96 ABSOLUTE (loser max 0.388) — genuine generalisation, not
a ranking artifact. Deploy via `wide_dynamics_adapter.py`; `solve_loop._load_models`
auto-detects a wide checkpoint and freezes the action to ACTION5. **Deploy ckpt:
`experiments/wide_world_model/runs/sp80_from_scratch_20260713/best.pt`** + the
unchanged merged_v4 `win_probe_d4`.

**3. What this buys: a ~16x BUDGET REDUCTION, not more winners.** Paired arms
(identical pools — the pool RNG is keyed on the seed alone) show ranked recovering the
SAME known winners (90005/90006/90023...) with **8 pours instead of 128**. Ranking does
not create winners; it makes each board 16x cheaper. **So spend it on 16x more BOARDS**
— `runs/harvest_queue.sh` (600 fresh seeds, top-8 budget) is queued to do exactly that.
Every extra L1 winner is a board that reaches L2, and only 4 boards have EVER reached L2.

**4. MANDATORY METHOD — hold out the winners' episodes (2702/2703/2712/2759).**
Trained on all of merged_v5 the chain looks transformed (AUC 0.9962) but that model
trained on the winners' own transitions; its clean twin scores 0.9566 = no change.
**The gain was pure memorisation.** merged_v5 also has **0 L1 val positives**, so
`val_auc`/`val_chain_auc` are L0-ONLY numbers.

**5. Falsified levers (do not re-try):** softmax-instead-of-argmax in the chain; more
arranged-L1 data; and the perceptual fidelity term (`--perceptual-probe`, which fixes
L0 beautifully — win-read agreement 0.998 — but does not move L1 ranks).

**6. Wide lane** (`experiments/wide_world_model/`, new): 25/25 games boot offline and
share one substrate (64x64, colours 0-15, actions 1-**7**). 264,520-transition corpus.
One action-conditioned model, **NO game-id input**, **val_changed 0.957** (identity
baseline 0.0 — and note identity scores 0.9916 OVERALL, so overall agreement is a
trap). **HONEST NEGATIVE: zero-shot transfer FAILS** — 0.021 on 5 unseen games, 0.048
on sp80 held out. The games share a substrate but not dynamics. The wide model is a
separate deliverable, NOT an SP80 improvement; what SP80 got from it is the
ARCHITECTURE.

**7. New data:** `merged_v6` (42,448 rows; **L2 512 → 6,656, 13x**). L2 hunt =
**6,144 pours, 0 wins** — L2 is a far harder needle and is still only 4 boards.
An L2-CAPABLE dynamics is training (merged_v6, `--holdout-mode l1wins`, because every
L2 row shares an episode with an L1 winner); the deployed model has ~zero L2 physics.

**8. Ops bugs fixed (both bit):** `pour_idx` is int8 and `np.int8(pours-1)` OVERFLOWS
at pour 128 — v8's 128-pool stopped at 127 by luck; a 384-deep L2 pool killed 4 runs.
Now saturates. And the fleet Pool uses **spawn**, so worker cmdlines are
`spawn_main(...)` and do NOT match "branch_fleet.py": killing by script name leaves
workers at 99% CPU forever. Kill by ppid — and never let a pkill pattern match your
own shell (exit 144).

**9. Pre-training does NOT help SP80 either — the gift was the ARCHITECTURE, not the
weights.** `finetune_sp80.py`, identical arch/data/schedule/holdout, only the init
differs: random-init reaches **0.9901 changed-pixel after ONE epoch**; the
wide-pretrained init starts *lower* (0.9777) because it must unlearn 25 games of other
dynamics. Consistent with the zero-shot negative. Do not sell the wide weights as an
SP80 improvement. (The one fair test left is FEW-SHOT on a genuinely new game —
`finetune_new_game.py`, 20 episodes of a game the wide ckpt HELD OUT; it is queued.)

**10. Linear-legal L1 solve (work-order §7 item 5) — ANSWERED, negative, with the
arithmetic.** It is PROPOSER-blocked, not ranker-blocked. Above L0 RESET is a full-game
restart, so a linear agent must arrange IN PLACE within ~45 actions / 4 pours. Winning
tapes are 14-33 actions ⇒ only **2-3 arrangements** are realisable linearly. The hunt
needs a **128-arrangement pool** to find a winner on 2.5% of boards. Even a PERFECT
ranker cannot rank an arrangement that was never realised: P ≈ 2.5% × 3/128 ≈ **0.06%**.
The next lever is a better PROPOSER, or an arrangement model that scores a tape WITHOUT
executing it (so candidates can be ranked before they are paid for in actions).

---

## IN FLIGHT AT SHUTDOWN (2026-07-13 ~15:15) — all `nohup`'d, they survive the session

**These keep running on the box.** If you want it quiet instead, see "to stop" below.

CPU chain (each blocks on the previous **completing**):
1. `runs/g2_wide_l1_{ranked,random}_20260713/` — paired L1 arms, pool 128 / budget 8,
   identical pools. **Ranked has already recovered 4/4 known winners with 8 pours each
   (v8 needed 128).** The `random` arm is the control and had not started at shutdown.
2. `runs/harvest_l1_wide_20260713/` (`runs/harvest_queue.sh`) — 600 FRESH seeds
   (91000+), top-8 budget. This is the point: spend the 16x budget cut on 16x more
   BOARDS. Every extra L1 winner is a board that reaches L2.
3. `runs/l2_attack_*_20260713/` (`runs/l2_attack_queue.sh`) — blocks on the harvest's
   `summary.json`, then hunts L2 on every L2-reachable board (harvest winners ∪ the
   original 4) with 512-deep pools and the L2-capable dynamics.

GPU chain: `sp80_from_wide` (pretrain-vs-scratch) → `newgame_queue.sh` (few-shot) →
`l2_dynamics_queue.sh` (merged_v6 L2-capable dynamics, `--holdout-mode l1wins`).

**To collect next session (in this order):**
```bash
cd synthesis_lab/experiments/learned_physics
# 1. the G2 paired result (the env-side proof of the ranking gain)
for M in ranked random; do python3 -c "import json;s=json.load(open('runs/g2_wide_l1_${M}_20260713/summary.json'));print('$M', json.dumps(s['per_level']))"; done
# 2. did the harvest produce many more L1 winners (=> more L2 boards)?
python3 -c "import json;s=json.load(open('runs/harvest_l1_wide_20260713/summary.json'));print(json.dumps(s['per_level']), s['winning_tapes_by_level'])"
# 3. FIRST L2 WIN?  (grep the attack summaries)
python3 -c "
import json,glob
for f in sorted(glob.glob('runs/l2_attack_s*_20260713/summary.json')):
    print(f, json.load(open(f))['per_level'])"
# 4. re-score the chain on any NEW L1 winners, holding their episodes out (MANDATORY)
python3 -B eval_l1_winner_rank.py --data <new shard>.pt \
  --pairs wide=../wide_world_model/runs/sp80_from_scratch_20260713/best.pt:runs/merged_v4_branch_l1_20260713/win_probe_d4/best.pt \
  --level 1 --out runs/<new>.json
# 5. verify any new win in a PLAIN LINEAR fresh run before it counts
python3 -B verify_linear.py --tapes runs/<run>/winning_tapes.json --out runs/<run>/linear_verification.json
```
**To stop everything instead:** kill the queue shells, then the fleet workers **by
ppid** — their cmdlines are `spawn_main(...)`, they do NOT match "branch_fleet.py", and
a `pkill` pattern that matches your own shell kills the shell (exit 144):
```bash
for p in harvest_queue l2_attack_queue newgame_queue l2_dynamics_queue g2_wide_queue; do
  ps -eo pid,args | awk -v p="$p" '$0 ~ p && !/awk/ {print $1}' | xargs -r kill
done
ps -eo ppid,pid,args | awk '/spawn_main/ {print $2}' | xargs -r kill -9
```

**NEXT LEVERS, in priority order:**
1. **L2 is the whole game now.** 0 wins in 6,144 pours across the only 4 boards that
   reach it. Ranking cannot fix that — it only cuts the budget to find a winner ALREADY
   IN THE POOL. L2 needs (a) more boards (the harvest) and (b) a better PROPOSER /
   deeper pools. Check whether a winning L2 arrangement exists at all before tuning the
   ranker further.
2. **A proposer / arrangement model** — the same missing piece blocks the linear
   headline (item 10). Predicting the arranged board from a tape without executing it
   would let us rank candidates before paying actions for them.
3. Re-train the win probe on the wide architecture too (it is still the small 2-level
   `WinProbe`; it is already perfect at L1 on real boards, but L2 is untested).

## 0-PREV (2026-07-13 ~11:00): L1 WALL BROKEN — 4 L1 WINS, ALL 4 LINEARLY VERIFIED (levels_completed=2); SAMPLER v8 + D4 PROBE WERE THE LEVERS

**Read first:** `synthesis_lab/notes/LP_TRAINING_SESSION_20260713.md` (full
session report incl. next-session work order §7). honesty_audit exit 0 at
session start and end (97/46/6, active 85/23/0, unchanged).

Headline events (all in `synthesis_lab/experiments/learned_physics/`):
- **First-ever L1 wins: 4/19,751 branch pours** (seeds 90005/90006/90023/90139,
  run `runs/branch_fleet_l1hunt_v8_20260713`, 20,421 env-graded pours total,
  512 first-ever L2 pours). **verify_linear.py: 4/4 chains replay in ONE plain
  linear fresh run each — env grades levels_completed=2.** The learned lane's
  linear-replay level count is now 2 (was 1). Program-wide L1 was 0/7,503
  before today.
- **Root cause of the old L1 wall found by env diagnostics** (diag_l1_moves.py,
  diag_l1_sampler.py): (a) `component_displacement` matches the component's
  ORIGINAL color, but selection RECOLORS responding movables → the v7 sampler
  retained almost nothing at L1, so the 07-09 hunt poured near-empty
  arrangements; (b) all attempts shared one probe branch, whose futile actions
  burned the ~45-action level budget (deaths were budget, not lethal moves).
  **Sampler v8** (branch_fleet.py, condition stage3_branch_fleet_v8): fresh
  branch per segment attempt + `_arrangement_changed` retention gate (persistent
  pixel change) + `--move-persistence` correlated drags (-1 = mixed pool) +
  pre-death prefix keeping. Full pools now realize (128/128 vs ~41/128).
- **D4 (rotation/flip) win probe** `runs/merged_v4_branch_l1_20260713/win_probe_d4`:
  new `probe_rotation_diagnostic.py` shows the plain probe's win sense is
  orientation-locked (rot180 AUC 0.579 = the L1-blindness mechanism); the D4
  probe passes at AUC 1.0 under all six transforms. It ranked seed-90023's
  winning L1 arrangement as the HIGHEST imagined score of the entire run
  (0.161; run p99 0.002). Disclosed generic symmetry prior (--augment-d4).
- **Chain eval is the deploy criterion** (`eval_imagination_auc.py`): probes
  score IMAGINED boards, and per-model metrics mislead — the changed-pixel-
  selected dynamics (battery L0 changed 0.9724!) collapses the chain to ~1e-4.
  **Deploy pair for round 2: merged_v4 dynamics/best.pt + win_probe_d4_imix
  (--imagined-mix trains the probe on real+imagined boards): chain L0
  imag-AUC 1.0 AND rotation diagnostic 1.0.**
- **merged_v4 dataset** (15,851 rows, L1 1,214→8,329) unblocked via explicit
  pour_idx schema migration in merge_transitions.py (--allow-branched-pour-idx,
  manifest-gated to branched_snapshot_search sources). hc/bfs smoke shards
  excluded (manifests can't prove t0-five exclusion).
- Ops: fleet workers MUST run with OMP/MKL/OPENBLAS_NUM_THREADS=1 (load 218
  without); a pkill whose pattern matches its own shell dies at exit 144 and
  leaves orphaned mp workers — kill worker PIDs explicitly.

NEXT (work order in LP_TRAINING_SESSION_20260713.md §7): merged_v5 with the
l1hunt_v8 shard (first L1 POSITIVES + first L2 rows) → retrain probe/dynamics →
round-2 hunt fast-pathing to L2 via the verified tapes → G2-style paired arms
at L1 → attempt linear-legal L1 solve (would move the honest agent-alone
headline from 1 toward 2+ with no branching caveat).

Working tree remains uncommitted; staging intentionally untouched.

## 0-PREV (2026-07-09 ~23:50): STAGE-3 LOOP CLOSED -- G2 PASSED (RANKED 9x RANDOM AT L0), 3 MECHANICS DISCOVERED, L1 WIN WALL IS THE NEXT LEVER

**Read first:** `synthesis_lab/notes/NETWORK_VIABILITY_REPORT_20260709.md` (this
session's full report, honesty_audit-led), then ONE_LANE_20260709.md, then
API_CONTRACT.md. Coding model was Codex 5.6 Terra per plan (codex CLI session
`019f486a-880b-7a80-8f96-f8974fcffb84`, resumable); Claude orchestrated.

Pre-registered gate results (report has full detail):
- G1 PASS: solve loop crash-free over 200+200+400 fresh episodes, full ledgers.
- G2 PASS (round 2): imagination-ranked 87/546 L0 pour wins (15.9%) vs random
  13/778 (1.7%), identical seeds/candidates/pour budget. Round 1 FAILED (2.1% vs
  2.6%) — root cause was a render-convention OOD bug, not the models.
- G3: agent-alone fresh-run level count = 1 (168/400 episodes win L0; 0/1,020 L1
  pour wins). Headline "2" NOT beaten; L1 win = the single next lever.

Mechanics discovered by acting (all legal, all encoded in solve_loop v6):
1. Pre-pour frames render 9-movables/pip-strip differently than pour frames
   (9->8, 14->0) — models must be fed via the ONLINE-LEARNED render map (no
   constants; solve_loop learns it from its own pours per level).
2. RESET = FULL-GAME restart everywhere (never level restart). RESET-replay
   candidate pools are valid ONLY at L0. Never reset past L0.
3. Per-level ACTION BUDGET ~45 actions (any type; clicks not lethal; RESET
   refreshes it but see 2). L1+ walk is budget-aware: 2 visits/pour, undo off.

Artifacts: solve_loop.py v6 + eval_per_level.py + merge_transitions.py (Codex,
reviewed, banned-scan clean); diary now 7,452 pours / 2,140 episodes / 404 wins,
L1 coverage 1,214 rows (12x). Deployed ckpts `runs/merged_v2/*` (L0 held-out:
probe AUC 1.0, dyn changed 0.64; L1 held-out: dyn changed 0.61, probe untestable
— zero L1 positives exist). Round-3 retrain on the full diary COMPLETED
in-session (`runs/merged_v3/`): dynamics changed-pixel 0.7724 (from 0.6030 this
morning); held-out L1 changed-pixel 0.612 -> 0.9105 (L1 physics ~ as sharp as
L0); probe winners 0.9995 vs losers MAX 0.0012 at L0. **Deploy merged_v3 ckpts
for the L1-win hunt** (pass --dynamics-ckpt/--win-probe-ckpt to solve_loop).

NEXT (in order): (1) find ONE L1 win inside the ~45-action budget — richer L1
candidate tapes (more moves/multi-component), imagination hill-climb, or brute
episodes (~0.5 s each; 400 eps ≈ 3 min); (2) the moment an L1 win exists,
retrain the probe (positives activate L1 ranking) and re-run G2-style paired
arms at L1; (3) re-eval per level with merged_v3 ckpts; the changed-pixel L1
number should move first. Standing rules unchanged; honesty_audit exit 0
verified at session end (97/46/6, active 85/23/0).

Working tree remains uncommitted; staging intentionally untouched.

## 0-PREV (2026-07-09 ~19:30): LEARNED-PHYSICS FOUNDATION BUILT -- ONE LANE IS PLAN OF RECORD; NEXT SESSION = STAGE 3, CODING MODEL = CODEX 5.6 TERRA

**Read first, in order:** `synthesis_lab/notes/ONE_LANE_20260709.md` (the whole
plan, one page), then `synthesis_lab/experiments/learned_physics/API_CONTRACT.md`
(authoritative env interface — do NOT re-derive it), then this entry.

Lane verdicts this cycle:
- `legal_search/` blend LoRA = **PARKED** with verdict: objective carried no
  task signal (change_acc 0.999 vs base 0.874, synth level rate 0.975, **zero**
  real Arcade levels — SP80 thrash GAME_OVER ~7 steps, ft09 no-ops). See
  `legal_search/runs/overnight_level/RESULT.md`. Do not resume as framed.
- Hand-coded sim3d = demoted to **offline baseline only**; nothing it computes
  feeds the agent. Specialist solvers = separate labeled lane, never the agent.
- New plan of record: **learned physics from own experience** (ONE_LANE doc).
  Permissibility test for every idea: knowledge from env responses to our own
  actions = legal; human-coded routes/rules/physics/source facts = not this lane.

Foundation state (all in `synthesis_lab/experiments/learned_physics/`; built by
4 Opus agents + main loop 2026-07-09, integration-verified on real env data,
honesty_audit exit 0 `97/46/6 active 85/23/0`, zero banned imports):
- Stage 0 collector: `collect_transitions.py`, `render_contact_sheet.py`.
- Stage 1/2 trainers: `lp_data.py`, `train_dynamics.py` (UNet-lite 1.06M,
  changed-pixel-weighted CE), `train_win_probe.py` (114k, pos_weight),
  `render_imagination.py` (before | imagined | real, val-only).
- Dataset `runs/stage0_v1/transitions_v1.pt`: **2,056 env-graded pours**, 900
  fresh-run episodes, 82 env wins (3.99%), levels {0:1955, 1:101}, episode-keyed
  810/90 split, t0-five (level-1 seed-1234) excluded. Collection ~40 s — data
  is CHEAP now; never farm set_level, fresh runs only.
- Honest numbers (real data, metrics.json verbatim): dynamics val overall
  **0.9639** (bar 0.90 met), **changed-pixel 0.6030 = THE frontier number**;
  win probe val AUC **1.00** / AP 1.00 (13 val positives, L0-dominated — expect
  this to degrade at higher levels; that is signal, not crisis).
- Commands for all four scripts: ONE_LANE STATUS section (miniconda python, GPU).

NEXT (the only work order): **Stage 3 solve loop** — propose arrangement tapes
(random + CEM over the (32,3) tape vocabulary in the contract) → imagine each
with `runs/stage0_v1/dynamics/best.pt` → rank with win probe → pour top-K for
real → env grades → append to diary → retrain both models. The loop's wins fix
the two known data gaps (level coverage past L1, sharper dynamics).

Pre-registered Stage-3 gates (report against these, win or lose):
- G1: loop runs N fresh episodes end-to-end, crash-free, full pour ledger written.
- G2: imagination-ranked top-K beats random-K on env win rate at L0, same real-pour budget.
- G3: fresh-run agent-alone level count — >2 = new honest headline; 2 = parity. Either is a valid report.

Division of labor next session: **Codex 5.6 Terra is the coding model** — it
implements `solve_loop.py` + retrain plumbing against API_CONTRACT.md; the
Claude session orchestrates, reviews diffs, runs gates, and enforces honesty
rules (Codex gets ONE_LANE + API_CONTRACT + this entry as its brief; hydra is
the comms surface if fleet-run). Standing rules unchanged: labels from env
only; no imports from `_invalid/`, `sp80_spill_*`, `sp80_player_harness`,
`spill_planner`, dropletsim, sim3d in the agent path; never read
`environment_files/sp80/589a99af/sp80.py` into agent code; pre-register gates
before runs; `honesty_audit.py` leads every report.

Working tree remains uncommitted; staging intentionally untouched.

## 0-PREV (2026-07-07 ~08:38): R11L TRACE PROBE LANDED -- LEVEL-0 TRANSITION REPRODUCED, LEVEL-1 COMPONENT PROBES NO 2/6

Current r11l local/offline state:
- New tool:
  `synthesis_lab/experiments/other_games/r11l_trace_probe.py`, schema
  `synthesis_lab.other_games.r11l_trace_probe.v2`.
- Condition: replay the recorded legal Tufa generic-agent ledger from
  `codexlab/runs/tufa_seed_sweep_cn04_r11l_600_20260707T111000Z/children/tufa_seed_sweep_cn04_r11l_600_20260707T111000Z_r11l-495a7899_seed_101/result.json`
  through the local OFFLINE `r11l-495a7899` env, then click current-frame
  component centroids on the reproduced level-1 frame.
- This is a mechanic-induction/replay diagnostic. It is not a live scorecard,
  not a specialized solver, and not route-generalization evidence.

Command:
`python3 synthesis_lab/experiments/other_games/r11l_trace_probe.py --ledger-result codexlab/runs/tufa_seed_sweep_cn04_r11l_600_20260707T111000Z/children/tufa_seed_sweep_cn04_r11l_600_20260707T111000Z_r11l-495a7899_seed_101/result.json --game r11l-495a7899 --seed 7 --env-dir synthesis_lab/experiments/other_games/environment_files --max-actions 120 --window 10 --component-probe --probe-colors 3,6,12,15 --probe-max-area 40 --max-probes 24 --out-dir synthesis_lab/experiments/other_games/runs/r11l_trace_probe_seed7_tufa101_components_20260707T1338Z`

Result:
- Replay reproduced the first r11l level transition exactly: `actions=42`,
  `transition_step=41`, click `(x=37,y=6)`, `levels_before=0`,
  `levels_after=1`, `state_after=GameState.NOT_FINISHED`.
- No ledger level-counter mismatches.
- Transition hashes:
  before `6e38c960ee9fe2f16b6594bdc5dd7864f01a8535bbe68fe4c90f56f89d3962ba`,
  after `1d2208ff15e1fb7d1018928f59029aff3467a147906c098331dd4360432667d7`.
- Level-1 component probe clicked 24 current-frame component centroids from
  colors `{3,6,12,15}`. None advanced to level 2. The active one-click
  responses were small local deltas: green plus/object-overlap clicks changed
  25 px, the color-6/color-12 object at `(24,12)` changed 108 px, most ring
  pixels changed 1 px. All rows stayed `levels_after=1`.

Review/artifacts:
- Run dir:
  `synthesis_lab/experiments/other_games/runs/r11l_trace_probe_seed7_tufa101_components_20260707T1338Z/`.
- Local HTTP server on `127.0.0.1:18787` returned `200 OK` for
  `index.html`, `result.json`, and `report_manifest.json`.
- Playwright screenshot was inspected and stored as `playwright_screenshot.png`.
- Temporary HTTP server was stopped.
- Hashes:
  `result.json`
  `2652e7c53392c08b902fb7f5e38c32a55162482d4abebb642e7fbab167dddafe`;
  `index.html`
  `22b4ad10f9f9bbed8383578422260ad522aad4dbf68e7281e11dec40a9100852`;
  `contact_sheet.png`
  `2b902e51d3b1abda81a7307fc05ff0c16aab5b12874828a41c511b9fa82000db`;
  `component_probe_candidates.png`
  `2f0fedea39eecda1da36747301eb3d3d0a458b1fd8f4fe9139049b2fc788e5e8`;
  `report_manifest.json`
  `6181b8ed499187b80ef56af8ab9e51b9a2c0511e9442315764659bf0e833ff6f`;
  `playwright_screenshot.png`
  `959b2585dc9030437df4924ad9afc5a1ec123a33e905f2bbae06c72a5db6b935`.

Boundary/next:
- r11l is still only `1/6` locally via generic-agent replay/probe evidence.
  Do not promote this to a prepared live condition.
- Do not repeat one-click component-centroid probes as the next r11l move.
  The next useful representation is a sequential relation/state-machine probe:
  track how clicks move or connect the colored nodes/lines across multiple
  controlled actions, or derive a candidate rule from current-frame geometry
  and test it fail-closed offline.
- SP80 and ft09 remain the only current real ARC SDK live scorecard closes,
  each 6/6 score 100, recorded below.
- Final gates after this entry: `py_compile` on runner/probe files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
- Working tree remains uncommitted; staging is intentionally untouched.

## 0-PREV (2026-07-07 ~08:24): LS20 ACTION-GRAMMAR JUMP-GRAPH PROBE LANDED -- MOTIF EDGES RECORDED, STILL NO 4/7

Current LS20 local/offline state:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py` now uses
  schema `synthesis_lab.other_games.ls20_corridor_probe.v5`.
- New knobs: `--color8-action-grammar-probe`,
  `--color8-grammar-steps`, and `--color8-grammar-window`.
- New diagnostic: `target_action_grammar_search`, which starts after a
  post-pickup color-8 target transition and explores action motifs over the
  local game-clone jump graph. It records grammar edge count, jump-edge count,
  motif count, states seen, and extension length in both JSON and HTML.

Focused grammar smoke:
- Run dir:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_grammar_smoke_20260707T1312Z/`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders c11_1,c11_2,item01,exit14_0 --post-pickup-color8-probe --color8-action-grammar-probe --graph-engine game_clone --max-ext 220 --max-nodes 5000 --deadline-seconds 60 --color8-probe-steps 100 --color8-grammar-steps 36 --color8-grammar-window 3 --color8-probe-min-area 10 --color8-contact-distance 1 --color8-jump-threshold 12 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_grammar_smoke_20260707T1312Z`
- Result: `color8_action_grammar_deadline`, not level advance.
  `best_levels_completed=3`, `best_route_len=97`,
  `best_final_state=GameState.NOT_FINISHED`.
- Grammar stats: `target_action_grammar_states_seen=150`,
  `target_action_grammar_edge_count=1913`,
  `target_action_grammar_jump_edge_count=212`,
  `target_action_grammar_motif_count=80`, no extension advance.

All-order grammar diagnostic:
- Run dir:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_grammar_orders_20260707T1318Z/`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders all_pickups_exit --post-pickup-color8-probe --color8-action-grammar-probe --graph-engine game_clone --max-ext 220 --max-nodes 2500 --deadline-seconds 35 --color8-probe-steps 100 --color8-grammar-steps 36 --color8-grammar-window 2 --color8-probe-min-area 10 --color8-contact-distance 1 --color8-jump-threshold 12 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_grammar_orders_20260707T1318Z`
- Result: six `color8_action_grammar_deadline` rows, zero rows advanced to
  4/7. `best_levels_completed=3`, `best_route_len=97`, final state
  `GameState.NOT_FINISHED`.
- Orders `0,1,2,4` ended with best extension `22` or `322231`,
  best actor `[11,40]`, and best phase `{"5":12,"8":8}`.
- Orders `3,5` ended with empty extension, best actor `[6,58]`, and best
  phase `{"12":20}`.
- Grammar edge/jump-edge counts by order:
  `1389/301`, `1384/301`, `1378/301`, `1347/233`,
  `1363/296`, `1339/229`.

Artifact/review checks:
- Local HTTP server on `127.0.0.1:18786` returned `200 OK` for grammar
  smoke/all-order `index.html` and `result.json`.
- Playwright screenshot of the all-order review surface was inspected and
  stored as
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_grammar_orders_20260707T1318Z/playwright_screenshot.png`.
- Temporary HTTP server was stopped.
- Hashes:
  smoke `result.json`
  `1fe9dea4597b52bae81e2313a688814363b12d51f6dd37d8d4a1b0246394fd02`;
  smoke `index.html`
  `18f164f7f4497ca7ce9c8e8b0a3e80893de50e8946e3510cb0305c47decde61c`;
  all-order `result.json`
  `9bb5651bec2579da5500171939133585c15308297bd8290145e0116291629a53`;
  all-order `index.html`
  `a4ae0b3fe961ac11aa4fee1ebc341b57ffa8d3d9110cc76da8da65df1e1e290a`;
  all-order `playwright_screenshot.png`
  `e59bc7b5eae0363d548992413c28908956163725b78c38ee90413864ec4768fa`.

Boundary/next:
- This was a representation-changing LS20 local/offline diagnostic. It is not
  a live/ranked scorecard, not a leaderboard result, and not a 4/7 route.
- Do not launch another plain action-grammar window/depth expansion as the
  next LS20 move. The next valid LS20 step needs a new objective/acceptance
  model, cross-level lock-action generalization, or a clearly labeled
  source-derived mechanic parse if explicitly allowed.
- SP80 and ft09 remain real ARC SDK live scorecard closes, each 6/6 score 100,
  recorded in the block below.
- Final gates after this entry: `py_compile` on runner/probe files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
- Working tree remains uncommitted; staging is intentionally untouched.

## 0-PREV (2026-07-07 ~08:03): SECOND LIVE SCORECARD CLOSED -- SP80 + ft09 EACH 6/6, SCORE 100

Current live state:
- SP80 live ARC SDK scorecard:
  `8820f639-fe4d-438f-9129-83abe0646843`.
  Run dir:
  `codexlab/runs/arc_live_sp80_megarexus_dw_directgrid_20260707T125653Z/`.
  Result: `sp80-589a99af` completed `6/6`, score `100.0`,
  `actions=157`, `resets=0`, run state `WIN`, `policy_error_count=0`.
- ft09 live ARC SDK scorecard:
  `2c33f8bb-9da5-44ee-b0c4-96f4d9c611c0`.
  Run dir:
  `codexlab/runs/arc_live_ft09_glyph_20260707T130252Z/`.
  Result: `ft09-0d8bbf25` completed `6/6`, score `100.0`,
  `actions=86`, `resets=0`, run state `WIN`, `policy_error_count=0`.

ft09 live command:
`python3 codexlab/arc_live_suite_runner.py --allow-live --policy ft09_glyph --games ft09-0d8bbf25 --max-actions 320 --no-perception --run-id arc_live_ft09_glyph_20260707T130252Z --out-dir codexlab/runs/arc_live_ft09_glyph_20260707T130252Z`

ft09 pre-live gate:
- Run dir:
  `codexlab/runs/arc_suite_ft09_glyph_current_20260707T130217Z/`.
- Command:
  `python3 codexlab/arc_suite_runner.py --policy ft09_glyph --games ft09-0d8bbf25 --env-dir synthesis_lab/experiments/other_games/environment_files --seed 7 --max-actions 320 --no-perception --run-id arc_suite_ft09_glyph_current_20260707T130217Z --out-dir codexlab/runs/arc_suite_ft09_glyph_current_20260707T130217Z`
- Result: current-code local/offline `6/6`, score `100.0`,
  `actions=86`, `resets=0`, `final_state=GameState.WIN`,
  `policy_error_count=0`; solver schema
  `synthesis_lab.other_games.ft09_glyph_solver.v5`.

ft09 SDK close payload:
- `score=100.0`, `total_levels_completed=6`, `total_actions=86`,
  `total_environments_completed=1/1`, `source_url=null`.
- Level actions `[4,7,14,27,21,13]` versus baselines
  `[43,12,23,28,65,37]`.

ft09 review/artifact checks:
- Local HTTP server returned `200 OK` for offline and live `index.html` plus
  `result.json`.
- Playwright screenshots were inspected and stored in each run dir.
- Live hashes:
  `result.json` `f908770ddf81803a8e700fd275a5aec4493b13f130c048659401f1ce6c091297`;
  `index.html` `7ef4510f4237af3e09fe46fb6466b005c95d81f3eea2e50be59db2f114bcda22`;
  `report_manifest.json` `0063c9f841a6c9ed0d1792874a1af6240ea1b9bfc23f8e768dded9dac9f65b28`;
  `report_trace.jsonl` `df294b9bc438a43ae78c123d1883d39013187c089534cdf02c3b77c77009c98c`;
  `review_screenshot.png` `380f0885dd5ede929ec1e62deda95d125e8fc4f7d51846661a3c45ec397155ab`.
- Offline gate hashes:
  `result.json` `536b210979f5702093c3d8d98601b7b1250bd78238f13b2d900229d90989b319`;
  `index.html` `a5977ef2211b800e400e4fc400e791c7891ab481c2770bd87e66df98cdee4912`;
  `report_manifest.json` `f5c0aa51f82730d7d8b2ce8c4bb06a46d297b2e90ade18be5ab961980d6b8064`;
  `report_trace.jsonl` `5645cebc6ab186ed8226ea0f44d156eae6ba556fd92101ad873d280e42e9dc96`;
  `review_screenshot.png` `da903bcdd4cce613c471511f2792b860fffeb8ac9b5c7441c7be65e4f0f9b778`.

Boundary/next:
- These are real ARC SDK live scorecard closes for two individual games, not a
  combined leaderboard submission and not a broad ARCPrize3 solved claim.
- The SDK returned `source_url=null` for both current live artifacts; public
  URL verification is therefore not available from the SDK payloads.
- Do not spend another generic live run. Next valid progress is either a
  prepared single-game live condition with a current-code offline 6/6 gate, or
  a representation-changing local lane for an unsolved game. Current LS20 is in
  a same-basin local 3/7 loop unless the representation changes beyond
  phase/underlay/budget/jump counts.
- Final gates after this entry: `py_compile` on runner/policy files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
- Working tree remains uncommitted; staging is still intentionally untouched.

## 0-PREV (2026-07-07 ~07:58): SP80 DW LIVE SCORECARD CLOSED -- 6/6, SCORE 100

Current result:
- Live ARC SDK scorecard:
  `8820f639-fe4d-438f-9129-83abe0646843`.
- Run dir:
  `codexlab/runs/arc_live_sp80_megarexus_dw_directgrid_20260707T125653Z/`
  with `result.json`, `index.html`, `report_manifest.json`,
  `report_trace.jsonl`, and `review_screenshot.png`.
- Command:
  `python3 codexlab/arc_live_suite_runner.py --allow-live --policy megarexus_dw_directgrid --games sp80-589a99af --max-actions 220 --run-id arc_live_sp80_megarexus_dw_directgrid_20260707T125653Z --out-dir codexlab/runs/arc_live_sp80_megarexus_dw_directgrid_20260707T125653Z`
- Result: exit `0`, `evidence_category=live_scorecard`,
  `arc_operation_mode=OperationMode.NORMAL`, `aggregate_score=100.0`,
  `total_levels_completed=6`, `policy_error_count=0`.
- SDK close payload: `score=100.0`, `total_levels_completed=6`,
  `total_actions=157`, `total_environments_completed=1/1`; game row
  `sp80-589a99af` completed `6/6`, `resets=0`, `score=100.0`,
  run state `WIN`.
- Per-level action counts: `[12,8,13,34,48,42]` versus baselines
  `[39,58,25,148,96,152]`.

Pre-live gate:
- Current dirty-tree offline gate:
  `codexlab/runs/arc_suite_megarexus_dw_directgrid_current_20260707T1308Z/`.
- Command:
  `python3 codexlab/arc_suite_runner.py --policy megarexus_dw_directgrid --games sp80-589a99af --max-actions 220 --run-id arc_suite_megarexus_dw_directgrid_current_20260707T1308Z --out-dir codexlab/runs/arc_suite_megarexus_dw_directgrid_current_20260707T1308Z`
- Result: exit `0`, local/offline `aggregate_score=100.0`,
  `total_levels_completed=6`, `actions=157`, `resets=0`,
  `final_state=GameState.WIN`, `policy_error_count=0`.

Review/artifact checks:
- Local HTTP server returned `200 OK` for offline and live `index.html` plus
  `result.json`.
- Playwright screenshots were inspected and stored in each run dir.
- Live artifact hashes:
  `result.json` `3d0869afd7781b8efc7232474010e7e9a6d32cbd7bc532a233ad2cf853acbf7e`;
  `index.html` `2215bc50c180de8f8b67a4285bc226d26763890cfd4c2e08cc6a9c61d0508efb`;
  `report_manifest.json` `109415c515e36360b469074bca0be5e92cdb4fba668e7f33c8504911633a69b4`;
  `report_trace.jsonl` `ab887dbca510500c102dc2a5f389349a35294a331bff1a3edd0fe11abfe428d1`;
  `review_screenshot.png` `7e5ac745168324fb90389e6d5cf6531b164959bc4e396637804f1b7093ec15f0`.
- Offline pre-live hashes:
  `result.json` `bb32edbac52b01b8c2ed1f40ca6cbec312fd6372e11c6a9bbe6a71b73feacb76`;
  `index.html` `14495a6798e5da9972a836d05b0b4ba53c1548411ea616e8b24ecfcde0876c30`;
  `report_manifest.json` `e9096a352c7134764b8d380f7bcd5d66255d2d60309c3e8368240ff045a4926d`;
  `report_trace.jsonl` `ab09c6d832d2d1985d8fa791665c027d8a9f1518d16f9d4623452df19dd16308`;
  `review_screenshot.png` `5995e8d72266b1de708f1224a5e786aa4c8c503d21d8981bb5da028fa11e6ec1`.

Boundary/next:
- This is a real live ARC SDK scorecard close for SP80 only. It is not a
  broad ARCPrize3 leaderboard or multi-game claim.
- The SDK returned `source_url=null`; public URL verification is therefore not
  available from this artifact.
- Next valid move is a controlled second live scorecard only if targeting a
  prepared policy/game condition; do not spend generic live runs.
- Final gates after this entry: `py_compile` on runner/policy files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
- Working tree remains uncommitted; staging is still intentionally untouched.

## 0-PREV (2026-07-07 ~07:47): LS20 TARGET-PHASE/UNDERLAY PROBE LANDED -- JUMP/BUDGET STATE EXPOSED, NO 4/7 CLAIM

Current integration after target-state continuation stale-loop:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py` schema is now
  `synthesis_lab.other_games.ls20_corridor_probe.v4`.
- New target-phase continuation mode:
  - `--color8-phase-probe` starts after a color-8 target transition event and
    continues legal local-engine search with actor-masked target underlay,
    actor-overlap count, budget-strip counts, and jump-like actor movement in
    the state/diagnostic surface.
  - `--color8-phase-steps` bounds the phase continuation depth.
  - `--color8-jump-threshold` controls the actor-position Manhattan delta
    counted as jump-like; current runs used threshold `12`.
  - `target_phase_continue_search` records phase histograms, phase transition
    events, jump events, and budget-strip events with compact frame deltas.
- This is a representation change relative to raw target-signature
  continuation: target cells covered by the actor become underlay-masked state,
  and volatile bottom-strip budget state is explicit rather than hidden inside
  full object signatures.

Focused phase smoke:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_phase_smoke_20260707T1232Z/index.html`
  with raw `result.json`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders c11_1,c11_2,item01,exit14_0 --post-pickup-color8-probe --color8-phase-probe --graph-engine game_clone --max-ext 220 --max-nodes 100000 --deadline-seconds 60 --color8-probe-steps 100 --color8-phase-steps 80 --color8-probe-min-area 10 --color8-contact-distance 1 --color8-jump-threshold 12 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_phase_smoke_20260707T1232Z`
- Result: `color8_phase_deadline`, not level advance.
  `best_levels_completed=3`, `best_route_len=97`.
- The focused `{12:20}` contact/order produced `target_phase_unique_phase_count=120`,
  `target_phase_transition_event_count=3971`, `target_phase_jump_event_count=569`,
  and `target_phase_budget_event_count=3971`.
- First high-signal jump event: after starting on the target (`visible={12:20}`,
  `underlay={}`, `overlap=20`), action `2` moved actor `[6,58] -> [36,30]`
  with Manhattan `58`, exposing target underlay/visible `{9:20}` and reducing
  energy11 `34 -> 32`.

All-order phase diagnostic:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_phase_orders_20260707T1238Z/index.html`
  with raw `result.json` and `playwright_screenshot.png`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders all_pickups_exit --post-pickup-color8-probe --color8-phase-probe --graph-engine game_clone --max-ext 220 --max-nodes 100000 --deadline-seconds 60 --color8-probe-steps 100 --color8-phase-steps 80 --color8-probe-min-area 10 --color8-contact-distance 1 --color8-jump-threshold 12 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_phase_orders_20260707T1238Z`
- Result: all six pickup permutations ended `color8_phase_deadline`; zero
  advanced to 4/7. `best_levels_completed=3`, `best_route_len=97`,
  `best_final_state=GameState.NOT_FINISHED`.
- Orders `0,2` (`{5:12,8:8}` transition): phase deadlines with
  `states_seen=1228/1226`, unique phases `90/90`, jump events `338/338`.
- Orders `1,4` (`{5:12,8:8}` transition): phase deadlines with
  `states_seen=1298/1295`, unique phases `111/111`, jump events `331/331`.
- Orders `3,5` (`{12:20}` transition/contact): phase deadlines with
  `states_seen=1346/1337`, unique phases `120/120`, jump events `569/569`.
- The `{12:20}` rows make the actor/underlay split explicit: overlap phase
  `visible={12:20}`, `underlay={}`, `overlap=20` transitions to exposed
  `visible={9:20}`, `underlay={9:20}`, `overlap=0` after a jump-like action.
  The `{5:12,8:8}` rows mostly cycle through visible/underlay mixtures such as
  `{5:12,8:8}` and `{5:8,8:12}` with budget levels split by energy11.

Artifact hashes:
- Phase smoke `result.json`
  `0432581547b6c44174ba3baed97369d8e79a031f2158daedeb4ab679ce3a06dc`
- Phase smoke `index.html`
  `e0c8f19f89b9cc2977f3424176f9be7ba6b0e53a10fcd8540ae7561717219002`
- Phase all-order `result.json`
  `06346d4b7b6051f510a3fb066e9d03d8fb145f05af5481bb5e39aef6ed53a2cd`
- Phase all-order `index.html`
  `d8fa3c3a08aaddedb549bb971076f2a80c6783894c7f1c47b85e5aa2357959ee`
- Phase all-order `playwright_screenshot.png`
  `af5a7b5d08fa9d3648d955115660fe0734017e3fe743ce14bec011ac07419882`

Review check:
- Served `synthesis_lab/experiments/other_games/runs` on
  `127.0.0.1:18783`.
- HTTP `200` returned for phase smoke/all-order `index.html` and `result.json`.
- Playwright screenshot succeeded for the all-order `index.html`; local image
  inspection showed a nonblank readable page with the phase columns populated.
- Temporary server stopped.

Boundary:
- This is controlled local/offline evidence only. It is not live/ranked proof,
  not a public leaderboard result, and not a 4/7 route.
- Do not rerun longer generic `actor_astar`, waypoint-order, old color-14
  exit, one-step color8 poke, plain target-state continuation, or plain
  target-phase continuation for ls20 L3 as the next move. The phase/underlay
  representation now has smoke plus all-order same-basin deadlines.
- Next valid LS20 move must expose a new mechanic degree of freedom beyond
  phase/underlay/budget/jump counts, e.g. action-sequence grammar over the
  jump graph, source-derived mechanic parse if explicitly allowed and clearly
  labeled, or a cross-level generalization that predicts the lock action rather
  than expanding the same local phase search.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
- Working tree remains uncommitted.

## 0-PREV-2 (2026-07-07 ~07:28): LS20 TARGET-STATE CONTINUATION PROBE LANDED -- SIGNATURE CYCLES FOUND, NO 4/7 CLAIM

Current integration after post-pickup color-8 target changes:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py` schema is now
  `synthesis_lab.other_games.ls20_corridor_probe.v3`.
- New target-state continuation mode:
  - `--color8-transition-continue-probe` starts after a color-8 target
    transition event and continues legal local-engine search with the target
    cell signature in the graph key.
  - `--color8-transition-steps` bounds the continuation depth.
  - `target_state_continue_search` does not stop at the next target signature
    change; it records target-signature histograms and transition events while
    continuing until level advance, deadline, node cap, depth cap, or terminal
    state.
- This is a representation change relative to one-step post-transition pokes:
  the changing target mask is explicit graph state, not a terminal event.

Focused continuation smoke:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_continue_smoke_20260707T121613Z/index.html`
  with raw `result.json`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders c11_1,c11_2,item01,exit14_0 --post-pickup-color8-probe --color8-transition-continue-probe --graph-engine game_clone --max-ext 220 --max-nodes 100000 --deadline-seconds 60 --color8-probe-steps 100 --color8-transition-steps 80 --color8-probe-min-area 10 --color8-contact-distance 1 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_continue_smoke_20260707T121613Z`
- Result: `color8_transition_continue_deadline`, not level advance.
  `best_levels_completed=3`, `best_route_len=97`.
- Continuation row: transition target signature starts `{12:20}`, deadline at
  `levels_completed=3`, `states_seen=1340`, `queue_popped=1119`,
  `expanded_edges=1339`, `target_transition_event_count=332`.
- Signature cycle examples included `{12:20}`, `{14:20}`, `{5:12,12:8}`,
  `{5:8,12:12}`, `{8:20}`, and `{9:20}`.

All-order continuation diagnostic:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_continue_orders_20260707T121758Z/index.html`
  with raw `result.json`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders all_pickups_exit --post-pickup-color8-probe --color8-transition-continue-probe --graph-engine game_clone --max-ext 220 --max-nodes 100000 --deadline-seconds 60 --color8-probe-steps 100 --color8-transition-steps 80 --color8-probe-min-area 10 --color8-contact-distance 1 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_continue_orders_20260707T121758Z`
- Result: all six pickup permutations ended
  `color8_transition_continue_deadline`; zero advanced to 4/7.
  `best_levels_completed=3`, `best_route_len=97`,
  `best_final_state=GameState.NOT_FINISHED`.
- Orders `0,2` (`{5:12,8:8}` transition): continuation deadlines with
  `states_seen=1184/1160`, event counts `35/35`.
- Orders `1,4` (`{5:12,8:8}` transition): continuation deadlines with
  `states_seen=1251/1229`, event counts `31/31`.
- Orders `3,5` (`{12:20}` transition/contact): continuation deadlines with
  `states_seen=1126/1099`, event counts `325/325`.
- Target-state dynamics are real but not sufficient: rows cycle through
  multiple signatures such as `{5:12,12:8}`, `{5:8,12:12}`, `{8:20}`,
  `{9:20}`, and `{14:20}` while staying at level 3.

Artifact hashes:
- Continuation smoke `result.json`
  `ea7c3848844877c5dc083c37c2825f849534b366d54c428b3aa1e8de9b65ab64`
- Continuation smoke `index.html`
  `79a871b0f74298c930b04a1f61e6a1d61a56162b2fc804bd1b3a6934344fa15a`
- Continuation all-order `result.json`
  `969dd094bbf75cc4c885e8570f00d3dbad948bfd59d4fac4c55fd13be36d05a4`
- Continuation all-order `index.html`
  `4633059290fc1d8c1ce6ab3ea1b837f98aa2a9c4623a0d947fc749f6535dc651`

Review check:
- Served `synthesis_lab/experiments/other_games/runs` on
  `127.0.0.1:18782`.
- HTTP `200` returned for continuation smoke and all-order `index.html` and
  `result.json`.
- Playwright screenshot succeeded for the all-order `index.html` and local
  image inspection showed a nonblank readable page.
- Temporary server stopped.

Boundary:
- This is controlled local/offline evidence only. It is not live/ranked proof,
  not a public leaderboard result, and not a 4/7 route.
- Do not rerun longer generic `actor_astar`, waypoint-order, old color-14
  exit, one-step color8 poke, or plain target-state continuation for ls20 L3
  as the next move. The target-state continuation class now has smoke plus
  all-order same-basin deadlines. The next valid ls20 representation needs a
  new degree of freedom: phase/action-budget model over signature cycles,
  explicit teleport/refill mechanic modeling, or a clearly labeled
  source-derived mechanic parse if explicitly allowed.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
- Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~07:10): LS20 POST-PICKUP COLOR-8 TRANSITION PROBE LANDED -- TARGET CHANGES, STILL NO 4/7 CLAIM

Current integration after the color-14-to-color-8 delta:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py` schema is now
  `synthesis_lab.other_games.ls20_corridor_probe.v2`.
- New explicit post-pickup color-8 mechanic mode:
  - `--post-pickup-color8-probe` stops each waypoint order at final-pickup
    progress, finds newly gained color-8 connected components, and targets
    those cells rather than the stale `exit14_0` color-14 goal.
  - `--color8-probe-steps`, `--color8-probe-min-area`, and
    `--color8-contact-distance` bound the contact/adjacency search.
  - `color8_contact_probe` records target-cell signatures, actor distance,
    transition action, before/after diff, and one-step pokes after contact or
    target change.
- This is a representation change relative to old-exit greedy probing: target
  identity is the observed gained color-8 mask, not the original color-14
  component label.

Color-8 smoke v2:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_smoke_v2_20260707T120456Z/index.html`
  with raw `result.json`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders item01,c11_1,c11_2,exit14_0 --post-pickup-color8-probe --graph-engine game_clone --max-ext 180 --max-nodes 80000 --deadline-seconds 45 --color8-probe-steps 80 --color8-probe-min-area 10 --color8-contact-distance 1 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_smoke_v2_20260707T120456Z`
- Result: `color8_target_changed`, not level advance. The gained target is
  one 20-cell color-8 component at bbox `[3,55,8,60]`.
- Controlled transition: target signature changes from `{8:20}` to
  `{5:12, 8:8}` on action `1`, parent extension `33111`, child extension
  `331111`, actor `[26,36] -> [26,30]`, levels remain `3`.

All-order color-8 diagnostic:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_orders_20260707T120530Z/index.html`
  with raw `result.json`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders all_pickups_exit --post-pickup-color8-probe --graph-engine game_clone --max-ext 220 --max-nodes 80000 --deadline-seconds 45 --color8-probe-steps 100 --color8-probe-min-area 10 --color8-contact-distance 1 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_orders_20260707T120530Z`
- Result: all six pickup permutations reached a `color8_target_changed`
  transition; zero advanced to 4/7. `best_levels_completed=3`,
  `best_route_len=97`, `best_final_state=GameState.NOT_FINISHED`.
- Shared target: all six rows gained the same 20-cell color-8 component at
  bbox `[3,55,8,60]`.
- Orders `0,1,2,4`: target `{8:20} -> {5:12, 8:8}`; no level advance.
- Orders `3,5`: target `{8:20} -> {12:20}` after action `2`, parent extension
  `1`, child extension `12`; actor `[36,26] -> [6,58]`, target distance `0`,
  levels remain `3`.

Focused post-transition contact smoke:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_contact_smoke_v3_20260707T120800Z/index.html`
  with raw `result.json`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders c11_1,c11_2,item01,exit14_0 --post-pickup-color8-probe --graph-engine game_clone --max-ext 220 --max-nodes 80000 --deadline-seconds 45 --color8-probe-steps 100 --color8-probe-min-area 10 --color8-contact-distance 1 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_contact_smoke_v3_20260707T120800Z`
- Result: the `{12:20}` contact state is not an immediate exit. Post-transition
  one-step pokes with actions `1,2,3,4` all stayed at `levels_completed=3`.
  Action `1` changed 51 pixels but did not advance; actions `2,3,4` only spent
  energy / stayed in place.

Artifact hashes:
- Color8 smoke v2 `result.json`
  `4087cd7682f405c9456f44424aebd88de912632b519d8fb6015c06f92e88496e`
- Color8 smoke v2 `index.html`
  `3b975c1d14470eaa13540daa33f822c4db401c89efeacb1e2106e69dcbecfee9`
- Color8 all-order `result.json`
  `21803cdf44a1e377df224a9cd1a13aeb53e7389176f01f9d65776ecc984c9a8d`
- Color8 all-order `index.html`
  `499f33a9d7c5fb2fea46e48f51c2953418e6b1daa2c4597049ed9044dabe14da`
- Color8 contact smoke v3 `result.json`
  `7f462da1f29065cdeaab5fe79a5780445731cf8be82fa3839d54847f7d170648`
- Color8 contact smoke v3 `index.html`
  `9b5893dd3b7154b3227d33888b17aaa74153cfac86311982152030326eb4ebba`

Review check:
- Served `synthesis_lab/experiments/other_games/runs` on
  `127.0.0.1:18781`.
- HTTP `200` returned for color8 smoke v2, all-order, and contact-smoke
  `index.html` and `result.json`.
- Temporary server stopped.

Boundary:
- This is controlled local/offline evidence only. It is not live/ranked proof,
  not a public leaderboard result, and not a 4/7 route.
- Do not go back to generic longer `actor_astar`, waypoint-order, or old
  color-14 exit probing for ls20 L3. The next valid ls20 move should model the
  post-transition target states explicitly (`{5:12,8:8}` and `{12:20}`), e.g.
  a target-state graph key with controlled multi-step continuation from the
  transition event, or a clearly labeled source-derived mechanic parse if
  explicitly allowed.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
- Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~06:56): LS20 L3 DELTA PROBE LANDED -- PICKUPS TRANSFORM COLOR-14 TO COLOR-8, NO 4/7 CLAIM

Current integration after waypoint-order negative:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py` now has a
  waypoint delta-probe mode:
  - `--waypoint-delta-probe` stops each waypoint order once final-pickup
    progress is reached, records the L3-start vs post-pickup frame delta, then
    runs a short controlled greedy probe toward the observed exit-like target.
  - `frame_diff_summary` records changed pixels, bbox, color gain/loss,
    actor before/after, counts, components, and frame hashes.
  - `greedy_exit_probe` records each legal local-engine action, before/after
    deltas, actor distance, and level/state after each step.
- This is a mechanic/delta representation artifact, not another longer
  waypoint search.

Delta smoke:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_delta_smoke_20260707T1205Z/index.html`
  with raw `result.json`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders item01,c11_1,c11_2,exit14_0 --waypoint-delta-probe --graph-engine game_clone --max-ext 180 --max-nodes 80000 --deadline-seconds 45 --exit-probe-steps 24 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_delta_smoke_20260707T1205Z`
- Result: `progress_found`, not level advance. Progress route length `123`,
  progress extension length `26`, progress actor pos `[36,50]`,
  `progress_levels_completed=3`.
- Pickup delta: `changed=78`, bbox `[3,5,58,60]`, colors lost
  `{3:17, 9:15, 11:16, 12:10, 14:20}`, colors gained
  `{3:33, 8:20, 9:15, 12:10}`.
- Exit probe: actions `332222222222222222222222`, 24 actions, stayed at
  `levels_completed=3`, final state `GameState.NOT_FINISHED`, actor pos
  `[26,50]`, distance to old exit target `28.4`.

All-order delta diagnostic:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_delta_orders_20260707T1210Z/index.html`
  with raw `result.json`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders all_pickups_exit --waypoint-delta-probe --graph-engine game_clone --max-ext 220 --max-nodes 80000 --deadline-seconds 45 --exit-probe-steps 24 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_delta_orders_20260707T1210Z`
- Result: six `progress_found` rows, zero `exit_probe_advanced` rows.
  `best_levels_completed=3`, `best_route_len=97`,
  `best_final_state=GameState.NOT_FINISHED`.
- All six pickup permutations reach final-pickup progress, but none of their
  24-step old-exit probes advance to level 4.
- Shared mechanic clue: pickup progress consistently removes the large
  color-14 component and gains a 20-cell color-8 component. The prior
  `exit14_0` target becomes color 8 after pickup progress; treating it as a
  color-14 exit is the wrong next model.

Artifact hashes:
- Delta smoke `result.json`
  `6a4511b59bac7f507828000ef6e993acaf1e0c4f7ac7ca2999705fae134837ed`
- Delta smoke `index.html`
  `5d7f24357373d61bbdbd32eff233f4a234e0bb109edf57eaba3d2077856a56a9`
- Delta all-order `result.json`
  `60a4b2076163fc4365c013836c5502380dd4c58a7c0d92d77e67fa4f84fcbab7`
- Delta all-order `index.html`
  `8984c36a2b1a0cfbf5e1c7ab19692d9017b3f263b782fe08e7bddae94985e6d6`

Review check:
- Served `synthesis_lab/experiments/other_games/runs` on
  `127.0.0.1:18780`.
- HTTP `200` returned for delta smoke `index.html` and `result.json`, and for
  delta all-order `index.html` and `result.json`.
- Temporary server stopped.

Boundary:
- This is controlled local/offline evidence only. It is not live/ranked proof,
  not a public leaderboard result, and not a 4/7 route.
- Do not keep aiming at `exit14_0` as a color-14 target. The next valid ls20
  representation should model the post-pickup color-8/door mechanic or a
  door/lock before-after delta transition explicitly.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
- Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~06:50): LS20 L3 WAYPOINT-ORDER PLANNER LANDED -- ALL PICKUP ORDERS BOUNDED NEGATIVE

Current integration after actor-aware L3:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py` now has a
  waypoint-order planner for ls20 L3:
  - `--waypoint-orders` accepts comma-separated observed goal names, or
    `all_pickups_exit` to expand all visible pickup permutations followed by
    the large color-14 exit-like component.
  - `waypoint_order_search_from_prefix` searches legal local-engine clone
    states with actor-aware `graph_actor_semantic` keys plus an explicit
    waypoint-progress dimension.
  - The planner records `progress_histogram`, `pickup_order_histogram`, visible
    navigation goals, and one row per waypoint order.
- This is a new search-space representation relative to the prior global
  `actor_astar`: it tests ordered pickup/exit hypotheses directly.

Waypoint smoke:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_waypoint_smoke_20260707T1145Z/index.html`
  with raw `result.json`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders item01,c11_1,c11_2,exit14_0 --graph-engine game_clone --max-ext 180 --max-nodes 80000 --deadline-seconds 45 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_waypoint_smoke_20260707T1145Z`
- Result: bounded negative. `best_levels_completed=3`,
  `best_route_len=97`, `best_final_state=GameState.NOT_FINISHED`.
- The row hit `status=deadline`, `elapsed_seconds=45.33`,
  `states_seen=1020`, `queue_popped=827`, `expanded_edges=1019`.
- It reached waypoint progress `3` in `42` popped states, meaning all three
  pickup waypoints were reachable under that order, but no level advance to
  4/7 happened before the cap.

Six-order waypoint sweep:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_waypoint_orders_20260707T1150Z/index.html`
  with raw `result.json`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders all_pickups_exit --graph-engine game_clone --max-ext 220 --max-nodes 80000 --deadline-seconds 45 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_waypoint_orders_20260707T1150Z`
- Result: bounded negative across all six pickup permutations. No
  `status=found` rows. `best_levels_completed=3`, `best_route_len=97`,
  `best_final_state=GameState.NOT_FINISHED`.
- Every row reached waypoint progress `3` in at least some states, but none
  advanced through `exit14_0` to level 4:
  - `item01,c11_1,c11_2,exit14_0`: progress-3 count `42`
  - `item01,c11_2,c11_1,exit14_0`: progress-3 count `72`
  - `c11_1,item01,c11_2,exit14_0`: progress-3 count `46`
  - `c11_1,c11_2,item01,exit14_0`: progress-3 count `41`
  - `c11_2,item01,c11_1,exit14_0`: progress-3 count `30`
  - `c11_2,c11_1,item01,exit14_0`: progress-3 count `29`

Artifact hashes:
- Smoke `result.json`
  `0eac991b9d7791826bb96e8ec9f82876aae7d7df738df23450e304eeb0e6e513`
- Smoke `index.html`
  `0530e620aec1768a7753007ba37601743ef391659283f15bd957c060105ae74c`
- Six-order `result.json`
  `4a8a75144fa88d518259813c1904c6222d79fb99e7c7a8567cad2543dc0b1dad`
- Six-order `index.html`
  `0c949e786f0d23c9a385b7d8fd400d7bb8a101e47f89db2254a4cfe5361101dd`

Review check:
- Served `synthesis_lab/experiments/other_games/runs` on
  `127.0.0.1:18779`.
- HTTP `200` returned for smoke `index.html` and `result.json`, and for
  six-order `index.html` and `result.json`.
- Temporary server stopped.

Boundary:
- This is controlled local/offline evidence only. It is not live/ranked
  proof, not a public leaderboard result, and not a 4/7 route.
- Do not run another longer ls20 L3 waypoint-order search as the next move.
  The waypoint-order class is now same-basin negative after smoke and all-order
  sweep. The next valid ls20 work needs a further mechanic representation:
  explicit door/lock before-after delta modeling, energy/action-budget
  admissibility, or a clearly labeled source-derived mechanic parse if that is
  allowed.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
- Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~06:38): LS20 L3 ACTOR-AWARE SEARCH LANDED -- BOUNDED NEGATIVE, NO 4/7 CLAIM

Current integration after the Tufa seed-map wrap:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py` now has an
  actor-aware representation for ls20:
  - `actor_component`, `actor_pos`, and `actor_bbox` pick the moving color-12
    actor component instead of averaging all color-12 pixels.
  - `update_pickup_order` now uses the actor component, not the old mixed
    `avatar_pos`.
  - New `graph_actor_semantic` state key tracks actor bbox, pickup order,
    object counts, and object-component signatures while excluding the volatile
    bottom energy strip and the moving actor component from the object
    signature.
  - New `--search-strategy actor_astar` prioritizes graph states by legal
    engine action depth plus distance from the actor to visible specials and
    large color-14 exit-like components.
- This is a real search representation/frontier change, not a relabel of the
  previous replay BFS.

Controlled diagnostic baseline:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_gameclone_baseline_20260707T1125Z/index.html`
  with raw `result.json`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --chain-levels 1 --graph-engine game_clone --max-ext 220 --max-nodes 200000 --deadline-seconds 60 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_gameclone_baseline_20260707T1125Z`
- Result: no 4/7 extension. Existing `graph_semantic` + game-clone baseline
  hit deadline at `60.003s`, `states_seen=1333`, `queue_popped=1147`,
  `expanded_edges=1332`, `best_levels_completed=3`.

Corrected actor-key run:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_actor_key_astar_20260707T1138Z/index.html`
  with raw `result.json`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --chain-levels 1 --modes graph_actor_semantic --graph-engine game_clone --search-strategy actor_astar --max-ext 220 --max-nodes 200000 --deadline-seconds 180 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_actor_key_astar_20260707T1138Z`
- Result: bounded negative. `best_levels_completed=3`,
  `best_route_len=97`, `best_final_state=GameState.NOT_FINISHED`,
  `game_clone_consistency.match=true`.
- Search row: `mode=graph_actor_semantic`, `search_strategy=actor_astar`,
  `status=deadline`, `elapsed_seconds=181.382`, `states_seen=3930`,
  `queue_popped=3011`, `expanded_edges=3929`, `max_queue=926`,
  `queue_remaining=919`, `target_level=3`.
- Visible navigation goals in the run: `item01`, `c11_1`, `c11_2`, and
  `exit14_0`.

Artifact hashes:
- Baseline `result.json`
  `841f567f466bfb7c0f38215189cffcea8bb3c5df50b7a1900be1307352fea5f8`
- Baseline `index.html`
  `8a500b093a7b4adad3fc5dc7448428f24ba3cedbead4cfee5624a53750c578cc`
- Actor-key `result.json`
  `8ee4bea46f69a833ae7fb37d1978a49f99ebbda6d2147452d518bb0cefd651f6`
- Actor-key `index.html`
  `f7d3baf311df0015188cdeaea96d62c5ecb3d7e932851c68347dc28b4784e611`

Review check:
- Served `synthesis_lab/experiments/other_games/runs` on
  `127.0.0.1:18778`.
- HTTP `200` returned for actor-key `index.html` and `result.json`, and for
  baseline `index.html` and `result.json`.
- Temporary server stopped.

Boundary:
- This is controlled local/offline evidence only. It is not live/ranked
  proof, not a public leaderboard result, and not a 4/7 route.
- Do not launch another longer `graph_actor_semantic`/`actor_astar` ls20 L3
  run as the next move; that would be same-basin. The next valid ls20 work
  needs a further representation change: e.g. explicitly model energy/action
  budget admissibility, door/lock transformations from before/after deltas, or
  a clearly labeled source-derived mechanic parse if allowed.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
- Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~06:22): TUFA SEED SWEEP + R11L SEED MAP INTEGRATED

Current integration after the checkpoint-safe Tufa wrapper:
- Added `synthesis_lab/experiments/other_games/tufa_seed_sweep.py`, a
  controlled local/offline runner that delegates each seed row to
  `codexlab/arc_suite_runner.py`, strips `ARC_ALLOW_LIVE` from child
  environments, writes a review surface, and records child seed/checkpoint
  trace fields.
- Added `TUFA_FALLBACK_SEED_MAP` and `ARC_TUFA_SEED_MAP` support in
  `codexlab/arc_play_harness.py`. `ARC_TUFA_SEED` still has top priority.
  The only built-in seed-map row is `r11l-495a7899 -> 101`.

Controlled local/offline seed sweep:
- Review surface:
  `codexlab/runs/tufa_seed_sweep_cn04_r11l_600_20260707T111000Z/index.html`
  with raw `result.json`, `report_manifest.json`, and `report_trace.jsonl`.
- Command:
  `python3 synthesis_lab/experiments/other_games/tufa_seed_sweep.py --games cn04-2fe56bfb r11l-495a7899 --tufa-seeds default 7 42 101 --env-dir synthesis_lab/experiments/other_games/environment_files --max-actions 600 --arc-seed 7 --zero-stop 2 --run-id tufa_seed_sweep_cn04_r11l_600_20260707T111000Z --out-dir codexlab/runs/tufa_seed_sweep_cn04_r11l_600_20260707T111000Z`
- Result: `rows=6`, `best_total_levels=1`,
  `best_total_score=1.3065543677788576`, `policy_error_count=0`.
- cn04 rows: default seed `868634289` -> `0` levels; seed `7` -> `0`
  levels. Early stop recorded
  `stale_loop_zero_level_after_2_seeds`.
- r11l rows: default seed `997786576` -> `1` level, score
  `0.2013068307067783`; seed `7` -> `1` level, score
  `0.25537528030602824`; seed `42` -> `0` levels; seed `101` -> `1`
  level, score `1.3065543677788576`.

Repeat check:
- Review surface:
  `codexlab/runs/tufa_seed_repeat_r11l_101_20260707T111500Z/index.html`.
- Command:
  `ARC_TUFA_SEED=101 ARC_TUFA_RUNS_DIR=/home/frosty40/megarexus/codexlab/runs/tufa_seed_repeat_r11l_101_20260707T111500Z/tufa_sidecars python3 codexlab/arc_suite_runner.py --policy arcade_portfolio_v1 --games r11l-495a7899 --env-dir synthesis_lab/experiments/other_games/environment_files --seed 7 --max-actions 600 --run-id tufa_seed_repeat_r11l_101_20260707T111500Z --out-dir codexlab/runs/tufa_seed_repeat_r11l_101_20260707T111500Z`
- Result reproduced: `total_levels_completed=1`,
  score `1.3065543677788576`, actions `566`, resets `34`,
  `policy_error_count=0`, seed source `ARC_TUFA_SEED`.

Integrated seed-map portfolio check:
- Review surface:
  `codexlab/runs/tufa_seedmap_portfolio_cn04_r11l_20260707T111800Z/index.html`.
- Command:
  `python3 codexlab/arc_suite_runner.py --policy arcade_portfolio_v1 --games cn04-2fe56bfb r11l-495a7899 --env-dir synthesis_lab/experiments/other_games/environment_files --seed 7 --max-actions 600 --run-id tufa_seedmap_portfolio_cn04_r11l_20260707T111800Z --out-dir codexlab/runs/tufa_seedmap_portfolio_cn04_r11l_20260707T111800Z`
- Result: `game_count=2`, `aggregate_score=0.6532771838894288`,
  `total_levels_completed=1`, `policy_error_count=0`.
- cn04 stayed at `0` levels with stable hash seed `868634289`.
- r11l used seed `101` from `tufa_fallback_seed_map` and reproduced `1`
  level, score `1.3065543677788576`, actions `566`, resets `34`.
- Trace fields for both rows: `checkpoints_enabled=false`,
  `checkpoint_numbered=false`.

Artifact hashes:
- Sweep `result.json`
  `82f4dc7fe956bd1194faaf7b1aaaa3b79a1c9730da3e072674e00a955c02d97d`
- Sweep `index.html`
  `675cd6078377dac5a6615350dd92b055a723c39b0cf5bbe5c24cf49d02a83e66`
- Sweep `report_manifest.json`
  `4bac13057bd6d5cc09893c401a321138f6960edd5e55bfa4d31cd8285630b986`
- Sweep `report_trace.jsonl`
  `c91ff54fca424bc9d4b5d1859b91db9da4fa392def53fdf8bb39b70bb7d8a49e`
- Repeat `result.json`
  `10d7713b067728ee515c0e18a7d6ec32ccab38f12747421f2c53908be534fc83`
- Repeat `index.html`
  `6ba6c0ab4bd3feb2aee7736a93f1fc939c6e27b19a6392cf4bce28210d6641af`
- Seed-map `result.json`
  `45b11a5cf829a2f7d1e1677b9b53f7910598e9cdc50368126cc679f8e2150847`
- Seed-map `index.html`
  `fadd08b8ac62cfdc2c675b29acc075f25a1b737249503af7b5b6209d56f53c1d`
- Seed-map `report_manifest.json`
  `e390e73552f598aa8b6e1bd11256f44336c359ddd86b58091ee536ed6a23fec8`
- Seed-map `report_trace.jsonl`
  `4935f51e6903eb65e0dfa5390bf3f2f7e7a00011dd6a1690248bd87374490ac2`

Filesystem and review checks:
- No `.pt` or `checkpoint_*.pt` files were found under the sweep, repeat, or
  seed-map artifact directories.
- Sizes: sweep `3.9M`, repeat `628K`, seed-map `744K`,
  `codexlab/runs/tufa_action_sidecars` `996K`, repo-root `runs/` `4.0K`.
- Local HTTP check returned `200` for seed-map `index.html`, `result.json`,
  `report_manifest.json`, and `report_trace.jsonl` on `127.0.0.1:18777`;
  temporary server was stopped.

Boundary:
- This is local/offline fallback hyperparameter evidence only. It is not
  live/ranked ARCPrize3 proof, not a leaderboard claim, and not evidence that
  cn04 or r11l is solved.
- The deterministic seed map improves reproducibility and r11l fallback score
  under the same 600-action local condition; it does not increase the local
  level count beyond `1` for the two fallback games.
- The old all-public cn04 `1` level did not reproduce under the controlled
  default/7 seed sweep; treat that old row as stochastic fallback evidence,
  not a stable cn04 capability.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
- Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~06:04): TUFA FALLBACK WRAPPER SAFE FOR REPEATED OFFLINE SCOUTS

Current integration added after the all-public scout:
- `codexlab/arc_play_harness.py` now wraps the external Tufa `Action` agent
  with safe default runtime controls:
  - Tufa experiment sidecars are redirected to
    `codexlab/runs/tufa_action_sidecars/` instead of top-level `runs/`.
  - Checkpoint writes are disabled by default with
    `ACTION_DISABLE_CHECKPOINTS=1` and `ACTION_CHECKPOINT_NUMBERED=0`.
  - Checkpoints can be explicitly re-enabled only with
    `ARC_TUFA_ENABLE_CHECKPOINTS=1`.
  - The Tufa fallback is now seeded with a stable per-game hash unless
    `ARC_TUFA_SEED` is set. The seed is recorded in `policy_trace`.
- This removes the previous `193G` checkpoint-spam failure mode from
  all-public fallback scouts while preserving local/offline Tufa action
  training behavior inside the episode.

Controlled local/offline smoke:
- Review surface:
  `codexlab/runs/tufa_fallback_nockpt_seeded_cn04_seed7_20260707T110400Z/index.html`
  with raw `result.json`, `report_manifest.json`, and `report_trace.jsonl`.
- Command:
  `python3 codexlab/arc_suite_runner.py --policy arcade_portfolio_v1 --games cn04-2fe56bfb --env-dir synthesis_lab/experiments/other_games/environment_files --seed 7 --max-actions 80 --run-id tufa_fallback_nockpt_seeded_cn04_seed7_20260707T110400Z --out-dir codexlab/runs/tufa_fallback_nockpt_seeded_cn04_seed7_20260707T110400Z`
- Result: `game_count=1`, `total_levels_completed=0`,
  `policy_error_count=0`, actions `79`, resets `1`.
- Trace fields: `checkpoints_enabled=false`, `checkpoint_numbered=false`,
  seed `868634289`, seed source `stable_game_id_hash`,
  Tufa experiment dir
  `/home/frosty40/megarexus/codexlab/runs/tufa_action_sidecars/20260707_060341`.
- Filesystem check:
  - no top-level `runs/20260707_*` directories were created.
  - `codexlab/runs/tufa_action_sidecars` was `492K`.
  - `runs/` remained `4.0K`.
  - `find codexlab/runs/tufa_action_sidecars/20260707_060341 -maxdepth 4 -type f`
    listed only `git_info.txt` and `logs.log`; no checkpoint files.
- Artifact hashes:
  - `result.json`
    `6bf2b1bd1b65441b91f5c0f1d77e28a4981c14182ac04185be0e2339bd732260`
  - `index.html`
    `95cd0e6117a52442db70be6e49da0452385f578d5de2ebdf650a74b82ab48614`
  - `report_manifest.json`
    `3c174dbf705d8df26ecc44f34abc8062301747cd0cf2a732e82952e3130729ca`
  - `report_trace.jsonl`
    `582fb54891e0b6f91d066b18155ca9ba9ba737d6df1ff7219f870eb602b022ab`
- Local HTTP check returned `200 OK` for `index.html`, `result.json`,
  `report_manifest.json`, and `report_trace.jsonl` on `127.0.0.1:18775`;
  temporary server was stopped.

Boundary:
- This is a runtime/control fix plus a local/offline smoke. It is not a new
  score result, not live/ranked proof, and not evidence that cn04 is solved.
- The purpose is to make future legal offline fallback/training loops safe to
  repeat without unmanaged disk growth and without wall-clock-seeded drift.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
- Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~05:57): ALL-PUBLIC LOCAL SUITE SCOUT -- 25 GAMES, 17 LOCAL LEVELS

Current integration added after the portfolio run:
- `synthesis_lab/experiments/other_games/download_game.py` now supports
  `--all`, repeated `--exclude`, `--force`, and a JSON download manifest.
  The downloader still uses only the public anonymous-key endpoint and fetches
  `/api/games`, per-game metadata, and per-game source; it does not construct
  an Arcade or open a scorecard.
- `codexlab/arc_play_harness.py` now gives `tufa_action_policy` an
  `export_trace()` and routes unsupported `arcade_portfolio_v1` games to a
  clearly marked `tufa_action_fallback`. The three specialized routes remain:
  SP80 -> `megarexus_dw_directgrid`, ft09 -> `ft09_glyph`, ls20 ->
  `ls20_corridor`.

Download manifest:
- `synthesis_lab/experiments/other_games/runs/download_manifest_20260707T102931Z.json`
- Catalog count `25`; requested `24` after excluding repo-root
  `sp80-589a99af`; downloaded `22`; skipped existing ft09/ls20 local copies;
  `opened_arcade_or_scorecard=false`.
- Manifest hash:
  `ed2bdd46a8307e3267000f6304bb9a1f472047d309f33e5ddb6ec915e0c06736`

Controlled local/offline all-public scout:
- Review surface:
  `codexlab/runs/arc_suite_arcade_portfolio_v1_allpublic_seed7_20260707T103000Z/index.html`
  with raw `result.json`, `report_manifest.json`, and `report_trace.jsonl`.
- Command:
  `python3 codexlab/arc_suite_runner.py --policy arcade_portfolio_v1 --env-dir environment_files --extra-env-dir synthesis_lab/experiments/other_games/environment_files --seed 7 --max-actions 600 --run-id arc_suite_arcade_portfolio_v1_allpublic_seed7_20260707T103000Z --out-dir codexlab/runs/arc_suite_arcade_portfolio_v1_allpublic_seed7_20260707T103000Z`
- Result: `game_count=25`, `aggregate_score=9.032793584260942`,
  `total_levels_completed=17`, `policy_error_count=0`.
- Nonzero rows:
  - SP80: `6/6`, score `100.0`, actions `157`, resets `0`,
    final state `GameState.WIN`, child policy `megarexus_dw_directgrid`.
  - ft09: `6/6`, score `100.0`, actions `86`, resets `0`,
    final state `GameState.WIN`, child policy `ft09_glyph`.
  - ls20: `3/7`, score `21.428571428571427`, actions `600`, resets `0`,
    final state `GameState.NOT_FINISHED`, child policy `ls20_corridor`.
  - cn04: fallback `tufa_action_fallback`, `1` level, score
    `0.03444038067063324`, `594` actions, `6` resets.
  - r11l: fallback `tufa_action_fallback`, `1` level, score
    `4.356827797281484`, `569` actions, `31` resets.
- All other fallback games remained at `0` levels under the 600-action cap.
- Artifact hashes:
  - `result.json`
    `989211f29cccdc72fb048c02ee0d4a2f6d53e760889805c11aad9410f4578036`
  - `index.html`
    `c3280f1ef3f62b605d65e0ab5a2ce4b35902c66275e37baea8e600ec228af0c5`
  - `report_manifest.json`
    `b4724e0f7b2f7dfc28ad875bef2c8f78ba4f5179c48996f83e33d9229126b6ec`
  - `report_trace.jsonl`
    `2566e4f39d3a4662e3a32bbc6cf0698e080c7fc1241acbda97bb9e7f3f75bff6`
- Local HTTP check returned `200 OK` for `index.html`, `result.json`,
  `report_manifest.json`, and `report_trace.jsonl` on `127.0.0.1:18773`;
  temporary server was stopped.

Side effect and cleanup:
- The external Tufa fallback created top-level `runs/20260707_*` checkpoint
  directories while running fallback games, about `193G` total. These were
  generated side effects, not the review artifact, and were removed after the
  suite result was captured. `runs/` is back to `4.0K`.
- Future all-public Tufa fallback sweeps should disable checkpoint writes or
  use a much smaller cap before repeating this breadth condition.

Boundary:
- This is controlled local/offline suite evidence over copied local
  environment files. It is not live/ranked ARCPrize3 proof and not a public
  leaderboard claim.
- `arcade_portfolio_v1` is still an integration dispatcher. Unknown games use
  an external generic Tufa action fallback and must not be described as
  specialized Megarexus solves.
- ls20 remains graph-route replay and remains partial at `3/7`.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
- Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~05:23): PORTFOLIO OFFLINE SUITE INTEGRATED -- SP80+ft09+ls20 = 15 LOCAL LEVELS

Current integration:
- `codexlab/arc_play_harness.py` now registers `POLICIES["arcade_portfolio_v1"]`.
  It dispatches by game id:
  - `sp80-589a99af` -> `megarexus_dw_directgrid`
  - `ft09-0d8bbf25` -> `ft09_glyph`
  - `ls20-9607627b` -> `ls20_corridor`
- `codexlab/arc_suite_runner.py` now supports repeated `--extra-env-dir`.
  When extras are supplied, it builds a run-local copied
  `merged_environment_files/` view and records the source dirs plus per-game
  runtime dirs in `env_merge_manifest`.

Controlled local/offline portfolio run:
- Review surface:
  `codexlab/runs/arc_suite_arcade_portfolio_v1_seed7_20260707T102138Z/index.html`
  with raw `result.json`, `report_manifest.json`, and `report_trace.jsonl`.
- Command:
  `python3 codexlab/arc_suite_runner.py --policy arcade_portfolio_v1 --games sp80-589a99af ft09-0d8bbf25 ls20-9607627b --env-dir environment_files --extra-env-dir synthesis_lab/experiments/other_games/environment_files --seed 7 --max-actions 260 --run-id arc_suite_arcade_portfolio_v1_seed7_20260707T102138Z --out-dir codexlab/runs/arc_suite_arcade_portfolio_v1_seed7_20260707T102138Z`
- Result: `aggregate_score=73.80952380952381`,
  `total_levels_completed=15`, `policy_error_count=0`.
- Per-game rows:
  - SP80: `levels_completed=6/6`, score `100.0`, actions `157`,
    resets `0`, final state `GameState.WIN`, child policy
    `megarexus_dw_directgrid`.
  - ft09: `levels_completed=6/6`, score `100.0`, actions `86`,
    resets `0`, final state `GameState.WIN`, child policy `ft09_glyph`.
  - ls20: `levels_completed=3/7`, score `21.428571428571427`,
    actions `260`, resets `0`, final state `GameState.NOT_FINISHED`, child
    policy `ls20_corridor`.
- Runtime env merge:
  - SP80 copied from `environment_files/sp80/589a99af`
  - ft09 copied from
    `synthesis_lab/experiments/other_games/environment_files/ft09/0d8bbf25`
  - ls20 copied from
    `synthesis_lab/experiments/other_games/environment_files/ls20/9607627b`
- Artifact hashes:
  - `result.json`
    `c1e11203fd6b4144e7839f97402d52e2db4adacc43857a0b1a6e2d4e9ca45900`
  - `index.html`
    `30c3db082a3429a31c2185b43d051c21d8916279cd468ca70e1e657475921f0b`
  - `report_manifest.json`
    `a0aabc8d1bf537bd90d037c525c326da24b106183276b27d290bab3b3bf685dc`
  - `report_trace.jsonl`
    `97980affb9f244ea6fed7352adf8b1f6d1dbcad28413701e85966d9442c0d400`
- Local HTTP check returned `200 OK` for `index.html`, `result.json`,
  `report_manifest.json`, and `report_trace.jsonl` on `127.0.0.1:18772`;
  temporary server was stopped.

Superseded artifact:
- `codexlab/runs/arc_suite_arcade_portfolio_v1_seed7_20260707T102054Z/`
  used symlinked env dirs and every `arc.make()` returned `None`.
  Treat it only as an environment-merge failure diagnostic, not score evidence.

Boundary:
- This is controlled local/offline suite evidence over copied local
  environment files. It is not live/ranked ARCPrize3 proof and not a public
  leaderboard claim.
- `arcade_portfolio_v1` is an integration dispatcher. Solver boundaries remain
  in each child `policy_trace`; ls20 is still graph-route replay and remains
  partial at `3/7`.
- Final gates after portfolio integration: `py_compile` on touched Python
  files exit `0`, `git diff --check` exit `0`, and `honesty_audit.py` exit
  `0` with `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
- Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~05:15): ROUND-5 ORDER REVALIDATED; ls20 L3 SEARCH BOUNDED NEGATIVE

The user-requested round-5 resume order is complete in the current working
tree and was re-run after the other-games edits:
`row_flash` column generalization with exact old-label regression →
`FlowTrail.wrap` knob → L4/L5 pour perception gate → arranger `{8,9,15}` +
L5 vertical-unit continuity gate → `r5_fit.py` full grid → L4 `dw_solve`
sweep.

Re-run commands and verdicts:
- `python3 synthesis_lab/orchestration/sim3d/r4_loader_regress.py`
  - PASS. Old 4px corpora masks/lava labels unchanged, and the generalized
    row-or-column flash labels exactly match the old rows-only labels on all
    five pre-r5 corpora. L3 still loads `40/40`, `0` skipped, all lava.
- `python3 synthesis_lab/orchestration/sim3d/r5_perception_pour_gate.py`
  - PASS. L4 `40/40` trusted, L5 `40/40` trusted, no unmodeled colors.
- `python3 synthesis_lab/orchestration/sim3d/perception_gate_eval.py`
  - PASS. Present-frame L0-L5 trusted `6/6`.
- `python3 synthesis_lab/orchestration/sim3d/arranger_gate.py --levels 4,5 --seeds 555,1234,42 --n 40 --out arranger_gate_L4_L5_20260706.json`
  - PASS. Each L4/L5 seed row has `40/40` byte matches, `0` mismatches,
    `0` kernel disagreements, and `0` post-pour mismatches.
- `python3 synthesis_lab/orchestration/sim3d/r5_fit.py`
  - PASS across the 832-config grid. Selected:
    `flow_trail emitters=all phase=uncharged_lead mouth=feed_die
    volume=unlimited merge=die blocker_contact=block wrap=immediate
    side_mouth=adjacent_feed`, `rule=fill_cap`, `stop=volume`.
  - L2 no-regression exam PASS: outcome `1.0`, lava `1.0`, `40` rows.
- L4 sweep:
  `python3 synthesis_lab/orchestration/sim3d/dw_solve.py --level 4 --seed <seed> --model r5 --restarts 0 --steps 0 --topk 4`
  for seeds `{1234,7,42,101,2026}`.
  - All five are `SOLVED@1`, each with `4296` kernel evals,
    `0` refused/dead, and `0` search pours in the command.

Round-5 artifact hashes after re-run:
- `sim3d/fixtures/r4_phaseb_loader_20260706.json`
  `4a4ca70769ea38a16729b3c4c4994a6074da5d20ff72e88fa861f9143e1756e5`
- `sim3d/fixtures/r5_perception_pour_gate_20260706.json`
  `31b3dd88faace14d2a87d994c9e62fb89bcb8a6aba67c0e0be27091d8770cb26`
- `sim3d/fixtures/perception_gate_L0_L5_20260706.json`
  `78246ad5cb73bea49e338464ea8b4a0c5c1fa2e40f70540636d3516b4b2c1c54`
- `sim3d/fixtures/arranger_gate_L4_L5_20260706.json`
  `9790d5e469b1492594c74aa743939f6554477d3acfb4cb1f19a76b146e900a8b`
- `sim3d/fixtures/dynamics_round5_20260706.json`
  `0be23dc270eb460413642b6c872f788f86687d40ee9ba3dd9ba857057f5c043b`
- L4 direct-grid solve JSON hashes:
  - s1234 `7964e2a61d11fc09d0a030cabe355b2a9dd51c650cda21aa21ee04baaa58eae6`
  - s7 `dab3d8b8868f4e90db1051fe1172bb25f5d62dde4a33ddc0dec7c79328e2b0c6`
  - s42 `ef9c4a5bcdd7e073e4e6b61881bbe8bc3939ff72f6aa218d220b6131a6dbe287`
  - s101 `a2fde2f5997d6db0dd772176b43d752ccbe93da93ece78be745e636f7f609e66`
  - s2026 `6546bdb418359a767fb5faff9be14812ab8c42cd79226763d3285315f737fdfb`

Other-games side result since the prior note:
- `ls20_corridor_probe.py` now has explicit graph engines:
  `wrapper`, `game_clone`, and deadline-bounded `replay`.
- A wrapper-vs-direct-game consistency smoke matched exactly on the known 97
  action 3/7 route.
- Durable level-3 replay search:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_replay_20260707T100846Z/index.html`
  with raw `result.json`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --chain-levels 1 --graph-engine replay --max-ext 180 --max-nodes 50000 --deadline-seconds 180 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_replay_20260707T100846Z`
- Result: `status=deadline`, no verified extension to 4/7, elapsed
  `180.049s`, `states_seen=1017`, `queue_popped=872`, `expanded_edges=1016`,
  `max_queue=147`, `queue_remaining=145`, `game_clone_consistency.match=true`.
- Hashes:
  - result `1eeddd86f4305e5ec2ad9ddc3cd2468fa02463873c0723152c2aeddba1f2bc1f`
  - index `ca664ea3e3a0bb56cf902952ce5b6ecbc75334ecfc1f0a2ef591cd6fce6b23b2`
- Local HTTP check returned `200 OK` for that `index.html` and `result.json`
  on `127.0.0.1:18771`; the temporary server was stopped.

Boundary:
- Round-5 evidence is controlled local/offline SP80 evidence over repo
  `environment_files`, not a live/ranked ARCPrize3 scorecard.
- The ls20 level-3 search is a bounded negative artifact. Do not claim 4/7 or
  a suite improvement from it.
- Final gates after notes: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
- Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~04:58): ls20 ADVANCED TO 3/7 — graph-derived route integrated into offline suite

Read the newest OTHER_GAMES_RECON_20260706.md entry:
`2026-07-07 ~04:58 — ls20 clone-backed graph search reaches 3/7 and suite policy records route boundary`.

Current state:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py` now has a
  clone-backed graph search mode:
  - `graph_position` keys by level/avatar/pickup order.
  - `graph_semantic` keys by level/avatar/pickup order, relevant object counts,
    and a stable frame hash with the volatile bottom budget strip masked.
- The level-1 and level-2 graph searches are reproducible from one CLI command
  with `--chain-levels 2`; together they produce a 97-action prefix that reaches
  `levels_completed=3/7`.
- The graph-derived route was verified across seeds `{7,101,1234}` with the
  same final frame hash.
- `codexlab/arc_play_harness.py` now registers `POLICIES["ls20_corridor"]`.
  It is explicitly a graph-derived route replay policy, not a learned policy.
- `codexlab/arc_suite_runner.py` passes `game_id` into `ls20_corridor` and its
  generic offline manifest now says route/replay status is policy-specific and
  recorded in `policy_trace`.

Evidence:
- Graph-chain review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_chain3_20260707T1005Z/index.html`
  with raw `result.json`.
- Graph-chain command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --chain-levels 2 --modes graph_semantic --max-ext 130 --max-nodes 5000 --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_chain3_20260707T1005Z`
- Graph-chain result:
  - target level 1: `status=found`, extension length `45`,
    `states_seen=1463`, `queue_popped=1365`, reaches `levels_completed=2`.
  - target level 2: `status=found`, extension length `39`,
    `states_seen=2050`, `queue_popped=1871`, reaches `levels_completed=3`.
  - best route length `97`, final state `GameState.NOT_FINISHED`,
    final frame hash `7f58b27c980459e6378bb4fda5d66e2fe2dc2516c36f4b0dbbd873f545e17522`.
- Seed-set verification:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_known_route3_seedset_20260707T1019Z/index.html`
  verifies that the 97-action route reaches `levels_completed=3` on
  seeds `{7,101,1234}` with the same final frame hash.
- Offline suite review surface:
  `codexlab/runs/arc_suite_ls20_corridor_seed7_graphroute3_20260707T1029Z/index.html`
  with raw `result.json`, `report_manifest.json`, and `report_trace.jsonl`.
- Suite command:
  `python3 codexlab/arc_suite_runner.py --policy ls20_corridor --games ls20-9607627b --env-dir synthesis_lab/experiments/other_games/environment_files --seed 7 --max-actions 97 --no-perception --run-id arc_suite_ls20_corridor_seed7_graphroute3_20260707T1029Z --out-dir codexlab/runs/arc_suite_ls20_corridor_seed7_graphroute3_20260707T1029Z`
- Controlled offline suite result:
  `aggregate_score=21.428571428571427`, `total_levels_completed=3`,
  game `levels_completed=3/7`, `actions=97`, `resets=0`,
  final state `GameState.NOT_FINISHED`, `policy_error_count=0`.
  Level actions: `[13,45,39,0,0,0,0]`.
- Suite policy trace boundary:
  `graph_derived_route_replay_from_local_offline_probe;not_live_scorecard_proof;not_human_route;not_source_derived;not_learned_policy`.
- Artifact hashes:
  - graph-chain `result.json`
    `0ae8c486acbc06b5db42913eda6f2f3f5b19d9211e68a87377b2b16dc2c338f3`
  - graph-chain `index.html`
    `936960142ed0bd889fd2888005e173d1b3024969a77c485495877c07eb73a9e9`
  - seed-set `result.json`
    `8b1714c72704df4ff40e1fee9acf7a75a88b60f0749d951327b12f476e8ce307`
  - seed-set `index.html`
    `803fbbe65f5a87c798acaba2fd322ba495aeef3394a22af0c0a4b5e98f5bd72a`
  - suite `result.json`
    `c4b3408b795bc55a7234702e5d58eacdd6f490f33b2572c402d2c8021d45aca5`
  - suite `index.html`
    `e601df425edf1ad496dde008118957e73f64dfd087c644a6be5d9a2e9ae751f9`
  - suite `report_manifest.json`
    `a85cc81af4b1aaad732e7592ccd68352f89d185df1523fb13f440ab767fb9237`
  - suite `report_trace.jsonl`
    `85d7bbe77021311397d94631aef2e71f650e92681836ca53a3ee7f1e38e53f07`
- Local HTTP reachability check returned `200 OK` for the graph-chain and
  seed-set `index.html`/`result.json` on `127.0.0.1:18769`, and for the suite
  `index.html`, `result.json`, `report_manifest.json`, and
  `report_trace.jsonl` on `127.0.0.1:18770`; temporary servers were stopped.

Boundary:
- This is controlled local/offline evidence over downloaded public-preview
  environment files. It is not a live/ranked ARCPrize3 scorecard and not public
  leaderboard evidence.
- `ls20_corridor` is a graph-derived route replay policy, not a learned model
  and not an autonomous current-frame general solver. It is still legal under
  the local/offline lane because the route was induced by SDK action search,
  not by game source, human answers, or live scorecard leakage.
- A level-3 graph search console probe was interrupted and produced no
  artifact; do not cite it as evidence.
- Next valid work: either continue ls20 from the 3/7 prefix with a changed
  level-3 representation and a bounded artifact, or move to another downloaded
  game for breadth. Live/ranked work still needs explicit scorecard intent.

Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~04:34): DW RUNNER VERIFIED — local/offline SP80 6/6 through suite path

Read the newest DW_LAB_LOG_20260706.md entries:
`2026-07-07 ~03:05 — RUNNER CLOSURE: full offline SP80 solved by DW policy, L5 transition fixed`
and
`2026-07-07 ~04:34 — DW runner current-code verification`.

Current state:
- The original round-5 resume order is complete in the current tree:
  `row_flash` row/column generalization, `FlowTrail.wrap`, L4/L5 pour gates,
  arranger `{8,9,15}` with L5 vertical unit parse, `r5_fit.py`, L4 and L5
  direct-grid local solves, and runner wiring.
- `codexlab/arc_play_harness.py` registers policy `megarexus_dw_directgrid`.
  It uses current-frame DW priors for L0-L3, r5 direct-prior/direct-grid for
  L4, and geometry-priority L5 direct-grid.
- A fresh current-code local/offline suite run verifies continuous SP80 closure
  through the actual `codexlab/arc_suite_runner.py` path.

Evidence:
- Review surface:
  `codexlab/runs/arc_suite_megarexus_dw_directgrid_verify_20260707T0935Z/index.html`
  with raw `result.json`, `report_manifest.json`, and `report_trace.jsonl`.
- Command:
  `python3 codexlab/arc_suite_runner.py --policy megarexus_dw_directgrid --games sp80-589a99af --max-actions 220 --run-id arc_suite_megarexus_dw_directgrid_verify_20260707T0935Z`
- Controlled local/offline result:
  `aggregate_score=100.0`, `total_levels_completed=6`, game
  `levels_completed=6/6`, `actions=157`, `resets=0`,
  final state `GameState.WIN`, scorecard state `WIN`,
  `policy_error_count=0`.
- Per-level action counts:
  `[12,8,13,34,48,42]`.
- Policy trace:
  `plans_built=6`, `direct_actions=157`, `fallback_actions=0`.
- Artifact hashes:
  - `result.json`
    `3ec784ab9b3d8857748a274e306792caae4f1e24979e82e7157360874e05c38a`
  - `index.html`
    `dd0df6be0d444caac5242fc2c7cf450753870afa743593448815ee1409dafcdd`
  - `report_manifest.json`
    `5a66e50bb27e340a411acbbc17dc03b891e7dbc402177ce71483c3f33786ded1`
  - `report_trace.jsonl`
    `a51b46d029508e251c0fd62c849b347194a602b35ea46e538e38d24bbe32ddff`
- Local HTTP reachability check returned `200 OK` for `index.html`,
  `result.json`, `report_manifest.json`, and `report_trace.jsonl` on
  `127.0.0.1:18768`; temporary server was stopped.

Boundary:
- This is real controlled local/offline SP80 evidence over repo
  `environment_files`, not a live/ranked ARCPrize3 scorecard, not public
  leaderboard evidence, and not a multi-game suite claim beyond SP80.
- L5 remains an induced-kernel under-score case: the winning topology-priority
  direct-grid candidate is env-confirmed but not certified by `imagined_win`.
- No live/ranked run has explicit permission in this handoff. Next valid work:
  either request/receive explicit live-scorecard intent for a ranked scout, or
  continue suite breadth with the offline downloaded games lane.

Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~04:28): ls20 LEVEL 0 VERIFIED — key-route reaches 1/7; level 1 remains open
Read the newest OTHER_GAMES_RECON_20260706.md entry:
`2026-07-07 ~04:28 — ls20 level-0 key route verified; level-1 search scaffold added`.

Current state:
- Added `synthesis_lab/experiments/other_games/ls20_corridor_probe.py`.
- It is local/offline only, uses legal keyboard actions `[1,2,3,4]`, and does
  not inspect the downloaded `ls20.py`.
- Level-0 controlled route from frames/env interaction:
  `3331111444111` (left-left-left, up-up-up-up, right-right-right,
  up-up-up) reaches `levels_completed=1/7`.
- Durable artifact uses `--skip-search`: it verifies the level-0 route and
  renders the level-0 and level-1 frames. It does NOT claim a level-1 solve.
- Level-1 frame-derived special objects detected:
  `item01` at centroid `[51.0,47.0]`, `c11_1` at `[16.0,17.0]`,
  `c11_2` at `[41.0,52.0]`.
- Interactive replay BFS probes before artifact packaging found no level-1
  completion for position-only, pickup-set, or pickup-order abstractions, but
  those broader sweeps were not completed as durable review artifacts. Treat
  that as planning signal only; next valid work is a cached/reusable replay
  graph or a changed representation for level 1.

Evidence:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_level0_20260707T092711Z/index.html`
  with raw `result.json`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --skip-search --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_level0_20260707T092711Z`
- Controlled local/offline result:
  `level0_route=3331111444111`, `level0_result.levels_completed=1`,
  final state after route `GameState.NOT_FINISHED`, avatar position `[31,40]`.
- Level-0 transition/frame hash:
  `0088262d9b328dca202d9e15a2707ed5ba3f6d1b843845b2637fec957114b1cd`.
- Artifact hashes:
  - `result.json`
    `4b43149977f5f02e41eb36fadfd732903b88990938c57367be50951c544b8a96`
  - `index.html`
    `ffbe60a7e7bd2d085799b582006e8ad103dc4b96f134f45a36aeaf9156778a64`
- Local HTTP reachability check returned `200 OK` for `index.html` and
  `result.json` on `127.0.0.1:18766`; temporary server was stopped.

Boundary:
- This is real controlled local/offline evidence for ls20 level 0 only. It is
  not a full ls20 solve, not a live/ranked ARCPrize3 scorecard, and not a
  source-derived route.
- Next valid work: optimize `ls20_corridor_probe.py` with cached replay/action
  graph state, then rerun a bounded level-1 search with a review artifact.

Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~04:10): ft09 SUITE INTEGRATED — `ft09_glyph` wins 6/6 through `codexlab/arc_suite_runner.py`
Read the newest OTHER_GAMES_RECON_20260706.md entry:
`2026-07-07 ~04:10 — ft09 wired into offline suite surface; seed 7 suite row wins 6/6`.

Current state:
- `codexlab/arc_play_harness.py` now registers `POLICIES["ft09_glyph"]`.
- The suite wrapper imports
  `synthesis_lab.experiments.other_games.ft09_glyph_solver`, not the downloaded
  `ft09.py`, and emits one queued `ACTION6` click per harness call.
- Important fix: the first stateless wrapper recomputed after every click and
  got stuck on transitional animation frames at level 1. Superseded diagnostic:
  `codexlab/runs/arc_suite_ft09_glyph_seed7_20260707T090722Z/` reached only
  `levels_completed=1/6`. Use the later queued wrapper artifact below.
- `codexlab/arc_suite_runner.py` now accepts explicit `--env-dir` and `--seed`
  and records both into `result.json` / `report_manifest.json`.

Evidence:
- Review surface:
  `codexlab/runs/arc_suite_ft09_glyph_seed7_20260707T090841Z/index.html`
  with raw `result.json`, `report_manifest.json`, and `report_trace.jsonl`.
- Command:
  `python3 codexlab/arc_suite_runner.py --policy ft09_glyph --games ft09-0d8bbf25 --env-dir synthesis_lab/experiments/other_games/environment_files --seed 7 --max-actions 320 --no-perception --run-id arc_suite_ft09_glyph_seed7_20260707T090841Z --out-dir codexlab/runs/arc_suite_ft09_glyph_seed7_20260707T090841Z`
- Controlled offline suite result:
  `aggregate_score=100.0`, `total_levels_completed=6`, game
  `levels_completed=6/6`, `actions=86`, `resets=0`,
  final state `GameState.WIN`, `policy_error_count=0`.
- Final frame hash:
  `7b71d45a704ca893afd54db1c321db38569971ed9a844958de249fd5d3c5ea1a`
  (matches the v5 native solver artifact).
- Artifact hashes:
  - `result.json`
    `4e04646edd9602027f8b02323189beba91a82ce8937c2aa9a4dc898b84c5010c`
  - `index.html`
    `cd11f4154bf8a4d531e30313cb08b784efeb3b298d246165e4c866170ec9fd1c`
  - `report_manifest.json`
    `83444d31c0c191c0809123201da3ee7b755537fa1a676233e66503e3ddac3861`
  - `report_trace.jsonl`
    `b02c46b1a8e0b82ee33cf90e1d9c739dd2dd583d5c73b5cac7e5b31fcae99a46`
- Local HTTP reachability check for the review surface and linked raw artifacts
  returned `200 OK` on `127.0.0.1:18765`; temporary server was stopped.

Boundary:
- This is real controlled local/offline suite evidence for one downloaded
  non-sp80 game. It is not a live/ranked ARCPrize3 scorecard and not a hidden
  leaderboard claim.
- Next valid work: continue another downloaded non-sp80 game with the same
  frame-derived/offline discipline, or request explicit live-scorecard intent
  before any live/ranked run.

Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~04:00): ft09 LOCAL/OFFLINE SOLVED 6/6 — frame-derived v5 policy wins all levels on three seeds
Read the newest OTHER_GAMES_RECON_20260706.md entry:
`2026-07-07 ~04:00 — ft09 solved locally/offline 6/6 with frame-derived v5 policy`.

Current state:
- `synthesis_lab/experiments/other_games/ft09_glyph_solver.py` is now schema
  `synthesis_lab.other_games.ft09_glyph_solver.v5`.
- It remains local/offline only through `Arcade(..., operation_mode=OFFLINE,
  environments_dir=synthesis_lab/experiments/other_games/environment_files)`.
- It still does NOT read `ft09.py`, does NOT use human answers, and does NOT
  open a live ARC scorecard.
- Added L4 color-14 lattice parsing: clue cells map center `14` and `15`
  polarities over glyph colors `{0,2,3}`; color-6 cells are plus toggles, so
  the planner simulates cross effects before repairing singles.
- Added final-level 11/14 domino planner: active cells are color-11 with
  color-6 markers, clue color `0 -> 14`, colors `2/3 -> 11`, and each click
  toggles its own cell plus the active cell directly above it.

Evidence:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_v5_fullwin_20260707T085858Z/index.html`
  with raw `result.json` and `contact_sheet.png`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ft09_glyph_solver.py --seeds 7 101 1234 --max-actions 320 --out-dir synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_v5_fullwin_20260707T085858Z`
- Controlled offline result across seeds `{7,101,1234}`:
  `levels_completed=6/6`, `win_levels=6`, `actions=86`,
  final state `GameState.WIN`, `policy_errors=0`.
- Final frame hash for all three seeds:
  `7b71d45a704ca893afd54db1c321db38569971ed9a844958de249fd5d3c5ea1a`.

Boundary:
- This is real controlled local/offline ft09 solve evidence, not a live/ranked
  ARCPrize3 scorecard and not suite-level leaderboard evidence.
- Next valid work toward the broader goal: wire this ft09 policy into a local
  multi-game suite surface, then continue another non-sp80 game under the same
  frame-derived/offline discipline. Any live/ranked run still requires an
  explicit live-scorecard boundary.

Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~03:50): ft09 FRAME POLICY ADVANCED TO 4/6 — center-polarity + ternary target colors landed; L4 color-14 layout parser is next
Read the newest OTHER_GAMES_RECON_20260706.md entry:
`2026-07-07 ~03:50 — ft09 frame policy reaches 4/6; L4 parser now blocked on color-14 layout`.

Current state:
- `synthesis_lab/experiments/other_games/ft09_glyph_solver.py` is now schema
  `synthesis_lab.other_games.ft09_glyph_solver.v3`.
- It remains local/offline only and still does not read `ft09.py`.
- L2 is solved by a frame-visible center-polarity rule:
  use color-2 marks only; center `8` means target = 2-marked cells, center
  `12` means target = non-2 cells.
- L3 is solved by explicit ternary target colors and the visible click cycle
  `8 -> 12 -> 9 -> 8`: center `12` maps glyph `0 -> 12`, `2 -> 8`;
  center `9` maps glyph `0 -> 9`, `2 -> 8`.
- The planner now supports explicit per-tile `target_color`, not just boolean
  on/off targets.

Evidence:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_v3_l4_stop_20260707T084835Z/index.html`
  with raw `result.json` and `contact_sheet.png`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ft09_glyph_solver.py --seeds 7 101 1234 --max-actions 180 --out-dir synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_v3_l4_stop_20260707T084835Z`
- Controlled offline result across seeds `{7,101,1234}`:
  `levels_completed=4/6`, `actions=52`, `stop_reason=no_boards`,
  `policy_errors=0`, final state `GameState.NOT_FINISHED`.
- Stop frame hash for all three seeds:
  `9d4eb0268e4dbd7efefe53edac6ec957bfe782d2c7a356bec2b30bb2849d6bdb`.

Important correction:
- The earlier fast transform diagnostic reused an env object across reset/replay
  loops. That was patched: transform candidate grading now uses fresh env
  replay. Corrected fresh replay artifact:
  `synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_l2_freshreplay_20260707T083006Z/`;
  it matched the old negative (`30245` candidates, no solution), but use the
  fresh replay artifact if citing that diagnostic.

L4 blocker:
- The L4 stop frame uses color-14 active tiles with a 14/15 click cycle and
  marker/operator colors including `3`, `6`, `12`, and `15`.
- The existing 3x3 glyph-board detector finds no boards because it only admits
  the earlier tile palettes. A strict custom color-14 scan found one 3x3 board
  at `(y=20,x=30)`, but the visual frame is a larger color-14 layout, so the
  next valid work is L4 perception/layout parsing, not another L2/L3 rule
  search.

Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~03:25): ft09 L2 DIAGNOSTIC CLOSED — color-12 tile transition recorded, final-state glyph relation stale_loop
Read the newest OTHER_GAMES_RECON_20260706.md entry:
`2026-07-07 ~03:25 — ft09 L2 diagnostic: 12/8 palette confirmed, final-state glyph relation exhausted`.

Current state:
- `synthesis_lab/experiments/other_games/ft09_glyph_solver.py` is now schema
  `synthesis_lab.other_games.ft09_glyph_solver.v2`.
- Active tile perception includes `{8,9,12}`. The L2 palette is no longer a
  guess: the recorded one-click probe from the L2 prefix clicked tile
  `(tile_y=4,tile_x=20)`, changed 37 pixels, and showed changed-color counts
  before `{8:36,12:1}` and after `{12:36,11:1}`. So the L2 active true tile
  color is `12`, with the one extra `12 -> 11` pixel being budget/counter burn.
- The runner still stops fail-closed on contradictory overlapping glyph
  constraints. It also now has optional bounded diagnostics:
  `--conflict-search` over marker/palette ambiguity families and
  `--transform-search` over D4 glyph transforms plus reductions.

Evidence:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_l2_diag_20260707T082439Z/index.html`
  with raw `result.json` and `contact_sheet.png`.
- Command:
  `python3 synthesis_lab/experiments/other_games/ft09_glyph_solver.py --seeds 7 101 1234 --max-actions 80 --conflict-search --conflict-max-candidates 2048 --transform-search --transform-max-candidates 50000 --single-click-probe --out-dir synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_l2_diag_20260707T082439Z`
- Controlled offline result across seeds `{7,101,1234}` remains
  `levels_completed=2/6`, `stop_reason=conflicting_glyph_constraints`,
  `policy_errors=0`, final state `GameState.NOT_FINISHED`.
- Stop frame hash remains
  `2123ce4d3135531fc4efd164f92fd791b250440bfd4002525a1598fa18944cbc`.
- Marker/palette conflict diagnostic: `candidate_space=520`,
  `candidates_tried=130`, `solution_found=false`, `stopped_at_limit=false`.
  Base planner palette is `true_color=12`, `false_color=8`.
- Transform diagnostic: `transform_sets_generated=24580`,
  `candidates_tried=30245`, `solution_found=false`,
  `stopped_at_limit=false`.

Honest next step: mark the final-state glyph-board relation family as
`stale_loop`. Do NOT launch more ft09 L2 final-target aggregation, marker-set,
palette, or D4-transform sweeps until the representation changes. The next
valid lane is a stateful/dynamic relation read: controlled click probes that
observe how one physical tile's toggle affects all overlapping boards, budget,
and any hidden acceptance timing, or a source-derived read explicitly labeled
`[SOURCE-DERIVED]` and excluded from autonomous solve claims.

Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~03:15): OTHER-GAMES LOOP STARTED — ft09 frame-derived solver reaches 2/6, L2 conflict basin recorded
Read the newest OTHER_GAMES_RECON_20260706.md entry:
`2026-07-07 ~03:15 — ft09 glyph solver scaffold: 2/6 controlled, L2 overlap rule open`.

Current state:
- `synthesis_lab/experiments/other_games/ft09_glyph_solver.py` is a reusable
  local/offline runner for ft09. It reads frames only: detects 6x6 tile
  components, anchors 3x3 glyph boards by component geometry, decodes center
  mini-map marker colors `{2,12}`, and clicks mismatched ring tiles.
- It does NOT read `ft09.py`, does NOT use human answers, and does NOT open a
  live ARC scorecard. It runs against
  `synthesis_lab/experiments/other_games/environment_files/`.
- The runner is fail-closed: if overlapping glyphs assign contradictory target
  states to the same physical tile, it stops instead of inventing an
  aggregation rule.

Evidence:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_conflict_20260707T0310Z/index.html`
  with raw `result.json` and `contact_sheet.png`.
- Controlled offline result across seeds `{7,101,1234}`: each case reaches
  `levels_completed=2/6` in 11 clicks, then stops at
  `conflicting_glyph_constraints`; `policy_errors=0`.
- Stop frame hash is identical across all three seeds:
  `2123ce4d3135531fc4efd164f92fd791b250440bfd4002525a1598fa18944cbc`.
- L2 diagnostic: six physical tiles receive contradictory glyph targets.
  Explicit ambiguity-bit search tried the full `2 * 2^6 = 128` palette/bit
  candidate space from that frame, and found no solution. So the next
  representation is NOT a simple OR/AND/majority/XOR/conflict-bit assignment.

Honest next step: solve ft09 L2 by adding a new observable relation, not more
aggregation guesses. Candidate direction: identify which glyph blocks are clues
versus active target boards, or infer a higher-order operation over the four
glyph mini-maps from controlled click/diff probes. Keep source reads out of the
lane unless explicitly labeled `[SOURCE-DERIVED]` and excluded from solve
claims. No live/ranked run yet.

Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~03:05): DW FULL OFFLINE SP80 RUNNER CLOSED — continuous local suite 6/6, not live/ranked
Read the newest DW_LAB_LOG_20260706.md entry:
`2026-07-07 ~03:05 — RUNNER CLOSURE: full offline SP80 solved by DW policy, L5 transition fixed`.

Current state:
- `codexlab/arc_play_harness.py` has policy
  `megarexus_dw_directgrid`.
- L0-L3 use current-frame scored env-confirmed DW fixture genome priors, with
  r4 genome search as fallback.
- L4 uses current-frame scored r5 direct priors. The transition-frame false
  positive is guarded by low-confidence defer, which triggers the existing
  presentation nudge before replanning.
- L5 now uses the topology-derived `direct_grid_l5_four_mouth` priority first
  (`direct_grid_limit=1`) and only falls back to direct priors if the topology
  proposer is unavailable. This fixes the continuous L5 transition where the
  fixture prior failed but the first geometry-priority direct-grid candidate
  wins in the real local env.
- `dw_solve.py` exposes importable runner helpers for genome priors, genome
  search, direct priors, and direct-grid candidates. `LOCAL_ENV_DIR` resolves
  this repo's active `environment_files/` layout.

Evidence:
- Continuous offline suite:
  `codexlab/runs/arc_suite_megarexus_dw_planner_l5direct_20260707T0300Z/`
  (`index.html`, `result.json`, `report_manifest.json`, `report_trace.jsonl`).
  Result: SP80 local/offline, score `100.0`, `levels_completed=6`,
  `policy_errors=0`, `resets=0`, `actions=157`, `final_state=GameState.WIN`,
  level_actions `[12,8,13,34,48,42]`.
- L5 trace in that run: base frame
  `abc1885d7d3099a3ed6b6d81241d611f2219f3246a9654a2cade864f5885b858`;
  proposal `direct_grid_l5_four_mouth`, deltas
  `[[4,-12],[5,2],[-8,0],[-3,-3]]`, scored `1/6561`,
  `imagined_win=false` but env win after pour. This is the known L5 kernel
  under-score boundary; env remains the decider.
- Isolated L5 runner-shaped smoke after the patch:
  `sim3d/fixtures/dw_policy_rehearsal_L5_directgrid_priority_smoke_20260707.json`
  and `.html`: advanced `1/1`, errors `0`, fallback actions `0`.
- Earlier isolated full L0-L5 runner smoke:
  `sim3d/fixtures/dw_policy_rehearsal_L0_L5_prior_smoke_20260707.json`
  and `.html`: advanced `6/6`, errors `0` before the L5 direct-grid-priority
  continuous fix.

Honest next step: this is the strongest local/offline SP80 evidence so far,
but it is still NOT a ranked ARCPrize3 scorecard and NOT live proof. Next live
step requires explicit live-scorecard intent (`ARC_ALLOW_LIVE=1` boundary) and
should preserve the same review-surface/report discipline.

honesty_audit at wrap: exit 0 — total 97/46/6, active 85/23/0.

Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~01:55): L4/L5 LOCAL SOLVED BY DEFAULT DIRECT-GRID PROPOSERS
Read the three newest DW_LAB_LOG_20260706.md entries:
`2026-07-07 ~01:10 — L4 SIDE-MOUTH REPAIR: r5 dynamics fixed, seeded solve passes, unseeded proposer open`
`2026-07-07 ~01:25 — L4 PROPOSER REPAIR: direct-grid integrated, five-seed L4 solved`
and
`2026-07-07 ~01:55 — L5 FOUR-MOUTH REPAIR: geometry-priority direct-grid five-seed solved`.

Current state:
- L4 side-mouth dynamics are fixed: r5 selected `side_mouth=adjacent_feed`
  and old holdout gates + L2 exam still pass.
- `dw_solve.py` now includes geometry-derived direct-placement proposers for
  L4 side-mouth topology and L5 four-mouth topology. They are enabled by
  default when the topology is present; use `--no-direct-grid` only for the old
  random-genome ablation.
- Controlled local L4 command:
  `python3 synthesis_lab/orchestration/sim3d/dw_solve.py --level 4 --seed <seed> --model r5 --restarts 0 --steps 0 --topk 4`
  solved seeds `{1234,7,42,101,2026}` at `SOLVED@1`, top-1 `env_win=true`.
  Each run used 4,296 kernel evals, 0 refusals, 0 search pours, and no external
  direct-delta fixture.
- Controlled local L5 command:
  `python3 synthesis_lab/orchestration/sim3d/dw_solve.py --level 5 --seed <seed> --model r5 --restarts 0 --steps 0 --topk 4`
  solved seeds `{1234,7,42,101,2026}` at `SOLVED@1`, top-1 `env_win=true`.
  Each run used 6,561 kernel evals, 0 refusals, 0 search pours, and no external
  direct-delta fixture.
- Old random-genome-only L4 negative remains as ablation evidence only:
  `--no-direct-grid` / old 64x150 random proposer missed.
- Old L5 random/direct scout negatives remain as ablation/diagnostic evidence
  only; they established the 3/4-target basin and the later 4/4-with-flash
  repair path.

Key review/evidence surface:
- `sim3d/fixtures/l4_side_mouth_repair_review_20260707.html`
- `sim3d/fixtures/l5_four_mouth_repair_review_20260707.html`

Key fixtures:
- `sim3d/fixtures/dynamics_round5_20260706.json`
- `sim3d/fixtures/dw_solve_r5_L4_s{1234,7,42,101,2026}_directgrid_20260706.json`
- `sim3d/fixtures/dw_r5_L4_s{1234,7,42,101,2026}_directgrid_win_side_by_side_20260706.png`
- `sim3d/fixtures/dw_solve_r5_L5_s{1234,7,42,101,2026}_directgrid_20260706.json`
- `sim3d/fixtures/dw_r5_L5_s{1234,7,42,101,2026}_directgrid_win_side_by_side_20260706.png`
- `sim3d/fixtures/l4_env_win_pour_20260707.npz` and `.provenance.json`
- `sim3d/fixtures/l5_env_win_pour_20260707.npz` and `.provenance.json`
- `sim3d/fixtures/dw_solve_r5_L4_s1234_20260706.json` (old random-genome negative ablation)
- `sim3d/fixtures/l5_flash_repair_evo_scout_s1234_20260707.json` (env-guided diagnostic/train scout)

Honest next step: local L4 and L5 are no longer the blockers. This does NOT
prove a ranked ARCPrize3 scorecard and does NOT prove continuous-episode runner
integration. Wire the default proposer into the runner/full-game rehearsal and
capture offline full-game evidence before any ranked run.

Important boundary: after adding train-only `l5_env_win`, r5 gates still pass
but train win-side accuracy is 12/13. The L5 solver succeeds by
geometry-priority direct-grid plus env confirmation, not by kernel imagined-win
ranking (`imagined_wins=0` in L5 direct-grid fixtures).

honesty_audit after L5 direct-grid proposer: exit 0 — total 97/46/6, active 85/23/0.
Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~01:25): L4 LOCAL SOLVED BY DEFAULT DIRECT-GRID PROPOSER — superseded by L5 block above
Read the two DW_LAB_LOG_20260706.md entries:
`2026-07-07 ~01:10 — L4 SIDE-MOUTH REPAIR: r5 dynamics fixed, seeded solve passes, unseeded proposer open`
and
`2026-07-07 ~01:25 — L4 PROPOSER REPAIR: direct-grid integrated, five-seed L4 solved`.

Current state:
- L4 side-mouth dynamics are fixed: r5 selected `side_mouth=adjacent_feed`
  and old holdout gates + L2 exam still pass.
- `dw_solve.py` includes a geometry-derived direct-placement proposer for
  L4 side-mouth topology.
- Controlled local L4 command:
  `python3 synthesis_lab/orchestration/sim3d/dw_solve.py --level 4 --seed <seed> --model r5 --restarts 0 --steps 0 --topk 4`
  solved seeds `{1234,7,42,101,2026}` at `SOLVED@1`, top-1 `env_win=true`.

## 0-PREV (2026-07-07 ~01:10): L4 SIDE-MOUTH REPAIR LANDED — seeded solve passes, unseeded proposer still open
Read the newest DW_LAB_LOG_20260706.md entry:
`2026-07-07 ~01:10 — L4 SIDE-MOUTH REPAIR: r5 dynamics fixed, seeded solve passes, unseeded proposer open`.

Current state:
- Real env-guided direct-delta repair found an L4 win at eval 564:
  deltas `[[9,-3],[6,-5],[3,7],[-2,-8]]`, seed 1234, all four targets
  full at t19, `levels_completed=1`. This is TRAIN/diagnostic evidence,
  not a holdout claim.
- `FlowTrail` now has `side_mouth ∈ {lateral_only, adjacent_feed}`.
  `adjacent_feed` lets a falling stream adjacent to a side-facing mouth feed
  that mouth; this fixes the L4 side-mouth miss.
- `r5_fit.py` full grid expanded to 832 configs and includes the L4 repair
  win as train-only observation data (`l4_env_win`). It selected:
  `flow_trail emitters=all phase=uncharged_lead mouth=feed_die
  volume=unlimited merge=die blocker_contact=block wrap=immediate
  side_mouth=adjacent_feed`, `rule=fill_cap`, `stop=volume`.
  Old holdouts l0/l1/solvetime/l3/l4/l5 PASS; L2 no-regression exam PASS.
- `dw_solve.py` now accepts explicit direct-delta seed proposals via
  `--direct-deltas` or `--direct-deltas-fixture`. With
  `--direct-deltas-fixture l4_env_repair_single_s1234_20260707.json`,
  L4 seed 1234 is **SOLVED@1** in-budget; top-1 candidate is the exact
  repair deltas and `env_win=true`.
- Unseeded proposer coverage is still open: a fresh unseeded
  `--model r5 --level 4 --seed 1234 --restarts 64 --steps 150` run after
  the dynamics repair still did **NOT** solve. It reached many modeled
  `fill=1.00` candidates with nonzero lava, but no clean in-budget env win.

Key review/evidence surface:
- `sim3d/fixtures/l4_side_mouth_repair_review_20260707.html`

Key fixtures:
- `sim3d/fixtures/l4_env_repair_single_s1234_20260707.json`
- `sim3d/fixtures/l4_env_inspect_win_s1234_20260707.json`
- `sim3d/fixtures/l4_env_win_pour_20260707.npz`
- `sim3d/fixtures/l4_env_win_pour_20260707.provenance.json`
- `sim3d/fixtures/dynamics_round5_20260706.json`
- `sim3d/fixtures/dw_solve_r5_L4_s1234_directseed_20260706.json`
- `sim3d/fixtures/dw_solve_r5_L4_s1234_20260706.json` (unseeded negative)
- `sim3d/fixtures/dw_r5_L4_s1234_directseed_win_side_by_side_20260706.png`

Honest next step: do not claim unseeded DW L4 is solved. The side-mouth
dynamics miss is closed and a seeded in-budget L4 solve is real. The remaining
ranked-run blocker is proposer coverage: promote the direct-delta repair into
a constructive L4 proposal family (or equivalent geometry-aware proposer) so
the solver can reach that basin without consuming an env-guided seed fixture.
honesty_audit after repair: exit 0 — total 97/46/6, active 85/23/0.
Working tree remains uncommitted.

## 0-PREV (2026-07-07 ~00:20): ROUND-5 LANE COMPLETE THROUGH L4 SWEEP — honest negative, ranked still blocked
Read the newest DW_LAB_LOG_20260706.md entry:
`2026-07-07 ~00:20 — ROUND-5 RESUME COMPLETED THROUGH L4 SWEEP`.

Current state:
- `row_flash` is now row OR column wall-band 14-flash; exact old-corpus
  label regression passed on all five pre-r5 corpora. L4/L5 load 40/40 usable
  and 40/40 lava.
- `FlowTrail` has the generic `wrap` knob; r5 fit selected the old-preserving
  `wrap=immediate`.
- Perception gates are clean: L0-L5 present-frame gate 6/6 trusted; L4/L5
  pour-frame gate L4 40/40 and L5 40/40 trusted.
- `virtual_arranger` now handles {8,9,15}, L5's 1-cell-wide vertical 8-bar,
  mixed 8/15 overlaps, and L5 post-pour continuity. Arranger gate PASS:
  L0-L3 and L4/L5 across seeds {555,1234,42}.
- `r5_fit.py` full grid (512 configs) PASS. Selected:
  `flow_trail emitters=all phase=uncharged_lead mouth=feed_die
  volume=unlimited merge=die blocker_contact=block wrap=immediate`,
  `rule=fill_cap`, `stop=volume`. Holdouts l0/l1/solvetime/l3/l4/l5 PASS;
  L2 no-regression exam PASS.
- `dw_solve.py` accepts `--model r5`; L4 sweep was run at the prior frontier
  budget `64x150` over seeds {1234,7,42,101,2026}. Result: **0/5 solved**,
  each with 9,664 kernel evals, 0 refusals, 0 search pours, top-4 env pours
  only. All plateaued at `fill=0.75`, low/zero lava, `imagined_win=False`.

Key fixtures:
- `sim3d/fixtures/r4_phaseb_loader_20260706.json`
- `sim3d/fixtures/perception_gate_L0_L5_20260706.json`
- `sim3d/fixtures/r5_perception_pour_gate_20260706.json`
- `sim3d/fixtures/arranger_gate_20260706.json`
- `sim3d/fixtures/arranger_gate_L4_L5_20260706.json`
- `sim3d/fixtures/dynamics_round5_20260706.json`
- `sim3d/fixtures/dw_solve_r5_L4_s{1234,7,42,101,2026}_20260706.json`

Honest next step: do NOT rework labels/perception first; those gates passed.
The current blocker is proposer/search capacity on L4. The r5 kernel ranks
trusted arrangements but the generic hill-climber cannot reach a 4/4-fill
candidate under the 6.6x budget. Ranked run still waits for L4 per user
ruling. Candidate next lanes: a constructive L4 proposer over the side-wall
mouth geometry, or a targeted analysis of why current genomes top out at three
of four modeled capacities.

SIDE LANE (user-requested, subagent, done): OTHER ARC GAMES unlocked — 25-game
catalog downloadable WITHOUT scorecard; ls20+ft09 obtained & observed offline
(ft09 win mechanic CONFIRMED by env grading, levels 0→2). Read
`notes/OTHER_GAMES_RECON_20260706.md`; tools in
`synthesis_lab/experiments/other_games/`. Honest, observation-only lane.
honesty_audit at wrap: exit 0 — total 97/46/6, active 85/23/0.
Working tree has sim3d edits, generated fixtures, notes, and pre-existing
session artifacts UNCOMMITTED (user decides staging).

## 0-PREV (2026-07-06 late, superseded by 0-NOW above): ROUND-5 MID-LANE HANDOFF — steps 1-2 DONE, step 3 kernel LANDED, resume at row_flash
Read the FOUR new DW_LAB_LOG_20260706.md entries at the bottom (decisions
ruled / step-1 / step-2 / step-3-partial). State:
- **All program-report decisions RULED by the user** (see 0-NEXT block below
  for the full text): Q1 flywheel-refit-first decides L2 split/die; Q2a greedy
  spend-all + no-idle rule; Q2b ≤3 ranked scorecards, report all; Q3 ranked
  run waits for L4; Q4 retry L3 s2026 at higher budget after the L4 lane.
- **Step 1 DONE:** color-15 contact semantics verified on ALL 16 L4 contact
  pours (entered_15=False everywhere → BLOCK; side contact seals edge-spill
  columns; top-adjacent bands pass over; L-notch = one-sided deflection).
  NEW: fail = wall-14 growth at ANY touched wall (pour 25, left wall).
  OPEN physics: wrap-rule asymmetry (pours 9/12 rise at obstacle ends;
  pour 25's band ran to the wall without rising) — must be a fit knob.
  Tool promoted: `sim3d/r5_l4_contact_verify.py`.
- **Step 2 DONE:** L5 Phase-A read — flow FALLS (gravity is per-level);
  2 sources (top charged + mid-board UNcharged embedded in a movable 8-bar);
  side-facing wall mouths + bottom cup; two L-shaped 15 blockers (block
  semantics hold, 0 entries across 40 pours); top row y0 = budget GAUGE;
  fail = SIDE-wall COLUMN 14-flash (row_flash MISSES it — see next-work);
  first VERTICAL movable (1×4-cell 8-bar) challenges the arranger's
  1-cell-tall band-split assumption.
- **Step 3 LANDED (kernel+scene), regression-verified:** PALETTE blocker=15
  (de-grey documented); Scene/KnownMap `blocker` mask; FlowTrail
  `blocker_contact ∈ {block, turn_l, turn_r, split, pass}`; gravity needs NO
  new knob (extract_scene flip canonicalization already infers direction —
  L4 flipped=True, L5 flipped=False, perception gate trusted on both).
  Exact regression: r4 member reproduces fixture holdouts to 10 decimals.
  L4 smoke: 40/40 pours load; knob members differentiate (block leads).
RESUME ORDER: (1) generalize `dynamics_fit.row_flash` to wall-band flash on
rows OR columns + EXACT label regression on all 5 old corpora; (2) add the
wrap-rule knob family (immediate vs run-to-end) to FlowTrail; (3) perception
gate re-run over ALL L4/L5 pours (should be clean now); (4) step 5 arranger
{8,9,15} + L5 vertical-unit parse + arranger_gate L4/L5; (5) `r5_fit.py`
full grid (r4 corpora + l4_calib + l5_calib; holdouts; L0-L3 no-regression;
fold env-graded win pours per Q1) ; (6) dw_solve L4 sweeps, env decides.
SIDE LANE (user-requested, subagent, done): OTHER ARC GAMES unlocked — 25-game
catalog downloadable WITHOUT scorecard; ls20+ft09 obtained & observed offline
(ft09 win mechanic CONFIRMED by env grading, levels 0→2). Read
`notes/OTHER_GAMES_RECON_20260706.md`; tools in
`synthesis_lab/experiments/other_games/`. Honest, observation-only lane.
honesty_audit at handoff: exit 0 — total 97/46/6, active 85/23/0.
Working tree has the sim3d edits + notes UNCOMMITTED (user decides staging).

## 0-NEXT (2026-07-06 ~21:45, superseded by 0-NOW above): SOLVE L4 LOCALLY FIRST — round-5 lane is ACTIVE
USER RULING: "fix level 4 now locally before the ranked run, figure out
orthogonal flow." The ranked run now WAITS for L4. Read the round-5 Phase-A
entry at the bottom of `notes/DW_LAB_LOG_20260706.md` — it has tonight's L4
findings (flow RISES; win/fail mirrored — lava = TOP-wall growth; color-15 =
MOVABLE blocker, blocks + splits streams like a static obstacle; selected it
renders 9, unselected 15) and the 6-step work order:
(1) verify contact semantics on the other 15 contact pours (scanner:
`sim3d/r5_l4_observe.py --level 4`; deep-read: `--pour N --ticks ... --window ...`),
(2) Phase-A read of the L5 corpus, (3) FlowTrail generic extension
(gravity-direction knob INFERRED not hardcoded + color-15 contact-knob family
{block, turn-L, turn-R, split, pass} — observation selects), (4) palette
manifest + perception gate for 15 (de-grey protocol), (5) virtual_arranger
movables {8,9}→{8,9,15} (selected-15 renders 9 — cold parse ambiguous,
continuity resolves) + arranger_gate L4/L5 rows, (6) r5 full-grid fit
(holdouts + L0-L3 no-regression) then dw_solve L4 sweeps, env decides.
Also this session: VIRTUAL ARRANGER GATE-CERTIFIED (see UPDATE in the block
below) — the ranked-run enabler is done, runner build resumes after L4.
Program report (network-viewable): `synthesis_lab/reports/
program_report_20260706.html` on the 8766 evidence server
(http://100.124.153.1:8766/... via Tailscale). ALL FOUR decision points are
now RULED (user, 2026-07-06 session pickup):
  Q1 (L2 split-vs-die member) = flywheel refit FIRST — fold the env-graded
      win-side pours into the r5 fit and let real data break the tie; only if
      the refit still ties do we revisit (options b/c not taken).
  Q2a (ranked budget policy) = GREEDY spend-all: levels are sequential, spend
      until the current level falls; NO-IDLE rule — if the kernel can't
      certify, pour best-imagined candidates anyway, never burn the budget on
      fail-closed no-ops (the old live runs wasted ~400 actions no-op'ing).
  Q2b (scorecards) = SMALL SET, report all: up to ~3 ranked scorecards, every
      run disclosed per lab honesty rules (set shows seed variance; best
      single AND full set both reported).
  Q3 = ranked run WAITS for L4 (user ruling above).
  Q4 (L3 s2026) = RETRY at higher budget, in-lane after the L4 work order —
      either L3 goes 6/6 or a documented negative.
honesty_audit at wrap: exit 0 — total 97/46/6, active 85/23/0.

## 0-PREV (2026-07-07): RANKED RUN — build the continuous-episode runner
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
