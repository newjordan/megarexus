# NETWORK VIABILITY REPORT — learned-physics lane, Stage 3 (2026-07-09, evening)

## Honesty audit (leads every report)
`honesty_audit.py` exit 0 — `summary: AUTONOMOUS_CANDIDATE=97, IMITATION_ASSISTED=46,
INVALID_HEADLINE=6; active: 85/23/0; quarantined: 12/23/6` — unchanged from session
start. Zero banned imports in every new lane file (scan in session log). No route,
human-mined, or imitation lineage anywhere in this lane. All evidence below is
LOCAL/OFFLINE, fresh runs from level 0, labels from env verdicts only, t0-five
untouched (seed 1234 never played). Coding model: **Codex 5.6 Terra** via codex CLI
(session `019f486a-880b-7a80-8f96-f8974fcffb84`); Claude session orchestrated,
reviewed diffs, probed mechanics, ran gates.

## The question asked
Does the network learn level 0, and does that learning hold on all the other
levels? Is the network viable?

## Verdict, one paragraph
**The network verifiably learned level 0 and its physics partially transfers; the
win sense does not transfer yet for lack of higher-level win examples.** With an
input-convention fix and a retrain, imagination-ranked pouring beats a matched
random baseline ~9x at L0 (15.9% vs 1.7% pour win rate; 43.5% vs 6.5% episode win
rate; 200 paired episodes per arm, identical seeds/candidates/budgets). Learned
paint physics transfers to L1 nearly intact (held-out changed-pixel agreement
0.61 at L1 vs 0.64 at L0). But the agent has still never won an L1 pour (0/1,020),
so the win probe has zero L1 positives and ranks L1 candidates blind. Agent-alone
fresh-run level count: **1** (baseline to beat was 2 — not beaten yet). The network
is viable as an L0 solver and as a physics engine; the L1 win wall is now purely a
data problem, which is this lane's designed regime.

## Pre-registered gates (win or lose, as registered in HANDOFF 0-NOW)
- **G1 (loop runs crash-free, full ledger): PASS.** 200+200+400 fresh episodes,
  full pour ledgers, shards + manifests + summaries written. (One crash found
  mid-session in the new L1 walk path — empty-frame response without fallback —
  fixed by Codex, rerun clean.)
- **G2 (ranked beats random at L0, same real-pour budget): PASS on round 2.**
  - Round 1 (stage0 ckpts, raw live frames): FAIL — ranked 2.1% vs random 2.6%,
    imagined-score→win AUC 0.36. Root cause found by tape replay: an
    input-rendering mismatch (below), not model failure.
  - Round 2 (retrained ckpts + self-calibrating render map, fresh seeds 60000+,
    identical seeds across arms): **ranked 87/546 L0 pour wins (15.9%) vs random
    13/778 (1.7%); episodes winning L0: 87/200 vs 13/200.**
- **G3 (fresh-run agent-alone level count; >2 = new headline): 1 — headline
  unchanged.** 400-episode run: 168/400 episodes won L0 (42%); 672 L1 pours, 0 L1
  wins. Honest boundary of the whole program remains 2 levels (sim3d-lane,
  assisted); this lane's pure-learned count is 1.

## Per-level network viability (val episodes only; eval_per_level.py)
Current deployed checkpoints = `runs/merged_v2/*` (trained on 3,770 pours).

| data | level | n | pos | dyn overall | dyn changed | probe AUC | p(win) win-rows | p(win) loss-rows |
|---|---|---|---|---|---|---|---|---|
| stage0+G2r1 (in-dist) | 0 | 367 | 18 | 0.977 | 0.738 | 1.00 | 0.929 mean | 0.046 mean / 0.514 max |
| stage0+G2r1 (in-dist) | 1 | 18 | 0 | 0.889 | 0.470 | n/a | — | 0.011 mean |
| fresh battery (held-out) | 0 | 246 | 26 | 0.961 | 0.639 | 1.00 | 0.932 mean | 0.048 mean / 0.528 max |
| fresh battery (held-out) | 1 | 104 | 0 | **0.932** | **0.612** | n/a (0 pos) | — | 0.013 mean |

Reading: dynamics (learned physics) holds up at L1 on 10x more L1 validation data
than existed this morning — 0.612 changed-pixel vs 0.639 at L0. The win probe is
perfect at L0 on held-out data and **untestable** at L1+ because no L1 win has ever
been observed. L2+ rows cannot exist yet (reachable only by winning L1 forward).
Retraining moved the L0 frontier number 0.6030 → 0.7054 (in-dist) and crushed the
probe's false-positive ceiling (loser max p(win) 0.984 → 0.514).

## Three mechanic discoveries (all from env responses to our own actions — legal)
1. **Pre-pour frames render differently than pour frames** (`9→8` on movables,
   `14→0` pip strip). This silently fed the models out-of-distribution input and
   flattened all scores to ~0.078 (G2 round-1 failure). Fix: the loop now learns a
   per-level render map online from its own pours (majority vote over observed
   live-vs-pour pixel pairs, min support 20 px, zero hard-coded constants) —
   learned `{0:9→8, 0:14→0, 1:9→8, 1:14→0}` in-run.
2. **RESET is a full-game restart** (levels_completed → 0 from any state), not a
   level restart. Discovered by the loop's own first-win runtime check (36
   independent ledger events). Consequence: candidate pools via RESET-replay are
   valid ONLY at level 0; the loop never resets past L0 now.
3. **Per-level action budget ≈ 45 actions** (any action type; specific clicks are
   NOT lethal; RESET refreshes it, but see 2). Probed with three controlled
   experiments at L1. The L1+ walk is now budget-aware (2 visits/pour, undo off by
   default); L1 episodes end via pour budget with the level intact instead of
   thrash GAME_OVER.

## Data flywheel (the lane working as designed)
Diary grew 2,056 → **7,452** env-graded pours (2,140 fresh-run episodes, 404 wins)
in one session, at zero human labeling cost. L1 coverage grew 101 → **1,214** pours
(12x). Round-3 retrain on the full diary (CPU, completed in-session,
`runs/merged_v3/`): dynamics changed-pixel **0.7724** on 751 val rows (was 0.6030
this morning); on the held-out battery val split, **L1 changed-pixel jumped 0.612 →
0.9105** — L1 physics is now nearly as sharp as L0 — and the probe's L0 separation
became extreme (winners 0.9995 mean vs losers 0.0012 MAX; the false-positive
ceiling collapsed 0.514 → ~0.001). The merged_v3 checkpoints are the recommended
deploy for the next L1-win hunt.

## What blocks level 2+ (the next lever, in order)
1. **An L1 win has to be found once.** Random candidates win L0 ~1.7% per pour, but
   0/1,020 at L1 — L1's solution density under the current candidate distribution
   is below ~0.1%. Options that stay inside the lane: bigger/smarter candidate
   tapes at L1 (more moves, multi-component), imagination-guided hill-climb within
   the 45-action budget (scores at L1 do vary slightly), or many more episodes
   (episodes are ~0.5 s).
2. Once a single L1 win exists, the flywheel repeats what it did at L0: probe gets
   positives → ranking activates → win rate compounds → L2 becomes reachable.
3. Dynamics at L1 (0.61) will keep sharpening from the retrain on 12x L1 data.

## Provenance
- Code (all by Codex 5.6 Terra, reviewed): `solve_loop.py` (v6),
  `eval_per_level.py`, `merge_transitions.py` + briefs
  `STAGE3_BRIEF_20260709.md`, `_V2/_V3/_V4_ADDENDUM.md` in
  `synthesis_lab/experiments/learned_physics/`.
- Runs: `runs/stage3_g2_{ranked,random}_20260709` (round 1),
  `runs/stage3_g2r2_{ranked,random}_v6_20260709` (round 2, paired),
  `runs/stage3_g3_ranked_v6_20260709` (G3, 400 eps),
  `runs/stage3_v5_walkfix_smoke`, `runs/stage3_v6_budget_smoke`, diagnostics.
- Evals: `runs/eval_prelretrain_20260709.json`, `runs/eval_postretrain_20260709.json`,
  `runs/eval_fresh_battery_20260709.json`. Datasets: `runs/merged_v2/`, `runs/merged_v3/`.
- Every shard manifest embeds checkpoint/script/brief SHA-256 hashes.

## Boundary
Local/offline evidence only; not a live scorecard, not a leaderboard claim. The
G2 lift is at L0 only. "Levels won from a fresh run, agent alone" — this lane's
number is 1; the program's honest boundary remains 2 (separately-labeled assisted
lane). No claim beyond these.
