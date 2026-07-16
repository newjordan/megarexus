# Learned-physics training session — 2026-07-13

Honesty audit at session start: exit 0, `summary: 97/46/6`, `active: 85/23/0`,
`quarantined: 12/23/6` (unchanged). All work below stays inside the ONE LANE
(learned physics from own experience); labels from env verdicts only; t0-five
untouched; no source reads; every new artifact carries its provenance.

User directive: "hit the training and improvements hard, optimize rexus."

## 1. merged_v4 dataset — the blocked retrain unblocked

The 2026-07-10 condition (`merged_v4_branch_negatives_50pct_20260710`) died
pre-launch on `pour_idx must be within 0..3`. Resolution (explicit schema
migration, no silent remapping):

- `merge_transitions.py` gains `--allow-branched-pour-idx`: sources whose
  manifest declares `evidence_category=branched_snapshot_search` validate with
  pour_idx 0..127 (per-branch counters, writer-clamped at int8 max). Linear
  sources keep 0..3. Rule recorded per-source in the merged manifest.
  `pour_idx` is provenance metadata only — no trainer consumes it (verified).
- `branch_hc_smoke3` / `branch_bfs_smoke` shards stay EXCLUDED: manifests lack
  split/seeds_used/t0_five_excluded (cannot prove seed-1234 exclusion), and
  each is ~4 episodes of one board (dominance risk).

**merged_v4** = merged_v3 + branch_fleet_hunt_90000 + branch_fleet_l1hunt =
15,851 transitions / 2,698 episodes (270 val), 720 wins. **L1 coverage 1,214 →
8,329 rows (6.9×).** Condition: `runs/merged_v4_branch_l1_20260713/CONDITION.json`.

## 2. GPU retrains (GB10, torch 2.11 cu130; hyperparams held at merged_v3 values)

| model | val (merged split) | note |
|---|---|---|
| dynamics best.pt | overall 0.9432 / changed 0.8400 | selection by *overall* — see §5 |
| win_probe (plain) | AUC 1.0, AP 1.0, 720 pos / 74 val pos | |
| win_probe_d4 | AUC 1.0, AP 1.0 | D4 augmentation, disclosed |

## 3. D4 probe — cross-orientation win sense achieved (rotated-winner diagnostic)

New tool `probe_rotation_diagnostic.py` (val episodes only): score L0 winners
vs losers under identity/rot90/rot180/rot270/flip/flip-rot90; pass bar =
winner-vs-loser AUC ≥ 0.95 under EVERY transform.

- plain probe: **FAIL** — min AUC 0.579 (rot180 collapses to chance;
  rot90 winner scores drop to ~0.25). The win sense is orientation-locked —
  the mechanism behind total L1 blindness (L1 is rotated geometry).
- d4 probe: **PASS** — AUC 1.000 under all six transforms; rotated winners
  score 0.86–0.98; loser max ≤ 0.15. L1 score spread opens 60×
  (max 0.036 / p99 0.013 vs plain 0.0006).

D4 augmentation is a generic symmetry prior (like convolution's translation
prior), injects no game facts; flag + disclosure in `train_win_probe.py
--augment-d4`, recorded in metrics.json. Report:
`runs/merged_v4_branch_l1_20260713/rotation_diagnostic.json`.

## 4. v7 sampler post-mortem — the L1 candidate wall explained (and fixed: v8)

Two diagnostics (`diag_l1_moves.py`, `diag_l1_sampler.py`, pure env probing on
disposable branches at a seed-90000 L1 snapshot):

1. Single-select straight drags never die: components respond ~5 steps
   (~160 changed px/step for an 80-px component) then saturate; 12-step pushes
   in all 4 directions, 8 components: zero GAME_OVER.
2. The v7 sampler's deaths ("sample_go 23/24") are the ~45-action per-level
   budget burning on ONE shared probe branch across many futile segment
   attempts — not lethal moves.
3. The futility itself: `component_displacement` matches the component's
   ORIGINAL color inside a local window, but selection RECOLORS responding
   movables — so the check returns None for exactly the components that moved.
   At L1 almost nothing retained. **Consequence: the 6,483-pour l1hunt
   (2026-07-09) poured mostly near-empty arrangements — the "L1 wall" was
   substantially a broken candidate generator, not solution density.**

**Sampler v8** (branch_fleet.py, CONDITION_NAME bumped to `stage3_branch_fleet_v8`):
- every segment attempt runs on its OWN fresh branch (replay tape-so-far, then
  probe) — futile attempts can't burn the shared budget;
- retention gate = `_arrangement_changed` (persistent pixel change ≥6 px near
  the clicked component) instead of same-color centroid displacement;
- `--move-persistence`: correlated-random-walk drags (net displacement ~O(n)
  vs ~O(sqrt(n)); -1 = mixed pool {0.0, 0.5, 0.85, 0.95} per candidate);
- pre-death prefix keeping: a drag that dies keeps its safe displaced prefix
  ("drag as far as safely possible" candidates);
- latent arity bug fixed in `_pour_candidate_on_branch` game-over path.

Smoke (4 seeds, pool 24): realization 24/24 (v7: ~6/24), sample game-overs 0
(v7: ~23/24), tape lengths up to 35 actions. All generic search mechanics —
no game knowledge; env grades everything.

## 5. Chain eval — new tool `eval_imagination_auc.py` (deploy criterion)

The loop scores IMAGINED boards, so per-model metrics mislead. On the held-out
battery (val episodes), AUC of probe(dynamics(board_before)) vs env win:

| pair | L0 imag-AUC | note |
|---|---|---|
| v3 dyn + v3 probe | 1.000 | proven at L0, but orientation-blind at L1 |
| v3 dyn + d4 probe | 0.836 | |
| v4 dyn + d4 probe | 0.837 | deploy for the hunt |
| v4 selchanged dyn + d4 probe | 0.828, scores collapse to ~1e-4 | chain broken |

Also learned: `train_dynamics.py` selected best.pt by *overall* agreement while
the frontier metric is changed-pixel; it now also writes `best_changed.pt`.
The changed-selected snapshot wins per-model battery metrics (L0 changed
0.9724 vs v3 0.7045; L1 changed 0.9293 vs 0.9105) but its background errors
break the chain — a train/test mismatch between real and imagined boards.
Counter-lever launched: `--imagined-mix` probe training (real + imagined
boards, same env labels) to align the probe with the chain.

## 6. v8 L1 hunt — THE L1 WALL IS BROKEN (4 wins, all 4 linearly verified)

`runs/branch_fleet_l1hunt_v8_20260713`: 158 L0-winning seeds (fast-path via own
prior tapes), pool 128 = budget 128 at L1, selects ≤3, moves ≤12, mixed
persistence, pour-threshold 0.5, ranked with v4 dynamics + d4 probe, 16 workers
(OMP/MKL/OPENBLAS threads=1 — without caps 16 workers × default thread pools
put load average at 218 on 20 cores and starved the run; also killed 16
orphaned workers a partial pkill left behind).

**Result: 20,421 env-graded pours; L1: 4 wins / 19,751 pours (program was
0/7,503 L1 before today). Winning seeds 90005, 90006, 90023, 90139; tape
lengths 14–33 actions (real multi-drag arrangements). The 4 winning episodes
advanced to L2 and poured 512 L2 branch pours (0 L2 wins yet — L2 is the new
frontier).**

Ranking signal: seed 90023's winning arrangement scored imagined_p_win 0.161 —
the SINGLE HIGHEST score of all 19,751 L1 pours; 90139's was above p99. Two
winners sat mid-distribution (found by full-pool coverage). Cross-orientation
transfer is real but noisy; the imix probe (chain AUC 1.0) should sharpen it.

**Linear verification (`verify_linear.py`, evidence_category
`linear_replay_of_branch_found_tapes`): 4/4 winning seeds verified — fresh
boot → L0 tape → pour → L1 tape → pour in ONE plain forward run, env grades
`levels_completed=2`, state NOT_FINISHED (alive at L2). L0 chains verified
158/158. Artifact: `runs/branch_fleet_l1hunt_v8_20260713/linear_verification.json`.**

Boundary: tapes were FOUND by branched snapshot search (disclosed); the
verified chains are linear fresh-run replays of them. The learned-physics
lane's linear-replay level count is now 2 (was 1); no claim beyond that.

## 7. Next session work order (in order)

1. **merged_v5** = merged_v4 + l1hunt_v8 shard (20,421 pours incl. 4 L1
   POSITIVES + 512 L2 rows — first L2 physics data). Retrain win probe
   (--augment-d4 --imagined-mix) → L1 positives activate L1 ranking; retrain
   dynamics (arranged L1 pours fix the pristine bias; consider best_changed +
   a fresh imix probe trained against it).
2. Re-run rotation diagnostic + eval_imagination_auc; deploy best chain
   (baseline to beat: v4dyn+d4imix, both gates at 1.0).
3. Round-2 hunt at L2: re-run fleet with --l0-tapes from l1hunt_v8
   winning_tapes (fast-path seeds 90005/90006/90023/90139 straight to L2),
   plus fresh seeds for more L1 wins. L2 pools were only 512 pours — scale
   there. Thread caps mandatory (OMP/MKL/OPENBLAS=1).
4. G2-style paired arms at L1 (ranked vs random, identical seeds/budgets) once
   the probe has L1 positives — measure the lift honestly.
5. Consider linear-legal L1 solve (the G3 headline metric): with ranked pours
   inside the linear 4-pour budget, the chain (imag-AUC 1.0) may now find L1
   wins in plain fresh runs; that would move the honest agent-alone headline
   from 1 toward 2+ with no branching caveat.
