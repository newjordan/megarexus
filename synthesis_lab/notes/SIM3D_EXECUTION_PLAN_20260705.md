# SIM3D EXECUTION PLAN — the 3D pour-simulator lane (2026-07-05)

**Status: SCAFFOLDED AND ALREADY PAST THE DECISIVE GATE.** This document is the
complete work order for agents of any capability to finish the lane. Read
`HANDOFF_SIMULATOR_20260705.md` §1–9 first (the goal, the wall, the failure map);
this doc assumes it. Binding contracts in `ROADMAP_WORLD_MODEL_20260705.md` §3
apply unchanged (C1 env-only grader, C2 honesty guards, C4 scoreboard, C5 interfaces).

## 0. What happened tonight (evidence snapshot)

The user's idea: extract a 3D representation of the board's objects and SIMULATE
the pour — literally pour spheres out of the bucket, run pathing/ray queries on the
extruded 2D shapes — instead of asking a neural net to hallucinate the rollout.
Built as `synthesis_lab/orchestration/sim3d/`. Results already on record:

| Check | Result | Artifact |
|---|---|---|
| STAGE S0 decisive 5-arrangement test (§7 of handoff), **stream lane** | **PASS — winner margin +1.000, all 4 FPs −1.000** (real: +1, 0, −1, −1, −1) | `sim3d/fixtures/s0_result_stream.json` |
| 55-pour L0 corpus, win-sign agreement, stream lane | **100 %** (all 5 real wins positive, all 50 losses ≤ 0) | `calibrate.py --report` |
| STAGE S0, **particle lane** (uncalibrated baseline) | FAIL (all five lava — spheres leak like true liquid) | `sim3d/fixtures/s0_result_particle.json` |
| honesty_audit before this report | ran, no new flags (this lane adds no shards/ckpts/imitation lineage) | terminal log |
| **STAGE S3 in-budget L1 solve, sim-guided** | **SOLVED@1 — 4,848 imagined evals, 0 search pours, FIRST real pour wins (1 of 4 budget). The winning genome is NEW (≠ the lava_search winner). Independently re-confirmed env win.** | `sim3d/fixtures/s3_l1_result.json` |

No training anywhere. The simulator is hand-coded forward physics; every rule was
derived by tick-matching **legal** L0 calibration pours (never the t0 five). This is
the first imaginer in the program's history that separates the verified L1 winner
from v52's false positives — the exact wall (§4/§5 of the handoff) that killed
v38–v56.

## 1. Why this works where the neural CA drifted (keep these properties!)

1. **Zero drift.** Hand-coded dynamics can't compound learned error over 40 ticks.
2. **Rotation-invariance by construction.** `scene.py` canonicalizes flow-down
   (flips rotated levels like L1). The exact failure that killed v48 (F35) is
   structurally impossible here.
3. **Dynamics-native lava.** The separator is WHERE streams route and hit the
   floor strip (§4 of handoff). The sim computes that directly instead of
   inferring it from statics.
4. **Same scorer as reality.** Sim rollouts are rendered back to SP80-colored
   frames and graded by `pertarget_labels` — the proven STAGE_PT ground-truth
   reader. NEVER build a parallel scoring path; if you need a new metric, add it
   to `metrics.py` as *native* diagnostics beside the rendered-frame labels.

## 2. File map and contracts (do not break these)

```
synthesis_lab/orchestration/sim3d/
  _paths.py        sys.path bootstrap (import first in every module)
  scene.py         frame [64,64] -> Scene: 16x16 macro grid, solids, spout,
                   floor strip, per-target glyph+mouth+capacity, canonical flip.
                   solid_boxes(scene) -> merged 3D AABBs for ray/particle work.
  raycast.py       Lane R HydraulicSim (THE working lane) + ray_aabb/drop_column
  engine.py        Lane P PourSim — 3D sphere PBD physics (user's literal idea);
                   deterministic, seeded jitter table only
  metrics.py       render_frames (sim -> SP80 frames) + score_rollout -> margin
  pour3d.py        imagine_pour(frame, lane=...) — THE public API
  decisive_test.py STAGE S0 gate (run after ANY rule/param change; must stay PASS)
  calibrate.py     --measure / --fit / --report on legal corpora
  s3_solve.py      in-budget solve, imag_search_solve protocol, sim-guided
  viz.py           3-panel PNG (pre-pour | canonical rollout | rendered settle)
  fixtures.py      env fixture dumper (rerun only to EXTEND fixtures)
  fixtures/        present frames, t0 arrangements (EVAL-ONLY), 55 calib pours
```

Contract for both lanes: `run()` returns `snapshots = [(occ [16,16], mouth_fill
[n_targets], lava bool)]` in CANONICAL coords + `mouth_fill`/`lava`/`ticks`.
`metrics.render_frames` un-flips. `imagine_pour(...)["margin"]` is the decisive
score `min_t(fill_t − lavan_t)` read from `pertarget_labels` on the rendered
rollout. Keep `imagine_pour`'s signature stable — search code depends on it.

## 3. Verified mechanics (evidence: fixtures/calib_l0_pours.npz; cite pours by index)

These were established by reading REAL pour animations — treat as ground truth,
extend the list with the same method (tick-by-tick ASCII diffs), never by guessing:

- The game runs on a **16×16 macro grid** (4px cells). Emission = exactly
  **1 macro cell of paint per tick** (`calibrate.py --measure`).
- Streams fall 1 cell/tick, 1 cell wide. Pools widen ~1 cell/side/tick on solid
  support (pour 5, ticks 4→12).
- **"The floor is lava" is literal**: paint touching the boundary strip behind the
  targets (color-1 rows; 60–63 on L0, 0–3 on L1) flashes the whole strip to 14 and
  **ends the pour ~2 ticks later** (pour 5 ticks 16–20; blocked pours plateau at
  ~13 emitted cells because of this stop — pour 0). The per-target lava=54 numbers
  in `t0_pertarget_20260705.json` are exactly `dilated-bbox ∩ flashed strip`.
- On a win the level resets ~2 frames after all-filled (DEVIATION-1; handled by
  `pertarget_labels`, do not re-derive).
- **Column-base shedding (THE core micro-rule, raycast.py §3):** lateral flow
  happens ONLY at a paint column's SOLID-supported base; the column above is a
  reservoir feeding it (base keeps 1.0 cap); full solid-based cells CONDUCT flow
  through to the first non-full cell (under-bar pockets). A column based in a
  mouth/hole first drains, then just STACKS — never spreads over the prong tops.
  This single rule reproduces: pour 5's early leak-while-filling, the L1 winner's
  leak-free three-way routing, and the L0 win's bar-under-source split (pour 6).
- Emission pauses when the spout cell stacks past **2×cap** (a shedding pool AT
  the spout hovers near cap — pour 6 has the bar directly under the source).

## 4. Honesty rules for THIS lane (additions to C1/C2)

1. The five t0 arrangements + their recorded labels are the TEST SET. **Never fit
   any rule or parameter against them.** All fitting: the 55-pour L0 corpus +
   fresh `sp80_variants` harvests (env-graded, legal). After any change, S0 must
   still PASS — if it stops passing, your change is wrong, not the test.
2. Honest caveat to carry into every report: tonight's micro-rules were iterated
   until S0 passed (S0 was used as a go/no-go signal, per the handoff §7 protocol).
   The antidote to accidental overfit is **S2 generalization** below — run it
   before celebrating any further claim.
3. The sim RANKS and pre-filters; the env DECIDES. An "imagined win" (margin +1)
   is never a headline. Headlines come only from `real_win`/gate pours (C1).
4. Lead every report with `python3 synthesis_lab/orchestration/honesty_audit.py`.
5. This lane imports nothing from `_invalid/`, `sp80_spill_*`, or
   `sp80_player_harness` — keep it that way (C2).

## 5. Stage ladder (execute in order; each has a gate; report per §7 template)

### S1 — Fidelity closure on the calibration corpus (stream lane) — **iter 2 ADOPTED; materially advanced**
**Iteration 2 (2026-07-06, `SIM3D_S1_RESULT_20260705.md`):** floor-gated ledge spill
(`HydroParams.ledge_spill_floor=True`, now the DEFAULT) — a bar-lip pool spills over a ledge
at rest ONLY when the drop reaches the floor strip (not a mouth). Full S2 (11,810 pours, both
seed batches): **precision 0.736→0.807, recall 0.992 unchanged, argmax nontrivial 0/2→1/1,
S0 PASS.** The argmax wall that blocked in-budget solves is broken on the S2 corpus. ~28
overshoot-FPs remain (iter-3 corpus). Below = the original iter-1 log.


**Iteration 1 (2026-07-05, `SIM3D_S1_RESULT_20260705.md`):** tick-matched the S2
overshoot-FPs (decisive board 336, `sim3d/s1_tickmatch.py`). DIAGNOSED: FPs flood via a
bar-lip cascade to the floor that the sim misses (its lateral shed moves only over-cap
excess, so 1-deep bar pools rest and paint self-settles at ~23 cells, never reaching the
floor; reality emits 41–50 and floods). FIX TRIED: `ledge_spill` (dribble over open lips
at rest) — fixed 5/8 of board 336's FPs, kept S0 PASS, but REGRESSED the full S2 corpus
(precision 0.559→0.447, argmax 8/9→6/9, +17 FPs). **Reverted.** LESSON: score every S1
change on the FULL S2 sweep, not one board. NEXT (iter 2): the real structure is a
WIN-vs-FLOOD timing race (sim fills the mouth and declares win before the overflow reaches
the floor; reality floods first) — model emission-stops-at-win + a calibrated cascade rate,
gate on full S2 (precision up, recall 1.0) AND S0.

Known gaps to close (all visible in `calibrate.py --report`):
  a. fp0-class mismatch: real fp0 fills ALL targets then flashes (fill 1.0,
     lavan 1.0); sim fills 2/3. Sim margin ordering is already right, but the
     fill pattern is wrong → the flow-split ratios / stop timing need refinement.
  b. Rendered-label windows on lava-stopped pours: real losses sustain fill
     through the flash (sus=80); sim stops within stop_on_lava=2 ticks so
     rendered fill can read 0.00 where native fill is 1.0. Consider extending
     post-lava snapshots (freeze state, emit 2–3 more frames) rather than
     changing pertarget_labels (NEVER change pertarget_labels).
  c. Emission cap: `emit_volume=60` is arbitrary; measure the true source volume
     (does an unblocked pour ever self-stop? design a variant with a bottomless
     route and count).
Method: tick-match harness — for chosen pours, render sim frames and diff against
real animation frames cell-by-cell per tick; fix the first divergence; repeat.
**Gate:** 55/55 win-sign agreement (already there) AND per-target |fill−fill| and
|lavan−lavan| MAE ≤ 0.10 across the corpus AND S0 still PASS.

### S2 — Generalization sweep (the overfit antidote) — **DONE 2026-07-05: GATE PASS, with a sharp caveat**
`sim3d/s2_generalize.py` + `s2_analyze.py`; full result in `SIM3D_S2_RESULT_20260705.md`.
1000 fresh multi-base boards (seed 30000/31000), 11,810 real pours, 118 env-wins.
**Gate PASS:** win-sign 0.9964 (≥0.95), margin AUC 0.9977 (≥0.98), S0 still PASS.
Stored `fixtures/s2_variants_20260705{,_b}.npz`. One code change was needed:
`scene.py` now falls back to the source(4) position when a variant frame lacks the
pre-charged color-6 reservoir (real levels always have 6 → S0 path untouched, re-verified).

**THE HONEST CAVEAT (read before any victory lap):** the win-sign gate is nearly
trivial here (1% win rate → always-loss baseline already scores 0.99), and AUC is
imbalance-flattered. The metrics that matter:
  * **recall 0.992** and **99.6% loss-rejection** → generalization is REAL; the sim is a
    strong FILTER/RANKER and did NOT overfit the t0 five. This is the genuine S2 win.
  * **precision 0.736** (42 overshoot false-positives: sim fills, env floods the floor).
  * **argmax on the 2 boards that actually require discrimination = 0/2** (boards 336,
    269: true winner tied at margin +1 with overshoot-FPs the sim can't separate).
So: the wall is broken for RANKING/FILTERING and for L1 (S3), **NOT** for general
argmax-discrimination on overshoot-contaminated boards. The 42 FPs (esp. boards 336,
269) are the S1 tick-match corpus — fixing the sim's floor-lava/overshoot timing there
should split the ties. Do NOT touch the t0 five.

### S3 — In-budget L1 solve — **DONE: SOLVED@1, seed-robust 5/5 (post-iter2)**
**Seed-robustness (2026-07-06, post S1-iter2):** `s3_solve --levels 1 --emit-volume 37`
across seeds {1234,7,42,101,2026} → **L1 SOLVED@1 on 5/5 seeds** (every one first-real-pour
wins, 4848 imagined evals each). The headline is ROBUST, not a lucky single run, and iter2's
floor-spill did not cost the solve. (Solve did NOT get cheaper in evals — the search is fixed
48×100; a "cheaper" check would lower restarts and re-measure solved-rate — future work.)

`s3_solve.py --levels 1 --restarts 48 --steps 100 --topk 4` → SOLVED@1
(`fixtures/s3_l1_result.json`): 4,848 imagined evals, zero search pours, first
real confirmation pour wins; winning genome `[(3,4,2),(0,1,2),(4,1,2),(2,2,3),
(5,1,3),(7,3,2)]` is a NEW arrangement (not the lava_search one); env win
independently re-confirmed. Protocol was imag_search_solve's, pre-registered:
search pours all IMAGINED; env poured only the final candidates.
**Headline: 2 levels completed in-budget (L0 CNN-alone, L1 sim-guided).** State
the "sim-guided, not CNN-alone" qualifier in every claim — C4's CNN-alone
counter for L1 remains open (that is Stage E amortization / the CNN maximizing
Contra-concept work, not this lane's job).
Remaining S3 work for agents: seed-robustness — repeat across ≥ 5 seeds
(`--seed`), report solved-rate; and re-run after S1/S2 fidelity upgrades to
check the solve gets cheaper (fewer restarts), not just luckier.

### S4 — Level reconnaissance and scene-extractor generalization (L2+)
Before ANY L2+ solve attempt: dump present frames for levels 2–7 (extend
`fixtures.py::dump_presents`), eyeball with `viz.py`, and extend `scene.py` for
whatever appears: sidewalls (L3 turns 4 streams into 2 —
`megarexus-level3-mechanic`), redirect blocks (`liolfvkveqg` sprites — unseen by
this extractor so far; if they deflect rather than split, model them as one-way
lateral feeds or 45° wedges in Lane P), horizontal flows (if source and targets
share rows, canonicalization needs transpose, not just flip). Then run the
level's DEFAULT pour for real once and tick-match before trusting the sim there.
**Gate per level:** default-pour sim margin sign matches env outcome + tick-match
first-divergence ≥ 10 ticks in.

### S5 — Lane P (sphere physics) calibration — **DEFERRED 2026-07-06 (lowest priority, probe below)**
Probe (`calibrate.py --fit --lane particle`): the coarse grid fit **did not finish in 5 min**
(sphere-PBD is expensive) and S0 particle **still FAILs** (winner −1.0, tied with all FPs).
Confirms Lane P is a deep, slow, finicky effort with LOW payoff — Lane R (stream) is the
validated ranker (S1/S2/S3 all built on it) and Lane P is explicitly the fallback. Deferred:
not worth grinding while Lane R passes and the high-value work (proposer/training) is elsewhere.
When resumed: needs rest-contact freezing + backpressure emission (see below), and a faster
particle sim (the fit grid must run in seconds, not minutes) before S0/S1 tuning is practical.

Lane P exists and runs (see `s0_result_particle.json` baseline: FAIL — spheres
leak like true liquid; the game's CA is NOT hydrostatic, see the column-base rule).
Goal: make Lane P match Lane R on S0+S1. Levers in `SimParams`: radius (bigger =
less leaky; try 0.45–0.5 = cell-quantized), friction ↑, per_cell ↓, and if needed
add: rest-contact freezing (spheres sleeping on contact chains), emission only
while spout region has < N spheres (backpressure), stop-on-floor-contact.
Why bother when Lane R passes: Lane P is the shape-GENERAL substrate (arbitrary
geometry, wedges, L4+ shapes the cell CA can't express), the natural home for the
user's ray-tracing/pathing queries in true 3D, and the visual artifact
(`viz.py --spheres`). It is the fallback if S4 meets geometry Lane R can't model.
**Gate:** S0 PASS + S1 corpus ≥ 95 % win-sign, wall-clock ≤ 200 ms/pour.

### S6 — Integration into the deliberation loop — **DONE 2026-07-06 (lane wired)**
`imagine_search_gate.py` now has a `--sim3d [--sim3d-lane stream|particle]` lane (no
ckpt needed) that swaps the neural `Imaginer` for the validated hydraulic sim, guarded
so the neural path is untouched (re-verified: `--ckpt` still required unless `--sim3d`).
Validation: `--sim3d --levels 0,1 --seeds 2` → **L0 agree_acc 1.000, tp=2 fp=0 fn=0**
(perfect win-agreement over 56 candidates), L1 fp=0 (its candidate set had no winner —
the proposer gap). CAVEAT: the A1 paint-mass secondary reads ~4.8× (render artifact —
the lava-strip flash inflates changed-pixel count; NOT the decision signal, which is
`margin>0`). Remaining S6 (NOT done, needs a live loop): the Stage-C portband-seeded
proposer using sim margin as fitness, and the Stage-F flywheel (emit real pours into the
calib corpus). Both need the proposer, which is out of this lane's scope.

Original spec: Wire `imagine_pour` as a lane in `imagine_search_gate` (behind a flag,
like `--ca`): sim = fast RANKER + geometric pre-decider, env = final grader. Then the
roadmap's Stage C proposer (portband-seeded CEM) uses sim margin as its relaxed fitness.

## 6. Open questions / risks (be honest about these in every report)

- **Multi-pour episodes.** Everything here models ONE ACTION5. The budget allows
  4; lock/key goal-seeking (`megarexus-lockkey-goal-seeker`) may require chosen
  pour COUNTS. Sim extension: run `imagine_pour` sequentially on the rendered
  settle? NOT validated — the env's second-pour semantics must be observed first.
- **Emission volume cap** (S1c) and **flow-split ratios** (S1a) are approximate.
- **Redirect blocks and L2+ sprites are unmodeled** (S4). The extractor treats
  every solid color {4,8,9,11,13,14} as inert boxes.
- **The t0-five iteration caveat** (§4.2). S2 is mandatory before any strong claim.
- Lane P determinism depends on the fixed jitter table + stable sorts — keep it
  that way (no Date.now/random anywhere in sim code paths).

## 7. Report template (extends the roadmap's)

```
SIM3D STAGE <S1..S6> RESULT — <date>
honesty_audit: <exit/summary>   t0-five untouched: YES
S0 decisive test after change: PASS/FAIL (must be PASS)
Gate criterion: <verbatim from this doc>  ->  PASS/FAIL
Corpus: <which fixtures/harvests, sizes>
Numbers: <the gate's numbers, before -> after>
New verified mechanics facts (if any): <pour indices + one-line rule>
Next per ladder: <stage>
```

## 8. Quick commands

```bash
cd synthesis_lab/orchestration/sim3d
python3 decisive_test.py --lane both            # the gate (seconds)
python3 calibrate.py --measure                  # emission stats from real pours
python3 calibrate.py --report --lane stream --emit-volume 60
python3 s3_solve.py --levels 1 --restarts 48 --steps 100 --topk 4
python3 viz.py --index 0 --lane stream --out /tmp/winner.png
python3 fixtures.py                             # only to EXTEND fixtures
```
