# STAGE CA SPEC — Precise Neural-CA Pour Simulator (agent-executable)

**Date:** 2026-07-05 · **Status:** DRAFT (awaiting human OK before any TRAINING run) —
CA0 (no-training baseline) is pre-authorized; CA2+ (training) needs a green light.
**Parent docs:** `ROADMAP_WORLD_MODEL_20260705.md` (Q1 verdict + Stage A2; contracts C1–C5),
`STAGE_PT_SPEC_20260705.md` (the lane this succeeds), `NIGHT_REGIME_20260705.md` F42–47.

**Audience:** a single agent session (overnight-capable). Execute CA0→CA4 in order. Every
stage has a pre-registered PASS/FAIL and a STOP rule. A clean FAIL, written up, is a
successful delivery — do NOT improvise past a failed gate (rule-7 scope lock).

---

## 0. WHY (one paragraph)

STAGE_PT proved (F42–47) that the winner↔false-positive separator is **pour DYNAMICS**, not
static per-target features: a head reading the static pre-pour arrangement predicts near-
identical (fill, lava) for the verified L1 winner and a lava-saturated FP (both ~0.16/0.95) —
blind to the arrangement difference, because what differs is *how the paint streams route*.
The per-target GROUND TRUTH (read from the REAL pour) separates them cleanly (T0 PASS), so the
signal exists; only a model that SIMULATES the routing can recover it. The roadmap's Q1 verdict
is exactly this lever: a **local, iterated Neural-CA** (`spill_ca`, already additive in model.py;
`train_spill_ca.py` exists) that rolls the pour out tick-by-tick and lets the geometric per-
target check read (fill_t, lava_t) off the SIMULATED settle. "The single highest-leverage
architectural change available." Known risk (the night's recurring wall): pixel-CA rollout has
repeatedly DRIFTED (over-paints, ~6× paint mass) — so fidelity is gated cheaply and early, and a
clean STOP escalates to the latent two-tier path (Stage B) rather than grinding.

**Counter targeted (C4):** in-budget env-graded L1 completion → headline 1 → 2 levels CNN-alone,
via a faithful pour SIMULATOR (what the static predictor could not be).

---

## 1. HONESTY FENCES (verbatim, all mandatory — inherit STAGE_PT §1 + roadmap C1–C5)

1. FIRST ACTION: `python3 synthesis_lab/orchestration/honesty_audit.py`. Exit 2 → STOP.
2. Every new shard/checkpoint through `no_imitation_guard.assert_no_imitation_lineage`.
3. NEVER import `_invalid/`, `sp80_spill_model*`, `sp80_spill_planner_backend`,
   `sp80_frame_spill_perception`, `sp80_player_harness`. Use `frame_components` + oracle/
   portband builders + `pertarget_labels` + `pour_settle` directly.
4. No human demos, no route lineage, no same-level answer data. Spill-tick pairs and variants
   vary the PROBLEM. Real levels are NEVER in training shards (held out, eval only).
5. **Env is the only decider.** Rollout fidelity / AUC are diagnostics; the DIRECT winner-vs-FP
   separation (CG1) and the env-graded in-budget solve (CG3) are the gates.
6. Pre-register every training condition (CONDITION.json, `evidence_class: autonomous_candidate`,
   shard hashes, exact command) BEFORE launch. CA2+ ALSO needs the human OK noted above.
7. Rule-7 scope lock: a failed/early-finished stage is NOT a work queue. Write the report, stop.
8. BANNED dead-ends (already mapped, do not retry): **the STAGE_PT static per-target head
   (F42–47 — DEAD)**; hard-negative mining (F39–40); rotation via `set_data` (F32 — use
   `base_level`); settled frame via `steps[-1]` (F16 — use `pour_settle`); pooled/global win-
   probe or one-shot settle as the L1 DECIDER (roadmap anti-patterns / F35).

---

## 2. GROUND FACTS (verified; do not re-derive)

- SP80 pour = deterministic LOCAL cellular automaton (fall / split-on-block / deflect-on-
  redirect / fill-on-U 11→13 / hit-wall→14). WIN = every U filled AND no wall hit. In spill mode
  `step()` ignores input → outcome fully determined by the pre-pour arrangement.
- Colors: 4=source, 6=paint, 11=target U, 13=satisfied, 12=bg, 1/14=wall (14-GROWTH above pre-
  pour baseline = lava). One ACTION5 press returns the whole animation LIST; the win state holds
  only ~2 frames before RESET. **DEVIATION-1 (STAGE_PT, applied in `pertarget_labels`):** read
  fill (2-frame window) + lava over PRE-RESET outcome frames only; strict/harvest values retained.
- The verified L1 winner genome (env-confirmed 240/240, lava≈1) and the FOUR v52 false-positive
  genomes + their REAL per-target labels are in
  `runs/t0_pertarget_20260705.json` (winner: all 3 targets fill=1/lava=0; every FP lava-saturated).
- v52 checkpoint (`runs/outcome_v52_multibase_unfrozen_apollo_20260705/best.pt`) carries the
  additive `spill_ca` head (near-identity init) + `forward_dynamics`. `imagine_search_gate.py`
  already implements CA rollout (`--ca`, `_active_mask`) and legacy `forward_dynamics` rollout.
- `train_spill_ca.py` already trains `spill_ca` (encoder frozen C3; active-front-masked 1-step
  loss; clean-window pairs via `pour_settle`). Pour shards with tick `steps`: `data/
  pour_balanced_tr.pt`, `data/pour_outcomes_train.pt`.

---

## 3. THE DECISIVE METRIC (pre-decided)

Given a pre-pour arrangement, a SIMULATOR produces a rolled-out frame sequence. Read per-target
(fill_t, lava_t) from that sequence with `pertarget_labels` (IDENTICAL code to the real-pour
labels — the simulator must reproduce the real per-target outcome). Two quantities:

- **Separation (CG1):** `arr_score = min_t(sigmoid-free: fill_t − lava_t)` computed on the
  SIMULATED rollout. Winner must be POSITIVE and STRICTLY ABOVE all four T0 FPs. (Direct counter
  to STAGE_PT-G1: −0.58/−0.79 static → must flip positive via dynamics.)
- **Fidelity (CG2):** simulated (fill_t, lava_t) vs REAL (from T0 / a fresh eval set): per-target
  **lava** Pearson r ≥ 0.7 and **fill** MAE ≤ 0.15, AND settle paint-mass ratio sim/real ∈
  [0.8, 1.2] (roadmap A1). Lava fidelity is the crux — it is the overshoot signal the static
  head could not represent.

**Permitted ablation (the ONLY tuning knob, since drift is the known wall):** the anti-drift
lever in CA2 — 1-step vs k-step rollout loss, and active-front mask radius R∈{2,3}. No new
objective, no new label scheme.

---

## 4. MODEL (pre-decided — `spill_ca` already exists; roadmap A2)

Local Neural-CA: `spill_ca = Conv(16→64,3)→SiLU→Conv(64→64,3)→SiLU→Conv(64→16,1)` (5×5 RF),
applied ITERATIVELY on the 16-colour one-hot grid, UPDATING ONLY the active front
(`_active_mask`: dilation r2 of paint(6)+source(4)), copying the rest. Additive; encoder + all
other heads frozen (C3). **Anti-drift amendment (the key change vs the drifting v38–42):** train
the loss on a **k-step rollout** (k≈3–5, scheduled sampling from teacher-forced toward free-run)
so compounding error is penalized directly, not just 1-step transitions. Rollout for scoring =
`imagine_pour` peak-fill logic already in `imagine_search_gate` (`--ca`), but read per-target
via `pertarget_labels` on the rolled-out frames.

---

## 5. STAGES

### CA0 — DECISIVE BASELINE & DRIFT MEASUREMENT (local, NO training, ~1h) — run FIRST
Roll out BOTH existing simulators — v52 `spill_ca` (`--ca`) and `forward_dynamics` (ACTION5→
ACTION0 ticks) — from the winner + 4 T0 FP arrangements on real L1. Read per-target (fill_t,
lava_t) via `pertarget_labels` on each rolled-out sequence.
- Report Separation (CG1 preview) and Fidelity (CG2 preview: sim vs the T0 REAL labels).
- **PASS-to-skip:** if an existing rollout ALREADY separates (winner positive & above all FPs)
  AND fidelity meets §3 → skip to CG3. (Unlikely per prior drift.)
- **EXPECTED:** existing rollout drifts. Record the drift magnitude (paint-mass ratio, lava r).
  This is the cheap first shot and the baseline CA2 must beat. If BOTH rollouts are catastrophic
  AND the k-step-loss ablation is judged unable to help → escalate to §6 (latent path), STOP.

### CA1 — SPILL-TICK PAIRS (local, ~1–2h)
Verify `train_spill_ca.extract_pairs` yields clean-window [pour_idx, settle_idx] tick pairs from
`pour_balanced_tr` + `pour_outcomes_train`. If rotated/L1-like geometry coverage is thin, harvest
additional tick trajectories on multi-base VARIANTS (bases 0–3, fresh seeds; record full `steps`),
real levels held out. Guard (fence 2). Out: `data/spill_ticks_multibase_20260705.pt` if harvested.

### CA2 — TRAIN Neural-CA (local GB10, ~1–2h) — NEEDS HUMAN OK (fence 6)
`train_spill_ca.py` + the §4 k-step anti-drift loss. New ckpt dir `outcome_ca_v57_neuralca_
<date>` (v52 untouched). Pre-register CONDITION.json. If GB10 shared, check `nvidia-smi` first.

### CA3 — GATES (pre-registered, in order; report each verbatim)
- **CG1 (THE gate):** on real L1, per-target (fill_t, lava_t) from the TRAINED CA rollout give the
  verified winner arr_score POSITIVE and STRICTLY ABOVE all four T0 FPs. (Counters STAGE_PT-G1.)
- **CG2 (fidelity):** on a fresh real L0+L1 eval set — per-target lava Pearson r ≥ 0.7, fill MAE
  ≤ 0.15, paint-mass ratio ∈ [0.8,1.2]. Reuse `pt_gates.harvest_real_eval`; labels via
  `pertarget_labels` on both the REAL pour and the SIMULATED rollout.
- **CG3 (headline, only if CG1 AND CG2 pass):** in-budget solve on real L1 — search (reuse
  `pt_gates.g3` machinery) with fitness = simulated-rollout arr_score; pour top-4 DIVERSE; env
  grades. PASS = `levels_completed` 0→1 in ≤4 pours ⇒ headline 2 levels CNN-alone (say exactly
  that). Then L2 (bonus).
- CG1 fail → pixel-CA cannot read per-target lava faithfully → STOP (§6). CG3 fail with CG1/CG2
  passed → PROPOSER problem (roadmap Stage C: portband-seeded CEM), not fidelity — hand off there.

### CA4 — REPORT (roadmap §7 template) + handoff
Lead with honesty_audit output. Update `HANDOFF.md` §0, append `NIGHT_REGIME` (F48+). Commit code
+ notes (NOT .pt).

---

## 6. WHAT SUCCESS AND FAILURE EACH MEAN

- CG1+CG2+CG3 pass: headline 2 levels CNN-alone, fresh-from-0, env-graded — via a faithful pour
  simulator. Next: roadmap Stage C (proposer) then D (TTA).
- CG1/CG2 pass, CG3 fail: fidelity solved; the residual is the PROPOSER (finding L1 winners) →
  roadmap Stage C. A clean, valuable localization.
- CG1 or CG2 fail: the pixel Neural-CA cannot be made faithful enough to read per-target lava
  (the night's recurring drift wall, now rigorously bounded). Escalate to the LATENT two-tier
  path (roadmap Stage B: settle-predictor + win-probe as ranker) or a non-CA simulator. That
  conclusion, cleanly evidenced, is the deliverable. Do not blur it.

---

## 7. INFRA
- Local GB10 cuda works and is fast (batched trunk ~0.1s/64); v52 is local. Check `nvidia-smi`
  before GPU work (shared rig). Apollo optional (see STAGE_PT §7 for detached-launch gotchas).
- Reuse the loop cadence: launch each test/training detached, monitor the log, rest during runs.

## 8. STOP RULES (any ends the lane with a report, not a workaround)
honesty_audit exit 2 · guard refusal · CG1 FAIL · CG2 FAIL (after the single §3 ablation) · any
banned file/technique (§1.3/§1.8) · any impulse to add a new objective/label/simulator mid-run ·
CA2 launched without the human OK. When in doubt: write `NEEDS_HUMAN_<topic>.json` and stop.
