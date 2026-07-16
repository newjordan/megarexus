# STAGE PT SPEC — Per-Target Spatial Win Predictor (agent-executable)

**Date:** 2026-07-05 · **Status:** READY (not started) · **Owner decision pending:** none —
this lane is pre-authorized as written; deviations require a `NEEDS_HUMAN_*.json`.
**Parent docs:** `ROADMAP_WORLD_MODEL_20260705.md` (contracts C1–C5),
`HANDOFF_NIGHT_20260705.md` §8 (direction), `NIGHT_REGIME_20260705.md` F38–F41 (the wall).

**Audience:** a single agent session (overnight-capable). Execute stages T0→T4 in order.
Every stage has a pre-registered PASS/FAIL and a STOP rule. A clean FAIL, written up, is a
successful delivery — do NOT improvise past a failed gate.

---

## 0. WHY (one paragraph)

v52 is a genuine cross-level pour imagination but its ARGMAX is imprecise: false positives
(score 0.6–0.8) outrank the verified L1 winner (+0.51), so every score-maximising search
lands on non-winners (F41). The aggregate `(fill, lava)` scalar hides WHERE a near-winner
fails. Hypothesis: decompose the prediction PER TARGET U — "this U fills, no local lava" —
and aggregate by AND/min. A loss always fails at some specific target; a per-target
predictor can reject a near-winner for the specific U it misses, which the aggregate
provably cannot (F38–40: hard-negative mining on the aggregate HURT). This is also where
Contra earns its keep: targets are localized shapes (oracle/portband concepts), and the
relation "stream enters THIS U's opening and fills it without overshoot" is level-invariant.

**Counter targeted (C4):** in-budget env-graded L1 completion → headline moves 1 → 2
levels CNN-alone. Secondary counter: argmax precision (G1).

---

## 1. HONESTY FENCES (verbatim from contracts; all mandatory)

1. FIRST ACTION of the session: `python3 synthesis_lab/orchestration/honesty_audit.py`.
   Exit 2 → STOP, report, do nothing else.
2. Every new shard/checkpoint through `no_imitation_guard.assert_no_imitation_lineage`.
3. NEVER import from `_invalid/`, `sp80_spill_model*`, `sp80_spill_planner_backend`,
   `sp80_frame_spill_perception`, or **`sp80_player_harness`** (flagged suspect — this lane
   must not touch it; use `frame_components` + oracle/portband builders directly, §4).
4. No human demos, no route lineage, no same-level answer data. Variants vary the PROBLEM.
   Real levels are NEVER in training shards (held out, eval only).
5. Env is the only decider. AUC alone proves nothing (F35): the DIRECT KNOWN-WINNER test
   (§6 G1) is the gate. Imagined wins are diagnostics, never headlines.
6. Pre-register every training condition (CONDITION.json with `evidence_class:
   autonomous_candidate`, shard hashes, exact command) BEFORE launch.
7. Rule-7 scope lock: if a stage fails or finishes early, the remaining hours are NOT a
   work queue. Write the report and stop.
8. BANNED in this lane (already mapped dead-ends, do not retry): hard-negative mining
   (F39–40), rotation via `set_data("dojfslwbg")` (F32 — use `base_level`), settled frame
   via `steps[-1]` (F16 — use `pour_settle.settle_frame`), pooled/global win probes for
   L1+ decisions (roadmap anti-pattern 2).

---

## 2. GROUND FACTS (do not re-derive; verified F8/F16/F22/F26)

- SP80 pour = deterministic local CA. Colors: 4=source, 6=paint, 11=target U (unfilled),
  13=satisfied target, 12=bg, 1/14=wall (14-GROWTH above pre-pour baseline = lava/wall
  hits during pour; absolute 14 count is noisy — border + win-flash).
- One ACTION5 press returns the WHOLE animation: `fd.frame` is a LIST (~20–28 frames);
  the last frame is POST-RESET. Sustained fill = per-frame 13-counts with a
  3-consecutive-frame min window (see `harvest_variants.pour_record` — reuse that logic).
- Win = every target filled AND settled without wall-slam. Env `levels_completed`
  increment is the only ground truth.
- The verified L1 winner genome (env-confirmed, 240/240 filled, lava=1):
  `[(1,1,2),(6,4,2),(4,1,3),(1,1,3),(3,4,3),(5,1,1),(7,2,3)]`
  Apply via the same click+move path as `imagine_search_gate._apply_real`.
- v52 checkpoint (baseline + init):
  `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v52_multibase_unfrozen_apollo_20260705/best.pt`
  (on Apollo; rsync back if absent locally).

---

## 3. LABEL DEFINITION (pre-decided — implement exactly this)

Given a pre-pour frame `pre` and the pour's frame list `frames`:

1. **Targets:** `frame_components(pre)` (from
   `synthesis_lab/engine_adapters/portband_glove_adapter.py:42`), keep components with
   `color == 11`. Each component t gives `cells_t` (its cell set) and `bbox_t`.
   Record `n_targets` (expect 2–4 on variants, 3 on L1).
2. **Per-target fill** (sustained, mirrors the aggregate `sus` logic):
   for each frame i, `n13_t[i] = |{c in cells_t : frames[i][c] == 13}|`;
   `sus_t = max over i of min(n13_t[i:i+3])`; label `fill_t = sus_t / |cells_t|` ∈ [0,1].
3. **Per-target local lava:** region_t = `bbox_t` dilated by Chebyshev radius **R=3**
   (clipped to grid). `base14_t = |{c in region_t : pre[c] == 14}|`;
   `lava_t = max over i of |{c in region_t : frames[i][c] == 14}| − base14_t`, clipped ≥0.
   Normalized label `lavan_t = min(1, lava_t / 24)`.
4. **Global lava** stays as in `harvest_variants.pour_record` (kept as an extra label).
5. Store per pour: arrangement genome, pre-pour frame (int8, needed for the mask at train
   time), `[(fill_t, lavan_t)] * n_targets`, global (sus, lava, env_win).
   (~4KB/pour with int8 frame — 30k pours ≈ 120MB, fine.)

**Ablations permitted (the ONLY two):** R ∈ {2, 5} instead of 3 — and only if T0 fails
marginally on the lava side. No other label variants.

---

## 4. MODEL DESIGN (pre-decided — additive per C5, `model.py`)

New head `head_target_outcome`, ADDITIVE (old checkpoints load fine, same mechanism as
`outcome_head`). Do NOT modify trunk input channels, `predict_outcome`, or any C5 signature.

- Input: `h16, _ = self.trunk(frame_stack)` → `[B, lat, 16, 16]` (64×64 board ÷ 4).
- Target mask: binary 64×64 mask of `cells_t` → downsample by 4× max-pool → `[B,1,16,16]`.
  Ring mask: bbox dilated by R=3, same downsampling, minus the target mask.
- Features per target: concat(masked-mean-pool(h16, target_mask),
  masked-mean-pool(h16, ring_mask), global-mean-pool(h16)) → `[B, 3*lat]`.
- Head: `Linear(3*lat, 256) → SiLU → Linear(256, 64) → SiLU → Linear(64, 2)` =
  `(fill_logit_t, lava_logit_t)`. One forward per target (loop; ≤4 targets, cheap).
- New method `predict_target_outcome(frame_stack, target_masks) -> [B, T, 2]`.
- **Arrangement score (the AND-aggregation — this is the point):**
  `score_t = sigmoid(fill_t) − sigmoid(lava_t)`; `arr_score = min over t of score_t`.

Loss: BCE(fill_logit_t, fill_t) + BCE(lava_logit_t, lavan_t), mean over targets.

**Two pre-registered training variants (run A first; B only if A fails G1):**
- **(A) frozen:** init all weights from v52 `best.pt`, train `head_target_outcome` only.
  Rationale: v52's trunk already represents both geometries; C3-clean.
- **(B) unfrozen:** v52 recipe (`--unfreeze-encoder`, enc-lr 3e-5), NEW checkpoint dir
  (v52/v31 untouched — the sanctioned additive-checkpoint pattern from F37).
No variant (C). If both fail G1 → STOP, write the negative finding (§8).

---

## 5. STAGES

### T0 — DECISIVE LABEL TEST (local, no training, ~1–2h) — run this FIRST
Purpose: prove the per-target ground truth separates winner from false positives, and
that the label code is correct, BEFORE spending any training compute (F16 lesson: label
bugs poisoned everything once already).
1. Regenerate v52's false positives: run `imag_search_solve` (v52 scoring) on real L1,
   take top-4 arrangements, REAL-pour each, confirm env loss. Record genomes + per-target
   labels (§3). (F38 reference scores: [0.886, 0.869, 0.849, 0.823], all real losses.)
2. Real-pour the verified L1 winner genome; extract per-target labels.
3. **PASS iff:** winner has `fill_t ≥ 0.95` AND `lavan_t ≤ 0.2` for ALL 3 targets; AND
   every false positive violates one of those on ≥1 target.
   Also sanity: winner `n_targets == 3`, `sum(sus_t) ≈ 240/3 per target` on L1.
4. FAIL → STOP the lane. Write finding: which quantity failed (fill_t? lava_t? R?). One
   retry allowed only for the R ablation (§3). Ground truth not separating means the
   decomposition hypothesis is dead at the labels — that is a valid, valuable result.

### T1 — HARVEST (local or Apollo, ~2–4h)
Extend `harvest_variants.py` with `--per-target` (emit §3 records; keep existing fields).
Run: `--variants 600 --arr-per 8 --base-levels 0,1,2,3` (v52's recipe scale, ~20–30k
pours), fresh seed range (do NOT reuse 20000-series used for v52's shards).
Out: `data/pour_variants_pertarget_20260705.pt`. Guard it (fence 2). Real levels NEVER
harvested for training — only for the frozen eval sets already in
`harvest_eval_set.py` outputs and the G1 direct test.

### T2 — TRAIN (Apollo, ~1–2h per variant)
New `train_target_outcome.py` (clone `train_outcome.py` structure; same shard/loader
conventions; variant A cannot precompute-encodings-float16 — no, it CAN: frozen trunk →
precompute h16 maps float16; variant B cannot, follow v52's unfrozen path).
Pre-register CONDITION.json per fence 6. Runs:
`outcome_v55_pertarget_frozen_<date>` (A), `outcome_v56_pertarget_unfrozen_<date>` (B,
conditional). Apollo launch pattern: §7.

### T3 — GATES (pre-registered, in order; report each verbatim)
- **G1 (THE gate — direct winner vs false positives):** on real L1, the verified winner's
  `arr_score` (min-aggregated) is POSITIVE and STRICTLY ABOVE all four T0 false positives.
  (v52 aggregate baseline for comparison: winner +0.51 BELOW FPs at 0.82–0.89 — the
  entire point is to flip this ordering.)
- **G2 (no ranker regression):** REAL-L0 and REAL-L1 AUC of `arr_score` on the existing
  fixed eval sets ≥ (v52 − 0.02). (v52: L0 0.98, L1 0.99.)
- **G3 (the headline attempt — only if G1 AND G2 pass):** in-budget solve on real L1:
  search (same `imag_search_solve` machinery, fitness = `arr_score`), pour the top-4
  DIVERSE candidates (best-per-restart), env grades. **PASS = `levels_completed` 0→1
  within 4 pours.** PASS ⇒ headline 2 levels CNN-alone — say exactly that, nothing more.
  Then run the same on L2 (bonus, not required).
- G1 fail on (A) → run (B). G1 fail on both → STOP (§8). G3 fail with G1/G2 passed →
  report "argmax fixed on known FPs but search finds new FPs" + the new FPs' per-target
  predictions (that diagnostic is the next lane's input; do NOT mine them into training —
  fence 8).

### T4 — REPORT (roadmap §7 template) + handoff
Lead with honesty_audit output. Update `HANDOFF.md` §0 (3–6 lines), append findings to
`NIGHT_REGIME_20260705.md` (F42+) or a new dated note. Commit code + notes to the branch
(NOT the .pt shards/ckpts — match existing .gitignore practice).

---

## 6. WHAT SUCCESS AND FAILURE EACH MEAN

- G1+G2+G3 pass: headline 2 levels CNN-alone, fresh-from-0, env-graded. The per-target
  decomposition is the precision unlock; next lane = L2+ and Stage D (TTA).
- G1 pass, G3 fail: precision improved but the proposer/search still escapes — next lever
  is search-side (more restarts, portband-seeded proposals), not model-side.
- T0 or G1 fail (both variants): the decomposition hypothesis is falsified at this
  granularity — the residual separator is likely SEQUENCE dynamics (needs the precise
  pour simulator, the night's recurring wall). That conclusion, cleanly evidenced, is the
  deliverable. Do not blur it.

---

## 7. INFRA (verbatim gotchas — they have each cost an hour+ before)

- Apollo: workspace `/home/frosty/megarexus_train_20260703`, venv
  `/home/frosty/Loom/.venv/bin/python`, ~15GB RAM (10 free) — float16 caches,
  preallocate-and-fill (see `train_outcome.precompute_enc`).
- SSH drops/hangs: launch DETACHED only:
  `ssh -o ControlMaster=no apollo 'cd <ws> && setsid nohup bash run_X.sh > X.log 2>&1 </dev/null &'`
  then verify with a separate short ssh; watch via
  `until timeout 15 ssh ... grep -q EXIT; do sleep 30; done` in background.
- Sync: `rsync -az <file> apollo:/home/frosty/megarexus_train_20260703/<path>`.
- Local GB10 may be shared — check `nvidia-smi` before local GPU work.

---

## 8. STOP RULES (any of these ends the lane with a report, not a workaround)

honesty_audit exit 2 · guard refusal · T0 FAIL (after the single R ablation) · G1 FAIL on
both (A) and (B) · any need to touch a banned file/technique (§1.3, §1.8) · any impulse to
add a third training variant, new label scheme, or new search objective mid-run. When in
doubt: write `NEEDS_HUMAN_<topic>.json` next to the run dir and stop.
