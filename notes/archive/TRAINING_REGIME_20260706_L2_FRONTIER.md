# TRAINING REGIME ADJUSTMENT — the L2 frontier (2026-07-06)

**This is the pick-up document for the next agent(s).** It supersedes the "next
direction" sections of `SIM3D_LOOP_STATUS_20260706.md` and adjusts the training
regime after the L2 wall. Binding context you must have read first:
`SIM3D_EXECUTION_PLAN_20260705.md` (the sim lane, its contracts, §4 honesty rules),
`SIM3D_LOOP_STATUS_20260706.md` (the wall + its two corrections),
`ROADMAP_WORLD_MODEL_20260705.md` §3–4 (contracts C1–C5, Stages C/E/F).
Standing rules apply unchanged: NO imitation (HANDOFF §0B), evidence contract 1–8
(HANDOFF), sim RANKS / env DECIDES, t0-five untouched, `honesty_audit.py` leads
every report, S0 must stay PASS after any sim change.

honesty_audit at writing: exit 0 — total 97/46/6, active 85/23/0 (no new flags).
Honest headline (unchanged, hardened): **2 levels in-budget — L0 CNN-alone,
L1 sim-guided SOLVED@1, robust 5/5 seeds.**

## 0. The wall, stated precisely (read this before proposing work)

The night loop finished the sim lane (S0–S4, S6 done; S5 deferred) and then hit
L2. The diagnosis went through TWO corrections — know all three states so you
don't regress to a falsified conclusion:

1. **First claim (WRONG):** "L2 is action-space-blocked — 0 winners in 1400+
   arrangements, fundamental wall." This was CONFOUNDED: the sim guiding the
   search mis-modeled L2.
2. **Correction A (extraction):** L2 has **3 paint sources** (cols 1, 9, 14);
   `extract_scene` had collapsed them to one central spout. Fixed —
   `scene.spouts`, one per source blob; single-source levels byte-identical
   (S0/S2 re-verified).
3. **Correction B (fidelity — STILL OPEN):** even with 3 spouts, the multi-source
   sim **under-reports fills**: on real L2 pours where the env fills 1–2 targets,
   sim reports 0. Two causes found: (a) `ledge_spill_floor` was calibrated
   single-source and over-floods 3-source pours — now GATED to single-source
   (kept, safe); (b) the sim counts SETTLED paint only, but the game fills a
   target the tick a droplet ENTERS its mouth ENCLOSED (game source L775) — a
   transient event the settled-only count misses. A "count falling too" hack
   over-filled (36/200 agreement) and was REVERTED.

**Therefore: L2 solvability is UNKNOWN, not walled.** The 0/1400 real-pour fact
stands (random + directed arrangements don't win), but every *guided* search so
far ran on a wrong gradient. The verdict must wait for a faithful multi-source sim.

**The game source is now read** (`environment_files/sp80/589a99af/sp80.py`,
facts in `SIM3D_LOOP_STATUS_20260706.md` last section + memory
`megarexus-sp80-mechanic`): win = ONE pour in which every target mouth is entered
by an ENCLOSED droplet (L775) with zero wall-touch (L815); 4 independent pours,
no accumulation (L522/638/717/730); droplets split around movables (L764); the
real redirect block is `tuvkdkhdokr`, deflects 90° (L787–803). L2's needle: the
middle source (col 9) aligns with NO mouth and must be bar-routed while sources
1/14 stay clean and nothing overshoots.

## 1. FALSIFIED / BANNED — do not spend cycles here again

- **Static pre-pour outcome CNN as decider or separator** — falsified at the
  model (STAGE_PT G1, v55/v56: predicts near-identical fill/lava for winner vs
  FP). No new "predict outcome from the static board" head, ever.
- **Blind/random arrangement search on L2** — 1400+ env-graded arrangements,
  0 wins. Re-running bigger random sweeps is not evidence, it's grinding.
- **Blanket `ledge_spill`** (regressed S2 0.736→0.447, reverted) and
  **mouth-fill = settled+falling** (over-fills, reverted). Both tried; the fix
  bar is the *enclosed-droplet entry* rule, not these heuristics.
- **Multi-pour accumulation strategies** — code-confirmed impossible (filled set
  resets every pour).
- **Variants v1 as-is for anything redirect- or multi-source-related** —
  `sp80_variants.py:27` `REDIR_TAG="liolfvkveqg"` is the PAINT DROPLET, so v1
  variants never contained a real redirect; and every v1 variant is
  single-source. These two coverage holes are exactly what burned v48 (F35) and
  the multi-source sim (Correction B): components validated only on v1 silently
  fail on real multi-source/redirect geometry.
- Everything already banned: imitation/route lineage, `_invalid/` imports,
  fitting anything against the t0 five, `sp80_spill_*`, `sp80_player_harness`.

## 2. THE REGIME ADJUSTMENT (what changes and why)

Old regime (implicit): *make the imaginer more faithful, search harder.*
That regime's single-source arm is DONE (S1 iter2 adopted, S2 gate passed,
L1 robust). Its residue — S1 iter3 (~28 single-source FPs), Lane P — is
**low-value polish; do not pick it up by default.**

New regime: **fidelity where the frontier is (multi-source), then a
mechanic-aware constructive proposer, then amortize the search into the CNN.**
Training in this program now means Stage-E/F training: the CNN learns to
PROPOSE from our own honest search's results, and ranker heads retrain on the
growing env-graded pour ledger. It does NOT mean another outcome predictor.

Execute the phases IN ORDER; each has a pre-registered gate; report per §4.

### P0 — Multi-source fill fidelity (the blocker; no training; sim calibration)
The single highest-leverage task. Nothing downstream is trustworthy until done.
- **Corpus:** `sim3d/fixtures/proposer_l2.npz` — 200 real env-graded L2
  arrangements (from `proposer_probe.py`), incl. the reality-fills-[1,0,0] and
  [1,1,0] cases the sim currently scores [0,0,0]. Extend with
  `s1_tickmatch.py`-style tick diffs on a handful of real L2 pours (legal:
  L2 default/probe pours are calibration, not t0 material).
- **The fix to implement (from the game source, not guessed):**
  (a) mouth-fill counts the tick a droplet ENTERS a mouth cell ENCLOSED
  (flanked both perpendicular sides by target) — a transient-entry event,
  distinct from settled occupancy; (b) the pour must not flood-STOP before
  crediting entries that already happened; (c) 3-source→mouth routing must be
  tick-matched (real L2 emits ~54 macro-cells ≈ 18/source; sim emitted 42 and
  flooded early — re-measure per-source emission with `calibrate.py --measure`
  logic on L2 pours).
- **GATE P0 (all four, pre-registered):**
  1. Single-source PRISTINE: S0 PASS byte-identical + full S2 sweep metrics
     unchanged (precision 0.807, recall 0.992, AUC ≥ 0.997).
  2. Multi-source fill agreement on the 200-arrangement L2 corpus ≥ 0.90
     (per-target fill-count match; currently ~0 on the fills).
  3. Tick-match first-divergence ≥ 10 ticks on ≥ 3 real L2 pours.
  4. The `s4_l2_multisrc` FP (sim +1 → env loss) must score ≤ 0.
- **Stop rule:** two failed iterations on the same sub-cause → write up, stop,
  next agent gets the tick diffs. Do not stack heuristics (the reverted pair in
  §1 is the cautionary tale).

### P1 — Variants v2 + multi-source generalization gate (data for sim AND future training)
- Fix `sp80_variants.py`: `REDIR_TAG = "tuvkdkhdokr"`; add a **multi-source
  family** (2–3 source blobs, L2/L4-like geometry) and a **true-redirect
  family**. Multi-base inheritance (the v52 lesson) stays.
- Harvest env-graded pours over v2 variants (legal, agent-generated,
  pre-registered) → `fixtures/s2ms_variants_*.npz`.
- **GATE P1 (S2-MS):** on ≥ 2000 multi-source variant pours: win-sign ≥ 0.95,
  recall ≥ 0.95, precision ≥ 0.70, AND single-source S2 unchanged. This is the
  overfit antidote for P0, exactly as S2 was for S1.
- v2 shards are ALSO the training corpus for P3 — build manifests +
  CONDITION-ready hashes as you go.

### P2 — Mechanic-aware constructive proposer → the L2 attempt (roadmap Stage C, upgraded)
Only after P0+P1 gates pass. Blind CEM is now known-insufficient on L2; the
proposer must use the mechanic:
- **Genome:** per-movable placements (as `s3_solve`), but **construction, not
  mutation**: enumerate/route per-source — assign each of L2's 3 sources a
  target mouth, place bars to route the middle source (col 9) to its mouth
  ENCLOSED, keep sources 1/14 clean, minimize wall exposure. Portband alignments
  seed the assignment (roadmap C2); sim margin (post-P0) is the fitness;
  `proposer_search.py`'s sum-of-clean-fills is the shaped variant if margin is
  too flat.
- **Budget:** imagined evals unlimited; real pours only for final top-K
  (imag_search_solve protocol, as S3).
- **GATE P2 (the L2 unlock):** env-confirmed L2 win within a 4-pour budget →
  **headline moves to 3 levels (L2 sim+proposer-guided; state the qualifier).**
- **Decision branch (pre-registered so nobody grinds):** if the faithful sim +
  constructive enumeration finds NO arrangement with margin > 0 over an
  exhaustive-ish route assignment sweep → that is real evidence L2 is not
  solvable by arrange-and-pour; STOP, write it up with the sweep artifact, and
  escalate to the user (the "intended L2 mechanic" question — e.g. does L2
  expect redirect-block use?). Do not silently fall back to random search.

### P3 — TRAIN: amortize the proposer + flywheel the heads (Stages E/F — this is the training)
This is where "always be training / master the field" now points. All
no-imitation-legal: the teacher is OUR OWN env-graded search, hindsight over
agent-generated episodes under pre-registered conditions.
- **E-lane (`head_proposer`):** additive head, trained to map present frame →
  the P2 search's final population distribution / winning genomes, across v2
  variants + real levels' own search outputs. Gate (roadmap E): proposer-seeded
  search reaches `real_wins ≥ 1` in ≤ half the evals of P2 on a held-out
  geometry. Falsifiable, cheap to gate.
- **F-lane (ranker flywheel):** every env-graded pour (P0 calib, P1 harvest,
  P2 attempts) appends to the pour ledger; retrain the outcome/ranker heads on
  it continuously (encoder frozen; fixed val; no-loss gates vs current). These
  heads pre-filter for the sim; they never decide (C1).
- **Mechanics:** every run = CONDITION.json first (evidence_class,
  assist_env_flags_allowed, hashes), launched ONLY via `train_watch.py`,
  no-imitation guard green, Apollo for GPU (workspace
  `$TRAIN_WORKSPACE`, venv `$LOOM_ROOT/.venv`).
  Commit toolchain before conditions record hashes (contract rule 8 — note the
  sim3d lane itself is currently UNCOMMITTED working-tree; committing it is the
  first prerequisite for any P3 condition).

### Parallel lane (only if blocked on P0–P2): L3–L5 extractor extensions
Sequence from `SIM3D_S4_RECON_20260705.md`: L3 (connected-mouth segmentation +
sidewalls) → L5 (redirects `tuvkdkhdokr`, down-orientation) → L4 (2 sources +
redirect — becomes tractable exactly when P0 lands). Gate per level: default-pour
tick-match first-divergence ≥ 10 ticks. These yield NO headline levels without
P2's proposer — recon/fidelity value only.

## 3. Priority order and what NOT to start

1. **P0** — everything else is downstream of a faithful multi-source sim.
2. **P1** — cheap once P0's rules exist; unlocks P3 data.
3. **P2** — the only lever that can move the headline past 2.
4. **P3** — start E-lane as soon as P2 produces its first search populations
   (even failed searches train the proposer's distribution); F-lane whenever
   the ledger grows.
- Do NOT start: S1 iter3 single-source FP polish, Lane P particle physics,
  another static outcome head, bigger random L2 sweeps, L3–L5 solves before P2.

## 4. Gates summary + report template

| Phase | Gate | Artifact to produce |
|---|---|---|
| P0 | S0+S2 unchanged; L2 fill-agreement ≥0.90; tick-match ≥10; known FP ≤0 | `fixtures/p0_multisrc_fidelity_result.json` |
| P1 | S2-MS: win-sign ≥0.95, recall ≥0.95, precision ≥0.70; S2 unchanged | `fixtures/s2ms_result.json` + v2 shards/manifests |
| P2 | env L2 win ≤4 pours (or pre-registered negative + escalation) | `fixtures/p2_l2_solve_result.json` |
| P3-E | proposer-seeded search: real_wins≥1 at ≤½ P2 evals, held-out geometry | run dir + CONDITION.json + gate JSON |
| P3-F | per-retrain: no-loss vs current heads on fixed val | gate JSONs |

Report template (extends the sim lane's):
```
REGIME L2-FRONTIER <P0..P3> RESULT — <date>
honesty_audit: <exit/summary>   t0-five untouched: YES   S0 after change: PASS
Gate criterion (verbatim from TRAINING_REGIME_20260706_L2_FRONTIER.md §2/§4): -> PASS/FAIL
Corpus/condition: <fixtures, sizes, CONDITION.json sha if training>
Numbers: <gate numbers, before -> after>
New verified mechanics facts: <game-source line / pour index + one-line rule>
Falsified this phase (add to §1 if any): <...>
Next per §3: <phase>
```

## 5. Quick start for a fresh agent

```bash
cd $REPO
python3 synthesis_lab/orchestration/honesty_audit.py            # must lead your report
cd synthesis_lab/orchestration/sim3d
python3 decisive_test.py --lane stream                          # S0 must PASS before+after your work
python3 proposer_probe.py --level 2 --n 200 --out /tmp/p0_check.npz   # reproduce the fill-mismatch
# P0 work surface: raycast.py (HydraulicSim: enclosed-entry fill, flood-stop timing,
#                  per-source emission), scene.py (scene.spouts — already correct on L2),
#                  s1_tickmatch.py (extend to L2 multi-source pours)
```
Key files: `sim3d/raycast.py` (the working stream sim), `sim3d/scene.py`
(extraction, `scene.spouts`), `proposer_search.py` (P2 seed),
`sp80_variants.py` (P1 fix at line 27), game source
`environment_files/sp80/589a99af/sp80.py` (mechanics ground truth).

First hour: reproduce Correction B (sim fills [0,0,0] where `proposer_l2.npz`
env fills [1,0,0]/[1,1,0]), then implement enclosed-entry fill counting, then
run GATE P0's four checks. If you change ANY sim rule, `decisive_test.py` and
the full `s2_generalize.py` sweep run BEFORE you write a single result line.
