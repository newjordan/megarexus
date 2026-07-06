# HANDOFF — Night 2026-07-05 (shape-based, lava-aware pour imagination)

Self-contained handoff for a fresh session. Read this first, then
`synthesis_lab/notes/NIGHT_REGIME_20260705.md` (FINDINGS 1-41) for the full log.

## 1. ONE-PARAGRAPH STATUS
We built an honest, Contra-informed CNN that IMAGINES SP80 pour outcomes and generalises
across level geometries. The core objective (user's insight): a pour WINS iff it fills every
target U (POSITIVE) AND avoids walls/"lava" (NEGATIVE = growth of wall-colour-14 above the
pre-pour baseline). **v52** is the deliverable: it correctly scores the verified L1 winner and
generalises to real unseen levels (REAL-L0 0.98, REAL-L1 0.99 AUC). What is NOT done: a
single-episode 4-pour CNN-alone SOLVE of L1/L2 — blocked by imagination precision at the
argmax (false positives outscore true winners). That needs a genuinely new capability, not
more training tweaks.

## 2. THE DELIVERABLE — v52
- Ckpt (on Apollo): `synthesis_lab/experiments/megarexus_cnn/runs/outcome_v52_multibase_unfrozen_apollo_20260705/best.pt`
- Model: `MegarexusCNN` with the additive `outcome_head` (predict_outcome -> [B,2] =
  fill_logit, lava_logit). See `synthesis_lab/experiments/megarexus_cnn/model.py`.
- Win-score for ranking = `sigmoid(fill) - lam*sigmoid(lava)` (lam=1).
- Trained on: `pour_variants_multibase_20260705.pt` (30k pours, bases 0-3 = both source-top
  and source-bottom/rotated geometry) with `--unfreeze-encoder` (the trunk fine-tunes so the
  latent represents rotated geometry — THIS was the key to L1). Real levels held out.

## 3. HOW TO VERIFY (the honesty gold standard)
AUC on a harvested set is MISLEADING (it rides the level-invariant lava correlation; see
FINDING 35 where v48's "L1 AUC 0.95" was an artifact). ALWAYS test the KNOWN winner directly:
the verified L1-winning arrangement is genome
`[(1,1,2),(6,4,2),(4,1,3),(1,1,3),(3,4,3),(5,1,1),(7,2,3)]` on level 1 (env-confirmed
levels_completed 0->1, 240/240 filled, lava=1). Apply it (frame_components centers -> click
+ moves), score with predict_outcome, and check fill high / lava low / score positive, and
that it ranks above random L1 arrangements. v52: fill 0.56, lava 0.045, score +0.51, above
97%. (v48 scored it fill 0.003 = a total loss — the over-claim tell.)

## 4. INFRASTRUCTURE (+ critical gotcha)
- Training box: Apollo (RTX 4080). Workspace `/home/frosty/megarexus_train_20260703`.
  venv `/home/frosty/Loom/.venv/bin/python`. Apollo has only ~15GB RAM (10 free) — cache
  encodings as float16, preallocate-and-fill (see train_outcome.precompute_enc), or OOM.
- Sync local->Apollo with `rsync -az <file> apollo:/home/frosty/megarexus_train_20260703/<path>`.
- **SSH GOTCHA (cost me an hour):** long-lived ssh sessions to Apollo DROP (exit 255) or HANG
  (exit 124). Short read-only commands (ps/tail/cat) mostly work. To launch training, use a
  wrapper script + FULLY DETACHED launch that survives the ssh dying:
  `ssh -o ControlMaster=no apollo 'cd <ws> && setsid nohup bash run_X.sh > X.log 2>&1 </dev/null &'`
  Then VERIFY with a separate short ssh (retry loop). Do NOT rely on the launch ssh returning.
  Wrapper scripts live in the scratchpad; pattern in `run_v52.sh` etc.
- Watch completion via a background bash `until timeout 15 ssh ... grep -q "EXIT"; do sleep 30; done`.

## 5. KEY FILES (all committed to branch `night-shape-lava-imagination-20260705`)
- `orchestration/pour_settle.py` — CORRECT settled-frame extraction (paint-mass plateau, NOT
  steps[-1] which is post-reset). This label fix unblocked EVERYTHING (FINDING 16).
- `orchestration/sp80_variants.py` — synthesize randomised valid SP80 boards; `base_level`
  param inherits each real level's geometry (incl. rotation). `make_env_variant`.
- `orchestration/harvest_variants.py` — harvest (arrangement, fill, lava, env_win) over many
  variants; `--base-levels 0,1,2,3` cycles geometries.
- `orchestration/train_outcome.py` — trains the outcome head; `--unfreeze-encoder`,
  multi-shard `--variants`. Precomputes encodings float16.
- `orchestration/harvest_eval_set.py` — fixed-level (arr, env_win, sustained_fill, lava) sets.
- `orchestration/imag_search_solve.py`, `v48_solve.py` — in-budget solve attempts (search +
  top-K real pours; `--diverse`). ALL fail on L1/L2 (see §7).
- `orchestration/harvest_hardneg.py` — hard-negative mining (tried; HURT, see §7).
- `orchestration/win_probe_gate.py` — rank+pour gate (`--mode outcome` uses predict_outcome).
- Earlier: `train_win_probe.py` (v47 win-probe, L0 solver), `lava_search.py` (env-feedback
  hill-climb; found the first L1 win), `imagine_search_gate.py`.
- Data on Apollo (`.../data/`, NOT committed — large): pour_variants_multibase_20260705.pt,
  pour_variants_mass_20260705.pt, pour_filllava_l012_20260705.pt, pour_hardneg_20260705.pt,
  pour_ondist_l0_20260705.pt.

## 6. THE MECHANIC (verified)
SP80 is a deterministic LOCAL cellular automaton. Colours: 4=source, 6=paint, 11=target U,
13=satisfied target, 12=bg, 1/14=wall/border. Arrange-then-pour: press ACTION5 once; the
WHOLE animation is `fd.frame` as a LIST (~20-28 frames). `_grid` returns f[-1] = POST-RESET
(bug source). The win is a SEQUENCE property (fill AND settle, no overshoot); NO single frame
determines it (peak-frame all-filled = 0.34 precision; terminal = post-reset). The env's
`levels_completed` increment is the ground-truth win. Levels: L0 source-top; L1 rotated 180
(dojfslwbg=180) so source renders at BOTTOM — this is why source-top-only variants failed L1.

## 7. DO NOT REPEAT (mapped dead-ends)
- Rotation via `set_data("dojfslwbg")` does NOT rotate the visible layout (only hidden flow)
  -> contradictory signal, poisons transfer (F32). Use `base_level` to inherit geometry instead.
- Naive mass scale-up (more/wider variants) HURT real transfer (F33).
- Hard-negative mining (v53 3x, v54 1x) HURT — false-positives are visually ~ true winners, so
  suppressing them suppresses winners (F39-40).
- In-budget SOLVE via search (CEM / hill-climb / diverse) all fail: v52's high-score region
  (0.6-0.8) is false positives that outscore the true winner (0.51). No score-maximising search
  reaches a winner (F41).

## 8. NEXT DIRECTION (proposed, unstarted)
Break the precision wall with a SPATIAL, PER-TARGET win predictor: use oracle/portband
(codexlab: `build_contra_perception_pipeline` + oracle actor/target builders — gives each
target U's bbox + open_direction) to LOCATE each target, then predict per-target
"filled-and-no-lava". Decomposing the aggregate (fill,lava) scalar into per-target predictions
(shape-grounded, Contra-informed) may separate visual near-winners from true winners where the
aggregate can't. This is the honest path the user's shape/Contra emphasis points to.
Alternative: the agent already SOLVES via env-feedback play (lava_search for L1) — if
multi-attempt play is acceptable, that is a legitimate honest solve.

## 9. HONESTY BOUNDARIES (the user has discarded ~500 attempts over violations)
Env is the only truth. No route replay / imitation of answers. Variants = vary the PROBLEM
(never inject the answer); real levels held out; every claim env-graded. Test the KNOWN winner
directly, never trust harvested-set AUC alone. Lead reports with what is VERIFIED vs claimed.
[[megarexus-lab-honesty-rules]] [[megarexus-cnn-deliberation-gap]] [[megarexus-contra-architecture]]
