# MegaRexus — Turning Contra into a trainable CNN (SP80). Technical report for external analysis
**Date:** 2026-07-05 · **Author:** overnight training loop · **Status:** live experiments in flight

---

## 0. What I need from you (the analyzing model)

This is an honest reinforcement-of-a-world-model problem with a hard no-cheating
constraint. We have hit a specific, well-characterized wall and are integrating
ideas from **AdaJEPA** to get past it. Please pressure-test the plan below.
Specifically:

1. Is a **pixel-rollout world model** or a **JEPA latent world model** the right
   substrate for imagining a *deterministic cellular automaton* (the game's "pour")?
2. Is **test-time adaptation** (AdaJEPA's core) worth it inside a **4-attempt**
   budget, and is our honesty fence around it sound?
3. How do we honestly generate/rank **winning arrangements for levels 1–5**, where
   *no solved example exists and no teacher can produce one*?
4. Is the whole **"world model + MPC + geometric win-check"** architecture correct,
   or is there a better honest path we're missing?

AdaJEPA (for your reference): **https://github.com/agentic-learning-ai-lab/adajepa**
— "AdaJEPA: An Adaptive Latent World Model," Wang, Bounou, LeCun, Ren (ICML 2024).
Underlying concept: **JEPA (Joint-Embedding Predictive Architecture)**, LeCun's
non-generative self-supervised framework — predict in *latent/embedding* space, not
pixels. See also I-JEPA (arXiv 2301.08243), V-JEPA, and LeCun's "A Path Towards
Autonomous Machine Intelligence" position paper for the JEPA + world-model + MPC
"H-JEPA" vision that this project has independently converged on.

---

## 1. The goal & the hard constraint

**Goal:** the user has a hand-engineered symbolic game-playing brain called
**Contra** (with subsystems *oracle*, *portband*, *ghost*). They want its
competence expressed as a **trainable CNN** — "turn Contra into a CNN." The user
understands Contra deeply but not neural nets; I own the ML.

**The hard constraint (non-negotiable, the user has discarded ~500 prior attempts
over violations):**
- The **environment is the only source of truth**. A result only counts if the env
  grades it.
- **No imitation of answers/routes.** No copying solution traces, no replaying
  human/mined completions, no injecting solved states (`on_set_level` farming to
  memorize a level is banned).
- **No reward-hacking.** A prior "spill model" that hard-coded wins from privileged
  sprite state was quarantined.
- **Contra is a teacher/informer, not an oracle of answers.** It may inform the CNN
  from *pixels* (no privileged state); the env decides what's actually learned.

Everything below is designed to stay inside this fence. The recurring failure mode
to guard against is a method that "works" only because it smuggled the answer in.

---

## 2. The game (SP80), mechanics decoded from source

A 6-level (0–5) puzzle. A **source** emits paint that **pours** downward; the player
first **arranges movable blocks**, then triggers the pour. Decoded from the
(obfuscated) game source `environment_files/sp80/589a99af/sp80.py`:

- **Movable block** (`plzwjbfyfli`): a falling paint stream hitting it **splits**
  sideways (spreader).
- **Redirect block** (`tuvkdkhdokr`): a stream hitting it **deflects 90°** (the
  "catch-and-redirect" mechanic).
- **Target "U"** (`repwkzbkhxl`, rendered color 11): **filled/satisfied** when a
  stream nestles into it (recolors 11→13).
- **Wall** (`waoewejnqzc`): a stream hitting it = **FAIL/overflow** (recolors →14,
  sets a fail flag).
- **Win** = *every* target U satisfied **AND** *no* wall hit.
- **Attempts:** up to **4 pours per level** (`lyremoheq >= 4 → lose`), plus a global
  ~50-action step budget.

**Critical property:** during the pour ("spill" mode) the game **ignores player
input** — the pour runs as an **autonomous, deterministic, local cellular
automaton**. Therefore **the outcome is fully determined by the pre-pour
arrangement.** L1+ are hard *only because winning arrangements are rare in a large
arrangement space*, not because they need reactive control. (This corrected an
earlier hypothesis that L1+ needed during-pour action — it does not.)

---

## 3. The model (MegarexusCNN)

~4.0M-param CNN. **Input:** `[B, 44, 64, 64]` = 2-frame stack × (16 one-hot color
planes + 6 Contra "perception" planes). The 6 Contra planes are the **informer
channel** — Contra's oracle/portband/ghost outputs fed as extra input planes, all
truth-boundaried (derived from pixels, no privileged state). Heads:
`dynamics` (next-frame prediction / world model), `change`, `catch`, `settle`+`level`
(pour-outcome), `control`, `inverse`, `goose_policy`(BC), `goal`.

- `forward_dynamics(frame, action) → next_frame_logits [B,16,64,64]` — the **pixel
  world model** (predicts the next grid one step ahead).
- `forward_settle(frame) → (settle_mask, level_logit)` — the **pour-outcome head**:
  given a *pre-pour arrangement*, predict whether a pour completes the level.
  (Label semantics verified: the level-up label attaches to the pre-pour frame of
  the pour transition, so this head is a legitimate "would this arrangement's pour
  win?" oracle.)

---

## 4. The capability map (what is proven, with evidence)

All numbers are from env-graded held-out evaluations, fresh-from-0 training on
Contra-guided fan-out over all 6 levels (run "v31").

| Capability | Status | Evidence |
|---|---|---|
| **Perception** (click localization, level-up detection, 1-step dynamics) | ✅ solid | click 0.6 px error; level-up true-positive 3/3; changed-cell acc ~1.0 |
| **Reactive policy** (feedforward action selection) | ❌ fails | 0/18 completions across: hand-coded policy, Behavioral Cloning of Contra (×4 data scales), and DAgger |
| **Move-imagination** (imagine k steps of block motion) | ✅ good | 6-step autoregressive changed-cell acc ~0.9; errors don't compound |
| **Pour-imagination** (imagine the spill CA to settle) | ❌ fails | v31 undercounts paint (imagined ~16 vs real ~96 cells); on the end-to-end gate it **misses 2/2 real L0 wins** (see §6) |
| **Pour-outcome head** (score an arrangement) | ⚠ partial | saturation bug fixed (was 1.0 everywhere), but overfit to ~23 positives (fresh-arrangement AUC ~0.79) → retraining on 3× positives (v37) |

**Headline finding:** a **feedforward** CNN reproduces Contra's *perception* but not
its *deliberation*. Behavioral cloning of a deliberative policy into a reactive net
loses the deliberation (and suffers covariate shift). Contra's own teacher ceiling
is **L0 only** (it completes level 0 but not 1–5). So imitation cannot exceed L0,
and the CNN was below even that.

---

## 5. The honest deliberation architecture (our answer to §4)

Since reactive/imitation fails, the honest path is **neural deliberation**: use the
CNN's own world model to *imagine and score* candidate plans, MPC-style. This is
literally "turn Contra's deliberation into a CNN," and it maps onto Contra's
subsystems:

- **oracle → which/where the target Us are** (goal).
- **portband → how paint routes** (the learned spill dynamics / world model).
- **ghost → the settled shape / relation memory** (the win-check).

**Planner (`imagine_search_gate.py`):**
1. **Propose** candidate arrangements (structured block shifts + random arrangements
   drawn from the same distribution that yields real wins).
2. **Imagine** each pour: autoregressively roll `forward_dynamics` (one ACTION5 then
   ACTION0 ticks) until the imagined frame settles.
3. **Geometric win-check** on the imagined settled frame: *all targets recolored
   11→13 AND no wall recolored →14.* This is a hard, non-learned check reading the
   game's own color semantics.
4. **Pour for real** on the best-scoring arrangement; **env grades** it.

**Why this can reach L1–L5 with zero L1+ training labels:** the spill CA rules are
**level-invariant**. If the world model imagines the CA accurately, the geometric
check identifies a winning arrangement on *any* level — no per-level "positive"
examples required. The two remaining sub-problems are (a) **imagination fidelity**
and (b) **proposer quality** (winning arrangements are rare, so the candidate set
must contain one and imagination must rank it into the top-4 pours).

---

## 6. The measurement that grades everything (`imagine_search_gate`)

For each candidate arrangement it records **both** the *imagined* win-score and the
**real** env outcome (reset, re-apply the arrangement, real pour). It then reports
whether imagination *predicts* reality, and whether the candidate set even
*contains* a real win (separating "search too small" from "imagination too weak").

**v31 baseline (L0, 3 seeds, 129 candidates):**
- `real_wins = 2` (a winning arrangement exists in the set for 2/3 seeds) → **search
  space is adequate**.
- `imagined_wins = 0`, `false_negatives = 2` → **the world model misses every real
  win**. This is the pour-imagination wall, measured end-to-end on the actual
  decision.

This `fn` is the number the next experiments must drive down.

---

## 7. The L1+ wall (the crux)

- **No legal source of L1–L5 completing-pour examples.** Random arrangement completes
  L0 ~3% (46/1600) but **L1/L2 = 0/1600**. Contra (the teacher) also completes only
  L0. So neither student exploration nor teacher can produce an L1+ positive.
- Therefore **no supervised pour-outcome head can learn L1+** (nothing to fit).
- **The only honest way through** is §5's imagination + geometric check (needs no
  positives), which reduces L1+ to two solvable-in-principle problems:
  1. **Imagination fidelity** on the CA → §8 experiments.
  2. **Proposer quality** → winning arrangements are rare; random search will rarely
     contain one for L1+. Candidates: (i) improve Contra's *portband* flow-router to
     propose arrangements; (ii) a learned proposer; (iii) larger/guided search.
     *This is the open question we most want outside input on.*

---

## 8. Experiments in flight + the AdaJEPA integration

**Compute:** single RTX 4080 ("Apollo"). Runs are sequential, auto-chained
(a watcher launches the next the instant the GPU frees).

**v37 (running, ~epoch 32/40):** retrain the pour-outcome `settle`/`level` heads
on 3× the L0 positives (~69) to fix the overfit. *Tests:* does fresh-arrangement
AUC beat v36's 0.79? (L0 discrimination for a learned scorer.)

**v38 (queued behind v37):** retrain `forward_dynamics` (decoder only; encoder
frozen to preserve perception) on **spill-tick transitions** — dense in a new 146k-
step pour harvest that was under-represented before. **This is the decisive test:**
was the pour-imagination failure just *under-exposure* to spill dynamics? *Graded
by* whether `imagine_search_gate` `fn` drops / `tp` rises on v38.

**AdaJEPA — integrated "where effective," staged by risk:**

| AdaJEPA idea | Where it plugs in | Status |
|---|---|---|
| **Test-time adaptation** (few grad steps/replan) | The **4-attempt** budget: on a missed pour, adapt `forward_dynamics` on the *observed* spill, then replan. Works with current pixel dynamics. | **Built** (`tta_search_gate.py`); plumbing validated; efficacy test pending v38 base |
| **JEPA latent prediction** (predict latent, not pixels) | Replace the pixel-rollout in `imagine_pour` with a latent predictor + a small learned win-probe. Directly targets the compounding-pixel-error failure. | **Gated on v38** — build only if pixels still fail |

**TTA honesty fence (important):** adaptation trains the *dynamics* on the env's own
response to *our* action (learning physics), never on an injected answer. The gate
**resets the model to init weights per seed** (no cross-episode memorization) and
runs a **no-adapt control arm** on identical seeds/candidates, so any gain is
provably within-episode physics correction — not level-memorization (which would be
the banned `on_set_level` farming).

**TTA smoke (v31, inconclusive by design):** NO-ADAPT 0/2, TTA 0/2 — because the
small candidate set contained no winner and v31 can't rank one anyway. It validated
the machinery (adapt loss computes at CE≈0.40, control arm works) but the real
efficacy test needs v38's imagination as the base + a larger candidate set.

---

## 9. Open questions for you (the analysis we want)

1. **Pixel vs latent world model for a deterministic CA.** Our CA is discrete,
   local, and fully deterministic. Pixel-rollout compounds error over ~40 ticks
   (measured). Would a **JEPA latent** predictor (predict a settled-state embedding,
   score with a learned "all-Us-filled / wall-hit" probe) be materially more robust,
   or is a deterministic local CA actually *better* served by a well-trained pixel
   (or even discrete/graph) model? Note the tension: our win-check currently needs
   pixel colors; a latent needs a learned probe (a new failure surface).

2. **Is TTA worth it in a 4-attempt budget?** AdaJEPA claims ~1 grad step/replan
   helps under distribution shift. Our "shift" is *unseen levels with no training
   positives*. Is per-episode dynamics adaptation from ≤4 observed pours a real win,
   or noise? Is our honesty fence (per-seed reset + control arm) the right rig, and
   what would you add to prove it isn't memorizing?

3. **The proposer problem for L1+ (our hardest).** With no solving teacher and no
   positives, how do we get a *winning arrangement into the candidate set* for L1–L5?
   Options on the table: improve the symbolic portband router to propose; learn a
   proposer from imagined outcomes (bootstrapping — honesty implications?); guided
   search using the world model as the search oracle (MCTS/CEM over arrangements).
   Which is most promising and most honesty-safe?

4. **Architecture sanity check.** Is "world model + MPC over arrangements + geometric
   win-check + optional TTA" the right frame, or is there a cleaner honest path
   (e.g., model-based RL with the geometric check as a sparse reward but *no* answer
   injection; amortized inference; energy-based EBM over arrangements à la JEPA)?

---

## 10. Code & artifact map

Repo: `~/megarexus`. Training on Apollo at `~/megarexus_train_20260703`.

- `synthesis_lab/experiments/megarexus_cnn/model.py` — MegarexusCNN + heads.
- `.../train.py`, `.../dataset.py` — training + label construction (spill-tick
  transitions become dynamics targets; pre-pour frame carries the pour-outcome label).
- `synthesis_lab/orchestration/collect_pour_outcomes.py` — pour-outcome data harvest.
- `.../build_pour_shard.py` — lean per-level shard builder.
- `.../probe_pour_oracle.py` — pour-outcome head fresh-arrangement AUC probe.
- `.../imagine_search_gate.py` — the planner + imagination-fidelity gate (§6).
- `.../tta_search_gate.py` — AdaJEPA test-time-adaptation planner (§8).
- `.../fanout_gate.py`, `.../bc_gate.py`, `.../collect_dagger.py` — reactive/BC/DAgger
  gates (the failed §4 approaches, kept as negative results).
- `codexlab/contra_head/` — the Contra symbolic brain (oracle / portband / ghost).
- Night log with full findings 1–9: `synthesis_lab/notes/NIGHT_REGIME_20260705.md`.

**Honesty note for the reader:** any result that headlines a completion must be
env-graded, from a fresh run, with no route-replay / human-mined / solved-state
lineage. Several tools above (`fanout_gate`, `bc_gate`, `collect_dagger`) document
*failures* — kept deliberately, because negative results under the no-cheating rule
are the load-bearing evidence here.
