# Dual-Hemisphere Brain — design (2026-07-15)

**Status:** design + scaffold. Not a scorecard claim.  
**Parent:** REXUS Dream Lattice (RDL).  
**Mining feed:** concept archaeology inventory + medium mine (dual systems).

---

## Verdict on the idea

**Yes — with three hard refinements.** This is not a new-age metaphor; you already
invented half the duals. Wiring them as **two hemispheres + a trainable central
cortex that oscillates** is the right way to make them one brain.

| Refinement | Why |
|---|---|
| **Asymmetric roles, not twin clones** | Two identical nets co-train → mode collapse or one dominates. One side must be **internal (dream/propose)**; the other **external (world/judge)**. |
| **Central cortex is the only open action authority during train phases that use both** | Hemispheres emit embeddings + scores. The center commits. Matches Rexus ownership + CONTRA fail-closed flags. |
| **Central serving LoRA = owned train trunk** | Train LoRA on **LegalBlend / Wide / CentralCortex**. Serving runtimes (e.g. vLLM “Laguna”) are **infra only**, not architecture. Dreamweaver ideas (Dream Kernel, trust gate, chronometric ranking) **are** in scope via ports in `sim3d` / bridge plans. |

### What you already had (do not reinvent)

| Existing dual | Map onto hemispheres |
|---|---|
| Imagined vs observed (V033 / Dream Kernel) | Internal dream ↔ External calibration |
| Ghost vs Oracle | External memory ↔ Internal future hyp |
| Lewis vs Rexus | Goal informer stays out; center owns act |
| Stream ranker vs sphere decider | Internal body plugs (both can feed I) |
| DualLatentBelief z_env ∥ z_hyp | External family vs internal mechanic hyp |
| Fast/slow + HRM H/L | Timescale inside center or I |
| Brake/Gas | Modes of center policy |
| LPRM / reflection judges | External process scores |
| Joint LeWM↔goal oscillation (`alternate`/`weighted`/`two_phase`) | **Training schedule precedent** |
| AdaJEPA online LoRA | **Center adapt step** after act |
| Continual MoE trust router | Central fusion of expert scores |
| Teacher/student PortBand | External teacher ordering vs live student |

---

## Architecture

```text
                    ┌─────────────────────────────────────┐
                    │     CENTRAL CORTEX (train LoRA)     │
                    │  fuse(h_I, h_E, δ) → action / mode  │
                    │  + optional frozen LLM judge token  │
                    └──────────────▲──────────────────────┘
                                   │ h_I, h_E, disagreement
              ┌────────────────────┴────────────────────┐
              │                                         │
     ┌────────┴────────┐                     ┌─────────┴────────┐
     │ HEMISPHERE_I    │                     │ HEMISPHERE_E     │
     │ INTERNAL / DREAM│   watches/judges    │ EXTERNAL / WORLD │
     │                 │ ◄────────────────── │                  │
     │ propose, imagine│   (scores I's plan  │ predict real     │
     │ roll futures    │    vs reality)      │ transitions      │
     │ field+ports     │                     │ LP/wide/legal    │
     │ DW kernel/sim3d │                     │ Ghost+Crucible   │
     └────────┬────────┘                     └────────┬─────────┘
              │                                       │
              └──────────────┬────────────────────────┘
                             │ frames / actions / env
                      L0 ENV TRUTH (only grader)
```

### Hemisphere_I — Internal (dream / propose)

**Job:** Generate candidates and dream their outcomes.

Inputs: trusted KnownMap / frame, CONTRA ports, signed field scores, available actions.  
Outputs:

- `h_I` embedding
- `plan` (action sequence or arrangement genome)
- `dream_next` (imagined next frame or latent)
- `p_win_imagined` (hypothesis only)

Plugs: Dreamweaver-ported kernel/trust + sim3d body/ray, stream/sphere/LP,
PortBand proposers, transcript macros. **Never admits solve truth.**

### Hemisphere_E — External (world / judge)

**Job:** Model what *actually* happens; score Internal honesty.

Inputs: real (frame_t, action, frame_t+1), episode memory, Ghost rows.  
Outputs:

- `h_E` embedding
- `pred_next` from real dynamics
- `judge_I` = how well I’s dream matched E’s prediction / reality
- `surprise`, `irreversible_flag`, `survival_risk`

Plugs: learned dynamics trained only on env transitions, TIMR cycle signals,
DeltaBand, Crucible discrepancy. **Judges with prediction error, not answer keys.**

### Central Cortex — trainable LoRA node

**Job:** The only module that **chooses** under uncertainty during closed-loop train.

Inputs (concat):

```text
[h_I | h_E | h_I - h_E | |h_I - h_E| | judge_I | surprise | legal_features | mode]
```

Outputs:

- action distribution (+ optional click)
- mode: gas | brake | probe | dream_more
- trust weights: α_I, α_E (how much to believe each side this step)

Training:

- **LoRA on** late layers of a small owned trunk (LegalBlend-style or dedicated
  `CentralCortex`). Rank 8–16, AdaJEPA online steps allowed.
- Center is in-repo torch LoRA. Optional vLLM is only a *serve* path for weights,
  not a design dependency. Dreamweaver surfaces may be ported in; do not treat
  CellWave as a solver unless a separate experiment says so.

---

## Oscillation schedule (the “literally oscillates” part)

Three phases per cycle (or per minibatch block). Precedent:
`architecture/src/lewm/joint_training.py` oscillation ∈ {alternate, weighted, two_phase}.

### Phase E — External learns reality (I frozen)

```text
loss_E = dynamics(frame_t, a → frame_t+1) + optional change_acc
```

Only Hemisphere_E (+ its encoder) gets gradients.  
Internal is stop-grad, may still run forward for logging.

### Phase I — Internal learns to dream (E frozen as teacher)

```text
loss_I = || dream_next - stopgrad(E.pred_next) ||   # teacher: world model
       + optional rank loss on env-graded pours when available
```

Internal tries to match External’s world model (and/or real frames).  
External is judge/target, not updated.

### Phase C — Cortex learns to arbitrate (I and E stop-grad)

```text
loss_C = policy_ssl (taken action weak label)
       + disagreement_use (predict which side was right *after* env return)
       + survival (penalize thrash GAME_OVER / pure no-op — legal signals only)
       + AdaJEPA one-step on LoRA after real transitions
```

Only central LoRA params train. Hemispheres provide **fixed features** this phase
(so the center learns *when* each hemisphere is trustworthy).

### Why this oscillation works

- Prevents the classic collapse: one network learns to fool a co-trained twin.
- External always tracks env truth (legal).
- Internal can stay creative (search/propose) without owning the scoreboard.
- Center learns the **meta-skill**: trust dream vs world, gas vs brake.

---

## Internal vs external *states* (explicit state machine)

| State | Who writes | Meaning |
|---|---|---|
| `S_ext` | Env + E | Last real frame, levels, game_over, measured deltas |
| `S_int` | I | Dreamed frames, open plans, potential fields, hypotheses |
| `S_meta` | Center | Trust α, mode, disagreement history, Ada buffer |
| `S_mem` | Ghost/Crucible | Confirmed relations, mechanic ledger (hypotheses) |

Training step always:

```text
S_ext, S_int → center → a → env → S_ext'
judge: compare I.dream(a) vs S_ext' and E.pred(a)
update phase losses; advance oscillation phase
```

---

## Central LoRA: practical choice

| Option | Trainable? | Role |
|---|---|---|
| **A. LegalBlendLoRA / dedicated CentralCortex (recommended v0)** | Yes, full torch | Real train + Ada online |
| **B. Wide dynamics encoder + LoRA head** | Yes | Multi-game body features |

**Recommendation:** v0 = A. Serve and train an owned in-repo checkpoint only.

---

## Honesty / legal fence

- Judge losses use **prediction error and env fields only** (levels, game_over,
  frame change). No spill_model, no fixture wins, no human routes as labels.
- Internal plans are hypotheses; `solve_condition_admitted=False` until env says so.
- Private-game wipe: reset S_mem + Ada buffer on new `game_id`.
- Teacher Opus signals (if used) stay process-supervision only.

---

## Minimal success criteria (pre-registered)

1. **Oscillation smoke:** 3-phase loop runs on synthetic (s,a,s′); losses move
   in the phase that owns params only.
2. **Judge signal:** after real SP80 random steps, `judge_I` correlates with
   actual dream-vs-real pixel error (held-out episodes).
3. **Center SSL:** Phase C change/action prediction beats base rate on legal
   collect rows (same bars as blend LoRA).
4. **No truth leakage:** honesty_audit + mining_gate unchanged.

Headline levels_completed remains env-only and is **out of scope** for v0 smoke.

---

## Scaffold map

```text
synthesis_lab/orchestration/rdl/brain/
  DESIGN.md           # this file
  __init__.py
  hemispheres.py      # Hemisphere_I / Hemisphere_E interfaces + stub nets
  central.py          # CentralCortex + LoRA
  oscillate.py        # 3-phase trainer
  smoke_oscillate.py  # synthetic smoke
```

---

## Build order

1. Smoke oscillation (done by scaffold).  
2. Wire Hemisphere_E to LP/wide dynamics checkpoint.  
3. Wire Hemisphere_I to dream kernel / stream imagine.  
4. Fuse legal features + CONTRA ports into center.  
5. Ada online on real Arcade steps.  
6. In-repo multi-game body plug (wide/LP) when ready.
