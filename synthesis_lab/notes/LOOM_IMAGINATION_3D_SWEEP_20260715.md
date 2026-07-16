# Loom Imagination / 3D World-Sim Sweep — 2026-07-15

**Purpose:** Inventory of *your* Loom-era imagination concepts so the megamind
base model can load **all of them** as seeds. Testing rises what works; absence
from the base is not acceptable.

**Sources:** `/home/frosty40/Loom`, Loom memory, Loom-gepa-bg scouts.

---

## North-star (memory)

### Active 3D injection + generative imagination
`synthetic_video_pretraining.md`

1. **Per-decision enrichment:** Dreamweaver 3D + raycast + physics probes
   (e.g. bouncing ball, collision color pings) stacked as encoder channels.
2. **Generative imagination:** model produces video of futures and assesses it.
3. **Video pretrain:** secondary only.

Fork: pixel decoder on LeWM **vs** Dreamweaver renderer + LeWM evaluator.

### Simulator self-play
`simulator_self_play_training.md`

```text
imagine rollouts → execute top-K in real env → keep level++ → gold train set
```

### Iterative lab (v0–v10)
`docs/iterative_world_model_lab.md`

observe → latent → simulate → residual → beliefs → experiments → goals → plan

---

## Concept catalog (must seed base model)

| ID | Concept | Loom home | Seed role in base |
|---|---|---|---|
| L1 | Multi-channel stack (frame+ray+3D+probe) | loom_direction Layer 1 | channel presence / density scores |
| L2 | Dual generative path (neural ⟷ structured) | loom_direction Layer 2 | disagreement / confidence |
| L3 | Physics probe library (ball, pings) | synthetic_video memory | probe hypothesis scores |
| L4 | Raycast topology (LOS, reach, block) | iterative lab v4 | topology scores |
| L5 | Simulation hypothesis bank F_k | iterative lab v5 | multi-hypothesis residuals |
| L6 | Bayesian residual selection | iterative lab v5 math | residual likelihoods |
| L7 | Controllability map | iterative lab v3 | control vs world split |
| L8 | Object / sprite perception | MultiSense, v1 | sprite/topo features |
| L9 | Fly-eye multi-horizon | fly_eye.py | multi-scale plan entropies |
| L10 | Soft imagination guidance | imagination_guidance.py | phase latent/action alignment |
| L11 | PUCT informers | lewm/imagination + informers | wall/affordance/path priors |
| L12 | Block-parallel latent rollouts | block_jacobi_rollout.py | rollout depth/cost proxies |
| L13 | Gaussian residual corrector | correction.py, GCD doc | calibration residual |
| L14 | HRM dual-timescale lookahead | hrm memory | H/L cycle deliberation proxy |
| L15 | Dual latent z_env ∥ z_hyp | arc_dual_latent_belief | env vs mechanic posteriors |
| L16 | Simulator self-play gold loop | simulator_self_play memory | gold-tape readiness flags |
| L17 | Local sim search / lookahead | 2026-05-13 local_sim scouts | branch-search density |
| L18 | Cross-game imagination guidance | microbuild imagination scout | global fallback guidance |
| L19 | Goal coordinator O_G | iterative lab v7 | goal posterior mass |
| L20 | Skill modules / transfer | iterative lab v9 | skill bank size |
| L21 | TIMR cycle consistency | TIMR_DESIGN | reversible vs irreversible |
| L22 | Meta-horizon anti-myopia | META_HORIZON | short vs long score |
| L23 | Dreamer-style dream hours | dreamer_v2 scout, wishlist | dream budget |
| L24 | Pixel decoder path | Layer 2 fork A | decoder presence |
| L25 | Dreamweaver render path | Layer 2 fork B | DW render presence |
| L26 | SAM / sprite objectness | MultiSense / SAM notes | object mask density |
| L27 | Probe animation channels | synthetic_video | probe channel flags |
| L28 | Confidence from dual-path agree | Layer 2 cross-check | agree score |
| L29 | Action-conditioned latent F | LeWM core | pred error |
| L30 | CEM/MPC over imagination | rex, cem_discrete | plan cost |

---

## Integration rule (base model)

```text
ALL idea IDs above MUST appear as named channels in megamind dream_summary
with status ok|degraded and numeric scores. Never omit an idea from the pad.
Training / eval rises weights; the garden stays planted.
```

**Artifact:** operators live under
`synthesis_lab/orchestration/rdl/brain/operators/loom_imagination.py`
and are registered in `hemisphere_i.OP_FAMILIES`.
