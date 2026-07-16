# Rexus Concept Sweep — 2026-05/06 archive into base model

**Source tree:** `$REPO`  
**Doctrine:** model is ground; testing rises what works; **all ideas planted**.

---

## What Rexus is

Fresh transfer (2026-05-25) of protected human data + base LeWM/Rexus
architecture + the full **CONTRA/SP80 lab** (`codexlab/`) out of polluted Loom.

Prime loop (canon):

```text
S_t → encode/memory → candidates A_k → O_pred_k → State Prober
    → scorer → action → env → O_true / scorecard
```

---

## Concept catalog (must seed base model)

| ID | Concept | Rexus home |
|---|---|---|
| R1 | Fail-closed admission policy | ADMISSION_POLICY.md |
| R2 | Paper-faithful LeWM chassis | architecture/ + base_checkpoints |
| R3 | Rexus 12-step action ownership | canon_architecture / Loom transfer |
| R4 | O_pred candidate transitions | GRAM++ / opred experiments |
| R5 | State Prober O_pred vs O_true | canon |
| R6 | GRAM++ pretrain gate surfaces | architecture docs |
| R7 | LPRM process reward | GRAM_LPRM |
| R8 | Full CONTRA system (not partial) | AGENTS.md CONTRA rule |
| R9 | Config + Oracle + View passes required | AGENTS.md |
| R10 | Action-state tags (generic, not paint-only) | AGENTS.md PortBand note |
| R11 | Action interaction graph | portband / cross_game probe |
| R12 | Action interaction matrix | same |
| R13 | PortBand fleet / lineup / edge-lock | portband.py |
| R14 | Structural sense (floor/wall abstain) | contra_structural_sense.py |
| R15 | Shape fit | contra_shape_fit.py |
| R16 | Fill sense (if present) | contra_fill_sense if any |
| R17 | Ghost relational memory | ghost.py + architecture doc |
| R18 | Crucible mechanic induction | crucible/* + plan doc |
| R19 | Source-edge outcome model | crucible_build_source_edge* |
| R20 | Activation transitions | crucible_mine_activation* |
| R21 | DeltaBand controlled before/after | deltaband + controlled_run |
| R22 | ExploreBand budget | exploreband.py |
| R23 | ActionModBand | actionmodband.py |
| R24 | Match calibration stage0/1 | match_calibration/ |
| R25 | Relationship ladder | relationship_ladder.py |
| R26 | Episode memory | episode_memory.py |
| R27 | Tabula rasa wipe | tabula_rasa.py |
| R28 | View system multi-pass | view_system.py |
| R29 | Action semantics ledger | action_semantics_ledger.py |
| R30 | Teacher/student signals (Opus) | CONTRA_TEACHER_STUDENT* |
| R31 | Human goal-discovery calculus | human/human_goal_discovery* |
| R32 | Human win-bank (eval-only fence) | human/win_bank* — **fence not train** |
| R33 | Per-game mechanic decode (25) | reference/arc_game_mechanic_decode* |
| R34 | Mechanic family clusters | mechanic_clusters.json |
| R35 | SAM hypothesis / admission gates | sp80_sam_* |
| R36 | Paint control skill | paint_control_skill.py |
| R37 | PufferLib RL loop | pufferlib_* |
| R38 | Cross-game isolation / color board | cross_game_* viewers |
| R39 | Angel / gameplay report SOP | CONTRA_GAMEPLAY_REPORTING_SOP |
| R40 | Groundhog paintflow conditions | conditions/*groundhog* |
| R41 | LoRA SP80 adapter | sp80_lora_adapter.py |
| R42 | Control drift tracker | sp80_control_drift_tracker.py |
| R43 | Recommendation optimum | recommendation_optimum |
| R44 | Goal localization | sp80_goal_localization* |
| R45 | Quarantine fence (spill) | spill FENCE scores only |
| R46 | Like-control prior | contra_verify_like_control_prior |
| R47 | Macroband object ranking | macroband verify |
| R48 | Live suite / scorecard runners | arc_*_suite_runner |
| R49 | Contra service contracts | contra_service/ |
| R50 | Players / primitives contract | players/ |
| R51 | Latent distance bridge | lewm_bridge/ |
| R52 | Corpus alignment (sp80 liquid pour) | corpus/sp80_liquid_pour |
| R53 | Novel systems inventory (human) | NOVEL_SYSTEMS_INVENTORY.md |
| R54 | Oversight layer | oversight.py |
| R55 | Perception layer | perception.py |
| R56 | Controls mapping | controls.py |
| R57 | Debug summary packet | debug.py / contra_debug_summary |
| R58 | Boundary pass queues | BOUNDARY_PASS_SPEC |
| R59 | Human replay L1 testbed | contra_human_replay_l1_testbed |
| R60 | Fail-closed not-prediction flags | controller response truth flags |

---

## Integration rule

Every `R*` ID must appear in megamind `dream_summary` as a named channel with
scores. Spill/win-bank appear only as **fence** channels (never labels).

**Code:** `synthesis_lab/orchestration/rdl/brain/operators/rexus_concepts.py`

---

## Planted status (2026-07-15)

| Item | Status |
|---|---|
| `REXUS_IDEA_IDS` R1–R60 | in `operators/rexus_concepts.py` |
| `OP_FAMILIES` | **127** (= physics/legal/contra + Loom 30 + Rexus 60) |
| `EXTRA_FEAT_DIM` | **2032** (127 × 16) |
| `_try_operators` step 4 | `run_all_rexus_seeds` after Loom |
| smoke_all_ideas | **Rexus 60/60 ok**, coverage **127/127** |
| smoke_full_stack | **PASS** (contra 20/20, core ok) |
| CentralConfig.extra_feats default | 2032 |

```bash
CUDA_VISIBLE_DEVICES= python3 -B \
  synthesis_lab/orchestration/rdl/brain/smoke_all_ideas.py
# ALL IDEAS IN BASE MODEL — garden complete (Loom + Rexus)
```
