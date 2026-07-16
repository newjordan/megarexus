# Mining Manifest — every system → legal consume

**Schema:** `legal_search.mining_manifest.v1`  
**Rule:** ship **algorithms**, strip **game answers**.  
**Entry:** `legal_search/consume/registry.py`  
**Order of jobs:** [`MINING_THEN_TRAIN.md`](MINING_THEN_TRAIN.md) — **mine first; training is gated**.

Legend:

| Tag | Meaning |
|---|---|
| **KEEP** | Algorithm/interface legal in the general agent |
| **STRIP** | Must not enter legal loop (bindings, priors, answer keys) |
| **RESEARCH** | May exist beside the agent with honest labels |
| **BAN** | Quarantine; never import into agent path |

---

## 1. Competition shell

| System | Source | KEEP algorithm | STRIP / BAN |
|---|---|---|---|
| arc_play_harness | `codexlab/arc_play_harness.py` | Offline/live episode driver, policy registry, env judge | Specialist policies as *default* general agent |
| arc_suite / live runners | `codexlab/arc_suite_runner.py`, `arc_live_suite_runner.py` | Scorecard evidence only | Using live as training label for win scripts |
| environment runner | `arc_agi` Arcade offline | Real env step | Reading `environment_files/**/*.py` as knowledge |

---

## 2. Stochastic Goose / Tufa

| System | KEEP | STRIP |
|---|---|---|
| Goose ActionModel | Online frame-change learning; masked stochastic sample | Hidden per-game seed maps as answer; external-only install confusion |
| `goose_action_adapter` | 4101-way sample | — |
| `tufa_action_policy` | Generic explore policy | Treating weak partial levels as solves |

**Consume id:** `stochastic_goose` → `explore.change_by_repetition.v1`

---

## 3. DreamWeaver / sim3d

| System | KEEP | STRIP |
|---|---|---|
| `genome_search` / hill-climb | Program search + plug-in scorer | Fixture env_win genome banks |
| `direct_grid_candidates` | Short delta enumeration | L4/L5 human strategy tables |
| `imagination_kernel` | Pluggable DynamicsHypothesis interface | Shipped r4/r5 JSON as permanent SP80 truth |
| `perception_gate` | Round-trip trust gate | — |
| `virtual_arranger` | Rigid-body move operator **if roles induced** | Hardcoded colors 8/9/12/15/6/11 |
| `dw_solve` CLI | Offline research fitter | Ranked path that loads win priors |
| `megarexus_dw_directgrid` | Research specialist (honest label) | Calling it the legal general agent |

**Consume id:** `dreamweaver_sim3d` → `search.imagine_then_rank.v1`  
**Specialist note:** also listed under `specialist_solvers` as non-default.

---

## 4. CONTRA stack

| System | KEEP | STRIP |
|---|---|---|
| `vision` / components | Connected components, edges | Semantic bucket labels |
| `controller_head` / `run_contra_view` | Fail-closed packet pipeline | Any `*_truth_admitted=True` |
| `perception` planes | Feature planes for models | — |
| `statecon` | Working set structure | Engine sprite truth |
| `controller_affordances` | Actor/target/support **candidates** | Admitted actor |
| `symmetry` | Axis scores | Symmetry lock as goal key |
| `view_system` | Multi-pass review views | Report-only paths claiming solve |
| `handoff` | Layer envelopes | — |
| `tabula_rasa` | Wipe bindings | — |
| `trainable_levers` | Online reweight knobs | Frozen SP80 lever tables as truth |

**Consume id:** `contra_head`, `contra_vision`, `statecon`, `controller_affordances`, `symmetry`, `tabula_rasa`

---

## 5. Bands

| System | KEEP | STRIP |
|---|---|---|
| **PortBand** | Edge ports, alignment candidates | Route-flow lock as solve; lineup as action authority |
| **GeoBand** | Spatial pair / boundary queues | Boundary as forced route |
| **DeltaBand** | Numeric before/after deltas | Delta as control truth |
| **ExploreBand** | Budgeted probe queue | Probe queue as solution |
| **ActionModBand** | Preference modulation from evidence | Forced action choice |
| **Action semantics ledger** | Effect tags from observation | Hardcoded SP80 verb map |

**Consume ids:** `portband`, `geoband`, `deltaband`, `exploreband`, `actionmodband`, `action_semantics_ledger`

---

## 6. Oracle / Ghost / memory

| System | KEEP | STRIP |
|---|---|---|
| **Oracle** | One missing-element assumption (review) | Assumption as control/goal/action/solve |
| **Ghost** | Confirmed relation hints this game | Ghost as action route / authority |
| **episode_memory** | Working episodic state | Cross-game permanent prior |
| **relationship_ladder** | Ordered relation hypotheses | Ladder complete ⇒ win |
| **match_calibration** | Reweight role candidates only | Admit actor/goal truth |

**Consume ids:** `oracle`, `ghost`, `episode_memory`, `relationship_ladder`, `match_calibration`

---

## 7. Loom / LeWM / Rexus

| System | KEEP | STRIP |
|---|---|---|
| Encoder + predictor objectives | Online/generic world model training | Unlocked runtime without evidence |
| Affordance head | `did_frame_change` | — |
| Candidate bank A→O_pred | Rank/prune interface | Q6 as action owner without gate |
| CEM / planning wrappers | Generic planners | Game-specific pretrain as live policy |
| GRAM++ path probes | Research row-shape | Claiming training-ready true from proxy |
| lewmtrain scripts | Trajectory tools | Imitation of human answer routes |

**Consume id:** `loom_lewm_rexus` → `worldmodel.candidate_predict_rank.v1`

---

## 8. Crucible / mechanics induction

| System | KEEP | STRIP |
|---|---|---|
| Transition mining | Induce from env interactions | Rules from game source |
| Grammar / forward_v0 | Hypothesis families | Answer-key mechanics |
| Discrepancy ledger | Fail cases | — |

**Consume id:** `crucible` → `induce.transition_mechanics.v1`

---

## 9. MegarexusCNN / synthesis_lab training

| System | KEEP | STRIP |
|---|---|---|
| Network architecture | Generic heads | Frozen SP80-only as universal agent |
| Outcome / catch heads | Online labels from pixels+env | Human-mined route datasets (`_invalid/`) |
| level_probe harness | Scout evidence | Clobbering canonical reports as proof |

**Consume id:** `megarexus_cnn`

---

## 10. Specialists (honest demotion)

| System | Status |
|---|---|
| SP80 DW directgrid (fixture priors) | RESEARCH specialist — live/ranked real, not legal-general |
| ft09_glyph | RESEARCH specialist — live 6/6, not general |
| ls20_corridor | RESEARCH partial |
| arcade_portfolio_v1 | Dispatch layer — legal only if children are legal |

**Consume id:** `specialist_solvers` (boundary note only)

---

## 11. Puffer / paint / control lab

| System | KEEP | STRIP |
|---|---|---|
| Generic RL loop | Optional trainer | SP80 paintflow-only policy as brain |
| paint_control_skill | Research | — |
| sp80_player_harness | Control contract research | Treating as scorecard primary |

**Consume id:** `pufferlib_rl`

---

## 12. BAN / quarantine (fence only)

| Path | Why |
|---|---|
| `codexlab/_invalid/sp80_spill_model*` | Hardcoded win + privileged state |
| `sim3d/dropletsim.py`, `native_scene.py` | Source-derived dynamics |
| `orchestration/_invalid/*` human routes | Scripted solutions |
| Any `environment_files/**/*.py` as knowledge | Game source |

**Consume id:** `quarantine_fence` → empty agent contribution

---

## Legal loop assembly

```text
tabula_rasa (wipe)
  → goose explore
  → contra/vision/portband/geoband (hypotheses)
  → deltaband + crucible + role_induce (from transitions)
  → dreamweaver program search (no fixture banks)
  → loom affordance rank (optional)
  → ghost/episode_memory (this game only)
  → env judges levels_completed
```

Implemented in `legal_search/loop.py` (`LegalAgent`).

---

## Counts

| Category | Systems registered |
|---|---|
| Proposal-capable | goose, dreamweaver, exploreband, ghost, episode_memory, loom_lewm |
| Informers | contra, portband, oracle, bands, vision, … |
| Guards | quarantine_fence, specialist_solvers, tabula_rasa |
| **Total consume adapters** | **25** (see `list_systems()`) |

## Full mine → LoRA (v2)

Every system has a **fixed 12-D slot** in `legal_search/train/features.py`
(`FEATURE_SCHEMA=legal_search.features.v2_fullmine`, `FEATURE_DIM=392`).

```text
consume_all(25) → encode_system_slot × 25 → LegalBlendLoRA fuse (LoRA) → train
```

See `docs/FULL_MINE_TO_LORA.md`. Train:

```bash
python3 -B legal_search/train/train_lora.py --out-dir legal_search/runs/blend_lora_v1_fullmine
```
