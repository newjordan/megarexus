# Mine inventory — used vs unused (JOB 1 → JOB 2 handoff)

**Date:** 2026-07-08  
**Gate:** opened for gated train after this inventory. Residual debt does **not** block train; it only limits claim language.

---

## USED — in the legal loop (feeds features / consume / train labels)

### Algorithms (`legal_search/algorithms/`)

| Module | Role in train path |
|---|---|
| `components.py` | Components, contact graph, change regions, global frame block |
| `ports.py` | PortBand ports, alignments, edge-lock, lineup, fleet, event surfaces |
| `geometry_rel.py` | GeoBand bearings / pairs / nearest |
| `shape_traces.py` | Vision traces + color groups (no roles) |
| `local_symmetry.py` | Global + component + pair symmetry |
| `affordances.py` | Actor/target/support *candidates* |
| `statecon_graph.py` | Working set, branch graph, support groups |
| `explore.py` | Goose change-by-repetition rates (pure rebuild) |
| `memory.py` + `ghost_relations.py` | Episode memory + confirmed relation rows |
| `role_induce.py` | Movable/static/click from transitions |
| `crucible_mine.py` | v0 forward, discrepancies, rules, falsify |
| `dynamics.py` | DW DynamicsHypothesis score + imagine |
| `perception_trust.py` | Frame self-consistency trust gate |
| `program_search.py` | Hill-climb programs (plug-in scorer) |
| `worldmodel.py` | Affordance ranks + multi-step consistency |
| `cem_plan.py` | Discrete CEM sequences (dynamics scorer) |
| `calibrate.py` | Reweight + stage0/1 (never admit truth) |
| `saliency.py` / `score.py` | Edges, hist, proxy search scores |

### All 25 consume systems

Registered and encoded into **392-D** `v2_fullmine` features every collect step (`run_consume=True`).

Proposal systems that can affect weak action labels: goose, DW, exploreband, ghost, episode_memory, loom_lewm.

### Train labels (legal only)

| Label | Source |
|---|---|
| `did_change` | Env/frame delta |
| weak taken-action | What was actually stepped |
| optional `levels_up` weight | Env `levels_completed` only |

---

## NOT USED — fenced, specialist-only, or residual debt

### Hard BAN (never import into collect/train)

| Item | Why |
|---|---|
| `sp80_spill_model` / `_invalid/` spill | Win condition + privileged state |
| `dropletsim` | Answer-key dynamics |
| `human_mined` / route replay datasets | Imitation of solutions |
| Fixture `env_win` genome / direct priors | Shipped public-game answers |
| `native_scene` as knowledge | Source as oracle |

### Boundary-only (present in registry, not solutions)

| System | Role |
|---|---|
| `quarantine_fence` | Ban list fence |
| `specialist_solvers` | Research/scorecard only |
| `megarexus_cnn` | Optional FA architecture flag — not SP80-only general agent |
| `pufferlib_rl` | Optional generic RL idea — not SP80 paint brain |
| `tabula_rasa` / `match_calibration` admit flags | Wipe / reweight fence |

### Residual debt (not in train weights; methods incomplete vs full source)

1. PortBand full Monte-Carlo fleet parity with 222k source  
2. Crucible full grammar atom exhaustiveness + corpus package  
3. DW multi-family gravity/flow fit without paint roles  
4. Real LeWM encoder **weight load** path (CEM runs without it)  
5. Match transfer_prior / macroband full subset  
6. Goose **CNN** online adapter (tabular explorer is used)  
7. Specialist 6/6 policies as default legal agent (**never**)

### Prior runs (scout history — not this gated job)

- `legal_search/runs/blend_lora_v0`  
- `legal_search/runs/blend_lora_v1_fullmine`  
- `legal_search/runs/blend_lora_v2_train`  

These were ungated/scout. New gated train: `blend_lora_gated_v1`.

---

## Claim boundary for JOB 2

- **May claim:** legal blend LoRA on full-mine features + env-judged labels  
- **May not claim:** ranked SP80 DW 6/6, portfolio ARC general agent, spill/droplet solve, fixture-prior search  
