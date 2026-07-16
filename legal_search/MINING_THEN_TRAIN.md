# Mining job → Training job (single gate)

**Schema:** `legal_search.mining_then_train.v1`  
**Rule:** **No training until mining is complete.**  
**Gate file:** `legal_search/MINING_COMPLETE.json` (must exist and `complete: true`)

This is the only document that orders the two jobs. Everything else defers to it.

---

## Status (read this first)

| Job | Status | How marked |
|---|---|---|
| **1. Mining** | Signed for JOB 2 — see `docs/USED_VS_UNUSED.md` | `MINING_COMPLETE.json` |
| **2. Training** | Allowed when gate `complete: true` | `train_lora.py` gate check |

Prior runs under `legal_search/runs/blend_lora_*` were **scout / ungated** experiments. They do **not** count as the gated training job.

**Honest debt log:** [`docs/MINING_DEBT.md`](docs/MINING_DEBT.md) — source sizes vs what is actually mined. Do not stamp the gate after checklist theatre.

**Mine probe (JOB 1, not train):**

```bash
python3 -B legal_search/smoke.py
python3 -B legal_search/mine_probe.py
python3 -B legal_search/mine_probe.py --arcade
```

Probe verifies registry, 25/25 features, depth on critical systems, pure transition
rebuild, private-game wipe, and legal import audit. It does **not** write the gate.

---

# JOB 1 — MINING (must finish first)

## Goal

Reduce **every** lab system into a **legal consumable** that feeds a fixed feature vector for the blend model. No game answer keys. No training claims during this job.

## Scope (mine all of these)

| # | System ID | Source (approx) | Algorithm kept | Must strip |
|---:|---|---|---|---|
| 1 | `stochastic_goose` | `stochastic_goose/`, tufa policy | change-by-repetition explore | seed hacks as answers |
| 2 | `dreamweaver_sim3d` | `sim3d/dw_solve.py` etc. | program search / imagine rank | fixture win banks, level strategy tables, shipped SP80 physics JSON as truth |
| 3 | `contra_head` | `codexlab/contra_head/` | components, edges, symmetry hyps | admitted control/goal/solve truth |
| 4 | `portband` | `portband.py` | edge ports, alignments | route-flow as solve, lineup as action authority |
| 5 | `oracle` | `oracle.py` | one missing-element assumption | assumption as action/solve |
| 6 | `ghost` | `ghost.py` | this-game relation memory | routes / action authority |
| 7 | `geoband` | `geoband.py` | spatial pair queues | forced route |
| 8 | `deltaband` | `deltaband.py` | numeric deltas | control truth |
| 9 | `exploreband` | `exploreband.py` | probe queue | probe as solution |
| 10 | `actionmodband` | `actionmodband.py` | preference from evidence | forced action |
| 11 | `statecon` | `statecon.py` | working set | engine sprite truth |
| 12 | `episode_memory` | `episode_memory.py` | episodic working state | cross-game prior |
| 13 | `loom_lewm_rexus` | `architecture/`, lewmtrain | candidate rank / affordance | unlocked runtime without evidence |
| 14 | `symmetry` | `symmetry.py` | axis scores | symmetry as goal key |
| 15 | `contra_vision` | `vision.py` | raw components | semantic bucket labels |
| 16 | `crucible` | `crucible/` | induce from transitions | source-derived rules |
| 17 | `megarexus_cnn` | `megarexus_cnn/` | optional FA role | SP80-only as general agent |
| 18 | `specialist_solvers` | ft09, ls20, DW ranked | **boundary only** | default legal agent |
| 19 | `pufferlib_rl` | puffer / paint | optional RL loop | SP80-only brain |
| 20 | `quarantine_fence` | `_invalid/`, spill, dropletsim | fence | **never train on** |
| 21 | `tabula_rasa` | `tabula_rasa.py` | wipe bindings | — |
| 22 | `relationship_ladder` | `relationship_ladder.py` | relation hyps | ladder = win |
| 23 | `match_calibration` | `match_calibration/` | reweight only | admit actor/goal truth |
| 24 | `controller_affordances` | affordances | candidates | admitted actor |
| 25 | `action_semantics_ledger` | ledger | effect tags from observation | hardcoded SP80 verbs |

**Also mine (algorithms, not always separate registry rows):**

- `algorithms/explore.py`, `program_search.py`, `memory.py`, `components.py`, `role_induce.py`, `saliency.py`, `score.py`
- AdaJEPA deploy pattern: plan–act–adapt–replan (`train/ada_adapt.py`) — **document + implement**, not a train substitute

## Mining deliverables (definition of done)

Mining is **complete** only when **all** boxes below are true:

### A. Registry & adapters

- [ ] All 25 systems registered in `consume/registry.py`
- [ ] Each has a `ConsumeAdapter` with `algorithm_id`, strip list, fail-closed truth boundary
- [ ] `python3 -B legal_search/smoke.py` → 25 systems, `illegal=[]`
- [ ] `quarantine_fence` present; banned paths listed and non-importable into train labels

### B. Feature mine → model input

- [ ] `features.py` schema documented (`FEATURE_SCHEMA`, `FEATURE_DIM`, per-system slots)
- [ ] **Every** system has a dedicated non-empty encoder path (not a silent zero forever)
- [ ] `feature_report` shows **25/25 present** on a real Arcade reset frame **and** a synthetic frame
- [ ] Specialist + quarantine slots encode **boundaries only** (not solutions)

### C. Depth (no shallow stubs)

- [ ] PortBand / CONTRA / Delta / Ghost / DW search each contribute **structured** fields (counts, scores, centers, rates) — not only `present=1`
- [ ] Goose rates update from real transitions in the mine path
- [ ] DW path does **not** load fixture win banks into features
- [ ] Role induction (movable/static) wired from transitions where applicable

### D. Docs (single package)

- [ ] This file is the gate doc
- [ ] `MINING_MANIFEST.md` matches registry (no orphan systems)
- [ ] `docs/MINING_HANDOFF.md`, `docs/FULL_MINE_TO_LORA.md`, `docs/ADAJEPA_PORT.md` current
- [ ] `PRIMARY_LANE.md` points here for order of operations

### E. Legal audit

- [ ] Train label sources listed: only `did_change`, weak taken-action, optional `levels_up` **weight**
- [ ] Explicit ban list: spill_model, dropletsim, native_scene, human_mined routes, fixture env_win banks as labels
- [ ] `honesty_audit` / manual check: no `_invalid/` paths in collect/train imports
- [ ] Private-game test written: same binary, wipe memory, new `game_id` still runs mine

### F. Gate file

- [ ] Write `legal_search/MINING_COMPLETE.json` with the schema below **only when A–E are done**
- [ ] `complete` must be `true` and `checklist` all true

```json
{
  "schema": "legal_search.mining_complete.v1",
  "complete": true,
  "completed_at": "ISO-8601",
  "feature_schema": "legal_search.features.v2_fullmine",
  "feature_dim": 392,
  "n_systems": 25,
  "checklist": {
    "registry_smoke": true,
    "features_25_present": true,
    "depth_not_stub": true,
    "docs_synced": true,
    "legal_audit": true
  },
  "notes": "human or agent sign-off sentence"
}
```

**Until that file exists with `complete: true`, mining is incomplete.**

## Mining is NOT

- Running `train_lora.py` for 8–20 seconds  
- Getting val change_acc = 1.0 on synthetic blobs  
- Having adapters that only set `present=1`  
- Ranked SP80 DW 6/6 (specialist research, separate lane)

---

# JOB 2 — TRAINING (blocked until mining complete)

## Goal

Train `LegalBlendLoRA` on **legal_search experience** with full-mine features, then optionally deploy with AdaJEPA online LoRA adapt.

## Preconditions (hard)

1. `legal_search/MINING_COMPLETE.json` exists  
2. `"complete": true`  
3. All `checklist` values `true`  
4. `train_lora.py` gate passes (see below)

If any fail → **exit non-zero, no data.pt write, no best.pt write.**

## Training steps (only after gate open)

| Step | Action | Output |
|---|---|---|
| T1 | Collect Arcade multi-game + synthetic under full mine | `runs/<run>/data.pt` |
| T2 | Train change + weak policy (+ optional latent) | `best.pt`, `train_log.jsonl` |
| T3 | Val metrics vs base rate; held-out game if available | `metrics.json` |
| T4 | Optional: Ada deploy smoke (`AdaBlendPolicy`) | log only |
| T5 | Write `RESULT.md` with claim boundary | honest labels |

## Default train command (post-gate)

```bash
# snack (smoke-scale) — not a claim regime
/home/frosty40/miniconda3/bin/python3 -B legal_search/train/train_lora.py \
  --out-dir legal_search/runs/blend_lora_gated_v1 \
  --epochs 20 --batch-size 48 --collect-episodes 40 --lr 2e-3 --seed 42

# huge regime (default scale for real claims)
/home/frosty40/miniconda3/bin/python3 -B legal_search/train/train_lora.py \
  --out-dir legal_search/runs/blend_lora_gated_v2_regime \
  --regime huge --force-collect --seed 42
# huge ⇒ ~600 synth eps + 40 arcade (multi-game) × 80 steps, 100 epochs, batch 128
```

Override only with explicit flag after human approval:

```bash
# DANGER: scout only — does not open the real gate
python3 -B legal_search/train/train_lora.py --allow-ungated-scout ...
```

Ungated scout runs **must** write under `legal_search/runs/_ungated_scout/` and set `"gated": false` in config. They are not production training.

## Training success criteria (when allowed)

- [ ] Loss improves vs epoch 0  
- [ ] Change acc **beats base rate** on val  
- [ ] Prefer: held-out env / game_id eval  
- [ ] Config records `mining_complete` hash + feature schema  
- [ ] Claim language: legal Tufa+++ / not ranked SP80 / not portfolio ARC  

## AdaJEPA (after offline train)

Deploy path only (does not replace mining or offline train):

```python
from legal_search.train.ada_adapt import AdaBlendPolicy
pol = AdaBlendPolicy("<gated_best.pt>", adapt=True)
```

---

# Enforcement

| Mechanism | Behavior |
|---|---|
| This document | Human/agent order of operations |
| `legal_search/train/mining_gate.py` | Shared gate check |
| `train_lora.py` | Calls gate at start; aborts if incomplete |
| `collect.py` | Same gate when used for train data (not pure mine probes) |

---

# Order of operations (no exceptions)

```text
1. Finish JOB 1 mining checklist A–E
2. Write MINING_COMPLETE.json (complete: true)
3. Only then run JOB 2 training
4. Only then treat best.pt as gated training output
```

**If you are about to train and the gate file is missing: stop. Mine first.**
