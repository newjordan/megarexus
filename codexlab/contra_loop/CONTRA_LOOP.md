# CONTRA Loop Harness — Canonical Training Doc

**Point an agent at this file.** It is the single source of truth for running CONTRA as a
self-improving loop that accumulates *verified algorithmic solvers*.

> CONTRA is an Oracle/controller **head** for ARC-AGI-3. SP80 is only the test game, not the
> product. The product is **this loop** and the growing library of solvers it produces.

---

## 1. Mental model

Turn CONTRA from "the thing that solves SP80" into a **library that accumulates verified
algorithmic solvers**, driven by a loop an agent rides. Each iteration adds (or rejects) a
solver. The library only ever grows with solvers backed by a reproducible artifact.

```
        ┌──────────────────── SOLVER LIBRARY ────────────────────┐
        │  verified algorithmic solvers + mechanic schemas,       │
        │  each carrying an immutable reproducible artifact        │
        └─────▲────────────────────────────────────────┬─────────┘
        admit │ (only if VERIFY gate passes)  retrieve  │ applicable solvers
              │                                          ▼
   ┌──────────┴──┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │   VERIFY    │◀──│ CRYSTALLIZE  │◀──│  DISCOVER     │◀──│   ATTEMPT     │
   │ reproducible│   │ mechanic →   │   │ mine deltas → │   │ CONTRA plays  │
   │ artifact or │   │ algorithmic  │   │ mechanic      │   │ w/ current    │
   │   REJECT    │   │ solver spec  │   │ hypothesis    │   │ library       │
   └─────────────┘   └──────────────┘   └──────────────┘   └───────┬──────┘
         │                                                          │
         └──────────────────────  next iteration  ─────────────────┘
```

The loop rides infrastructure that **already exists** — it is mostly assembly, not greenfield:

| Stage | Reuses (already in repo) |
|---|---|
| ATTEMPT | `codexlab/sp80_run_contra_levels.py` + the `contra_head/` informer bands (PortBand/ExploreBand/Delta/Geo/Oracle/Ghost) — the *current, implicit* solver library |
| DISCOVER | `codexlab/arc_opus48_playthrough_miner.py` (mines playthroughs → hypotheses), generalized to mine CONTRA's *own* episode deltas |
| CRYSTALLIZE | `contra_head/ghost.py` + `ghost_mechanics_overlay.py` — the Ghost mechanic schema is the registry record template |
| VERIFY | `codexlab/sp80_verify_episode_run.py` + `python codexlab/contra.py check` — the anti-reward-hacking spine |
| ADAPT | `codexlab/players/sp80/` registry (`game_player.v1`) → a `GameAdapter` a new game implements |

---

## 2. THE ASSIGNED TARGET (first loop objective)

> ### 🎯 Discover and crystallize the **SP80 Level-1 mechanic**, beat L1, and register the solver.
>
> **Done when ALL hold (reproducible, no reward-hacking surface):**
> 1. A run records `levels_completed_delta > 0` on `level_01`.
> 2. `python codexlab/sp80_verify_episode_run.py --run-dir <that level_01 dir>` returns 0.
> 3. The new solver is admitted to the registry with **that run_dir as immutable provenance**.
>
> **Why L1:** it is the known wall — *not scorer-starved; it needs the mechanic.* L0 is solved
> (see seed solvers below); L1 is the loop's reason to exist, and its success is reproducibly
> checkable. See `[[contra-l1-blocker-diagnosis]]`.
>
> **Next target after L1:** onboard one *new* game via `GameAdapter` — proving the harness
> generalizes past SP80.

---

## 3. The per-iteration agent contract

Put an agent on the loop with this contract. **Deterministic code** does ATTEMPT and VERIFY;
**agent judgment** does DISCOVER and CRYSTALLIZE. The agent must never claim a solve without
the artifact.

Each iteration:

1. **ATTEMPT (code).** Run `python codexlab/contra.py loop --target sp80:1` (wraps
   `sp80_run_contra_levels.py`). Capture the run_dir, `levels_completed_delta`, and the
   per-packet decision trail (`packets.jsonl`).
2. **Check the artifact (code).** If `levels_completed_delta > 0` on the target level **and**
   the verifier returns 0 → the active solver hypothesis is **confirmed**; go to CRYSTALLIZE
   to register it. Else → go to DISCOVER.
3. **DISCOVER (agent).** Read the failing episode's packets + frames. Form ONE concrete
   *mechanic hypothesis*: "the level needs X to be true; the current solvers never make X
   true." Ground it in the frame deltas — what changed vs what needed to change. Mine prior
   *successful* playthroughs (`seed/algorithm_pointers.md`) for transferable structure.
4. **CRYSTALLIZE (agent → code).** Express the hypothesis as an **algorithm pointer**: a
   named, threshold-precise procedure (preconditions → ordered steps with concrete triggers →
   mechanic → expected effect). Encode it as a candidate solver (a scored candidate-producer
   over the recommendation packet, like the existing bands).
5. **VERIFY (code).** Re-run ATTEMPT with the candidate active. Admit to the registry **only
   if** the artifact bar in §2 is met *and* it does not regress existing verified solves.
   Otherwise reject and record why (the rejection is also data for the next hypothesis).
6. **LOOP.** Repeat until the target's success criterion is met, then move to the next target.

**Guardrails (cardinal):**
- A "solve" is real **only** with `levels_completed_delta > 0` + verifier rc 0. Nothing else.
- `status: nominal` means the episode/scorer contract held — **not** that the level was solved.
- Every admitted solver carries its proving `run_dir` as immutable provenance. The library can
  never accumulate a solver without one.
- Never edit a verifier to make a red bar green. If a verifier is wrong, diagnose it first
  (see `[[sp80-episode-verifier-preexisting-fail]]` for the standard of proof).

---

## 4. How to run

The engine is `loop.py` — a 9-step protocol (GROUND → DISCOVER-GAP → HYPOTHESIZE → KEYSTONE →
WIRE → LIVE-RUN → VERIFY-GATE → ADMIT → STALL-CHECK) wrapping the real `contra run` +
`sp80_verify_episode_run` spine, behind a **5-layer gate** and a **self-verifying, content-
addressed registry** (`registry/solvers.jsonl`) that the agent can never write directly and that
re-verifies every record on read. Reward-hacking is structural, not trust-based.

```bash
# Prove the spine end-to-end (GROUND + gate layers; no admission required):
python codexlab/contra.py loop iterate --target sp80:1 --dry-run

# Inspect the library + honest negatives (rejections.jsonl) + journal:
python codexlab/contra.py loop status

# One real iteration that attempts an admission for a wired mechanic:
python codexlab/contra.py loop iterate --target sp80:1 --mechanic <SOLVER_NAME>

# Gate / admit a specific run_dir (admit NEVER bypasses the gate):
python codexlab/contra.py loop gate  --target sp80:1 --run-dir <path>
python codexlab/contra.py loop admit --target sp80:1 --run-dir <path> --mechanic <NAME>
```

`--target <game>:<level>` selects the `GameAdapter` and level. `sp80` is the only adapter today;
onboarding a new game = adding a `players/<game>/` + an `adapter.py` `GameAdapter` entry.
Validated day-one: `iterate --dry-run` runs the full spine and correctly returns
`would_admit: false` on L1 (`target_delta=0`); 16/16 anti-reward-hacking tests pass; the registry
self-rejects a forged `run_dir`.

---

## 5. Seed solver library (verified L0 — the agent starts from these)

These are **algorithm pointers** mined from CONTRA's own *verified* wins. Detailed, packet-cited
specs live in **`codexlab/contra_loop/seed/algorithm_pointers.md`** (populated by the seed
miner). The anchor pointer:

### `ALIGN_SOURCE_THEN_ACTIVATE_FLOW`  *(SP80 L0 — VERIFIED)*
- **applies_when:** a selectable source actor exists; a target port/sink exists; an `activate_flow`
  (ACTION5) primitive is available.
- **procedure:** select the source actor → issue PortBand lineup moves toward the target port
  (value_score ≈ 0.82 rows sourced from `portband.lineup_matrix.rows`) → when the PortBand
  alignment signal saturates (`portband_score == 1.0` and display-edge-aligned) → issue
  `activate_flow`. Repeat per remaining source until the level completes.
- **mechanic:** liquid flows only when the source actor is edge-aligned with the target port;
  `activate_flow` while aligned commits the flow.
- **effect:** `levels_completed_delta` increments.
- **provenance (reproducible — same 16-action solution across runs):**
  - `codexlab/runs/contra_sp80_episode_levels_00_01_20260608T173344Z/level_00` (delta=1, 16 actions)
  - `codexlab/runs/contra_sp80_episode_levels_00_01_20260604T171457Z/level_00` (delta=1, 16 actions)
- **generalization_hypothesis:** "align actor to sink, then commit" is a general
  *transport/plumbing* primitive; SP80-specific parts are the action ids and the edge-alignment
  geometry. The general form is the seed for non-SP80 games.

---

## 6. Wanted mechanic: SP80 L1 — surface/orthogonal paint flow, chained (the assigned target)

**The mechanic (human-confirmed):** paint is captured at the **top of the largest action
object**; its **orthogonal sides re-direct the paint flow** — a surface/orthogonal flow dictates
paint direction. L1 needs several such redirects **chained** (a staircase) to walk paint across to
all buckets. L1 ends `budget_exhausted`, `delta=0`; it is *not* scorer-starved — the controller
never assembles the chained multi-source flow.

**Progress so far (3 partial rules; L1 still UNSOLVED — honest negatives the loop must remember):**

| Rule | Status | Measured effect |
|---|---|---|
| `SURFACE_ORTHOGONAL_PAINT_REDIRECT` | **committed** (`sp80_player_harness.surface_orthogonal_flow_redirect`) | boosts moves orthogonal to the largest object's paint-flow axis (real signals: `closed_shape_traces.area_pixels`, `edge_touches.top`, `flow_direction_hypothesis`). L0 safe (0.46 < 0.48 win), gate green. Did not solve L1 alone. |
| `STABLE_MULTI_SOURCE_RELAY_LOCK` (HOLD) | **committed** (`relay_lock_hold_bonus`) | persists a committed relay leg until stuck. Thrash 14→12. L0 safe. |
| `STAIRSTEP_PAINT_CHAIN` | **reverted** | full version made a real staircase (pkts 11–14 `move_left,move_up,move_left,move_up`), thrash 14→9, but **regressed L0** (its +0.07 stacked on the orthogonal +0.18 → L0 pkt15 `move_left` 0.53 > the 0.48 winning `activate_flow`). Correctly `git checkout`-reverted. |

**Precise L0-safe next step (next agent on the loop):** re-implement the stairstep as **flow-leg-
only** — boost *only* the along-flow move, *only* in the expect-flow phase, and **skip any row that
already carries `surface_orthogonal_flow_redirect_bonus`** (delegate every redirect leg to the
orthogonal rule). That keeps the along-flow bonus under L0's 0.17 pre-pour margin
(`move_down` 0.31+0.14=0.45 < 0.48) — L0-safe by construction — while still imposing the staircase
alternation on L1. The deeper lift the bonuses can't reach: real per-leg `target_object_id`
sequencing (ADVANCE) + a DEFER gate so `activate_flow` only fires once all legs are locked.

See `[[sp80-fuel-and-relay-solve]]` and `[[contra-l1-blocker-diagnosis]]`.

---

## 7. Files in this package

| File | Role |
|---|---|
| `CONTRA_LOOP.md` | **this doc** — the canonical training doc |
| `loop.py` | **the engine** — `contra.py loop` CLI: the 9-step protocol + admit/reject |
| `baseline.py` | GROUND step: run episode, parse `summary.jsonl`, emit `baseline.json` |
| `ground_truth.py` | `GroundTruthOracle` — pixel-exact frame pairs (win-bank / engine / recordings) |
| `gap_finder.py` | DISCOVER-GAP: diff predicted-vs-observed fill → the only hypothesis seed |
| `crystallize.py` | pure Crystallizer: the ONLY writer of `registry/solvers.jsonl` |
| `gate.py` | the 5-layer VERIFY-GATE (+ SHA `gate_pin.json`) |
| `registry.py` | append-only, content-addressed registry; re-verifies every record on read |
| `solver_registry.py` | admission primitives (`load_registry`/`register_solver`/`_provenance_is_real`) `loop.py` builds on |
| `adapter.py` | `GameAdapter` — new-game onboarding seam over the generic `run_contra_view` |
| `solver_interface.py` | `SolverProtocol` — uniform shape for a registered solver |
| `stalls.py` | STALL-CHECK: reinvent / thrash / zero-coverage / unreproducible detectors |
| `keystones/` | standalone forward-model validators (discovery-only; a keystone ≠ a solve) |
| `tests/` | anti-reward-hacking suite (16/16 green: the gate rejects forged run_dirs) |
| `registry/` | `solvers.jsonl` (4 verified) · `rejections.jsonl` (honest negatives) · `gate_pin.json` |
| `seed/algorithm_pointers.md` | packet-cited seed solvers mined from verified wins |

The deterministic steps (GROUND, GATE, ADMIT) are real today and bracket the agent's judgment
(HYPOTHESIZE, KEYSTONE, WIRE) so nothing counts as solved without a re-runnable engine artifact.
(`loop_driver.py` is the superseded v1 minimal entry, kept for reference.)
