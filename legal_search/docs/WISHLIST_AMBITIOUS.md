# Wishlist — craziest high-impact ideas (legal lane)

**Date:** 2026-07-09  
**Schema:** `legal_search.wishlist.v1`  
**Context:** Path B overnight left **synth floor 0.975**, Arcade best-ckpt still **0 levels** (SP80 thrash-die / ft09 no-op). Specialist lane already has SP80/ft09 scorecards; this list is for **legal general agent** moonshots only.  
**Hard constraints:** no spill/dropletsim oracles, no human-route BC, no fixture win banks as labels. Env remains the only judge.

Ranked **#1 = highest potential impact** if it works. Impact = chance of real Arcade/portfolio levels_completed × leverage on the whole stack, not ease.

---

## 1. Level-oracle self-play on *abstract* mechanics (not game answers)

**Idea:** Learn a *family* of invented grid games whose win conditions are *sampled programs* (reach object, sort colors, open portal, conserve mass, match target mask). Train a single legal policy + world-model to **level-up many random mechanics**, then transfer zero-shot / few-shot to Arcade SP80/ft09/etc.

**Why crazy:** training set never contains SP80 answers; only *the skill of discovering rules and completing levels*.  
**Why impact:** if transfer works, one model becomes a general ARC-game player instead of a per-game specialist.  
**Legal angle:** labels = env level counters on synthetic games only; consume stack stays legal.

---

## 2. Closed-loop AdaJEPA “dream hours” — replan on imagined frames that must re-enter reality

**Idea:** Scale AdaJEPA plan–act–adapt into multi-hour GPU loops: latent rollouts propose 50–200-step plans; **only trajectories that re-predict real Arcade frames** keep gradients. Kill thrash/no-op policies by imagination–reality KL + survival length.

**Why crazy:** full agentic learning loop, not offline epochs.  
**Why impact:** directly attacks Path B’s failure modes (SP80 dies in 7 steps; ft09 freezes 150). Online LoRA-only steps already sketched in `ADAJEPA_PORT.md`.  
**Kill criteria:** imagination that never matches Arcade is discarded (no free fake wins).

---

## 3. Population of legal “species” + tournament evolution of *algorithms*, not weights

**Idea:** Treat each consume system (Goose, DreamWeaver, PortBand, Ghost memory, Loom affordances, CONTRA, …) as a genome of *procedures*. CEM / MAP-Elites evolves **composition graphs** (who proposes, who vetoes, who searches) scored only on Arcade `levels_completed` + survival. Weights secondary; **wiring is primary**.

**Why crazy:** automatic scientific method over the 25-system mine.  
**Why impact:** Path B showed the blend LoRA can fit change_acc while the *decision architecture* still thrash-dies or no-ops — evolution might find the missing controller topology.  
**Output:** evolved LegalAgent DAGs + receipts, not black-box cheats.

---

## 4. Success-window time machine from collect milestones

**Idea:** Cycles 9/11/18/31 logged Arcade level-ups during overnight collect. Build a **forensic miner**: for every env level increment, backtrack N steps, re-encode fullmine features, train a *level-edge* specialist head + reverse-curriculum that only rewards approaching that manifold. Multiply rare events via legal counterfactuals (legal action rewrites that still change frame the same way).

**Why crazy:** turn four accidental pings into the entire curriculum.  
**Why impact:** highest *near-term* shot at locking `any_arcade_level=true` without banned lineages.  
**Risk:** overfit those four episodes — mitigate with multi-game + held-out Arcade games.

---

## 5. Dual-head “brake / gas” policy with hard survival bar

**Idea:** Explicit two-mode controller: **Gas** maximizes informative change + search proposals; **Brake** maximizes not-dying / not-noop. Hierarchical gate learns *when* to thrash-explore vs freeze-observe. Train with asymmetric costs: SP80 GAME_OVER in <K steps = huge penalty; ft09 change_rate=0 for M steps = forced ε-explore burst.

**Why crazy:** treat failure modes as first-class physics of the agent.  
**Why impact:** Path B best ckpt literally fails as pure extremes of one continuum — dual control could unlock both games with one architecture.  
**Ambitious upgrade:** learned options library (macro actions: “scan edges”, “nudge once”, “commit click”).

---

## 6. World-model crucible: predict `levels` and `GAME_OVER` before acting

**Idea:** Train a legal frame+feature world model that forecasts next frame, **game state**, and **level counter delta**. Policy is planned with CEM/MCTS in the model with objective `E[Δlevels] − λ P(GAME_OVER)` under legal action sets only. Recalibrate online when reality diverges.

**Why crazy:** full model-based RL on ARC Arcade with level as terminal signal.  
**Why impact:** levels are sparse; a model that *imagines* level-up conditions multiplies sample efficiency 10–100×.  
**Legal:** model cannot be seeded from fixture genomes or spill physics dumps — only legal rollouts + synthetic mechanics family (#1).

---

## 7. Cross-game skill library + symbolic program induction co-trained with LoRA

**Idea:** DreamWeaver-class program search and the neural blend LoRA share a **skill tape**: short programs (move-until, flood-stop, match-color-click) that either execute or condition the policy. When search finds a program that level-ups once, distill it into LoRA *and* keep the symbol. Portfolio = growing library of legal skills, not one blob weight file.

**Why crazy:** neuro-symbolic ARC agent that writes its own playbook.  
**Why impact:** specialists already encode game-specific wins; a legal skill library could approach specialist scores without answer banks.  
**Ambition peak:** auto-export skills as human-readable markdown for audit.

---

## 8. Fleet-scale multi-game overnight: 20 games × continuous promote-on-level

**Idea:** Stop optimizing SP80+ft09 only. Continuous 24/7 loop across every offline Arcade title available: separate `best_synth.pt`, `best_arcade_any.pt`, `best_per_game/{id}.pt`. Any level-up anywhere freezes a lineage and spawns a child with heavier success-window weight. Path B’s single-synth promote rule dies.

**Why crazy:** industrialize the overnight script into a **scorecard factory**.  
**Why impact:** breadth finds games the current policy *can* crack; those trajectories teach transfer to hard ones.  
**Ops:** GPU schedule, disk budget, honest scoreboard JSON the human can trust.

---

## 9. Perception → object permanence → affordance graph as first-class state

**Idea:** Replace flat 392-D pooled features as the *only* policy state with a live **object graph**: components, ports, last-seen positions, symmetry groups, “what I can click that moves.” Graph neural LoRA acts on nodes/edges; Ghost memory becomes node attributes. Rebuild CONTRA/PortBand/Delta as continuous graph maintainers every frame.

**Why crazy:** rebuild legal perception as a real cognitive architecture, not a bag of stats.  
**Why impact:** thrash/no-op often mean “policy doesn’t know what the objects are.” Graph state is the missing inductive bias for ARC-like games.  
**Cost:** big encoder rewrite; highest research risk, huge upside if objects are the bottleneck.

---

## 10. Meta-goal: “become the legal twin of the specialist portfolio”

**Idea:** For every game where specialist evidence exists (SP80 DW-directgrid, ft09 glyph, …), run a **shadow contest**: legal agent must match specialist *levels_completed* on the same scorecard runners *without* specialist answer paths. Use specialist only as a **black-box behavior teacher of action distributions on public frames** if ever allowed — default: **no distillation from specialists**, only env-matched score targets. Public dashboard: legal vs specialist gap per game.

**Why crazy:** explicit moonshot KPI for the whole lab lane.  
**Why impact:** turns “legal general agent” from a vibe into a portfolio race; forces every research bet to move the gap.  
**Guardrail:** never import specialist win banks, genomes, or private routes into legal labels — gap closes by env performance only.

---

## Impact ranking summary

| Rank | Codename | Core bet |
|---:|---|---|
| 1 | **Mechanics self-play** | Generalize level-up skill across invented games |
| 2 | **Ada dream hours** | Closed-loop imagination constrained by reality |
| 3 | **Evolve algorithms** | MAP-Elites over consume system wiring |
| 4 | **Milestone time machine** | Forensic mine of cycles 9/11/18/31 level-ups |
| 5 | **Brake/gas dual policy** | Kill thrash + no-op with explicit modes |
| 6 | **Level world model** | Plan for Δlevels − death |
| 7 | **Skill tape** | Neuro-symbolic program + LoRA co-train |
| 8 | **Scorecard factory** | Multi-game continuous promote-on-level |
| 9 | **Object graph mind** | Replace pooled features with live affordance graphs |
| 10 | **Legal twin KPI** | Match specialist portfolio levels without answer theft |

## Suggested sequencing (if someone actually builds)

Near-term spike (days): **#4 → #5 → #8**  
Medium bet (weeks): **#2 + #6** sharing world model  
Moonshot tracks (parallel research): **#1**, **#3**, **#9**  
North star metric: **#10**

## Non-goals on this wishlist

- Importing spill / dropletsim / human routes / fixture banks “just this once”
- Claiming portfolio solve from synth floor alone
- Infinite Path-B restarts with the same promote-on-synth rule

## Pointers

- Last result: `docs/OVERNIGHT_PATH_B_HANDOFF.md`
- Best ckpt: `runs/overnight_level/best/best.pt`
- Gate: `MINING_THEN_TRAIN.md` / `MINING_COMPLETE.json`
- Lane split: `../PRIMARY_LANE.md`
