# Regime review — re-knock the arrow (2026-07-04)

Requested by the user after the Mythos boundary correction: spot the weakness,
define it, strengthen the training regime. Plan only — execution belongs to
subagent lanes. Read with `CODEX_HANDOFF_20260704_MYTHOS_BOUNDARY.md` and
`CNN_TRAINING_REGIME_20260704_NO_IMITATION.md` (the regime doc; it stands —
this note operationalizes it and fixes its gaps).

## Diagnosis — why the arrow is off target

The CNN has not learned anything real since v8 (2026-07-02). Every live gain
after v8 came from action-side wiring; the one lane that "moved" the model
(route replay v16–v26) was imitation and is quarantined. That is not an
architecture failure or a gate failure alone — it is the interaction of three
structural defects:

**D1 — Data starvation.** The admitted clean corpus is tiny and narrow:
`control_play` (200 episodes, no manifest), `course_correction_v10`
(12 episodes, 0 level-2 successes), `perturb_recovery_mining_v11`
(28 trajectories). The v27 clean smoke proved the pipeline learns exactly what
the data contains (dynamics/control/catch fine) and fails exactly what it
lacks (level positives 0/3, click acc 0.043 vs v8's 0.48). The regime banned
the contaminated data engine but never built the replacement at scale. The
route-replay cheat was a symptom of this vacuum, not its cause.

**D2 — The no-loss-vs-v8 gate is a no-op ratchet.** `gate_eval.py` requires
head-by-head no-regression on v8's own fixed valset. History: v9, v10 fresh/ft/
sel, v11 all failed it; the only passes (v10dynonly, v12) changed nothing
measurable. Training on *different* (clean) data while forbidding any movement
on v8's *distributional* valset structurally selects for "identical to v8."
The gate measures fidelity to v8's data distribution, not competence.

**D3 — Bootstrap gap in the goal lane.** Regime Stage D requires goal labels
from the agent's *own* completions, but the clean agent completes ~level 0–1.
Stage20/32 completion frames are assisted lineage and inadmissible. Ground
truth for levels 2+ is unreachable by grinding from level 0 — the flywheel
has no crank. Unlock verified today: the offline engine
(`environment_files/sp80/589a99af/sp80.py`) exposes `on_set_level` /
`next_level`, so clean interaction can be farmed AT any level directly,
seed-disjoint from all gate/live surfaces. Nobody has used this.

Secondary defects:

- **D4 — Guard is filename-based, both leaky and unauditable.**
  `no_imitation_guard.py` substring-matches paths. `course_correction_v10`
  passes while its manifest records a `demo_recovery` collection policy;
  `control_play` has no manifest at all (violates Binding Data Rule #5).
  Lineage must be a declared, checked manifest field, not a filename accident.
- **D5 — No imagination metric.** Pour-count calibration and level-3 vertical
  routing depend on multi-step rollout of dynamics/settle across the 20-frame
  ACTION5 pour animation. No gate measures k-step rollout accuracy;
  `n_input_frames` maxes at 2, no recurrence. The regime's own rule ("no third
  same-basin run without a representation change" — Stage F) is already
  triggered by the v9/v10/v11 failures.
- **D6 — Two ladders conflated.** The 28.57/3-level headline stack contains
  components now banned at runtime (Stage17 graph-demo templates, v14/v15 goal
  heads). v12's admitted status is unstated in the handoff. There is no
  component-level classification of either stack and no explicit twin
  scoreboard (clean lane vs frozen legacy reference).

## Re-aimed lanes (in order; owners = subagents)

### Lane 0 — Boundary manifest + guard hardening (cheap, blocks everything)

1. Produce `BOUNDARY_MANIFEST` table: every component of both stacks —
   v8, v12, graph-adaptive policy, Stage17 template family, goal heads
   v14/v15, goal-provider wiring, each active shard — classified
   admitted / assisted / invalid under the corrected boundary, with the reason.
   Explicitly rule on v12 and on the graph-adaptive policy (is it graph-DEMO
   derived, or autonomous search?).
2. Harden `no_imitation_guard.py`: read manifest contents (collection
   policies, parent checkpoints, collectors), not just path substrings.
   Add a required `lineage:` field to the manifest schema. Backfill a
   manifest for `control_play_*.pt`. Rule on the `demo_recovery` rows inside
   `course_correction_v10` (strip rows or quarantine shard).
3. Publish the twin scoreboard in HANDOFF: **clean ladder** (current: 1 level
   CNN-alone) and **legacy assisted reference** (28.57/3, frozen, never a
   training target). All future reports cite both, labeled.

### Lane 1 — Clean interaction corpus at scale (the core fix; long pole)

Industrialize regime Stage B. One pre-registered collector condition:

1. `collect_level_interactions.py`: use offline `on_set_level` to farm
   episodes at EVERY level (0–5), not just from level 0. Policy mix per
   level: random, scripted primitive sweeps (click / select / move / pour),
   current-clean-model policy, perturbation harness. Record frame-before /
   action / frame-after / env feedback / seed / provenance per regime rules.
2. Capture every pour with its full 20-frame animation window (feeds Lane 2's
   rollout gate and Lane 3's completion-window labels).
3. Scale target: 10–100x the current corpus, balanced per-level coverage.
   Pre-register seed ranges: train range A, val range B, gate range C,
   live untouched. Level-instance disjointness = seed disjointness (same
   convention the perturb gate already uses). **Decision point for user:**
   confirm this ruling — offline level instantiation for interaction farming
   is Stage-B-legal (the ban is answer routes and eval-surface leakage, not
   env interaction at higher levels). Recommended: legal.
4. Sub-lane 1b (immediately, zero risk): scale `control_play` 200 → 2000+
   episodes with a click-dense curriculum. Directly targets click
   localization (0.043 clean vs 0.48 v8) — the cheapest named skill gap.

### Lane 2 — Gate redesign: clean ladder is primary

1. For the clean lane, promotion = clean benchmark suite: mechanics heads vs
   fixed baselines on disjoint clean val + adapter gate + perturb gate +
   new rollout gate. The v8 fixed-valset becomes a *regression report* with
   pre-named acceptable tradeoffs, not a hard gate. Keep strict
   no-loss-vs-v8 only for the claim "replaces v8 inside the legacy stack."
2. New **rollout/imagination gate**: from the pre-pour frame, predict the
   pour outcome (settle mask, catch, level-up) across the 20-frame window;
   metric = per-pour outcome prediction accuracy on held-out clean pours.
   This is the capability pour-count planning depends on; it gets a number.

### Lane 3 — Goal-head rebuild on clean lineage

Labels from Lane 1 completions only (all levels reachable via level farming
+ hindsight completion windows). Rebuild the goal-label dataset (channels
`target_lock, key_block, paint_output`), train from a clean parent, rerun the
designation probes and the provider-enabled adapter gate. The provider wiring
and lock-inventory→pour-cap derivation already exist and are legal Stage E
perception use — only the weights need clean rebirth.

### Lane 4 — Representation change for temporal dynamics (one condition)

Licensed by the regime's own two-failure rule. Cheapest first:
`n_input_frames > 2` (or temporal aggregation over the pour window for
settle/catch heads). Single pre-registered condition, gated by the Lane 2
suite. Do NOT run as another v8 fine-tune under the old gate.

### Lane 5 — Pour count as a planned quantity (action-side, after Lane 2.2)

Plan = (arrangement, N pours); score by rollout-gate-verified imagination;
stop pouring when the predicted goal is met. Gate on wasted pours AND levels.
Replaces every budget-cap heuristic. Legal under Stage E: predictions as
perception, candidates judged by env outcome.

## Sequencing & success metrics

Lane 0 → Lane 1 (+1b immediately) ∥ Lane 2 → Lane 3 + Lane 4 → Lane 5.
Cap: at most two active lanes at once (one model-side, one action-side).
Compute: keep the GB10 preflight rule; pre-register Apollo fallback with the
torch-2.5 venv in the condition, not ad hoc.

The arrow now points at one number: **clean-lane CNN-alone levels completed**
(currently 1). Leading indicators: click acc trend, rollout-outcome accuracy,
wasted-pour count, clean goal-head IoU on held-out completions. The legacy
28.57 stays on the scoreboard as the frozen assisted reference until the
clean stack passes it honestly.
