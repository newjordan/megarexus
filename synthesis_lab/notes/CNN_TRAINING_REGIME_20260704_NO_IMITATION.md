# CNN Training Regime - No Imitation, No Answer Training

Date: 2026-07-04

This document supersedes any plan that trains on human demonstrations, mined
routes, replay traces, same-level solved/eval states, graph-demo action plans,
or answer-derived labels. The user ruling is absolute: imitation training is
never authorized.

## First Principle

The CNN exists to learn causal environment competence, not to memorize answers.

Allowed training target shape:

- From `(frame, action)`, predict the next frame.
- From `(frame, object, action)`, predict whether that object moves.
- From `(frame, pour/action)`, predict catch, settle mask, and level-up.
- From live pixels after the agent itself reaches a visible objective state,
  infer the current objective schema.
- From disjoint held-out episodes, prove that the learned representation
  generalizes.

Banned training target shape:

- "Here is the solved state; output the solved state."
- "Here is a human route; imitate the route."
- "Here are actions mined from the target level; replay them."
- "Here is an eval/live answer frame; train the model to emit that answer."
- "Here is a graph-demo plan; count the result as CNN-alone."

## Binding Data Rule

No data source is eligible for CNN-alone training unless all of these are true:

1. It was produced by a pre-registered non-imitation condition.
2. It records the action sequence or environment transition provenance.
3. It has no `human`, `demo`, `route`, `replay`, `answer`, `graphdemo`, or
   `graph_demo_plan` lineage.
4. It is split away from the gate/live evaluation surface by seed, episode,
   and level instance.
5. The condition records script paths, code hashes, dataset hashes, seed,
   budget, loader mode, output paths, and metrics before launch.

If any field is missing, the lane stops. Do not repair it by inference.

## Immediate Artifact Ruling

The following artifacts are not valid clean CNN-alone training products:

- `stage25_goal_label_dataset_20260704/goal_label_dataset.pt`
- `outcome_v14_goal_head_v12parent_20260704/best.pt`
- `stage33_paint_output_label_dataset_20260704/goal_label_dataset.pt`
- `outcome_v15_paint_output_goal_head_v12parent_20260704/best.pt`

Reason: the recorded Stage25 and Stage33 dataset-build commands included
`level3_human_mined_inclusive_stage22b_20260704.pt`. Those artifacts may be
used only as forensic diagnostics for what the old lane attempted to learn.
They cannot be used as parents, training targets, gates, or live scorecard
support for a CNN-alone/autonomous claim.

Stage20/Stage32 remain live evidence with graph-demo-plan assistance only.
CNN-alone live evidence remains the pre-incident one-level boundary unless a
new no-imitation condition proves more.

## Clean Skill Targets

The critical skills from last night's legitimate analysis are still valid as
targets, but they must be learned from clean interaction data:

- Level-0 objective reading: after the agent completes level 0 by its own
  non-imitation policy, the next visible frames expose target locks, key-like
  blocks, and paint outputs. The CNN may learn to segment those roles from
  clean self/environment episodes only.
- Key/lock inventory: count and locate targets/locks from current pixels, then
  derive pour/stream budget from that inventory. Never copy a historical pour
  count or answer route.
- Paint-output role: identify every small output block of the paint color from
  live/current pixels. Do not train from human-solved level-3 output states.
- Vertical control and catch routing: train controllability, catch, settle, and
  level-up heads on controlled environment interventions, not on human
  placements.

## Regime Stages

### Stage A - Enforcement Preflight

Before any training condition is built:

1. Run `python3 synthesis_lab/orchestration/honesty_audit.py`.
2. Reject any active invalid headline.
3. Reject any proposed train/val shard or policy-frame root whose path or
   manifest contains `human`, `demo`, `route`, `replay`, `answer`,
   `graphdemo`, or `graph_demo_plan`.
4. Reject any runtime condition that sets `MEGAREXUS_GRAPH_DEMO_PLAN` or any
   `MEGAREXUS_ROUTE_*` variable.
5. Write the resolved allowlist of data files and hashes into `CONDITION.json`
   before launch.

### Stage B - Clean Interaction Collection

Build a new no-imitation data corpus from controlled environment interaction:

- Randomized and model-policy episodes are allowed only if they do not use
  human/demo/route assistance.
- Every episode must record frame before, action, frame after, levels
  completed, seed, action mapping, and raw provenance.
- Level completions can provide environment-feedback labels only for the
  training split that produced them. They must not be copied from held-out
  eval/live surfaces.
- Failed episodes are useful; they teach dynamics, controllability, catch,
  miss, and non-level-up outcomes.

Output should be a new dated clean corpus, for example:
`synthesis_lab/experiments/megarexus_cnn/data/no_imitation_interactions_20260704_train.pt`
and a disjoint validation shard.

### Stage C - Mechanics Heads

Train `MegarexusCNN` on causal mechanics first:

- `forward_dynamics`: next-frame logits.
- `forward_control`: object displacement under action.
- `forward_catch`: pour touches catch/glove target.
- `forward_settle`: persistent settle mask.
- `head_level`: level-up probability.
- inverse dynamics only as a diagnostic, not as an action route source.

Use `train.py` with a pre-registered condition and only clean shards. Existing
heads already support these losses. Parent checkpoints must themselves be
clean or explicitly marked diagnostic-only.

### Stage D - Objective Informer Rebuild

Rebuild the goal-label dataset without any human/demo/route/eval-answer source.

Legal label source: clean training-split completions produced by the
non-imitation agent/environment condition. The label is the visible objective
state exposed by the environment after completion, not a human solution.

Train `train_goal_head.py` only after the clean dataset manifest proves:

- no banned lineage in inputs,
- disjoint train/val groups,
- current-frame labels, not future moved-object labels,
- channel order `target_lock`, `key_block`, `paint_output`,
- review page plus raw manifest.

The goal head must be evaluated on held-out clean completions and separate
post-level-0 current-frame probes. Passing label IoU alone is not solve
evidence.

### Stage E - Policy Use Without Answer Plans

The live policy may consume CNN predictions as perception inputs:

- target/lock count,
- key/object candidate masks,
- paint-output masks,
- control/catch/settle/level-up scores.

It may not consume human routes, route heads, graph-demo action plans, or
same-level solved state. Candidate actions must be selected by predicted
mechanics and current observations, then judged by environment outcome.

### Stage F - Gates Before Live

Required gate stack:

1. Data-lineage gate: all training inputs pass the no-imitation allowlist.
2. Label gate: goal head passes held-out clean label metrics.
3. Current-frame probe: after level 0, predictions match current objects, not
   future moved positions.
4. Mechanics gate: dynamics/control/catch/settle/level metrics beat fixed
   baselines on disjoint validation.
5. Offline adapter gate: no assist env vars, paired random baseline, review
   page, raw JSON.
6. Perturb/generalization gate: held-out seeds/layouts, no same-level replay.
7. Live scout: only after all above pass, with condition recorded beforehand.

Any failure blocks promotion. Do not launch another same-basin run after two
verified failures without a representation change.

## Launch Rule

"Train for N hours" means exactly one command:

```bash
python3 synthesis_lab/orchestration/train_watch.py --condition <pre_registered_clean_CONDITION.json> --max-hours N
```

The condition must already name one clean training run. The watcher must not
mine new data, invent a follow-up condition, alter code, add assist flags, or
promote a checkpoint.

## What To Do Next

The first enforcement task is now done in active code:
`synthesis_lab/orchestration/no_imitation_guard.py` is wired into
`train_watch.py`, `build_goal_label_dataset.py`, `dataset.py`, `train.py`, and
`train_goal_head.py`.

The next valid engineering task is to collect a clean no-imitation interaction
corpus under a pre-registered condition, then rebuild the goal-informer lane
from scratch from that corpus.
