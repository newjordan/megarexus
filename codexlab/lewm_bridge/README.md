# LeWM Bridge (pickup note — 2026-06-12)

Pre-work for the eventual CONTRA → LeWM integration. **Nothing here is wired
into any CONTRA solver path, the contra_loop, or dev_check** — that is
deliberate. The strategy decision (2026-06-12) is CONTRA-first: keep pushing
CONTRA to solve levels on its own, and only bring LeWM in when a trigger fires
(see "When to pick this up" below). This folder just keeps the future bridge
cheap and pre-answers the riskiest assumption.

LeWM = LeWorldModel, the JEPA-style latent world model in `/home/frosty40/Loom`
(ViT-Tiny encoder → AdaLN action-conditioned predictor, CEM planner). The
intended integration: CONTRA informs the **goal function** (CEM `cost_fn`) of
LeWM — LeWM owns "what happens if I act", CONTRA owns "what counts as progress".

## What's here

| File | Purpose |
| --- | --- |
| `export_episodes.py` | Exports **full ordered episodes** (every executed step: action signature, before/after grid refs + shas, goal progress, levels_completed_delta) from admitted runs to `codexlab/datasets/sp80_lewm_episodes_v0/`. Schema `codexlab.lewm_episode.v1`. This is the stable LeWM training-feed surface — unlike the crucible transition corpus it is not deduped, so trajectories stay intact for sequence training. |
| `latent_distance_probe.py` | One-shot sanity test: encodes a solved episode through Loom's Phase-0 encoder and checks whether latent L2 distance to the final (solved) frame decreases over the trajectory. Writes a report artifact under `codexlab/runs/_lewm_bridge/`. |

Run them from the repo root:

```bash
python3 -B codexlab/lewm_bridge/export_episodes.py
python3 -B codexlab/lewm_bridge/latent_distance_probe.py            # longest solved episode
python3 -B codexlab/lewm_bridge/latent_distance_probe.py --episode-id <id>
```

The probe needs the Loom repo (`--loom-root`, default `/home/frosty40/Loom`)
and its Phase-0 checkpoint
(`experiments/2026-05-23_phase0_production_rtx6k/bs64_lr1e-4/best.pt`).

## Export status (2026-06-12)

`sp80_lewm_episodes_v0`: **91 episodes, 2,385 steps, 4 solved episodes**
(all level_00), 54 inter-step discontinuities (surfaced per episode as
`discontinuity_count`, not hidden). Per-step npy sha verification on by default.

## Latent-distance probe result (2026-06-12): **FAILS — do not build a latent-distance goal cost on the Phase-0 backbone**

All 4 solved L0 episodes, Phase-0 checkpoint, cuda:

| episode | frames | spearman ρ (d_final vs t, excl. final) | decreasing-step frac | verdict |
| --- | --- | --- | --- | --- |
| 2f6b69d4 | 17 | **+0.50** | 0.69 | does_not_support |
| b946f279 | 17 | **+0.50** | 0.69 | does_not_support |
| fc242dfa | 7  | **+0.31** | 0.67 | does_not_support |
| 73b03303 | 15 | **+0.51** | 0.64 | does_not_support |

Reports: `codexlab/runs/_lewm_bridge/latent_distance_probe_20260612T*/report.json`.

What the series actually show (see `2f6b69d4` report):

- **Intra-episode latent collapse.** Distance from each frame to the *initial*
  frame stays ≈0.000–0.001 for the whole episode — the encoder maps every
  intermediate SP80 board state to nearly the same latent point, even when
  paint moves across the board. There is no usable progress gradient in this
  latent space regardless of which frame you pick as the goal.
- **The final-frame drop is scene change, not progress.** The solved frame
  differs from every prior frame by ~1,100 pixels (level-transition render),
  so d_final sits flat at ≈0.005 and falls to 0 only at the last frame.

This is consistent with Loom's own finding that the multi-game Phase-0 backbone
is blurry (single-game specialization gave 2.8× sharper pairwise latent
distances, 57%→0% dead action pairs — see Loom `RESEARCH.md`).

**Implication for the bridge:** "encode CONTRA's goal state, use latent
distance as CEM cost" is falsified for the current checkpoint. Before the
goal-cost bridge is designed, one of these has to happen first:

1. An **SP80-specialized** LeWM encoder trained on `sp80_lewm_episodes_v0`
   (now exists, full trajectories) — then re-run this probe; or
2. Loom's **Sub-JEPA** variant (33× better pred MSE, not yet integrated) —
   re-run this probe against it; or
3. A non-latent-distance cost formulation (e.g. CONTRA-evaluated symbolic
   predicates on decoded/predicted states — requires Loom's planned decoder).

Re-running the probe against a new checkpoint is one command:
`python3 -B codexlab/lewm_bridge/latent_distance_probe.py --checkpoint <path>`.

## When to pick this up

Integrate CONTRA→LeWM only when a CONTRA-side trigger fires **and** the
LeWM-side gate is clear:

- **Trigger A:** the crucible mechanic-induction loop runs dry on L1
  (consecutive iterations admit nothing new, no solve) — symbolic rule grammar
  exhausted.
- **Trigger B:** new levels keep demanding level-specific rules that don't
  transfer — evidence the generalizing component is due.
- **LeWM gate:** Loom clears its O_pred blocker (B2 in
  `Loom/lewm_preflight_analysis.md`) so the world model's predictions are
  inspectable, plus a checkpoint that *passes this probe*.

First steps on pickup: re-run `export_episodes.py` (picks up all newly admitted
runs), train/obtain an SP80-specialized or Sub-JEPA checkpoint in Loom on
`sp80_lewm_episodes_v0`, re-run the probe, and only then design the CEM
`cost_fn` bridge. Also resolve the canon tension in Loom: its ownership matrix
says the Lewis-role informer "cannot shape reward directly" — CONTRA supplying
a cost function amends that, and should be an explicit decision.

## Boundaries

- Episode rows and probe reports carry truth-boundary blocks; they are
  observation/diagnostic evidence, never control or solve truth.
- Nothing in this folder is registered in `dev_check.py` (cross-repo torch
  dependency; bridge is quarantined from the CONTRA loop on purpose).
