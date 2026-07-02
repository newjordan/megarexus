# Manifest: megarexus_cnn — SCOUT_EXPERIMENT

One valid convolutional network synthesizing the three engines. Local,
seeded, offline. **Not a benchmark claim; no live scorecard was used.**

## Roles

| Label | What it contributes | Source it was synthesized from |
|---|---|---|
| `LEWM_REX_ACTION_OWNER` | Action-conditioned world model: conv trunk + FiLM-modulated (AdaLN-zero-style, zero-init) residual blocks → next-frame logits `[16, 64, 64]`; family-structured action embedding (base + offset + click coords) | `architecture/src/lewm/model.py`, `predictor.py` (AdaLN idea), `action_embedder.py`, `data/arc_agi_3.py` (one-hot contract) |
| `CONTRA_THINKING_ENGINE` | 6 perception input planes per frame (background, objectness, component area, centroid offset, edges, color frequency) computed by Contra's real component code | `codexlab/contra_head/controller_head.py::build_raw_color_component_packet`, `vision.py::raw_pixel_components_with_labels` |
| `STOCHASTIC_GOOSE_RUNTIME` | Goose-exact head: 5 simple-action logits + 4096 click logits (4101 total); Goose's self-supervised "did the frame change" BCE objective; Goose's sigmoid/masked/renormalized stochastic sampler (drives rollout collection) | `stochastic_goose/custom_agents/action.py::ActionModel`, `_sample_from_combined_output` |
| `GAMEOPS_SIGNAL_TRANSPORT` | Frame/action normalization: 64×64 int grids 0–15, one-hot `[16, 64, 64]`, actions 0–7, ACTION6 carries (x, y); action 0 doubles as the animation "tick" | shared contract per `codexlab/arc_play_harness.py`, `lewm_bridge/latent_distance_probe.py` |
| PORTBAND (present-state geometry) | The glove in catch-play: moveable object identified by controlled interaction (click + move + observed displacement), its per-side edge cell sets from Portband's real `_object_ports`, entering the CNN as a glove plane for the catch head | `codexlab/contra_head/portband.py::_object_ports`, `_port_row`; select-then-move contract per `codexlab/players/sp80/registry.py` |
| `ARC_SCORECARD_EVIDENCE` | **Untouched.** No scorecard opened; no level-completion claims made | — |

## Data (all real, no synthetic answers)

- `corpus/sp80_liquid_pour`: 6 human episodes, 1,245 frames → 1,239-ish
  transitions after boundary filtering; actions 1–6.
- `corpus/s5i5_snake_modulo`: 1 human playthrough, 215 frames, all ACTION6.
- Goose-driven local SP80 rollouts (`orchestration/collect_sp80_rollouts.py`,
  OFFLINE mode, uniform or flow-biased Goose sampling; ACTION5 animation
  frames expanded into tick transitions): train-only dynamics data.
- Catch-play curriculum (`orchestration/collect_catch_play.py`): glove
  found by interaction probing, swept side-to-side under real pours;
  hindsight caught/missed labels from pixels (train seeds 20261001+,
  val seeds 20262001+ — disjoint).
- Action/frame alignment convention empirically verified against SP80's
  known directional display deltas before training (row t's action produces
  frame t+1; 67/91 motion agreement vs 49/91 for the alternative).

## Anti-cheating provisions

- Validation split at episode/level granularity: sp80 `episode_005` +
  snake final level. No within-episode leakage.
- Forward dynamics is judged against the persistence baseline
  (predict next = current) on the same validation cells.
- Inverse dynamics / BC judged against the majority-action baseline;
  change head against its base rate.
- No hardcoded answers, no environment access at forward time, no
  test-time peeking at the next frame.

## Ownership language

MegarexusCNN's world-model heads act in the LeWM/Rex role; Contra informed
the input representation; Goose contributed the policy-head geometry,
objective, and rollout runtime. Nothing here "beat a level"; random
rollouts reached `levels_completed=0`, and the human corpus levels were
completed by the human player, not by any engine.
