# SP80 Spill-Planner Solve — Handoff (2026-06-15)

Branch: `spike/pufferlib-rl-contra`. Commits this session: `e822905b`, `4a50ca0a`, `84d27320` (pushed to `origin`).

## Verified status

SP80 (`sp80-589a99af`) is solved **6/6 levels (L0-L5)** on the live engine, at the strict gated bar:
- `levels_completed_delta=1` per level; **exactly one `activate_flow` each** (the DEFER gate holds — no premature flow); both `sp80_verify_episode_run.py` **and** `angel_report_verify.py --strict-png-ledger --require-gameplay-visual-contract` pass on all six. Re-run and re-verified independently (not taken on a subagent's word).
- All plans are produced by forward-model search at run time; nothing is a replay of a stored/human sequence.
- Evidence: `codexlab/runs/_gated_final/level_0[0-5]/` (each has `result.json`, `packets.jsonl`, `brain_calls.jsonl`, `angel_report/`).

## What exists now

- `codexlab/contra_head/sp80_spill_model.py` — engine-faithful reimplementation of SP80's ACTION5 spill (recovered from git `5c4d4953`; the old `contra/` dir was renamed to `contra_head/`). `extract_state(game)` + `simulate_spill(state, platform_positions={idx:(x,y)}) -> {win, overflow, filled}`. **Re-validated 48/48 exact vs the live engine** (`python -B codexlab/sp80_spill_model_validate.py`). Testbed: `codexlab/contra_human_replay_l1_testbed.py`.
- Planners/executors (standalone, all reach 6/6): `sp80_spill_solver.py`, `sp80_spill_multiflow_planner.py`, `sp80_hard_level_solver.py`, `sp80_spill_planner_groundhog_run.py` (groundhog metrics 1.0 on catch/all-bucket/completion, overflow 0.0, vs random 0.0).
- Live brain backend: **`--backend spill_planner`** (`codexlab/sp80_spill_planner_backend.py` + additive wiring in `sp80_player_harness.py`). Per decision: `extract_state` → if `simulate_spill(current)` predicts a win, emit `activate_flow`; else search a winning layout and move the next off-target movable toward it. Rotation-seeded move-map (no probe waste), BFS routing through the engine-legal rest graph, strict DEFER gate. This is the live-play path that hits 6/6 gated.
- Perception changes: goal de-shimmed (`sp80_base_controller.bucket_goal_labels` sources the goal from a perceptual lock signature — yellow `v11`/filled `v13` — verified against the `repwkzbkhxl` tag, tag demoted to verifier; `sp80_goal_localization_regression.py`; `sp80_verify_run.py` updated). Role taxonomy in `contra_head/controller_affordances.py` (oracle key→`payload_source`, target→`goal_sink`, calibration actor→`effector`). Additive `paint_objective` (payload→sink) in `bucket_goal_condition`/`goal_progress` (measurement only, not wired into control).
- Crucible Stage C/D built (`grammar.py`, `falsify.py`, `crucible_*_rules.py`, mechanic ledger, `model.disagreements`, keystone `sp80_l1_surface_redirect_chain.py`). **Keystone is a NO-GO (0/107):** the per-cell redirect grammar is the wrong representation for SP80 (the win mechanic is global/per-object flow, not per-cell actor deltas). The mechanic ledger is NOT wired into live control. Stage B discrepancy ledger was rebuilt over the re-mined 956-row corpus.

## Engine facts (source + live verified)

- Win condition: every bucket filled AND no liquid reaches a wall (`waoewejnqzc`).
- **Single-flow wins.** The engine resets all buckets on any non-winning flow (`lpqbikobah`; counter `lyremoheq`; `lose()` at ≥4 failed flows). There is **no cross-flow accumulation** — a level is won by one flow filling every bucket. The win-bank's "multiple flows per level" are human failed retries, not engine stages.
- L0-L3 are platform-only; **L4/L5 require moving DIVERTERS** (L-shaped color-15 `tuvkdkhdokr`, turn the stream 90°; verified load-bearing — displacing one breaks the win).
- The controllable EFFECTOR is the movable platform/diverter, NOT the magenta liquid (`v6`) that oracle's lock/key pass labels as actor. Boards carry a per-level rotation (`rotation_quarters`) — the executor discovers the move→grid-delta map per level; do not assume a fixed mapping.

## Reproduce / verify

```
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=codexlab python3 -B \
  codexlab/sp80_spill_model_validate.py            # 48/48 vs engine
  codexlab/sp80_spill_solver.py                    # 6/6 standalone, delta=1
# gated (per level N=0..5):
  codexlab/sp80_player_harness.py --backend spill_planner --mode episode --level-index N --max-actions 130 --out-dir runs/X/level_0N
  codexlab/gameplay_report_harness.py --run-dir runs/X/level_0N      # builds angel_report/
  codexlab/sp80_verify_episode_run.py --run-dir runs/X/level_0N      # episode verifier
  codexlab/angel_report_verify.py --strict-png-ledger --require-gameplay-visual-contract runs/X/level_0N/angel_report
```

## Open work — the board step

Goal: register activity on the ARC-AGI-3 board (one full game won + losses on others is sufficient).
- SDK `arc_agi` is installed: `Arcade`, `RemoteEnvironmentWrapper` (online), `LocalEnvironmentWrapper` (offline), `ScorecardManager`, `RestAPI`. Flow: `Arcade(arc_api_key=...)` → `open_scorecard()` → `make(game_id, scorecard_id)` → play → `close_scorecard()`. A real key (online) registers on the board.
- `codexlab/arc_play_harness.py` recovered (from `6ef1a90e`), but its perception deps `contra_shape_fit.py` / `contra_structural_sense.py` are **missing** (would need recovery or replacement). It reads the key from `.arc_api_key` (`ARC_API_KEY_PATH`) or the `ARC_API_KEY` env var. No key is currently reachable by the agent's tool-shell; the user holds it (in their interactive session). `.arc_api_key` is gitignored.
- **The only legitimate path:** the agent must PLAY the live game — the `spill_planner` policy perceiving the board from the live frame (pixels) each step and acting in real time. The online SDK provides frames, not the privileged sprite state the planner currently uses (`extract_state` reads game sprite objects), so the real task is reconstructing platform/bucket/diverter positions from the rendered frame (perceive-from-pixels — the role-taxonomy/de-shim work is the start of this). Only then does a scorecard reflect the system actually playing.

### Caution (recorded misstep)
This agent earlier proposed extracting the winning action sequences and **replaying them** through the SDK to register a board score. That is cheating — a canned script, not the agent perceiving and solving — and it is brittle and pointless. It was rejected by the user and must not be repeated. Get on the board only by the agent playing the live game for real.

## Pointers
- Memories: `sp80-spill-model-planner`, `sp80-goal-admission-calibration`, `sp80-actor-truth`, `sp80-engine-verified-mechanic`, `no-reward-hacking-cardinal-rule`, `avoid-word-honest`.
- The cardinal rule stands: a solve is only `levels_completed_delta>0` from the live engine with passing verifiers; never claim otherwise; never read/replay the answer.
