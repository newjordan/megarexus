# HANDOFF — synthesis_lab (megarexus), written 2026-07-01

For a fresh agent session picking up this work. Read this top to bottom
before touching anything. The two notes files are the ground truth:
`notes/megarexus_cnn_scout.md` (full experiment detail) and
`notes/RESEARCH_LOG.md` (dated summary entries). User memory lives in
`~/.claude/projects/-home-frosty40-megarexus/memory/`.

## 0. ACTIVE SESSION HANDOFF (2026-07-02, ~12:45)

Fresh session started mid-/loop for token efficiency (user's call).
Sufficient context = this section + §2 (rules) + §4 (results) +
ADDING_CNN_ELEMENTS.md (repo root) + the last two RESEARCH_LOG
entries. Do NOT re-read model.py/dataset.py/train.py or old probe
transcripts unless you are editing them — probe JSONs and run configs
are the evidence. Python is `/home/frosty40/miniconda3/bin/python3 -B`;
run from the repo root with absolute or repo-relative paths (cd does
not persist); NOT a git repo.

**Active /loop directive (user, verbatim):** "[interval = test based,
sleep between cnn runs] train one small element at a time, ensuring no
loss, focusing on optimizing all existing untrained systems before
adding new ones, use logic, best candidate applies, nothing is sacred,
no cheating." Protocol: long jobs (collect/train/probe) run as
BACKGROUND Bash tasks; end every turn with ScheduleWakeup (3600s max —
match shorter test durations when known) carrying the /loop prompt
verbatim; the old session's background-completion notifications do NOT
reach a new session — poll with `pgrep -af train.py` + `ls` on the run
dir instead.

**UPDATE 2026-07-02 15:20 — BABYSITTER MODE (lighter-model session OK
until late afternoon).** v9 at epoch ~34/40, ETA ~15:50 CDT. Gate
steps 1–2 below are AUTOMATED: `synthesis_lab/orchestration/
run_v9_gates.sh` is already running in the background — it waits for
train.py to exit, then runs gate_eval ×3 (v9 last / v9 best / v8
baseline, on v8's valset) and level_probe + arrange_probe on symlinked
scratch run dirs. Outputs → `runs/outcome_v9_gates/` (timeline.txt =
progress, DONE = finished, train_log_tail.txt = final train summary).

Babysitter scope — each wake do ONLY this: (1) `pgrep -af train.py`;
(2) `tail runs/outcome_v9_gates/timeline.txt` + check for DONE;
(3) glance at the newest `runs/outcome_v9_gates/*.log|*.txt` for
tracebacks. If the runner died before DONE and training is gone,
relaunch `bash synthesis_lab/orchestration/run_v9_gates.sh` as a
background task (it is idempotent, overwrites its own outputs). Do
NOT: interpret gate numbers, make release decisions, edit code or
notes, start new training/collection, or write into canonical
`runs/outcome_v9`. When DONE appears, report "gates complete, awaiting
adjudication" and keep sleeping. End EVERY turn with ScheduleWakeup
carrying the /loop prompt verbatim (§ directive below): ~1200s near
the ~15:50 train→gates handover, 2700–3600s otherwise. Old sessions'
background-task notifications do NOT reach a new session — poll files.
Adjudication (no-loss comparison, probe 20/20 & 10/10 verdicts,
release decision, docs, next element) is reserved for the main
session: it resumes at "WHEN v9 LANDS" step 1 using the already-
written outputs in `runs/outcome_v9_gates/` instead of re-running.

**IN FLIGHT: outcome_v9 training** (current element, task #5 = pure
object-control curriculum; collector `orchestration/
collect_control_play.py`, shards verified: train 27,713 transitions /
10,935 control samples both labels all four axes / 2,460 click rows).
Launched 2026-07-02 ~12:20, 40 epochs, ETA ~15:45–16:15 (v8 took
11,884s on less data). Recipe = v8 shard list + control_play_train/val
(recoverable from `runs/outcome_v9/*.pt` → `ckpt["config"]`). Log
(absolute, survives session change):
`/tmp/claude-1000/-home-frosty40-megarexus/011566a1-0d5a-40de-b357-367afd13eb4b/tasks/b4jen916k.output`
Done when `runs/outcome_v9/last.pt` exists and the log prints the
final summary.

**WHEN v9 LANDS — gate sequence (all commands verified this session):**

1. Same-valset no-loss gate, BOTH checkpoints, vs v8 on v8's valset:
   `python3 -B synthesis_lab/orchestration/gate_eval.py --ckpt
   synthesis_lab/experiments/megarexus_cnn/runs/outcome_v9/last.pt
   --val-shards-from .../runs/outcome_v8/last.pt` — then the same with
   `--ckpt .../outcome_v9/best.pt` and `--ckpt .../outcome_v8/last.pt`
   (the baseline side), compare head-by-head. No head may lose more
   than borderline noise (see RESEARCH_LOG on v5's 1-sample catch dip).
2. Behavioral gates (probes load `<run-dir>/best.pt` by NAME — make a
   scratch dir in YOUR scratchpad, symlink the gated checkpoint as
   best.pt, point --run-dir there, never at the canonical run dir):
   - `orchestration/level_probe.py` — must reproduce 20/20 model vs
     5/20 random (seeds as in the script defaults).
   - `orchestration/arrange_probe.py --search exhaustive-x` — must
     hold 10/10 model vs 1/10 random (paired seeds 20280001–10).
3. v9-specific checks (the element under test is control coverage):
   control/acc must hold 1.0 on v8's valset AND be strong on
   control_play_val (in-training log prints it); worth checking
   whether plain `rank_movables` (non-imagined) now ranks the
   platforms positive — if yes, note it: the imagined-click
   composition in arrange_probe could simplify.
4. Then: release decision, update §3/§4 here + RESEARCH_LOG entry,
   mark task #5 done, pick the next element per the loop directive
   (§6/§7 backlog; level-3 generalization of the planner is the
   natural probe-side candidate).

Known open observation for a future element: the learned glove finder
is appearance-sensitive — a selected (recolored) glove breaks it
(collect_control_play.py header, smoke eps 0/3). Level-1 probing
stays off until the finder tolerates recoloring.

## 1. Mission

Synthesize the user's separate systems into ONE genuine trainable CNN in
`synthesis_lab/` — "without cheating":

- **LeWM/Rex** — world model / action owner (conv dynamics, FiLM action
  conditioning),
- **Contra** — perception/thinking (components as input planes),
- **Stochastic Goose** — runtime (4101-logit action head, stochastic
  collection),
- **Portband** — the "present": moveable objects + per-side port edges
  (gloves, catch labels).

Target game: ARC-AGI-3 SP80 ("liquid pour", game_id `sp80-589a99af`),
core mechanic "catch and redirect paint". 7 levels (0–6). All runs are
local OFFLINE (`arc_agi.Arcade`, `environment_files/`), labeled
SCOUT_EXPERIMENT — never benchmark claims, never live scorecards.

## 2. Honesty rules (user rulings — NON-NEGOTIABLE)

1. **Answer-key ban** (user, emphatic): hand-authored models of a
   specific game's mechanics/win conditions (e.g.
   `codexlab/contra_head/sp80_spill_model.py`) must NEVER touch training
   — not as labels, informer, agreement metric, or ghost verifier. "We
   are training systems, not the answers." Labels never come from live
   engine internals, sprite tags, or game source.
2. Win/outcome semantics are learned from **environment fields +
   pixels in hindsight only** (`levels_completed`, `state`, frame
   diffs). Human demonstrations (corpus) ARE legitimate experience.
3. Every trained element gets baselines (persistence, majority, base
   rate), unseen-seed val shards, one backlog element per run,
   ablation-gated. Negative results are documented, not buried.
4. Ownership language: say which role owned an outcome (world model =
   imagination, controllability head = selection, env fields = level
   evidence). Scripted/demonstration-derived rollouts must never be
   presented as model capability (see §5, level-2 completion).
5. User guidance is legal as research direction (what to measure), and
   demonstrations are legal as data; guidance must not become a
   hand-coded label/reward.

## 3. Key files

- Model: `synthesis_lab/experiments/megarexus_cnn/model.py` —
  MegarexusCNN, 3.64M params. Trunk (h16/h64, 2-frame stack), FiLM
  dynamics, Goose head, catch head (glove-plane masked pooling), inverse
  dynamics, and outcome_v4 heads: `forward_settle` (settle+level-up),
  `forward_control(frame, plane, action_id)` (controllability).
- Data: `.../dataset.py` — hindsight labels; control samples via
  `MOVE_ACTION_AXIS` + displacement rule (`on_axis*sign >= 0.4`,
  `|off_axis| <= 1.0`); `has_control` aux rows gated OUT of main losses.
- Train: `.../train.py` — run: `python train.py --run-name <name>` from
  the experiment dir. Best checkpoint → `runs/<name>/best.pt`.
- Probes (in `synthesis_lab/orchestration/`):
  - `level_probe.py` — paired model-vs-random full episodes. Current v3:
    learned glove finder (`--finder learned`), miss-gated + budget-aware
    `careful_place`, per-pour side-block/gauge logging. Reports →
    `runs/<run>/level_probe*.json`.
  - `behavioral_catch_probe.py` — `imagine_pour_column` (world-model
    ghost), `move_n`, `model_place`, `random_place`.
  - `collect_catch_play.py` — curriculum collector (`find_gloves`
    interaction probe, `EpisodeRecorder`, `step`, `track_glove`).
- Adapter: `synthesis_lab/engine_adapters/portband_glove_adapter.py` —
  `frame_components`, `glove_ports` (Portband `_object_ports`),
  `catch_label(frames, mask, ports, sides=..., max_frames=...)`,
  `component_displacement`.
- Human corpus: `corpus/sp80_liquid_pour/manifest.jsonl` — 1245 frames,
  6 episodes, per-frame `level`/`state`/`action`/PNG + spoken
  transcript. All episodes reach level 4+; one full WIN
  (`morning_liquid_pour_2..._episode_02_winbank_005`).
- Current release checkpoint: `runs/outcome_v8/last.pt` (NOT best.pt —
  release is chosen by the external same-valset + behavioral gates;
  probes load `best.pt` by name, so symlink last.pt as best.pt in a
  scratch dir).

## 4. Verified results so far (all on unseen-seed val / paired seeds)

- World model: h=1 changed-cell acc 1.00, flow acc 0.987, beats
  persistence; inverse dynamics 0.990.
- Catch head (v3): 1.000 vs 0.528 base (106 pours).
- Behavioral catch: 77/80 strict catches vs 30/80 random.
- Controllability head (outcome_v4): 1.000 vs 0.528 base (1,848
  samples); ranks level-2 bar first (+4.31 vs ≤−14).
- Level-up head: 3/3 tp, 0 fp on 84 val pours (thin positives — 18
  train).
- Level probe: model **20/20 level-1 completions** (learned finder;
  was 10/20 with interaction probing), random baseline 5/20 reproduced
  exactly. Settle head: dead on thin v4 data, alive at IoU 0.656 with
  v8's plumbing coverage.
- outcome_v8 (release, 2026-07-02): level head 109/109 tp 0 fp on
  diverse positives, control 0.999 (n=5,682), all v7 gates hold.
- **Arrange planner (2026-07-02): model 10/10 plumbing completions vs
  1/10 paired random** (`orchestration/arrange_probe.py --search
  exhaustive-x`). Offsets × last-mover enumerated in imagination,
  scored by the model's own level head; chosen plan = platform2 +2,
  platform1 +3, bar +1, bar last (+19.95, runner-up +3.09);
  imagined-vs-real agreement 1.0. First completions where arrangement,
  scoring, and execution order all came from learned heads. Scope
  caveats in RESEARCH_LOG (level-2 layout deterministic across seeds;
  plan sits in the demo basin; level-3 generalization untested).

## 5. Level-2 mechanic — established 2026-07-01 (big session)

Chain: level probe v3 → diagnostics (session scratchpad, gone; results
in scout notes) → human-corpus mining → verified completion.

- **Pour supply**: each level grants ~4 animated pours; the 5th pour
  call returns 1 frame and instant GAME_OVER. Pours are the scarce
  resource — the user's "you can't miss".
- **Gauge quirks**: drains COLOR→COLOR (14→0) so non-background counts
  are blind; row 0 on level 1 but **row 63 on level 2**; ~1.5
  cells/action; NOT the level-2 killer (moves-only survives to 58/64).
- **Flow**: level 2 is an UPWARD jet from a bottom spout (x40-43); it
  splits on the bar's underside; risers climb both flanks (bar_x0−4..−1
  and bar_x1+1..+4) toward three funnel mouths (x16-19/32-35/48-51).
- **Movables**: the bar AND both 8-colored platforms move via
  click-select + move, all 4 axes, ±4px/step. Bar position alone
  changes nothing; the platforms are the real flow-blockers.
- **Offline vs live**: humans pour first and steer live; the offline
  env settles the whole flow inside `step(5)`, so live steering does
  NOT transfer. The offline solve is **arrange plumbing first, pour
  last**: platform2 right×2 → [36,24,47,27], platform1 right×3 →
  [20,16,31,19], bar right×1 → [24,36,43,39], then one pour →
  `levels_completed=2`. Verified on seed 20280001; level-2 layout is
  deterministic across seeds.
- **This completion is demonstration-derived and scripted** — never
  present it as model capability; never hardcode the configuration as
  label/reward. Level 3: three spouts, three funnels, wider platforms.
- Settle/fill signal: level 2 has NO persistent fill pixels (settled
  frames differ only by gauge cells). Outcome semantics = level-up
  head. Do not resurrect the fill-delta idea without new evidence.

## 6. THE NEXT TASK

~~Plumbing-play curriculum + outcome_v5 + planner~~ — **DONE 2026-07-02**
(outcome_v8 + arrange_probe 10/10 vs 1/10; see §4 and RESEARCH_LOG for
the v5→v8 saga: checkpoint policy, label leakage, coverage holes,
selection-conditional control, last-mover state).

Now: **pure object-control curriculum** (user-endorsed as the missed
foundational concept), under the active /loop directive "one small
element at a time, optimize existing systems before adding new ones":

1. Collector `orchestration/collect_control_play.py` — broad
   click/select/move coverage as its OWN curriculum, not a side effect
   of arranging: clicks on movables, non-movables, and empty cells;
   select→switch-selection transitions; moves with nothing selected
   (negatives); all four axes from varied positions. Selection recolors
   movables (9=selected, 8=not) — track by click-point/position+area.
2. This feeds the EXISTING control + dynamics heads (no new head). The
   planner provably depends on their composition
   (`rank_movables_if_clicked`: imagine the click, then query control
   on the imagined frame) — that composition path is what the
   curriculum should make robust.
3. Gate as always: same-valset vs outcome_v8, level probe 20/20,
   arrange probe 10/10 must all hold.

## 7. Remaining backlog (see scout notes §"Shape/map systems audit")

Death head P(GAME_OVER within k | stack, action) — note the pour-supply
mechanic makes "pours since level start" the key feature and the
2-frame stack can't count; consider a supply-plane input or recurrence.
Cost-to-intercept head. Edge-adjoin head (70 trials in hand). v4
curriculum re-collection (old shards have the strip-glove bug —
completions there weren't glove-driven). Tier 4: Contra scene-surface
distillation (floor/wall/checkerboard/facing/contact-graph). Tier 5:
symmetry/lock-key + port alignment. Ghost re-run on human segments.
Outcome-weighted self-imitation once model-guided completions exist.

## 8. Gotchas

- `catch_label` sides: `("top",)` + `max_frames=8` is the strict catch;
  side labels fire on EVERY normal catch (liquid spreads down the
  flanks) — they do not isolate blocking as-is.
- Level probe pairing: random baseline must keep the interaction-probe
  finder and pour-always policy, or the 5/20 baseline breaks.
- `Arcade` OFFLINE prints scorecard-manager INFO lines — they are local
  artifacts, not live scorecard use.
- Level transitions change the scene AND the selected object; re-find
  and re-click the movable after every level-up.
- Manifest convention: row t's action produces frame t+1. Human corpus
  `level` field = levels_completed (0-indexed current level).
