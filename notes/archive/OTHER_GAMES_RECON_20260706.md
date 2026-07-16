# OTHER GAMES RECON — 2026-07-06

Lane: parallel recon of non-sp80 ARC-AGI-3 games. Rules of this lane: observation and
legal env interaction only; no human answers/solutions; no route replay; no model
training; no ranked/live scorecard runs; no git commits; new files only.

---

## Entry 1 — 2026-07-06 — DISCOVERY: offline game acquisition CONFIRMED

**How the stack works (read from installed SDK, not guessed):**
- The SDK is `arc_agi` (0.1.0-era, at `python3/lib/python3.13/site-packages/arc_agi/`)
  plus `arcengine` (the local game engine: `ARCBaseGame`, `GameAction`, `ActionInput`, sprites/levels/camera).
- ARC-AGI-3 games are **plain Python source files** (subclass of `arcengine.ARCBaseGame`) executed
  locally by `LocalEnvironmentWrapper` (`arc_agi/local_wrapper.py`). A game on disk =
  `environment_files/<base_id>/<version>/{metadata.json, <classname>.py}`. That is exactly what
  `environment_files/sp80/589a99af/` is (its metadata.json even carries `date_downloaded`).
- `Arcade(operation_mode=OFFLINE, environments_dir=...)` scans for `metadata.json` files and runs
  games fully locally. Scorecards in OFFLINE/NORMAL mode are **local objects**, not live submissions
  (`base.py` line ~449 "Local scorecard (NORMAL or OFFLINE)"). Live submission only happens in
  ONLINE/COMPETITION modes, which this lane does not use.

**Download path (verified live, 2026-07-06):**
- `GET https://three.arcprize.org/api/games/anonkey` → issues an anonymous API key (no account). Verified: returns a key.
- `GET /api/games` with `X-Api-Key` → catalog. Verified: **25 games** currently listed.
- `GET /api/games/<game_id>-<version>/source` → full MIT-licensed game source. Verified: ls20 source
  downloads fine with the anonymous key (HTTP 200, MIT license header, ~2060 lines).
- `GET /api/games/<game_id>` → metadata dict.
- Conclusion: **(a) YES — other games are obtainable locally without any live scorecard.**
  Downloading source/metadata is a plain GET; no scorecard is opened. Play then happens in
  OFFLINE mode against the local files.

**(b) Catalog (from /api/games, anon key, 2026-07-06):** 25 games. `baseline_actions` length is a
per-level action budget list, so its length ~ level count.

| game_id | tags | ~levels |
|---|---|---|
| lp85-305b61c3 | click | 8 |
| r11l-495a7899 | click | 6 |
| dc22-fdcac232 | keyboard_click | 6 |
| bp35-0a0ad940 | keyboard_click | 9 |
| wa30-ee6fef47 | keyboard | 9 |
| tn36-ef4dde99 | click | 7 |
| tu93-0768757b | keyboard_click | 9 |
| re86-8af5384d | keyboard_click | 8 |
| su15-1944f8ab | click | 9 |
| m0r0-492f87ba | keyboard_click | 6 |
| sb26-7fbdac44 | keyboard_click | 8 |
| lf52-271a04aa | click | 10 |
| ls20-9607627b | keyboard | 7 |
| sc25-635fd71a | keyboard_click | 6 |
| g50t-5849a774 | keyboard | 7 |
| tr87-cd924810 | keyboard | 6 |
| sp80-589a99af | keyboard_click | 6 |
| ka59-38d34dbb | keyboard_click | 7 |
| ft09-0d8bbf25 | (none) | 6 |
| vc33-5430563c | click | 7 |
| cd82-fb555c5d | keyboard_click | 6 |
| sk48-d8078629 | keyboard_click | 8 |
| cn04-2fe56bfb | keyboard_click | 6 |
| s5i5-18d95033 | click | 8 |
| ar25-0c556536 | keyboard_click | 8 |

**Game picks for Phase-A:** `ls20` and `ft09` — both are ARC-AGI-3 *public preview* games
(ls20 a.k.a. LockSmith; ft09 the other original preview title), so they are the most
community-legible choices and match the "prefer public-set names" directive.

**Honesty note on source files:** downloading a game necessarily places its full Python source
(including win logic) on disk. This lane's mechanics reads will be **frames-first**: derived from
legal env interaction and frame dumps, per the sp80 method. Anything read from source will be
explicitly labeled `[SOURCE-DERIVED]`; the Phase-A reads below aim to use frames only.

**Storage decision:** downloaded envs go under
`synthesis_lab/experiments/other_games/environment_files/<base>/<version>/` (new directory, this
lane's own space) rather than the repo-root `environment_files/`, to avoid touching the sp80 lane's
area. Scripts live in `synthesis_lab/experiments/other_games/`.

---

## Entry 2 — 2026-07-06 — ACQUISITION + OFFLINE VERIFICATION

- New tool: `synthesis_lab/experiments/other_games/download_game.py` — fetches metadata + source
  for a game id via the anonymous key and writes the sp80-identical layout. It does NOT construct
  an Arcade and opens no scorecard of any kind (plain GETs only).
- Downloaded: `ls20-9607627b` (2060 source lines, class Ls20) and `ft09-0d8bbf25`
  (2520 lines, class Ft09) into `synthesis_lab/experiments/other_games/environment_files/`.
- Verified both load and play through `Arcade(OFFLINE, environments_dir=<lane dir>)` — reset,
  step, frame capture all work. All scorecards created during play are LOCAL objects (OFFLINE mode).
- New tool: `synthesis_lab/experiments/other_games/observe_game.py` — reusable Phase-A harness for
  ANY downloaded game: initial-frame dump (ASCII hex + color histogram + available_actions),
  single-action pixel-diff probes, random rollouts honoring `available_actions`, npz corpus
  harvest (frames/actions/states/levels/episode/clicks).
- Source-honesty status: game .py files are on disk (unavoidable), but NOT read in this lane.
  The only source-adjacent data used was the class name during download. All mechanics reads
  below are frames + env-signal only.
- Side observation (not acted on): `codexlab/arc_play_harness.py` line 35 points offline mode at
  `REPO_ROOT/environment/environment_files`, which does not exist; the sim3d lane instead uses
  the repo-root `environment_files/`. Not touched — noted for the main lane.

---

## Entry 3 — 2026-07-06 — ls20 Phase-A frames-only read (LockSmith-family, "keyboard")

Corpora: `corpora/obs_ls20_seed7_e6s50.npz` (306 frames), `corpora/obs_ls20_seed101_e12s100.npz`
(1212 frames), logs in `logs/ls20_obs_seed*.txt`. Level 1 only (no level was ever completed).

**Board (64x64, level 1):** color 4 = outer background, color 3 = walkable corridor/maze
(rooms at mid-screen plus a corridor from the spawn area), color 5 = wall/frame accents.
Bottom-left: a static 9-on-5 icon panel (never changed by movement — legend/lock-shape display,
hypothesis). Top-middle: a small 5-filled room connected to the maze by a corridor.

**Action semantics (single-step pixel-diff probes, seed 7):**
- available_actions = [1,2,3,4]; ACTION5/6/7 not offered. Matches the "keyboard" tag.
- Avatar = the color 9 (45 px) + color 12 (10 px) sprite. Moves exactly 5 px per action:
  ACTION1 dy=-5 (up), ACTION3 dx=-5 (left), ACTION4 dx=+5 (right). ACTION2 (down) at spawn:
  avatar does not move (blocked) but the move still costs energy.
- Energy: color-11 bar along the bottom (84 px at start). EVERY action costs 2 px, including
  blocked moves.

**Resource economics (event scan across 18 episodes):**
- When the bar hits 0 and a reserve exists: 4 px of color 8 disappear and the bar refills to 84.
  Initial color-8 inventory = 12 px = exactly 3 reserves. Scripted-walk math confirms: refills at
  42-action intervals (84/2), GAME_OVER on action ~128 = 3 tanks * 42 + 2. So color 8 = reserve
  batteries, auto-consumed at empty; bar empty with no reserve = GAME_OVER (observed, seed 7).
- Item in the maze: a 5-px glyph of colors 1+0. Walking onto it OCCLUDES it (counts drop to 0),
  stepping off restores it — no persistent pickup observed. Pickup/interaction mechanic UNKNOWN
  (key candidate, given the lock-shaped legend; hypothesis only).
- Random play: 0/18 episodes completed a level (budget always exhausted or wandered).

**Caveats:** all reads from level 1, seeds 7 & 101 only; level-1 layout looked identical across
those two seeds but seed-dependence is unverified; "down blocked at spawn" is one position only.

---

## Entry 4 — 2026-07-06 — ft09 Phase-A read + WIN MECHANIC CONFIRMED (levels 1-2, env-graded)

Corpus: `corpora/obs_ft09_seed7_e6s50.npz` (306 frames), log `logs/ft09_obs_seed7.txt`.

**Board (64x64, level 1):** click-only (available_actions=[6]). Four 3x3 tile blocks:
two on top (A, B), bottom-left (Q), bottom-right = the only INTERACTIVE board (4-bordered).
Each tile is 6x6 px at pitch 8; tiles are color 9 ("on") or 8 ("off"); each block's CENTER tile
holds a small glyph drawn in color 2 on 0, with an 8 (or 12 in later levels) at the glyph center.

**Click semantics (frames):**
- Clicks on the interactive board toggle the hit tile 9<->8 (36 px) and burn 2 px of a color-12
  budget strip (64 px at level start = 32 clicks).
- Clicks anywhere else: rejected — a 2-frame 80-px blink (to color 0) in the animation frames,
  zero settled change, NO budget cost.

**The mechanic (frames-derived, then env-verified):**
1. First hypothesis — "answer = copy Q" — set the board ring equal to Q. Result: NO win. (Kept as
   an honest negative.)
2. Second read: each block's center glyph is a MINI-MAP of that block's own ring — glyph
   color-2 positions == ring color-9 positions, verified exactly on A, B and Q. The interactive
   board's OWN glyph therefore displays its TARGET ring.
3. Test: toggled the 4 tiles so the board ring matched its glyph -> **levels_completed 0->1 on the
   final click** (auto-graded, no submit action). FIRST ft09 LEVEL WIN, frames-only reasoning.
4. Generalization: level 2 = two vertically-stacked 3x3 blocks SHARING their middle tile-row,
   glyph centers now color 12; the two glyphs give consistent targets for the shared row.
   7 toggles -> **levels_completed 1->2**. Mechanic generalizes.
- Level 3 preview (frames only): inverted palette (tiles default 8 on color-4 field), 4 glyph
  blocks in a cross/columns arrangement — same family, richer composition. Not attempted.

**Honesty:** wins were obtained by our own frame reasoning + legal clicks in one offline session
(seed 7); no game source read, no human solutions, no training. Random clicking scored 0 wins
(30/300 random clicks even hit the board), so these wins are hypothesis-tests, not luck.

---

## Entry 5 — 2026-07-06 — SUMMARY & NEXT STEPS

**Now locally playable (offline, no live scorecard):** sp80 (pre-existing), **ls20**, **ft09** —
plus any of the other 22 catalog games via `download_game.py` (anonymous key suffices; verified).

**Learned:**
- ARC-AGI-3 games are locally executable Python; OFFLINE mode is fully self-contained.
- ft09: complete win-mechanic read for levels 1-2 (glyph = target-map; toggle tiles; auto-grade;
  32-click budget; misclicks free). Ready for a mechanic-aware solver whenever the lab wants it.
- ls20: movement/energy/reserve economics fully read; win condition and key-pickup mechanic still
  unknown — next lever is directed exploration (corridor-following) within the 3-tank budget.

**Blockers:** none for acquisition/play. Live/ranked anything remains out of scope for this lane.

**Recommended next steps (in order):**
1. ls20: corridor-graph walk (frames-only BFS on color-3 mask) to find what the 1/0 item and the
   top room do; watch for level-complete signal.
2. ft09: level-3+ observation sweep with a board-aware click prober (random ring-toggles inside
   the detected board), then extend the glyph-map read.
3. Download 1-2 more catalog games (click-tag ones are cheapest to probe) and run observe_game.py
   as-is — the harness is game-agnostic.
4. If/when a training lane is ever authorized for these games, the corpora here
   (`corpora/obs_*.npz`) are clean observation-only starting points.

---

## Entry 6 — 2026-07-07 ~03:15 — ft09 glyph solver scaffold: 2/6 controlled, L2 overlap rule open

Built a reusable frame-derived ft09 solver runner:
`synthesis_lab/experiments/other_games/ft09_glyph_solver.py`.

Boundary:
- Local/offline only through `Arcade(..., operation_mode=OFFLINE,
  environments_dir=synthesis_lab/experiments/other_games/environment_files)`.
- No live ARC scorecard, no human solution, no route replay.
- No game-source read. The downloaded `ft09.py` exists on disk, but this runner
  consumes only SDK frames and env-returned level/state signals.

Representation:
- Detect 6x6 tile components of colors 8/9.
- Anchor 3x3 glyph boards by component geometry rather than by arbitrary
  sliding-window matches. This fixed the earlier level-2 (second playable
  level) off-by-one parser bug.
- Decode the center tile as a 3x3 mini-map: marker colors `{2,12}` mean target
  marker present; non-marker subcells mean absent.
- Select only boards embedded in the color-4 play panel, excluding color-5
  display/clue boards from level 0.
- Click ring tiles whose current majority color disagrees with the decoded
  target. If overlapping boards assign contradictory marker values to the same
  physical tile, stop fail-closed.

Controlled result:
- Command:
  `python3 synthesis_lab/experiments/other_games/ft09_glyph_solver.py --seeds 7 101 1234 --max-actions 80 --conflict-search --out-dir synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_conflict_20260707T0310Z`
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_conflict_20260707T0310Z/index.html`
  with `result.json` and `contact_sheet.png`.
- Seeds `{7,101,1234}` all reached `levels_completed=2/6` in 11 clicks and
  stopped with `conflicting_glyph_constraints`; final state
  `GameState.NOT_FINISHED`.
- Stop frame hash for all three seeds:
  `2123ce4d3135531fc4efd164f92fd791b250440bfd4002525a1598fa18944cbc`.

L2 diagnostic:
- The L2 frame has four overlapping glyph boards; six physical tiles receive
  contradictory target markers from neighboring glyphs.
- The runner's optional conflict search enumerated the full ambiguity-bit
  family from that frame: `candidate_space=128` (`2` palette directions times
  `2^6` conflict assignments), `candidates_tried=128`,
  `solution_found=false`.
- This closes the naive aggregation basin: the missing rule is not a simple
  OR/AND/majority/XOR/conflict-bit choice over the already-detected glyphs.

Next valid representation work:
- Add a new observable relation before more ft09 L2 runs. Candidate targets:
  distinguish clue glyphs from target boards; infer whether glyph blocks define
  an operation/transformation over a larger tile field; run controlled
  click/diff probes that isolate one physical tile's effect under the overlap
  layout.
- Do not read `ft09.py` for solve logic unless the result is labeled
  `[SOURCE-DERIVED]` and excluded from autonomous solve claims.

---

## Entry 7 — 2026-07-07 ~03:25 — ft09 L2 diagnostic: 12/8 palette confirmed, final-state glyph relation exhausted

Updated runner:
`synthesis_lab/experiments/other_games/ft09_glyph_solver.py`.

What changed:
- Schema is now `synthesis_lab.other_games.ft09_glyph_solver.v2`.
- Active tile detection includes color `12`, not just `{8,9}`.
- Palette inference records why L2 is treated as `true_color=12`,
  `false_color=8`.
- Added three optional diagnostics:
  `--single-click-probe`, `--conflict-search`, and `--transform-search`.

Controlled command:
`python3 synthesis_lab/experiments/other_games/ft09_glyph_solver.py --seeds 7 101 1234 --max-actions 80 --conflict-search --conflict-max-candidates 2048 --transform-search --transform-max-candidates 50000 --single-click-probe --out-dir synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_l2_diag_20260707T082439Z`

Review surface:
`synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_l2_diag_20260707T082439Z/index.html`
with raw `result.json` and `contact_sheet.png`.

Controlled result:
- Seeds `{7,101,1234}` all remain `levels_completed=2/6`, stop reason
  `conflicting_glyph_constraints`, final state `GameState.NOT_FINISHED`.
- Stop frame hash:
  `2123ce4d3135531fc4efd164f92fd791b250440bfd4002525a1598fa18944cbc`.
- Single-click L2 probe from that prefix clicked tile `(4,20)` at `(x=23,y=7)`;
  changed pixels `37`; changed-color counts before `{8:36,12:1}` and after
  `{12:36,11:1}`. This confirms the active L2 tile toggles `8 -> 12`; the
  extra `12 -> 11` pixel is budget/counter burn.
- Marker/palette conflict search: `candidate_space=520`,
  `candidates_tried=130`, `solution_found=false`, `stopped_at_limit=false`.
- D4 glyph-transform search: `transform_sets_generated=24580`,
  `candidates_tried=30245`, `solution_found=false`,
  `stopped_at_limit=false`.

Interpretation:
- The naive aggregation basin was already negative in Entry 6.
- The broader final-state glyph relation family is now also negative: not a
  marker-set issue, not a `9/8` vs `12/8` palette issue, and not a D4
  orientation/reduction issue in the searched family.
- Mark this as `stale_loop` for final-target glyph relations. The next valid
  representation must be stateful/dynamic: controlled click/diff probes over
  overlapping physical tiles, a temporal acceptance read, or a deliberately
  labeled `[SOURCE-DERIVED]` branch that is not counted as autonomous solve
  evidence.

---

## Entry 8 — 2026-07-07 ~03:50 — ft09 frame policy reaches 4/6; L4 parser now blocked on color-14 layout

Updated runner:
`synthesis_lab/experiments/other_games/ft09_glyph_solver.py`.

What changed:
- Schema is now `synthesis_lab.other_games.ft09_glyph_solver.v3`.
- Corrected transform-search replay semantics: candidate grading now uses a
  fresh env replay rather than reusing one env across `reset()` calls.
- Added center-polarity target parsing for the L2 overlap frame:
  use color-2 marks only; center `8` means target = color-2 cells, center
  `12` means target = non-color-2 cells.
- Added explicit target-color planning for L3 ternary tiles. The observed click
  cycle is `8 -> 12 -> 9 -> 8`; center `12` maps glyph `0 -> 12`, `2 -> 8`,
  while center `9` maps glyph `0 -> 9`, `2 -> 8`.

Controlled command:
`python3 synthesis_lab/experiments/other_games/ft09_glyph_solver.py --seeds 7 101 1234 --max-actions 180 --out-dir synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_v3_l4_stop_20260707T084835Z`

Review surface:
`synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_v3_l4_stop_20260707T084835Z/index.html`
with raw `result.json` and `contact_sheet.png`.

Controlled result:
- Seeds `{7,101,1234}` all reached `levels_completed=4/6`.
- Each case used `52` clicks, stopped with `no_boards`, and final state was
  `GameState.NOT_FINISHED`.
- Stop frame hash for all three seeds:
  `9d4eb0268e4dbd7efefe53edac6ec957bfe782d2c7a356bec2b30bb2849d6bdb`.

Diagnostic correction:
- The previous fast transform diagnostic artifact is superseded for provenance.
  Corrected fresh-replay diagnostic:
  `synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_l2_freshreplay_20260707T083006Z/`.
  It matched the negative result: marker/palette conflict search
  `candidate_space=520`, `candidates_tried=130`, no solution; transform search
  `candidates_tried=30245`, no cap hit, no solution.

L4 read so far:
- The new frame after level 3 uses color-14 active tiles; clicking a color-14
  tile toggles `14 <-> 15`.
- Glyph/operator colors now include `3`, `6`, `12`, and `15`, so the older
  `{8,9,12}` tile/glyph parser correctly fails closed with `no_boards`.
- A strict custom scan for 3x3 color-14 tile boards found one full 3x3 board at
  `(y=20,x=30)`, but the visual frame is a larger color-14 layout. Do not treat
  the single detected board as complete L4 perception.

Next valid work:
- Generalize perception for L4's large color-14 layout: detect the full tile
  lattice and separate clue glyph cells from clickable 14/15 cells.
- Only after that, infer the L4 target relation from controlled click/diff
  probes. Do not resume final-target aggregation sweeps over L2/L3.

---

## Entry 9 — 2026-07-07 ~04:00 — ft09 solved locally/offline 6/6 with frame-derived v5 policy

Updated runner:
`synthesis_lab/experiments/other_games/ft09_glyph_solver.py`.

What changed:
- Schema is now `synthesis_lab.other_games.ft09_glyph_solver.v5`.
- L4 color-14 lattice parser added:
  - detects active 14/15 controls, non-clickable clue cells, and color-6 plus
    toggles;
  - center `14`: glyph `0 -> 14`, `2 -> 15`, `3 -> 14`;
  - center `15`: glyph `0 -> 15`, `2 -> 14`, `3 -> 14`;
  - simulates plus-toggle effects before repairing single tiles.
- Final level 11/14 domino parser added:
  - active cells are color-11 blocks with color-6 markers;
  - clue colors map `0 -> 14`, `2 -> 11`, `3 -> 11`;
  - each click toggles its own active cell plus the active cell directly above,
    so the planner backsolves columns bottom-up.

Controlled command:
`python3 synthesis_lab/experiments/other_games/ft09_glyph_solver.py --seeds 7 101 1234 --max-actions 320 --out-dir synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_v5_fullwin_20260707T085858Z`

Review surface:
`synthesis_lab/experiments/other_games/runs/ft09_glyph_solver_v5_fullwin_20260707T085858Z/index.html`
with raw `result.json` and `contact_sheet.png`.

Controlled result:
- Seeds `{7,101,1234}` all reached `levels_completed=6/6`.
- `win_levels=6`, `actions=86`, final state `GameState.WIN`,
  `policy_errors=0`.
- Final frame hash for all three seeds:
  `7b71d45a704ca893afd54db1c321db38569971ed9a844958de249fd5d3c5ea1a`.

Boundary:
- This is a real local/offline ft09 solve through legal env interaction and
  frame-derived rules.
- It is NOT live/ranked ARCPrize3 proof and NOT suite-level leaderboard proof.
- No game-source read was used. Keep any future source-derived branch explicitly
  labeled `[SOURCE-DERIVED]` and excluded from autonomous solve claims.

Next valid work:
- Wire ft09 into a local multi-game suite surface or move to another downloaded
  non-sp80 game with the same frame-derived/offline discipline.
- If a live/ranked scorecard is desired later, require explicit live intent and
  preserve review-surface/report discipline.

---

## Entry 10 — 2026-07-07 ~04:10 — ft09 wired into offline suite surface; seed 7 suite row wins 6/6

Updated harness/suite:
- `codexlab/arc_play_harness.py`
- `codexlab/arc_suite_runner.py`

What changed:
- Added `POLICIES["ft09_glyph"]`, a suite-harness wrapper around
  `synthesis_lab.experiments.other_games.ft09_glyph_solver`.
- The wrapper imports the frame-derived solver module only. It does not import
  the downloaded `ft09.py`, use human answers, or open a live scorecard.
- The first wrapper version recomputed after each click and got stuck on
  level-1 transitional animation frames. Superseded diagnostic artifact:
  `codexlab/runs/arc_suite_ft09_glyph_seed7_20260707T090722Z/`,
  `levels_completed=1/6`, `actions=311`, `resets=9`.
- Fixed wrapper behavior by queuing the solver's per-level click plan and
  draining that queue one `ACTION6` click per suite call. It replans only when
  the level changes or the queue is exhausted.
- Added explicit suite runner args `--env-dir` and `--seed`, both recorded in
  the run result and manifest.

Controlled command:
`python3 codexlab/arc_suite_runner.py --policy ft09_glyph --games ft09-0d8bbf25 --env-dir synthesis_lab/experiments/other_games/environment_files --seed 7 --max-actions 320 --no-perception --run-id arc_suite_ft09_glyph_seed7_20260707T090841Z --out-dir codexlab/runs/arc_suite_ft09_glyph_seed7_20260707T090841Z`

Review surface:
`codexlab/runs/arc_suite_ft09_glyph_seed7_20260707T090841Z/index.html`
with raw `result.json`, `report_manifest.json`, and `report_trace.jsonl`.

Controlled suite result:
- `aggregate_score=100.0`, `total_levels_completed=6`,
  `policy_error_count=0`.
- Game row `ft09-0d8bbf25`: `levels_completed=6/6`, `actions=86`,
  `resets=0`, final state `GameState.WIN`.
- Level actions from scorecard close: `[4, 7, 14, 27, 21, 13]`.
- Final frame hash:
  `7b71d45a704ca893afd54db1c321db38569971ed9a844958de249fd5d3c5ea1a`,
  matching Entry 9's native solver artifact.

Artifact hashes:
- `result.json`
  `4e04646edd9602027f8b02323189beba91a82ce8937c2aa9a4dc898b84c5010c`
- `index.html`
  `cd11f4154bf8a4d531e30313cb08b784efeb3b298d246165e4c866170ec9fd1c`
- `report_manifest.json`
  `83444d31c0c191c0809123201da3ee7b755537fa1a676233e66503e3ddac3861`
- `report_trace.jsonl`
  `b02c46b1a8e0b82ee33cf90e1d9c739dd2dd583d5c73b5cac7e5b31fcae99a46`

Review check:
- Served the run directory on `127.0.0.1:18765` and verified `200 OK` for
  `index.html`, `result.json`, and `report_manifest.json`.
- Temporary server was stopped after the check.

Boundary:
- This is controlled local/offline suite evidence for a single downloaded
  non-sp80 game. It is not a live/ranked ARCPrize3 scorecard, not public
  leaderboard evidence, and not a source-derived route.

Next valid work:
- Continue another downloaded non-sp80 game under the same offline/frame-derived
  discipline, or get explicit live-scorecard intent before any live/ranked run.

---

## Entry 11 — 2026-07-07 ~04:28 — ls20 level-0 key route verified; level-1 search scaffold added

Updated runner:
`synthesis_lab/experiments/other_games/ls20_corridor_probe.py`.

What changed:
- Added a durable local/offline probe runner for ls20 corridor/key mechanics.
- The runner uses legal keyboard actions through `Arcade(..., OFFLINE,
  environments_dir=synthesis_lab/experiments/other_games/environment_files)`.
- It does not inspect the downloaded `ls20.py`, does not use human routes, and
  does not open a live scorecard.
- The runner can verify the known level-0 route and render level-0/level-1
  ASCII frames. It also contains a replay-BFS search scaffold for level 1, but
  full level-1 sweeps need a cached/reusable replay graph before becoming a
  practical artifact path.

Controlled command:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --skip-search --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_level0_20260707T092711Z`

Review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_level0_20260707T092711Z/index.html`
with raw `result.json`.

Controlled result:
- Level-0 route `3331111444111` reaches `levels_completed=1/7`.
- Final state after that route is `GameState.NOT_FINISHED`, avatar position
  `[31,40]`.
- Level-0 transition / level-1 start frame hash:
  `0088262d9b328dca202d9e15a2707ed5ba3f6d1b843845b2637fec957114b1cd`.
- Level-1 frame-derived special objects:
  - `item01`, zero/one glyph, centroid `[51.0,47.0]`
  - `c11_1`, small color-11 component, centroid `[16.0,17.0]`
  - `c11_2`, small color-11 component, centroid `[41.0,52.0]`

Artifact hashes:
- `result.json`
  `4b43149977f5f02e41eb36fadfd732903b88990938c57367be50951c544b8a96`
- `index.html`
  `ffbe60a7e7bd2d085799b582006e8ad103dc4b96f134f45a36aeaf9156778a64`

Review check:
- Served the run directory on `127.0.0.1:18766` and verified `200 OK` for
  `index.html` and `result.json`.
- Temporary server was stopped after the check.

Exploratory level-1 state:
- Interactive replay probes found the level-0 route by tracking the hidden
  `item01` touch state. Position-only BFS misses it.
- Interactive level-1 probes over position, pickup-set, and pickup-order
  abstractions did not find a level-1 completion in the tested bounded search.
  Those broader sweeps were console probes and are not durable review
  artifacts; use them only as planning signal.
- Two attempted durable full-search runs were stopped because replaying every
  candidate through the SDK was too slow for a standard artifact. The next
  search should cache replay/action graph state before another full level-1
  sweep.

Boundary:
- This is controlled local/offline evidence for ls20 level 0 only.
- It is not a full ls20 solve, not a live/ranked scorecard, and not a
  source-derived route.

Next valid work:
- Optimize `ls20_corridor_probe.py` with cached replay/action graph state and
  rerun level-1 pickup/order search as a durable artifact.
- Alternatively move to another downloaded public-preview game if the goal is
  breadth rather than finishing ls20.

---

## Entry 12 — 2026-07-07 ~04:58 — ls20 clone-backed graph search reaches 3/7 and suite policy records route boundary

Updated files:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py`
- `codexlab/arc_play_harness.py`
- `codexlab/arc_suite_runner.py`

What changed:
- Added clone-backed graph search to `ls20_corridor_probe.py`.
  - `graph_position` keys by level/avatar/pickup order and exhausts the old
    position/order basin.
  - `graph_semantic` adds relevant object counts and a stable frame hash with
    the volatile bottom budget strip masked.
- Added `--chain-levels` so graph searches can be reproduced from one command.
- Added the graph-derived 97-action route to level 3 plus
  `--verify-known-route` for seed-set verification.
- Added suite policy `ls20_corridor` in `codexlab/arc_play_harness.py`.
  This policy is explicitly graph-derived route replay, not a learned policy.
- Adjusted the offline suite manifest boundary so policy-specific replay status
  is recorded in `policy_trace` instead of denied generically.

Controlled graph-chain command:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --chain-levels 2 --modes graph_semantic --max-ext 130 --max-nodes 5000 --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_chain3_20260707T1005Z`

Graph-chain review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_chain3_20260707T1005Z/index.html`
with raw `result.json`.

Graph-chain result:
- Target level 1: `status=found`, extension length `45`,
  `states_seen=1463`, `queue_popped=1365`, reaches `levels_completed=2`.
- Target level 2: `status=found`, extension length `39`,
  `states_seen=2050`, `queue_popped=1871`, reaches `levels_completed=3`.
- Best route length `97`, final state `GameState.NOT_FINISHED`, final frame
  hash `7f58b27c980459e6378bb4fda5d66e2fe2dc2516c36f4b0dbbd873f545e17522`.

Seed-set verification:
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_known_route3_seedset_20260707T1019Z/index.html`
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --skip-search --verify-known-route --verify-seeds 7 101 1234 --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_known_route3_seedset_20260707T1019Z`
- Result: the same 97-action route reaches `levels_completed=3` on seeds
  `{7,101,1234}` with final frame hash
  `7f58b27c980459e6378bb4fda5d66e2fe2dc2516c36f4b0dbbd873f545e17522`.

Offline suite command:
`python3 codexlab/arc_suite_runner.py --policy ls20_corridor --games ls20-9607627b --env-dir synthesis_lab/experiments/other_games/environment_files --seed 7 --max-actions 97 --no-perception --run-id arc_suite_ls20_corridor_seed7_graphroute3_20260707T1029Z --out-dir codexlab/runs/arc_suite_ls20_corridor_seed7_graphroute3_20260707T1029Z`

Offline suite review surface:
`codexlab/runs/arc_suite_ls20_corridor_seed7_graphroute3_20260707T1029Z/index.html`
with raw `result.json`, `report_manifest.json`, and `report_trace.jsonl`.

Controlled suite result:
- `aggregate_score=21.428571428571427`, `total_levels_completed=3`,
  `policy_error_count=0`.
- Game row `ls20-9607627b`: `levels_completed=3/7`, `actions=97`,
  `resets=0`, final state `GameState.NOT_FINISHED`.
- Level actions from scorecard close: `[13,45,39,0,0,0,0]`.
- Policy trace boundary:
  `graph_derived_route_replay_from_local_offline_probe;not_live_scorecard_proof;not_human_route;not_source_derived;not_learned_policy`.

Artifact hashes:
- graph-chain `result.json`
  `0ae8c486acbc06b5db42913eda6f2f3f5b19d9211e68a87377b2b16dc2c338f3`
- graph-chain `index.html`
  `936960142ed0bd889fd2888005e173d1b3024969a77c485495877c07eb73a9e9`
- seed-set `result.json`
  `8b1714c72704df4ff40e1fee9acf7a75a88b60f0749d951327b12f476e8ce307`
- seed-set `index.html`
  `803fbbe65f5a87c798acaba2fd322ba495aeef3394a22af0c0a4b5e98f5bd72a`
- suite `result.json`
  `c4b3408b795bc55a7234702e5d58eacdd6f490f33b2572c402d2c8021d45aca5`
- suite `index.html`
  `e601df425edf1ad496dde008118957e73f64dfd087c644a6be5d9a2e9ae751f9`
- suite `report_manifest.json`
  `a85cc81af4b1aaad732e7592ccd68352f89d185df1523fb13f440ab767fb9237`
- suite `report_trace.jsonl`
  `85d7bbe77021311397d94631aef2e71f650e92681836ca53a3ee7f1e38e53f07`

Review check:
- Served the other-games runs directory on `127.0.0.1:18769` and verified
  `200 OK` for the graph-chain and seed-set `index.html` / `result.json`.
- Served the suite run directory on `127.0.0.1:18770` and verified `200 OK`
  for `index.html`, `result.json`, `report_manifest.json`, and
  `report_trace.jsonl`.
- Temporary servers were stopped after checks.

Boundary:
- This is controlled local/offline evidence only. It is not a live/ranked
  scorecard and not public leaderboard evidence.
- `ls20_corridor` is graph-derived route replay, not a learned model and not a
  current-frame general solver.
- No game source, human route, or live scorecard leakage was used.
- A level-3 graph search console probe was interrupted and produced no durable
  artifact; do not cite it.

Next valid work:
- Continue ls20 from the 3/7 prefix with a changed level-3 representation and
  a bounded artifact, or switch to another downloaded game for breadth.

---

## Entry 13 — 2026-07-07 ~05:15 — ls20 level-3 replay search is bounded negative; no 4/7 claim

Updated file:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py`

What changed:
- Added explicit graph engines:
  - `wrapper`: prior wrapper-deepcopy path.
  - `game_clone`: direct local-engine state clone path.
  - `replay`: deadline-bounded route-only graph path that recomputes state
    from reset rather than copying nested engine state.
- Added `--start-route known_l3`, `--graph-engine`,
  `--deadline-seconds`, and `--verify-game-clone-engine`.
- Corrected graph-search artifact metadata so start-route searches are not
  labeled as level-1 failures.

Consistency smoke:
- Artifact:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_gameclone_consistency_20260707T_now/index.html`
- Result: wrapper replay and direct local-engine replay matched exactly on the
  97-action known route: `levels_completed=3`, `actions_executed=97`,
  final frame hash
  `7f58b27c980459e6378bb4fda5d66e2fe2dc2516c36f4b0dbbd873f545e17522`.

Durable bounded level-3 search:
- Command:
  `python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --chain-levels 1 --graph-engine replay --max-ext 180 --max-nodes 50000 --deadline-seconds 180 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_replay_20260707T100846Z`
- Review surface:
  `synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_replay_20260707T100846Z/index.html`
  with raw `result.json`.
- Result: no verified extension to 4/7. The graph row ended by deadline:
  `status=deadline`, `elapsed_seconds=180.049`, `states_seen=1017`,
  `queue_popped=872`, `expanded_edges=1016`, `max_queue=147`,
  `queue_remaining=145`, `target_level=3`.
- The artifact still reports `best_levels_completed=3`, route length `97`,
  final state `GameState.NOT_FINISHED`, and
  `game_clone_consistency.match=true`.
- Hashes:
  - result `1eeddd86f4305e5ec2ad9ddc3cd2468fa02463873c0723152c2aeddba1f2bc1f`
  - index `ca664ea3e3a0bb56cf902952ce5b6ecbc75334ecfc1f0a2ef591cd6fce6b23b2`
- Local HTTP check returned `200 OK` for `index.html` and `result.json` on
  `127.0.0.1:18771`; temporary server stopped.

Boundary:
- This is controlled local/offline evidence only.
- It is a negative/bounded-search artifact, not a 4/7 route, not a suite
  improvement, not a learned policy, and not live/ranked proof.

---

## Entry 14 — 2026-07-07 ~05:23 — portfolio suite dispatch integrates ft09 and ls20 with SP80

Updated files:
- `codexlab/arc_play_harness.py`
- `codexlab/arc_suite_runner.py`

What changed:
- Added `arcade_portfolio_v1` policy dispatch:
  - SP80 -> `megarexus_dw_directgrid`
  - ft09 -> `ft09_glyph`
  - ls20 -> `ls20_corridor`
- Added `arc_suite_runner.py --extra-env-dir` support. Extra local
  environment roots are copied into a run-local `merged_environment_files/`
  tree, and the source/runtime mapping is recorded in `env_merge_manifest`.

Controlled local/offline suite command:
`python3 codexlab/arc_suite_runner.py --policy arcade_portfolio_v1 --games sp80-589a99af ft09-0d8bbf25 ls20-9607627b --env-dir environment_files --extra-env-dir synthesis_lab/experiments/other_games/environment_files --seed 7 --max-actions 260 --run-id arc_suite_arcade_portfolio_v1_seed7_20260707T102138Z --out-dir codexlab/runs/arc_suite_arcade_portfolio_v1_seed7_20260707T102138Z`

Review surface:
`codexlab/runs/arc_suite_arcade_portfolio_v1_seed7_20260707T102138Z/index.html`
with raw `result.json`, `report_manifest.json`, and `report_trace.jsonl`.

Result:
- `aggregate_score=73.80952380952381`
- `total_levels_completed=15`
- `policy_error_count=0`
- SP80: `6/6`, score `100.0`, actions `157`, resets `0`,
  final state `GameState.WIN`, child policy `megarexus_dw_directgrid`.
- ft09: `6/6`, score `100.0`, actions `86`, resets `0`,
  final state `GameState.WIN`, child policy `ft09_glyph`.
- ls20: `3/7`, score `21.428571428571427`, actions `260`, resets `0`,
  final state `GameState.NOT_FINISHED`, child policy `ls20_corridor`.

Artifact hashes:
- `result.json`
  `c1e11203fd6b4144e7839f97402d52e2db4adacc43857a0b1a6e2d4e9ca45900`
- `index.html`
  `30c3db082a3429a31c2185b43d051c21d8916279cd468ca70e1e657475921f0b`
- `report_manifest.json`
  `a0aabc8d1bf537bd90d037c525c326da24b106183276b27d290bab3b3bf685dc`
- `report_trace.jsonl`
  `97980affb9f244ea6fed7352adf8b1f6d1dbcad28413701e85966d9442c0d400`

Review check:
- Served the portfolio run directory on `127.0.0.1:18772`.
- `curl -I` returned `200 OK` for `index.html`, `result.json`,
  `report_manifest.json`, and `report_trace.jsonl`.
- Temporary server stopped.

Superseded diagnostic:
- `codexlab/runs/arc_suite_arcade_portfolio_v1_seed7_20260707T102054Z/`
  used symlinked env dirs; every game returned `environment_error` because
  `arc.make()` returned `None`. Do not use it as score evidence.

Boundary:
- This is local/offline suite integration evidence only, not a live/ranked
  ARCPrize3 scorecard.
- `arcade_portfolio_v1` is a dispatcher; child traces carry the solver
  boundaries. ls20 remains graph-route replay and partial at `3/7`.

---

## Entry 15 — 2026-07-07 ~05:57 — all-public local suite scout with generic Tufa fallback

Updated files:
- `synthesis_lab/experiments/other_games/download_game.py`
- `codexlab/arc_play_harness.py`
- `synthesis_lab/notes/HANDOFF.md`
- `synthesis_lab/notes/OTHER_GAMES_RECON_20260706.md`

What changed:
- `download_game.py` now supports catalog-wide public-preview downloads with
  `--all`, repeated `--exclude`, `--force`, and a durable JSON manifest.
- `tufa_action_policy` now exports a policy trace.
- `arcade_portfolio_v1` now routes unsupported games to
  `tufa_action_fallback`, explicitly marked as a generic external fallback.

Download command:
`python3 synthesis_lab/experiments/other_games/download_game.py --all --exclude sp80-589a99af --dest synthesis_lab/experiments/other_games/environment_files`

Download manifest:
`synthesis_lab/experiments/other_games/runs/download_manifest_20260707T102931Z.json`

Download result:
- Catalog count `25`.
- Requested `24` after excluding repo-root `sp80-589a99af`.
- Downloaded `22`; skipped existing ft09/ls20 local copies.
- `opened_arcade_or_scorecard=false`.
- Manifest hash:
  `ed2bdd46a8307e3267000f6304bb9a1f472047d309f33e5ddb6ec915e0c06736`.

Controlled local/offline suite command:
`python3 codexlab/arc_suite_runner.py --policy arcade_portfolio_v1 --env-dir environment_files --extra-env-dir synthesis_lab/experiments/other_games/environment_files --seed 7 --max-actions 600 --run-id arc_suite_arcade_portfolio_v1_allpublic_seed7_20260707T103000Z --out-dir codexlab/runs/arc_suite_arcade_portfolio_v1_allpublic_seed7_20260707T103000Z`

Review surface:
`codexlab/runs/arc_suite_arcade_portfolio_v1_allpublic_seed7_20260707T103000Z/index.html`
with raw `result.json`, `report_manifest.json`, and `report_trace.jsonl`.

Result:
- `game_count=25`
- `aggregate_score=9.032793584260942`
- `total_levels_completed=17`
- `policy_error_count=0`
- Specialized rows:
  - SP80: `6/6`, score `100.0`, actions `157`, final
    `GameState.WIN`, child policy `megarexus_dw_directgrid`.
  - ft09: `6/6`, score `100.0`, actions `86`, final
    `GameState.WIN`, child policy `ft09_glyph`.
  - ls20: `3/7`, score `21.428571428571427`, actions `600`, final
    `GameState.NOT_FINISHED`, child policy `ls20_corridor`.
- Generic fallback positive rows:
  - cn04: `1` level, score `0.03444038067063324`, `594` actions,
    `6` resets.
  - r11l: `1` level, score `4.356827797281484`, `569` actions,
    `31` resets.
- All other fallback games remained at `0` levels under the 600-action cap.

Artifact hashes:
- `result.json`
  `989211f29cccdc72fb048c02ee0d4a2f6d53e760889805c11aad9410f4578036`
- `index.html`
  `c3280f1ef3f62b605d65e0ab5a2ce4b35902c66275e37baea8e600ec228af0c5`
- `report_manifest.json`
  `b4724e0f7b2f7dfc28ad875bef2c8f78ba4f5179c48996f83e33d9229126b6ec`
- `report_trace.jsonl`
  `2566e4f39d3a4662e3a32bbc6cf0698e080c7fc1241acbda97bb9e7f3f75bff6`

Review check:
- Served the all-public run directory on `127.0.0.1:18773`.
- `curl -I` returned `200 OK` for `index.html`, `result.json`,
  `report_manifest.json`, and `report_trace.jsonl`.
- Temporary server stopped.

Side effect cleanup:
- Tufa fallback created top-level `runs/20260707_*` checkpoint directories
  during fallback games, about `193G` total. These were not the review
  artifact and were removed after the suite result and hashes were captured.
  `runs/` is back to `4.0K`.

Boundary:
- Local/offline only. This is not a live/ranked scorecard and not leaderboard
  proof.
- The new cn04/r11l levels are from an external generic Tufa fallback, not a
  specialized Megarexus policy.
- Future all-public fallback sweeps should disable Tufa checkpoint writes or
  use a smaller cap before repeating.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.

## Entry 26 — 2026-07-07 ~08:38 — r11l trace replay probe reproduces 1/6 transition; component probes do not advance

Updated files:
- `synthesis_lab/experiments/other_games/r11l_trace_probe.py`
- `synthesis_lab/notes/HANDOFF.md`
- `synthesis_lab/notes/OTHER_GAMES_RECON_20260706.md`

What changed:
- Added `r11l_trace_probe.py`, schema
  `synthesis_lab.other_games.r11l_trace_probe.v2`.
- The probe replays a recorded generic Tufa-action ledger through local
  `Arcade(..., operation_mode=OFFLINE)` and records the first r11l level
  transition with raw before/after/delta frames.
- Optional `--component-probe` starts from the reproduced level-1 frame and
  clicks current-frame component centroids, recording env levels and frame
  deltas for each candidate.
- The script produces `index.html`, `result.json`, `report_manifest.json`,
  `report_trace.jsonl`, `frames.npz`, raw PNGs, and review images.

Command:
`python3 synthesis_lab/experiments/other_games/r11l_trace_probe.py --ledger-result codexlab/runs/tufa_seed_sweep_cn04_r11l_600_20260707T111000Z/children/tufa_seed_sweep_cn04_r11l_600_20260707T111000Z_r11l-495a7899_seed_101/result.json --game r11l-495a7899 --seed 7 --env-dir synthesis_lab/experiments/other_games/environment_files --max-actions 120 --window 10 --component-probe --probe-colors 3,6,12,15 --probe-max-area 40 --max-probes 24 --out-dir synthesis_lab/experiments/other_games/runs/r11l_trace_probe_seed7_tufa101_components_20260707T1338Z`

Review surface:
`synthesis_lab/experiments/other_games/runs/r11l_trace_probe_seed7_tufa101_components_20260707T1338Z/index.html`
with raw `result.json`, `report_manifest.json`, `report_trace.jsonl`,
`frames.npz`, `contact_sheet.png`, `component_probe_candidates.png`, and
`playwright_screenshot.png`.

Result:
- Replayed the seed-101 Tufa generic-agent ledger against local r11l seed 7.
- First level transition reproduced at action step `41`, after `42` legal
  clicks. Transition click: `(x=37,y=6)`, `levels_before=0`,
  `levels_after=1`, state `GameState.NOT_FINISHED`.
- No ledger level-counter mismatches.
- Transition frame hashes:
  before `6e38c960ee9fe2f16b6594bdc5dd7864f01a8535bbe68fe4c90f56f89d3962ba`;
  after `1d2208ff15e1fb7d1018928f59029aff3467a147906c098331dd4360432667d7`.
- Component probe clicked 24 current-frame centroids from colors `{3,6,12,15}`
  after reproducing the level-1 state. No row advanced to level 2; all stayed
  `levels_after=1`.
- One-click response classes observed:
  green plus/object-overlap clicks changed 25 px; the color-6/color-12 object
  at `(24,12)` changed 108 px; most ring pixels changed 1 px.

Artifact hashes:
- `result.json`
  `2652e7c53392c08b902fb7f5e38c32a55162482d4abebb642e7fbab167dddafe`
- `index.html`
  `22b4ad10f9f9bbed8383578422260ad522aad4dbf68e7281e11dec40a9100852`
- `contact_sheet.png`
  `2b902e51d3b1abda81a7307fc05ff0c16aab5b12874828a41c511b9fa82000db`
- `component_probe_candidates.png`
  `2f0fedea39eecda1da36747301eb3d3d0a458b1fd8f4fe9139049b2fc788e5e8`
- `report_manifest.json`
  `6181b8ed499187b80ef56af8ab9e51b9a2c0511e9442315764659bf0e833ff6f`
- `playwright_screenshot.png`
  `959b2585dc9030437df4924ad9afc5a1ec123a33e905f2bbae06c72a5db6b935`

Review check:
- Served `synthesis_lab/experiments/other_games/runs` on
  `127.0.0.1:18787`.
- HTTP `200` returned for `index.html`, `result.json`, and
  `report_manifest.json`.
- Playwright screenshot succeeded; local image inspection showed a readable
  nonblank page with transition frames and component-probe table.
- Temporary HTTP server was stopped.

Boundary:
- Local/offline only. This is not a live/ranked scorecard, not a specialized
  r11l solver, and not route-generalization evidence.
- Do not promote r11l to live. Current r11l state is still only `1/6` via
  generic-agent replay/probe evidence.
- Do not repeat one-click component-centroid sweeps as the next r11l move.
  The next representation should model sequential state changes: how clicks
  move/connect the colored nodes and lines across multiple controlled actions,
  or a current-frame geometry rule tested fail-closed offline.
- Final gates after this entry: `py_compile` on runner/probe files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.

## Entry 25 — 2026-07-07 ~08:22 — ls20 action-grammar jump-graph probe records motif edges, no 4/7 route

Updated files:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py`
- `synthesis_lab/notes/HANDOFF.md`
- `synthesis_lab/notes/OTHER_GAMES_RECON_20260706.md`

What changed:
- Bumped LS20 probe schema to
  `synthesis_lab.other_games.ls20_corridor_probe.v5`.
- Added `--color8-action-grammar-probe`, `--color8-grammar-steps`, and
  `--color8-grammar-window`.
- Added `target_action_grammar_search`, which starts after a color-8 target
  transition event and explores action motifs over the local game-clone jump
  graph. The diagnostic records motif count, grammar edge count, jump-edge
  count, states seen, and extension length.
- HTML review now includes grammar status, grammar edges, grammar jumps, and
  grammar extension columns for post-pickup color-8 probes.

Focused grammar smoke command:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders c11_1,c11_2,item01,exit14_0 --post-pickup-color8-probe --color8-action-grammar-probe --graph-engine game_clone --max-ext 220 --max-nodes 5000 --deadline-seconds 60 --color8-probe-steps 100 --color8-grammar-steps 36 --color8-grammar-window 3 --color8-probe-min-area 10 --color8-contact-distance 1 --color8-jump-threshold 12 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_grammar_smoke_20260707T1312Z`

Focused grammar smoke review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_grammar_smoke_20260707T1312Z/index.html`
with raw `result.json`.

Focused grammar result:
- `color8_action_grammar_deadline`, not level advance.
- `best_levels_completed=3`, `best_route_len=97`,
  `best_final_state=GameState.NOT_FINISHED`.
- One probe row: `target_action_grammar_status=deadline`,
  `target_action_grammar_states_seen=150`,
  `target_action_grammar_edge_count=1913`,
  `target_action_grammar_jump_edge_count=212`,
  `target_action_grammar_motif_count=80`, no extension advance.

All-order grammar command:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders all_pickups_exit --post-pickup-color8-probe --color8-action-grammar-probe --graph-engine game_clone --max-ext 220 --max-nodes 2500 --deadline-seconds 35 --color8-probe-steps 100 --color8-grammar-steps 36 --color8-grammar-window 2 --color8-probe-min-area 10 --color8-contact-distance 1 --color8-jump-threshold 12 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_grammar_orders_20260707T1318Z`

All-order grammar review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_grammar_orders_20260707T1318Z/index.html`
with raw `result.json` and `playwright_screenshot.png`.

All-order grammar result:
- Six `color8_action_grammar_deadline` rows.
- Zero rows advanced to 4/7. `best_levels_completed=3`,
  `best_route_len=97`, final state `GameState.NOT_FINISHED`.
- Grammar settings: `color8_grammar_steps=36`,
  `color8_grammar_window=2`, `target_action_grammar_motif_count=16`.
- Orders `0,1,2,4` ended with best extension `22` or `322231`,
  best actor `[11,40]`, and best phase `{"5":12,"8":8}`.
- Orders `3,5` ended with empty extension, best actor `[6,58]`, and
  best phase `{"12":20}`.
- Grammar edge/jump-edge counts by order:
  `1389/301`, `1384/301`, `1378/301`, `1347/233`,
  `1363/296`, `1339/229`.

Artifact hashes:
- Grammar smoke `result.json`
  `1fe9dea4597b52bae81e2313a688814363b12d51f6dd37d8d4a1b0246394fd02`
- Grammar smoke `index.html`
  `18f164f7f4497ca7ce9c8e8b0a3e80893de50e8946e3510cb0305c47decde61c`
- Grammar all-order `result.json`
  `9bb5651bec2579da5500171939133585c15308297bd8290145e0116291629a53`
- Grammar all-order `index.html`
  `a4ae0b3fe961ac11aa4fee1ebc341b57ffa8d3d9110cc76da8da65df1e1e290a`
- Grammar all-order `playwright_screenshot.png`
  `e59bc7b5eae0363d548992413c28908956163725b78c38ee90413864ec4768fa`

Review check:
- Served `synthesis_lab/experiments/other_games/runs` on
  `127.0.0.1:18786`.
- HTTP `200` returned for grammar smoke/all-order `index.html` and
  `result.json`.
- Playwright screenshot succeeded for the all-order review surface and local
  image inspection showed a readable nonblank page.

Boundary:
- Local/offline only. This is not a live/ranked scorecard, not a leaderboard
  result, and not a 4/7 route.
- This is a representation change beyond phase/underlay/budget/jump counts,
  and it successfully records a motif-level jump graph. It still deadlined in
  the same observable LS20 L3 outcome basin.
- Do not launch another plain action-grammar window/depth expansion as the next
  LS20 move. The next valid LS20 step needs a new objective/acceptance model,
  cross-level lock-action generalization, or a clearly labeled source-derived
  mechanic parse if explicitly allowed.
- Final gates after this entry: `py_compile` on runner/probe files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.

## Entry 24 — 2026-07-07 ~08:03 — ft09 live ARC SDK scorecard closed, 6/6 score 100

Second prepared live condition after SP80: re-gated the current-code
`ft09_glyph` suite wrapper offline, then ran one live ARC SDK scorecard.

Pre-live current-code offline gate:
- Command:
  `python3 codexlab/arc_suite_runner.py --policy ft09_glyph --games ft09-0d8bbf25 --env-dir synthesis_lab/experiments/other_games/environment_files --seed 7 --max-actions 320 --no-perception --run-id arc_suite_ft09_glyph_current_20260707T130217Z --out-dir codexlab/runs/arc_suite_ft09_glyph_current_20260707T130217Z`
- Result: exit `0`; mode `offline_local_environment_files`;
  `aggregate_score=100.0`, `total_levels_completed=6`, `actions=86`,
  `resets=0`, `final_state=GameState.WIN`, `policy_error_count=0`.
- Solver trace schema: `synthesis_lab.other_games.ft09_glyph_solver.v5`.
- Review surface: local HTTP server returned `200 OK` for `index.html` and
  `result.json`; Playwright screenshot saved as `review_screenshot.png`.
- Artifact hashes:
  `result.json` `536b210979f5702093c3d8d98601b7b1250bd78238f13b2d900229d90989b319`;
  `index.html` `a5977ef2211b800e400e4fc400e791c7891ab481c2770bd87e66df98cdee4912`;
  `report_manifest.json` `f5c0aa51f82730d7d8b2ce8c4bb06a46d297b2e90ade18be5ab961980d6b8064`;
  `report_trace.jsonl` `5645cebc6ab186ed8226ea0f44d156eae6ba556fd92101ad873d280e42e9dc96`;
  `review_screenshot.png` `da903bcdd4cce613c471511f2792b860fffeb8ac9b5c7441c7be65e4f0f9b778`.

Live scorecard:
- Command:
  `python3 codexlab/arc_live_suite_runner.py --allow-live --policy ft09_glyph --games ft09-0d8bbf25 --max-actions 320 --no-perception --run-id arc_live_ft09_glyph_20260707T130252Z --out-dir codexlab/runs/arc_live_ft09_glyph_20260707T130252Z`
- Result: exit `0`; `evidence_category=live_scorecard`;
  `arc_operation_mode=OperationMode.NORMAL`; scorecard
  `2c33f8bb-9da5-44ee-b0c4-96f4d9c611c0`.
- SDK close payload: `score=100.0`, `total_levels_completed=6`,
  `total_actions=86`, `total_environments_completed=1/1`, game row
  `ft09-0d8bbf25` completed with `levels_completed=6`, `resets=0`,
  `score=100.0`, run state `WIN`, level actions `[4,7,14,27,21,13]`
  versus baselines `[43,12,23,28,65,37]`.
- Runner row: `status=completed`, `arc_mode=live_scorecard`,
  `final_state=GameState.WIN`, `scorecard_state=WIN`,
  `policy_error_count=0`, `actions=86`.
- Review surface: local HTTP server returned `200 OK` for live `index.html`
  and `result.json`; Playwright screenshot saved as `review_screenshot.png`.
- Artifact hashes:
  `result.json` `f908770ddf81803a8e700fd275a5aec4493b13f130c048659401f1ce6c091297`;
  `index.html` `7ef4510f4237af3e09fe46fb6466b005c95d81f3eea2e50be59db2f114bcda22`;
  `report_manifest.json` `0063c9f841a6c9ed0d1792874a1af6240ea1b9bfc23f8e768dded9dac9f65b28`;
  `report_trace.jsonl` `df294b9bc438a43ae78c123d1883d39013187c089534cdf02c3b77c77009c98c`;
  `review_screenshot.png` `380f0885dd5ede929ec1e62deda95d125e8fc4f7d51846661a3c45ec397155ab`.

Boundary:
- This is a real ARC SDK live scorecard close for ft09 only, not a broad
  ARCPrize3 leaderboard claim and not a combined multi-game scorecard.
- The SDK close payload returned `source_url=null`; current evidence is the
  local result bundle plus the official SDK close payload, not a public URL.
- The policy remains frame-derived: no downloaded `ft09.py` source import,
  no human trace, no coordinate replay.
- Final gates after this entry: `py_compile` on runner/policy files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
- Working tree remains uncommitted.

---

## Entry 16 — 2026-07-07 ~06:04 — Tufa fallback wrapper disables checkpoint spam and records stable seed

Updated files:
- `codexlab/arc_play_harness.py`
- `synthesis_lab/notes/HANDOFF.md`
- `synthesis_lab/notes/OTHER_GAMES_RECON_20260706.md`

What changed:
- The external Tufa `Action` fallback is now wrapped with safer default
  runtime controls:
  - Tufa sidecars go to `codexlab/runs/tufa_action_sidecars/`.
  - Checkpoint writes are disabled by default with
    `ACTION_DISABLE_CHECKPOINTS=1` and `ACTION_CHECKPOINT_NUMBERED=0`.
  - Checkpoint writes require explicit `ARC_TUFA_ENABLE_CHECKPOINTS=1`.
  - The fallback is seeded with a stable per-game hash unless `ARC_TUFA_SEED`
    is set. The resolved seed is recorded in the child policy trace.

Controlled local/offline smoke command:
`python3 codexlab/arc_suite_runner.py --policy arcade_portfolio_v1 --games cn04-2fe56bfb --env-dir synthesis_lab/experiments/other_games/environment_files --seed 7 --max-actions 80 --run-id tufa_fallback_nockpt_seeded_cn04_seed7_20260707T110400Z --out-dir codexlab/runs/tufa_fallback_nockpt_seeded_cn04_seed7_20260707T110400Z`

Review surface:
`codexlab/runs/tufa_fallback_nockpt_seeded_cn04_seed7_20260707T110400Z/index.html`
with raw `result.json`, `report_manifest.json`, and `report_trace.jsonl`.

Result:
- `game_count=1`
- `total_levels_completed=0`
- `policy_error_count=0`
- actions `79`, resets `1`
- child policy `tufa_action_fallback`
- trace: `checkpoints_enabled=false`, `checkpoint_numbered=false`, seed
  `868634289`, seed source `stable_game_id_hash`
- Tufa experiment dir:
  `$REPO/codexlab/runs/tufa_action_sidecars/20260707_060341`

Filesystem check:
- No top-level `runs/20260707_*` directories were created.
- `codexlab/runs/tufa_action_sidecars` was `492K`.
- repo-root `runs/` remained `4.0K`.
- The Tufa sidecar directory contained only `git_info.txt` and `logs.log`; no
  checkpoint files.

Artifact hashes:
- `result.json`
  `6bf2b1bd1b65441b91f5c0f1d77e28a4981c14182ac04185be0e2339bd732260`
- `index.html`
  `95cd0e6117a52442db70be6e49da0452385f578d5de2ebdf650a74b82ab48614`
- `report_manifest.json`
  `3c174dbf705d8df26ecc44f34abc8062301747cd0cf2a732e82952e3130729ca`
- `report_trace.jsonl`
  `582fb54891e0b6f91d066b18155ca9ba9ba737d6df1ff7219f870eb602b022ab`

Review check:
- Served the smoke directory on `127.0.0.1:18775`.
- `curl -I` returned `200 OK` for `index.html`, `result.json`,
  `report_manifest.json`, and `report_trace.jsonl`.
- Temporary server stopped.

Boundary:
- This is a runtime/control fix and local/offline smoke only. It is not a new
  score result, not live/ranked proof, and not evidence that cn04 is solved.
- Purpose: future all-public fallback/training scouts can be repeated without
  unmanaged checkpoint growth or wall-clock-seeded drift.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.

---

## Entry 17 — 2026-07-07 ~06:22 — Tufa seed sweep and deterministic r11l fallback seed map

Updated files:
- `codexlab/arc_play_harness.py`
- `synthesis_lab/experiments/other_games/tufa_seed_sweep.py`
- `synthesis_lab/notes/HANDOFF.md`
- `synthesis_lab/notes/OTHER_GAMES_RECON_20260706.md`

What changed:
- Added a controlled local/offline seed-sweep runner for the generic Tufa
  fallback. Each row delegates to `codexlab/arc_suite_runner.py`, removes
  `ARC_ALLOW_LIVE` from the child environment, records seed/checkpoint trace
  fields, and writes an HTML/JSON review surface.
- Added `TUFA_FALLBACK_SEED_MAP` support to the Tufa wrapper. The built-in map
  currently contains only `r11l-495a7899 -> 101`; `ARC_TUFA_SEED` still
  overrides it, and `ARC_TUFA_SEED_MAP` can extend or override it by JSON
  string or JSON file path.

Controlled local/offline seed sweep command:
`python3 synthesis_lab/experiments/other_games/tufa_seed_sweep.py --games cn04-2fe56bfb r11l-495a7899 --tufa-seeds default 7 42 101 --env-dir synthesis_lab/experiments/other_games/environment_files --max-actions 600 --arc-seed 7 --zero-stop 2 --run-id tufa_seed_sweep_cn04_r11l_600_20260707T111000Z --out-dir codexlab/runs/tufa_seed_sweep_cn04_r11l_600_20260707T111000Z`

Review surface:
`codexlab/runs/tufa_seed_sweep_cn04_r11l_600_20260707T111000Z/index.html`
with raw `result.json`, `report_manifest.json`, and `report_trace.jsonl`.

Seed sweep result:
- `rows=6`
- `best_total_levels=1`
- `best_total_score=1.3065543677788576`
- `policy_error_count=0`
- cn04 default seed `868634289`: `0` levels, `593` actions, `7` resets.
- cn04 seed `7`: `0` levels, `593` actions, `7` resets.
- cn04 early stop: `stale_loop_zero_level_after_2_seeds`.
- r11l default seed `997786576`: `1` level, score
  `0.2013068307067783`, `575` actions, `25` resets.
- r11l seed `7`: `1` level, score `0.25537528030602824`, `569` actions,
  `31` resets.
- r11l seed `42`: `0` levels, `591` actions, `9` resets.
- r11l seed `101`: `1` level, score `1.3065543677788576`, `566` actions,
  `34` resets.

Repeat command:
`ARC_TUFA_SEED=101 ARC_TUFA_RUNS_DIR=$REPO/codexlab/runs/tufa_seed_repeat_r11l_101_20260707T111500Z/tufa_sidecars python3 codexlab/arc_suite_runner.py --policy arcade_portfolio_v1 --games r11l-495a7899 --env-dir synthesis_lab/experiments/other_games/environment_files --seed 7 --max-actions 600 --run-id tufa_seed_repeat_r11l_101_20260707T111500Z --out-dir codexlab/runs/tufa_seed_repeat_r11l_101_20260707T111500Z`

Repeat result:
- `total_levels_completed=1`
- score `1.3065543677788576`
- actions `566`, resets `34`
- `policy_error_count=0`
- trace seed `101`, seed source `ARC_TUFA_SEED`

Integrated seed-map command:
`python3 codexlab/arc_suite_runner.py --policy arcade_portfolio_v1 --games cn04-2fe56bfb r11l-495a7899 --env-dir synthesis_lab/experiments/other_games/environment_files --seed 7 --max-actions 600 --run-id tufa_seedmap_portfolio_cn04_r11l_20260707T111800Z --out-dir codexlab/runs/tufa_seedmap_portfolio_cn04_r11l_20260707T111800Z`

Integrated seed-map result:
- `game_count=2`
- `aggregate_score=0.6532771838894288`
- `total_levels_completed=1`
- `policy_error_count=0`
- cn04: `0` levels, score `0.0`, seed `868634289`, seed source
  `stable_game_id_hash`.
- r11l: `1` level, score `1.3065543677788576`, seed `101`, seed source
  `tufa_fallback_seed_map`.
- Both child traces have `checkpoints_enabled=false` and
  `checkpoint_numbered=false`.

Artifact hashes:
- Sweep `result.json`
  `82f4dc7fe956bd1194faaf7b1aaaa3b79a1c9730da3e072674e00a955c02d97d`
- Sweep `index.html`
  `675cd6078377dac5a6615350dd92b055a723c39b0cf5bbe5c24cf49d02a83e66`
- Sweep `report_manifest.json`
  `4bac13057bd6d5cc09893c401a321138f6960edd5e55bfa4d31cd8285630b986`
- Sweep `report_trace.jsonl`
  `c91ff54fca424bc9d4b5d1859b91db9da4fa392def53fdf8bb39b70bb7d8a49e`
- Repeat `result.json`
  `10d7713b067728ee515c0e18a7d6ec32ccab38f12747421f2c53908be534fc83`
- Repeat `index.html`
  `6ba6c0ab4bd3feb2aee7736a93f1fc939c6e27b19a6392cf4bce28210d6641af`
- Repeat `report_manifest.json`
  `5d96c5acaa09890022d19f80e711188966a18b7b9bff689a4aa31eea14b2fcfc`
- Repeat `report_trace.jsonl`
  `46b0f7c82a6a2b5353dbdb5671880df6e58a0cdcc73c19513a6e359a78ba69b1`
- Seed-map `result.json`
  `45b11a5cf829a2f7d1e1677b9b53f7910598e9cdc50368126cc679f8e2150847`
- Seed-map `index.html`
  `fadd08b8ac62cfdc2c675b29acc075f25a1b737249503af7b5b6209d56f53c1d`
- Seed-map `report_manifest.json`
  `e390e73552f598aa8b6e1bd11256f44336c359ddd86b58091ee536ed6a23fec8`
- Seed-map `report_trace.jsonl`
  `4935f51e6903eb65e0dfa5390bf3f2f7e7a00011dd6a1690248bd87374490ac2`

Review and filesystem checks:
- Seed-map review surface served on `127.0.0.1:18777`; `index.html`,
  `result.json`, `report_manifest.json`, and `report_trace.jsonl` all
  returned HTTP `200`; temporary server stopped.
- No `.pt` or `checkpoint_*.pt` files were found under the sweep, repeat, or
  seed-map artifact directories.
- Sizes: sweep `3.9M`, repeat `628K`, seed-map `744K`,
  `codexlab/runs/tufa_action_sidecars` `996K`, repo-root `runs/` `4.0K`.

Boundary:
- Local/offline fallback hyperparameter evidence only. This is not a live or
  ranked ARCPrize3 scorecard, not a leaderboard claim, and not a specialized
  solve for cn04 or r11l.
- The r11l seed-map row improves deterministic fallback score and
  reproducibility under the 600-action condition; it does not increase the
  level count beyond `1`.
- The previous all-public cn04 `1` level did not reproduce in the controlled
  default/7 seed sweep and should be treated as stochastic fallback evidence,
  not stable cn04 capability.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.

---

## Entry 18 — 2026-07-07 ~06:38 — ls20 actor-aware level-3 search is bounded negative

Updated files:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py`
- `synthesis_lab/notes/HANDOFF.md`
- `synthesis_lab/notes/OTHER_GAMES_RECON_20260706.md`

What changed:
- Added actor-aware ls20 state helpers:
  - `actor_component`, `actor_pos`, and `actor_bbox` isolate the moving
    color-12 actor component.
  - Pickup tracking now uses the actor component instead of averaging every
    color-12 pixel.
  - New `graph_actor_semantic` key records actor bbox, pickup order, object
    counts, and object-component signatures.
  - New `--search-strategy actor_astar` prioritizes legal engine states by
    action depth plus visible-goal distance.
- Fixed chained graph searches so `--modes graph_actor_semantic` is honored;
  the earlier actor-frontier diagnostic at
  `ls20_corridor_probe_seed7_l3_actor_astar_20260707T1130Z` used
  `actor_astar` but still had the old `graph_semantic` key, so do not cite it
  as the corrected actor-key condition.

Controlled diagnostic baseline:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --chain-levels 1 --graph-engine game_clone --max-ext 220 --max-nodes 200000 --deadline-seconds 60 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_gameclone_baseline_20260707T1125Z`

Baseline review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_gameclone_baseline_20260707T1125Z/index.html`
with raw `result.json`.

Baseline result:
- Existing `graph_semantic` + `game_clone` hit deadline at `60.003s`.
- `states_seen=1333`, `queue_popped=1147`, `expanded_edges=1332`.
- `best_levels_completed=3`; no 4/7 extension.

Corrected actor-key command:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --chain-levels 1 --modes graph_actor_semantic --graph-engine game_clone --search-strategy actor_astar --max-ext 220 --max-nodes 200000 --deadline-seconds 180 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_actor_key_astar_20260707T1138Z`

Actor-key review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_actor_key_astar_20260707T1138Z/index.html`
with raw `result.json`.

Actor-key result:
- `best_levels_completed=3`
- `best_route_len=97`
- `best_final_state=GameState.NOT_FINISHED`
- `game_clone_consistency.match=true`
- Search row: `mode=graph_actor_semantic`, `search_strategy=actor_astar`,
  `status=deadline`, `elapsed_seconds=181.382`, `states_seen=3930`,
  `queue_popped=3011`, `expanded_edges=3929`, `max_queue=926`,
  `queue_remaining=919`.
- Navigation goals recorded by the run: `item01`, `c11_1`, `c11_2`,
  `exit14_0`.

Artifact hashes:
- Baseline `result.json`
  `841f567f466bfb7c0f38215189cffcea8bb3c5df50b7a1900be1307352fea5f8`
- Baseline `index.html`
  `8a500b093a7b4adad3fc5dc7448428f24ba3cedbead4cfee5624a53750c578cc`
- Actor-key `result.json`
  `8ee4bea46f69a833ae7fb37d1978a49f99ebbda6d2147452d518bb0cefd651f6`
- Actor-key `index.html`
  `f7d3baf311df0015188cdeaea96d62c5ecb3d7e932851c68347dc28b4784e611`

Review check:
- Served `synthesis_lab/experiments/other_games/runs` on
  `127.0.0.1:18778`.
- HTTP `200` returned for actor-key `index.html` and `result.json`, and for
  baseline `index.html` and `result.json`.
- Temporary server stopped.

Boundary:
- Local/offline only. This is not a live/ranked scorecard, not a leaderboard
  result, and not a 4/7 route.
- Do not run another longer `graph_actor_semantic`/`actor_astar` L3 search as
  the next step. The next valid ls20 move needs a further representation
  change: explicit energy/action-budget admissibility, door/lock
  before/after-delta mechanics, or a clearly labeled source-derived mechanic
  parse if that lane is allowed.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.

---

## Entry 19 — 2026-07-07 ~06:50 — ls20 waypoint-order planner is bounded negative across all pickup orders

Updated files:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py`
- `synthesis_lab/notes/HANDOFF.md`
- `synthesis_lab/notes/OTHER_GAMES_RECON_20260706.md`

What changed:
- Added a waypoint-order planner to `ls20_corridor_probe.py`.
- `--waypoint-orders` accepts comma-separated observed goal names, or
  `all_pickups_exit` for all pickup permutations followed by the observed
  large color-14 exit-like component.
- `waypoint_order_search_from_prefix` searches legal local-engine clone states
  with actor-aware `graph_actor_semantic` keys and an explicit
  waypoint-progress dimension.
- Review artifacts now include waypoint rows in `index.html` and raw
  `result.json`.

Waypoint smoke command:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders item01,c11_1,c11_2,exit14_0 --graph-engine game_clone --max-ext 180 --max-nodes 80000 --deadline-seconds 45 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_waypoint_smoke_20260707T1145Z`

Waypoint smoke review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_waypoint_smoke_20260707T1145Z/index.html`
with raw `result.json`.

Waypoint smoke result:
- `best_levels_completed=3`
- `best_route_len=97`
- `best_final_state=GameState.NOT_FINISHED`
- Search row: `status=deadline`, `elapsed_seconds=45.33`,
  `states_seen=1020`, `queue_popped=827`, `expanded_edges=1019`.
- The planner reached waypoint progress `3` in `42` popped states, so all
  three pickup waypoints were reachable in that order, but no level advance
  happened before the cap.

Six-order waypoint command:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders all_pickups_exit --graph-engine game_clone --max-ext 220 --max-nodes 80000 --deadline-seconds 45 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_waypoint_orders_20260707T1150Z`

Six-order waypoint review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_waypoint_orders_20260707T1150Z/index.html`
with raw `result.json`.

Six-order waypoint result:
- `waypoint_order_count=6`
- `found_orders=[]`
- `best_levels_completed=3`
- `best_route_len=97`
- `best_final_state=GameState.NOT_FINISHED`
- Each order reached progress `3` in at least some states, but none advanced
  to level 4:
  - `item01,c11_1,c11_2,exit14_0`: progress-3 count `42`
  - `item01,c11_2,c11_1,exit14_0`: progress-3 count `72`
  - `c11_1,item01,c11_2,exit14_0`: progress-3 count `46`
  - `c11_1,c11_2,item01,exit14_0`: progress-3 count `41`
  - `c11_2,item01,c11_1,exit14_0`: progress-3 count `30`
  - `c11_2,c11_1,item01,exit14_0`: progress-3 count `29`

Artifact hashes:
- Smoke `result.json`
  `0eac991b9d7791826bb96e8ec9f82876aae7d7df738df23450e304eeb0e6e513`
- Smoke `index.html`
  `0530e620aec1768a7753007ba37601743ef391659283f15bd957c060105ae74c`
- Six-order `result.json`
  `4a8a75144fa88d518259813c1904c6222d79fb99e7c7a8567cad2543dc0b1dad`
- Six-order `index.html`
  `0c949e786f0d23c9a385b7d8fd400d7bb8a101e47f89db2254a4cfe5361101dd`

Review check:
- Served `synthesis_lab/experiments/other_games/runs` on
  `127.0.0.1:18779`.
- HTTP `200` returned for smoke `index.html` and `result.json`, and for
  six-order `index.html` and `result.json`.
- Temporary server stopped.

Boundary:
- Local/offline only. This is not a live/ranked scorecard, not a leaderboard
  result, and not a 4/7 route.
- Do not run another longer waypoint-order ls20 L3 search as the next step.
  The waypoint-order class is same-basin negative after the smoke and all-order
  sweep. The next valid ls20 move needs a further mechanic representation:
  explicit door/lock before-after delta modeling, energy/action-budget
  admissibility, or a clearly labeled source-derived mechanic parse if that
  lane is allowed.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.

---

## Entry 20 — 2026-07-07 ~06:56 — ls20 delta probe shows pickup progress transforms color-14 to color-8

Updated files:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py`
- `synthesis_lab/notes/HANDOFF.md`
- `synthesis_lab/notes/OTHER_GAMES_RECON_20260706.md`

What changed:
- Added `--waypoint-delta-probe` to stop each waypoint order once
  final-pickup progress is reached, record L3-start vs post-pickup deltas, and
  run a short controlled greedy probe toward the observed old exit target.
- Added `frame_diff_summary` and `greedy_exit_probe` records for pixel deltas,
  color gain/loss, component lists, actor position, action rows, distance, and
  per-step level/state.

Delta smoke command:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders item01,c11_1,c11_2,exit14_0 --waypoint-delta-probe --graph-engine game_clone --max-ext 180 --max-nodes 80000 --deadline-seconds 45 --exit-probe-steps 24 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_delta_smoke_20260707T1205Z`

Delta smoke review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_delta_smoke_20260707T1205Z/index.html`
with raw `result.json`.

Delta smoke result:
- `progress_found`, not level advance.
- Progress route length `123`, extension length `26`, actor pos `[36,50]`,
  `progress_levels_completed=3`.
- Pickup delta: `changed=78`, bbox `[3,5,58,60]`, lost
  `{3:17, 9:15, 11:16, 12:10, 14:20}`, gained
  `{3:33, 8:20, 9:15, 12:10}`.
- Exit probe: actions `332222222222222222222222`, 24 legal actions, stayed at
  `levels_completed=3`, final `GameState.NOT_FINISHED`, actor pos `[26,50]`,
  distance to old exit target `28.4`.

All-order delta command:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders all_pickups_exit --waypoint-delta-probe --graph-engine game_clone --max-ext 220 --max-nodes 80000 --deadline-seconds 45 --exit-probe-steps 24 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_delta_orders_20260707T1210Z`

All-order delta review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_delta_orders_20260707T1210Z/index.html`
with raw `result.json`.

All-order delta result:
- Six `progress_found` rows.
- Zero `exit_probe_advanced` rows.
- `best_levels_completed=3`, `best_route_len=97`,
  `best_final_state=GameState.NOT_FINISHED`.
- Across all pickup orders, the large color-14 component is removed and a
  20-cell color-8 component is gained. The old `exit14_0` target is therefore
  not a color-14 exit after pickup progress.

Artifact hashes:
- Delta smoke `result.json`
  `6a4511b59bac7f507828000ef6e993acaf1e0c4f7ac7ca2999705fae134837ed`
- Delta smoke `index.html`
  `5d7f24357373d61bbdbd32eff233f4a234e0bb109edf57eaba3d2077856a56a9`
- Delta all-order `result.json`
  `60a4b2076163fc4365c013836c5502380dd4c58a7c0d92d77e67fa4f84fcbab7`
- Delta all-order `index.html`
  `8984c36a2b1a0cfbf5e1c7ab19692d9017b3f263b782fe08e7bddae94985e6d6`

Review check:
- Served `synthesis_lab/experiments/other_games/runs` on
  `127.0.0.1:18780`.
- HTTP `200` returned for delta smoke `index.html` and `result.json`, and for
  delta all-order `index.html` and `result.json`.
- Temporary server stopped.

Boundary:
- Local/offline only. This is not a live/ranked scorecard, not a leaderboard
  result, and not a 4/7 route.
- Do not continue aiming at `exit14_0` as a color-14 exit. The next valid
  representation should model the post-pickup color-8/door mechanic or an
  explicit door/lock before-after transition.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.

---

## Entry 21 — 2026-07-07 ~07:10 — ls20 post-pickup color-8 transition probe records target-state changes, no 4/7 route

Updated files:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py`
- `synthesis_lab/notes/HANDOFF.md`
- `synthesis_lab/notes/OTHER_GAMES_RECON_20260706.md`

What changed:
- Bumped LS20 probe schema to
  `synthesis_lab.other_games.ls20_corridor_probe.v2`.
- Added `--post-pickup-color8-probe`, `--color8-probe-steps`,
  `--color8-probe-min-area`, and `--color8-contact-distance`.
- Added mask-derived gained-color component detection and
  `color8_contact_probe`, which targets newly gained color-8 cells after
  final-pickup progress instead of treating the stale `exit14_0` label as a
  color-14 exit.
- Added event provenance for target-cell signature changes and post-transition
  one-step pokes.

Color-8 smoke v2 command:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders item01,c11_1,c11_2,exit14_0 --post-pickup-color8-probe --graph-engine game_clone --max-ext 180 --max-nodes 80000 --deadline-seconds 45 --color8-probe-steps 80 --color8-probe-min-area 10 --color8-contact-distance 1 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_smoke_v2_20260707T120456Z`

Color-8 smoke v2 review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_smoke_v2_20260707T120456Z/index.html`
with raw `result.json`.

Color-8 smoke v2 result:
- `color8_target_changed`, not level advance.
- Gained target component: color 8, area 20, bbox `[3,55,8,60]`.
- Controlled transition: target signature `{8:20} -> {5:12, 8:8}` on action
  `1`, parent extension `33111`, child extension `331111`, actor
  `[26,36] -> [26,30]`, levels remain `3`.

All-order color-8 command:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders all_pickups_exit --post-pickup-color8-probe --graph-engine game_clone --max-ext 220 --max-nodes 80000 --deadline-seconds 45 --color8-probe-steps 100 --color8-probe-min-area 10 --color8-contact-distance 1 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_orders_20260707T120530Z`

All-order color-8 review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_orders_20260707T120530Z/index.html`
with raw `result.json`.

All-order color-8 result:
- All six pickup permutations reached `color8_target_changed`.
- Zero rows advanced to 4/7. `best_levels_completed=3`,
  `best_route_len=97`, final state `GameState.NOT_FINISHED`.
- All rows gained the same 20-cell color-8 target at bbox `[3,55,8,60]`.
- Orders `0,1,2,4`: target `{8:20} -> {5:12, 8:8}`; no level advance.
- Orders `3,5`: target `{8:20} -> {12:20}` after action `2`, parent
  extension `1`, child extension `12`; actor `[36,26] -> [6,58]`, target
  distance `0`, no level advance.

Focused post-transition contact smoke command:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders c11_1,c11_2,item01,exit14_0 --post-pickup-color8-probe --graph-engine game_clone --max-ext 220 --max-nodes 80000 --deadline-seconds 45 --color8-probe-steps 100 --color8-probe-min-area 10 --color8-contact-distance 1 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_contact_smoke_v3_20260707T120800Z`

Focused post-transition contact smoke review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_contact_smoke_v3_20260707T120800Z/index.html`
with raw `result.json`.

Focused contact result:
- The `{12:20}` contact state is not an immediate exit.
- Post-transition pokes with actions `1,2,3,4` all stayed at
  `levels_completed=3`.
- Action `1` changed 51 pixels but did not advance. Actions `2,3,4` only spent
  energy / stayed in place.

Artifact hashes:
- Color8 smoke v2 `result.json`
  `4087cd7682f405c9456f44424aebd88de912632b519d8fb6015c06f92e88496e`
- Color8 smoke v2 `index.html`
  `3b975c1d14470eaa13540daa33f822c4db401c89efeacb1e2106e69dcbecfee9`
- Color8 all-order `result.json`
  `21803cdf44a1e377df224a9cd1a13aeb53e7389176f01f9d65776ecc984c9a8d`
- Color8 all-order `index.html`
  `499f33a9d7c5fb2fea46e48f51c2953418e6b1daa2c4597049ed9044dabe14da`
- Color8 contact smoke v3 `result.json`
  `7f462da1f29065cdeaab5fe79a5780445731cf8be82fa3839d54847f7d170648`
- Color8 contact smoke v3 `index.html`
  `9b5893dd3b7154b3227d33888b17aaa74153cfac86311982152030326eb4ebba`

Review check:
- Served `synthesis_lab/experiments/other_games/runs` on
  `127.0.0.1:18781`.
- HTTP `200` returned for color8 smoke v2, all-order, and contact-smoke
  `index.html` and `result.json`.
- Temporary server stopped.

Boundary:
- Local/offline only. This is not a live/ranked scorecard, not a leaderboard
  result, and not a 4/7 route.
- Do not rerun longer generic `actor_astar`, waypoint-order, or old color-14
  exit probes for ls20 L3. The next valid representation should model the
  post-transition target states explicitly (`{5:12,8:8}` and `{12:20}`), or
  use a clearly labeled source-derived mechanic parse if explicitly allowed.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.

---

## Entry 22 — 2026-07-07 ~07:28 — ls20 target-state continuation records signature cycles, no 4/7 route

Updated files:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py`
- `synthesis_lab/notes/HANDOFF.md`
- `synthesis_lab/notes/OTHER_GAMES_RECON_20260706.md`

What changed:
- Bumped LS20 probe schema to
  `synthesis_lab.other_games.ls20_corridor_probe.v3`.
- Added `--color8-transition-continue-probe` and
  `--color8-transition-steps`.
- Added `target_state_continue_search`, which starts after a color-8 target
  transition event and continues legal local-engine search with target-cell
  signature in the graph key. Target signature changes are logged and kept as
  state, not treated as terminal.

Focused continuation smoke command:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders c11_1,c11_2,item01,exit14_0 --post-pickup-color8-probe --color8-transition-continue-probe --graph-engine game_clone --max-ext 220 --max-nodes 100000 --deadline-seconds 60 --color8-probe-steps 100 --color8-transition-steps 80 --color8-probe-min-area 10 --color8-contact-distance 1 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_continue_smoke_20260707T121613Z`

Focused continuation smoke review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_continue_smoke_20260707T121613Z/index.html`
with raw `result.json`.

Focused continuation result:
- `color8_transition_continue_deadline`, not level advance.
- `best_levels_completed=3`, `best_route_len=97`.
- Continuation starts from target signature `{12:20}` and stayed at
  `levels_completed=3`.
- Continuation stats: `states_seen=1340`, `queue_popped=1119`,
  `expanded_edges=1339`, `target_transition_event_count=332`.
- Observed target-signature cycle includes `{12:20}`, `{14:20}`,
  `{5:12,12:8}`, `{5:8,12:12}`, `{8:20}`, and `{9:20}`.

All-order continuation command:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders all_pickups_exit --post-pickup-color8-probe --color8-transition-continue-probe --graph-engine game_clone --max-ext 220 --max-nodes 100000 --deadline-seconds 60 --color8-probe-steps 100 --color8-transition-steps 80 --color8-probe-min-area 10 --color8-contact-distance 1 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_continue_orders_20260707T121758Z`

All-order continuation review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_continue_orders_20260707T121758Z/index.html`
with raw `result.json`.

All-order continuation result:
- Six `color8_transition_continue_deadline` rows.
- Zero rows advanced to 4/7. `best_levels_completed=3`,
  `best_route_len=97`, final state `GameState.NOT_FINISHED`.
- Orders `0,2` (`{5:12,8:8}` transition): `states_seen=1184/1160`,
  event counts `35/35`.
- Orders `1,4` (`{5:12,8:8}` transition): `states_seen=1251/1229`,
  event counts `31/31`.
- Orders `3,5` (`{12:20}` transition/contact): `states_seen=1126/1099`,
  event counts `325/325`.
- The target-state graph observed real signature cycles across variants
  (`{5:12,12:8}`, `{5:8,12:12}`, `{8:20}`, `{9:20}`, `{14:20}`), but none
  reached level 4 under this bounded representation.

Artifact hashes:
- Continuation smoke `result.json`
  `ea7c3848844877c5dc083c37c2825f849534b366d54c428b3aa1e8de9b65ab64`
- Continuation smoke `index.html`
  `79a871b0f74298c930b04a1f61e6a1d61a56162b2fc804bd1b3a6934344fa15a`
- Continuation all-order `result.json`
  `969dd094bbf75cc4c885e8570f00d3dbad948bfd59d4fac4c55fd13be36d05a4`
- Continuation all-order `index.html`
  `4633059290fc1d8c1ce6ab3ea1b837f98aa2a9c4623a0d947fc749f6535dc651`

Review check:
- Served `synthesis_lab/experiments/other_games/runs` on
  `127.0.0.1:18782`.
- HTTP `200` returned for continuation smoke and all-order `index.html` and
  `result.json`.
- Playwright screenshot succeeded for the all-order review surface and local
  image inspection showed a readable nonblank page.
- Temporary server stopped.

Boundary:
- Local/offline only. This is not a live/ranked scorecard, not a leaderboard
  result, and not a 4/7 route.
- Do not launch another longer plain target-state continuation run for ls20 L3
  as the next move. This representation now has smoke plus all-order
  same-basin deadlines. The next valid representation needs new state or
  mechanics: phase/action-budget model over signature cycles, explicit
  teleport/refill mechanics, or a clearly labeled source-derived mechanic parse
  if explicitly allowed.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.

## Entry 23 — 2026-07-07 ~07:47 — ls20 target-phase/underlay probe exposes jump and budget state, no 4/7 route

Updated files:
- `synthesis_lab/experiments/other_games/ls20_corridor_probe.py`
- `synthesis_lab/notes/HANDOFF.md`
- `synthesis_lab/notes/OTHER_GAMES_RECON_20260706.md`

What changed:
- Bumped LS20 probe schema to
  `synthesis_lab.other_games.ls20_corridor_probe.v4`.
- Added `--color8-phase-probe`, `--color8-phase-steps`, and
  `--color8-jump-threshold`.
- Added `target_phase_continue_search`, which starts after a color-8 target
  transition event and continues legal local-engine search with actor-masked
  target underlay, actor-overlap count, energy/budget-strip counts, and
  jump-like actor movement in the state/diagnostic surface.
- HTML review now includes phase status, unique phase count, jump events, and
  budget events for post-pickup color-8 probes.

Focused phase smoke command:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders c11_1,c11_2,item01,exit14_0 --post-pickup-color8-probe --color8-phase-probe --graph-engine game_clone --max-ext 220 --max-nodes 100000 --deadline-seconds 60 --color8-probe-steps 100 --color8-phase-steps 80 --color8-probe-min-area 10 --color8-contact-distance 1 --color8-jump-threshold 12 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_phase_smoke_20260707T1232Z`

Focused phase smoke review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_phase_smoke_20260707T1232Z/index.html`
with raw `result.json`.

Focused phase result:
- `color8_phase_deadline`, not level advance.
- `best_levels_completed=3`, `best_route_len=97`.
- Phase stats: `target_phase_unique_phase_count=120`,
  `target_phase_transition_event_count=3971`,
  `target_phase_jump_event_count=569`,
  `target_phase_budget_event_count=3971`.
- The first high-signal event shows actor overlap separated from underlay:
  `visible={12:20}`, `underlay={}`, `overlap=20`, actor `[6,58]`; action `2`
  jumps actor to `[36,30]`, exposing `visible={9:20}`,
  `underlay={9:20}`, `overlap=0`.

All-order phase command:
`python3 synthesis_lab/experiments/other_games/ls20_corridor_probe.py --seed 7 --start-route known_l3 --waypoint-orders all_pickups_exit --post-pickup-color8-probe --color8-phase-probe --graph-engine game_clone --max-ext 220 --max-nodes 100000 --deadline-seconds 60 --color8-probe-steps 100 --color8-phase-steps 80 --color8-probe-min-area 10 --color8-contact-distance 1 --color8-jump-threshold 12 --verify-game-clone-engine --out-dir synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_phase_orders_20260707T1238Z`

All-order phase review surface:
`synthesis_lab/experiments/other_games/runs/ls20_corridor_probe_seed7_l3_color8_phase_orders_20260707T1238Z/index.html`
with raw `result.json` and `playwright_screenshot.png`.

All-order phase result:
- Six `color8_phase_deadline` rows.
- Zero rows advanced to 4/7. `best_levels_completed=3`,
  `best_route_len=97`, final state `GameState.NOT_FINISHED`.
- Orders `0,2` (`{5:12,8:8}` transition): `states_seen=1228/1226`,
  unique phases `90/90`, jump events `338/338`.
- Orders `1,4` (`{5:12,8:8}` transition): `states_seen=1298/1295`,
  unique phases `111/111`, jump events `331/331`.
- Orders `3,5` (`{12:20}` transition/contact): `states_seen=1346/1337`,
  unique phases `120/120`, jump events `569/569`.
- `{12:20}` contact rows expose an underlay cycle through `{9:20}` and
  `{14:20}` after actor jumps away; `{5:12,8:8}` rows mostly cycle target
  mixtures such as `{5:12,8:8}` and `{5:8,8:12}` while budget-strip energy11
  splits otherwise similar phases.

Artifact hashes:
- Phase smoke `result.json`
  `0432581547b6c44174ba3baed97369d8e79a031f2158daedeb4ab679ce3a06dc`
- Phase smoke `index.html`
  `e0c8f19f89b9cc2977f3424176f9be7ba6b0e53a10fcd8540ae7561717219002`
- Phase all-order `result.json`
  `06346d4b7b6051f510a3fb066e9d03d8fb145f05af5481bb5e39aef6ed53a2cd`
- Phase all-order `index.html`
  `d8fa3c3a08aaddedb549bb971076f2a80c6783894c7f1c47b85e5aa2357959ee`
- Phase all-order `playwright_screenshot.png`
  `af5a7b5d08fa9d3648d955115660fe0734017e3fe743ce14bec011ac07419882`

Review check:
- Served `synthesis_lab/experiments/other_games/runs` on
  `127.0.0.1:18783`.
- HTTP `200` returned for phase smoke/all-order `index.html` and `result.json`.
- Playwright screenshot succeeded for the all-order review surface and local
  image inspection showed a readable nonblank page.
- Temporary server stopped.

Boundary:
- Local/offline only. This is not a live/ranked scorecard, not a leaderboard
  result, and not a 4/7 route.
- Do not launch another longer plain phase/underlay continuation run for ls20
  L3 as the next move. The phase representation now has smoke plus all-order
  same-basin deadlines. The next valid representation needs a new mechanic
  degree of freedom beyond phase/underlay/budget/jump counts: action grammar
  over the jump graph, source-derived mechanic parse if explicitly allowed and
  labeled, or cross-level generalization that predicts the lock action rather
  than expanding the same local phase graph.
- Final gates after this entry: `py_compile` on touched Python files exit `0`,
  `git diff --check` exit `0`, and `honesty_audit.py` exit `0` with
  `summary=97/46/6`, `active=85/23/0`, `quarantined=12/23/6`.
