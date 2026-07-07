# OTHER GAMES RECON — 2026-07-06

Lane: parallel recon of non-sp80 ARC-AGI-3 games. Rules of this lane: observation and
legal env interaction only; no human answers/solutions; no route replay; no model
training; no ranked/live scorecard runs; no git commits; new files only.

---

## Entry 1 — 2026-07-06 — DISCOVERY: offline game acquisition CONFIRMED

**How the stack works (read from installed SDK, not guessed):**
- The SDK is `arc_agi` (0.1.0-era, at `/home/frosty40/miniconda3/lib/python3.13/site-packages/arc_agi/`)
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
