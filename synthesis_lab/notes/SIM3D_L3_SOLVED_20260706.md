# L3 SOLVED, env-confirmed — HEADLINE 4 LEVELS — 2026-07-06

**L0 (CNN-alone), L1 (sim-guided SOLVED@1), L2 (droplet-CA-guided), L3 (droplet-CA-guided) —
all in-budget, all env-confirmed.** L3 real_win = [True, True, True] (fresh env builds), all 4
buckets filled, zero floor-touch.

```
Gate: env-confirmed L3 win within budget -> headline 4 levels.  Result: PASS.
No imitation: win FOUND by search guided by a hand-coded exact sim; reachable via legal moves.
```

## What L3 needed (two fixes, both from the sp80 mechanic, not guessing)
The droplet CA (L2's substrate) was right; the EXTRACTOR was wrong for L3:
1. **Native-sprite extraction** (`native_scene.py`). The 16x16/4px frame extractor mis-sized L3
   (native cell ≈ 3px, frame = native×3+2), turning 1-cell U-buckets into "wide connected
   mouths" and finding 2 targets instead of 4. Building the grid directly from sprite pixels in
   the game's own coords fixes it — the same grid the droplet logic (L749-820) runs on.
2. **Source-emission fix.** L3's "wide source" is a 7-wide bar whose pixels are mostly block
   (color 8) with ONE color-4 emitter embedded (`[8 8 8 4 8 8 8]`). Only color-4 pixels emit;
   the block-colored cells are solid. Emitting from every source cell caused a systematic +1
   phantom bucket-fill.

**Fidelity after both: EXACT.** L3 native-CA vs env — filled-set 100/100, flood 120/120,
win 120/120. (L2 frame droplet_pour still 60/60.) The confirmed sp80 mechanic (user): a bucket
is painted the tick a droplet enters it; touching the floor fails; LEFTOVER paint is
irrelevant — win = every bucket painted AND floor never touched.

## The solve
`l3_solve.py` hill-climbs move-genomes (real env moves, reachable/legit), scores by the exact
native CA on the flood-severity gradient (n_filled↑, n_wall↓). Found a win on the first restart:
14-move genome → native filled=[0,1,2,3], n_wall=0, win=True → **env real_win=True ×3**.
Artifacts: `fixtures/l3_solve.json`, `fixtures/L3_solve_h264.mp4` (env pour),
`fixtures/L3_sim_h264.mp4` (droplet-CA animation).

## Known TODO (not blocking L3)
`native_scene` and the frame extractor disagree on the L2-WIN arrangement (native spuriously
reports 1 wall-hit; frame + env say clean win). Frame is correct for L0-L2; native is exact for
L3. Reconcile native for L2 (likely a width/disposal-edge detail at 32 vs 16 wide) to have one
extractor for everything.

## Next
L4/L5 add the real redirect (`tuvkdkhdokr`, 90° deflect) — the droplet CA must gain redirect
handling (currently no-redirect only). The native extractor already reads redirect sprites into
a `redir` grid; wire the 90° deflection (game L787-803) and validate. Then search L4/L5.
