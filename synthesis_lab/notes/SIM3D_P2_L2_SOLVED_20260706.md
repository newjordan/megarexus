# REGIME L2-FRONTIER — L2 SOLVED, env-confirmed — 2026-07-06

**HEADLINE MOVES TO 3 LEVELS: L0 (CNN-alone), L1 (sim-guided SOLVED@1), L2 (droplet-CA-guided,
env-confirmed).** This SUPERSEDES `SIM3D_P2_L2_DECISION_20260706.md` — that decision-branch
NEGATIVE was WRONG. Correction credit: the user ("simulate the whole pour with a ball that
splits into two orthogonal directions and falls off the side of a wall … floor is bad").

```
honesty_audit: exit 0   t0-five untouched: YES   S0: PASS
Gate criterion (§2 P2): env-confirmed L2 win within a 4-pour budget -> headline 3 levels
Result: L2 real_win = [True, True, True] (3 fresh env builds), single pour. -> PASS
No imitation: solution FOUND by search guided by a hand-coded exact sim; no route/human lineage,
no answer injection; genome reached via legal ACTION6+move sequence (frame_of/real_win).
```

## What I got wrong, and why
My P2 negative rested on a capacity argument: L2 emits ~54 paint cells, mouths+bars hold ~24,
so ~30 cells "must overflow to the floor → flood unavoidable." **That is false.** The fail
sensor (`waoewejnqzc`) is ONLY the bottom floor; the LEFT/RIGHT board edges are open. Excess
paint is DISPOSED off the sides (a droplet whose next cell is empty/off-board just keeps going,
L756), so it never needs to be held. My hydraulic lane POOLS fluid and cannot represent
side-disposal, so it (and every search it guided) reported floods where a proper arrangement
sheds the excess off the edges. Empirical tell I initially under-weighted: 40/200 real
arrangements already retained <45 of the ~54 emitted cells — paint WAS leaving the board.

## The fix: a faithful droplet CA (`dropletsim.py`)
A direct port of the game's droplet stepping (sp80.py L749-820). On no-redirect levels (L2)
EVERY droplet moves straight DOWN; a block forks it to x±1 (L764); a droplet fills a target
when it ENTERS the enclosed mouth cell (L775); a droplet whose next cell is the bottom wall
FAILS (L815); a droplet that runs off an x-edge is DISPOSED. Volume is irrelevant to win/fail
(one droplet fills a mouth; one touch fails), so the outcome is an EXACT deterministic
reachability BFS — microseconds per pour.

**Fidelity: EXACT.** On 150 real L2 arrangements the droplet CA matches the env on filled-set
(150/150), flood (150/150), and win (150/150). Defaults match on L0/L1/L2. So a droplet-win IS
an env-win, and a droplet-negative is definitive (not sim-limited).

## The solve
- `p2_grid_search.py` (direct bar-placement search on the exact CA, no env in the loop) found a
  clean win config: all 3 mouths filled, **n_wall = 0** (every excess branch shed off the sides).
- `p2_droplet_solve.py` then found a REACHABLE win via legal move-genomes (hill-climb on the
  flood-severity gradient n_filled↑/n_wall↓), scored by the exact CA: 12-move genome
  `[(1,3,6),(5,1,8),(5,3,2),(0,3,2),(6,2,4),(6,2,2),(1,2,4),(5,2,3),(1,3,3),(5,2,5),(0,4,8),(1,4,8)]`
  → droplet win (filled [0,1,2], n_wall 0) → **env real_win = True (×3 fresh builds)**.
- Artifacts: `fixtures/p2_grid_l2_result.json`, `fixtures/p2_droplet_solve.json`.

## Mechanic facts locked (from source + user)
- **Floor = the only fail sensor** (bottom `waoewejnqzc`); side edges are open → excess is shed
  off the sides. This is THE L2 mechanic.
- Walls are barriers that stop/redirect flow (route AROUND them; touching a wall fails) — a tool,
  per the user; relevant when field-walls appear (L4/L5). To model when we get there.

## Status / next
Headline **3 levels, in-budget, env-confirmed**. The droplet CA is the right substrate for the
remaining field (L3/L4/L5) — L4/L5 add the real redirect (`tuvkdkhdokr`, 90° deflect) which the
CA must gain; L3 adds connected mouths + sidewalls. The hydraulic lane remains the L0/L1 ranker
(P0/P1 gates stand); the droplet CA is exact for no-redirect levels.
