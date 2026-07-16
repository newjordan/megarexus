# Primitive Game Designs Sweep — 2026-07-15

**What they are:** basic **primitive game designs**, not Atari ports or a
training corpus of old games.

**Where:** `/home/frosty40/rexus/.external/pufferlib-4.0-newjordan/ocean/`  
(~62 minimal envs in C + bindings)

**Doctrine:** plant every design primitive into the base model; training rises
what works.

---

## Framing

Each Ocean env is a **compressed design atom** — the smallest playable
expression of a mechanic class:

| Atom | Meaning |
|---|---|
| paddle-ball | continuous deflection + score lines |
| projectile | aim / velocity / wrap / lives |
| lane timing | cross traffic without collision |
| maze path | discrete navigation + goal |
| push-block | irreversible spatial planning |
| growth trail | body extends; self-collision |
| chase-avoid | multi-entity grid pursuit |
| balance dynamics | unstable continuous control |
| flip puzzle | local toggle → global state |
| gravity-align | stack-to-N win |
| falling pack | tetromino packing under gravity |
| slide-merge | 2048-style combine |
| perfect-info board | chess/go-class zero-sum |
| multiagent forage | share / convert / compete on stars |
| coordination cook | Overcooked-style joint plan |
| logistics | warehouse / pickup routes |
| survival craft | open world craft+survive |
| reaction | hit targets under time |
| paintflow | domain SP80 liquid pour (Rexus) |

These are the **alphabet** for more complex ARC/SP80-style worlds — not
labels, not win-banks, not scorecard targets by themselves.

---

## Catalog on the pad (P1–P32)

| ID | Primitive | Realizers (ocean/) |
|---|---|---|
| P1 | paddle-ball deflect | pong, breakout, slimevolley |
| P2 | projectile shooter | asteroids, blastar |
| P3 | lane timing dodge | freeway |
| P4 | racing pass/dodge | enduro, whisker_racer, drive |
| P5 | maze pathfind | maze, tmaze, tower_climb |
| P6 | push-block sokoban | boxoban |
| P7 | growth trail snake | snake |
| P8 | chase-avoid grid | pacman |
| P9 | balance dynamics | cartpole, double_pendulum |
| P10 | continuous vehicle | drone, docking |
| P11 | flip toggle puzzle | lightsout |
| P12 | gravity-align connect | connect4 |
| P13 | falling pack | tetris, drmario |
| P14 | slide merge | g2048 |
| P15 | perfect-info board | chess, checkers, go, hex, tripletriad |
| P16 | memory match | memory |
| P17 | multiagent forage | battle, convert*, scape, target, onlyfish, minimal, squared* |
| P18 | coordination cook | overcooked |
| P19 | warehouse logistics | rware, trash_pickup |
| P20 | survival craft | craftax*, nmmo3, nethack |
| P21 | platform jump | dino |
| P22 | flock boids | boids |
| P23 | laser-mirror puzzle | laser_puzzle |
| P24 | combat arena | impulse_wars, moba, tactical, battle |
| P25 | abstract MDP chain | chain_mdp, onestateworld |
| P26 | reaction whack | whackamole |
| P27 | terrain terraform | terraform |
| P28 | shared pool commons | shared_pool |
| P29 | domain paintflow | rexus_sp80_paintflow* |
| P30 | scaffold template | template, benchmark, matsci |
| P31 | ocean library present | meta |
| P32 | primitive coverage | meta |

**Code:** `operators/primitive_game_designs.py`
