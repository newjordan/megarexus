"""Pin the overlap-click tie-break for EQUAL-width units (L1 s42: u0 and u1
are both 12px wide). Four constructed cases varying selection state and
left/right placement, reading which unit turns 9 after the overlap click."""
import sys
from pathlib import Path
import numpy as np

LAB = Path('/home/frosty40/megarexus/synthesis_lab')
for p in (LAB, LAB / 'orchestration', LAB / 'experiments' / 'megarexus_cnn',
          LAB / 'orchestration' / 'sim3d'):
    sys.path.insert(0, str(p))

from imagine_search_gate import _present, _grid, GAME_ID, ACTION_OF_ID
from arc_agi import Arcade
from arc_agi.base import OperationMode

arc = Arcade(arc_api_key="anonymous",
             environments_dir='/home/frosty40/megarexus/environment_files',
             operation_mode=OperationMode.OFFLINE)

def fresh():
    env = arc.make(GAME_ID, seed=42)
    _present(env, 1)
    return env

def step(env, aid, x=None, y=None):
    data = {"x": int(x), "y": int(y)} if aid == 6 else None
    return _grid(env.step(ACTION_OF_ID[aid], data=data))

def region(g, tag):
    # u0 home rows y16-19, u1 parked rows y16-19 after up2; print both bands
    print(tag)
    for yy in range(15, 21):
        print('  ', ''.join(f"{g[yy,xx]:X}" for xx in range(0, 44)))

# base L1 s42: u0 x8-19 y16-19 (idx0), u1 x28-39 y24-27 (idx1), u2 x20-39 y32-35
# A: u1 selected, moved to x16-27 y16-19 (overlap u0 x16-19); click overlap.
env = fresh()
step(env, 6, 33, 25)                       # select u1
for _ in range(2): step(env, 1)            # up x2 -> y16-19
for _ in range(3): step(env, 3)            # left x3 -> x16-27
g = step(env, 6, 17, 18)                   # click overlap (u0+u1)
region(g, "A: tie, u1 SELECTED+rightmost+later-scan; click (17,18):")

# B: same geometry, but neither overlap unit selected (u2 selected); click.
env = fresh()
step(env, 6, 33, 25)
for _ in range(2): step(env, 1)
for _ in range(3): step(env, 3)
step(env, 6, 33, 33)                       # select u2 (away)
g = step(env, 6, 17, 18)
region(g, "B: tie, neither selected; u0 leftmost+earlier, u1 rightmost+later; click (17,18):")

# C: mirror — u1 moved LEFT of u0 (x4-15), overlap x8-15; u2 selected; click.
env = fresh()
step(env, 6, 33, 25)
for _ in range(2): step(env, 1)
for _ in range(6): step(env, 3)            # left x6 -> x4-15
step(env, 6, 33, 33)                       # select u2
g = step(env, 6, 9, 18)
region(g, "C: tie, neither selected; u1 LEFTMOST+later, u0 rightmost+earlier; click (9,18):")

# D: A's geometry but u0 selected; click overlap.
env = fresh()
step(env, 6, 33, 25)
for _ in range(2): step(env, 1)
for _ in range(3): step(env, 3)
step(env, 6, 9, 18)                        # select u0 (u0-only cell)
g = step(env, 6, 17, 18)
region(g, "D: tie, u0 SELECTED+leftmost+earlier; click (17,18):")
