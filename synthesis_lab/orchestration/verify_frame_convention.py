"""Task 1: verify the frame-convention hypothesis offline.

Part A: raw FrameData.frame lengths and first-vs-last frame divergence
        around a pour on the pristine L0 scene.
Part B: drive the UNMODIFIED arc_suite_runner.run_game code path (the
        exact path the live scout used) offline with the same v9retry
        checkpoint, and check whether the arrange roster/plan reproduces
        the live failure signature (glove-geometry roster entry, flat
        plan scores).
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/frosty40/megarexus")
os.environ["MEGAREXUS_CKPT"] = str(
    REPO / "synthesis_lab/experiments/megarexus_cnn/runs/outcome_v9_retry_20260702/last.pt"
)
sys.path.insert(0, str(REPO / "codexlab"))

from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402

ENV_DIR = REPO / "environment_files"
GAME = "sp80-589a99af"


def comps_summary(grid):
    sys.path.insert(0, str(REPO / "synthesis_lab"))
    from engine_adapters.portband_glove_adapter import frame_components
    return [(c["color"], c["bbox_xyxy"], c["area"]) for c in frame_components(np.asarray(grid, dtype=np.int8))[:8]]


print("=== Part A: raw frame conventions ===")
arc = Arcade(arc_api_key="anonymous", environments_dir=str(ENV_DIR), operation_mode=OperationMode.OFFLINE)
card = arc.open_scorecard()
env = arc.make(GAME, scorecard_id=card)
fd = env.reset()
frames = fd.frame if isinstance(fd.frame, list) else [fd.frame]
print(f"reset: n_frames={len(frames)} levels={fd.levels_completed} state={fd.state}")

fd = env.step(GameAction.ACTION5)  # pour on pristine L0, no arrangement
frames = fd.frame if isinstance(fd.frame, list) else [fd.frame]
f0 = np.asarray(frames[0], dtype=np.int8)
fl = np.asarray(frames[-1], dtype=np.int8)
print(f"ACTION5: n_frames={len(frames)} levels={fd.levels_completed} state={fd.state} "
      f"first_vs_last_diff_cells={int((f0 != fl).sum())}")

fd = env.step(GameAction.ACTION3)  # single move
frames = fd.frame if isinstance(fd.frame, list) else [fd.frame]
print(f"ACTION3: n_frames={len(frames)} levels={fd.levels_completed} state={fd.state}")

print()
print("=== Part B: unmodified run_game path, offline, v9retry last ===")
from arc_suite_runner import run_game  # noqa: E402

res = run_game(
    arc=arc,
    scorecard_id=card,
    game_id=GAME,
    policy_name="megarexus_cnn",
    max_actions=60,
    solution_root=None,
    auto_reset=True,
    include_perception=False,
)
print(f"status={res['status']} levels={res['levels_completed']} resets={res['resets']} "
      f"final_state={res['final_state']} policy_errors={res['policy_error_count']}")
trace = res.get("policy_trace") or {}
print("plan_summaries:")
print(json.dumps(trace.get("plan_summaries"), indent=1))
print("ledger:")
for row in (trace.get("ledger") or [])[:22]:
    print(f"  step={row['step']} lv={row['levels_completed']} {row['source']} "
          f"a={row['action']} x={row['x']} y={row['y']} info={row['info']}")

try:
    arc.close_scorecard(card)
except Exception:
    pass
print("done")
