# stochastic-goose — provenance & admission

Brought in on request ("use the stochastic goose ... no cheat or lookahead").
"Stochastic goose" = **StochasticGoose**, ARC-AGI-3 agent by Tufa Labs (hence "tufa").

## Source
- Repo: https://github.com/DriesSmit/ARC3-solution  (git a6e77bb, 2026-01-27)
- Authors: Dries Smit (Tufa Labs), adviser Jack Cole (MindsAI).
- Brought in: custom_agents/{action.py,view_utils.py,__init__.py}, custom_agent.py,
  utils.py, README.md, requirements.txt, Makefile.
- NOT brought in: the vendored ARC-AGI-3-Agents git submodule (we use the current
  official framework already cloned at ../ARC-AGI-3-Agents instead).

## What it is (integrity screen: PASSED — no lookahead, no cheat)
- A CNN that predicts which action will change the CURRENT frame; trained online with
  reward = 1.0 if the frame changed else 0.0 (self-supervised explorer).
- Picks ONE action per step for the observed frame and executes it in the real env.
- NO forward rollouts / simulator lookahead, NO answer/label/win-bank access.
  ("scoring" = CNN action-change prob; "loop" = online train/act loop.)

## Port applied (Jan-2026 Preview API -> current May-2026 framework)
- custom_agents/action.py:
  - `from agents.structs import ...`  ->  `from arcengine import FrameData, GameAction, GameState`
  - `latest_frame.score`  ->  `latest_frame.levels_completed`  (FrameData field renamed in v0.9.3)
- `available_actions` is now NATIVE on FrameData, so the README's manual structs.py
  patch is NOT needed.

## Still TODO
- Wire registration: import Action into ../ARC-AGI-3-Agents/agents/__init__.py so
  --agent=action registers (framework auto-registers Agent subclasses by lowercased name).
- Make utils.get_git_info tolerate running outside a git repo (git rev-parse can crash startup).
- Install torch/tensorboard/pillow into the framework venv (in progress).
- Smoke test: uv run main.py --agent=action --game=<id> for a few steps on the RTX 3090.
