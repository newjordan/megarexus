#!/usr/bin/env python3
"""Local / OFFLINE runner for ARC-AGI-3 agents (StochasticGoose, etc.).

Runs an agent against LOCAL game environments through the arc_agi OFFLINE engine:
NO network, NO API key, no calls to three.arcprize.org. Use this for all iteration
until we have a good score (then switch to live).

StochasticGoose sets MAX_ACTIONS=inf and only stops on WIN/8h, so an explicit
--max-actions cap is required for a bounded evaluation. We construct the agent
directly (not via Swarm) so we can set that cap.

Examples:
  python run_local.py --game ls20 --max-actions 500
  python run_local.py --game ls20,vc33 --env-dir synthesis_lab/experiments/other_games/environment_files
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

FRAMEWORK = os.environ.get("ARC_FRAMEWORK", "").strip()
GOOSE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(GOOSE, ".."))
_DEFAULT_ENV = os.path.join(
    _REPO, "synthesis_lab", "experiments", "other_games", "environment_files"
)
for _p in (FRAMEWORK, GOOSE, os.path.join(GOOSE, "custom_agents")):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

# Import the StochasticGoose agent directly. Avoids the framework's
# template-heavy agents/__init__.py (langgraph/smolagents/llm deps we don't need).
from custom_agents.action import Action  # noqa: E402  (pulls torch + agents.agent base)
from arc_agi import Arcade, OperationMode  # noqa: E402

AGENTS = {"action": Action}

logger = logging.getLogger("run_local")


def run_one(arc: Arcade, agent_name: str, game: str, card_id: str,
            max_actions: int, record: bool):
    env = arc.make(game, scorecard_id=card_id)
    if env is None:
        print(f"[{game}] make() returned None — not available offline in this env-dir; skipping")
        return
    agent_cls = AGENTS[agent_name]
    agent = agent_cls(
        card_id=card_id,
        game_id=game,
        agent_name=agent_name,
        ROOT_URL="offline://local",
        record=record,
        arc_env=env,
        tags=["local"],
    )
    if not hasattr(agent, "action_counter"):
        agent.action_counter = 0
    agent.MAX_ACTIONS = max_actions  # cap the otherwise-infinite loop
    try:
        agent.main()
    except Exception as e:  # noqa: BLE001 — report per-game, keep going
        logger.exception(f"[{game}] agent crashed: {e}")
    print(f"[{game}] finished: actions={getattr(agent,'action_counter',None)} "
          f"levels_completed={getattr(agent,'levels_completed',None)} state={getattr(agent,'state',None)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Local/OFFLINE ARC-AGI-3 agent runner (no network)")
    ap.add_argument("-a", "--agent", default="action")
    ap.add_argument("-g", "--game", default="ls20", help="comma-separated game ids")
    ap.add_argument("--max-actions", type=int, default=300)
    ap.add_argument("--env-dir", default=_DEFAULT_ENV,
                    help="local metadata.json corpus (default: in-repo other_games suite)")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Force offline everywhere (belt-and-suspenders vs any env-var leakage).
    os.environ["OPERATION_MODE"] = "offline"
    arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=args.env_dir)
    avail = sorted({e.game_id.split("-", 1)[0] for e in arc.available_environments})
    print(f"OFFLINE runner | env-dir={args.env_dir} | available_games={avail}")

    if args.agent not in AGENTS:
        print(f"agent '{args.agent}' not registered. Have: {sorted(AGENTS)}")
        sys.exit(2)

    card_id = arc.open_scorecard(tags=["local", args.agent])
    games = [g.strip() for g in args.game.split(",") if g.strip()]
    for g in games:
        print(f"\n===== {args.agent} on {g}  (<= {args.max_actions} actions, OFFLINE) =====")
        run_one(arc, args.agent, g, card_id, args.max_actions, args.record)

    sc = arc.close_scorecard(card_id)
    print("\n===== LOCAL SCORECARD =====")
    if sc is None:
        print("(no scorecard returned)")
        return
    d = sc.model_dump()
    print(f"score={d.get('score')} | levels {d.get('total_levels_completed')}/{d.get('total_levels')} "
          f"| envs_completed {d.get('total_environments_completed')}/{d.get('total_environments')} "
          f"| total_actions={d.get('total_actions')}")
    for e in d.get("environments", []) or []:
        if isinstance(e, dict):
            print(f"  {e.get('game_id')}: levels_completed={e.get('levels_completed')} "
                  f"score={e.get('score')} actions={e.get('total_actions', e.get('actions'))}")
        else:
            print(f"  {e}")


if __name__ == "__main__":
    main()
