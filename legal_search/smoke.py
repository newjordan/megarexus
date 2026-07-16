#!/usr/bin/env python3
"""Smoke test: consume every registered system + run a fake legal episode."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# repo root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_search.consume.registry import consume_all, list_systems  # noqa: E402
from legal_search.contract import LEGAL_RULES, TRUTH_BOUNDARY  # noqa: E402
from legal_search.loop import LegalAgent  # noqa: E402
from legal_search.types import Action, Transition  # noqa: E402


def fake_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = np.full((64, 64), 12, dtype=np.int16)
    g[0, :] = 14
    g[10:14, 20:40] = 8
    g[40:48, 30:38] = 11
    g[50:55, 10:50] = 6
    noise = rng.integers(0, 3, size=(8, 8))
    g[2:10, 2:10] = noise
    return g


def main() -> int:
    frame = fake_frame(1)
    systems = list_systems()
    print(f"registered_systems={len(systems)}")
    for row in systems:
        print(f"  - {row['system_id']}: {row['algorithm_id']} ({row['role']})")

    # transitions for inducers
    frame2 = frame.copy()
    frame2[10:14, 21:41] = 8  # shifted bar
    frame2[10:14, 20] = 12
    tr = Transition(
        game_id="smoke-game",
        step=0,
        frame_before=frame,
        action=Action(4, source="smoke"),
        frame_after=frame2,
        levels_before=0,
        levels_after=0,
        frame_changed=True,
        levels_up=False,
    )
    bundle = consume_all(
        frame2,
        game_id="smoke-game",
        transitions=[tr],
        available_actions=[1, 2, 3, 4, 5, 6],
        levels_completed=0,
    )
    # validate receipts legal
    bad = []
    for sid, pkt in bundle["systems"].items():
        rec = pkt.get("receipt") or {}
        if rec.get("legal") is False:
            bad.append(sid)
        for hyp in pkt.get("hypotheses") or []:
            if hyp.get("admitted") is True:
                bad.append(f"{sid}:admitted")
    print(f"consume_all systems={bundle['n_systems']} illegal={bad}")

    agent = LegalAgent(game_id="smoke-game")
    agent.reset_for_game("smoke-game")
    a1 = agent.decide(frame, available_actions=[1, 2, 3, 4, 5, 6])
    agent.observe_transition(tr)
    a2 = agent.decide(frame2, available_actions=[1, 2, 3, 4, 5, 6])
    print(f"legal_agent actions: {a1.to_dict()} -> {a2.to_dict()}")
    print(f"truth_boundary={TRUTH_BOUNDARY}")
    print(f"rules={[r.rule_id for r in LEGAL_RULES]}")

    # Private-game wipe: new game_id must clear episodic memory (R7)
    n_before = agent.memory.snapshot()["n_transitions"]
    agent.reset_for_game("smoke-private-never-seen")
    wiped = agent.memory.snapshot()
    a3 = agent.decide(frame, available_actions=[1, 2, 3, 4, 5, 6])
    private_ok = (
        n_before >= 1
        and wiped["n_transitions"] == 0
        and wiped["cross_game_memory_admitted"] is False
        and a3.action_id in range(1, 7)
    )
    print(f"private_game_wipe ok={private_ok} n_before={n_before} wiped={wiped}")
    if not private_ok:
        bad.append("private_game_wipe")

    out = {
        "ok": not bad,
        "n_systems": bundle["n_systems"],
        "illegal": bad,
        "sample_action": a2.to_dict(),
        "memory": agent.memory.snapshot(),
        "private_game_wipe": private_ok,
    }
    print(json.dumps(out, indent=2))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
