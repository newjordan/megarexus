#!/usr/bin/env python3
"""Durable tests for legal shaft mines — drive shipped functions only.

No golden scores hardcoded as magic pass conditions beyond structure/legality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_search.algorithms.calibrate import mine_calibration  # noqa: E402
from legal_search.algorithms.cem_plan import discrete_cem_plan  # noqa: E402
from legal_search.algorithms.components import connected_components  # noqa: E402
from legal_search.algorithms.crucible_mine import mine_crucible  # noqa: E402
from legal_search.algorithms.dynamics import mine_dynamics  # noqa: E402
from legal_search.algorithms.perception_trust import gate_frame_structure  # noqa: E402
from legal_search.algorithms.ports import mine_ports_full  # noqa: E402
from legal_search.algorithms.shape_traces import mine_shape_traces  # noqa: E402
from legal_search.algorithms.statecon_graph import mine_statecon  # noqa: E402
from legal_search.consume.registry import consume_all  # noqa: E402
from legal_search.train.collect import synthetic_episode  # noqa: E402
from legal_search.train.mining_gate import mining_is_complete  # noqa: E402
from legal_search.types import Action, Transition  # noqa: E402


def _blob_frame() -> np.ndarray:
    g = np.full((64, 64), 12, dtype=np.int16)
    g[10:16, 10:20] = 8
    g[10:16, 30:40] = 11
    g[40:48, 40:48] = 6
    return g


def _transitions_from_synth(seed: int = 2, steps: int = 24) -> tuple[np.ndarray, list[Transition]]:
    rows = synthetic_episode(np.random.default_rng(seed), max_steps=steps)
    trs: list[Transition] = []
    for i, r in enumerate(rows):
        after = rows[i + 1]["frame"] if i + 1 < len(rows) else r["frame"]
        trs.append(
            Transition(
                game_id="test_synth",
                step=i,
                frame_before=r["frame"],
                action=Action(
                    int(r["action_id"]),
                    int(r["x"]) if r.get("x") else None,
                    int(r["y"]) if r.get("y") else None,
                    source=str(r.get("source") or "t"),
                ),
                frame_after=after,
                levels_before=0,
                levels_after=int(r.get("levels_after") or 0),
                frame_changed=bool(r.get("did_change")),
                levels_up=bool(r.get("levels_up")),
            )
        )
    return rows[-1]["frame"], trs


def test_ports_fleet_and_edge_lock_structured():
    frame = _blob_frame()
    comps = connected_components(frame)
    mined = mine_ports_full(comps, frame_shape=frame.shape[:2], transitions=None)
    assert mined["n_ports"] == len(comps) * 4
    assert isinstance(mined["alignments"], list)
    assert "edge_lock_ledger" in mined
    assert "fleet" in mined
    assert "ranked_candidates" in mined["fleet"]
    assert mined["fleet"].get("admitted") is False
    assert mined["edge_lock_ledger"].get("not_solve_truth") is True


def test_ports_event_surfaces_from_transitions():
    frame, trs = _transitions_from_synth()
    comps = connected_components(frame)
    mined = mine_ports_full(comps, frame_shape=frame.shape[:2], transitions=trs[-12:])
    events = mined["event_surfaces"]
    assert "events" in events
    assert events.get("admitted") is False
    # when transitions change frames, expect some events or empty list with counts
    assert events["n_events"] == len(events["events"])


def test_statecon_branch_graph():
    mined = mine_statecon(_blob_frame())
    g = mined["branch_graph"]
    assert g["n_nodes"] >= 1
    assert g["n_edges"] >= 0
    assert mined["n_support_groups"] >= 1
    assert mined.get("any_admitted") is False
    assert mined.get("not_engine_truth") is True


def test_crucible_falsify_and_rules():
    _, trs = _transitions_from_synth()
    mined = mine_crucible(trs)
    assert mined["n_transitions"] > 0
    assert mined["source_rules_used"] is False
    assert "rule_candidates" in mined
    assert mined["n_rules"] == len(mined["rule_candidates"])
    for r in mined["rule_candidates"]:
        assert r.get("admitted") is False
        assert r.get("falsified") is not True


def test_dynamics_no_fixtures():
    frame, trs = _transitions_from_synth()
    dyn = mine_dynamics(frame, trs[-16:])
    assert dyn["fixture_priors_loaded"] is False
    assert dyn["shipped_dynamics_json_loaded"] is False
    assert len(dyn["hypotheses"]) >= 2
    assert dyn["best_hypothesis"]


def test_perception_trust_no_sp80_scene():
    trust = gate_frame_structure(_blob_frame())
    assert trust["fixture_scene_loaded"] is False
    assert trust["sp80_roles_used"] is False
    assert 0.0 <= trust["trust_score"] <= 1.0
    assert trust["n_checks"] >= 3


def test_cem_plan_structured():
    frame, trs = _transitions_from_synth(steps=16)
    plan = discrete_cem_plan(frame, transitions=trs[-12:], seed=1)
    assert len(plan["best_sequence"]) == plan["horizon"]
    assert plan["requires_lewm_weights"] is False
    assert plan["proposals"]
    assert all(p.actions for p in plan["proposals"])


def test_calibration_stage0_stage1():
    frame, trs = _transitions_from_synth()
    cal = mine_calibration(frame, trs)
    assert cal["may_admit_truth"] is False
    assert cal["admitted_actor"] is False
    assert "stage0" in cal and cal["stage0"]["n_ranked"] >= 1
    assert "stage1" in cal and "change_rate" in cal["stage1"]
    assert cal["stage0"]["may_admit_truth"] is False


def test_shape_traces_color_groups_no_roles():
    traces = mine_shape_traces(_blob_frame())
    assert traces["n_traces"] >= 1
    assert traces["semantic_labels_assigned"] is False
    assert traces["n_color_groups"] >= 1
    for g in traces["color_groups"]:
        assert g["semantic_role"] is None


def test_consume_all_high_debt_shafts_structured():
    frame, trs = _transitions_from_synth()
    bundle = consume_all(
        frame,
        game_id="test_synth",
        transitions=trs[-16:],
        available_actions=[1, 2, 3, 4, 5, 6],
    )
    assert bundle["n_systems"] == 25
    systems = bundle["systems"]

    pb = systems["portband"]["hypotheses"]
    kinds = {h["kind"] for h in pb}
    assert "fleet_combos" in kinds
    assert "event_surfaces" in kinds
    fleet = next(h["payload"] for h in pb if h["kind"] == "fleet_combos")
    assert "n_candidates" in fleet

    sc = systems["statecon"]["hypotheses"][0]["payload"]
    assert "branch_graph" in sc
    assert sc["branch_graph"]["n_nodes"] >= 1
    assert sc.get("any_admitted") is False

    dw = systems["dreamweaver_sim3d"]["hypotheses"][0]["payload"]
    assert dw["fixture_priors_loaded"] is False
    assert "perception_trust" in dw
    assert dw["perception_trust"]["fixture_scene_loaded"] is False

    loom = systems["loom_lewm_rexus"]["hypotheses"][0]["payload"]
    assert "cem" in loom
    assert loom["cem"]["requires_lewm_weights"] is False
    assert loom["cem"]["best_sequence"]

    cru = systems["crucible"]["hypotheses"][0]["payload"]
    assert cru["source_rules_used"] is False
    assert "n_falsified" in cru or "rule_candidates" in cru

    cal = systems["match_calibration"]["hypotheses"][0]["payload"]
    assert cal["may_admit_truth"] is False
    assert "stage0" in cal

    vis = systems["contra_vision"]["hypotheses"][0]["payload"]
    assert vis["semantic_labels_assigned"] is False
    assert "color_groups" in vis

    # legality: no hyp admitted
    for sid, pkt in systems.items():
        for h in pkt.get("hypotheses") or []:
            assert h.get("admitted") is not True, sid


def test_gate_stays_closed():
    assert mining_is_complete() is False


if __name__ == "__main__":
    # allow running without pytest installed
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:
                fails += 1
                print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if fails else 0)
