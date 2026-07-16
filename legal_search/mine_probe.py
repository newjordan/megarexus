#!/usr/bin/env python3
"""JOB 1 mining probe — depth + legal audit (not training).

Run:
  python3 -B legal_search/mine_probe.py
  python3 -B legal_search/mine_probe.py --arcade

Does NOT write MINING_COMPLETE.json. That gate is human/agent sign-off
only after this probe and MINING_THEN_TRAIN.md A–E are honestly green.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_search.consume.registry import (  # noqa: E402
    CONSUME_REGISTRY,
    consume_all,
    list_systems,
)
from legal_search.contract import (  # noqa: E402
    BANNED_PRIOR_PATTERNS,
    BANNED_SOURCE_MARKERS,
    TRUTH_BOUNDARY,
    scan_text_for_bans,
)
from legal_search.loop import LegalAgent  # noqa: E402
from legal_search.train.collect import synthetic_episode  # noqa: E402
from legal_search.train.features import (  # noqa: E402
    FEATURE_DIM,
    FEATURE_SCHEMA,
    GLOBAL_DIM,
    SYS_SLOT,
    SYSTEM_ORDER,
    encode_from_frame,
    feature_report,
)
from legal_search.train.mining_gate import mining_is_complete  # noqa: E402
from legal_search.types import Action, Transition  # noqa: E402

# Systems that must show structured depth (checklist C)
DEPTH_CRITICAL = (
    "portband",
    "contra_head",
    "deltaband",
    "ghost",
    "dreamweaver_sim3d",
    "stochastic_goose",
)

LEGAL_SEARCH = ROOT / "legal_search"
SCAN_GLOBS = (
    "legal_search/**/*.py",
    "legal_search/**/*.md",
)
# Train/collect may mention bans only as ban-list strings — allow in these files.
ALLOW_BAN_MENTION = {
    "legal_search/contract.py",
    "legal_search/consume/misc.py",
    "legal_search/mine_probe.py",
    "legal_search/MINING_MANIFEST.md",
    "legal_search/MINING_THEN_TRAIN.md",
    "legal_search/docs/MINING_HANDOFF.md",
    "legal_search/docs/FULL_MINE_TO_LORA.md",
    "legal_search/README.md",
    "PRIMARY_LANE.md",
}


def _rows_to_transitions(rows: list[dict], game_id: str) -> list[Transition]:
    out: list[Transition] = []
    for i, r in enumerate(rows):
        after = rows[i + 1]["frame"] if i + 1 < len(rows) else r["frame"]
        out.append(
            Transition(
                game_id=game_id,
                step=i,
                frame_before=r["frame"],
                action=Action(
                    int(r["action_id"]),
                    int(r["x"]) if r.get("x") else None,
                    int(r["y"]) if r.get("y") else None,
                    source=str(r.get("source") or "mine"),
                ),
                frame_after=after,
                levels_before=0,
                levels_after=int(r.get("levels_after") or 0),
                frame_changed=bool(r.get("did_change")),
                levels_up=bool(r.get("levels_up")),
            )
        )
    return out


def _slot(vec: np.ndarray, sid: str) -> np.ndarray:
    idx = SYSTEM_ORDER.index(sid)
    base = GLOBAL_DIM + idx * SYS_SLOT
    return vec[base : base + SYS_SLOT]


def probe_features_synthetic(seed: int = 7) -> dict:
    rows = synthetic_episode(np.random.default_rng(seed), max_steps=36)
    game_id = "synthetic_blob"
    transitions = _rows_to_transitions(rows, game_id)
    frame = rows[-1]["frame"]
    cold = encode_from_frame(frame, game_id=game_id, transitions=None)
    hot = encode_from_frame(
        frame,
        game_id=game_id,
        transitions=transitions[-24:],
        available_actions=[1, 2, 3, 4, 5, 6],
        levels_completed=int(rows[-1].get("levels_after") or 0),
    )
    cold_rep = feature_report(cold)
    hot_rep = feature_report(hot)

    # Goose pure rebuild: two consumes with same window must match rates
    g = CONSUME_REGISTRY["stochastic_goose"]
    p1 = g.consume(frame, game_id=game_id, transitions=transitions[-16:])
    snap1 = p1.hypotheses[0].payload.get("action_stats")
    p2 = g.consume(frame, game_id=game_id, transitions=transitions[-16:])
    snap2 = p2.hypotheses[0].payload.get("action_stats")
    goose_pure = snap1 == snap2

    depth_critical = {}
    for sid in DEPTH_CRITICAL:
        sl = _slot(hot, sid)
        depth_critical[sid] = {
            "nz_after_present": int(np.count_nonzero(sl[1:])),
            "depth_ok": bool(np.count_nonzero(sl[1:]) >= 3),
            "slot_head": [round(float(x), 4) for x in sl[:8]],
        }

    return {
        "n_rows": len(rows),
        "n_changes": sum(1 for r in rows if r["did_change"]),
        "cold_present": cold_rep["n_systems_present"],
        "hot_present": hot_rep["n_systems_present"],
        "hot_depth_ok": hot_rep["n_systems_depth_ok"],
        "feature_schema": FEATURE_SCHEMA,
        "feature_dim": FEATURE_DIM,
        "goose_pure_rebuild": goose_pure,
        "depth_critical": depth_critical,
        "all_critical_depth_ok": all(v["depth_ok"] for v in depth_critical.values()),
        "systems": {
            sid: {
                "present": m["present"],
                "nz": m["nz_after_present"],
                "depth_ok": m["depth_ok"],
            }
            for sid, m in hot_rep["systems"].items()
        },
    }


def probe_arcade() -> dict | None:
    try:
        from arc_agi import Arcade
        from arc_agi.base import OperationMode
    except Exception as exc:
        return {"ok": False, "reason": f"import: {exc}"}

    env_dir = ROOT / "environment_files"
    if not env_dir.exists():
        return {"ok": False, "reason": "no environment_files"}

    try:
        arc = Arcade(
            arc_api_key="anonymous",
            environments_dir=str(env_dir),
            operation_mode=OperationMode.OFFLINE,
        )
        card = arc.open_scorecard()
        env = arc.make("sp80-589a99af", scorecard_id=card)
        fr = env.reset()
    except Exception as exc:
        return {"ok": False, "reason": f"reset: {exc}"}

    grid = np.array(fr.frame)
    if grid.ndim == 3:
        grid = grid[0]
    grid = np.asarray(grid, dtype=np.int16)
    levels = int(getattr(fr, "levels_completed", 0) or 0)
    vec = encode_from_frame(
        grid,
        game_id="sp80-589a99af",
        transitions=[],
        available_actions=[1, 2, 3, 4, 5, 6],
        levels_completed=levels,
    )
    rep = feature_report(vec)
    return {
        "ok": True,
        "game_id": "sp80-589a99af",
        "frame_shape": list(grid.shape),
        "n_systems_present": rep["n_systems_present"],
        "n_systems_depth_ok": rep["n_systems_depth_ok"],
        "feature_dim": FEATURE_DIM,
    }


def private_game_wipe_test() -> dict:
    """Same binary, wipe memory, new game_id still mines."""
    frame = np.full((64, 64), 12, dtype=np.int16)
    frame[8:12, 8:20] = 8
    frame2 = frame.copy()
    frame2[8:12, 9:21] = 8
    frame2[8:12, 8] = 12

    agent = LegalAgent(game_id="game-alpha")
    agent.reset_for_game("game-alpha")
    tr = Transition(
        game_id="game-alpha",
        step=0,
        frame_before=frame,
        action=Action(4, source="mine"),
        frame_after=frame2,
        levels_before=0,
        levels_after=0,
        frame_changed=True,
        levels_up=False,
    )
    agent.observe_transition(tr)
    mem_alpha = agent.memory.snapshot()["n_transitions"]

    agent.reset_for_game("game-beta-private")
    mem_beta = agent.memory.snapshot()
    a = agent.decide(frame, available_actions=[1, 2, 3, 4, 5, 6])
    bundle = consume_all(
        frame,
        game_id="game-beta-private",
        transitions=[],
        available_actions=[1, 2, 3, 4, 5, 6],
    )
    return {
        "ok": (
            mem_alpha >= 1
            and mem_beta["n_transitions"] == 0
            and mem_beta["cross_game_memory_admitted"] is False
            and bundle["n_systems"] == 25
            and a.action_id in range(1, 7)
        ),
        "mem_alpha_n": mem_alpha,
        "mem_beta": mem_beta,
        "sample_action": a.to_dict(),
        "n_systems": bundle["n_systems"],
    }


def legal_import_audit() -> dict:
    """AST-scan legal_search train/collect/consume for banned path imports."""
    hits: list[dict] = []
    py_files = sorted(LEGAL_SEARCH.rglob("*.py"))
    ban_needles = (
        "_invalid",
        "spill_model",
        "dropletsim",
        "native_scene",
        "env_win_genome",
        "fixture_genome_prior",
    )
    for path in py_files:
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        text = path.read_text(encoding="utf-8", errors="replace")
        # import-level: look for import of banned modules
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            hits.append({"file": rel, "kind": "syntax_error", "detail": str(exc)})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    for ban in ban_needles:
                        if ban in name:
                            hits.append({"file": rel, "kind": "import", "detail": name})
            elif isinstance(node, ast.ImportFrom) and node.module:
                for ban in ban_needles:
                    if ban in node.module:
                        hits.append(
                            {"file": rel, "kind": "import_from", "detail": node.module}
                        )
        # string markers outside allowlist (train labels must not use ban sources)
        if rel in ALLOW_BAN_MENTION or rel.startswith("legal_search/docs/"):
            continue
        if "/train/" in f"/{rel}" or rel.endswith("collect.py") or "/consume/" in f"/{rel}":
            for ban in ban_needles:
                # only flag if looks like a path use, not a ban list comment
                if ban in text and f'"{ban}' not in text and f"'{ban}" not in text:
                    # fence files list banned names as strings — ok if in list/tuple
                    continue
            # hard: no import of codexlab._invalid etc
            if "codexlab._invalid" in text or "orchestration._invalid" in text:
                hits.append({"file": rel, "kind": "invalid_path", "detail": "_invalid"})

    # quarantine fence must list bans
    q = CONSUME_REGISTRY["quarantine_fence"].consume(np.zeros((8, 8), dtype=np.int16))
    banned = (q.hypotheses[0].payload or {}).get("banned") or []
    return {
        "ok": len(hits) == 0 and len(banned) >= 3,
        "import_hits": hits,
        "quarantine_banned": banned,
        "truth_boundary": dict(TRUTH_BOUNDARY),
        "label_sources": [
            "did_change (env-measured frame delta)",
            "weak taken-action (what agent actually did)",
            "optional levels_up weight (env levels_completed only)",
        ],
        "ban_list": list(BANNED_SOURCE_MARKERS) + list(BANNED_PRIOR_PATTERNS),
    }


def registry_manifest_sync() -> dict:
    reg = set(CONSUME_REGISTRY.keys())
    order = set(SYSTEM_ORDER)
    manifest = (LEGAL_SEARCH / "MINING_MANIFEST.md").read_text(encoding="utf-8")
    missing_in_manifest = [s for s in sorted(reg) if f"`{s}`" not in manifest and s not in manifest]
    # softer: system id as bare word
    missing_soft = [s for s in sorted(reg) if s not in manifest]
    return {
        "n_registry": len(reg),
        "n_feature_order": len(order),
        "registry_eq_order": reg == order,
        "missing_in_manifest": missing_soft,
        "ok": reg == order and len(reg) == 25 and not missing_soft,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="JOB 1 mining probe")
    ap.add_argument("--arcade", action="store_true", help="also probe Arcade reset frame")
    ap.add_argument("--json-out", type=str, default="", help="write full report JSON")
    args = ap.parse_args()

    print("=== JOB 1 mine probe (not training) ===")
    systems = list_systems()
    print(f"registered_systems={len(systems)}")

    synth = probe_features_synthetic()
    print(
        f"[features/synth] present cold={synth['cold_present']} hot={synth['hot_present']} "
        f"depth_ok={synth['hot_depth_ok']}/25 goose_pure={synth['goose_pure_rebuild']}"
    )
    for sid, d in synth["depth_critical"].items():
        flag = "OK" if d["depth_ok"] else "THIN"
        print(f"  [{flag}] {sid:24s} nz={d['nz_after_present']} head={d['slot_head']}")

    priv = private_game_wipe_test()
    print(f"[private_game] ok={priv['ok']} mem_alpha={priv['mem_alpha_n']} beta={priv['mem_beta']}")

    audit = legal_import_audit()
    print(
        f"[legal_audit] ok={audit['ok']} import_hits={len(audit['import_hits'])} "
        f"quarantine={audit['quarantine_banned']}"
    )

    sync = registry_manifest_sync()
    print(
        f"[manifest] ok={sync['ok']} n={sync['n_registry']} "
        f"missing={sync['missing_in_manifest']}"
    )

    arcade = None
    if args.arcade:
        arcade = probe_arcade()
        print(f"[arcade] {arcade}")

    debt_path = LEGAL_SEARCH / "docs" / "MINING_DEBT.md"
    debt_exists = debt_path.is_file()
    # Surface checklist can be green while algorithm debt remains. Gate requires
    # human/agent sign-off against MINING_DEBT.md — probe never auto-opens gate.
    checklist = {
        "registry_smoke": len(systems) == 25 and sync["registry_eq_order"],
        "features_25_present": synth["hot_present"] == 25
        and (arcade is None or not arcade.get("ok") or arcade.get("n_systems_present") == 25),
        "depth_not_stub": synth["all_critical_depth_ok"] and synth["goose_pure_rebuild"],
        "docs_synced": sync["ok"] and debt_exists,
        "legal_audit": audit["ok"] and priv["ok"],
    }
    print("--- surface checklist (necessary, NOT sufficient for gate) ---")
    for k, v in checklist.items():
        print(f"  {k}: {v}")
    print(f"mining_is_complete()={mining_is_complete()}")
    print(f"surface_checklist_true={all(checklist.values())}")
    print(f"mining_debt_doc={debt_exists} path=legal_search/docs/MINING_DEBT.md")
    print("ready_for_gate=False  # pay MINING_DEBT.md first; never auto-write gate from probe")

    report = {
        "schema": "legal_search.mine_probe.v1",
        "checklist_preview": checklist,
        "surface_checklist_true": all(checklist.values()),
        "ready_for_gate": False,
        "mining_debt_doc": str(debt_path) if debt_exists else None,
        "synthetic": synth,
        "private_game": priv,
        "legal_audit": audit,
        "manifest": sync,
        "arcade": arcade,
        "gate_open": mining_is_complete(),
        "note": (
            "Surface checklist != mining complete. Read docs/MINING_DEBT.md. "
            "Do not write MINING_COMPLETE.json until real algorithm debt is paid."
        ),
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")

    # Exit 0 if mine probe machinery works; checklist may still be partial
    ok_run = (
        synth["hot_present"] == 25
        and synth["goose_pure_rebuild"]
        and priv["ok"]
        and audit["ok"]
    )
    return 0 if ok_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
