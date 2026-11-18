"""ALL Rexus-folder ideas as base-model seeds.

Source: external Rexus archive when REXUS_ROOT is set (+ notes/REXUS_CONCEPT_SWEEP_20260715.md).
Doctrine: plant everything; training rises what works. Spill/win-bank = fence only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from synthesis_lab.orchestration.rdl.brain.brain_types import KnownMapView, OperatorResult, TruthFlags

_MEGA = Path(__file__).resolve().parents[5]
_REXUS = Path(os.environ.get("REXUS_ROOT", _MEGA / ".external" / "rexus"))

REXUS_IDEA_IDS: tuple[str, ...] = (
    "R1_admission_fail_closed",
    "R2_paper_faithful_chassis",
    "R3_rexus_action_owner",
    "R4_opred_candidates",
    "R5_state_prober",
    "R6_gram_plus_gate",
    "R7_lprm_process_reward",
    "R8_full_contra_system",
    "R9_config_oracle_view_required",
    "R10_action_state_tags",
    "R11_action_interaction_graph",
    "R12_action_interaction_matrix",
    "R13_portband_fleet_lineup",
    "R14_structural_sense",
    "R15_shape_fit",
    "R16_fill_sense",
    "R17_ghost_memory",
    "R18_crucible_induction",
    "R19_source_edge_outcome",
    "R20_activation_transitions",
    "R21_deltaband_controlled",
    "R22_exploreband",
    "R23_actionmodband",
    "R24_match_calibration",
    "R25_relationship_ladder",
    "R26_episode_memory",
    "R27_tabula_rasa",
    "R28_view_system",
    "R29_action_semantics_ledger",
    "R30_teacher_student",
    "R31_human_goal_calculus",
    "R32_winbank_eval_fence",
    "R33_mechanic_decode_25",
    "R34_mechanic_clusters",
    "R35_sam_gates",
    "R36_paint_control_skill",
    "R37_puffer_rl",
    "R38_cross_game_isolation",
    "R39_angel_report_sop",
    "R40_groundhog_paintflow",
    "R41_sp80_lora_adapter",
    "R42_control_drift",
    "R43_recommendation_optimum",
    "R44_goal_localization",
    "R45_spill_quarantine_fence",
    "R46_like_control_prior",
    "R47_macroband_ranking",
    "R48_live_suite_runners",
    "R49_contra_service",
    "R50_players_primitives",
    "R51_lewm_bridge",
    "R52_sp80_corpus",
    "R53_novel_systems_inventory",
    "R54_oversight",
    "R55_perception",
    "R56_controls",
    "R57_debug_summary",
    "R58_boundary_pass",
    "R59_human_replay_testbed",
    "R60_not_prediction_flags",
)


def _ok(name: str, scores: dict[str, float], **kw: Any) -> OperatorResult:
    return OperatorResult(
        name=name,
        status="ok",
        scores={k: float(v) for k, v in scores.items()},
        hypothesis_only=True,
        truth_flags=TruthFlags(),
        meta={"catalog": "rexus", **(kw.pop("meta", {}) or {})},
        **kw,
    )


def _seed(name: str, scores: dict[str, float], msg: str, **kw: Any) -> OperatorResult:
    return OperatorResult(
        name=name,
        status=kw.pop("status", "ok"),
        scores={k: float(v) for k, v in scores.items()},
        messages=[msg],
        hypothesis_only=True,
        truth_flags=TruthFlags(),
        meta={"catalog": "rexus", "seed": True, **(kw.pop("meta", {}) or {})},
        **kw,
    )


def _exists(*parts: str) -> float:
    return 1.0 if (_REXUS.joinpath(*parts)).exists() else 0.0


def _mega_exists(*parts: str) -> float:
    return 1.0 if (_MEGA.joinpath(*parts)).exists() else 0.0


def _partial(ctx: dict[str, Any]) -> dict[str, Any]:
    return ctx.get("dream_summary_partial") or {}


def _pkt(ctx: dict[str, Any], key: str) -> dict[str, Any]:
    p = _partial(ctx).get(key) or {}
    return p if isinstance(p, dict) else {}


def _scores(ctx: dict[str, Any], key: str) -> dict[str, Any]:
    return (_pkt(ctx, key).get("scores") or {}) if isinstance(_pkt(ctx, key), dict) else {}


# ── seeds ────────────────────────────────────────────────────────────────


def R1(mv, ctx):
    contra = ctx.get("contra_packet") or {}
    return _ok(
        "R1_admission_fail_closed",
        {
            "policy_on_disk": _exists("ADMISSION_POLICY.md"),
            "admission_status_not_admitted": 1.0
            if str(contra.get("admission_status") or "") in ("not_admitted", "fail_closed", "")
            else 0.0,
            "control_truth_admitted": 1.0 if contra.get("control_truth_admitted") else 0.0,
            "solve_condition_admitted": 1.0 if contra.get("solve_condition_admitted") else 0.0,
            "n_fail_closed_reasons": float(len(contra.get("fail_closed_reasons") or [])),
            "fail_closed_seed": 1.0,
        },
        messages=["admission_policy"],
        meta={"backend": "rexus_ADMISSION+contra"},
    )


def R2(mv, ctx):
    ckpt = _REXUS / "base_checkpoints" / "2026-05-09_chassis_paper_faithful_v1" / "best.pt"
    mega = _MEGA / "architecture" / "src" / "lewm"
    return _ok(
        "R2_paper_faithful_chassis",
        {
            "chassis_ckpt": 1.0 if ckpt.is_file() else 0.0,
            "lewm_src": 1.0 if mega.is_dir() or (_REXUS / "architecture" / "src" / "lewm").is_dir() else 0.0,
            "upstream_le_wm": _exists("architecture", "upstream_le_wm"),
            "paper_faithful_seed": 1.0,
        },
        messages=["chassis_base"],
        meta={"backend": "base_checkpoints"},
    )


def R3(mv, ctx):
    return _ok(
        "R3_rexus_action_owner",
        {
            "central_commits": 1.0,
            "lewis_does_not_act": 1.0,
            "action_owner_seed": 1.0,
            "loop_12_step_seed": 1.0,
        },
        messages=["rexus_owns_action"],
        meta={"backend": "megamind_central"},
    )


def R4(mv, ctx):
    world = ctx.get("world_summary") or {}
    return _ok(
        "R4_opred_candidates",
        {
            "opred_seed": 1.0,
            "pred_norm": float(world.get("pred_norm") or 0),
            "wm_live": 1.0 if (world.get("wm_plug") or {}).get("status") == "ok" else 0.0,
            "n_candidates_proxy": float(len(ctx.get("available_actions") or [1, 2, 3, 4, 5, 6])),
        },
        messages=["opred"],
        meta={"backend": "hemisphere_E_pred"},
    )


def R5(mv, ctx):
    world = ctx.get("world_summary") or {}
    return _ok(
        "R5_state_prober",
        {
            "disagreement": float(world.get("disagreement") or ctx.get("disagreement") or 0),
            "judge_I": float(world.get("judge_I") or ctx.get("judge_I") or 0),
            "prober_seed": 1.0,
        },
        messages=["state_prober"],
        meta={"backend": "I_vs_E_delta"},
    )


def R6(mv, ctx):
    return _ok(
        "R6_gram_plus_gate",
        {
            "gram_plus_seed": 1.0,
            "docs_present": max(
                _exists("architecture", "docs", "canon_architecture", "README.md"),
                _mega_exists("architecture", "docs", "canon_architecture", "README.md"),
            ),
            "path_fan_seed": 1.0,
            "lattice_seed": 1.0,
        },
        messages=["gram_plus"],
        meta={"backend": "canon_docs"},
    )


def R7(mv, ctx):
    return _ok(
        "R7_lprm_process_reward",
        {
            "lprm_seed": 1.0,
            "process_not_terminal_only": 1.0,
            "vector_reward_seed": 1.0,
        },
        messages=["lprm"],
        meta={"backend": "process_reward_seed"},
    )


def R8(mv, ctx):
    sc = _scores(ctx, "contra_view")
    return _ok(
        "R8_full_contra_system",
        {
            "contra_view_ok": 1.0 if _pkt(ctx, "contra_view").get("status") == "ok" else 0.0,
            "n_keys": float(sc.get("n_keys") or 0),
            "full_system_seed": 1.0,
        },
        messages=["full_contra"],
        meta={"backend": "run_contra_view"},
    )


def R9(mv, ctx):
    parts = ["contra_oracle", "contra_view", "contra_admission"]
    live = sum(1 for p in parts if _pkt(ctx, p).get("status") == "ok")
    # also config_pass via contra packet
    pkt = ctx.get("contra_packet") or {}
    has_config = 1.0 if pkt.get("config_pass") else 0.0
    has_oracle = 1.0 if pkt.get("oracle_pass") else 0.0
    has_view = 1.0 if pkt.get("view_system_passes") else 0.0
    return _ok(
        "R9_config_oracle_view_required",
        {
            "config_pass": has_config,
            "oracle_pass": has_oracle,
            "view_passes": has_view,
            "required_triple": float(has_config * has_oracle * has_view),
            "layers_ok": float(live),
        },
        messages=["config_oracle_view"],
        meta={"backend": "contra_packet_required_passes"},
    )


def R10(mv, ctx):
    sc = _scores(ctx, "contra_semantics")
    return _ok(
        "R10_action_state_tags",
        {
            "semantics_present": float(sc.get("present") or 0),
            "n_keys": float(sc.get("n_keys") or 0),
            "generic_not_paint_only": 1.0,
            "tag_seed": 1.0,
        },
        messages=["action_state_tags"],
        meta={"backend": "action_semantics_ledger"},
    )


def R11(mv, ctx):
    # graph from portband/conceptual alignments as proxy for interaction graph
    pb = _scores(ctx, "portband")
    return _ok(
        "R11_action_interaction_graph",
        {
            "n_alignments": float(pb.get("n_alignments") or 0),
            "n_ports": float(pb.get("n_ports") or 0),
            "graph_seed": 1.0,
            "module_disk": max(
                _exists("codexlab", "cross_game_action_interaction_probe.py"),
                _mega_exists("codexlab", "cross_game_action_interaction_probe.py"),
            ),
        },
        messages=["interaction_graph"],
        meta={"backend": "portband_as_graph"},
    )


def R12(mv, ctx):
    pb = _scores(ctx, "portband")
    return _ok(
        "R12_action_interaction_matrix",
        {
            "n_top_lineup": float(pb.get("n_top_lineup") or pb.get("n_alignments") or 0),
            "best_score": float(pb.get("best_lineup_score") or pb.get("best_alignment_score") or 0),
            "matrix_seed": 1.0,
            "not_action_authority": 1.0,
        },
        messages=["interaction_matrix"],
        meta={"backend": "lineup_matrix_scores"},
    )


def R13(mv, ctx):
    pb = _scores(ctx, "portband")
    cp = _scores(ctx, "contra_portband")
    return _ok(
        "R13_portband_fleet_lineup",
        {
            "legal_portband_ok": 1.0 if _pkt(ctx, "portband").get("status") == "ok" else 0.0,
            "contra_portband_ok": 1.0 if _pkt(ctx, "contra_portband").get("status") == "ok" else 0.0,
            "best_lineup": float(pb.get("best_lineup_score") or cp.get("n_lineup_rows") or 0),
            "n_ports": float(pb.get("n_ports") or cp.get("n_ports") or 0),
            "fleet_seed": 1.0,
        },
        messages=["portband_fleet"],
        meta={"backend": "portband+contra_portband"},
    )


def R14(mv, ctx):
    frame = np.asarray(mv.frame) if mv.frame is not None else None
    scores = {"structural_sense_seed": 1.0}
    try:
        # prefer rexus structural sense
        sys.path.insert(0, str(_REXUS / "codexlab"))
        from contra_structural_sense import detect_structure  # type: ignore

        if frame is not None:
            st = detect_structure(frame)
            floors = st.get("floors") or st.get("floor") or []
            walls = st.get("walls") or st.get("wall") or []
            scores["n_floors"] = float(len(floors) if hasattr(floors, "__len__") else (1 if floors else 0))
            scores["n_walls"] = float(len(walls) if hasattr(walls, "__len__") else (1 if walls else 0))
            scores["abstain_capable"] = 1.0
            scores["backend_live"] = 1.0
            return _ok("R14_structural_sense", scores, messages=["structural_sense_live"], meta={"backend": "rexus.contra_structural_sense"})
    except Exception as exc:
        scores["backend_live"] = 0.0
        scores["err"] = 0.0
        scores["msg"] = 0.0
        # geometric fallback
        comps = list(mv.components or [])
        scores["n_components"] = float(len(comps))
        return _ok("R14_structural_sense", scores, messages=[f"fallback:{type(exc).__name__}"], meta={"backend": "fallback_geom"})


def R15(mv, ctx):
    scores = {
        "shape_fit_seed": 1.0,
        "module_disk": max(
            _exists("codexlab", "contra_shape_fit.py"),
            _mega_exists("codexlab", "contra_shape_fit.py"),
        ),
    }
    # use portband alignment as live fit signal
    pb = _scores(ctx, "portband")
    scores["best_fit_proxy"] = float(pb.get("best_lineup_score") or 0)
    return _ok("R15_shape_fit", scores, messages=["shape_fit"], meta={"backend": "shape_fit+lineup"})


def R16(mv, ctx):
    body = _scores(ctx, "body")
    return _ok(
        "R16_fill_sense",
        {
            "mouth_fill_sum": float(body.get("mouth_fill_sum") or 0),
            "winner_ok": float(body.get("winner_ok") or 0),
            "margin": float(body.get("margin") or 0),
            "fill_sense_seed": 1.0,
            "module_disk": max(
                _exists("codexlab", "contra_fill_sense.py"),
                _mega_exists("codexlab", "contra_fill_sense.py"),
            ),
        },
        messages=["fill_sense"],
        meta={"backend": "body_fill+optional_module"},
    )


def R17(mv, ctx):
    g = _scores(ctx, "ghost")
    cg = _scores(ctx, "contra_ghost")
    return _ok(
        "R17_ghost_memory",
        {
            "ghost_ok": 1.0 if _pkt(ctx, "ghost").get("status") == "ok" else 0.0,
            "contra_ghost_ok": 1.0 if _pkt(ctx, "contra_ghost").get("status") == "ok" else 0.0,
            "n_confirmed": float(g.get("n_confirmed") or 0),
            "ghost_seed": 1.0,
        },
        messages=["ghost"],
        meta={"backend": "ghost+contra_ghost"},
    )


def R18(mv, ctx):
    c = _scores(ctx, "crucible")
    return _ok(
        "R18_crucible_induction",
        {
            "crucible_ok": 1.0 if _pkt(ctx, "crucible").get("status") == "ok" else 0.0,
            "n_rules": float(c.get("n_rules") or 0),
            "n_discrepancies": float(c.get("n_discrepancies") or 0),
            "n_falsified": float(c.get("n_falsified") or 0),
            "v0_match": float(c.get("v0_match_fraction") or 0),
            "crucible_seed": 1.0,
        },
        messages=["crucible_full"],
        meta={"backend": "mine_crucible"},
    )


def R19(mv, ctx):
    return _ok(
        "R19_source_edge_outcome",
        {
            "module_disk": max(
                _exists("codexlab", "crucible_build_source_edge_outcome_model.py"),
                _mega_exists("codexlab", "crucible_build_source_edge_outcome_model.py"),
            ),
            "seed": 1.0,
            "dataset_disk": max(
                _exists("codexlab", "datasets", "sp80_source_edge_outcome_v0"),
                _mega_exists("codexlab", "datasets", "sp80_source_edge_outcome_v0"),
            ),
        },
        messages=["source_edge"],
        meta={"backend": "crucible_source_edge"},
    )


def R20(mv, ctx):
    return _ok(
        "R20_activation_transitions",
        {
            "module_disk": max(
                _exists("codexlab", "crucible_mine_activation_transitions.py"),
                _mega_exists("codexlab", "crucible_mine_activation_transitions.py"),
            ),
            "dataset_disk": max(
                _exists("codexlab", "datasets", "sp80_crucible_activation_transitions_v0"),
                _mega_exists("codexlab", "datasets", "sp80_crucible_activation_transitions_v0"),
            ),
            "seed": 1.0,
        },
        messages=["activation_transitions"],
        meta={"backend": "crucible_activation"},
    )


def R21(mv, ctx):
    d = _scores(ctx, "delta")
    return _ok(
        "R21_deltaband_controlled",
        {
            "delta_ok": 1.0 if _pkt(ctx, "delta").get("status") == "ok" else 0.0,
            "deltaband_present": float(d.get("deltaband_present") or 0),
            "delta_pixels": float(d.get("delta_pixels") or 0),
            "controlled_run_disk": max(
                _exists("codexlab", "sp80_deltaband_controlled_run.py"),
                _mega_exists("codexlab", "sp80_deltaband_controlled_run.py"),
            ),
            "seed": 1.0,
        },
        messages=["deltaband"],
        meta={"backend": "delta_full"},
    )


def R22(mv, ctx):
    sc = _scores(ctx, "contra_explore")
    return _ok(
        "R22_exploreband",
        {
            "contra_explore_ok": 1.0 if _pkt(ctx, "contra_explore").get("status") == "ok" else 0.0,
            "n_keys": float(sc.get("n_keys") or 0),
            "seed": 1.0,
        },
        messages=["exploreband"],
        meta={"backend": "contra_exploreband"},
    )


def R23(mv, ctx):
    return _ok(
        "R23_actionmodband",
        {
            "module_disk": max(
                _exists("codexlab", "contra_head", "actionmodband.py"),
                _mega_exists("codexlab", "contra_head", "actionmodband.py"),
            ),
            "verify_disk": max(
                _exists("codexlab", "contra_verify_actionmodband.py"),
                _mega_exists("codexlab", "contra_verify_actionmodband.py"),
            ),
            "seed": 1.0,
        },
        messages=["actionmodband"],
        meta={"backend": "actionmodband_module"},
    )


def R24(mv, ctx):
    return _ok(
        "R24_match_calibration",
        {
            "module_disk": max(
                _exists("codexlab", "contra_head", "match_calibration"),
                _mega_exists("codexlab", "contra_head", "match_calibration"),
            ),
            "stage0_seed": 1.0,
            "stage1_seed": 1.0,
            "reweight_only_not_truth": 1.0,
        },
        messages=["match_calibration"],
        meta={"backend": "match_calibration"},
    )


def R25(mv, ctx):
    sc = _scores(ctx, "contra_ladder")
    return _ok(
        "R25_relationship_ladder",
        {
            "ladder_ok": 1.0 if _pkt(ctx, "contra_ladder").get("status") == "ok" else 0.0,
            "n_keys": float(sc.get("n_keys") or 0),
            "seed": 1.0,
        },
        messages=["relationship_ladder"],
        meta={"backend": "contra_ladder"},
    )


def R26(mv, ctx):
    sc = _scores(ctx, "contra_episode")
    n_tr = float(len(ctx.get("transitions") or []))
    return _ok(
        "R26_episode_memory",
        {
            "episode_ok": 1.0 if _pkt(ctx, "contra_episode").get("status") == "ok" else 0.0,
            "n_transitions": n_tr,
            "seed": 1.0,
        },
        messages=["episode_memory"],
        meta={"backend": "contra_episode+local"},
    )


def R27(mv, ctx):
    return _ok(
        "R27_tabula_rasa",
        {
            "module_disk": max(
                _exists("codexlab", "contra_head", "tabula_rasa.py"),
                _mega_exists("codexlab", "contra_head", "tabula_rasa.py"),
            ),
            "wipe_seed": 1.0,
            "private_game_ready": 1.0,
        },
        messages=["tabula_rasa"],
        meta={"backend": "tabula_rasa"},
    )


def R28(mv, ctx):
    pkt = ctx.get("contra_packet") or {}
    return _ok(
        "R28_view_system",
        {
            "view_passes": 1.0 if pkt.get("view_system_passes") else 0.0,
            "n_view_keys": float(len(pkt.get("view_system_passes") or {})),
            "seed": 1.0,
        },
        messages=["view_system"],
        meta={"backend": "view_system_passes"},
    )


def R29(mv, ctx):
    sc = _scores(ctx, "contra_semantics")
    return _ok(
        "R29_action_semantics_ledger",
        {
            "ok": 1.0 if _pkt(ctx, "contra_semantics").get("status") == "ok" else 0.0,
            "n_keys": float(sc.get("n_keys") or 0),
            "seed": 1.0,
        },
        messages=["action_semantics"],
        meta={"backend": "action_semantics_ledger"},
    )


def R30(mv, ctx):
    return _ok(
        "R30_teacher_student",
        {
            "doc_disk": max(
                _exists("codexlab", "docs", "CONTRA_TEACHER_STUDENT_SIGNAL_PLAN_2026-06-01.md"),
                _mega_exists("codexlab", "docs", "CONTRA_TEACHER_STUDENT_SIGNAL_PLAN_2026-06-01.md"),
            ),
            "dataset_disk": max(
                _exists("codexlab", "datasets", "arc_opus48_teacher_signals_v0"),
                _mega_exists("codexlab", "datasets", "arc_opus48_teacher_signals_v0"),
            ),
            "process_supervision_only": 1.0,
            "seed": 1.0,
        },
        messages=["teacher_student"],
        meta={"backend": "opus_teacher_signals"},
    )


def R31(mv, ctx):
    return _ok(
        "R31_human_goal_calculus",
        {
            "human_dir": _exists("human", "human_goal_discovery_calculus_2026-05-19"),
            "audio_dir": _exists("human", "raw_incoming_audio_and_sidecars"),
            "mega_corpus": _mega_exists("corpus", "sp80_liquid_pour"),
            "seed": 1.0,
            "not_train_labels": 1.0,
        },
        messages=["human_calculus"],
        meta={"backend": "human_transcripts"},
    )


def R32(mv, ctx):
    # FENCE: win bank exists for eval only — never as train truth
    return _ok(
        "R32_winbank_eval_fence",
        {
            "winbank_present": _exists("human", "win_bank_2026-05-16_human_cheat_map_v1"),
            "fence_active": 1.0,
            "train_use_forbidden": 1.0,
            "eval_only": 1.0,
        },
        messages=["winbank_FENCE"],
        meta={"backend": "admission_fence"},
    )


def R33(mv, ctx):
    d = _REXUS / "reference" / "arc_game_mechanic_decode_2026-05-19"
    n = len(list(d.glob("*.md"))) if d.is_dir() else 0
    return _ok(
        "R33_mechanic_decode_25",
        {
            "n_decode_md": float(n),
            "dir_present": 1.0 if d.is_dir() else 0.0,
            "seed": 1.0,
        },
        messages=["mechanic_decode"],
        meta={"backend": "arc_game_mechanic_decode"},
    )


def R34(mv, ctx):
    p = _REXUS / "reference" / "arc_game_mechanic_decode_2026-05-19" / "mechanic_clusters.json"
    scores = {"seed": 1.0, "file": 1.0 if p.is_file() else 0.0}
    if p.is_file():
        try:
            d = json.loads(p.read_text())
            scores["cluster_count"] = float(d.get("cluster_count") or len(d.get("clusters") or []))
        except Exception:
            scores["cluster_count"] = 0.0
    return _ok("R34_mechanic_clusters", scores, messages=["mechanic_clusters"], meta={"backend": "clusters_json"})


def R35(mv, ctx):
    return _ok(
        "R35_sam_gates",
        {
            "sam_scoring": max(_exists("codexlab", "sp80_sam_scoring.py"), _mega_exists("codexlab", "sp80_sam_scoring.py")),
            "sam_admission": max(
                _exists("codexlab", "sp80_sam_contra_admission_gate.py"),
                _mega_exists("codexlab", "sp80_sam_contra_admission_gate.py"),
            ),
            "sam_acceptance": max(
                _exists("codexlab", "sp80_sam_acceptance_gate.py"),
                _mega_exists("codexlab", "sp80_sam_acceptance_gate.py"),
            ),
            "seed": 1.0,
        },
        messages=["sam_gates"],
        meta={"backend": "sp80_sam_*"},
    )


def R36(mv, ctx):
    return _ok(
        "R36_paint_control_skill",
        {
            "module": max(_exists("codexlab", "paint_control_skill.py"), _mega_exists("codexlab", "paint_control_skill.py")),
            "motion_mine": max(
                _exists("codexlab", "paint_control_skill_motion_effect_mine.py"),
                _mega_exists("codexlab", "paint_control_skill_motion_effect_mine.py"),
            ),
            "seed": 1.0,
        },
        messages=["paint_control"],
        meta={"backend": "paint_control_skill"},
    )


def R37(mv, ctx):
    return _ok(
        "R37_puffer_rl",
        {
            "probe": max(_exists("codexlab", "sp80_pufferlib_probe.py"), _mega_exists("codexlab", "sp80_pufferlib_probe.py")),
            "policy": max(_exists("codexlab", "sp80_puffer_policy.py"), _mega_exists("codexlab", "sp80_puffer_policy.py")),
            "native_train": max(
                _exists("codexlab", "pufferlib_native_train_run.py"),
                _mega_exists("codexlab", "pufferlib_native_train_run.py"),
            ),
            "seed": 1.0,
            "optional_not_default_brain": 1.0,
        },
        messages=["puffer_rl"],
        meta={"backend": "pufferlib"},
    )


def R38(mv, ctx):
    return _ok(
        "R38_cross_game_isolation",
        {
            "isolation_viewer": max(
                _exists("codexlab", "cross_game_isolation_viewer.py"),
                _mega_exists("codexlab", "cross_game_isolation_viewer.py"),
            ),
            "color_board": max(
                _exists("codexlab", "cross_game_color_board_viewer.py"),
                _mega_exists("codexlab", "cross_game_color_board_viewer.py"),
            ),
            "action_probe": max(
                _exists("codexlab", "cross_game_action_interaction_probe.py"),
                _mega_exists("codexlab", "cross_game_action_interaction_probe.py"),
            ),
            "seed": 1.0,
        },
        messages=["cross_game"],
        meta={"backend": "cross_game_*"},
    )


def R39(mv, ctx):
    return _ok(
        "R39_angel_report_sop",
        {
            "sop_disk": max(
                _exists("codexlab", "docs", "CONTRA_GAMEPLAY_REPORTING_SOP.md"),
                _mega_exists("codexlab", "docs", "CONTRA_GAMEPLAY_REPORTING_SOP.md"),
            ),
            "angel_verify": max(
                _exists("codexlab", "angel_report_verify.py"),
                _mega_exists("codexlab", "angel_report_verify.py"),
            ),
            "seed": 1.0,
        },
        messages=["angel_sop"],
        meta={"backend": "gameplay_reporting"},
    )


def R40(mv, ctx):
    cond = list((_REXUS / "codexlab" / "conditions").glob("*groundhog*") if (_REXUS / "codexlab" / "conditions").is_dir() else [])
    cond_m = list((_MEGA / "codexlab" / "conditions").glob("*groundhog*") if (_MEGA / "codexlab" / "conditions").is_dir() else [])
    return _ok(
        "R40_groundhog_paintflow",
        {
            "n_conditions": float(len(cond) + len(cond_m)),
            "seed": 1.0,
        },
        messages=["groundhog"],
        meta={"backend": "conditions_groundhog"},
    )


def R41(mv, ctx):
    return _ok(
        "R41_sp80_lora_adapter",
        {
            "adapter": max(_exists("codexlab", "sp80_lora_adapter.py"), _mega_exists("codexlab", "sp80_lora_adapter.py")),
            "central_lora_live": 1.0,
            "seed": 1.0,
        },
        messages=["lora"],
        meta={"backend": "sp80_lora+central"},
    )


def R42(mv, ctx):
    return _ok(
        "R42_control_drift",
        {
            "module": max(
                _exists("codexlab", "sp80_control_drift_tracker.py"),
                _mega_exists("codexlab", "sp80_control_drift_tracker.py"),
            ),
            "seed": 1.0,
        },
        messages=["control_drift"],
        meta={"backend": "drift_tracker"},
    )


def R43(mv, ctx):
    return _ok(
        "R43_recommendation_optimum",
        {
            "module": max(
                _exists("codexlab", "sp80_recommendation_optimum_regression.py"),
                _mega_exists("codexlab", "sp80_recommendation_optimum_regression.py"),
            ),
            "seed": 1.0,
        },
        messages=["recommendation_optimum"],
        meta={"backend": "recommendation"},
    )


def R44(mv, ctx):
    sc = _scores(ctx, "contra_target")
    return _ok(
        "R44_goal_localization",
        {
            "target_hyps": float(sc.get("n_items") or sc.get("n_keys") or 0),
            "module": max(
                _exists("codexlab", "sp80_goal_localization_regression.py"),
                _mega_exists("codexlab", "sp80_goal_localization_regression.py"),
            ),
            "seed": 1.0,
        },
        messages=["goal_localization"],
        meta={"backend": "targets+goal_loc"},
    )


def R45(mv, ctx):
    # FENCE — spill is banned; channel records the fence is active
    return _ok(
        "R45_spill_quarantine_fence",
        {
            "fence_active": 1.0,
            "never_train": 1.0,
            "never_solve": 1.0,
            "quarantine_marker": max(
                _mega_exists("codexlab", "contra_head", "sp80_spill_model.QUARANTINED.md"),
                _exists("codexlab", "contra_head", "sp80_spill_model.py"),  # presence = know to fence
            ),
            "seed": 1.0,
        },
        messages=["spill_FENCE"],
        meta={"backend": "quarantine_fence"},
    )


def R46(mv, ctx):
    return _ok(
        "R46_like_control_prior",
        {
            "module": max(
                _exists("codexlab", "contra_verify_like_control_prior.py"),
                _mega_exists("codexlab", "contra_verify_like_control_prior.py"),
            ),
            "seed": 1.0,
        },
        messages=["like_control_prior"],
        meta={"backend": "like_control"},
    )


def R47(mv, ctx):
    return _ok(
        "R47_macroband_ranking",
        {
            "module": max(
                _exists("codexlab", "contra_verify_macroband_object_ranking.py"),
                _mega_exists("codexlab", "contra_verify_macroband_object_ranking.py"),
            ),
            "seed": 1.0,
        },
        messages=["macroband"],
        meta={"backend": "macroband"},
    )


def R48(mv, ctx):
    return _ok(
        "R48_live_suite_runners",
        {
            "suite_runner": max(_exists("codexlab", "arc_suite_runner.py"), _mega_exists("codexlab", "arc_suite_runner.py")),
            "live_suite": max(
                _exists("codexlab", "arc_live_suite_runner.py"),
                _mega_exists("codexlab", "arc_live_suite_runner.py"),
            ),
            "seed": 1.0,
        },
        messages=["suite_runners"],
        meta={"backend": "arc_suite"},
    )


def R49(mv, ctx):
    return _ok(
        "R49_contra_service",
        {
            "service_dir": max(_exists("codexlab", "contra_service"), _mega_exists("codexlab", "contra_service")),
            "seed": 1.0,
        },
        messages=["contra_service"],
        meta={"backend": "contra_service"},
    )


def R50(mv, ctx):
    return _ok(
        "R50_players_primitives",
        {
            "players_dir": max(_exists("codexlab", "players"), _mega_exists("codexlab", "players")),
            "seed": 1.0,
        },
        messages=["players"],
        meta={"backend": "players"},
    )


def R51(mv, ctx):
    return _ok(
        "R51_lewm_bridge",
        {
            "bridge": max(_exists("codexlab", "lewm_bridge"), _mega_exists("codexlab", "lewm_bridge")),
            "seed": 1.0,
        },
        messages=["lewm_bridge"],
        meta={"backend": "lewm_bridge"},
    )


def R52(mv, ctx):
    return _ok(
        "R52_sp80_corpus",
        {
            "rexus_corpus": _exists("corpus", "sp80_liquid_pour"),
            "mega_corpus": _mega_exists("corpus", "sp80_liquid_pour"),
            "seed": 1.0,
        },
        messages=["sp80_corpus"],
        meta={"backend": "corpus"},
    )


def R53(mv, ctx):
    p = _REXUS / "human" / "human_goal_discovery_calculus_2026-05-19" / "NOVEL_SYSTEMS_INVENTORY.md"
    return _ok(
        "R53_novel_systems_inventory",
        {
            "file": 1.0 if p.is_file() else 0.0,
            "seed": 1.0,
        },
        messages=["novel_systems"],
        meta={"backend": "NOVEL_SYSTEMS_INVENTORY"},
    )


def R54(mv, ctx):
    return _ok(
        "R54_oversight",
        {
            "module": max(
                _exists("codexlab", "contra_head", "oversight.py"),
                _mega_exists("codexlab", "contra_head", "oversight.py"),
            ),
            "seed": 1.0,
        },
        messages=["oversight"],
        meta={"backend": "oversight"},
    )


def R55(mv, ctx):
    return _ok(
        "R55_perception",
        {
            "module": max(
                _exists("codexlab", "contra_head", "perception.py"),
                _mega_exists("codexlab", "contra_head", "perception.py"),
            ),
            "vision_ok": 1.0 if _pkt(ctx, "contra_vision").get("status") == "ok" else 0.0,
            "seed": 1.0,
        },
        messages=["perception"],
        meta={"backend": "perception+vision"},
    )


def R56(mv, ctx):
    return _ok(
        "R56_controls",
        {
            "module": max(
                _exists("codexlab", "contra_head", "controls.py"),
                _mega_exists("codexlab", "contra_head", "controls.py"),
            ),
            "seed": 1.0,
        },
        messages=["controls"],
        meta={"backend": "controls"},
    )


def R57(mv, ctx):
    pkt = ctx.get("contra_packet") or {}
    return _ok(
        "R57_debug_summary",
        {
            "debug_present": 1.0 if pkt.get("debug_summary") else 0.0,
            "n_debug_keys": float(len(pkt.get("debug_summary") or {})),
            "seed": 1.0,
        },
        messages=["debug_summary"],
        meta={"backend": "contra_debug_summary"},
    )


def R58(mv, ctx):
    sc = _scores(ctx, "contra_boundary")
    return _ok(
        "R58_boundary_pass",
        {
            "boundary_ok": 1.0 if _pkt(ctx, "contra_boundary").get("status") == "ok" else 0.0,
            "n_queues": float(sc.get("n_items") or sc.get("n_keys") or 0),
            "spec_disk": max(
                _exists("codexlab", "contra_head", "BOUNDARY_PASS_SPEC.md"),
                _mega_exists("codexlab", "contra_head", "BOUNDARY_PASS_SPEC.md"),
            ),
            "seed": 1.0,
        },
        messages=["boundary_pass"],
        meta={"backend": "boundary_queues"},
    )


def R59(mv, ctx):
    return _ok(
        "R59_human_replay_testbed",
        {
            "module": max(
                _exists("codexlab", "contra_human_replay_l1_testbed.py"),
                _mega_exists("codexlab", "contra_human_replay_l1_testbed.py"),
            ),
            "seed": 1.0,
            "eval_tool_not_train_default": 1.0,
        },
        messages=["human_replay_testbed"],
        meta={"backend": "human_replay_l1"},
    )


def R60(mv, ctx):
    pkt = ctx.get("contra_packet") or {}
    return _ok(
        "R60_not_prediction_flags",
        {
            "not_prediction": 1.0 if pkt.get("not_prediction", True) else 0.0,
            "not_control_truth": 1.0 if pkt.get("not_control_truth", True) else 0.0,
            "not_solve_truth": 1.0 if pkt.get("not_solve_truth", True) else 0.0,
            "all_flags_fail_closed": 1.0
            if (
                pkt.get("not_prediction", True)
                and pkt.get("not_control_truth", True)
                and pkt.get("not_solve_truth", True)
                and not pkt.get("control_truth_admitted")
                and not pkt.get("solve_condition_admitted")
            )
            else 0.0,
            "seed": 1.0,
        },
        messages=["truth_flags"],
        meta={"backend": "contra_truth_flags"},
    )


SEED_FNS: dict[str, Callable[..., OperatorResult]] = {
    "R1_admission_fail_closed": R1,
    "R2_paper_faithful_chassis": R2,
    "R3_rexus_action_owner": R3,
    "R4_opred_candidates": R4,
    "R5_state_prober": R5,
    "R6_gram_plus_gate": R6,
    "R7_lprm_process_reward": R7,
    "R8_full_contra_system": R8,
    "R9_config_oracle_view_required": R9,
    "R10_action_state_tags": R10,
    "R11_action_interaction_graph": R11,
    "R12_action_interaction_matrix": R12,
    "R13_portband_fleet_lineup": R13,
    "R14_structural_sense": R14,
    "R15_shape_fit": R15,
    "R16_fill_sense": R16,
    "R17_ghost_memory": R17,
    "R18_crucible_induction": R18,
    "R19_source_edge_outcome": R19,
    "R20_activation_transitions": R20,
    "R21_deltaband_controlled": R21,
    "R22_exploreband": R22,
    "R23_actionmodband": R23,
    "R24_match_calibration": R24,
    "R25_relationship_ladder": R25,
    "R26_episode_memory": R26,
    "R27_tabula_rasa": R27,
    "R28_view_system": R28,
    "R29_action_semantics_ledger": R29,
    "R30_teacher_student": R30,
    "R31_human_goal_calculus": R31,
    "R32_winbank_eval_fence": R32,
    "R33_mechanic_decode_25": R33,
    "R34_mechanic_clusters": R34,
    "R35_sam_gates": R35,
    "R36_paint_control_skill": R36,
    "R37_puffer_rl": R37,
    "R38_cross_game_isolation": R38,
    "R39_angel_report_sop": R39,
    "R40_groundhog_paintflow": R40,
    "R41_sp80_lora_adapter": R41,
    "R42_control_drift": R42,
    "R43_recommendation_optimum": R43,
    "R44_goal_localization": R44,
    "R45_spill_quarantine_fence": R45,
    "R46_like_control_prior": R46,
    "R47_macroband_ranking": R47,
    "R48_live_suite_runners": R48,
    "R49_contra_service": R49,
    "R50_players_primitives": R50,
    "R51_lewm_bridge": R51,
    "R52_sp80_corpus": R52,
    "R53_novel_systems_inventory": R53,
    "R54_oversight": R54,
    "R55_perception": R55,
    "R56_controls": R56,
    "R57_debug_summary": R57,
    "R58_boundary_pass": R58,
    "R59_human_replay_testbed": R59,
    "R60_not_prediction_flags": R60,
}


def run_all_rexus_seeds(map_view: KnownMapView, context: dict[str, Any]) -> dict[str, OperatorResult]:
    out: dict[str, OperatorResult] = {}
    ctx = dict(context)
    # ensure partial includes current packet for R* that read channels
    partial = dict(ctx.get("dream_summary_partial") or {})
    ctx["dream_summary_partial"] = partial
    for name, fn in SEED_FNS.items():
        try:
            res = fn(map_view, ctx)
        except Exception as exc:
            res = _seed(name, {"planted": 1.0}, f"error:{type(exc).__name__}", status="degraded")
        out[name] = res
        partial[name] = {"name": name, "status": res.status, "scores": res.scores, "hypothesis_only": True}
    for cid in REXUS_IDEA_IDS:
        if cid not in out:
            out[cid] = _seed(cid, {"planted": 1.0}, "forced_seed", status="degraded")
    return out
