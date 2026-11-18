"""Internal hemisphere full stack: map + operators + dream/propose net.

Composes existing InternalHemisphere encoder with optional WP-A/B map_trust
and operators (body/ray/field/conceptual). Degrades to empty operator
summaries when those modules are absent.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .hemispheres import InternalHemisphere
from .brain_types import KnownMapView, OperatorResult

# Full load: physics + legal mine + FULL CONTRA view fanout.
# Center learns which seeds grow. OP_SLOT scalars each.
OP_SLOT = 16
OP_FAMILIES = (
    # physics / map
    "body",
    "ray",
    "field",
    "perception_trust",
    # legal mine
    "conceptual",
    "portband",
    "macros",
    "symmetry",
    "geo",
    "saliency",
    "affordances",
    "statecon",
    "delta",
    "ghost",
    "crucible",
    "explore",
    "legal392",
    # FULL contra_head.run_contra_view fanout (not stand-ins)
    "contra_view",
    "contra_vision",
    "contra_control",
    "contra_surfaces",
    "contra_symmetry",
    "contra_actor",
    "contra_target",
    "contra_support",
    "contra_statecon",
    "contra_geoband",
    "contra_boundary",
    "contra_ladder",
    "contra_delta_gate",
    "contra_explore",
    "contra_oracle",
    "contra_ghost",
    "contra_episode",
    "contra_semantics",
    "contra_admission",
    "contra_portband",
    # ALL Loom imagination / 3D world-sim ideas (LOOM_IMAGINATION_3D_SWEEP)
    "L1_multichannel_stack",
    "L2_dual_generative",
    "L3_physics_probe_lib",
    "L4_raycast_topology",
    "L5_sim_hypothesis_bank",
    "L6_bayesian_residual",
    "L7_controllability",
    "L8_object_sprite_perception",
    "L9_flyeye_multiscale",
    "L10_imagination_guidance",
    "L11_puct_informers",
    "L12_block_parallel_rollout",
    "L13_gaussian_residual",
    "L14_hrm_lookahead",
    "L15_dual_latent_belief",
    "L16_sim_selfplay_gold",
    "L17_local_sim_search",
    "L18_crossgame_imag_guidance",
    "L19_goal_coordinator",
    "L20_skill_modules",
    "L21_timr_cycle",
    "L22_meta_horizon",
    "L23_dreamer_hours",
    "L24_pixel_decoder_path",
    "L25_dreamweaver_render_path",
    "L26_sam_objectness",
    "L27_probe_animation_channels",
    "L28_dual_path_confidence",
    "L29_action_conditioned_F",
    "L30_cem_mpc_over_imagination",
    # ALL Rexus-folder ideas (REXUS_CONCEPT_SWEEP) — plant everything; training rises
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
    # Basic primitive game designs (Puffer Ocean) — design atoms, not Atari ports
    "P1_paddle_ball_deflect",
    "P2_projectile_shooter",
    "P3_lane_timing_dodge",
    "P4_racing_pass_dodge",
    "P5_maze_pathfind",
    "P6_push_block_sokoban",
    "P7_growth_trail_snake",
    "P8_chase_avoid_grid",
    "P9_balance_dynamics",
    "P10_continuous_vehicle",
    "P11_flip_toggle_puzzle",
    "P12_gravity_align_connect",
    "P13_falling_pack",
    "P14_slide_merge",
    "P15_perfect_info_board",
    "P16_memory_match",
    "P17_multiagent_forage",
    "P18_coordination_cook",
    "P19_warehouse_logistics",
    "P20_survival_craft",
    "P21_platform_jump",
    "P22_flock_boids",
    "P23_laser_mirror_puzzle",
    "P24_combat_arena",
    "P25_abstract_mdp_chain",
    "P26_reaction_whack",
    "P27_terrain_terraform",
    "P28_shared_pool_commons",
    "P29_domain_paintflow",
    "P30_scaffold_template",
    "P31_ocean_library_present",
    "P32_primitive_coverage",
    # Dynamic multi-variate pattern growth (geometry expanded; not SP80-bound)
    "G1_multichannel_variables",
    "G2_density_field",
    "G3_connectivity_growth",
    "G4_expansion_frontier",
    "G5_accretion_erosion",
    "G6_branching_topology",
    "G7_merge_split_dynamics",
    "G8_multiscale_pyramid",
    "G9_anisotropy_orientation",
    "G10_periodicity_lattice",
    "G11_symmetry_multiaxis",
    "G12_multivariate_potential",
    "G13_reaction_diffusion_proxy",
    "G14_cellular_rule_affinity",
    "G15_trail_growth_persistence",
    "G16_packing_compaction",
    "G17_void_cavities",
    "G18_topological_holes",
    "G19_aspect_ratio_spectrum",
    "G20_color_adjacency_graph",
    "G21_local_rewrite_rules",
    "G22_joint_state_vector",
    "G23_growth_rate_estimate",
    "G24_front_curvature",
    "G25_blob_competition",
    "G26_hierarchy_nesting",
    "G27_lattice_vs_organic",
    "G28_mass_conservation",
    "G29_irreversible_growth",
    "G30_pattern_phase",
    "G31_variable_coupling",
    "G32_spatial_frequency",
    "G33_geodesic_distance_field",
    "G34_support_scaffold",
    "G35_transform_invariance",
    "G36_generative_grammar_fit",
    "G37_multiobjective_conflict",
    "G38_not_sp80_bound",
    "G39_frame_size_agnostic",
    "G40_pattern_growth_coverage",
    # Weco AIDE² recursive self-improvement (bi-level / RSI ladder / anti-hack)
    "W1_bilevel_outer_inner",
    "W2_outer_rewrites_inner_harness",
    "W3_keep_only_if_better",
    "W4_rsi_ladder_l0_delegation",
    "W5_rsi_ladder_l1_net_positive",
    "W6_rsi_ladder_l2_ignition",
    "W7_rsi_ladder_l3_inflection",
    "W8_public_private_split",
    "W9_private_score_selection",
    "W10_first_order_generalization",
    "W11_second_order_generalization",
    "W12_third_order_ignition_test",
    "W13_fixed_cost_budget",
    "W14_efficiency_not_brute_force",
    "W15_task_heterogeneity",
    "W16_multi_game_as_hetero",
    "W17_sustained_multistep_trend",
    "W18_human_baseline_compare",
    "W19_anti_reward_hacking_emerge",
    "W20_antihack_prompt_layer",
    "W21_antihack_hard_guard",
    "W22_antihack_statistical_layer",
    "W23_search_draft_diverse",
    "W24_bandit_lineage_arms",
    "W25_greedy_in_lineage",
    "W26_fork_on_stall",
    "W27_context_compress_minimal",
    "W28_reinvest_tokens_as_steps",
    "W29_eval_repair_not_exploit",
    "W30_reject_fancy_under_budget",
    "W31_blackbox_module_interfaces",
    "W32_weco_rsi_coverage",
)
EXTRA_FEAT_DIM = OP_SLOT * len(OP_FAMILIES)


def _as_frame_tensor(frame: Any, device: torch.device) -> torch.Tensor:
    """Normalize frame to [1,H,W] long color indices."""
    if isinstance(frame, torch.Tensor):
        t = frame
    else:
        t = torch.as_tensor(frame)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    elif t.dim() == 3 and t.shape[0] not in (1,) and t.shape[-1] in (1, 3):
        t = t[..., 0]
        if t.dim() == 2:
            t = t.unsqueeze(0)
    elif t.dim() == 3 and t.shape[0] > 1 and t.shape[0] not in (1,):
        # already [B,H,W]
        pass
    t = t.long()
    return t.to(device)


def _empty_operator_packet() -> dict[str, Any]:
    return {
        fam: {
            "name": fam,
            "scores": {},
            "status": "skipped",
            "hypothesis_only": True,
        }
        for fam in OP_FAMILIES
    }


def _result_to_dict(result: OperatorResult | dict[str, Any], fam: str) -> dict[str, Any]:
    if isinstance(result, OperatorResult):
        return {
            "name": result.name or fam,
            "scores": {
                k: (float(v) if isinstance(v, (int, float)) else bool(v) if isinstance(v, bool) else v)
                for k, v in (result.scores or {}).items()
                if isinstance(v, (int, float, bool, str)) or v is None
            },
            "status": result.status,
            "hypothesis_only": bool(getattr(result, "hypothesis_only", True)),
            "messages": list(result.messages or [])[:8],
            # keep compact proposal count only (JSON-serializable scalars)
            "n_proposals": len(result.proposals or []),
            "summary_keys": list((result.summary or {}).keys())[:16],
        }
    if isinstance(result, dict):
        scores = result.get("scores") or {}
        return {
            "name": str(result.get("name", fam)),
            "scores": {
                k: (float(v) if isinstance(v, (int, float)) else bool(v) if isinstance(v, bool) else v)
                for k, v in scores.items()
                if isinstance(v, (int, float, bool, str)) or v is None
            },
            "status": str(result.get("status", "ok")),
            "hypothesis_only": bool(result.get("hypothesis_only", True)),
            "messages": list(result.get("messages") or [])[:8],
            "n_proposals": len(result.get("proposals") or []),
        }
    return {"name": fam, "scores": {}, "status": "degraded", "hypothesis_only": True}


def pack_operator_feats(
    dream_summary: dict[str, Any],
    dim: int = EXTRA_FEAT_DIM,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Flatten operator scores into a fixed-dim padded vector [1, dim]."""
    device = device or torch.device("cpu")
    vals: list[float] = []
    for fam in OP_FAMILIES:
        packet = dream_summary.get(fam) or {}
        if isinstance(packet, OperatorResult):
            slot = packet.scalar_list(OP_SLOT)
        elif isinstance(packet, dict):
            scores = packet.get("scores") or {}
            slot: list[float] = []
            for v in scores.values():
                if isinstance(v, bool):
                    slot.append(1.0 if v else 0.0)
                elif isinstance(v, (int, float)):
                    slot.append(float(v))
            if len(slot) < OP_SLOT:
                slot = slot + [0.0] * (OP_SLOT - len(slot))
            slot = slot[:OP_SLOT]
        else:
            slot = [0.0] * OP_SLOT
        vals.extend(slot)
    if len(vals) < dim:
        vals = vals + [0.0] * (dim - len(vals))
    vals = vals[:dim]
    return torch.tensor(vals, dtype=torch.float32, device=device).unsqueeze(0)


def _frame_numpy(frame: Any):
    import numpy as np

    if isinstance(frame, torch.Tensor):
        t = frame.detach().cpu()
        if t.dim() == 3:
            t = t[0]
        return t.numpy()
    return np.asarray(frame)


def _try_map_trust(frame: Any) -> KnownMapView:
    """Call map_trust.frame_to_map if WP-B present; else fail-closed view."""
    try:
        from .map_trust import frame_to_map

        view = frame_to_map(_frame_numpy(frame))
        if isinstance(view, KnownMapView):
            return view
    except Exception as exc:
        return KnownMapView(
            frame=_frame_numpy(frame) if frame is not None else None,
            trusted=False,
            gate_failures=[f"map_trust:{type(exc).__name__}:{exc}"],
            status="degraded",
        )
    return KnownMapView(
        frame=_frame_numpy(frame) if frame is not None else None,
        trusted=False,
        gate_failures=["map_trust:missing"],
        status="degraded",
    )


def _try_operators(map_view: KnownMapView, context: dict[str, Any]) -> dict[str, Any]:
    """Run full operator garden. CONTRA runs once and fans out all layers."""
    packet = _empty_operator_packet()
    import importlib

    # 1) FULL CONTRA view first — real run_contra_view, fanout every layer
    try:
        from .operators.contra_full import ContraFullOp

        cop = ContraFullOp()
        primary = cop.run(map_view, context)
        packet["contra_view"] = _result_to_dict(primary, "contra_view")
        if cop.last_packet is not None:
            context["contra_packet"] = cop.last_packet
        for name, res in (cop.last_fanout or {}).items():
            packet[name] = _result_to_dict(res, name)
    except Exception as exc:
        packet["contra_view"] = {
            "name": "contra_view",
            "scores": {},
            "status": "degraded",
            "hypothesis_only": True,
            "messages": [f"contra_full:{type(exc).__name__}:{exc}"],
        }

    # 2) Physics + legal + full crucible/delta (not proxies)
    op_specs: list[tuple[str, list[tuple[str, str]]]] = [
        ("body", [("operators", "BodyOp"), ("operators.body", "BodyOp")]),
        ("ray", [("operators", "RayOp"), ("operators.ray_ops", "RayOp")]),
        ("field", [("operators", "FieldOp"), ("operators.field_ops", "FieldOp")]),
        (
            "conceptual",
            [("operators", "ConceptualOp"), ("operators.conceptual", "ConceptualOp")],
        ),
        (
            "portband",
            [("operators", "PortBandOp"), ("operators.portband_op", "PortBandOp")],
        ),
        ("symmetry", [("operators.thick_ops", "SymmetryOp")]),
        ("geo", [("operators.thick_ops", "GeoOp")]),
        ("saliency", [("operators.thick_ops", "SaliencyOp")]),
        ("affordances", [("operators.thick_ops", "AffordanceOp")]),
        ("statecon", [("operators.thick_ops", "StateConOp")]),
        ("delta", [("operators.deltaband_full", "DeltaBandFullOp")]),
        ("ghost", [("operators.thick_ops", "GhostOp")]),
        ("crucible", [("operators.crucible_full", "CrucibleFullOp")]),
        ("explore", [("operators.thick_ops", "ExploreOp")]),
        ("legal392", [("operators.thick_ops", "LegalFullmineOp")]),
        ("perception_trust", [("operators.thick_ops", "PerceptionTrustOp")]),
    ]

    for fam, paths in op_specs:
        if fam in packet and packet[fam].get("status") == "ok" and fam.startswith("contra_"):
            continue
        op = None
        for mod_rel, cls_name in paths:
            try:
                mod = importlib.import_module(f".{mod_rel}", package=__package__)
                cls = getattr(mod, cls_name, None)
                if cls is None:
                    continue
                op = cls()
                break
            except Exception:
                continue
        if op is None or not hasattr(op, "run"):
            continue
        try:
            result = op.run(map_view, context)
            packet[fam] = _result_to_dict(result, fam)
        except Exception as exc:
            packet[fam] = {
                "name": fam,
                "scores": {},
                "status": "degraded",
                "hypothesis_only": True,
                "messages": [f"{type(exc).__name__}:{exc}"],
            }

    # 3) ALL Loom imagination / 3D world-sim ideas (must every ID appear)
    try:
        from .operators.loom_imagination import LOOM_IDEA_IDS, run_all_loom_imagination_seeds

        # world_summary may be filled by caller later; pass empty if absent
        context = dict(context)
        context["dream_summary_partial"] = {
            k: v for k, v in packet.items() if isinstance(v, dict)
        }
        loom_out = run_all_loom_imagination_seeds(map_view, context)
        for name, res in loom_out.items():
            packet[name] = _result_to_dict(res, name)
        # assert catalog completeness into packet
        for cid in LOOM_IDEA_IDS:
            if cid not in packet:
                packet[cid] = {
                    "name": cid,
                    "scores": {"planted": 1.0},
                    "status": "degraded",
                    "hypothesis_only": True,
                    "messages": ["forced_catalog_seed"],
                }
    except Exception as exc:
        for cid in (
            "L1_multichannel_stack",
            "L2_dual_generative",
            "L3_physics_probe_lib",
            "L4_raycast_topology",
            "L5_sim_hypothesis_bank",
            "L6_bayesian_residual",
            "L7_controllability",
            "L8_object_sprite_perception",
            "L9_flyeye_multiscale",
            "L10_imagination_guidance",
            "L11_puct_informers",
            "L12_block_parallel_rollout",
            "L13_gaussian_residual",
            "L14_hrm_lookahead",
            "L15_dual_latent_belief",
            "L16_sim_selfplay_gold",
            "L17_local_sim_search",
            "L18_crossgame_imag_guidance",
            "L19_goal_coordinator",
            "L20_skill_modules",
            "L21_timr_cycle",
            "L22_meta_horizon",
            "L23_dreamer_hours",
            "L24_pixel_decoder_path",
            "L25_dreamweaver_render_path",
            "L26_sam_objectness",
            "L27_probe_animation_channels",
            "L28_dual_path_confidence",
            "L29_action_conditioned_F",
            "L30_cem_mpc_over_imagination",
        ):
            packet[cid] = {
                "name": cid,
                "scores": {},
                "status": "degraded",
                "hypothesis_only": True,
                "messages": [f"loom_pack:{type(exc).__name__}"],
            }

    # 4) ALL Rexus-folder ideas (must every R1–R60 appear)
    # contra_packet already set from ContraFullOp when available — do not overwrite
    try:
        from .operators.rexus_concepts import REXUS_IDEA_IDS, run_all_rexus_seeds

        context = dict(context)
        context["dream_summary_partial"] = {
            k: v for k, v in packet.items() if isinstance(v, dict)
        }
        rexus_out = run_all_rexus_seeds(map_view, context)
        for name, res in rexus_out.items():
            packet[name] = _result_to_dict(res, name)
        for cid in REXUS_IDEA_IDS:
            if cid not in packet:
                packet[cid] = {
                    "name": cid,
                    "scores": {"planted": 1.0},
                    "status": "degraded",
                    "hypothesis_only": True,
                    "messages": ["forced_catalog_seed"],
                }
    except Exception as exc:
        for cid in (
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
        ):
            packet[cid] = {
                "name": cid,
                "scores": {"planted": 1.0},
                "status": "degraded",
                "hypothesis_only": True,
                "messages": [f"rexus_pack:{type(exc).__name__}"],
            }

    # 5) Basic primitive game designs (Ocean) — design atoms, not Atari ports
    try:
        from .operators.primitive_game_designs import (
            PRIMITIVE_IDEA_IDS,
            run_all_primitive_design_seeds,
        )

        context = dict(context)
        context["dream_summary_partial"] = {
            k: v for k, v in packet.items() if isinstance(v, dict)
        }
        prim_out = run_all_primitive_design_seeds(map_view, context)
        for name, res in prim_out.items():
            packet[name] = _result_to_dict(res, name)
        for cid in PRIMITIVE_IDEA_IDS:
            if cid not in packet:
                packet[cid] = {
                    "name": cid,
                    "scores": {"planted": 1.0},
                    "status": "degraded",
                    "hypothesis_only": True,
                    "messages": ["forced_catalog_seed"],
                }
    except Exception as exc:
        for cid in (
            "P1_paddle_ball_deflect",
            "P2_projectile_shooter",
            "P3_lane_timing_dodge",
            "P4_racing_pass_dodge",
            "P5_maze_pathfind",
            "P6_push_block_sokoban",
            "P7_growth_trail_snake",
            "P8_chase_avoid_grid",
            "P9_balance_dynamics",
            "P10_continuous_vehicle",
            "P11_flip_toggle_puzzle",
            "P12_gravity_align_connect",
            "P13_falling_pack",
            "P14_slide_merge",
            "P15_perfect_info_board",
            "P16_memory_match",
            "P17_multiagent_forage",
            "P18_coordination_cook",
            "P19_warehouse_logistics",
            "P20_survival_craft",
            "P21_platform_jump",
            "P22_flock_boids",
            "P23_laser_mirror_puzzle",
            "P24_combat_arena",
            "P25_abstract_mdp_chain",
            "P26_reaction_whack",
            "P27_terrain_terraform",
            "P28_shared_pool_commons",
            "P29_domain_paintflow",
            "P30_scaffold_template",
            "P31_ocean_library_present",
            "P32_primitive_coverage",
        ):
            packet[cid] = {
                "name": cid,
                "scores": {"planted": 1.0},
                "status": "degraded",
                "hypothesis_only": True,
                "messages": [f"primitive_pack:{type(exc).__name__}"],
            }

    # 6) Dynamic multi-variate pattern growth (geometry for ANY field, not SP80)
    try:
        from .operators.pattern_growth import (
            PATTERN_GROWTH_IDEA_IDS,
            run_all_pattern_growth_seeds,
        )

        context = dict(context)
        context["dream_summary_partial"] = {
            k: v for k, v in packet.items() if isinstance(v, dict)
        }
        grow_out = run_all_pattern_growth_seeds(map_view, context)
        for name, res in grow_out.items():
            packet[name] = _result_to_dict(res, name)
        for cid in PATTERN_GROWTH_IDEA_IDS:
            if cid not in packet:
                packet[cid] = {
                    "name": cid,
                    "scores": {"planted": 1.0},
                    "status": "degraded",
                    "hypothesis_only": True,
                    "messages": ["forced_catalog_seed"],
                }
    except Exception as exc:
        for cid in (
            "G1_multichannel_variables",
            "G2_density_field",
            "G3_connectivity_growth",
            "G4_expansion_frontier",
            "G5_accretion_erosion",
            "G6_branching_topology",
            "G7_merge_split_dynamics",
            "G8_multiscale_pyramid",
            "G9_anisotropy_orientation",
            "G10_periodicity_lattice",
            "G11_symmetry_multiaxis",
            "G12_multivariate_potential",
            "G13_reaction_diffusion_proxy",
            "G14_cellular_rule_affinity",
            "G15_trail_growth_persistence",
            "G16_packing_compaction",
            "G17_void_cavities",
            "G18_topological_holes",
            "G19_aspect_ratio_spectrum",
            "G20_color_adjacency_graph",
            "G21_local_rewrite_rules",
            "G22_joint_state_vector",
            "G23_growth_rate_estimate",
            "G24_front_curvature",
            "G25_blob_competition",
            "G26_hierarchy_nesting",
            "G27_lattice_vs_organic",
            "G28_mass_conservation",
            "G29_irreversible_growth",
            "G30_pattern_phase",
            "G31_variable_coupling",
            "G32_spatial_frequency",
            "G33_geodesic_distance_field",
            "G34_support_scaffold",
            "G35_transform_invariance",
            "G36_generative_grammar_fit",
            "G37_multiobjective_conflict",
            "G38_not_sp80_bound",
            "G39_frame_size_agnostic",
            "G40_pattern_growth_coverage",
        ):
            packet[cid] = {
                "name": cid,
                "scores": {"planted": 1.0},
                "status": "degraded",
                "hypothesis_only": True,
                "messages": [f"pattern_growth_pack:{type(exc).__name__}"],
            }

    # 7) Weco AIDE² RSI concepts (bi-level outer/inner, ladder, anti-hack)
    try:
        from .operators.weco_rsi import WECO_RSI_IDEA_IDS, run_all_weco_rsi_seeds

        context = dict(context)
        context["dream_summary_partial"] = {
            k: v for k, v in packet.items() if isinstance(v, dict)
        }
        context.setdefault("suite_n_games", 24)
        weco_out = run_all_weco_rsi_seeds(map_view, context)
        for name, res in weco_out.items():
            packet[name] = _result_to_dict(res, name)
        for cid in WECO_RSI_IDEA_IDS:
            if cid not in packet:
                packet[cid] = {
                    "name": cid,
                    "scores": {"planted": 1.0},
                    "status": "degraded",
                    "hypothesis_only": True,
                    "messages": ["forced_catalog_seed"],
                }
    except Exception as exc:
        for cid in (
            "W1_bilevel_outer_inner",
            "W2_outer_rewrites_inner_harness",
            "W3_keep_only_if_better",
            "W4_rsi_ladder_l0_delegation",
            "W5_rsi_ladder_l1_net_positive",
            "W6_rsi_ladder_l2_ignition",
            "W7_rsi_ladder_l3_inflection",
            "W8_public_private_split",
            "W9_private_score_selection",
            "W10_first_order_generalization",
            "W11_second_order_generalization",
            "W12_third_order_ignition_test",
            "W13_fixed_cost_budget",
            "W14_efficiency_not_brute_force",
            "W15_task_heterogeneity",
            "W16_multi_game_as_hetero",
            "W17_sustained_multistep_trend",
            "W18_human_baseline_compare",
            "W19_anti_reward_hacking_emerge",
            "W20_antihack_prompt_layer",
            "W21_antihack_hard_guard",
            "W22_antihack_statistical_layer",
            "W23_search_draft_diverse",
            "W24_bandit_lineage_arms",
            "W25_greedy_in_lineage",
            "W26_fork_on_stall",
            "W27_context_compress_minimal",
            "W28_reinvest_tokens_as_steps",
            "W29_eval_repair_not_exploit",
            "W30_reject_fancy_under_budget",
            "W31_blackbox_module_interfaces",
            "W32_weco_rsi_coverage",
        ):
            packet[cid] = {
                "name": cid,
                "scores": {"planted": 1.0},
                "status": "degraded",
                "hypothesis_only": True,
                "messages": [f"weco_rsi_pack:{type(exc).__name__}"],
            }
    return packet


class InternalHemisphereFull(nn.Module):
    """Full Internal path: map → operators → encode → propose/dream."""

    def __init__(
        self,
        net: InternalHemisphere | None = None,
        emb: int = 128,
        n_actions: int = 8,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.emb = emb
        self.n_actions = n_actions
        self.device = torch.device(device)
        self.net = net or InternalHemisphere(emb=emb, n_actions=n_actions)
        self.net.to(self.device)
        self._frame_prev = None
        self._episode_transitions: list[dict[str, Any]] = []
        self._game_id: str = "megamind"
        self._contra_packet_prev: dict[str, Any] | None = None
        self._last_action_id: int = 0

    def parameters_net(self):
        return self.net.parameters()

    def reset_episode(self, game_id: str | None = None) -> None:
        self._frame_prev = None
        self._episode_transitions = []
        self._contra_packet_prev = None
        self._last_action_id = 0
        if game_id is not None:
            self._game_id = str(game_id)

    def remember_transition(
        self,
        frame_before: Any,
        action_id: int,
        frame_after: Any,
        *,
        levels_before: int = 0,
        levels_after: int = 0,
        frame_changed: bool | None = None,
        levels_up: bool = False,
    ) -> None:
        """Feed episode memory for Ghost/Delta/Explore seeds."""
        import numpy as np

        fb = np.asarray(frame_before)
        fa = np.asarray(frame_after)
        if frame_changed is None:
            frame_changed = bool(np.any(fb != fa))
        self._episode_transitions.append(
            {
                "step": len(self._episode_transitions),
                "frame_before": fb,
                "frame_after": fa,
                "action_id": int(action_id),
                "levels_before": int(levels_before),
                "levels_after": int(levels_after),
                "frame_changed": bool(frame_changed),
                "levels_up": bool(levels_up),
            }
        )
        if len(self._episode_transitions) > 64:
            self._episode_transitions = self._episode_transitions[-64:]
        self._frame_prev = fa
        self._last_action_id = int(action_id)

    def run(
        self,
        frame: Any,
        available_actions: list[int] | None = None,
        action_for_dream: int | None = None,
    ) -> dict[str, Any]:
        """I.run(frame, available_actions) → h_I, plan, dream, operator packet.

        Does not admit any solve/control truth.
        """
        available_actions = list(available_actions or list(range(1, min(7, self.n_actions))))
        frames = _as_frame_tensor(frame, self.device)

        map_view = _try_map_trust(frame)
        context: dict[str, Any] = {
            "available_actions": available_actions,
            "hypothesis_only": True,
            "frame": map_view.frame,
            "components": list(map_view.components),
            "background": int(map_view.background),
            "frame_prev": self._frame_prev,
            "transitions": list(self._episode_transitions),
            "game_id": self._game_id,
            "contra_packet_prev": self._contra_packet_prev,
            "last_action_id": self._last_action_id,
            "action_invocation_id": f"act{self._last_action_id}",
        }
        dream_summary = _try_operators(map_view, context)
        # retain full CONTRA packet for next-step DeltaBand
        if context.get("contra_packet"):
            self._contra_packet_prev = context.get("contra_packet")
        # transcript macros (proposal priors — not labels)
        try:
            from .transcript_macros import apply_macro_bias_to_logits, macro_packet

            dream_summary["macros"] = macro_packet(list(map_view.components or []))
        except Exception as exc:
            dream_summary["macros"] = {
                "name": "macros",
                "status": "degraded",
                "scores": {},
                "messages": [f"macros:{type(exc).__name__}"],
                "hypothesis_only": True,
            }
        dream_summary["trusted"] = bool(map_view.trusted)
        dream_summary["gate_failures"] = list(map_view.gate_failures or [])
        dream_summary["map_status"] = str(map_view.status)
        dream_summary["n_components"] = len(map_view.components or [])

        was_training = self.net.training
        self.net.eval()
        with torch.no_grad():
            h_I = self.net.encode(frames)
            propose = self.net.propose_logits(frames)
            try:
                from .transcript_macros import apply_macro_bias_to_logits

                propose = apply_macro_bias_to_logits(propose, available_actions)
            except Exception:
                pass
            plan = propose.clone()
            if available_actions:
                mask = torch.ones(self.n_actions, dtype=torch.bool, device=self.device)
                for a in available_actions:
                    if 0 <= int(a) < self.n_actions:
                        mask[int(a)] = False
                plan = plan.masked_fill(mask.unsqueeze(0), float("-inf"))

            if action_for_dream is None:
                action_for_dream = int(propose[0].argmax().item())
            act_t = torch.tensor(
                [int(action_for_dream) % self.n_actions], device=self.device
            )
            dream_next = self.net.dream_next(frames, act_t)
        if was_training:
            self.net.train()

        # keep prev frame for next delta seed even without explicit remember
        try:
            import numpy as np

            if self._frame_prev is None:
                self._frame_prev = np.asarray(map_view.frame)
        except Exception:
            pass

        extra = pack_operator_feats(
            dream_summary, dim=EXTRA_FEAT_DIM, device=self.device
        )

        return {
            "h_I": h_I,
            "dream_next": dream_next,
            "propose_logits": propose,
            "plan_candidates": plan,
            "dream_summary": dream_summary,
            "map_view": map_view,
            "extra_feats": extra,
            "action_for_dream": int(action_for_dream),
            "trusted": bool(map_view.trusted),
            "gate_failures": list(map_view.gate_failures or []),
        }

    def forward(self, frames: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Net-only forward for OscillationTrainer compatibility."""
        return self.net(frames, actions)
