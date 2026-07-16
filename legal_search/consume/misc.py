"""Remaining systems: crucible, symmetry, vision, CNN, specialists, puffer, etc."""

from __future__ import annotations

import numpy as np

from legal_search.algorithms.components import connected_components
from legal_search.algorithms.role_induce import induce_movable_colors, induce_static_colors
from legal_search.algorithms.saliency import symmetry_score_horizontal, symmetry_score_vertical
from legal_search.consume.base import ConsumeAdapter
from legal_search.types import Hypothesis, LegalPacket, Transition


class SymmetryConsume(ConsumeAdapter):
    system_id = "symmetry"
    algorithm_id = "saliency.symmetry_axes.v1"
    source_paths = (
        "codexlab/contra_head/symmetry.py",
        "legal_search/algorithms/local_symmetry.py",
    )
    stripped_bindings = ("symmetry_lock_as_solve_key",)

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        from legal_search.algorithms.local_symmetry import mine_symmetry

        comps = connected_components(frame)
        mined = mine_symmetry(frame, comps)
        receipt = self.receipt(
            allowed_outputs=["axis_scores", "component_self_symmetry", "mirror_pairs"],
            banned_outputs=["symmetry_as_goal_truth"],
            notes="Symmetry → global + local geometry scores only (not goal key).",
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [
                Hypothesis(
                    "symmetry",
                    {
                        "horizontal": mined["global_horizontal"],
                        "vertical": mined["global_vertical"],
                        "component_self": mined["component_self"],
                        "mirror_pairs": mined["mirror_pairs"],
                        "best_self": mined["best_self"],
                        "best_pair": mined["best_pair"],
                        "n_pairs": mined["n_pairs"],
                        "not_goal_key": True,
                    },
                    0.55,
                    source_system=self.system_id,
                )
            ],
            [],
            {},
            receipt.to_dict(),
        )


class VisionConsume(ConsumeAdapter):
    system_id = "contra_vision"
    algorithm_id = "perception.raw_components.v1"
    source_paths = (
        "codexlab/contra_head/vision.py",
        "legal_search/algorithms/components.py",
        "legal_search/algorithms/shape_traces.py",
    )
    stripped_bindings = ("shape_trace_as_bucket_label",)

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        from legal_search.algorithms.components import enrich_components
        from legal_search.algorithms.shape_traces import mine_shape_traces

        scene = enrich_components(frame)
        traces = mine_shape_traces(frame)
        receipt = self.receipt(
            allowed_outputs=[
                "raw_components",
                "contact_graph",
                "component_features",
                "shape_traces",
            ],
            banned_outputs=["semantic_bucket_labels"],
            notes="Vision → components + contacts + shape-trace geometry (no floor/wall roles).",
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [
                Hypothesis(
                    "raw_components",
                    {
                        "components": scene["components"][:20],
                        "contact_graph": scene["contact_graph"],
                        "component_features": scene["component_features"][:12],
                        "n_components": scene["n_components"],
                        "background": scene["background"],
                        "shape_traces": traces["traces"][:12],
                        "by_geometry_hint": traces["by_geometry_hint"],
                        "thin_line_count": traces["thin_line_count"],
                        "edge_connected_count": traces["edge_connected_count"],
                        "color_groups": traces.get("color_groups") or [],
                        "n_color_groups": traces.get("n_color_groups") or 0,
                        "semantic_labels_assigned": False,
                    },
                    0.75,
                    source_system=self.system_id,
                )
            ],
            [],
            {},
            receipt.to_dict(),
        )


class CrucibleConsume(ConsumeAdapter):
    system_id = "crucible"
    algorithm_id = "induce.transition_mechanics.v1"
    source_paths = (
        "codexlab/contra_head/crucible/",
        "codexlab/contra_head/crucible/forward_v0.py",
        "codexlab/contra_head/crucible/grammar.py",
        "legal_search/algorithms/crucible_mine.py",
    )
    stripped_bindings = ("mechanics_from_game_source", "answer_key_rules")

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        from legal_search.algorithms.crucible_mine import mine_crucible

        mined = mine_crucible(transitions)
        receipt = self.receipt(
            allowed_outputs=[
                "induced_movable",
                "induced_static",
                "v0_discrepancies",
                "rule_candidates",
            ],
            banned_outputs=["source_transcribed_rules"],
            notes="Crucible → v0 forward model + discrepancy ledger + grammar candidates from transitions only.",
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [
                Hypothesis(
                    "induced_roles",
                    mined,
                    0.55 if mined.get("n_transitions") else 0.3,
                    source_system=self.system_id,
                ),
            ],
            [],
            {},
            receipt.to_dict(),
        )


class MegarexusCNNConsume(ConsumeAdapter):
    system_id = "megarexus_cnn"
    algorithm_id = "model.online_or_generic_heads.v1"
    source_paths = (
        "synthesis_lab/experiments/megarexus_cnn/",
        "synthesis_lab/orchestration/megarexus_live_policy.py",
    )
    stripped_bindings = (
        "sp80_only_checkpoint_as_universal_agent",
        "imitation_of_human_routes",
    )

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        receipt = self.receipt(
            allowed_outputs=["optional_change_logits_if_online_or_generic"],
            banned_outputs=["claim_solve_from_sp80_checkpoint_alone"],
            notes=(
                "CNN is legal as architecture + online training. "
                "A frozen SP80-only checkpoint is a specialist, not the general agent."
            ),
            legal=True,
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [
                Hypothesis(
                    "cnn_status",
                    {
                        "role": "optional_function_approximator",
                        "requires": "online_or_cross_game_generic_training",
                        "admitted_as_general_agent": False,
                    },
                    0.3,
                    source_system=self.system_id,
                )
            ],
            [],
            {},
            receipt.to_dict(),
        )


class SpecialistNoteConsume(ConsumeAdapter):
    """Specialist solvers (ft09 glyph, ls20 corridor, ranked DW SP80) demoted."""

    system_id = "specialist_solvers"
    algorithm_id = "research.specialist_not_general.v1"
    source_paths = (
        "synthesis_lab/experiments/other_games/ft09_glyph_solver.py",
        "synthesis_lab/experiments/other_games/ls20_corridor_probe.py",
        "codexlab/arc_play_harness.py::megarexus_dw_directgrid_policy",
    )
    stripped_bindings = (
        "ft09_glyph_as_portfolio_general",
        "ls20_known_routes",
        "sp80_dw_fixture_priors_in_legal_loop",
    )

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        receipt = self.receipt(
            allowed_outputs=["research_reference_only"],
            banned_outputs=["use_as_default_legal_agent"],
            notes=(
                "Specialists may exist for research/scorecards with honest labels. "
                "They are NOT consumed into the legal general loop."
            ),
            legal=True,
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [
                Hypothesis(
                    "specialist_boundary",
                    {
                        "included_in_legal_loop": False,
                        "ok_as": "honest_specialist_research",
                        "game_id_seen": game_id,
                    },
                    1.0,
                    source_system=self.system_id,
                )
            ],
            [],
            {},
            receipt.to_dict(),
        )


class PufferRLConsume(ConsumeAdapter):
    system_id = "pufferlib_rl"
    algorithm_id = "rl.generic_policy_gradient.v1"
    source_paths = (
        "codexlab/pufferlib_envs/",
        "codexlab/sp80_puffer_policy.py",
        "codexlab/paint_control_skill.py",
    )
    stripped_bindings = ("env_shaped_for_sp80_only_as_general",)

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        receipt = self.receipt(
            allowed_outputs=["generic_rl_loop_if_env_agnostic"],
            banned_outputs=["sp80_paintflow_env_as_only_brain"],
            notes="Puffer/RL kept as optional generic trainer, not SP80 answer.",
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [Hypothesis("rl_optional", {"admitted": False}, 0.2, source_system=self.system_id)],
            [],
            {},
            receipt.to_dict(),
        )


class QuarantineFenceConsume(ConsumeAdapter):
    system_id = "quarantine_fence"
    algorithm_id = "guard.banned_lineage.v1"
    source_paths = (
        "codexlab/_invalid/",
        "synthesis_lab/orchestration/_invalid/",
        "codexlab/contra_head/sp80_spill_model.QUARANTINED.md",
        "synthesis_lab/notes/INVALID_DROPLETSIM_ANSWERKEY_20260706.md",
    )
    stripped_bindings = ("ALL — never consume into agent",)

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        receipt = self.receipt(
            allowed_outputs=[],
            banned_outputs=["spill_model", "dropletsim", "human_routes", "source_rules"],
            notes="Fence only: banned systems produce empty legal packets.",
            legal=True,
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [
                Hypothesis(
                    "quarantine",
                    {
                        "banned": [
                            "sp80_spill_model",
                            "dropletsim",
                            "native_scene",
                            "human_mined_routes",
                        ],
                        "agent_may_import": False,
                    },
                    1.0,
                    source_system=self.system_id,
                )
            ],
            [],
            {},
            receipt.to_dict(),
        )


class TabulaRasaConsume(ConsumeAdapter):
    system_id = "tabula_rasa"
    algorithm_id = "reset.blank_slate.v1"
    source_paths = ("codexlab/contra_head/tabula_rasa.py",)
    stripped_bindings = ()

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        receipt = self.receipt(
            allowed_outputs=["reset_signal"],
            banned_outputs=[],
            notes="Tabula rasa → wipe game-specific state (legal).",
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [Hypothesis("blank_slate", {"wipe_game_bindings": True}, 1.0, source_system=self.system_id)],
            [],
            {},
            receipt.to_dict(),
        )


class RelationshipLadderConsume(ConsumeAdapter):
    system_id = "relationship_ladder"
    algorithm_id = "relations.ladder_hypotheses.v1"
    source_paths = (
        "codexlab/contra_head/relationship_ladder.py",
        "legal_search/algorithms/geometry_rel.py",
        "legal_search/algorithms/ports.py",
    )
    stripped_bindings = ("ladder_as_solve_proof",)

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        from legal_search.algorithms.geometry_rel import mine_geo_relations
        from legal_search.algorithms.ports import mine_ports_and_alignments

        comps = connected_components(frame)[:10]
        geo = mine_geo_relations(comps)
        ports = mine_ports_and_alignments(comps, frame_shape=np.asarray(frame).shape[:2])
        # Completeness stages from geometry evidence density — never "win"
        n_pairs = geo["n_pairs"]
        n_align = ports["n_alignments"]
        stages = [
            {"stage": 0, "name": "components_seen", "score": min(1.0, len(comps) / 4.0), "status": "hypothesis"},
            {"stage": 1, "name": "spatial_pairs", "score": min(1.0, n_pairs / 6.0), "status": "hypothesis"},
            {"stage": 2, "name": "port_alignments", "score": min(1.0, n_align / 4.0), "status": "hypothesis"},
            {
                "stage": 3,
                "name": "nearest_links",
                "score": min(1.0, len(geo["nearest"]) / 4.0),
                "status": "hypothesis",
            },
        ]
        ladder = [
            {
                "level": i,
                "component_id": c["component_id"],
                "status": "hypothesis",
                "nearest": next(
                    (n for n in geo["nearest"] if n["component_id"] == c["component_id"]),
                    None,
                ),
            }
            for i, c in enumerate(comps)
        ]
        completeness = float(np.mean([s["score"] for s in stages]))
        receipt = self.receipt(
            allowed_outputs=["relation_ladder_hypotheses", "geometry_completeness_stages"],
            banned_outputs=["ladder_complete_as_win"],
            notes="Relationship ladder → ordered relation hypotheses from geometry density.",
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [
                Hypothesis(
                    "ladder",
                    {
                        "rungs": ladder,
                        "stages": stages,
                        "completeness": completeness,
                        "not_win_proof": True,
                    },
                    0.4,
                    source_system=self.system_id,
                )
            ],
            [],
            {},
            receipt.to_dict(),
        )


class MatchCalibrationConsume(ConsumeAdapter):
    system_id = "match_calibration"
    algorithm_id = "calibrate.role_reweight_only.v1"
    source_paths = (
        "codexlab/contra_head/match_calibration/",
        "legal_search/algorithms/calibrate.py",
    )
    stripped_bindings = (
        "calibration_as_actor_truth",
        "transfer_prior_as_solve_truth",
    )

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        from legal_search.algorithms.calibrate import mine_calibration

        mined = mine_calibration(frame, transitions)
        receipt = self.receipt(
            allowed_outputs=["role_candidate_reweight"],
            banned_outputs=["admitted_actor_truth", "admitted_goal_truth"],
            notes="Match calibration reweights candidates from transitions — never admits truth.",
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [
                Hypothesis(
                    "calibration_boundary",
                    mined,
                    0.6 if mined.get("n_transitions") else 0.4,
                    source_system=self.system_id,
                )
            ],
            [],
            {},
            receipt.to_dict(),
        )


class ControllableAffordancesConsume(ConsumeAdapter):
    system_id = "controller_affordances"
    algorithm_id = "afford.actor_target_support_hypotheses.v1"
    source_paths = (
        "codexlab/contra_head/controller_affordances.py",
        "legal_search/algorithms/affordances.py",
    )
    stripped_bindings = ("affordance_as_admitted_control",)

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        from legal_search.algorithms.affordances import mine_affordance_candidates

        mined = mine_affordance_candidates(frame)
        candidates = []
        for row in mined["actors"][:3]:
            candidates.append({**row, "roles": ["actor_candidate"]})
        for row in mined["targets"][:3]:
            candidates.append({**row, "roles": ["target_candidate"]})
        for row in mined["supports"][:2]:
            candidates.append({**row, "roles": ["support_candidate"]})
        receipt = self.receipt(
            allowed_outputs=["actor_target_support_candidates"],
            banned_outputs=["admitted_actor", "admitted_goal"],
            notes="Affordances stay ranked candidates — never admitted control truth.",
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [
                Hypothesis(
                    "affordances",
                    {
                        "candidates": candidates,
                        "actors": mined["actors"],
                        "targets": mined["targets"],
                        "supports": mined["supports"],
                        "any_admitted": False,
                        "may_admit_truth": False,
                        "n_components_scored": mined["n_components_scored"],
                    },
                    0.4,
                    source_system=self.system_id,
                )
            ],
            [],
            {},
            receipt.to_dict(),
        )


class ActionSemanticsConsume(ConsumeAdapter):
    system_id = "action_semantics_ledger"
    algorithm_id = "semantics.action_effect_tags.v1"
    source_paths = ("codexlab/contra_head/action_semantics_ledger.py",)
    stripped_bindings = ("semantics_as_hardcoded_sp80_verbs",)

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        tags = []
        rates: dict[str, list[float]] = {str(i): [0.0, 0.0] for i in range(1, 7)}
        for tr in (transitions or [])[-12:]:
            tags.append(
                {
                    "action": tr.action.to_dict(),
                    "frame_changed": tr.frame_changed,
                    "levels_up": tr.levels_up,
                    "tag": "changes_frame" if tr.frame_changed else "noop_like",
                }
            )
            key = str(int(tr.action.action_id))
            if key in rates:
                rates[key][1] += 1.0
                rates[key][0] += 1.0 if tr.frame_changed else 0.0
        effect_rates = {
            k: (v[0] / v[1] if v[1] else 0.0) for k, v in rates.items()
        }
        receipt = self.receipt(
            allowed_outputs=["effect_tags_from_observation"],
            banned_outputs=["preloaded_game_verb_map"],
            notes="Action semantics from observed effects only.",
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [
                Hypothesis(
                    "action_tags",
                    {
                        "rows": tags,
                        "effect_rates": effect_rates,
                        "n_tagged": len(tags),
                    },
                    0.55,
                    source_system=self.system_id,
                )
            ],
            [],
            {},
            receipt.to_dict(),
        )
