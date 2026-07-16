"""GeoBand / DeltaBand / ExploreBand / ActionModBand / StateCon / EpisodeMemory.

Each band collapses to a legal signal class — algorithms mined from
codexlab/contra_head/* into legal_search.algorithms.*.
"""

from __future__ import annotations

import numpy as np

from legal_search.algorithms.components import (
    change_regions,
    connected_components,
    enrich_components,
    frame_delta_count,
)
from legal_search.algorithms.geometry_rel import mine_geo_relations
from legal_search.algorithms.role_induce import (
    induce_click_sensitive_cells,
    induce_movable_colors,
    induce_static_colors,
)
from legal_search.consume.base import ConsumeAdapter
from legal_search.types import Action, Hypothesis, LegalPacket, Proposal, Transition


class GeoBandConsume(ConsumeAdapter):
    system_id = "geoband"
    algorithm_id = "geometry.boundary_queues.v1"
    source_paths = (
        "codexlab/contra_head/geoband.py",
        "legal_search/algorithms/geometry_rel.py",
    )
    stripped_bindings = ("boundary_queue_as_solve_path",)

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        comps = connected_components(frame)
        mined = mine_geo_relations(comps)
        receipt = self.receipt(
            allowed_outputs=["spatial_pair_queue", "bearings", "directional_gaps"],
            banned_outputs=["admitted_route"],
            notes="GeoBand → ordered spatial relations only (bearing/gap/lateral).",
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [
                Hypothesis(
                    "geo_pairs",
                    {
                        "pairs": mined["pairs"],
                        "nearest": mined["nearest"],
                        "n_pairs": mined["n_pairs"],
                        "mean_manhattan": mined["mean_manhattan"],
                        "min_manhattan": mined["min_manhattan"],
                        "n_components": mined["n_components"],
                    },
                    0.5,
                    source_system=self.system_id,
                )
            ],
            [],
            {"n_pairs": mined["n_pairs"], "mean_manhattan": mined["mean_manhattan"]},
            receipt.to_dict(),
        )


class DeltaBandConsume(ConsumeAdapter):
    system_id = "deltaband"
    algorithm_id = "delta.action_result_numeric.v1"
    source_paths = (
        "codexlab/contra_head/deltaband.py",
        "legal_search/algorithms/components.py",
    )
    stripped_bindings = ("delta_as_control_truth",)

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        rows = []
        by_aid: dict[str, list[float]] = {}
        all_regions: list[dict] = []
        for tr in (transitions or [])[-16:]:
            dpx = float(frame_delta_count(tr.frame_before, tr.frame_after))
            ld = float(tr.levels_after - tr.levels_before)
            regions = change_regions(tr.frame_before, tr.frame_after, max_regions=4)
            rows.append(
                {
                    "action": tr.action.to_dict(),
                    "delta_pixels": dpx,
                    "levels_delta": ld,
                    "frame_changed": bool(tr.frame_changed),
                    "n_change_regions": regions["n_regions"],
                    "changed_fraction": regions["changed_fraction"],
                    "top_region": regions["regions"][0] if regions["regions"] else None,
                }
            )
            all_regions.extend(regions["regions"][:2])
            key = str(int(tr.action.action_id))
            by_aid.setdefault(key, []).append(dpx)
        mean_by_aid = {k: float(np.mean(v)) for k, v in by_aid.items() if v}
        all_dpx = [r["delta_pixels"] for r in rows]
        receipt = self.receipt(
            allowed_outputs=["numeric_deltas", "change_regions"],
            banned_outputs=["admitted_control"],
            notes="DeltaBand → measured before/after deltas + change-region geometry.",
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [
                Hypothesis(
                    "deltas",
                    {
                        "rows": rows,
                        "mean_delta": float(np.mean(all_dpx)) if all_dpx else 0.0,
                        "max_delta": float(np.max(all_dpx)) if all_dpx else 0.0,
                        "change_rate": (
                            sum(1 for r in rows if r["frame_changed"]) / max(1, len(rows))
                        ),
                        "mean_delta_by_action": mean_by_aid,
                        "recent_change_regions": all_regions[:8],
                        "n_delta_rows": len(rows),
                    },
                    0.7,
                    source_system=self.system_id,
                )
            ],
            [],
            {"n_delta_rows": len(rows)},
            receipt.to_dict(),
        )


class ExploreBandConsume(ConsumeAdapter):
    system_id = "exploreband"
    algorithm_id = "explore.budgeted_probe_queue.v1"
    source_paths = (
        "codexlab/contra_head/exploreband.py",
        "legal_search/algorithms/components.py",
    )
    stripped_bindings = ("explore_queue_as_solution",)

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        scene = enrich_components(frame)
        comps = scene["components"][:8]
        feats = {f["component_id"]: f for f in scene["component_features"]}
        # spent probes from transitions (ACTION6 clicks)
        spent: set[tuple[int, int]] = set()
        for tr in (transitions or [])[-32:]:
            if tr.action.action_id == 6 and tr.action.x is not None and tr.action.y is not None:
                spent.add((int(tr.action.x) // 4, int(tr.action.y) // 4))
        probes = []
        budget = 6
        for c in comps:
            if len(probes) >= budget:
                break
            x, y = int(round(c["center_xy"][0])), int(round(c["center_xy"][1]))
            bin_key = (x // 4, y // 4)
            if bin_key in spent:
                continue
            f = feats.get(c["component_id"], {})
            # budget priority: small compact objects, not border walls
            score = 1.0 / (1.0 + c["area"] / 100.0)
            score *= 1.0 - 0.5 * float(f.get("border_contact") or 0.0)
            score *= 0.5 + 0.5 * float(f.get("bbox_fill_ratio") or 0.5)
            probes.append(
                Proposal(
                    actions=[Action(6, x=x, y=y, source=self.algorithm_id)],
                    score=float(score),
                    kind="explore_probe_click",
                    evidence={
                        "component_id": c["component_id"],
                        "color": c["color"],
                        "area": c["area"],
                        "stage": "component_center",
                        "spent_bins": len(spent),
                    },
                    algorithm_id=self.algorithm_id,
                )
            )
        probes.sort(key=lambda p: -p.score)
        # stage-1: also probe contact-boundary midpoints if budget left
        for contact in scene["contact_graph"].get("contacts") or []:
            if len(probes) >= budget:
                break
            # weak: re-use first component center as proxy contact probe
            ca = contact.get("component_a")
            match = next((c for c in comps if c["component_id"] == ca), None)
            if not match:
                continue
            x0, y0, x1, y1 = match["bbox_xyxy"]
            x, y = (x0 + x1) // 2, y1  # bottom-edge probe
            probes.append(
                Proposal(
                    actions=[Action(6, x=int(x), y=int(y), source=self.algorithm_id)],
                    score=0.15,
                    kind="explore_probe_contact_edge",
                    evidence={"contact": contact, "stage": "contact_edge"},
                    algorithm_id=self.algorithm_id,
                )
            )
        receipt = self.receipt(
            allowed_outputs=["probe_queue", "budgeted_stages"],
            banned_outputs=["mandatory_route"],
            notes="ExploreBand → budgeted probe clicks; spent bins suppress repeats.",
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [
                Hypothesis(
                    "explore_budget",
                    {
                        "budget": budget,
                        "n_probes": len(probes),
                        "spent_bins": len(spent),
                        "n_components_considered": len(comps),
                    },
                    0.5,
                    source_system=self.system_id,
                )
            ],
            probes[:budget],
            {},
            receipt.to_dict(),
        )


class ActionModBandConsume(ConsumeAdapter):
    system_id = "actionmodband"
    algorithm_id = "modulate.action_preference_from_evidence.v1"
    source_paths = (
        "codexlab/contra_head/actionmodband.py",
        "legal_search/algorithms/role_induce.py",
    )
    stripped_bindings = ("modulation_as_admitted_action_choice",)

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        trs = list(transitions or [])
        movable = induce_movable_colors(trs)
        static = induce_static_colors(trs)
        clicks = induce_click_sensitive_cells(trs)
        # per-action response rates (effect kind proxy)
        response: dict[str, list[float]] = {str(i): [0.0, 0.0] for i in range(1, 7)}
        for tr in trs[-24:]:
            key = str(int(tr.action.action_id))
            if key in response:
                response[key][1] += 1.0
                response[key][0] += 1.0 if tr.frame_changed else 0.0
        effect_rates = {k: (v[0] / v[1] if v[1] else 0.0) for k, v in response.items()}
        receipt = self.receipt(
            allowed_outputs=["preference_hints", "effect_response_rates"],
            banned_outputs=["forced_action"],
            notes="ActionModBand → reweight proposals from observed evidence only.",
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [
                Hypothesis(
                    "movable_color_rates",
                    {"rates": movable, "static_rates": static},
                    0.5,
                    source_system=self.system_id,
                ),
                Hypothesis(
                    "click_sensitive",
                    {"cells": clicks[:12]},
                    0.5,
                    source_system=self.system_id,
                ),
                Hypothesis(
                    "action_response",
                    {"effect_rates": effect_rates, "n_transitions": len(trs)},
                    0.55,
                    source_system=self.system_id,
                ),
            ],
            [],
            {},
            receipt.to_dict(),
        )


class StateConConsume(ConsumeAdapter):
    system_id = "statecon"
    algorithm_id = "state.structured_working_set.v1"
    source_paths = (
        "codexlab/contra_head/statecon.py",
        "legal_search/algorithms/components.py",
        "legal_search/algorithms/geometry_rel.py",
        "legal_search/algorithms/statecon_graph.py",
    )
    stripped_bindings = ("statecon_as_engine_truth",)

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        from legal_search.algorithms.statecon_graph import mine_statecon

        mined = mine_statecon(frame)
        receipt = self.receipt(
            allowed_outputs=[
                "working_set_components",
                "contacts",
                "spatial_links",
                "branch_graph",
                "support_groups",
            ],
            banned_outputs=["engine_sprite_truth"],
            notes="StateCon → working set + branch graph + support groups (hypotheses only).",
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [
                Hypothesis(
                    "working_set",
                    mined,
                    0.6,
                    source_system=self.system_id,
                )
            ],
            [],
            {
                "levels_completed": levels_completed,
                "n_components": mined.get("n_components", 0),
                "n_support_groups": mined.get("n_support_groups", 0),
                "n_graph_nodes": (mined.get("branch_graph") or {}).get("n_nodes", 0),
            },
            receipt.to_dict(),
        )


class EpisodeMemoryConsume(ConsumeAdapter):
    system_id = "episode_memory"
    algorithm_id = "memory.episode_working_state.v1"
    source_paths = ("codexlab/contra_head/episode_memory.py",)
    stripped_bindings = ("episode_memory_as_universal_prior",)

    def __init__(self):
        from legal_search.algorithms.memory import EpisodicMemory

        self.memory = EpisodicMemory()

    def consume(self, frame, *, game_id="", transitions=None, available_actions=None, levels_completed=0):
        # Pure rebuild when a transition window is supplied (mine/feature path).
        if transitions is not None:
            self.memory.wipe()
            if game_id:
                self.memory.bind_game(game_id)
            for tr in transitions:
                self.memory.record_transition(tr)
        elif game_id:
            self.memory.bind_game(game_id)
        snap = self.memory.snapshot()
        receipt = self.receipt(
            allowed_outputs=["this_episode_snapshot"],
            banned_outputs=["cross_game_prior"],
            notes="Episode memory → working state for current game only.",
        )
        return LegalPacket(
            self.system_id,
            self.algorithm_id,
            [
                Hypothesis(
                    "episode_memory",
                    {
                        **snap,
                        "n_window": len(transitions or []),
                        "levels_completed": levels_completed,
                    },
                    0.6,
                    source_system=self.system_id,
                )
            ],
            self.memory.proposals_from_memory(),
            {},
            receipt.to_dict(),
        )
