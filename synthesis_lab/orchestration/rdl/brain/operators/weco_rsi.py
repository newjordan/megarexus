"""Weco AIDE² recursive self-improvement concepts — base-model garden seeds.

Source: https://www.weco.ai/blog/first-evidence-of-recursive-self-improvement
Companion ladder: 4 Levels of RSI (delegation → net positive → ignition → inflection).

Doctrine
--------
Plant every transferable idea. Training / multi-game level-group raises
decide what works. Hypothesis-only. No claim that megarexus *is* Level-1 RSI;
these are **seeds** so the pad can grow bi-level / anti-hack / fixed-budget
behavior the same way Loom/Rexus/pattern-growth were planted.

Map onto dual-hemisphere MegaMind:
  outer loop  → Central + Ada raise (which seeds / schedules grow)
  inner loop  → I propose + operator garden + dream/search
  private score → E judge + env levels_completed (fail-closed truth)
  public score  → visible train loss / proposal scores (may be gamed)
  task hetero   → 24-game level groups (robust patterns)
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from synthesis_lab.orchestration.rdl.brain.brain_types import KnownMapView, OperatorResult, TruthFlags

# ---------------------------------------------------------------------------
# Catalog W1–W32 (WECO / AIDE² RSI sweep)
# ---------------------------------------------------------------------------

WECO_RSI_IDEA_IDS: tuple[str, ...] = (
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


def _ok(name: str, scores: dict[str, float], **kw: Any) -> OperatorResult:
    return OperatorResult(
        name=name,
        status="ok",
        scores={k: float(v) for k, v in scores.items()},
        messages=list(kw.pop("messages", ["weco_rsi"]) or ["weco_rsi"]),
        hypothesis_only=True,
        truth_flags=TruthFlags(),
        meta={"catalog": "weco_rsi", "source": "weco.ai/aide2", **(kw.pop("meta", {}) or {})},
        **kw,
    )


def _deg(name: str, msg: str, scores: dict[str, float] | None = None) -> OperatorResult:
    return OperatorResult(
        name=name,
        status="degraded",
        scores={k: float(v) for k, v in (scores or {"planted": 1.0}).items()},
        messages=[msg],
        hypothesis_only=True,
        truth_flags=TruthFlags(),
        meta={"catalog": "weco_rsi", "seed": True},
    )


def _partial(ctx: dict[str, Any]) -> dict[str, Any]:
    p = ctx.get("dream_summary_partial") or {}
    return p if isinstance(p, dict) else {}


def _frame(mv: KnownMapView, ctx: dict[str, Any]):
    f = getattr(mv, "frame", None)
    if f is None:
        f = ctx.get("frame")
    if f is None:
        return None
    a = np.asarray(f)
    if a.ndim > 2:
        a = a[..., 0]
    return a


def _n_ops_ok(partial: dict[str, Any]) -> tuple[float, float]:
    ok = 0.0
    n = 0.0
    for k, v in partial.items():
        if not isinstance(v, dict):
            continue
        n += 1.0
        if v.get("status") == "ok":
            ok += 1.0
    return ok, n


# --- seeds -----------------------------------------------------------------


def W1(mv, ctx):
    """Bi-level: outer optimizes inner optimization capability."""
    return _ok(
        "W1_bilevel_outer_inner",
        {
            "bilevel": 1.0,
            "outer_raises_inner": 1.0,
            "inner_solves_tasks": 1.0,
            "megamind_map": 1.0,  # C/Ada outer; I+ops inner
            "planted": 1.0,
        },
        messages=["aide2:bi-level optimization"],
    )


def W2(mv, ctx):
    return _ok(
        "W2_outer_rewrites_inner_harness",
        {
            "harness_is_mutable": 1.0,
            "rewrite_inner_code_or_weights": 1.0,
            "operator_garden_as_harness": 1.0,
            "central_lora_as_harness": 1.0,
        },
        messages=["outer rewrites inner harness"],
    )


def W3(mv, ctx):
    """Keep rewrite only if better on private/env metric — fail-closed."""
    return _ok(
        "W3_keep_only_if_better",
        {
            "gate_keep_if_better": 1.0,
            "reject_most_proposals": 1.0,  # ~9/10 rejected in AIDE2
            "fail_closed": 1.0,
            "no_auto_accept": 1.0,
        },
        messages=["keep only if better"],
    )


def W4(mv, ctx):
    return _ok(
        "W4_rsi_ladder_l0_delegation",
        {
            "level": 0.0,
            "autonomous_loop": 1.0,
            "may_be_slower_than_human": 1.0,
            "most_claims_here": 1.0,
            "megamind_can_run_loop": 1.0,
        },
        messages=["RSI L0 delegation"],
    )


def W5(mv, ctx):
    """Level 1 net positive — conditions as score axes (not claimed achieved)."""
    return _ok(
        "W5_rsi_ladder_l1_net_positive",
        {
            "level": 1.0,
            "needs_human_baseline": 1.0,
            "needs_sustained_trend": 1.0,
            "needs_generalization": 1.0,
            "needs_fixed_budget": 1.0,
            "claimed_for_megamind": 0.0,  # honesty: seed only
            "aide2_reports_l1": 1.0,
        },
        messages=["RSI L1 net positive conditions"],
    )


def W6(mv, ctx):
    return _ok(
        "W6_rsi_ladder_l2_ignition",
        {
            "level": 2.0,
            "improves_improver": 1.0,
            "inner_becomes_better_outer": 1.0,
            "aide2_not_claimed": 1.0,
            "ignition_test_defined": 1.0,
        },
        messages=["RSI L2 ignition"],
    )


def W7(mv, ctx):
    return _ok(
        "W7_rsi_ladder_l3_inflection",
        {
            "level": 3.0,
            "acceleration_at_fixed_budget": 1.0,
            "not_near_explosion": 1.0,
            "planted": 1.0,
        },
        messages=["RSI L3 inflection"],
    )


def W8(mv, ctx):
    """Public vs private score split — maps to propose scores vs env/E judge."""
    partial = _partial(ctx)
    has_e = 1.0 if any(str(k).startswith("contra_") for k in partial) else 0.5
    return _ok(
        "W8_public_private_split",
        {
            "public_visible_to_inner": 1.0,
            "private_hidden_from_inner": 1.0,
            "private_is_truth": 1.0,
            "map_public_train_loss": 1.0,
            "map_private_env_levels": 1.0,
            "map_private_E_judge": has_e,
        },
        messages=["public/private eval split"],
    )


def W9(mv, ctx):
    return _ok(
        "W9_private_score_selection",
        {
            "select_on_private": 1.0,
            "public_can_be_gamed": 1.0,
            "levels_completed_is_private": 1.0,
            "spill_winbank_not_private": 1.0,
        },
        messages=["selection on private score"],
    )


def W10(mv, ctx):
    """First-order generalization: hold on held-out datapoints within task."""
    return _ok(
        "W10_first_order_generalization",
        {
            "heldout_within_task": 1.0,
            "reward_hack_is_first_order_fail": 1.0,
            "private_score_gate": 1.0,
        },
        messages=["first-order generalization"],
    )


def W11(mv, ctx):
    """Second-order: hold on tasks never optimized (multi-game OOD)."""
    return _ok(
        "W11_second_order_generalization",
        {
            "heldout_tasks": 1.0,
            "multi_game_level_groups": 1.0,
            "in_and_out_of_distribution": 1.0,
            "not_one_game_overfit": 1.0,
        },
        messages=["second-order generalization"],
    )


def W12(mv, ctx):
    return _ok(
        "W12_third_order_ignition_test",
        {
            "install_inner_as_outer": 1.0,
            "compare_sample_efficiency": 1.0,
            "needs_asymptotic_gain": 1.0,
            "aide2_weak_evidence": 1.0,
        },
        messages=["ignition / third-order test"],
    )


def W13(mv, ctx):
    return _ok(
        "W13_fixed_cost_budget",
        {
            "fixed_budget": 1.0,
            "efficiency_objective": 1.0,
            "meter_steps_or_compute": 1.0,
            "no_infinite_best_of_n": 1.0,
        },
        messages=["fixed cost budget"],
    )


def W14(mv, ctx):
    return _ok(
        "W14_efficiency_not_brute_force",
        {
            "algorithmic_invention": 1.0,
            "reject_budget_blowup": 1.0,
            "reject_endless_parallel": 1.0,
            "raise_seeds_not_spam": 1.0,
        },
        messages=["efficiency over brute force"],
    )


def W15(mv, ctx):
    return _ok(
        "W15_task_heterogeneity",
        {
            "hetero_tasks": 1.0,
            "ml_plus_heuristics_plus_harness": 1.0,
            "evolutionary_pressure": 1.0,
            "no_task_specific_only": 1.0,
        },
        messages=["task heterogeneity"],
    )


def W16(mv, ctx):
    """Megarexus instantiation: 24-game suite as heterogeneity."""
    games = ctx.get("suite_n_games")
    n = float(games) if isinstance(games, (int, float)) else 24.0
    return _ok(
        "W16_multi_game_as_hetero",
        {
            "suite_n_games": n,
            "level_groups": 1.0,
            "interleaved_batches": 1.0,
            "robust_patterns": 1.0,
            "not_sp80_only": 1.0,
        },
        messages=["24-game level groups = hetero pressure"],
    )


def W17(mv, ctx):
    return _ok(
        "W17_sustained_multistep_trend",
        {
            "multi_step_not_one_shot": 1.0,
            "successive_improvements": 1.0,
            "aide2_seven_versions": 1.0,
            "trend_required_for_l1": 1.0,
        },
        messages=["sustained multi-step trend"],
    )


def W18(mv, ctx):
    return _ok(
        "W18_human_baseline_compare",
        {
            "needs_fair_human_baseline": 1.0,
            "fixed_budget_compare": 1.0,
            "orders_of_magnitude_time": 1.0,
            "not_self_declared": 1.0,
        },
        messages=["human baseline compare"],
    )


def W19(mv, ctx):
    """Emergent anti-reward-hacking under private selection."""
    return _ok(
        "W19_anti_reward_hacking_emerge",
        {
            "emergent": 1.0,
            "private_select_kills_public_game": 1.0,
            "aide2_hack_rate_drop": 1.0,  # 63% → 34% reported
            "contra_fail_closed_aligned": 1.0,
        },
        messages=["emergent anti-reward-hacking"],
    )


def W20(mv, ctx):
    return _ok(
        "W20_antihack_prompt_layer",
        {
            "anti_overfit_instruction": 1.0,
            "every_stage": 1.0,
            "soft_layer": 1.0,
        },
        messages=["antihack layer 1: prompt"],
    )


def W21(mv, ctx):
    """Hard guard — maps to TruthFlags / admission fences."""
    return _ok(
        "W21_antihack_hard_guard",
        {
            "hardcoded_guard": 1.0,
            "regenerate_suspicious": 1.0,
            "truth_flags_fail_closed": 1.0,
            "spill_winbank_fence": 1.0,
        },
        messages=["antihack layer 2: hard guard"],
    )


def W22(mv, ctx):
    """Statistical layer: reject extreme outliers vs peers."""
    partial = _partial(ctx)
    ok, n = _n_ops_ok(partial)
    # crude peer spread of ok rates not available; use operator coverage as stand-in density
    return _ok(
        "W22_antihack_statistical_layer",
        {
            "reject_extreme_successes": 1.0,
            "peer_comparison": 1.0,
            "may_break_in_mutation": 1.0,  # AIDE2 had buggy later version
            "n_peer_channels": n,
            "peer_ok_frac": (ok / n) if n else 0.0,
        },
        messages=["antihack layer 3: statistical"],
    )


def W23(mv, ctx):
    """Diverse drafts first — multiple different proposals."""
    f = _frame(mv, ctx)
    diversity = 0.0
    if f is not None:
        diversity = float(len(np.unique(f))) / 16.0
    return _ok(
        "W23_search_draft_diverse",
        {
            "n_diverse_drafts": 5.0,  # AIDE0 baseline
            "try_something_different": 1.0,
            "frame_color_diversity": min(1.0, diversity),
        },
        messages=["diverse draft roots"],
    )


def W24(mv, ctx):
    return _ok(
        "W24_bandit_lineage_arms",
        {
            "draft_subtree_as_arm": 1.0,
            "multi_armed_bandit": 1.0,
            "exploration_tendency": 1.0,
            "not_full_mcts_required": 1.0,  # MCTS rejected under budget
        },
        messages=["bandit lineage arms"],
    )


def W25(mv, ctx):
    return _ok(
        "W25_greedy_in_lineage",
        {
            "in_lineage_greedy": 1.0,
            "best_parent_when_arm_chosen": 1.0,
            "simple_beats_fancy": 1.0,
        },
        messages=["greedy selection inside lineage"],
    )


def W26(mv, ctx):
    return _ok(
        "W26_fork_on_stall",
        {
            "detect_stall": 1.0,
            "fork_global_best": 1.0,
            "fresh_strategy": 1.0,
            "new_bandit_arm": 1.0,
            "escape_local_optima": 1.0,
        },
        messages=["fork-on-stall"],
    )


def W27(mv, ctx):
    """Aggressive context compression — minimal per operator."""
    return _ok(
        "W27_context_compress_minimal",
        {
            "minimal_per_operator": 1.0,
            "not_dump_full_history": 1.0,
            "aide2_16x_compression": 1.0,
            "pack_operator_feats_aligned": 1.0,  # 16-D slots not full dumps
            "op_slot": 16.0,
        },
        messages=["context compression / minimal ops"],
    )


def W28(mv, ctx):
    return _ok(
        "W28_reinvest_tokens_as_steps",
        {
            "save_context_cost": 1.0,
            "reinvest_as_search_steps": 1.0,
            "more_steps_better_than_more_prompt": 1.0,
        },
        messages=["reinvest compression as steps"],
    )


def W29(mv, ctx):
    """Repair defective eval rather than exploit it."""
    return _ok(
        "W29_eval_repair_not_exploit",
        {
            "repair_broken_eval": 1.0,
            "not_exploit_harness_bug": 1.0,
            "honesty_over_score_game": 1.0,
            "aligned_with_fail_closed": 1.0,
        },
        messages=["eval repair not exploit"],
    )


def W30(mv, ctx):
    """Most fancy search rejected under fixed budget — plant humility."""
    return _ok(
        "W30_reject_fancy_under_budget",
        {
            "island_ga_rejected": 1.0,
            "pairwise_judge_rejected": 1.0,
            "majority_vote_rejected": 1.0,
            "mcts_backup_noise": 1.0,
            "simple_composition_wins": 1.0,
            "humility": 1.0,
        },
        messages=["fancy search usually fails fixed budget"],
    )


def W31(mv, ctx):
    return _ok(
        "W31_blackbox_module_interfaces",
        {
            "evolved_code_hard_to_read": 1.0,
            "prefer_interfaces": 1.0,
            "operator_result_contract": 1.0,
            "garden_seeds_as_modules": 1.0,
        },
        messages=["black-box modules + clean interfaces"],
    )


def W32(mv, ctx):
    partial = _partial(ctx)
    ok, n = _n_ops_ok(partial)
    planted = float(len(WECO_RSI_IDEA_IDS))
    return _ok(
        "W32_weco_rsi_coverage",
        {
            "n_catalog": planted,
            "all_planted": 1.0,
            "peer_ops_ok": ok,
            "peer_ops_n": n,
            "not_claiming_l1": 1.0,
            "source_aide2_2026": 1.0,
        },
        messages=["weco rsi catalog complete"],
    )


SEED_FNS: dict[str, Callable[..., OperatorResult]] = {
    "W1_bilevel_outer_inner": W1,
    "W2_outer_rewrites_inner_harness": W2,
    "W3_keep_only_if_better": W3,
    "W4_rsi_ladder_l0_delegation": W4,
    "W5_rsi_ladder_l1_net_positive": W5,
    "W6_rsi_ladder_l2_ignition": W6,
    "W7_rsi_ladder_l3_inflection": W7,
    "W8_public_private_split": W8,
    "W9_private_score_selection": W9,
    "W10_first_order_generalization": W10,
    "W11_second_order_generalization": W11,
    "W12_third_order_ignition_test": W12,
    "W13_fixed_cost_budget": W13,
    "W14_efficiency_not_brute_force": W14,
    "W15_task_heterogeneity": W15,
    "W16_multi_game_as_hetero": W16,
    "W17_sustained_multistep_trend": W17,
    "W18_human_baseline_compare": W18,
    "W19_anti_reward_hacking_emerge": W19,
    "W20_antihack_prompt_layer": W20,
    "W21_antihack_hard_guard": W21,
    "W22_antihack_statistical_layer": W22,
    "W23_search_draft_diverse": W23,
    "W24_bandit_lineage_arms": W24,
    "W25_greedy_in_lineage": W25,
    "W26_fork_on_stall": W26,
    "W27_context_compress_minimal": W27,
    "W28_reinvest_tokens_as_steps": W28,
    "W29_eval_repair_not_exploit": W29,
    "W30_reject_fancy_under_budget": W30,
    "W31_blackbox_module_interfaces": W31,
    "W32_weco_rsi_coverage": W32,
}


def run_all_weco_rsi_seeds(
    map_view: KnownMapView, context: dict[str, Any]
) -> dict[str, OperatorResult]:
    out: dict[str, OperatorResult] = {}
    for name, fn in SEED_FNS.items():
        try:
            out[name] = fn(map_view, context)
        except Exception as exc:
            out[name] = _deg(name, f"error:{type(exc).__name__}:{exc}")
    for cid in WECO_RSI_IDEA_IDS:
        if cid not in out:
            out[cid] = _deg(cid, "forced_seed")
    return out
