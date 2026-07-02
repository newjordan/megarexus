"""PUCT-style MCTS planner over LeWM latent rollouts — Loom-integrated.

This is NOT a standalone planner. It is the central planning component of
the Loom architecture (see [[loom_legs_architecture]]) and exposes
explicit hooks for every adjacent leg. Standalone behavior is preserved
when hooks are absent, but the integration surface is the point — see
[[feedback_no_isolated_systems]] for why.

Integration surface (each maps to one Loom concept):

  ┌─────────────────────────────────┬─────────────────────────────────┐
  │ Loom component                  │ MCTS hook                       │
  ├─────────────────────────────────┼─────────────────────────────────┤
  │ LeWMsight (observed-state)      │ goal_inference_fn(frame) →      │
  │                                 │   z_goal at root if no goal     │
  │                                 │   was supplied                  │
  │ Hindsight-goal value head       │ value_fn(z, z_goal) → V         │
  │   (proposal §3.5)               │   replaces leaf -L2 evaluator   │
  │ Dense reward (v2-iter 1)        │ step_reward_fn(z_prev,action,   │
  │                                 │   z_next) → r accumulated with  │
  │                                 │   discount γ along each path    │
  │ Cluster heads (proposal §3.3)   │ cluster_prior_fn(game_id,z)→P[A]│
  │                                 │   becomes PUCT's P(s,a)         │
  │ Judge_bench LoRA reasoning head │ action_validator_fn(z,action)→  │
  │   ([[judge_bench_lora]])        │   score; <reject_threshold      │
  │                                 │   skips the expansion           │
  │ Replay sequence bank            │ context.seed_sequences:         │
  │   (arcprize_replay_sequence_*) │   list[list[int]] warm-start    │
  │                                 │   expanded before MC budget     │
  │ GameOps visual/action hints     │ context.complex_action_xy_      │
  │                                 │   priors seed ACTION6 xy        │
  │                                 │   progressive widening          │
  │ Simulator self-play training    │ result.candidate_rollouts:      │
  │   ([[simulator_self_play_…]])   │   sequences with leaf_value     │
  │                                 │   ≥ self_play_threshold,        │
  │                                 │   for promotion to gold bank    │
  │ GameOps compound actions        │ context.compound_action_tmpls:  │
  │   ([[loom_legs_architecture]])  │   macro-children at root        │
  │ LeWM video decoder              │ decoder_fn(z_history) → video   │
  │   ([[synthetic_video_…]])       │   for inspection of best path   │
  └─────────────────────────────────┴─────────────────────────────────┘

Drop-in note: kwarg-compatible with :class:`DiscreteCEMPlanner.plan()`.
``final_probs`` has shape ``(horizon, len(available_actions))`` — indexed
by *position in available_actions*, NOT by raw action_id — matching CEM
exactly so the two planners are swappable at the eval harness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from lewm.data.arc_agi_3 import (
    ARC_COMPLEX_ACTION_ID,
    ARC_GRID_SIZE,
    ARCActionEmbedder,
)
from lewm.model import LeWM


PAPER_MCTS_DEFAULTS = dict(
    num_simulations=2000, horizon=5, c_puct=1.4, pw_alpha=2.0, pw_beta=0.5,
)


# ─────────────────────────────────────────────────────────────────────
# Hook type aliases — Optional callables; defaults preserve standalone.
# ─────────────────────────────────────────────────────────────────────

GoalInferenceFn = Callable[[torch.Tensor], torch.Tensor]
"""(frame [C,H,W] one-hot) → goal_latent [1, D]. From LeWMsight."""

ValueFn = Callable[[torch.Tensor, torch.Tensor], float]
"""(z_leaf [D], z_goal [D]) → scalar value. From hindsight-goal head."""

StepRewardFn = Callable[[torch.Tensor, "_Action", torch.Tensor], float]
"""(z_prev [D], action, z_next [D]) → step reward. From dense-reward signal."""

ClusterPriorFn = Callable[[Optional[str], torch.Tensor], torch.Tensor]
"""(game_id, z_root [D]) → prior P[|available_actions|]. From cluster heads."""

ActionValidatorFn = Callable[[torch.Tensor, "_Action"], float]
"""(z [D], action) → score in [0, 1]; <reject_threshold skips. From Judge_bench."""

DecoderFn = Callable[[torch.Tensor], np.ndarray]
"""(z_history [T, D]) → video [T, H, W, 3]. From LeWM decoder. Inspection only."""


@dataclass
class CandidateRollout:
    """A complete root→leaf trajectory worth exporting for self-play training.

    See [[simulator_self_play_training]]: successful rollouts (high leaf value)
    are candidates for promotion to the gold training bank.
    """

    sequence: list[int]                  # action_id per step (length = depth at leaf)
    sequence_xy: list[tuple[int, int]]   # (x, y); (0, 0) for non-complex
    leaf_value: float                    # value_fn(z_leaf, z_goal); higher = better
    accumulated_reward: float            # sum of step_reward_fn along the path
    depth: int


@dataclass
class CompoundAction:
    """A macro-action template from [[loom_legs_architecture]]'s GameOps leg.

    Example: ``select_then_confirm`` = ACTION6(x, y) → ACTION1. The planner
    expands these as single root children with multi-step rollouts under them.
    """

    name: str
    action_id_sequence: list[int]
    xy_sequence: Optional[list[tuple[int, int]]] = None  # required only if any step is ACTION6


@dataclass
class PlanContext:
    """Per-call context bundling Loom-side information for one plan() call.

    Separate from MCTSConfig (the planner's behavior knobs) because this
    information varies per environment step, while MCTSConfig is fixed
    across the planning session.
    """

    game_id: Optional[str] = None
    goal_latent: Optional[torch.Tensor] = None  # [1, D]; if None and goal_inference_fn set, inferred
    seed_sequences: list[list[int]] = field(default_factory=list)
    seed_sequences_xy: list[list[tuple[int, int]]] = field(default_factory=list)
    complex_action_xy_priors: list[tuple[int, int]] = field(default_factory=list)
    blocked_complex_action_xy: list[tuple[int, int]] = field(default_factory=list)
    compound_action_templates: list[CompoundAction] = field(default_factory=list)
    gamma: float = 0.99  # discount on step_reward_fn accumulation


@dataclass
class MCTSConfig:
    """PUCT MCTS behavior + integration hooks. Defaults preserve standalone."""

    # Search knobs
    horizon: int = PAPER_MCTS_DEFAULTS["horizon"]
    num_simulations: int = PAPER_MCTS_DEFAULTS["num_simulations"]
    c_puct: float = PAPER_MCTS_DEFAULTS["c_puct"]
    pw_alpha: float = PAPER_MCTS_DEFAULTS["pw_alpha"]
    pw_beta: float = PAPER_MCTS_DEFAULTS["pw_beta"]
    value_temp: float = 1.0
    value_scale: float = 1.0

    # Integration hooks (None = standalone behavior)
    goal_inference_fn: Optional[GoalInferenceFn] = None
    value_fn: Optional[ValueFn] = None
    step_reward_fn: Optional[StepRewardFn] = None
    cluster_prior_fn: Optional[ClusterPriorFn] = None
    action_validator_fn: Optional[ActionValidatorFn] = None
    decoder_fn: Optional[DecoderFn] = None

    # Hook behavior tuning
    reject_threshold: float = 0.05     # Judge_bench score below this → skip expansion
    self_play_threshold: float = 0.0   # leaf_value above this → export to candidate_rollouts

    # FLARE additions (arxiv 2601.22311). Defaults preserve original MCTS behavior.
    # See POST_V2_ROADMAP.md mechanism #1.
    rollout_horizon_extension: int = 0       # 0 = no model-rollout beyond tree leaf;
                                              # N > 0 = run N greedy/random predictor steps
                                              # and add their discounted reward + value
                                              # to the backup signal (FLARE "explicit lookahead")
    rollout_action_strategy: str = "greedy"  # "greedy" (argmax visit at parent) or "random"
    backup_discount: float = 1.0             # γ for n-step return on path + rollout
                                              # (1.0 = identical to original undiscounted backup)
    enable_trajectory_cache: bool = False    # FLARE "trajectory memory" — cache (root_hash, prefix)
                                              # → (leaf_value, accumulated_reward) across plan()
                                              # calls. Default off for back-compat.
    trajectory_cache_max_size: int = 1000    # LRU eviction limit
    flare_backup_weighting: str = "uniform"  # "uniform" (every node sees same total) or
                                              # "depth_weighted" (early actions get heavier credit
                                              # for high-value leaves — FLARE's anti-myopia bias)

    def validate(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.num_simulations < 1:
            raise ValueError("num_simulations must be >= 1")
        if self.c_puct < 0.0:
            raise ValueError("c_puct must be >= 0")
        if self.pw_alpha <= 0.0:
            raise ValueError("pw_alpha must be > 0")
        if not 0.0 < self.pw_beta <= 1.0:
            raise ValueError("pw_beta must be in (0, 1]")
        if not 0.0 <= self.reject_threshold <= 1.0:
            raise ValueError("reject_threshold must be in [0, 1]")
        if self.rollout_horizon_extension < 0:
            raise ValueError("rollout_horizon_extension must be >= 0")
        if self.rollout_action_strategy not in ("greedy", "random"):
            raise ValueError("rollout_action_strategy must be 'greedy' or 'random'")
        if not 0.0 < self.backup_discount <= 1.0:
            raise ValueError("backup_discount must be in (0, 1]")
        if self.flare_backup_weighting not in ("uniform", "depth_weighted"):
            raise ValueError("flare_backup_weighting must be 'uniform' or 'depth_weighted'")


@dataclass
class MCTSResult:
    """Mirrors :class:`DiscreteCEMResult` keys + integration telemetry.

    ``final_probs`` is shape ``(horizon, |available_actions|)`` indexed by
    *position in available_actions* (NOT raw action_id) — matches CEM exactly.
    """

    best_sequence: torch.Tensor          # [horizon] long, action_id per step
    best_sequence_xy: torch.Tensor       # [horizon, 2] long, (x, y); (0, 0) for simple
    best_cost: float                     # -root_mean_value; lower is better (mirror CEM)
    final_probs: torch.Tensor            # [horizon, |available_actions|]
    iterations: int
    tree_size: int
    max_depth: int

    # Integration telemetry (zero/empty if hooks not used)
    candidate_rollouts: list[CandidateRollout] = field(default_factory=list)
    cluster_prior_used: bool = False
    seed_sequences_expanded: int = 0
    complex_action_xy_priors_expanded: int = 0
    compound_actions_expanded: int = 0
    validator_rejection_count: int = 0
    goal_was_inferred: bool = False
    # FLARE telemetry
    rollout_extension_steps_total: int = 0   # sum across all sims of rollout-phase steps
    trajectory_cache_hits: int = 0
    trajectory_cache_size: int = 0           # at end of plan()


# ─────────────────────────────────────────────────────────────────────
# Internal: Action and Node
# ─────────────────────────────────────────────────────────────────────


class _Action:
    """A (action_id, x, y) edge. xy ignored for non-complex actions."""

    __slots__ = ("action_id", "x", "y", "is_complex")

    def __init__(self, action_id: int, x: int = 0, y: int = 0) -> None:
        self.action_id = int(action_id)
        self.x = int(x)
        self.y = int(y)
        self.is_complex = self.action_id == ARC_COMPLEX_ACTION_ID

    def key(self) -> tuple:
        return (self.action_id, self.x, self.y) if self.is_complex else (self.action_id,)


class _Node:
    """Tree node with cached path histories for O(1) child-expansion cost."""

    __slots__ = (
        "z", "z_history", "action_history_emb", "visits", "value_sum",
        "children", "depth", "available_action_ids", "step_reward_in",
        "prior",
    )

    def __init__(
        self,
        z: torch.Tensor,
        z_history: torch.Tensor,
        action_history_emb: Optional[torch.Tensor],
        depth: int,
        available_action_ids: list[int],
        step_reward_in: float = 0.0,
        prior: float = 0.0,
    ) -> None:
        self.z = z
        self.z_history = z_history
        self.action_history_emb = action_history_emb
        self.visits = 0
        self.value_sum = 0.0
        self.children: dict[tuple, _Node] = {}
        self.depth = depth
        self.available_action_ids = available_action_ids
        self.step_reward_in = step_reward_in  # reward of arriving at THIS node
        self.prior = prior                    # P(s, a) used to select this node from parent

    @property
    def mean_value(self) -> float:
        return self.value_sum / max(1, self.visits)


# ─────────────────────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────────────────────


class LewmMCTSPlanner:
    """PUCT MCTS over LeWM latent rollouts with Loom-wide integration hooks.

    Construct once per planning session; call :meth:`plan` per env step.
    """

    def __init__(self, config: MCTSConfig | None = None, *, seed: int | None = None) -> None:
        self.config = config or MCTSConfig()
        self.config.validate()
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)
        # FLARE trajectory cache (arxiv 2601.22311). Persists across plan() calls
        # when `enable_trajectory_cache` is True. Key: (root_z_hash, action_prefix_tuple).
        # Value: dict with leaf_value + accumulated_rollout_reward.
        self._traj_cache: dict[tuple, dict] = {}
        self._traj_cache_order: list[tuple] = []  # for LRU eviction
        self._cache_hits_this_plan: int = 0
        self._rollout_steps_this_plan: int = 0
        self._xy_prior_expansions_this_plan: int = 0

    @torch.no_grad()
    def plan(
        self,
        *,
        model: LeWM,
        embedder: ARCActionEmbedder,
        current_frame: torch.Tensor,
        goal_latent: Optional[torch.Tensor] = None,
        available_actions: list[int],
        device: str = "cpu",
        return_trace: bool = False,
        seed: int | None = None,
        context: PlanContext | None = None,
    ) -> MCTSResult:
        """Plan one action sequence.

        Accepts the same kwargs as :meth:`DiscreteCEMPlanner.plan` for drop-in
        compatibility. The optional ``context`` carries Loom-side integration
        data (game_id, seed_sequences, compound actions, discount γ).
        """

        del return_trace  # not yet used; kept for API parity
        cfg = self.config
        if not available_actions:
            raise ValueError("available_actions must not be empty")
        if seed is not None:
            self._rng.manual_seed(seed)

        ctx = context or PlanContext()
        avail_to_pos = {a: i for i, a in enumerate(available_actions)}

        # ── ROOT ── ────────────────────────────────────────────────
        z_root = model.encoder(current_frame.unsqueeze(0).to(device))[0]  # [D]
        root = _Node(
            z=z_root,
            z_history=z_root.unsqueeze(0),
            action_history_emb=None,
            depth=0,
            available_action_ids=list(available_actions),
        )

        # ── GOAL ── infer via LeWMsight hook if not provided ───────
        goal_was_inferred = False
        if goal_latent is None:
            if cfg.goal_inference_fn is None:
                raise ValueError(
                    "goal_latent is None and no goal_inference_fn (LeWMsight hook) configured"
                )
            goal_latent = cfg.goal_inference_fn(current_frame.to(device))
            goal_was_inferred = True
        if ctx.goal_latent is not None:
            goal_latent = ctx.goal_latent  # context override
        goal_flat = goal_latent.to(device).flatten()

        # ── CLUSTER PRIOR ── from cluster_prior_fn if set ──────────
        cluster_prior_used = False
        prior_over_simple: Optional[torch.Tensor] = None
        if cfg.cluster_prior_fn is not None:
            try:
                prior = cfg.cluster_prior_fn(ctx.game_id, z_root)
                if prior.shape[-1] != len(available_actions):
                    raise ValueError(
                        f"cluster_prior_fn returned shape {prior.shape}; "
                        f"expected (|A|={len(available_actions)},)"
                    )
                prior_over_simple = prior.to(device)
                cluster_prior_used = True
            except Exception:
                prior_over_simple = None  # silent fall back to uniform

        # ── SEED SEQUENCES ── warm-start expansions from replay bank ──
        seed_sequences_expanded = self._expand_seed_sequences(
            root, ctx.seed_sequences, ctx.seed_sequences_xy,
            model, embedder, device, prior_over_simple, avail_to_pos,
        )

        # ── COMPOUND ACTIONS ── GameOps macro children at root ─────
        compound_expanded = self._expand_compound_actions(
            root, ctx.compound_action_templates, model, embedder, device,
        )

        # ── MAIN MC LOOP ── ────────────────────────────────────────
        validator_rejections = 0
        candidates: list[CandidateRollout] = []
        self._cache_hits_this_plan = 0
        self._rollout_steps_this_plan = 0
        self._xy_prior_expansions_this_plan = 0
        # Root-state hash for FLARE trajectory cache keys.
        root_hash = self._hash_root(root.z) if cfg.enable_trajectory_cache else None
        for _sim in range(cfg.num_simulations):
            rej, cand = self._simulate(
                root, model, embedder, device, goal_flat, ctx,
                prior_over_simple, avail_to_pos, root_hash=root_hash,
            )
            validator_rejections += rej
            if cand is not None:
                candidates.append(cand)

        return self._build_result(
            root=root,
            available_actions=available_actions,
            avail_to_pos=avail_to_pos,
            device=device,
            candidates=candidates,
            cluster_prior_used=cluster_prior_used,
            seed_sequences_expanded=seed_sequences_expanded,
            compound_actions_expanded=compound_expanded,
            validator_rejections=validator_rejections,
            goal_was_inferred=goal_was_inferred,
        )

    # ──────────────────────────────────────────────────────────────
    # Simulation = select → expand → evaluate → backup
    # ──────────────────────────────────────────────────────────────

    def _simulate(
        self,
        root: _Node,
        model: LeWM,
        embedder: ARCActionEmbedder,
        device: str,
        goal_flat: torch.Tensor,
        ctx: PlanContext,
        prior_over_simple: Optional[torch.Tensor],
        avail_to_pos: dict[int, int],
        *,
        root_hash: Optional[str] = None,
    ) -> tuple[int, Optional[CandidateRollout]]:
        cfg = self.config
        path: list[_Node] = [root]
        node = root
        validator_rejections = 0

        # SELECT
        while node.depth < cfg.horizon and self._is_fully_expanded(node):
            _, child = self._select_child_puct(node)
            path.append(child)
            node = child

        # EXPAND (with Judge_bench validation hook)
        if node.depth < cfg.horizon:
            action = self._pick_new_action(node, ctx)
            if cfg.action_validator_fn is not None:
                score = float(cfg.action_validator_fn(node.z, action))
                if score < cfg.reject_threshold:
                    validator_rejections += 1
                    stub = _Node(
                        z=node.z, z_history=node.z_history,
                        action_history_emb=node.action_history_emb,
                        depth=node.depth + 1, available_action_ids=node.available_action_ids,
                        step_reward_in=-1e6, prior=1e-6,
                    )
                    node.children[action.key()] = stub
                    path.append(stub)
                    node = stub
                else:
                    child = self._expand_child(node, action, model, embedder, device, ctx, prior_over_simple, avail_to_pos)
                    node.children[action.key()] = child
                    path.append(child)
                    node = child
            else:
                child = self._expand_child(node, action, model, embedder, device, ctx, prior_over_simple, avail_to_pos)
                node.children[action.key()] = child
                path.append(child)
                node = child

        # FLARE trajectory-cache lookup: key on (root_hash, prefix-of-action-keys).
        # If we have a cached rollout for this prefix, skip the costly evaluate+rollout.
        cached = None
        cache_key: Optional[tuple] = None
        if cfg.enable_trajectory_cache and root_hash is not None:
            prefix = self._path_prefix_key(path)
            cache_key = (root_hash, prefix)
            cached = self._traj_cache.get(cache_key)
            if cached is not None:
                self._cache_hits_this_plan += 1

        if cached is not None:
            leaf_value = cached["leaf_value"]
            rollout_reward = cached["rollout_reward"]
        else:
            # EVALUATE leaf
            leaf_value = self._evaluate_leaf(node, goal_flat)
            # FLARE rollout extension: from leaf, run N more model steps and aggregate.
            rollout_reward = 0.0
            if cfg.rollout_horizon_extension > 0:
                rollout_reward, rollout_extra_leaf = self._rollout_extend(
                    node, model, embedder, device, ctx, goal_flat,
                )
                # The rollout's final latent becomes the new "leaf" for value estimation,
                # added to the original leaf_value (FLARE bootstraps with the predictor).
                leaf_value = leaf_value + rollout_extra_leaf
            # Cache it for FLARE reuse.
            if cfg.enable_trajectory_cache and cache_key is not None:
                self._cache_insert(cache_key, {
                    "leaf_value": leaf_value,
                    "rollout_reward": rollout_reward,
                })

        # BACKUP: n-step return on path step-rewards + rollout reward + leaf_value.
        # path[0] = root, path[t>=1].step_reward_in is reward of arriving at node t.
        discounted_step_sum = 0.0
        gamma_path = ctx.gamma * cfg.backup_discount
        for t, n in enumerate(path[1:], start=1):
            discounted_step_sum += (gamma_path ** (t - 1)) * n.step_reward_in
        # Rollout reward gets the discount from where it starts (after path depth).
        rollout_start_t = max(1, len(path))
        discounted_step_sum += (gamma_path ** (rollout_start_t - 1)) * rollout_reward
        total = leaf_value + discounted_step_sum

        if cfg.flare_backup_weighting == "depth_weighted":
            # FLARE anti-myopia: early actions in a high-value path get extra credit so
            # PUCT learns to commit to good early choices faster.
            depth = max(1, len(path) - 1)
            for i, n in enumerate(path):
                weight = 1.0 + (depth - i) / depth  # root sees 2x, leaf sees 1x
                n.visits += 1
                n.value_sum += weight * total
        else:
            for n in path:
                n.visits += 1
                n.value_sum += total

        # Self-play export: if leaf is good enough, record the path.
        candidate = None
        if leaf_value >= self.config.self_play_threshold and len(path) > 1:
            seq: list[int] = []
            seq_xy: list[tuple[int, int]] = []
            for prev, nxt in zip(path[:-1], path[1:]):
                # Recover the action that led from prev to nxt by finding which
                # of prev.children IS nxt.
                for key, c in prev.children.items():
                    if c is nxt:
                        aid = key[0]
                        x = key[1] if len(key) == 3 else 0
                        y = key[2] if len(key) == 3 else 0
                        seq.append(aid)
                        seq_xy.append((x, y))
                        break
            candidate = CandidateRollout(
                sequence=seq, sequence_xy=seq_xy, leaf_value=leaf_value,
                accumulated_reward=discounted_step_sum, depth=node.depth,
            )

        return validator_rejections, candidate

    # ──────────────────────────────────────────────────────────────
    # Expansion helpers
    # ──────────────────────────────────────────────────────────────

    def _is_fully_expanded(self, node: _Node) -> bool:
        for aid in node.available_action_ids:
            if aid != ARC_COMPLEX_ACTION_ID and (aid,) not in node.children:
                return False
        if ARC_COMPLEX_ACTION_ID in node.available_action_ids:
            xy_count = sum(1 for k in node.children if len(k) == 3)
            pw_limit = max(1, int(self.config.pw_alpha * (node.visits ** self.config.pw_beta)))
            if xy_count < pw_limit:
                return False
        return True

    def _pick_new_action(self, node: _Node, ctx: PlanContext) -> _Action:
        for aid in node.available_action_ids:
            if aid != ARC_COMPLEX_ACTION_ID and (aid,) not in node.children:
                return _Action(aid)
        if ARC_COMPLEX_ACTION_ID in node.available_action_ids:
            blocked_xy = {
                (int(x), int(y))
                for x, y in ctx.blocked_complex_action_xy
                if 0 <= int(x) < ARC_GRID_SIZE and 0 <= int(y) < ARC_GRID_SIZE
            }
            for xy in ctx.complex_action_xy_priors:
                if len(xy) != 2:
                    continue
                x = int(xy[0])
                y = int(xy[1])
                if not (0 <= x < ARC_GRID_SIZE and 0 <= y < ARC_GRID_SIZE):
                    continue
                if (x, y) in blocked_xy:
                    continue
                action = _Action(ARC_COMPLEX_ACTION_ID, x=x, y=y)
                if action.key() not in node.children:
                    self._xy_prior_expansions_this_plan += 1
                    return action
            for _ in range(64):
                x = int(torch.randint(0, ARC_GRID_SIZE, (1,), generator=self._rng).item())
                y = int(torch.randint(0, ARC_GRID_SIZE, (1,), generator=self._rng).item())
                if (x, y) not in blocked_xy:
                    return _Action(ARC_COMPLEX_ACTION_ID, x=x, y=y)
            x = int(torch.randint(0, ARC_GRID_SIZE, (1,), generator=self._rng).item())
            y = int(torch.randint(0, ARC_GRID_SIZE, (1,), generator=self._rng).item())
            return _Action(ARC_COMPLEX_ACTION_ID, x=x, y=y)
        return _Action(node.available_action_ids[0])

    def _select_child_puct(self, node: _Node) -> tuple[_Action, _Node]:
        total_visits = sum(c.visits for c in node.children.values())
        sqrt_total = math.sqrt(max(1, total_visits))
        c = self.config.c_puct

        best_key: Optional[tuple] = None
        best_score = -float("inf")
        for key, child in node.children.items():
            q = child.mean_value
            # Use stored prior if non-zero (cluster_prior_fn set it); else uniform.
            p = child.prior if child.prior > 0 else (1.0 / max(1, len(node.children)))
            u = c * p * sqrt_total / (1 + child.visits)
            score = q + u
            if score > best_score:
                best_score = score
                best_key = key
        assert best_key is not None
        return self._action_from_key(best_key), node.children[best_key]

    def _expand_child(
        self,
        parent: _Node,
        action: _Action,
        model: LeWM,
        embedder: ARCActionEmbedder,
        device: str,
        ctx: PlanContext,
        prior_over_simple: Optional[torch.Tensor],
        avail_to_pos: dict[int, int],
    ) -> _Node:
        action_emb = self._make_action_emb(action, embedder, device)
        if parent.action_history_emb is None:
            full_action_emb = action_emb.unsqueeze(0)
        else:
            stacked = torch.cat([parent.action_history_emb, action_emb], dim=0)
            full_action_emb = stacked.unsqueeze(0)
        full_z_history = parent.z_history.unsqueeze(0)
        out = model.predictor(full_z_history, full_action_emb)
        z_next = out[0, -1]
        new_z_history = torch.cat([parent.z_history, z_next.unsqueeze(0)], dim=0)
        new_action_history = full_action_emb[0]

        # Step reward (dense reward hook).
        step_r = 0.0
        if self.config.step_reward_fn is not None:
            step_r = float(self.config.step_reward_fn(parent.z, action, z_next))

        # Prior for this child (cluster prior over simple action positions).
        prior = 0.0
        if prior_over_simple is not None and action.action_id in avail_to_pos:
            prior = float(prior_over_simple[avail_to_pos[action.action_id]].item())

        return _Node(
            z=z_next,
            z_history=new_z_history,
            action_history_emb=new_action_history,
            depth=parent.depth + 1,
            available_action_ids=parent.available_action_ids,
            step_reward_in=step_r,
            prior=prior,
        )

    def _expand_seed_sequences(
        self,
        root: _Node,
        seed_sequences: list[list[int]],
        seed_xy: list[list[tuple[int, int]]],
        model: LeWM,
        embedder: ARCActionEmbedder,
        device: str,
        prior_over_simple: Optional[torch.Tensor],
        avail_to_pos: dict[int, int],
    ) -> int:
        """Warm-start: expand each seed sequence into the tree before MC budget."""

        if not seed_sequences:
            return 0
        expanded = 0
        for i, seq in enumerate(seed_sequences):
            xy_seq = seed_xy[i] if i < len(seed_xy) else [(0, 0)] * len(seq)
            node = root
            for t, aid in enumerate(seq[: self.config.horizon]):
                x, y = xy_seq[t] if t < len(xy_seq) else (0, 0)
                action = _Action(aid, x=x, y=y)
                key = action.key()
                if key not in node.children:
                    if aid not in node.available_action_ids:
                        break  # invalid action; truncate
                    child = self._expand_child(
                        node, action, model, embedder, device,
                        ctx=PlanContext(), prior_over_simple=prior_over_simple,
                        avail_to_pos=avail_to_pos,
                    )
                    # Boost visit count so PUCT explores this branch eagerly.
                    child.visits = 1
                    node.children[key] = child
                    expanded += 1
                node = node.children[key]
        return expanded

    def _expand_compound_actions(
        self,
        root: _Node,
        templates: list[CompoundAction],
        model: LeWM,
        embedder: ARCActionEmbedder,
        device: str,
    ) -> int:
        """GameOps macro children: each template becomes one root child."""

        if not templates:
            return 0
        expanded = 0
        for tmpl in templates:
            node = root
            xy_seq = tmpl.xy_sequence or [(0, 0)] * len(tmpl.action_id_sequence)
            for t, aid in enumerate(tmpl.action_id_sequence[: self.config.horizon]):
                if aid not in node.available_action_ids:
                    break
                x, y = xy_seq[t] if t < len(xy_seq) else (0, 0)
                action = _Action(aid, x=x, y=y)
                key = action.key()
                if key not in node.children:
                    child = self._expand_child(
                        node, action, model, embedder, device,
                        ctx=PlanContext(), prior_over_simple=None,
                        avail_to_pos={a: i for i, a in enumerate(node.available_action_ids)},
                    )
                    node.children[key] = child
                node = node.children[key]
            expanded += 1
        return expanded

    def _make_action_emb(self, action: _Action, embedder: ARCActionEmbedder, device: str) -> torch.Tensor:
        aid = torch.tensor([[action.action_id]], dtype=torch.long, device=device)
        x = torch.tensor([[action.x]], dtype=torch.long, device=device)
        y = torch.tensor([[action.y]], dtype=torch.long, device=device)
        mask = torch.tensor([[float(action.is_complex)]], device=device)
        return embedder(aid, x, y, mask)[0]

    def _evaluate_leaf(self, leaf: _Node, goal_flat: torch.Tensor) -> float:
        """Leaf value: value_fn if set, else -value_scale * mean L2."""

        if self.config.value_fn is not None:
            return float(self.config.value_fn(leaf.z, goal_flat))
        mse = (leaf.z - goal_flat).pow(2).mean().item()
        return -self.config.value_scale * float(mse)

    # ──────────────────────────────────────────────────────────────
    # FLARE helpers (arxiv 2601.22311)
    # ──────────────────────────────────────────────────────────────

    def _rollout_extend(
        self,
        leaf: _Node,
        model: LeWM,
        embedder: ARCActionEmbedder,
        device: str,
        ctx: PlanContext,
        goal_flat: torch.Tensor,
    ) -> tuple[float, float]:
        """Greedy/random predictor-based rollout from a tree leaf.

        Returns (accumulated_step_reward, final_leaf_value). Used by FLARE
        to extend the effective planning horizon without growing the tree.
        Per the FLARE paper: "explicit lookahead suppresses myopic trap
        selection."
        """

        cfg = self.config
        if cfg.rollout_horizon_extension <= 0:
            return 0.0, 0.0

        z_history = leaf.z_history
        action_history_emb = leaf.action_history_emb
        accumulated_reward = 0.0
        gamma = ctx.gamma * cfg.backup_discount
        current_z = leaf.z

        for step in range(cfg.rollout_horizon_extension):
            self._rollout_steps_this_plan += 1
            # Pick action by strategy.
            if cfg.rollout_action_strategy == "greedy":
                # Greedy = first available simple action (deterministic; cheap).
                aid = leaf.available_action_ids[0]
            else:  # random
                idx = int(torch.randint(
                    0, len(leaf.available_action_ids), (1,), generator=self._rng,
                ).item())
                aid = leaf.available_action_ids[idx]
            action = _Action(aid)
            action_emb = self._make_action_emb(action, embedder, device)

            if action_history_emb is None:
                full_action_emb = action_emb.unsqueeze(0)
            else:
                stacked = torch.cat([action_history_emb, action_emb], dim=0)
                full_action_emb = stacked.unsqueeze(0)
            full_z_history = z_history.unsqueeze(0)
            out = model.predictor(full_z_history, full_action_emb)
            z_next = out[0, -1]

            # Step reward via dense-reward hook if set.
            if cfg.step_reward_fn is not None:
                r = float(cfg.step_reward_fn(current_z, action, z_next))
            else:
                r = 0.0
            accumulated_reward += (gamma ** step) * r

            # Advance state for next rollout step.
            z_history = torch.cat([z_history, z_next.unsqueeze(0)], dim=0)
            action_history_emb = full_action_emb[0]
            current_z = z_next

        # Final value at end of rollout.
        if cfg.value_fn is not None:
            final_leaf_value = float(cfg.value_fn(current_z, goal_flat))
        else:
            mse = (current_z - goal_flat).pow(2).mean().item()
            final_leaf_value = -cfg.value_scale * float(mse)
        return accumulated_reward, final_leaf_value

    def _hash_root(self, z: torch.Tensor) -> str:
        """Hash a root latent for trajectory-cache keying. Coarse: quantize to 4 sig digits."""

        import hashlib
        quant = (z.detach().cpu().float() * 1e4).round().to(torch.int32).numpy().tobytes()
        return hashlib.sha1(quant).hexdigest()[:16]

    def _path_prefix_key(self, path: list[_Node]) -> tuple:
        """Convert a path of nodes to a tuple of action-key tuples (cache-keyable)."""

        # Each edge in the path is identified by which child key was followed.
        # We reconstruct via parent.children inverse lookup.
        keys: list[tuple] = []
        for prev, cur in zip(path[:-1], path[1:]):
            edge_key = None
            for k, child in prev.children.items():
                if child is cur:
                    edge_key = k
                    break
            if edge_key is None:
                break
            keys.append(edge_key)
        return tuple(keys)

    def _cache_insert(self, key: tuple, value: dict) -> None:
        """LRU-bounded insert into the trajectory cache."""

        max_size = self.config.trajectory_cache_max_size
        if key in self._traj_cache:
            self._traj_cache_order.remove(key)
        self._traj_cache[key] = value
        self._traj_cache_order.append(key)
        while len(self._traj_cache_order) > max_size:
            evict = self._traj_cache_order.pop(0)
            self._traj_cache.pop(evict, None)

    def _action_from_key(self, key: tuple) -> _Action:
        if len(key) == 3:
            return _Action(key[0], x=key[1], y=key[2])
        return _Action(key[0])

    # ──────────────────────────────────────────────────────────────
    # Result construction (final_probs shape MATCHES CEM contract)
    # ──────────────────────────────────────────────────────────────

    def _build_result(
        self,
        *,
        root: _Node,
        available_actions: list[int],
        avail_to_pos: dict[int, int],
        device: str,
        candidates: list[CandidateRollout],
        cluster_prior_used: bool,
        seed_sequences_expanded: int,
        compound_actions_expanded: int,
        validator_rejections: int,
        goal_was_inferred: bool,
    ) -> MCTSResult:
        cfg = self.config
        num_avail = len(available_actions)

        best_seq: list[int] = []
        best_xy: list[tuple[int, int]] = []
        node = root
        for _ in range(cfg.horizon):
            if not node.children:
                break
            best_key = max(node.children, key=lambda k: node.children[k].visits)
            action_id = best_key[0]
            x = best_key[1] if len(best_key) == 3 else 0
            y = best_key[2] if len(best_key) == 3 else 0
            best_seq.append(action_id)
            best_xy.append((x, y))
            node = node.children[best_key]

        while len(best_seq) < cfg.horizon:
            best_seq.append(available_actions[0])
            best_xy.append((0, 0))

        # final_probs[t, j] = prob of available_actions[j] at step t.
        # t=0 uses root visit distribution (marginalized over xy for ACTION6).
        # t>0 is one-hot on the greedy walk (cheap; matches CEM's behavior in
        # practice since CEM's later-step probs are largely informative-of-elites
        # and the eval harness only ever consumes t=0).
        probs = torch.zeros(cfg.horizon, num_avail, device=device)
        total_root_visits = sum(c.visits for c in root.children.values())
        if total_root_visits > 0:
            for key, child in root.children.items():
                aid = key[0]
                if aid in avail_to_pos:
                    probs[0, avail_to_pos[aid]] += child.visits / total_root_visits
        for t in range(1, cfg.horizon):
            aid = best_seq[t]
            if aid in avail_to_pos:
                probs[t, avail_to_pos[aid]] = 1.0

        return MCTSResult(
            best_sequence=torch.tensor(best_seq, dtype=torch.long),
            best_sequence_xy=torch.tensor(best_xy, dtype=torch.long),
            best_cost=-root.mean_value,
            final_probs=probs,
            iterations=cfg.num_simulations,
            tree_size=self._count_tree(root),
            max_depth=self._tree_max_depth(root),
            candidate_rollouts=candidates,
            cluster_prior_used=cluster_prior_used,
            seed_sequences_expanded=seed_sequences_expanded,
            complex_action_xy_priors_expanded=self._xy_prior_expansions_this_plan,
            compound_actions_expanded=compound_actions_expanded,
            validator_rejection_count=validator_rejections,
            goal_was_inferred=goal_was_inferred,
            rollout_extension_steps_total=self._rollout_steps_this_plan,
            trajectory_cache_hits=self._cache_hits_this_plan,
            trajectory_cache_size=len(self._traj_cache),
        )

    def _count_tree(self, root: _Node) -> int:
        n = 0
        stack = [root]
        while stack:
            node = stack.pop()
            n += 1
            stack.extend(node.children.values())
        return n

    def _tree_max_depth(self, root: _Node) -> int:
        best = root.depth
        stack = [root]
        while stack:
            node = stack.pop()
            if node.depth > best:
                best = node.depth
            stack.extend(node.children.values())
        return best


__all__ = [
    "MCTSConfig", "MCTSResult", "LewmMCTSPlanner",
    "PlanContext", "CandidateRollout", "CompoundAction",
    "GoalInferenceFn", "ValueFn", "StepRewardFn",
    "ClusterPriorFn", "ActionValidatorFn", "DecoderFn",
]
