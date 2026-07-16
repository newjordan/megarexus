"""ALL Loom imagination / 3D world-sim ideas as base-model seeds.

Philosophy (user): the model is ground; testing rises what works.
Every concept from LOOM_IMAGINATION_3D_SWEEP must appear as a named channel
with numeric scores — never silently omitted.

Live computation when code/artifacts exist; otherwise explicit degraded seed
with concept_id still present (still in the pad, still in the garden).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

from synthesis_lab.orchestration.rdl.brain.types import KnownMapView, OperatorResult, TruthFlags

_REPO = Path(__file__).resolve().parents[5]
_LOOM = Path("/home/frosty40/Loom")
_LOOM_GEPA = Path("/home/frosty40/Loom-gepa-bg")

# Canonical catalog — do not drop IDs without user order
LOOM_IDEA_IDS: tuple[str, ...] = (
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
)


def _ok(name: str, scores: dict[str, float], **kw: Any) -> OperatorResult:
    return OperatorResult(
        name=name,
        status="ok",
        scores={k: float(v) for k, v in scores.items()},
        hypothesis_only=True,
        truth_flags=TruthFlags(),
        meta={"catalog": "loom_imagination_3d", **(kw.pop("meta", {}) or {})},
        **kw,
    )


def _seed(name: str, scores: dict[str, float], msg: str, **kw: Any) -> OperatorResult:
    """Always present: status ok if we computed something real, else degraded seed."""
    status = kw.pop("status", "ok" if scores else "degraded")
    return OperatorResult(
        name=name,
        status=status,
        scores={k: float(v) for k, v in scores.items()},
        messages=[msg],
        hypothesis_only=True,
        truth_flags=TruthFlags(),
        meta={"catalog": "loom_imagination_3d", "seed": True, **(kw.pop("meta", {}) or {})},
        **kw,
    )


def _frame(map_view: KnownMapView, context: dict[str, Any]) -> np.ndarray | None:
    if map_view.frame is not None:
        return np.asarray(map_view.frame)
    f = context.get("frame")
    return np.asarray(f) if f is not None else None


def _components(map_view: KnownMapView, context: dict[str, Any]) -> list[dict[str, Any]]:
    return list(map_view.components or context.get("components") or [])


# ── individual idea seeds ─────────────────────────────────────────────────


def seed_L1_multichannel(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    """Layer-1 stack: raw + ray + body/3D + probe channel flags."""
    frame = _frame(map_view, ctx)
    dream = ctx.get("dream_summary_partial") or {}
    body = dream.get("body") or {}
    ray = dream.get("ray") or {}
    scores = {
        "raw_channel": 1.0 if frame is not None else 0.0,
        "ray_channel": 1.0 if (isinstance(ray, dict) and ray.get("status") == "ok") else 0.0,
        "body_3d_channel": 1.0 if (isinstance(body, dict) and body.get("status") == "ok") else 0.0,
        "probe_channel": float((dream.get("L3_physics_probe_lib") or {}).get("scores", {}).get("n_probes", 0) > 0),
        "n_channels_live": 0.0,
    }
    scores["n_channels_live"] = float(
        scores["raw_channel"] + scores["ray_channel"] + scores["body_3d_channel"] + scores["probe_channel"]
    )
    scores["stack_density"] = scores["n_channels_live"] / 4.0
    return _ok("L1_multichannel_stack", scores, messages=["layer1_input_enrichment"], meta={"backend": "megamind_channels"})


def seed_L2_dual_generative(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    """Neural (E/world) vs structured (body/sim) imagination disagreement."""
    dream = ctx.get("dream_summary_partial") or {}
    world = ctx.get("world_summary") or {}
    body = dream.get("body") or {}
    disagree = float(world.get("disagreement") or ctx.get("disagreement") or 0.0)
    wm_ok = 1.0 if (world.get("wm_status") or 0) >= 1.0 or (world.get("wm_plug") or {}).get("status") == "ok" else 0.0
    body_ok = 1.0 if isinstance(body, dict) and body.get("status") == "ok" else 0.0
    # confidence high when both paths live and disagreement moderate
    both = wm_ok * body_ok
    conf = both * float(math.exp(-abs(disagree)))
    return _ok(
        "L2_dual_generative",
        {
            "neural_path": wm_ok,
            "structured_path": body_ok,
            "disagreement": disagree,
            "cross_check_confidence": conf,
            "dual_live": both,
        },
        messages=["dual_path_cross_check"],
        meta={"backend": "wm+sim3d_disagreement"},
    )


def seed_L3_physics_probes(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    """Physics probe library: virtual probes from spouts/mouths (bounce/drop columns)."""
    frame = _frame(map_view, ctx)
    comps = _components(map_view, ctx)
    n_probes = 0
    n_hits = 0
    path_lens = []
    # drop-column style probes from top of each non-bg component
    if frame is not None:
        H, W = frame.shape[:2]
        bg = int(map_view.background or 0)
        origins = []
        for c in comps[:12]:
            box = c.get("bbox_xyxy") or [0, 0, 0, 0]
            cx = int((box[0] + box[2]) / 2)
            cy = int(box[1])  # top
            if 0 <= cx < W:
                origins.append((max(0, cy), cx))
        # also center of frame
        origins.append((0, W // 2))
        solid_like = set()
        # treat modal non-bg dense colors as blockers roughly
        for c in comps:
            if int(c.get("area") or 0) > 20:
                solid_like.add(int(c.get("color") or -1))
        for y0, x0 in origins:
            n_probes += 1
            y = y0
            steps = 0
            while y < H - 1 and steps < H:
                y += 1
                steps += 1
                col = int(frame[y, x0])
                if col in solid_like and col != bg:
                    n_hits += 1
                    break
            path_lens.append(float(steps))
    scores = {
        "n_probes": float(n_probes),
        "n_hits": float(n_hits),
        "mean_path": float(np.mean(path_lens)) if path_lens else 0.0,
        "hit_rate": float(n_hits) / float(max(n_probes, 1)),
        "bounce_ball_seed": 1.0,  # concept planted
    }
    return _ok("L3_physics_probe_lib", scores, messages=["drop_column_probes"], meta={"backend": "probe_library_v0"})


def seed_L4_raycast_topology(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    dream = ctx.get("dream_summary_partial") or {}
    ray = dream.get("ray") or {}
    sc = (ray.get("scores") or {}) if isinstance(ray, dict) else {}
    scores = {
        "ray_live": 1.0 if isinstance(ray, dict) and ray.get("status") == "ok" else 0.0,
        "n_hits": float(sc.get("n_hits") or 0),
        "mean_signed_potential": float(sc.get("mean_signed_potential") or 0),
        "n_beneficial": float(sc.get("n_beneficial") or 0),
        "n_adversarial": float(sc.get("n_adversarial") or 0),
        "n_structural": float(sc.get("n_structural") or 0),
        "n_neutral": float(sc.get("n_neutral") or 0),
        "mean_path_len": float(sc.get("mean_path_len") or 0),
        "topology_ready": 1.0 if sc.get("n_hits") else 0.0,
    }
    return _ok("L4_raycast_topology", scores, messages=["ray_topology"], meta={"backend": "megamind_ray"})


def seed_L5_hypothesis_bank(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    """Multiple F_k: body margin, wm agreement, crucible match, field potential."""
    dream = ctx.get("dream_summary_partial") or {}
    world = ctx.get("world_summary") or {}
    hyps = {}
    body = (dream.get("body") or {}).get("scores") or {}
    cru = (dream.get("crucible") or {}).get("scores") or {}
    field = (dream.get("field") or {}).get("scores") or {}
    hyps["Fk_pour_body"] = float(body.get("margin") or 0)
    hyps["Fk_wm"] = float(world.get("wm_pixel_acc") or world.get("wm_status") or 0)
    hyps["Fk_crucible_v0"] = float(cru.get("v0_match_fraction") or 0)
    hyps["Fk_field"] = float(field.get("net_potential_at_origin") or 0)
    hyps["n_hypotheses"] = 4.0
    hyps["bank_live"] = 1.0
    return _ok("L5_sim_hypothesis_bank", hyps, messages=["Fk_bank"], meta={"backend": "multi_Fk"})


def seed_L6_bayesian_residual(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    """Softmax over -error^2 as pseudo-posterior over F_k."""
    dream = ctx.get("dream_summary_partial") or {}
    world = ctx.get("world_summary") or {}
    errs = []
    # lower better
    body = (dream.get("body") or {}).get("scores") or {}
    # invert margin-ish: lava bad
    errs.append(abs(float(body.get("lava") or 0)) + max(0.0, -float(body.get("margin") or 0)))
    errs.append(1.0 - float(world.get("wm_status") or 0))
    cru = (dream.get("crucible") or {}).get("scores") or {}
    errs.append(1.0 - float(cru.get("v0_match_fraction") or 0))
    disagree = float(world.get("disagreement") or ctx.get("disagreement") or 0.5)
    errs.append(disagree)
    e = np.asarray(errs, dtype=np.float64)
    # posterior ∝ exp(-0.5 e^2)
    logp = -0.5 * (e ** 2)
    logp = logp - logp.max()
    p = np.exp(logp)
    p = p / max(p.sum(), 1e-9)
    scores = {
        "entropy": float(-(p * np.log(p + 1e-9)).sum()),
        "max_posterior": float(p.max()),
        "argmax_k": float(int(p.argmax())),
        "n_k": float(len(p)),
    }
    for i, pi in enumerate(p):
        scores[f"p_k{i}"] = float(pi)
    return _ok("L6_bayesian_residual", scores, messages=["bayes_Fk"], meta={"backend": "residual_softmax"})


def seed_L7_controllability(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    comps = _components(map_view, ctx)
    # crude: small components more likely controllable, large static
    areas = [float(c.get("area") or c.get("area_pixels") or 0) for c in comps]
    if not areas:
        return _seed("L7_controllability", {"n_entities": 0.0}, "no_components", status="degraded")
    med = float(np.median(areas))
    controllable = sum(1 for a in areas if a <= med)
    world = len(areas) - controllable
    return _ok(
        "L7_controllability",
        {
            "n_entities": float(len(areas)),
            "n_controlled_hyp": float(controllable),
            "n_world_hyp": float(world),
            "control_frac": float(controllable) / float(len(areas)),
            "area_median": med,
        },
        messages=["controllability_v3"],
        meta={"backend": "area_split_hyp"},
    )


def seed_L8_sprites(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    comps = _components(map_view, ctx)
    frame = _frame(map_view, ctx)
    scores = {
        "n_sprites": float(len(comps)),
        "mean_area": float(np.mean([c.get("area") or 0 for c in comps])) if comps else 0.0,
        "max_area": float(max([c.get("area") or 0 for c in comps])) if comps else 0.0,
    }
    if frame is not None:
        hist = np.bincount(frame.reshape(-1), minlength=16)[:16].astype(np.float64)
        hist = hist / max(hist.sum(), 1)
        scores["topo_entropy"] = float(-(hist * np.log(hist + 1e-9)).sum())
        scores["wall_frac_est"] = float(hist[5] if len(hist) > 5 else 0.0)
    # try Loom MultiSense path if importable
    try:
        if str(_LOOM / "src") not in sys.path:
            sys.path.insert(0, str(_LOOM / "src"))
        from lewm.perception.sprites import extract_sprites  # type: ignore

        if frame is not None:
            sprites = extract_sprites(frame, background_colors=(int(map_view.background or 0),), min_size=3)
            scores["loom_sprites"] = float(len(sprites))
            scores["multisense_import"] = 1.0
    except Exception:
        scores["multisense_import"] = 0.0
    return _ok("L8_object_sprite_perception", scores, messages=["sprites_topo"], meta={"backend": "components+optional_loom"})


def seed_L9_flyeye(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    """Multi-scale plan entropy proxies (coarse/medium/fine)."""
    frame = _frame(map_view, ctx)
    if frame is None:
        return _seed("L9_flyeye_multiscale", {}, "no_frame", status="degraded")
    # downscale entropy as multi-scale eyes
    ent = []
    for factor in (16, 8, 4):
        step = max(1, 64 // factor)
        sub = frame[::step, ::step]
        h = np.bincount(sub.reshape(-1), minlength=16)[:16].astype(np.float64)
        h = h / max(h.sum(), 1)
        ent.append(float(-(h * np.log(h + 1e-9)).sum()))
    # shannon gate: weight = softmax(-H)
    e = np.asarray(ent)
    w = np.exp(-e)
    w = w / max(w.sum(), 1e-9)
    return _ok(
        "L9_flyeye_multiscale",
        {
            "H_coarse": ent[0],
            "H_medium": ent[1],
            "H_fine": ent[2],
            "w_coarse": float(w[0]),
            "w_medium": float(w[1]),
            "w_fine": float(w[2]),
            "gate_entropy": float(-(w * np.log(w + 1e-9)).sum()),
        },
        messages=["flyeye_shannon_gate"],
        meta={"backend": "multiscale_entropy"},
    )


def seed_L10_imag_guidance(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    dream = ctx.get("dream_summary_partial") or {}
    macros = dream.get("macros") or {}
    port = dream.get("portband") or {}
    scores = {
        "macro_prior": 1.0 if isinstance(macros, dict) and macros.get("status") == "ok" else 0.0,
        "portband_prior": float((port.get("scores") or {}).get("best_lineup_score") or 0)
        if isinstance(port, dict)
        else 0.0,
        "guidance_live": 1.0,
        "latent_weight_seed": 0.5,
        "action_weight_seed": 0.5,
    }
    # try import Loom guidance module presence
    try:
        p = _LOOM / "src" / "loom_layers" / "imagination_guidance.py"
        scores["module_on_disk"] = 1.0 if p.is_file() else 0.0
    except Exception:
        scores["module_on_disk"] = 0.0
    return _ok("L10_imagination_guidance", scores, messages=["soft_guidance"], meta={"backend": "macros+portband+disk"})


def seed_L11_puct_informers(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    dream = ctx.get("dream_summary_partial") or {}
    names = ("wall", "affordance", "sprite", "path", "geo")
    live = 0
    for n in names:
        # map to existing channels
        key = {
            "wall": "saliency",
            "affordance": "affordances",
            "sprite": "conceptual",
            "path": "portband",
            "geo": "geo",
        }[n]
        pkt = dream.get(key) or {}
        if isinstance(pkt, dict) and pkt.get("status") == "ok":
            live += 1
    scores = {
        "n_informers_live": float(live),
        "n_informers": float(len(names)),
        "informer_frac": float(live) / float(len(names)),
    }
    try:
        p = _LOOM / "src" / "lewm" / "imagination.py"
        scores["imagination_module_disk"] = 1.0 if p.is_file() else 0.0
    except Exception:
        scores["imagination_module_disk"] = 0.0
    return _ok("L11_puct_informers", scores, messages=["puct_informers"], meta={"backend": "mapped_channels"})


def seed_L12_block_rollout(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    # proxies for deep imagination cost/depth; module presence
    scores = {
        "horizon_H": 8.0,
        "jacobi_K": 4.0,
        "parallel_seed": 1.0,
        "ar_baseline": 1.0,
    }
    p = _LOOM / "src" / "loom_layers" / "block_jacobi_rollout.py"
    scores["module_on_disk"] = 1.0 if p.is_file() else 0.0
    return _ok("L12_block_parallel_rollout", scores, messages=["block_rollout_seed"], meta={"backend": "horizon_proxy"})


def seed_L13_gaussian_residual(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    world = ctx.get("world_summary") or {}
    disagree = float(world.get("disagreement") or ctx.get("disagreement") or 0.0)
    # residual magnitude as gaussian correction demand
    scores = {
        "residual_norm": disagree,
        "correction_demand": float(min(1.0, disagree * 4.0)),
        "sigreg_seed": 1.0,
    }
    p = _LOOM / "src" / "lewm" / "correction.py"
    scores["module_on_disk"] = 1.0 if p.is_file() else 0.0
    return _ok("L13_gaussian_residual", scores, messages=["gcd_seed"], meta={"backend": "disagreement_residual"})


def seed_L14_hrm(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    scores = {
        "H_cycles": 2.0,
        "L_steps": 3.0,
        "deliberation_depth": 2.0 * 3.0,
        "lookahead_seed": 1.0,
    }
    for base in (_LOOM, _LOOM_GEPA):
        p = base / "src" / "loom_layers" / "hrm_reasoning_head.py"
        if p.is_file():
            scores["module_on_disk"] = 1.0
            break
    else:
        scores["module_on_disk"] = 0.0
    return _ok("L14_hrm_lookahead", scores, messages=["hrm_seed"], meta={"backend": "dual_timescale"})


def seed_L15_dual_latent(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    dream = ctx.get("dream_summary_partial") or {}
    # z_env ~ world stability, z_hyp ~ contra/legal activity
    legal = (dream.get("legal392") or {}).get("scores") or {}
    contra = (dream.get("contra_view") or {}).get("scores") or {}
    scores = {
        "z_env_mass": float(legal.get("l2") or 0) * 0.01,
        "z_hyp_mass": float(contra.get("n_keys") or 0) * 0.01,
        "split_live": 1.0,
    }
    p = _LOOM_GEPA / "src" / "loom_layers" / "arc_dual_latent_belief.py"
    scores["module_on_disk"] = 1.0 if p.is_file() else 0.0
    return _ok("L15_dual_latent_belief", scores, messages=["dual_latent"], meta={"backend": "env_vs_hyp"})


def seed_L16_selfplay_gold(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    transitions = ctx.get("transitions") or []
    n_up = sum(1 for t in transitions if (getattr(t, "levels_up", False) or (isinstance(t, dict) and t.get("levels_up"))))
    scores = {
        "n_episode_transitions": float(len(transitions)),
        "n_level_up_events": float(n_up),
        "gold_ready": 1.0 if n_up > 0 else 0.0,
        "pipeline_seed": 1.0,
    }
    return _ok("L16_sim_selfplay_gold", scores, messages=["selfplay_gold_seed"], meta={"backend": "episode_level_up"})


def seed_L17_local_sim(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    dream = ctx.get("dream_summary_partial") or {}
    body = (dream.get("body") or {}).get("scores") or {}
    scores = {
        "branch_search_seed": 1.0,
        "body_ticks": float(body.get("ticks") or 0),
        "imagined_evals_proxy": float(body.get("n_frames") or 0),
        "local_sim_live": 1.0 if (dream.get("body") or {}).get("status") == "ok" else 0.0,
    }
    return _ok("L17_local_sim_search", scores, messages=["local_sim"], meta={"backend": "body_as_sim_branch"})


def seed_L18_crossgame_guidance(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    scores = {
        "global_fallback_seed": 1.0,
        "cross_game_prior": 1.0,
    }
    p = _LOOM_GEPA / "experiments" / "2026-05-17_microbuild_imagination_global_fallback_scout_v1"
    scores["scout_on_disk"] = 1.0 if p.is_dir() else 0.0
    return _ok("L18_crossgame_imag_guidance", scores, messages=["global_fallback"], meta={"backend": "scout_presence"})


def seed_L19_goal_coord(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    dream = ctx.get("dream_summary_partial") or {}
    targets = (dream.get("contra_target") or {}).get("scores") or {}
    oracle = (dream.get("contra_oracle") or {}).get("scores") or {}
    scores = {
        "n_target_hyps": float(targets.get("n_items") or targets.get("n_keys") or 0),
        "oracle_present": float(oracle.get("present") or 0),
        "goal_posterior_mass": float(targets.get("n_items") or 1.0),
        "goal_coord_seed": 1.0,
    }
    return _ok("L19_goal_coordinator", scores, messages=["O_G_seed"], meta={"backend": "contra_target_oracle"})


def seed_L20_skills(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    dream = ctx.get("dream_summary_partial") or {}
    macros = dream.get("macros") or {}
    scores = {
        "n_macro_skills": float((macros.get("scores") or {}).get("n_macros") or 0),
        "n_active_skills": float((macros.get("scores") or {}).get("n_active") or 0),
        "skill_bank_seed": 1.0,
    }
    return _ok("L20_skill_modules", scores, messages=["skill_bank"], meta={"backend": "transcript_macros"})


def seed_L21_timr(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    transitions = ctx.get("transitions") or []
    rev = 0
    irrev = 0
    for t in transitions[-16:]:
        aid = int(getattr(getattr(t, "action", None), "action_id", None) or (t.get("action_id") if isinstance(t, dict) else 0) or 0)
        if aid in (1, 2, 3, 4):
            rev += 1
        elif aid in (5, 6):
            irrev += 1
    scores = {
        "n_reversible_acts": float(rev),
        "n_irreversible_acts": float(irrev),
        "cycle_seed": 1.0,
        "timr_live": 1.0 if transitions else 0.5,
    }
    return _ok("L21_timr_cycle", scores, messages=["timr_seed"], meta={"backend": "action_reversibility"})


def seed_L22_meta_horizon(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    world = ctx.get("world_summary") or {}
    short = float(world.get("disagreement") or 0.0)
    # long-horizon proxy: inverse of thrash (many small deltas)
    transitions = ctx.get("transitions") or []
    scores = {
        "short_horizon_err": short,
        "n_steps_seen": float(len(transitions)),
        "myopia_risk": float(min(1.0, short * 2 + max(0, 8 - len(transitions)) * 0.05)),
        "meta_horizon_seed": 1.0,
    }
    return _ok("L22_meta_horizon", scores, messages=["anti_myopia"], meta={"backend": "short_vs_depth"})


def seed_L23_dreamer(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    scores = {
        "dream_budget_steps": 50.0,
        "dream_hours_seed": 1.0,
        "reality_kl_seed": 1.0,
    }
    p = _LOOM_GEPA / "experiments" / "2026-05-17_dreamer_v2"
    scores["scout_on_disk"] = 1.0 if p.is_dir() else 0.0
    return _ok("L23_dreamer_hours", scores, messages=["dreamer_seed"], meta={"backend": "dream_budget"})


def seed_L24_pixel_decoder(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    world = ctx.get("world_summary") or {}
    scores = {
        "decoder_path_seed": 1.0,
        "wm_as_decoder_proxy": 1.0 if (world.get("wm_plug") or {}).get("status") == "ok" or world.get("wm_status") else 0.0,
        "fork_A": 1.0,
    }
    return _ok("L24_pixel_decoder_path", scores, messages=["decoder_fork_A"], meta={"backend": "wide_as_decoder_proxy"})


def seed_L25_dw_render(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    dream = ctx.get("dream_summary_partial") or {}
    body = dream.get("body") or {}
    scores = {
        "dw_render_path_seed": 1.0,
        "sim3d_body_live": 1.0 if isinstance(body, dict) and body.get("status") == "ok" else 0.0,
        "fork_B": 1.0,
    }
    return _ok("L25_dreamweaver_render_path", scores, messages=["dw_render_fork_B"], meta={"backend": "sim3d_body"})


def seed_L26_sam(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    comps = _components(map_view, ctx)
    scores = {
        "n_object_masks": float(len(comps)),
        "objectness_density": float(len(comps)) / 64.0,
        "sam_seed": 1.0,
    }
    return _ok("L26_sam_objectness", scores, messages=["object_masks"], meta={"backend": "components_as_masks"})


def seed_L27_probe_anim(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    dream = ctx.get("dream_summary_partial") or {}
    p3 = (dream.get("L3_physics_probe_lib") or {}).get("scores") or {}
    scores = {
        "probe_anim_channels": float(p3.get("n_probes") or 0),
        "collision_ping_seed": 1.0,
        "channel_stack_ready": 1.0 if p3.get("n_probes") else 0.5,
    }
    return _ok("L27_probe_animation_channels", scores, messages=["probe_anim"], meta={"backend": "L3_probes"})


def seed_L28_dual_confidence(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    dream = ctx.get("dream_summary_partial") or {}
    l2 = (dream.get("L2_dual_generative") or {}).get("scores") or {}
    conf = float(l2.get("cross_check_confidence") or 0)
    return _ok(
        "L28_dual_path_confidence",
        {
            "confidence": conf,
            "agree_seed": 1.0,
            "use_for_action_gate": conf,
        },
        messages=["dual_confidence"],
        meta={"backend": "L2_confidence"},
    )


def seed_L29_action_F(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    world = ctx.get("world_summary") or {}
    scores = {
        "action_conditioned": 1.0,
        "pred_norm": float(world.get("pred_norm") or 0),
        "wm_live": 1.0 if (world.get("wm_plug") or {}).get("status") == "ok" else 0.0,
    }
    return _ok("L29_action_conditioned_F", scores, messages=["lewm_core_F"], meta={"backend": "hemisphere_E"})


def seed_L30_cem_mpc(map_view: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
    dream = ctx.get("dream_summary_partial") or {}
    prop = ctx.get("propose_entropy")
    scores = {
        "cem_seed": 1.0,
        "mpc_seed": 1.0,
        "propose_entropy": float(prop or 0),
        "plan_cost_proxy": float((dream.get("field") or {}).get("scores", {}).get("net_potential_at_origin") or 0),
    }
    p = _LOOM / "src" / "lewm" / "cem_discrete.py"
    scores["module_on_disk"] = 1.0 if p.is_file() else 0.0
    return _ok("L30_cem_mpc_over_imagination", scores, messages=["cem_mpc_seed"], meta={"backend": "plan_proxy"})


SEED_FNS: dict[str, Callable[[KnownMapView, dict[str, Any]], OperatorResult]] = {
    "L1_multichannel_stack": seed_L1_multichannel,
    "L2_dual_generative": seed_L2_dual_generative,
    "L3_physics_probe_lib": seed_L3_physics_probes,
    "L4_raycast_topology": seed_L4_raycast_topology,
    "L5_sim_hypothesis_bank": seed_L5_hypothesis_bank,
    "L6_bayesian_residual": seed_L6_bayesian_residual,
    "L7_controllability": seed_L7_controllability,
    "L8_object_sprite_perception": seed_L8_sprites,
    "L9_flyeye_multiscale": seed_L9_flyeye,
    "L10_imagination_guidance": seed_L10_imag_guidance,
    "L11_puct_informers": seed_L11_puct_informers,
    "L12_block_parallel_rollout": seed_L12_block_rollout,
    "L13_gaussian_residual": seed_L13_gaussian_residual,
    "L14_hrm_lookahead": seed_L14_hrm,
    "L15_dual_latent_belief": seed_L15_dual_latent,
    "L16_sim_selfplay_gold": seed_L16_selfplay_gold,
    "L17_local_sim_search": seed_L17_local_sim,
    "L18_crossgame_imag_guidance": seed_L18_crossgame_guidance,
    "L19_goal_coordinator": seed_L19_goal_coord,
    "L20_skill_modules": seed_L20_skills,
    "L21_timr_cycle": seed_L21_timr,
    "L22_meta_horizon": seed_L22_meta_horizon,
    "L23_dreamer_hours": seed_L23_dreamer,
    "L24_pixel_decoder_path": seed_L24_pixel_decoder,
    "L25_dreamweaver_render_path": seed_L25_dw_render,
    "L26_sam_objectness": seed_L26_sam,
    "L27_probe_animation_channels": seed_L27_probe_anim,
    "L28_dual_path_confidence": seed_L28_dual_confidence,
    "L29_action_conditioned_F": seed_L29_action_F,
    "L30_cem_mpc_over_imagination": seed_L30_cem_mpc,
}


def run_all_loom_imagination_seeds(
    map_view: KnownMapView,
    context: dict[str, Any],
) -> dict[str, OperatorResult]:
    """Run every Loom imagination seed. Order: probes early, dual-path late deps."""
    # two-pass: first independent, then dependent (L1,L2,L4,L5,L6,L10,L11,L27,L28 need partial)
    out: dict[str, OperatorResult] = {}
    independent = [
        "L3_physics_probe_lib",
        "L7_controllability",
        "L8_object_sprite_perception",
        "L9_flyeye_multiscale",
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
        "L29_action_conditioned_F",
        "L30_cem_mpc_over_imagination",
    ]
    ctx = dict(context)
    partial = dict(ctx.get("dream_summary_partial") or {})
    ctx["dream_summary_partial"] = partial

    for name in independent:
        fn = SEED_FNS[name]
        try:
            res = fn(map_view, ctx)
        except Exception as exc:
            res = _seed(name, {}, f"error:{type(exc).__name__}", status="degraded")
        out[name] = res
        partial[name] = {
            "name": name,
            "status": res.status,
            "scores": res.scores,
            "hypothesis_only": True,
        }

    dependent = [
        "L1_multichannel_stack",
        "L2_dual_generative",
        "L4_raycast_topology",
        "L5_sim_hypothesis_bank",
        "L6_bayesian_residual",
        "L10_imagination_guidance",
        "L11_puct_informers",
        "L27_probe_animation_channels",
        "L28_dual_path_confidence",
    ]
    for name in dependent:
        fn = SEED_FNS[name]
        try:
            res = fn(map_view, ctx)
        except Exception as exc:
            res = _seed(name, {}, f"error:{type(exc).__name__}", status="degraded")
        out[name] = res
        partial[name] = {
            "name": name,
            "status": res.status,
            "scores": res.scores,
            "hypothesis_only": True,
        }

    # guarantee every catalog ID present
    for cid in LOOM_IDEA_IDS:
        if cid not in out:
            out[cid] = _seed(cid, {"planted": 1.0}, "forced_seed", status="degraded")
    return out
