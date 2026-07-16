"""Basic primitive game designs (Puffer Ocean) as base-model seeds.

These are NOT "Atari ports to train on." They are **minimal game-design
primitives** — the elementary patterns a world/controller brain should know:

  paddle-ball · projectile dodge · lane timing · maze path · growth-trail ·
  balance dynamics · flip-puzzle · gravity-align · packing · merge · board
  perfect-info · memory · multiagent forage · coordination · logistics ·
  survival craft · continuous race · jump · flock · laser-mirror · combat ·
  abstract MDP · reaction · domain paintflow …

Source: `/home/frosty40/rexus/.external/pufferlib-4.0-newjordan/ocean/`
Doctrine: plant every primitive; training rises what works.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from synthesis_lab.orchestration.rdl.brain.types import KnownMapView, OperatorResult, TruthFlags

_OCEAN = Path("/home/frosty40/rexus/.external/pufferlib-4.0-newjordan/ocean")

# Design primitive → realizing envs (canonical names under ocean/)
PRIMITIVE_CATALOG: dict[str, tuple[str, ...]] = {
    "P1_paddle_ball_deflect": ("pong", "breakout", "slimevolley"),
    "P2_projectile_shooter": ("asteroids", "blastar"),
    "P3_lane_timing_dodge": ("freeway",),
    "P4_racing_pass_dodge": ("enduro", "whisker_racer", "drive"),
    "P5_maze_pathfind": ("maze", "tmaze", "tower_climb"),
    "P6_push_block_sokoban": ("boxoban",),
    "P7_growth_trail_snake": ("snake",),
    "P8_chase_avoid_grid": ("pacman",),
    "P9_balance_dynamics": ("cartpole", "double_pendulum"),
    "P10_continuous_vehicle": ("drone", "docking"),
    "P11_flip_toggle_puzzle": ("lightsout",),
    "P12_gravity_align_connect": ("connect4",),
    "P13_falling_pack": ("tetris", "drmario"),
    "P14_slide_merge": ("g2048",),
    "P15_perfect_info_board": ("chess", "checkers", "go", "hex", "tripletriad"),
    "P16_memory_match": ("memory",),
    "P17_multiagent_forage": (
        "battle",
        "convert",
        "convert_circle",
        "scape",
        "target",
        "onlyfish",
        "minimal",
        "squared",
        "squared_continuous",
    ),
    "P18_coordination_cook": ("overcooked",),
    "P19_warehouse_logistics": ("rware", "trash_pickup"),
    "P20_survival_craft": ("craftax", "craftax_classic", "nmmo3", "nethack"),
    "P21_platform_jump": ("dino",),
    "P22_flock_boids": ("boids",),
    "P23_laser_mirror_puzzle": ("laser_puzzle",),
    "P24_combat_arena": ("impulse_wars", "moba", "tactical", "battle"),
    "P25_abstract_mdp_chain": ("chain_mdp", "onestateworld"),
    "P26_reaction_whack": ("whackamole",),
    "P27_terrain_terraform": ("terraform",),
    "P28_shared_pool_commons": ("shared_pool",),
    "P29_domain_paintflow": ("rexus_sp80_paintflow", "rexus_sp80_paintflow_l1_multi"),
    "P30_scaffold_template": ("template", "benchmark", "matsci"),
}

PRIMITIVE_IDEA_IDS: tuple[str, ...] = tuple(PRIMITIVE_CATALOG.keys()) + (
    "P31_ocean_library_present",
    "P32_primitive_coverage",
)


def _ok(name: str, scores: dict[str, float], **kw: Any) -> OperatorResult:
    return OperatorResult(
        name=name,
        status="ok",
        scores={k: float(v) for k, v in scores.items()},
        hypothesis_only=True,
        truth_flags=TruthFlags(),
        meta={"catalog": "primitive_game_design", **(kw.pop("meta", {}) or {})},
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
        meta={"catalog": "primitive_game_design", "seed": True, **(kw.pop("meta", {}) or {})},
        **kw,
    )


def _env_present(name: str) -> float:
    return 1.0 if (_OCEAN / name).is_dir() else 0.0


def _family_scores(envs: tuple[str, ...]) -> dict[str, float]:
    present = [_env_present(e) for e in envs]
    n = float(len(envs)) or 1.0
    n_ok = float(sum(present))
    scores: dict[str, float] = {
        "n_realizers": n,
        "n_present": n_ok,
        "coverage": n_ok / n,
        "ocean_root": 1.0 if _OCEAN.is_dir() else 0.0,
        "is_design_primitive": 1.0,
        "not_atari_port_label": 1.0,
    }
    for e, p in zip(envs, present):
        scores[f"env_{e}"] = p
    return scores


def _make_family_fn(prim_id: str, envs: tuple[str, ...]) -> Callable[..., OperatorResult]:
    def _fn(mv: KnownMapView, ctx: dict[str, Any]) -> OperatorResult:
        scores = _family_scores(envs)
        return _ok(
            prim_id,
            scores,
            messages=[f"primitive:{prim_id}", f"realizers:{','.join(envs)}"],
            meta={"backend": "puffer_ocean", "realizers": list(envs)},
        )

    return _fn


def P31(mv, ctx):
    if not _OCEAN.is_dir():
        return _seed(
            "P31_ocean_library_present",
            {"present": 0.0, "n_envs": 0.0},
            "ocean_missing",
            status="degraded",
        )
    envs = sorted(p.name for p in _OCEAN.iterdir() if p.is_dir())
    return _ok(
        "P31_ocean_library_present",
        {
            "present": 1.0,
            "n_envs": float(len(envs)),
            "has_classic_paddle": float(int("pong" in envs and "breakout" in envs)),
            "has_maze": float(int("maze" in envs)),
            "has_snake": float(int("snake" in envs)),
            "has_balance": float(int("cartpole" in envs)),
            "has_paintflow": float(int("rexus_sp80_paintflow" in envs)),
            "is_design_primitive_lib": 1.0,
        },
        messages=["ocean_lib", f"n={len(envs)}"],
        meta={"backend": "puffer_ocean", "envs_sample": envs[:12]},
    )


def P32(mv, ctx):
    """How many design families have ≥1 realizing env on disk."""
    covered = 0
    total = len(PRIMITIVE_CATALOG)
    for envs in PRIMITIVE_CATALOG.values():
        if any(_env_present(e) > 0 for e in envs):
            covered += 1
    return _ok(
        "P32_primitive_coverage",
        {
            "n_families": float(total),
            "n_covered": float(covered),
            "coverage": float(covered) / float(total or 1),
            "all_planted": 1.0,
        },
        messages=["primitive_coverage"],
        meta={"backend": "puffer_ocean"},
    )


SEED_FNS: dict[str, Callable[..., OperatorResult]] = {
    pid: _make_family_fn(pid, envs) for pid, envs in PRIMITIVE_CATALOG.items()
}
SEED_FNS["P31_ocean_library_present"] = P31
SEED_FNS["P32_primitive_coverage"] = P32


def run_all_primitive_design_seeds(
    map_view: KnownMapView, context: dict[str, Any]
) -> dict[str, OperatorResult]:
    out: dict[str, OperatorResult] = {}
    for name, fn in SEED_FNS.items():
        try:
            out[name] = fn(map_view, context)
        except Exception as exc:
            out[name] = _seed(
                name, {"planted": 1.0}, f"error:{type(exc).__name__}", status="degraded"
            )
    for cid in PRIMITIVE_IDEA_IDS:
        if cid not in out:
            out[cid] = _seed(cid, {"planted": 1.0}, "forced_seed", status="degraded")
    return out
