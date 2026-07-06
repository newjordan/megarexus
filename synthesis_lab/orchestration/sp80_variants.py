"""Synthesize randomised SP80 variant boards (NIGHT FINDING 29, user's "move the obj
blocks around" idea) to FORCE the imagination to learn transferable routing SKILL instead
of memorising one fixed layout.

Levels are explicit sprite layouts; we clone the source/target/block/wall sprite templates
from a base level and re-place them at randomised, VALID positions (source top, targets
bottom opening up, blocks in the middle, no overlaps), then pour with the REAL CA physics.
Honest curriculum augmentation: we vary the PROBLEM, never inject an answer; every pour is
env-graded; the real unmodified levels stay held out for evaluation.

    from sp80_variants import make_env_variant, pour_variant
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

_LAB = Path(__file__).resolve().parents[1]
for p in (_LAB / "orchestration", _LAB / "experiments" / "megarexus_cnn"):
    sys.path.insert(0, str(p))
from imagine_search_gate import _present, _grid, GAME_ID, ACTION_OF_ID, game_live  # noqa: E402

SRC_TAG = "sowlljgtjvn"      # source
TGT_TAG = "repwkzbkhxl"      # U-target
WALL_TAG = "waoewejnqzc"     # wall bar
BLOCK_TAG = "plzwjbfyfli"    # movable split block
REDIR_TAG = "tuvkdkhdokr"    # redirect block (P1 fix 2026-07-06: was "liolfvkveqg", the PAINT
                             # DROPLET — a mislabel; the real 90°-deflect redirect is
                             # tuvkdkhdokr (game L787-803, memory megarexus-sp80-mechanic). Only
                             # affects tpl["redir"], which the single-source path never places, so
                             # single-source variants / S2 are unchanged; it is used only by the
                             # new redirect family below.
REDIR_LEVELS = (4, 5)        # base levels that actually carry a tuvkdkhdokr template


def _templates(level):
    """Grab one clone-able sprite of each role from a base level."""
    def one(tag):
        s = level.get_sprites_by_tag(tag)
        return s[0] if s else None
    return {"src": one(SRC_TAG), "tgt": one(TGT_TAG), "wall": one(WALL_TAG),
            "block": one(BLOCK_TAG), "redir": one(REDIR_TAG)}


def make_env_variant(arc, seed, rng, n_targets=None, n_blocks=None, rotate=None, base_level=0,
                     n_sources=None, with_redirect=False):
    """Build an env, replace the BASE level's sprites with a randomised valid layout, present
    it. base_level selects which real level's GEOMETRY/rotation to inherit (0=source-top,
    1=source-bottom/rotated-180, etc.) — the honest way to cover rotated real-level geometries
    (NIGHT FINDING 35). Returns (env, frame) or (None, None) if the layout renders degenerate.
    Target/source placement follows the base's target row so the inherited flow reaches them.

    P1 (2026-07-06) v2 families — the multi-source/redirect coverage v1 lacked (doc §1):
      n_sources    : None (default) = the byte-identical single-source path (S2 unchanged);
                     int >=2 = the MULTI-SOURCE family (2-3 spaced source blobs -> distinct
                     spouts, L2/L4-like — the P0 overfit antidote).
      with_redirect: True places one real redirect (tuvkdkhdokr) in the middle — the TRUE-
                     REDIRECT family (base_level in REDIR_LEVELS so the template exists).
    Both new branches draw from `rng` ONLY when engaged, so the default path is unchanged."""
    env = arc.make(GAME_ID, seed=seed)
    env.reset(); env._game.set_level(int(base_level))
    game = env._game
    cl = game.current_level
    tpl = _templates(cl)
    if tpl["src"] is None or tpl["tgt"] is None or tpl["wall"] is None:
        return None, None
    # remove every existing sprite, then add our randomised set
    for s in list(cl.get_sprites()):
        cl.remove_sprite(s)
    nt = n_targets or int(rng.integers(2, 5))          # 2-4 targets (VISIBLE diversity)
    nb = n_blocks if n_blocks is not None else int(rng.integers(1, 6))  # 1-5 blocks
    # NOTE: do NOT vary dojfslwbg (rotation) here — it does NOT rotate the rendered layout,
    # only the HIDDEN pour-flow rotation, so it gives identical-looking arrangements
    # different outcomes = contradictory training signal that poisoned real-L1 transfer
    # (NIGHT FINDING 32). Every outcome-determining factor must be VISIBLE in the frame.
    # targets at bottom row y=13, non-overlapping columns (each ~3 wide on 16-grid)
    tgt_cols = sorted(rng.choice(range(1, 13), size=min(nt, 12), replace=False).tolist())
    tgt_cols = [c for i, c in enumerate(tgt_cols) if all(abs(c - p) >= 3 for p in tgt_cols[:i])]
    occupied = set()
    for c in tgt_cols:
        cl.add_sprite(tpl["tgt"].clone().set_position(int(c), 13))
        for dx in range(-1, 2):
            occupied.add((c + dx, 13))
    # source(s) at top row y=0. Single-source path (default) is byte-identical.
    if n_sources is None:
        sx = int(rng.integers(1, 15))
        cl.add_sprite(tpl["src"].clone().set_position(sx, 0))
    else:                                          # MULTI-SOURCE family (L2/L4-like)
        # Anchor sources ABOVE target columns (one per target, small jitter) so boards are
        # WINNABLE-but-routed: a straight/near-straight drop can fill each mouth, while blocks
        # and arrangements create the flood/miss losses. Pure-random source columns give a
        # 0% win rate (multi-source win is a needle, like real L2) -> no positive class to
        # gate recall/precision on. Jitter spans easy(aligned)->routed(offset), covering the
        # L2 geometry where sources sit 1-2 cols off their mouths.
        ns = min(int(n_sources), len(tgt_cols))
        pick = sorted(rng.choice(len(tgt_cols), size=ns, replace=False).tolist())
        src_cols = []
        for ti in pick:
            c = int(min(14, max(1, tgt_cols[ti] + int(rng.integers(-1, 2)))))
            if all(abs(c - u) >= 2 for u in src_cols):   # spacing -> distinct spouts
                src_cols.append(c)
            elif all(abs(tgt_cols[ti] - u) >= 2 for u in src_cols):
                src_cols.append(int(tgt_cols[ti]))
        for c in sorted(set(src_cols)):
            cl.add_sprite(tpl["src"].clone().set_position(int(c), 0))
    # blocks scattered in the middle (rows 3..11), avoiding source column and target span
    if tpl["block"] is not None:
        placed = 0
        for _ in range(nb * 4):
            if placed >= nb:
                break
            bx, by = int(rng.integers(1, 15)), int(rng.integers(3, 12))
            if (bx, by) in occupied:
                continue
            cl.add_sprite(tpl["block"].clone().set_position(bx, by))
            occupied.add((bx, by)); placed += 1
    # TRUE-REDIRECT family: one real 90°-deflect block in the middle (game L787-803)
    if with_redirect and tpl["redir"] is not None:
        for _ in range(8):
            rx, ry = int(rng.integers(1, 15)), int(rng.integers(3, 12))
            if (rx, ry) not in occupied:
                cl.add_sprite(tpl["redir"].clone().set_position(rx, ry))
                occupied.add((rx, ry)); break
    cl.add_sprite(tpl["wall"].clone().set_position(0, 15))
    game.on_set_level(cl)
    fd = env.step(ACTION_OF_ID[1])
    if fd is None or not game_live(fd):
        return None, None
    g = _grid(fd)
    if int((g == 11).sum()) == 0 or int((g == 4).sum()) == 0:  # need targets + source
        return None, None
    return env, g


def pour_variant(env, base_wall, ntarg, arrangement=None):
    """Pour the (optionally rearranged) variant; return (frame_before_pour, fill_frac_peak,
    sustained_fill, lava, env_win)."""
    from imagine_search_gate import _apply_real
    fd = env.step(ACTION_OF_ID[1]); cur = _grid(fd)
    base = int(getattr(fd, "levels_completed", 0) or 0)
    if arrangement:
        fd, cur, _ = _apply_real(env, fd, cur, arrangement)
        if not game_live(fd):
            return cur, 0.0, 0.0, 999, False
    fd = env.step(ACTION_OF_ID[5])
    won = int(getattr(fd, "levels_completed", 0) or 0) > base
    f = fd.frame
    frames = [np.asarray(x, dtype=np.int8) for x in f] if isinstance(f, list) else [np.asarray(f, dtype=np.int8)]
    n13 = [int((x == 13).sum()) for x in frames]
    peak = max(n13) / max(ntarg, 1)
    sus = 0
    for i in range(len(n13) - 2):
        sus = max(sus, min(n13[i:i + 3]))
    lava = max(int((x == 14).sum()) for x in frames) - base_wall
    return cur, peak, sus / max(ntarg, 1), int(lava), bool(won)


if __name__ == "__main__":
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((_LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    rng = np.random.default_rng(0)
    ok = 0
    for i in range(12):
        env, g = make_env_variant(arc, seed=100 + i, rng=rng)
        if env is None:
            print(f"variant {i}: degenerate"); continue
        ntarg = int((g == 11).sum()); base_wall = int((g == 14).sum())
        srcx = int(np.nonzero(g == 4)[1].mean() // 4)
        tcols = sorted(set((np.nonzero(g == 11)[1] // 4).tolist()))
        _, peak, sus, lava, won = pour_variant(env, base_wall, ntarg)
        ok += 1
        print(f"variant {i}: src_col={srcx} tgt_cols={tcols} n_targets={ntarg} "
              f"| default pour: peak_fill={peak:.2f} lava={lava} win={won}")
    print(f"{ok}/12 valid variants")
