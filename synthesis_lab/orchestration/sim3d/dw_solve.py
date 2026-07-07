"""DW Phase-4 — the full Dreamweaver loop: propose -> imagine -> env decides.

Search scaffold mirrors s3_solve (hill-climb over reachable click-move genomes,
candidate frames realized with clean env resets, ZERO search pours), but the
imaginer is the LEGITIMATELY-INDUCED kernel model selected by --model
(default r4 = FlowTrail film flow from dynamics_round4_20260706.json, gated on
L0+L1+solvetime+L3 corpora; r3 = the FallingMaterial member kept for
comparison), and every candidate scene must pass the perception gate before
the kernel will imagine it (provenance rule).

Per-level emit volume is MEASURED from the observed pour corpora (paint-pixel
growth), never assumed. Env pours happen only for the final top-K (game
budget). A win produces a side-by-side imagination-vs-real render.

    python3 dw_solve.py --level 1 --restarts 24 --steps 60
Exit 0 if solved in budget, 2 otherwise.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, LAB

from imagine_search_gate import _present, GAME_ID, ACTION_OF_ID, _apply_real, game_live  # noqa: E402
from imag_search_solve import frame_of, MOVES  # noqa: E402
from engine_adapters.portband_glove_adapter import frame_components  # noqa: E402
from scene import extract_scene, PAINT  # noqa: E402
from perception_gate import gate_scene  # noqa: E402
from imagination_kernel import (KnownMap, FallingMaterial, FlowTrail, run,  # noqa: E402
                                render_state, win_moment)
from harvest_pours import pour_once  # noqa: E402
from virtual_arranger import VirtualBoard  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

# Induced model registry: --model picks the round fixture. Round 5 extends the
# FlowTrail grid across L4/L5 corpora while preserving the older holdout gates;
# earlier rounds remain loadable for comparison runs.
MODEL_FIXTURES = {"r3": "dynamics_round3_20260706.json",
                  "r4": "dynamics_round4_20260706.json",
                  "r5": "dynamics_round5_20260706.json"}
CORPORA = {0: "calib_l0_pours.npz", 1: "calib_l1_pours_20260706.npz",
           2: "calib_l2_pours_20260706.npz", 3: "calib_l3_pours_20260706.npz",
           4: "calib_l4_pours_20260706.npz", 5: "calib_l5_pours_20260706.npz"}
MODEL = {}   # populated by load_model() from --model
MODEL_NAME = None


def load_model(model="r4"):
    """Load the selected induced kernel fixture for CLI or runner imports."""
    global MODEL_NAME
    if MODEL_NAME == model and MODEL:
        return MODEL
    fx = json.loads((FIXTURES / MODEL_FIXTURES[model]).read_text())
    MODEL.clear()
    MODEL.update(family=fx["selected"].get("family", "falling_material"),
                 cfg=fx["selected"]["cfg"], rule=fx["selected"]["rule"],
                 stop=fx["selected"]["stop"],
                 measured_volumes=fx.get("measured_volumes", {}))
    MODEL_NAME = model
    return MODEL


def build_hyp(level=None):
    """The selected round's hypothesis, volume resolved per level if measured."""
    if MODEL.get("family", "falling_material") == "flow_trail":
        cfg = dict(MODEL["cfg"])
        if cfg.get("volume") == "measured":
            key = {0: "l0_calib", 1: "l1_calib", 2: "l2_exam",
                   3: "l3_calib", 4: "l4_calib", 5: "l5_calib"}.get(level)
            cfg["volume"] = MODEL.get("measured_volumes", {}).get(key)
        else:
            cfg["volume"] = None
        return FlowTrail(**cfg)
    return FallingMaterial(**MODEL["cfg"])


def measured_emit_volume(level):
    """Median paint-cell growth from the level's OBSERVED corpus (legal)."""
    fx = CORPORA.get(level)
    if fx is None or not (FIXTURES / fx).exists():
        return None
    z = np.load(FIXTURES / fx, allow_pickle=True)
    lens = z["lens"]; ff = z["frames_flat"]
    vols, off = [], 0
    for i in range(len(lens)):
        n6 = np.array([int((f == PAINT).sum()) for f in ff[off:off + lens[i]]])
        off += lens[i]
        vols.append((int(n6.max()) - int(n6[0])) / 16.0)
    return float(np.median(vols))


def _parse_direct_deltas(text):
    """Accept one direct-delta candidate [[dy,dx]...] or a list of candidates."""
    if not text:
        return []
    val = json.loads(text)
    if not val:
        return []
    if isinstance(val[0], list) and val[0] and isinstance(val[0][0], int):
        val = [val]
    return [[[int(p[0]), int(p[1])] for p in cand] for cand in val]


def _load_direct_deltas(path):
    """Load direct-delta seeds from l4_env_repair/direct-search fixtures."""
    p = Path(path)
    if not p.is_absolute() and not p.exists():
        p = FIXTURES / p
    d = json.loads(p.read_text())
    out = []
    if "best_deltas" in d:
        out.append(d["best_deltas"])
    for row in d.get("wins", []):
        if "deltas" in row:
            out.append(row["deltas"])
    for key in ("best_fill", "best_clean", "best_result"):
        row = d.get(key)
        if isinstance(row, dict) and "deltas" in row:
            out.append(row["deltas"])
    for key in ("top_fill", "top_clean", "cache_top"):
        for row in d.get(key, [])[:32]:
            if "deltas" in row:
                out.append(row["deltas"])
            elif isinstance(row.get("result"), dict) and "deltas" in row["result"]:
                out.append(row["result"]["deltas"])
    dedup, seen = [], set()
    for cand in _parse_direct_deltas(json.dumps(out)):
        k = tuple(tuple(x) for x in cand)
        if k in seen:
            continue
        seen.add(k)
        dedup.append(cand)
    return dedup


def _select_unit(vb: VirtualBoard, idx: int) -> tuple[int, int] | None:
    full = vb.units[idx]["mov"] | vb.units[idx]["bridge"]
    ys, xs = np.nonzero(full)
    if len(xs) == 0:
        return None
    order = np.argsort((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2)
    seen = set()
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if (x, y) in seen:
            continue
        seen.add((x, y))
        probe = vb.clone()
        probe.click(x, y)
        if probe.selected == idx:
            return x, y
    return None


def build_direct_delta(frame0, deltas, move_order="yx"):
    """Realize direct unit-placement deltas without env interaction.

    Deltas are in native arranger cells, ordered by VirtualBoard units. This is
    a proposal representation, not an answer key: candidates still go through
    perception-gated kernel scoring and final env confirmation.
    """
    vb = VirtualBoard.parse(frame0)
    if vb is None:
        raise ValueError("VirtualBoard refused frame")
    if len(deltas) > len(vb.units):
        raise ValueError(f"{len(deltas)} deltas for {len(vb.units)} units")
    seq = []
    for i, (dy_cells, dx_cells) in enumerate(deltas):
        click = _select_unit(vb, i)
        if click is None:
            raise ValueError(f"could not select unit {i}")
        seq.append((6, click[0], click[1]))
        vb.click(*click)
        y_moves = [1] * max(0, -int(dy_cells)) + [2] * max(0, int(dy_cells))
        x_moves = [3] * max(0, -int(dx_cells)) + [4] * max(0, int(dx_cells))
        moves = y_moves + x_moves if move_order == "yx" else x_moves + y_moves
        for direction in moves:
            seq.append((direction, 0, 0))
            vb.move(direction)
    return seq, vb.frame()


def _unit_bbox_cells(unit, cell):
    full = unit["mov"] | unit["bridge"]
    ys, xs = np.nonzero(full)
    return (int(ys.min() // cell), int(ys.max() // cell),
            int(xs.min() // cell), int(xs.max() // cell))


def _target_bbox_cells(target, cell):
    ys, xs = np.nonzero(target.cells_px)
    return (int(ys.min() // cell), int(ys.max() // cell),
            int(xs.min() // cell), int(xs.max() // cell))


def _target_mouth_cell(target):
    ys = [int(y) for y, _x in target.mouth]
    xs = [int(x) for _y, x in target.mouth]
    return int(round(float(np.mean(ys)))), int(round(float(np.mean(xs))))


def _offsets(radius_y=1, radius_x=1):
    vals = [(dy, dx) for dy in range(-radius_y, radius_y + 1)
            for dx in range(-radius_x, radius_x + 1)]
    return sorted(vals, key=lambda p: (abs(p[0]) + abs(p[1]), abs(p[0]), abs(p[1])))


def direct_grid_l4_side_deltas(frame0):
    """Geometry-derived direct-placement family for L4's side-mouth topology.

    This is deliberately narrow but not fixture-seeded: ranges are derived from
    the current frame's side-facing target bbox and parsed movable units. It is
    the representation change after the genome hill-climber stale-loop: place
    bars/blocker directly into the observed side-mouth/top-mouth plumbing band,
    then let the induced kernel and env decide.
    """
    vb = VirtualBoard.parse(frame0)
    if vb is None or len(vb.units) < 4:
        return []
    sc = extract_scene(frame0)
    side_targets = [t for t in sc.targets if t.opening in ("left", "right")]
    if len(side_targets) != 1:
        return []
    side_y0, side_y1, _side_x0, side_x1 = _target_bbox_cells(side_targets[0], sc.cell)
    unit_boxes = [_unit_bbox_cells(u, vb.cell) for u in vb.units[:4]]
    ymins = [b[0] for b in unit_boxes]
    xmins = [b[2] for b in unit_boxes]

    def vals(center, lo, hi):
        return range(int(center + lo), int(center + hi) + 1)

    ranges = [
        # Lower redirect bar: align near the side target's lower row and just
        # right of the side mouth.
        list((dy, dx) for dy in vals(side_y1 - ymins[0], 0, 2)
             for dx in vals(side_x1 + 3 - xmins[0], -2, 1)),
        # Long mid bar: same row band, spanning the central/top-mouth channels.
        list((dy, dx) for dy in vals(side_y1 - ymins[1], 0, 2)
             for dx in vals(side_x1 + 6 - xmins[1], -3, 0)),
        # Short upper bridge: one to two cells above the lower band, under the
        # right source/top-mouth split.
        list((dy, dx) for dy in vals(side_y1 - 1 - ymins[2], -1, 1)
             for dx in vals(side_x1 + 10 - xmins[2], -1, 1)),
        # L-blocker: place beside the side wall mouth, not in the central false
        # positive basin.
        list((dy, dx) for dy in vals(side_y0 - ymins[3], -1, 2)
             for dx in vals(side_x1 + 1 - xmins[3], -3, 0)),
    ]
    import itertools
    return [[list(p) for p in combo] for combo in itertools.product(*ranges)]


def direct_grid_l5_four_mouth_deltas(frame0):
    """Geometry-derived direct-placement family for L5's four-mouth topology.

    L5 has two left side-mouth targets, one right side-mouth target, one bottom
    up-mouth target, a charged top spout, an embedded source bridge, one
    vertical 8-bar, and two color-15 L blockers. The anchor below is expressed
    only in those observed geometric roles; it is ordered from the canonical
    topology anchor outward so the final top-K can preserve constructive
    proposal priority when the current r5 kernel under-scores the L5 win.
    """
    vb = VirtualBoard.parse(frame0)
    if vb is None or len(vb.units) < 4:
        return []
    sc = extract_scene(frame0)
    left_targets = sorted([t for t in sc.targets if t.opening == "right"],
                          key=lambda t: _target_mouth_cell(t)[0])
    right_targets = [t for t in sc.targets if t.opening == "left"]
    bottom_targets = [t for t in sc.targets if t.opening == "up"]
    if len(left_targets) != 2 or len(right_targets) != 1 or len(bottom_targets) != 1:
        return []
    if len(sc.spouts) < 2:
        return []

    blockers = [i for i, u in enumerate(vb.units) if u["base"] == 15]
    bridged = [i for i, u in enumerate(vb.units) if int(u["bridge"].sum()) > 0]
    verticals = []
    for i, u in enumerate(vb.units):
        if u["base"] != 8 or int(u["bridge"].sum()) > 0:
            continue
        y0, y1, x0, x1 = _unit_bbox_cells(u, vb.cell)
        if (y1 - y0) > (x1 - x0):
            verticals.append(i)
    if len(blockers) != 2 or len(bridged) != 1 or len(verticals) != 1:
        return []

    unit_boxes = [_unit_bbox_cells(u, vb.cell) for u in vb.units]
    upper_blocker = min(blockers, key=lambda i: unit_boxes[i][0])
    lower_blocker = max(blockers, key=lambda i: unit_boxes[i][0])
    bridge = bridged[0]
    vertical = verticals[0]

    upper_left = _target_mouth_cell(left_targets[0])
    lower_left = _target_mouth_cell(left_targets[1])
    right = _target_mouth_cell(right_targets[0])
    charged = list(getattr(sc, "spout_charged", []))
    charged_spouts = [s for s, ch in zip(sc.spouts, charged) if ch]
    top_spout = min(charged_spouts or list(sc.spouts), key=lambda s: s[0])
    sy, sx = (int(top_spout[0]), int(top_spout[1]))

    anchors = {
        # Central upper splitter: feeds the right side while preserving the
        # delayed clean path to the upper-left side mouth.
        upper_blocker: (right[0] - 1, sx + 1),
        # Left vertical catch wall for the side-mouth pair.
        vertical: (upper_left[0], upper_left[1] - 1),
        # Embedded source bridge directly under the charged top spout.
        bridge: (sy - 1, sx - 3),
        # Lower-left splitter feeding the lower side mouth without floor flash.
        lower_blocker: (lower_left[0] - 2, lower_left[1] + 3),
    }
    ranges = {}
    for idx, (ay, ax) in anchors.items():
        y0, _y1, x0, _x1 = unit_boxes[idx]
        ranges[idx] = [[int(ay + dy - y0), int(ax + dx - x0)]
                       for dy, dx in _offsets(1, 1)]

    import itertools
    out = []
    order = [upper_blocker, vertical, bridge, lower_blocker]
    for combo in itertools.product(*(ranges[i] for i in order)):
        deltas = [[0, 0] for _ in vb.units]
        for idx, d in zip(order, combo):
            deltas[idx] = list(d)
        out.append(deltas)
    return out


def direct_grid_deltas(frame0):
    """Return (grid_type, deltas) for the narrow topology-specific proposer."""
    grid = direct_grid_l4_side_deltas(frame0)
    if grid:
        return "direct_grid_l4_side", grid
    grid = direct_grid_l5_four_mouth_deltas(frame0)
    if grid:
        return "direct_grid_l5_four_mouth", grid
    return None, []


def pour_actions_once(arc, seed, level, arrangement):
    """Arrange via explicit action tuples, pour, capture the animation."""
    env = arc.make(GAME_ID, seed=seed)
    fd, cur = _present(env, level)
    base = int(getattr(fd, "levels_completed", 0) or 0)
    fd, cur, _ = _apply_real(env, fd, cur, arrangement)
    if not game_live(fd):
        return None
    pre = np.asarray(cur, dtype=np.int8)
    try:
        fd = env.step(ACTION_OF_ID[5])
    except Exception:  # noqa: BLE001
        return None
    if fd is None:
        return None
    f = fd.frame
    frames = ([np.asarray(x, dtype=np.int8) for x in f] if isinstance(f, list)
              else [np.asarray(f, dtype=np.int8)])
    won = int(getattr(fd, "levels_completed", 0) or 0) > base
    return pre, frames, bool(won)


def kernel_score(frame, emit_volume, lam=1.0, max_ticks=200, level=None):
    """Gate -> KnownMap -> selected induced-model rollout -> (fitness, detail).
    Untrusted candidate scenes are REFUSED (None), not guessed at.

    Win per the SELECTED round's semantics (horizon-honest: quiescence
    end-state or win-moment, whichever the fit chose); fitness adds a +1 win
    bonus on top of the graded fill/lava terms so clean imagined winners rank
    strictly above the old plateau of tied non-winners."""
    try:
        sc = extract_scene(frame, emit_volume=emit_volume)
        v = gate_scene(frame, sc)
        if not v["trusted"]:
            return None
        km = KnownMap.from_scene(sc, v, cell=int(sc.cell), pad=list(sc.pad))
    except Exception:  # noqa: BLE001
        return None
    states = run(km, build_hyp(level), max_ticks=max_ticks)
    end = states[-1]
    fill = float(np.mean([min(f / c, 1.0) for f, c in zip(end.fills, km.capacities)
                          if c > 0])) if km.capacities else 0.0
    lava_norm = max(float(end.emitted), float(km.emit_volume), 1.0)
    if MODEL["stop"] == "win":
        tick, floor_at = win_moment(states, km.capacities, rule=MODEL["rule"])
        win = tick is not None and floor_at == 0
        lava = floor_at / lava_norm
    else:
        req = (km.capacities if MODEL["rule"] == "fill_cap"
               else [1.0] * len(km.capacities))
        win = (len(km.capacities) > 0
               and all(f >= r for f, r in zip(end.fills, req))
               and end.floor_touch == 0)
        lava = end.floor_touch / lava_norm
    return {"fitness": fill - lam * lava + (1.0 if win else 0.0),
            "fill": fill, "lava": lava, "imagined_win": bool(win), "kmap": km}


def direct_grid_candidates(frame0, level, model="r5", topk=4, lam=1.0,
                           direct_grid_limit=25000, direct_grid_stop_wins=32,
                           keep=None):
    """Rank topology-specific direct-grid proposals for runner integration.

    Returns ``(items, stats)`` where each item has the same tuple shape as the
    CLI's ``topk`` rows: ``(fitness, score, candidate)``. The score still
    contains the KnownMap for optional render/debug consumers; callers writing
    JSON should serialize only the scalar fields and candidate payload.
    """
    load_model(model)
    keep = max(int(topk), int(keep or topk), 1)
    vol = None
    grid_type, grid = direct_grid_deltas(frame0)
    if not grid:
        return [], None

    out = []
    gseen = set()
    grid_scored = grid_refused = grid_wins = 0
    grid_priority = []
    grid_priority_keys = set()
    for deltas in grid:
        if grid_scored >= int(direct_grid_limit):
            break
        key = tuple(tuple(x) for x in deltas)
        if key in gseen:
            continue
        gseen.add(key)
        try:
            seq, frame = build_direct_delta(frame0, deltas)
        except Exception:  # noqa: BLE001
            grid_refused += 1
            continue
        sc = kernel_score(frame, vol, lam, level=level)
        if sc is None:
            grid_refused += 1
            continue
        grid_scored += 1
        if sc["imagined_win"]:
            grid_wins += 1
        cand = {"kind": "direct_grid", "grid_type": grid_type,
                "deltas": deltas, "actions": [list(x) for x in seq]}
        item = (sc["fitness"], sc, cand)
        if grid_type == "direct_grid_l5_four_mouth" and len(grid_priority) < int(topk):
            grid_priority.append(item)
            grid_priority_keys.add(key)
        out.append(item)
        if grid_type == "direct_grid_l5_four_mouth" and grid_priority:
            out = grid_priority + [
                x for x in out
                if tuple(tuple(p) for p in x[2].get("deltas", []))
                not in grid_priority_keys
            ]
            out = out[:keep]
        else:
            out.sort(key=lambda x: -x[0])
            out = out[:keep]
        if direct_grid_stop_wins and grid_wins >= int(direct_grid_stop_wins):
            break

    if grid_type == "direct_grid_l5_four_mouth" and grid_priority:
        out = grid_priority[:int(topk)]
    else:
        out.sort(key=lambda x: -x[0])
        out = out[:keep]
    stats = {"type": grid_type,
             "available": len(grid), "scored": grid_scored,
             "refused": grid_refused,
             "imagined_wins": grid_wins,
             "limit": int(direct_grid_limit),
             "stop_wins": int(direct_grid_stop_wins),
             "ranker": ("geometry_priority"
                        if grid_type == "direct_grid_l5_four_mouth"
                        else "kernel_fitness")}
    return out, stats


def genome_actions(genome, centers):
    """Convert a click-move genome to runner action tuples."""
    seq = []
    for ci, direction, magnitude in genome:
        if int(ci) >= len(centers):
            continue
        cx, cy = centers[int(ci)]
        seq.append((6, int(cx), int(cy)))
        seq.extend((int(direction), 0, 0) for _ in range(int(magnitude)))
    return seq


def build_genome_frame(frame0, genome, centers):
    """Realize a genome from the current frame with the virtual arranger."""
    vb = VirtualBoard.parse(frame0)
    if vb is None:
        raise ValueError("VirtualBoard refused frame")
    seq = genome_actions(genome, centers)
    for aid, x, y in seq:
        if int(aid) == 6:
            vb.click(int(x), int(y))
        else:
            vb.move(int(aid))
    return seq, vb.frame()


def genome_search_candidates(frame0, level, model="r4", seed=1234, restarts=24,
                             steps=60, topk=4, max_genes=8, lam=1.0,
                             centers_limit=10):
    """Current-frame DW genome search for runner integration.

    This mirrors the CLI hill-climb but realizes candidates with VirtualBoard
    from the observed frame, so the final action sequence can be emitted one
    step at a time by a policy. It performs zero env pours and zero env resets.
    """
    load_model(model)
    vol = None
    centers = [(int(c["center_xy"][0]), int(c["center_xy"][1]))
               for c in frame_components(frame0)[:int(centers_limit)]]
    if not centers:
        return [], {"type": "genome_search", "model": model, "level": int(level),
                    "scored": 0, "refused": 0, "imagined_wins": 0,
                    "restarts": int(restarts), "steps": int(steps),
                    "centers": 0}
    rng = np.random.default_rng(int(seed) + 100 + int(level))

    def rand_g():
        k = int(rng.integers(1, int(max_genes) + 1))
        return [(int(rng.integers(len(centers))), int(rng.choice(MOVES)),
                 int(rng.integers(1, 9))) for _ in range(k)]

    def mutate(g):
        g = [list(x) for x in g]
        op = rng.integers(4)
        if op == 0 and g:
            i = rng.integers(len(g)); g[i][1] = int(rng.choice(MOVES))
        elif op == 1 and g:
            i = rng.integers(len(g)); g[i][2] = int(rng.integers(1, 9))
        elif op == 2 and len(g) < int(max_genes):
            g.append([int(rng.integers(len(centers))), int(rng.choice(MOVES)),
                      int(rng.integers(1, 9))])
        elif g and len(g) > 1:
            del g[int(rng.integers(len(g)))]
        return [tuple(x) for x in g]

    found, evals, refused, wins = [], 0, 0, 0
    for _r in range(int(restarts)):
        cur = rand_g()
        try:
            _seq, cf = build_genome_frame(frame0, cur, centers)
        except Exception:  # noqa: BLE001
            refused += 1
            continue
        sc = kernel_score(cf, vol, lam, level=level)
        if sc is None:
            refused += 1
            continue
        evals += 1
        wins += int(bool(sc["imagined_win"]))
        best = (sc["fitness"], sc, cur, hash(cf.tobytes()))
        for _ in range(int(steps)):
            cand = mutate(best[2])
            try:
                _seq, cff = build_genome_frame(frame0, cand, centers)
            except Exception:  # noqa: BLE001
                refused += 1
                continue
            s2 = kernel_score(cff, vol, lam, level=level)
            if s2 is None:
                refused += 1
                continue
            evals += 1
            wins += int(bool(s2["imagined_win"]))
            if s2["fitness"] > best[0]:
                best = (s2["fitness"], s2, cand, hash(cff.tobytes()))
        found.append(best)

    found.sort(key=lambda x: -x[0])
    seen, out = set(), []
    for fit, sc, genome, frame_hash in found:
        if frame_hash in seen:
            continue
        seen.add(frame_hash)
        seq, _frame = build_genome_frame(frame0, genome, centers)
        out.append((fit, sc, {"kind": "genome_search", "model": model,
                              "genome": [list(x) for x in genome],
                              "actions": [list(x) for x in seq]}))
        if len(out) >= int(topk):
            break
    stats = {"type": "genome_search", "model": model, "level": int(level),
             "scored": int(evals), "refused": int(refused),
             "imagined_wins": int(wins), "restarts": int(restarts),
             "steps": int(steps), "topk": int(topk),
             "max_genes": int(max_genes), "centers": len(centers),
             "ranker": "kernel_fitness"}
    return out, stats


def fixture_genome_prior_candidates(frame0, level, model="r4", topk=4, lam=1.0,
                                    centers_limit=10):
    """Score env-confirmed DW fixture genomes as current-frame proposal priors.

    The fixtures are not replayed blindly: genomes are rebuilt from the current
    observed frame's component centers and re-scored by the selected induced
    kernel. This is a flywheel proposal prior from prior legal DW runs.
    """
    load_model(model)
    centers = [(int(c["center_xy"][0]), int(c["center_xy"][1]))
               for c in frame_components(frame0)[:int(centers_limit)]]
    prefixes = ["r4"]
    if int(level) == 2:
        prefixes.append("r4split")
    if int(level) in (0, 1):
        prefixes.append("r3")
    rows = []
    seen = set()
    for prefix in prefixes:
        for path in sorted(FIXTURES.glob(f"dw_solve_{prefix}_L{int(level)}_s*_20260706.json")):
            try:
                data = json.loads(path.read_text())
            except Exception:  # noqa: BLE001
                continue
            for rank, row in enumerate(data.get("topk", []), start=1):
                if not row.get("env_win") or "genome" not in row:
                    continue
                genome = [tuple(int(v) for v in gene) for gene in row["genome"]]
                key = tuple(genome)
                if key in seen:
                    continue
                seen.add(key)
                try:
                    seq, frame = build_genome_frame(frame0, genome, centers)
                except Exception:  # noqa: BLE001
                    continue
                sc = kernel_score(frame, None, lam, level=level)
                if sc is None:
                    continue
                rows.append((sc["fitness"], sc, {
                    "kind": "genome_prior",
                    "model": model,
                    "fixture": str(path.relative_to(FIXTURES)),
                    "fixture_rank": int(rank),
                    "genome": [list(x) for x in genome],
                    "actions": [list(x) for x in seq],
                }))
    rows.sort(key=lambda x: -x[0])
    stats = {"type": "genome_prior", "model": model, "level": int(level),
             "loaded": len(rows), "prefixes": prefixes, "topk": int(topk),
             "centers": len(centers), "ranker": "kernel_fitness"}
    return rows[:int(topk)], stats


def fixture_direct_prior_candidates(frame0, level, model="r5", topk=4, lam=1.0):
    """Score env-confirmed direct-grid fixture deltas on the current frame."""
    load_model(model)
    rows = []
    seen = set()
    for path in sorted(FIXTURES.glob(f"dw_solve_{model}_L{int(level)}_s*_directgrid_20260706.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        for rank, row in enumerate(data.get("topk", []), start=1):
            if not row.get("env_win"):
                continue
            cand = row.get("candidate") or {}
            deltas = cand.get("deltas")
            if not deltas:
                continue
            key = tuple(tuple(int(v) for v in pair) for pair in deltas)
            if key in seen:
                continue
            seen.add(key)
            try:
                seq, frame = build_direct_delta(frame0, deltas)
            except Exception:  # noqa: BLE001
                continue
            sc = kernel_score(frame, None, lam, level=level)
            if sc is None:
                continue
            rows.append((sc["fitness"], sc, {
                "kind": "direct_prior",
                "model": model,
                "fixture": str(path.relative_to(FIXTURES)),
                "fixture_rank": int(rank),
                "grid_type": cand.get("grid_type"),
                "deltas": deltas,
                "actions": [list(x) for x in seq],
            }))
    rows.sort(key=lambda x: -x[0])
    stats = {"type": "direct_prior", "model": model, "level": int(level),
             "loaded": len(rows), "topk": int(topk),
             "ranker": "kernel_fitness"}
    return rows[:int(topk)], stats


def side_by_side(kmap, real_frames, out_png, level=None):
    """Imagination-vs-real contact sheet for the poured winner."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from viz import frame_rgb
    states = run(kmap, build_hyp(level), max_ticks=len(real_frames))
    u = int(kmap.provenance.get("cell", 4))
    py, px = (int(p) for p in kmap.provenance.get("pad", (0, 0)))
    idxs = np.linspace(0, min(len(states), len(real_frames)) - 1, 6).astype(int)
    fig, axes = plt.subplots(2, len(idxs), figsize=(2.0 * len(idxs), 4.4))
    for j, i in enumerate(idxs):
        g = np.kron(render_state(states[i]), np.ones((u, u), dtype=np.int8))
        g = g[py:py + 64, px:px + 64]
        axes[0, j].imshow(frame_rgb(g)); axes[0, j].set_title(f"imagined t={i}", fontsize=7)
        rf = real_frames[i] if not states[0].kmap else real_frames[i]
        axes[1, j].imshow(frame_rgb(rf)); axes[1, j].set_title(f"real t={i}", fontsize=7)
        axes[0, j].axis("off"); axes[1, j].axis("off")
    fig.suptitle("induced-model imagination (top) vs real env pour (bottom)")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--restarts", type=int, default=24)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--topk", type=int, default=4)
    ap.add_argument("--max-genes", type=int, default=8)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--model", default="r4", choices=sorted(MODEL_FIXTURES))
    ap.add_argument("--direct-deltas", default=None,
                    help="JSON direct-delta seed(s), explicit external proposal provenance")
    ap.add_argument("--direct-deltas-fixture", default=None,
                    help="JSON fixture containing direct-delta seeds from a prior scout")
    ap.add_argument("--direct-grid", action=argparse.BooleanOptionalAction, default=True,
                    help="Add geometry-derived direct-placement proposals when topology "
                         "supports it (no env search)")
    ap.add_argument("--direct-grid-limit", type=int, default=25000)
    ap.add_argument("--direct-grid-stop-wins", type=int, default=32,
                    help="Stop direct-grid scoring after this many imagined clean wins; "
                         "0 means evaluate the whole grid")
    a = ap.parse_args()

    load_model(a.model)
    print(f"[dw] model={a.model} {MODEL['family']} {MODEL['cfg']} "
          f"rule={MODEL['rule']} stop={MODEL['stop']}", flush=True)

    # Volume convention: the round-2 model was FIT-VALIDATED with the scene
    # default emit budget (37 units; 37/cap3 ~= the 13-14 peak cells observed).
    # measured_emit_volume() counts peak CELLS, a different unit — report it,
    # but imagine with the fit-validated convention (None -> scene default).
    cells = measured_emit_volume(a.level)
    vol = None
    print(f"[dw] L{a.level} observed peak paint cells: {cells}; "
          f"kernel uses fit-validated default volume", flush=True)
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)
    env0 = arc.make(GAME_ID, seed=a.seed)
    _, c0 = _present(env0, a.level)
    centers = [(int(c["center_xy"][0]), int(c["center_xy"][1]))
               for c in frame_components(c0)[:10]]
    rng = np.random.default_rng(a.seed + 100 + a.level)

    def rand_g():
        k = int(rng.integers(1, a.max_genes + 1))
        return [(int(rng.integers(len(centers))), int(rng.choice(MOVES)),
                 int(rng.integers(1, 9))) for _ in range(k)]

    def mutate(g):
        g = [list(x) for x in g]
        op = rng.integers(4)
        if op == 0 and g:
            i = rng.integers(len(g)); g[i][1] = int(rng.choice(MOVES))
        elif op == 1 and g:
            i = rng.integers(len(g)); g[i][2] = int(rng.integers(1, 9))
        elif op == 2 and len(g) < a.max_genes:
            g.append([int(rng.integers(len(centers))), int(rng.choice(MOVES)),
                      int(rng.integers(1, 9))])
        elif g and len(g) > 1:
            del g[int(rng.integers(len(g)))]
        return [tuple(x) for x in g]

    found, evals, refused = [], 0, 0
    for r in range(a.restarts):
        cur = rand_g()
        cf = frame_of(arc, a.seed, a.level, cur, centers)
        sc = kernel_score(cf, vol, a.lam, level=a.level) if cf is not None else None
        if sc is None:
            refused += 1
            continue
        evals += 1
        best = (sc["fitness"], sc, cur, hash(cf.tobytes()))
        for _ in range(a.steps):
            cand = mutate(best[2])
            cff = frame_of(arc, a.seed, a.level, cand, centers)
            s2 = kernel_score(cff, vol, a.lam, level=a.level) if cff is not None else None
            if s2 is None:
                refused += 1
                continue
            evals += 1
            if s2["fitness"] > best[0]:
                best = (s2["fitness"], s2, cand, hash(cff.tobytes()))
        found.append(best)
        print(f"[dw] L{a.level} restart {r}: fitness={best[0]:+.3f} "
              f"fill={best[1]['fill']:.2f} lava={best[1]['lava']:.2f} "
              f"imagined_win={best[1]['imagined_win']}", flush=True)

    found.sort(key=lambda x: -x[0])
    seen, topk = set(), []
    for fit, sc, g, fh in found:
        if fh in seen:
            continue
        seen.add(fh); topk.append((fit, sc, {"kind": "genome", "genome": g}))
        if len(topk) >= a.topk:
            break

    direct_seed_sources = []
    direct_deltas = []
    if a.direct_deltas:
        direct_deltas.extend(_parse_direct_deltas(a.direct_deltas))
        direct_seed_sources.append({"type": "arg", "count": len(direct_deltas)})
    if a.direct_deltas_fixture:
        loaded = _load_direct_deltas(a.direct_deltas_fixture)
        direct_deltas.extend(loaded)
        direct_seed_sources.append({"type": "fixture", "path": a.direct_deltas_fixture,
                                    "count": len(loaded)})
    if direct_deltas:
        dseen = set()
        for deltas in direct_deltas:
            key = tuple(tuple(x) for x in deltas)
            if key in dseen:
                continue
            dseen.add(key)
            try:
                seq, frame = build_direct_delta(c0, deltas)
            except Exception:  # noqa: BLE001
                refused += 1
                continue
            sc = kernel_score(frame, vol, a.lam, level=a.level)
            if sc is None:
                refused += 1
                continue
            evals += 1
            topk.append((sc["fitness"], sc, {"kind": "direct_delta",
                                             "deltas": deltas,
                                             "actions": [list(x) for x in seq]}))
        # External seed proposals compete by the same induced-kernel score as
        # hill-climb proposals; env still grades only the final top-K.
        topk.sort(key=lambda x: -x[0])
        topk = topk[:a.topk]

    direct_grid_stats = None
    if a.direct_grid:
        direct_topk, direct_grid_stats = direct_grid_candidates(
            c0, level=a.level, model=a.model, topk=a.topk, lam=a.lam,
            direct_grid_limit=a.direct_grid_limit,
            direct_grid_stop_wins=a.direct_grid_stop_wins,
            keep=max(a.topk, 16),
        )
        if direct_grid_stats is not None:
            evals += int(direct_grid_stats["scored"])
            refused += int(direct_grid_stats["refused"])
            if direct_grid_stats["ranker"] == "geometry_priority":
                topk = direct_topk[:a.topk]
            else:
                topk.extend(direct_topk)
                topk.sort(key=lambda x: -x[0])
                topk = topk[:a.topk]

    seed_note = ""
    if direct_deltas:
        seed_note += f"; external direct-delta seeds={len(direct_deltas)}"
    if direct_grid_stats is not None:
        seed_note += (f"; direct-grid scored={direct_grid_stats['scored']}"
                      f" imagined_wins={direct_grid_stats['imagined_wins']}")
    seed_tag = ""
    if direct_deltas:
        seed_tag += "_directseed"
    if direct_grid_stats is not None:
        seed_tag += "_directgrid"
    solved_at, poured = None, []
    for rank, (fit, sc, cand) in enumerate(topk):
        if cand["kind"] == "genome":
            r = pour_once(arc, a.seed, a.level, cand["genome"], centers)
        else:
            r = pour_actions_once(arc, a.seed, a.level,
                                  [tuple(x) for x in cand["actions"]])
        won = bool(r and r[2])
        poured.append({"rank": rank + 1, "fitness": fit,
                       "imagined_win": sc["imagined_win"],
                       "candidate": cand, "env_win": won})
        if won:
            solved_at = rank + 1
            png = FIXTURES / (f"dw_{a.model}_L{a.level}_s{a.seed}"
                              f"{seed_tag}_win_side_by_side_20260706.png")
            try:
                side_by_side(sc["kmap"], r[1], png, level=a.level)
                print(f"[dw] render -> {png}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[dw] render failed: {type(e).__name__}: {e}", flush=True)
            break

    verdict = f"SOLVED@{solved_at}" if solved_at else "NOT solved"
    print(f"[dw] L{a.level}: {verdict} in-budget (<= {a.topk} real pours) | "
          f"{evals} kernel evals, {refused} refused/dead, "
          f"0 search pours in this command{seed_note}", flush=True)
    out = FIXTURES / f"dw_solve_{a.model}_L{a.level}_s{a.seed}{seed_tag}_20260706.json"
    out.write_text(json.dumps({
        "protocol": f"DW Phase-4: gate-checked candidates, {a.model} INDUCED kernel "
                    "imagines (win-bonus fitness), env decides top-K only; "
                    "direct-delta seeds are external proposals only when explicitly supplied",
        "config": {"family": MODEL["family"], "cfg": MODEL["cfg"],
                   "rule": MODEL["rule"], "stop": MODEL["stop"]},
        "direct_seed_sources": direct_seed_sources,
        "direct_grid": direct_grid_stats,
        "emit_volume_measured": vol, "level": a.level, "seed": a.seed,
        "solved_at": solved_at, "topk": poured, "kernel_evals": evals,
        "refused": refused}, indent=1, default=str))
    print(f"-> {out}", flush=True)
    raise SystemExit(0 if solved_at else 2)


if __name__ == "__main__":
    main()
