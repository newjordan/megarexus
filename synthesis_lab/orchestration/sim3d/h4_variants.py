"""H4 certification — the machinery, not the knowledge, is what ships.

Pre-registered claim (DREAMWEAVER_INTEGRATION_PLAN §2 H4): the identical
perception + kernel machinery, with NO game-specific input beyond the scene.py
PALETTE manifest, runs end-to-end on synthesized `sp80_variants` boards; and
the dynamics-fit harness re-run FROM SCRATCH on variant pours alone re-derives
the same dynamics knowledge that rounds 1-3 induced from real-level pours.

Protocol:
  1. Synthesize variant boards (single-source / rotated-base / multi-source
     families; the redirect family is EXCLUDED — its sprite-tag fix is
     source-derived and redirect physics is unmodeled, so it stays out of any
     machinery-certification claim).
  2. Per board: extract_scene -> perception gate -> (trusted only) KnownMap ->
     round-3 rollout, with a determinism double-run check. Untrusted boards are
     REFUSED and counted, never guessed at.
  3. Pour each usable board ONCE in the real env (full animation, env-graded
     win) -> a fresh observed corpus disjoint from every prior fit corpus.
  4. Re-run the FULL round-3 selection grid from scratch on variant pours only
     (train/holdout i%4 split). H4 PASSES if the round-3 member
     {deflect=none, spread=lateral, mouth=fill, edge=dispose, cap=1, absorb=
     unlimited, rule=fill_cap, stop=volume} sits in the top tie-block by
     train (outcome+lava) AND the from-scratch selected member passes the
     holdout gates. The full table is reported either way.

The variant GENERATOR (sp80_variants) manipulates env sprites — sanctioned
curriculum/test scaffolding (it varies the PROBLEM, never injects an answer);
the machinery under certification never touches it as an input.

    python3 h4_variants.py [--per-family 12]
Exit 0 if H4 passes, 2 otherwise. Fixture: fixtures/h4_variants_20260706.json
"""
from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

import _paths  # noqa: F401
from _paths import FIXTURES, LAB

from imagine_search_gate import ACTION_OF_ID, game_live  # noqa: E402
from sp80_variants import make_env_variant  # noqa: E402
from scene import extract_scene  # noqa: E402
from perception_gate import gate_scene  # noqa: E402
from imagination_kernel import KnownMap, FallingMaterial, run  # noqa: E402
from dynamics_fit import load_pours  # noqa: E402
from r3_fit import rollout_eval, score, SEMANTICS  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402

R3_SELECTED = {"cfg": {"deflect": "none", "spread": "lateral", "mouth": "fill",
                       "edge": "dispose", "cell_cap": 1.0, "absorb": "unlimited"},
               "rule": "fill_cap", "stop": "volume"}
FAMILIES = [("single", {}), ("rotated", {"base_level": 1}),
            ("multi2", {"n_sources": 2}), ("multi3", {"n_sources": 3})]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=12)
    a = ap.parse_args()
    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)

    fam_stats = {}
    pres, frames_flat, lens, wins = [], [], [], []
    det_fail = 0
    for fam, kw in FAMILIES:
        rng = np.random.default_rng(20260706)
        st = {"boards": 0, "degenerate": 0, "extract_error": 0, "trusted": 0,
              "untrusted": 0, "gate_failures": {}, "poured": 0, "wins": 0}
        for i in range(a.per_family):
            env, g = make_env_variant(arc, seed=7000 + i, rng=rng, **kw)
            if env is None:
                st["degenerate"] += 1
                continue
            st["boards"] += 1
            try:
                sc = extract_scene(g)
            except Exception as e:  # noqa: BLE001
                st["extract_error"] += 1
                st["gate_failures"][f"extract:{type(e).__name__}"] = \
                    st["gate_failures"].get(f"extract:{type(e).__name__}", 0) + 1
                continue
            v = gate_scene(g, sc)
            if not v["trusted"]:
                st["untrusted"] += 1
                for gf in v["gate_failures"]:
                    st["gate_failures"][gf] = st["gate_failures"].get(gf, 0) + 1
                continue
            st["trusted"] += 1
            km = KnownMap.from_scene(sc, v, family=fam, board=i)
            s1 = run(km, FallingMaterial(**R3_SELECTED["cfg"]), max_ticks=200)
            s2 = run(km, FallingMaterial(**R3_SELECTED["cfg"]), max_ticks=200)
            if [s.signature() for s in s1] != [s.signature() for s in s2]:
                det_fail += 1
            # one real pour: fresh observed dynamics data, env-graded
            fd = env.step(ACTION_OF_ID[1])
            if fd is None or not game_live(fd):
                continue
            base = int(getattr(fd, "levels_completed", 0) or 0)
            fd = env.step(ACTION_OF_ID[5])
            if fd is None:
                continue
            f = fd.frame
            frames = ([np.asarray(x, dtype=np.int8) for x in f] if isinstance(f, list)
                      else [np.asarray(f, dtype=np.int8)])
            won = int(getattr(fd, "levels_completed", 0) or 0) > base
            pres.append(np.asarray(g, dtype=np.int8))
            frames_flat.extend(frames); lens.append(len(frames)); wins.append(bool(won))
            st["poured"] += 1; st["wins"] += int(won)
        fam_stats[fam] = st
        print(f"[h4] {fam}: {st}", flush=True)

    npz = FIXTURES / "h4_variant_pours_20260706.npz"
    np.savez_compressed(npz, pre=np.stack(pres), frames_flat=np.stack(frames_flat),
                        lens=np.asarray(lens, dtype=np.int32),
                        wins=np.asarray(wins, dtype=bool))
    (FIXTURES / "h4_variant_pours_20260706.provenance.json").write_text(json.dumps(
        {"provenance": "sp80_variants boards (env-synthesized problems, no answers) "
                       "poured once each in the real env; observation data for the "
                       "H4 from-scratch dynamics re-selection", "families": fam_stats},
        indent=1))

    # From-scratch selection on VARIANT pours only (identical protocol to r3_fit)
    pours, skipped = load_pours(npz)
    train = [p for p in pours if p["i"] % 4 != 3]
    hold = [p for p in pours if p["i"] % 4 == 3]
    print(f"[h4] fit corpus: {len(pours)} usable ({len(skipped)} gate-skipped), "
          f"{sum(p['obs_win'] for p in pours)} wins -> train {len(train)} / hold {len(hold)}",
          flush=True)
    grid = [dict(zip(("deflect", "spread", "mouth", "edge", "cell_cap", "absorb"), v))
            for v in itertools.product(
                ["none", "dl", "dr", "both"], ["none", "lateral"], ["fill", "entry"],
                ["dispose", "halt"], [1.0, 2.0, 3.0], ["cap", "unlimited"])]
    table = []
    for gi, cfg in enumerate(grid):
        rows = [rollout_eval(cfg, p) for p in train]
        for rule, stop in SEMANTICS:
            table.append({"cfg": cfg, "rule": rule, "stop": stop,
                          "train": score(rows, train, rule, stop)})
        if (gi + 1) % 48 == 0:
            print(f"[h4] {gi + 1}/{len(grid)} rollout configs", flush=True)
    table.sort(key=lambda e: (e["train"]["outcome_acc"] + e["train"]["lava_acc"],
                              e["train"]["mean_iou"]), reverse=True)

    top_key = round(table[0]["train"]["outcome_acc"] + table[0]["train"]["lava_acc"], 6)
    tie_block = [e for e in table
                 if round(e["train"]["outcome_acc"] + e["train"]["lava_acc"], 6) == top_key]
    r3_in_tie = any(e["cfg"] == R3_SELECTED["cfg"] and e["rule"] == R3_SELECTED["rule"]
                    and e["stop"] == R3_SELECTED["stop"] for e in tie_block)
    best = table[0]
    hrows = [rollout_eval(best["cfg"], p) for p in hold]
    hr = score(hrows, hold, best["rule"], best["stop"])
    gates = {"r3_member_in_top_tie_block": r3_in_tie,
             "holdout_outcome": hr["outcome_acc"] >= 0.90,
             "holdout_win_side": hr["win_side_acc"] in (None, 1.0)}
    passed = all(gates.values())

    print(f"[h4] top tie-block: {len(tie_block)} members (train out+lava={top_key})")
    print(f"[h4] from-scratch argmax: {best['cfg']} rule={best['rule']} stop={best['stop']}")
    print(f"[h4] round-3 member in top tie-block: {r3_in_tie}")
    print(f"[h4] holdout of argmax: {hr}")
    print(f"[h4] determinism double-run failures: {det_fail}")
    print(f"[h4] GATES {gates} -> {'H4 PASS' if passed else 'H4 FAIL (report table)'}")

    out = FIXTURES / "h4_variants_20260706.json"
    out.write_text(json.dumps({
        "protocol": "H4: palette-manifest-only machinery on sp80_variants boards; "
                    "from-scratch full-grid dynamics re-selection on variant pours "
                    "(disjoint from all prior fit corpora); redirect family excluded "
                    "(source-derived tag; unmodeled physics)",
        "families": fam_stats, "determinism_failures": det_fail,
        "fit": {"n_pours": len(pours), "gate_skipped": len(skipped),
                "n_wins": int(sum(p["obs_win"] for p in pours))},
        "r3_selected": R3_SELECTED, "from_scratch_argmax":
            {"cfg": best["cfg"], "rule": best["rule"], "stop": best["stop"]},
        "top_tie_block_size": len(tie_block), "r3_member_in_top_tie_block": r3_in_tie,
        "holdout": hr, "table_top40": table[:40], "gates": gates,
        "pass": bool(passed)}, indent=1, default=str))
    print(f"-> {out}", flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
