"""DW Phase-1 harness — run the V031 perception trust gate over real levels.

EVAL-ONLY reads of the offline arcade (same convention as s4_recon.py); no
answer injection, no engine internals. Writes a per-level verdict fixture and
exits 0 when every level is trusted, 2 otherwise (V031 convention).

    python3 perception_gate_eval.py                    # L0..L5, default seed
    python3 perception_gate_eval.py --levels 0,1 --out fixtures/pg_smoke.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _paths  # noqa: F401
from _paths import LAB

from imagine_search_gate import _present, GAME_ID  # noqa: E402
from scene import extract_scene  # noqa: E402
from perception_gate import gate_scene  # noqa: E402
from arc_agi import Arcade  # noqa: E402
from arc_agi.base import OperationMode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--levels", default="0,1,2,3,4,5")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    arc = Arcade(arc_api_key="anonymous",
                 environments_dir=str((LAB.parent / "environment_files").resolve()),
                 operation_mode=OperationMode.OFFLINE)

    results = {}
    print(f"{'lvl':>3} {'verdict':>10} {'failures':<38} {'tgt(px/sc)':>10} {'spout(hs/sc)':>12}")
    for lvl in [int(x) for x in a.levels.split(",")]:
        env = arc.make(GAME_ID, seed=a.seed)
        fd, cur = _present(env, lvl)
        try:
            sc = extract_scene(cur)
        except Exception as e:  # noqa: BLE001
            results[f"L{lvl}"] = {"trusted": False,
                                  "gate_failures": ["extract_scene_exception"],
                                  "error": f"{type(e).__name__}: {e}"}
            print(f"{lvl:>3} {'EXTRFAIL':>10} {type(e).__name__:<38}")
            continue
        v = gate_scene(cur, sc)
        results[f"L{lvl}"] = v
        t = v["checks"]["targets"]; s = v["checks"]["spouts"]
        print(f"{lvl:>3} {'TRUSTED' if v['trusted'] else 'UNTRUSTED':>10} "
              f"{','.join(v['gate_failures']) or '-':<38} "
              f"{t['pixel_components']}/{t['scene_targets']:>2}       "
              f"{s['housing_components']}/{s['scene_spouts']}")

    out = Path(a.out) if a.out else (Path(__file__).parent / "fixtures"
                                     / "perception_gate_L0_L5_20260706.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": a.seed, "game_id": GAME_ID,
               "provenance": "observed offline-arcade frames vs scene.extract_scene; "
                             "no engine internals (DW plan Phase 1)",
               "results": results}
    out.write_text(json.dumps(payload, indent=1, default=str))
    n_trusted = sum(1 for r in results.values() if r.get("trusted"))
    print(f"\ntrusted {n_trusted}/{len(results)} -> {out}")
    raise SystemExit(0 if n_trusted == len(results) else 2)


if __name__ == "__main__":
    main()
