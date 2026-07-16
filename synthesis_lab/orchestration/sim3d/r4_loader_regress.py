"""R4/R5 loader regression gate — lattice-aware loader vs fixed references.

PASS iff on every 4px corpus: per-pour obs_masks byte-identical AND obs_lava
identical to the old (macro_any + >200 delta) labels; and on all five pre-r5
corpora the generalized row_or_column flash labels exactly match the previous
rows-only wall-band labels. L3 must still load 0-skip with all-40 lava (the
observed truth the old fixed-delta rule misread).
Writes fixtures/r4_phaseb_loader_20260706.json.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import _paths  # noqa
from _paths import FIXTURES
from scene import extract_scene, M, CELL, WALL
from perception_gate import gate_scene
from dynamics_fit import load_pours, macro_any

OLD_DELTA = 200
CORPORA_4PX = ["calib_l0_pours.npz", "calib_l1_pours_20260706.npz",
               "calib_l2_pours_20260706.npz", "r3_solvetime_pours_20260706.npz"]
CORPORA_ROWS_ONLY_REF = CORPORA_4PX + ["calib_l3_pours_20260706.npz"]


def rows_only_flash(pre, frames):
    pre_rows = (np.asarray(pre) == WALL).mean(axis=1) > 0.5
    for f in frames:
        rows = (np.asarray(f) == WALL).mean(axis=1) > 0.5
        if int((rows & ~pre_rows).sum()) >= 2:
            return True
    return False


def old_load(path):
    d = np.load(path, allow_pickle=True)
    pre, ff, lens, wins = d["pre"], d["frames_flat"], d["lens"], d["wins"]
    out, off = {}, 0
    for i in range(len(lens)):
        frames = ff[off:off + lens[i]]; off += lens[i]
        sc = extract_scene(pre[i])
        v = gate_scene(pre[i], sc)
        if not v["trusted"]:
            continue
        fr = [np.flipud(f) if sc.flipped else f for f in frames]
        reservoir = macro_any((np.flipud(pre[i]) if sc.flipped else pre[i]) == 6)
        base14 = int((pre[i] == 14).sum())
        masks = [macro_any(f == 6) & ~reservoir for f in fr]
        lava = any(int((f == 14).sum()) > base14 + OLD_DELTA for f in frames)
        out[i] = (masks, lava)
    return out


def rows_only_labels(path):
    d = np.load(path, allow_pickle=True)
    pre, ff, lens = d["pre"], d["frames_flat"], d["lens"]
    out, off = {}, 0
    for i in range(len(lens)):
        frames = ff[off:off + lens[i]]; off += lens[i]
        sc = extract_scene(pre[i])
        v = gate_scene(pre[i], sc)
        if not v["trusted"]:
            continue
        out[i] = rows_only_flash(pre[i], frames)
    return out

report = {"corpora": {}, "pass": True}
for fx in CORPORA_4PX:
    new, skipped = load_pours(FIXTURES / fx)
    old = old_load(FIXTURES / fx)
    old_rows = rows_only_labels(FIXTURES / fx)
    mask_id, lava_id, lava_diff = True, True, []
    rows_only_id, rows_only_diff = True, []
    for p in new:
        o_masks, o_lava = old[p["i"]]
        if len(o_masks) != len(p["obs_masks"]) or any(
                not np.array_equal(a, b) for a, b in zip(o_masks, p["obs_masks"])):
            mask_id = False
        if o_lava != p["obs_lava"]:
            lava_id = False
            lava_diff.append(p["i"])
        if old_rows[p["i"]] != p["obs_lava"]:
            rows_only_id = False
            rows_only_diff.append(p["i"])
    ok = mask_id and lava_id and rows_only_id
    report["corpora"][fx] = {"n": len(new), "skipped": len(skipped),
                             "masks_identical": mask_id, "lava_identical": lava_id,
                             "lava_diff_pours": lava_diff,
                             "rows_only_lava_identical": rows_only_id,
                             "rows_only_lava_diff_pours": rows_only_diff,
                             "pass": ok}
    report["pass"] &= ok
    print(f"[{fx}] n={len(new)} masks_identical={mask_id} "
          f"lava_identical={lava_id} rows_only_identical={rows_only_id} "
          f"{'PASS' if ok else 'FAIL delta=' + str(lava_diff) + ' rows=' + str(rows_only_diff)}")

l3, skipped = load_pours(FIXTURES / "calib_l3_pours_20260706.npz")
old_rows_l3 = rows_only_labels(FIXTURES / "calib_l3_pours_20260706.npz")
rows_only_l3_diff = [p["i"] for p in l3 if old_rows_l3[p["i"]] != p["obs_lava"]]
l3_ok = (len(l3) == 40 and len(skipped) == 0 and all(p["obs_lava"] for p in l3)
         and not rows_only_l3_diff
         and all(p["obs_masks"][0].shape == p["kmap"].solid.shape for p in l3))
report["corpora"]["calib_l3_pours_20260706.npz"] = {
    "n": len(l3), "skipped": len(skipped),
    "all_lava": all(p["obs_lava"] for p in l3),
    "rows_only_lava_identical": not rows_only_l3_diff,
    "rows_only_lava_diff_pours": rows_only_l3_diff,
    "mask_shape": list(l3[0]["obs_masks"][0].shape), "pass": l3_ok}
report["pass"] &= l3_ok
print(f"[calib_l3] n={len(l3)} skipped={len(skipped)} all_lava={all(p['obs_lava'] for p in l3)} "
      f"rows_only_identical={not rows_only_l3_diff} "
      f"mask_shape={l3[0]['obs_masks'][0].shape} {'PASS' if l3_ok else 'FAIL'}")

rows_only_pass = all(report["corpora"][fx].get("rows_only_lava_identical", False)
                     for fx in CORPORA_ROWS_ONLY_REF)
report["rows_only_label_regression"] = {
    "corpora": CORPORA_ROWS_ONLY_REF,
    "pass": bool(rows_only_pass),
}
report["pass"] &= rows_only_pass

out = FIXTURES / "r4_phaseb_loader_20260706.json"
out.write_text(json.dumps({
    "protocol": "round-4 Phase B gate: lattice-aware load_pours + row-flash lava rule; "
                "regression = byte-identical masks + identical lava labels on all 4px "
                "corpora; r5 row_or_column flash must exactly preserve the old rows-only "
                "labels on all five pre-r5 corpora; L3 loads 0-skip, all-40 lava, "
                "native-lattice mask shapes",
    **report}, indent=1))
print(f"-> {out}  OVERALL {'PASS' if report['pass'] else 'FAIL'}")
sys.exit(0 if report["pass"] else 2)
