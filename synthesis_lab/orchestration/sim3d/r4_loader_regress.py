"""R4 Phase B regression gate — lattice-aware loader vs the old fixed-16x16 path.

PASS iff on every 4px corpus: per-pour obs_masks byte-identical AND obs_lava
identical to the old (macro_any + >200 delta) labels; and the L3 corpus loads
0-skip with all-40 lava (the observed truth the old rule misread).
Writes fixtures/r4_phaseb_loader_20260706.json.
"""
import sys, json
sys.path.insert(0, "/home/frosty40/megarexus/synthesis_lab/orchestration/sim3d")
import numpy as np
import _paths  # noqa
from _paths import FIXTURES
from scene import extract_scene, M, CELL
from perception_gate import gate_scene
from dynamics_fit import load_pours, macro_any

OLD_DELTA = 200
CORPORA_4PX = ["calib_l0_pours.npz", "calib_l1_pours_20260706.npz",
               "calib_l2_pours_20260706.npz", "r3_solvetime_pours_20260706.npz"]

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

report = {"corpora": {}, "pass": True}
for fx in CORPORA_4PX:
    new, skipped = load_pours(FIXTURES / fx)
    old = old_load(FIXTURES / fx)
    mask_id, lava_id, lava_diff = True, True, []
    for p in new:
        o_masks, o_lava = old[p["i"]]
        if len(o_masks) != len(p["obs_masks"]) or any(
                not np.array_equal(a, b) for a, b in zip(o_masks, p["obs_masks"])):
            mask_id = False
        if o_lava != p["obs_lava"]:
            lava_id = False
            lava_diff.append(p["i"])
    ok = mask_id and lava_id
    report["corpora"][fx] = {"n": len(new), "skipped": len(skipped),
                             "masks_identical": mask_id, "lava_identical": lava_id,
                             "lava_diff_pours": lava_diff, "pass": ok}
    report["pass"] &= ok
    print(f"[{fx}] n={len(new)} masks_identical={mask_id} lava_identical={lava_id} "
          f"{'PASS' if ok else 'FAIL ' + str(lava_diff)}")

l3, skipped = load_pours(FIXTURES / "calib_l3_pours_20260706.npz")
l3_ok = (len(l3) == 40 and len(skipped) == 0 and all(p["obs_lava"] for p in l3)
         and all(p["obs_masks"][0].shape == p["kmap"].solid.shape for p in l3))
report["corpora"]["calib_l3_pours_20260706.npz"] = {
    "n": len(l3), "skipped": len(skipped),
    "all_lava": all(p["obs_lava"] for p in l3),
    "mask_shape": list(l3[0]["obs_masks"][0].shape), "pass": l3_ok}
report["pass"] &= l3_ok
print(f"[calib_l3] n={len(l3)} skipped={len(skipped)} all_lava={all(p['obs_lava'] for p in l3)} "
      f"mask_shape={l3[0]['obs_masks'][0].shape} {'PASS' if l3_ok else 'FAIL'}")

out = FIXTURES / "r4_phaseb_loader_20260706.json"
out.write_text(json.dumps({
    "protocol": "round-4 Phase B gate: lattice-aware load_pours + row-flash lava rule; "
                "regression = byte-identical masks + identical lava labels on all 4px "
                "corpora; L3 loads 0-skip, all-40 lava, native-lattice mask shapes",
    **report}, indent=1))
print(f"-> {out}  OVERALL {'PASS' if report['pass'] else 'FAIL'}")
sys.exit(0 if report["pass"] else 2)
