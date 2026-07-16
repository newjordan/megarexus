#!/usr/bin/env python3
"""Driver for the megarexus synthesis_lab SP80 stack (SCOUT_EXPERIMENT).

Everything here is local OFFLINE arc_agi — no live scorecards, no
benchmark claims. Three subcommands:

  preflight   full sanity pass: env, offline Arcade boot, Portband
              adapter on a real frame, outcome_v4 checkpoint + forward
              passes, stored reports, human corpus, data shards.
  probe       real paired model-vs-random episodes via
              synthesis_lab/orchestration/level_probe.py, run against a
              scratch run-dir so canonical reports are never clobbered.
  render      boot the offline env, reset, save the frame as a PNG
              (project palette, nearest-neighbor upscale).

Run with the rig's miniconda python:
  python3 -B .claude/skills/run-synthesis-lab/driver.py preflight
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LAB = REPO / "synthesis_lab"
EXP = LAB / "experiments" / "megarexus_cnn"
DEFAULT_RUN = EXP / "runs" / "outcome_v4"
GAME_ID = "sp80-589a99af"
DEFAULT_SEED = 20280001

for _p in (LAB, LAB / "orchestration", EXP):
    sys.path.insert(0, str(_p))

FAILURES: list[str] = []


def section(name: str) -> None:
    print(f"\n{'=' * 64}\n[{name}]\n{'=' * 64}")


def fail(name: str, exc: Exception) -> None:
    FAILURES.append(name)
    print(f"FAIL: {name}: {type(exc).__name__}: {exc}")
    traceback.print_exc()


def boot_env(seed: int):
    """OFFLINE Arcade → (env, first_frame_data, grid[64,64] int64)."""
    import numpy as np
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(REPO / "environment_files"),
    )
    env = arcade.make(GAME_ID, seed=seed)
    fd = env.reset()
    frames = fd.frame if isinstance(fd.frame, list) else [fd.frame]
    grid = np.asarray(frames[-1], dtype=np.int64)
    return env, fd, grid


# ------------------------------------------------------------------ preflight
def cmd_preflight(args: argparse.Namespace) -> int:
    section("1. ENVIRONMENT")
    try:
        import numpy as np
        import torch

        print(f"python           {sys.version.split()[0]}  ({sys.executable})")
        print(f"numpy            {np.__version__}")
        print(f"torch            {torch.__version__}")
        print(f"cuda available   {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"gpu              {torch.cuda.get_device_name(0)}")
        import arc_agi

        print(f"arc_agi          {Path(arc_agi.__file__).parent}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception as e:  # noqa: BLE001
        fail("environment", e)
        return 1

    section(f"2. OFFLINE ARCADE ({GAME_ID}, seed {args.seed})")
    grid = None
    try:
        env, fd, grid = boot_env(args.seed)
        print(f"reset ok         frame {grid.shape}")
        print(f"state            {getattr(fd, 'state', None)}")
        print(f"levels_completed {getattr(fd, 'levels_completed', None)}")
        print(f"colors present   {sorted(np.unique(grid).tolist())}")
    except Exception as e:  # noqa: BLE001
        fail("offline_arcade", e)

    section("3. PORTBAND ADAPTER on the real reset frame")
    try:
        from engine_adapters.portband_glove_adapter import (
            catch_label,  # noqa: F401 — import check
            component_displacement,  # noqa: F401
            frame_components,
            glove_ports,
        )

        assert grid is not None, "no reset frame from section 2"
        comps = frame_components(grid)
        print(f"frame_components {len(comps)} components")
        for c in comps[:5]:
            print(f"  {c['component_id']:<12} color={c['color']:2d} "
                  f"bbox={c['bbox_xyxy']} area={c['area']}")
        top = comps[0]
        # NOTE: glove_ports takes a single bbox, not the component list
        ports = glove_ports(top["bbox_xyxy"], top["component_id"])
        print(f"glove_ports      {len(ports)} port rows "
              f"(sides: {[p.get('side') for p in ports]})")
    except Exception as e:  # noqa: BLE001
        fail("adapter", e)

    section("4. MODEL + CHECKPOINT")
    try:
        import torch

        from eval_ghost import encode
        from model import MegarexusCNN, MegarexusCNNConfig

        ckpt_path = args.run_dir / "best.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        stack_n = ckpt["config"].get("stack_frames", 2)
        model = (MegarexusCNN(MegarexusCNNConfig(n_input_frames=stack_n))
                 .to(device).eval())
        model.load_state_dict(ckpt["model"])
        print(f"checkpoint       {ckpt_path}")
        print(f"params           {sum(p.numel() for p in model.parameters()):,}")
        if "val" in ckpt:
            print("stored val metrics:")
            print(json.dumps(ckpt["val"], indent=2, sort_keys=True))
        assert grid is not None
        enc = encode(grid)
        stack = torch.cat([enc] * stack_n, dim=0).unsqueeze(0).to(device)
        with torch.no_grad():
            settle_logits, level_logit = model.forward_settle(stack)
            from dataset import MOVE_ACTION_AXIS

            plane = torch.zeros(1, 1, 16, 16, device=device)
            plane[0, 0, 8, 8] = 1.0
            aid = sorted(MOVE_ACTION_AXIS.keys())[0]
            ctrl = model.forward_control(
                stack, plane, torch.tensor([aid], device=device))
        print(f"forward_settle   settle {tuple(settle_logits.shape)}, "
              f"level_logit {level_logit.item():+.3f}")
        print(f"forward_control  action {aid} → logit {ctrl.item():+.3f}")
        print(f"MOVE_ACTION_AXIS {MOVE_ACTION_AXIS}")
    except Exception as e:  # noqa: BLE001
        fail("model_checkpoint", e)

    section("5. STORED LEVEL-PROBE REPORTS")
    for report in sorted(args.run_dir.glob("level_probe*.json")):
        try:
            data = json.loads(report.read_text())
            print(f"--- {report.name} ---")
            for k, v in data.items():
                if k == "detail":
                    for pol, rows in v.items():
                        lv = [r["levels_completed"] for r in rows]
                        print(f"  detail.{pol}: {len(rows)} eps, "
                              f"levels_completed={lv}")
                else:
                    print(f"  {k}: {json.dumps(v)}")
        except Exception as e:  # noqa: BLE001
            fail(report.name, e)

    section("6. HUMAN CORPUS (corpus/sp80_liquid_pour)")
    try:
        rows = [json.loads(l) for l in
                (REPO / "corpus" / "sp80_liquid_pour" / "manifest.jsonl")
                .read_text().splitlines() if l.strip()]
        eps: dict[str, list] = {}
        for r in rows:
            # NOTE: the grouping key is episode_id, not episode
            eps.setdefault(r["episode_id"], []).append(r)
        print(f"manifest rows    {len(rows)}")
        for name, rs in sorted(eps.items()):
            lv = max(r.get("level", 0) for r in rs)
            print(f"  {name:<55} frames={len(rs):4d} max_level={lv}")
        n_png = sum(1 for _ in
                    (REPO / "corpus" / "sp80_liquid_pour" / "frames")
                    .rglob("*.png"))
        print(f"frames on disk   {n_png} PNGs")
    except Exception as e:  # noqa: BLE001
        fail("corpus", e)

    section("7. DATA SHARDS")
    import torch

    for shard in sorted((EXP / "data").glob("*.pt")):
        try:
            obj = torch.load(shard, map_location="cpu", weights_only=False)
            desc = (f"list[{len(obj)}]" if isinstance(obj, list)
                    else type(obj).__name__)
            print(f"{shard.name:<28} {desc}")
        except Exception as e:  # noqa: BLE001
            fail(shard.name, e)

    section("PRE-FLIGHT VERDICT")
    if FAILURES:
        print(f"FAILED sections: {FAILURES}")
        return 1
    print("ALL SECTIONS PASSED — output above is real command output "
          "(SCOUT_EXPERIMENT, offline only).")
    return 0


# ---------------------------------------------------------------------- probe
def cmd_probe(args: argparse.Namespace) -> int:
    """Paired model-vs-random episodes without touching canonical reports.

    level_probe.py writes its report INTO --run-dir, so we symlink the
    checkpoint into a scratch dir and point the probe there.
    """
    scratch = Path(tempfile.mkdtemp(prefix="sp80_level_probe_"))
    (scratch / "best.pt").symlink_to(args.run_dir.resolve() / "best.pt")
    cmd = [
        sys.executable, "-B",
        str(LAB / "orchestration" / "level_probe.py"),
        "--run-dir", str(scratch),
        "--episodes", str(args.episodes),
        "--seed", str(args.seed),
        "--finder", args.finder,
        "--out-name", "level_probe_report.json",
    ]
    print("+ " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, check=False).returncode
    report = scratch / "level_probe_report.json"
    if rc == 0 and report.exists():
        data = json.loads(report.read_text())
        print("\nheadline:")
        for pol in ("model", "random"):
            s = data[pol]
            print(f"  {pol:<7} {s['episodes_with_level_completed']}"
                  f"/{s['episodes']} episodes completed a level "
                  f"(pours={s['total_pours']}, "
                  f"strict_catches={s['total_strict_catches']})")
        print(f"\nfull report: {report}")
    return rc


# --------------------------------------------------------------------- render
def cmd_render(args: argparse.Namespace) -> int:
    import numpy as np
    from PIL import Image

    from dataset import PALETTE

    env, fd, grid = boot_env(args.seed)
    rgb = PALETTE[grid]
    img = Image.fromarray(rgb).resize(
        (64 * args.scale, 64 * args.scale), Image.NEAREST)
    out = Path(args.out) if args.out else Path(
        tempfile.gettempdir()) / f"sp80_reset_seed{args.seed}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"state={getattr(fd, 'state', None)} "
          f"levels_completed={getattr(fd, 'levels_completed', None)} "
          f"colors={sorted(np.unique(grid).tolist())}")
    print(f"frame → {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("preflight", help="full stack sanity pass")
    sp.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    sp.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sp.set_defaults(fn=cmd_preflight)

    sp = sub.add_parser("probe", help="paired model-vs-random episodes")
    sp.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    sp.add_argument("--episodes", type=int, default=2)
    sp.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sp.add_argument("--finder", choices=("probe", "learned"),
                    default="learned")
    sp.set_defaults(fn=cmd_probe)

    sp = sub.add_parser("render", help="render the reset frame to PNG")
    sp.add_argument("--seed", type=int, default=DEFAULT_SEED)
    sp.add_argument("--scale", type=int, default=8)
    sp.add_argument("--out", default=None)
    sp.set_defaults(fn=cmd_render)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
