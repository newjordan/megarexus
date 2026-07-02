#!/usr/bin/env python3
"""Install repo-owned Rexus PufferLib env sources into a PufferLib checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUFFERLIB = ROOT / ".external" / "pufferlib-4.0-newjordan"
ENV_ROOT = ROOT / "codexlab" / "pufferlib_envs"


def install_env(pufferlib_path: Path, env_name: str) -> list[Path]:
    source = ENV_ROOT / env_name
    env_dir = pufferlib_path / "ocean" / env_name
    config_path = pufferlib_path / "config" / f"{env_name}.ini"
    env_dir.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    written = []
    header = f"{env_name}.h"
    for name in (header, "binding.c"):
        dst = env_dir / name
        shutil.copy2(source / name, dst)
        written.append(dst)
    shutil.copy2(source / f"{env_name}.ini", config_path)
    written.append(config_path)
    return written


def install_rexus_sp80_paintflow(pufferlib_path: Path) -> list[Path]:
    return install_env(pufferlib_path, "rexus_sp80_paintflow")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pufferlib-path", type=Path, default=DEFAULT_PUFFERLIB)
    parser.add_argument(
        "--env",
        action="append",
        choices=["rexus_sp80_paintflow", "rexus_sp80_paintflow_l1_multi"],
        help="Env name to install. Defaults to all repo-owned envs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pufferlib_path = args.pufferlib_path.expanduser().resolve()
    if not pufferlib_path.exists():
        raise SystemExit(f"PufferLib checkout not found: {pufferlib_path}")
    envs = args.env or ["rexus_sp80_paintflow", "rexus_sp80_paintflow_l1_multi"]
    written = []
    for env_name in envs:
        written.extend(install_env(pufferlib_path, env_name))
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
