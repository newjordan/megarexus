#!/usr/bin/env python3
"""PufferLib readiness gate for Rexus RL environment work."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CODEXLAB = ROOT / "codexlab"
DEFAULT_REPO_PUFFERLIB = ROOT / ".external" / "pufferlib-4.0-newjordan"
LEGACY_TMP_PUFFERLIB = Path("/tmp/pufferlib-4.0-newjordan")
CHECK_SCHEMA = "codexlab.pufferlib_env_check.v1"


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def candidate_pufferlib_path(arg_path: str | None) -> Path | None:
    if arg_path:
        return Path(arg_path).expanduser().resolve()
    env_path = os.environ.get("PUFFERLIB_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    if DEFAULT_REPO_PUFFERLIB.exists():
        return DEFAULT_REPO_PUFFERLIB.resolve()
    if LEGACY_TMP_PUFFERLIB.exists():
        return LEGACY_TMP_PUFFERLIB.resolve()
    return None


def repo_git_metadata(path: Path | None) -> dict[str, Any]:
    if path is None or not (path / ".git").exists():
        return {"available": False}
    result: dict[str, Any] = {"available": True}
    for label, args in {
        "commit": ["git", "-C", str(path), "rev-parse", "HEAD"],
        "branch": ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
        "status_short": ["git", "-C", str(path), "status", "--short"],
    }.items():
        completed = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        result[label] = completed.stdout.strip() if completed.returncode == 0 else None
        if completed.returncode != 0:
            result[f"{label}_error"] = completed.stderr.strip()
    return result


def native_extension_metadata(module_path: str | None) -> dict[str, Any]:
    if not module_path:
        return {"available": False}
    package_dir = Path(module_path).resolve().parent
    candidates = sorted(package_dir.glob("_C*.so"))
    payload: dict[str, Any] = {
        "available": bool(candidates),
        "path": display_path(candidates[0]) if candidates else None,
        "sha256": sha256_file(candidates[0]) if candidates else None,
    }
    if candidates:
        completed = subprocess.run(
            ["file", str(candidates[0])],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        payload["file"] = completed.stdout.strip() if completed.returncode == 0 else None
        if completed.returncode != 0:
            payload["file_error"] = completed.stderr.strip()
    return payload


def inspect_pufferlib(path: Path | None) -> dict[str, Any]:
    started = time.perf_counter()
    if path is not None:
        sys.path.insert(0, str(path))
    payload: dict[str, Any] = {
        "path": display_path(path),
        "path_exists": bool(path and path.exists()),
        "path_inserted": bool(path),
    }
    try:
        module = importlib.import_module("pufferlib")
        try:
            version = importlib.metadata.version("pufferlib")
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", None)
        payload.update(
            {
                "top_level_available": True,
                "version": version,
                "module_file": getattr(module, "__file__", None),
            }
        )
    except Exception as exc:  # noqa: BLE001 - preflight records exact runtime failure.
        payload.update(
            {
                "available": False,
                "top_level_available": False,
                "pufferl_available": False,
                "native_backend_available": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
        return payload

    try:
        native = importlib.import_module("pufferlib._C")
        payload.update(
            {
                "native_backend_available": True,
                "native_backend_file": getattr(native, "__file__", None),
                "compiled_env": getattr(native, "env_name", None),
                "gpu_backend": bool(getattr(native, "gpu", False)),
                "has_create_pufferl": hasattr(native, "create_pufferl"),
                "has_create_vec": hasattr(native, "create_vec"),
                "has_rollouts": hasattr(native, "rollouts"),
                "has_train": hasattr(native, "train"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "native_backend_available": False,
                "native_error_type": type(exc).__name__,
                "native_error_message": str(exc),
            }
        )

    try:
        pufferl = importlib.import_module("pufferlib.pufferl")
        payload.update(
            {
                "pufferl_available": True,
                "pufferl_file": getattr(pufferl, "__file__", None),
                "has_load_config": hasattr(pufferl, "load_config"),
                "has_train_entrypoint": hasattr(pufferl, "train"),
                "has_eval_entrypoint": hasattr(pufferl, "eval"),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "pufferl_available": False,
                "pufferl_error_type": type(exc).__name__,
                "pufferl_error_message": str(exc),
            }
        )

    payload["legacy_python_api"] = {
        "PufferEnv": hasattr(sys.modules.get("pufferlib"), "PufferEnv"),
        "vector": import_available("pufferlib.vector"),
        "emulation": import_available("pufferlib.emulation"),
    }
    payload["torch_backend"] = inspect_torch_backend()
    payload["native_extension"] = native_extension_metadata(payload.get("module_file"))
    payload["runtime_ready"] = bool(
        payload.get("top_level_available") and payload.get("native_backend_available") and payload.get("pufferl_available")
    )
    payload["native_cuda_trainer_ready"] = bool(
        payload.get("has_create_pufferl") and payload.get("has_rollouts") and payload.get("has_train")
    )
    payload["torch_trainer_ready"] = bool((payload.get("torch_backend") or {}).get("available"))
    payload["available"] = bool(payload["runtime_ready"] and (payload["native_cuda_trainer_ready"] or payload["torch_trainer_ready"]))
    payload["latency_ms"] = (time.perf_counter() - started) * 1000.0
    return payload


def inspect_torch_backend() -> dict[str, Any]:
    try:
        module = importlib.import_module("pufferlib.torch_pufferl")
        backend = getattr(module, "PuffeRL", None)
        return {
            "available": backend is not None,
            "module_file": getattr(module, "__file__", None),
            "has_create_pufferl": hasattr(backend, "create_pufferl") if backend is not None else False,
            "has_rollouts": hasattr(backend, "rollouts") if backend is not None else False,
            "has_train": hasattr(backend, "train") if backend is not None else False,
            "truth_boundary": "torch --slowly trainer path; not native CUDA trainer path",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def import_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:  # noqa: BLE001 - availability only.
        return False


def load_known_config(path: Path | None, env_name: str) -> dict[str, Any]:
    if path is not None:
        sys.path.insert(0, str(path))
    try:
        pufferl = importlib.import_module("pufferlib.pufferl")
        old_argv = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]]
            args = pufferl.load_config(env_name)
        finally:
            sys.argv = old_argv
    except Exception as exc:  # noqa: BLE001
        return {
            "env_name": env_name,
            "available": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    return {
        "env_name": env_name,
        "available": True,
        "config_env_name": args.get("env_name"),
        "total_agents": (args.get("vec") or {}).get("total_agents"),
        "horizon": (args.get("train") or {}).get("horizon"),
        "total_timesteps": (args.get("train") or {}).get("total_timesteps"),
        "train_gpus": (args.get("train") or {}).get("gpus"),
    }


def run_trainer_smoke(path: Path | None, env_name: str, backend_name: str) -> dict[str, Any]:
    if path is not None:
        sys.path.insert(0, str(path))
    try:
        pufferl = importlib.import_module("pufferlib.pufferl")
        old_argv = sys.argv[:]
        try:
            sys.argv = [sys.argv[0]]
            args = pufferl.load_config(env_name)
        finally:
            sys.argv = old_argv
        native_ready = all(hasattr(pufferl._C, attr) for attr in ("create_pufferl", "rollouts", "train"))
        if backend_name == "auto":
            backend_name = "native" if native_ready else "torch"
        if backend_name == "native" and not native_ready:
            raise RuntimeError("native trainer backend is not available")
        args["slowly"] = backend_name == "torch"
        args["wandb"] = False
        args["vec"]["total_agents"] = 8
        args["vec"]["num_buffers"] = 1
        args["vec"]["num_threads"] = 1
        args["train"]["horizon"] = 4
        args["train"]["minibatch_size"] = 32
        args["train"]["total_timesteps"] = 32
        backend = pufferl._resolve_backend(args)
        puffer = backend.create_pufferl(args)
        try:
            backend.rollouts(puffer)
            backend.train(puffer)
            logs = dict(pufferl.unroll_nested_dict(backend.log(puffer)))
            backend_label = getattr(backend, "__name__", str(backend))
            backend_module = getattr(backend, "__module__", None)
            backend_file = getattr(backend, "__file__", None)
            return {
                "status": "passed",
                "backend": f"{backend_module}.{backend_label}" if backend_module else backend_label,
                "backend_file": backend_file,
                "requested_backend": backend_name,
                "env_name": env_name,
                "global_step": int(puffer.global_step),
                "log_keys": sorted(str(key) for key in logs.keys())[:40],
                "loss_policy_present": "loss/policy_loss" in logs or "loss/policy" in logs,
                "gpu_backend": bool(getattr(backend, "gpu", False)),
                "truth_boundary": (
                    f"tiny PufferLib-owned trainer smoke on built {env_name} env; "
                    "not CONTRA gameplay solve evidence"
                ),
            }
        finally:
            backend.close(puffer)
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "failed",
            "env_name": env_name,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }


def inspect_repo_adapter(path: Path | None) -> dict[str, Any]:
    if path is not None:
        sys.path.insert(0, str(path))
    sys.path.insert(0, str(CODEXLAB))
    try:
        from sp80_puffer_policy import inspect_pufferlib as inspect_adapter

        result = inspect_adapter()
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
    return {"available": bool(result.get("available")), "inspect_pufferlib": result}


def build_result(args: argparse.Namespace) -> dict[str, Any]:
    puffer_path = candidate_pufferlib_path(args.pufferlib_path)
    pufferlib = inspect_pufferlib(puffer_path)
    known_config = None
    if args.known_env:
        known_config = load_known_config(puffer_path, args.known_env)
    trainer_smoke = None
    if args.run_trainer_smoke:
        trainer_smoke = run_trainer_smoke(puffer_path, args.known_env, args.trainer_backend)
    repo_adapter = inspect_repo_adapter(puffer_path)
    checks = {
        "pufferlib_path_exists": bool(puffer_path and puffer_path.exists()),
        "pufferlib_runtime_ready": bool(pufferlib.get("runtime_ready")),
        "pufferlib_trainer_path_ready": bool(pufferlib.get("available")),
        "compiled_env_matches_known_env": (
            True if not args.known_env else pufferlib.get("compiled_env") in (None, args.known_env)
        ),
        "repo_adapter_sees_runtime": bool(repo_adapter.get("available")),
        "known_config_loads": True if known_config is None else bool(known_config.get("available")),
        "trainer_smoke_passed": True if trainer_smoke is None else trainer_smoke.get("status") == "passed",
    }
    result = {
        "schema": CHECK_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "python_executable": sys.executable,
            "cwd": str(ROOT),
        },
        "pufferlib": pufferlib,
        "pufferlib_git": repo_git_metadata(puffer_path),
        "repo_adapter": repo_adapter,
        "known_config": known_config,
        "trainer_smoke": trainer_smoke,
        "boundaries": [
            "This preflight proves local PufferLib runtime readiness, not SP80 native trainer integration.",
            "For Rexus RL runs, PufferLib is ready only when pufferlib.pufferl and pufferlib._C both import.",
            "Top-level import pufferlib alone is intentionally insufficient.",
        ],
        "next_native_training_requirement": (
            "Add/register a PufferLib env_name for each Rexus environment, then train through "
            "python -m pufferlib.pufferl train <env_name> or an equivalent pufferlib.pufferl API call."
        ),
    }
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pufferlib-path", default=None)
    parser.add_argument("--known-env", default="minimal")
    parser.add_argument("--run-trainer-smoke", action="store_true")
    parser.add_argument("--trainer-backend", choices=["auto", "native", "torch"], default="auto")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_result(args)
    if args.out:
        out_path = args.out if args.out.is_absolute() else ROOT / args.out
        write_json(out_path, result)
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
