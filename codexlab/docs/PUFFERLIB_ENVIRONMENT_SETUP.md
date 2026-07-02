# PufferLib Environment Setup

This is the repo-local setup contract for PufferLib-backed Rexus RL environments.

## Readiness Boundary

PufferLib is considered ready only when all of these pass:

- `pufferlib` imports.
- `pufferlib._C` imports.
- `pufferlib.pufferl` imports.
- Either the native CUDA trainer methods are present, or the torch `--slowly`
  trainer path is present for the CPU build.
- The repo adapter in `codexlab/sp80_puffer_policy.py` sees the same runtime.

A top-level `import pufferlib` is not enough. On this host, earlier PyPI builds
failed by linking `raylib-5.5_linux_amd64` into an `aarch64` build, and a source
checkout could import top-level while the native backend was still missing.

## Local Environment

The persistent local checkout should live here:

```bash
/home/frosty40/rexus/.external/pufferlib-4.0-newjordan
```

The previous spike checkout may also exist here:

```bash
/tmp/pufferlib-4.0-newjordan
```

`.external/` is intentionally git-ignored. Do not vendor PufferLib into this
repository history.

Before PufferLib-backed RL work:

```bash
source codexlab/pufferlib_env.sh
```

Then run the hard readiness gate:

```bash
PYTHONPATH=codexlab:$PUFFERLIB_PATH /home/frosty40/miniconda3/bin/python3 -B codexlab/pufferlib_env_check.py
```

For a machine-checkable report:

```bash
PYTHONPATH=codexlab:$PUFFERLIB_PATH /home/frosty40/miniconda3/bin/python3 -B codexlab/pufferlib_env_check.py \
  --out codexlab/runs/_system_run/pufferlib_env_check.json
```

To also run a tiny PufferLib-owned trainer smoke on the built `minimal` env:

```bash
PYTHONPATH=codexlab:$PUFFERLIB_PATH /home/frosty40/miniconda3/bin/python3 -B codexlab/pufferlib_env_check.py \
  --run-trainer-smoke \
  --trainer-backend native \
  --out codexlab/runs/_system_run/pufferlib_env_check.json
```

## Existing PufferLib 4.0 Host Patch

The host patch is recorded at:

```bash
codexlab/docs/pufferlib_4_0_arm64_build.patch
```

The known-good source checkout was built from `newjordan/PufferLib` branch
`4.0`, commit `9836f0d2e78889c1aaf189c04d161b6fc61a9386`, with:

```bash
NVCC_ARCH=sm_120 bash ./build.sh minimal
```

Expected native backend shape on this host:

```text
pufferlib/_C.cpython-313-aarch64-linux-gnu.so
```

Current local build note: the `minimal` build is a native CUDA trainer build.
It exposes `_C.create_pufferl`, `_C.rollouts`, and `_C.train`; the recorded
preflight reports `native_cuda_trainer_ready=true`.

The native build uses bf16 precision (`precision_bytes=2`). That is fine for
the native CUDA backend, but it disables the torch `--slowly` path unless the
checkout is rebuilt with `--float`.

## Rexus RL Environment Rule

For each new RL environment:

1. Add a condition file under `codexlab/conditions/`.
2. Add or reuse a controlled runner that records action provenance, packets,
   brain calls, raw frames when visual, and a readable review surface.
3. Keep PufferLib training claims separate from PufferLib-gated repo harness
   claims.
4. Native PufferLib training requires a real PufferLib `env_name` and execution
   through `python -m pufferlib.pufferl train <env_name>` or an equivalent
   `pufferlib.pufferl` API call.
5. Evaluate any learned/exported policy back through the controlled
   CONTRA/SP80 harness before presenting it as gameplay evidence.

Current SP80 paintflow status: the repo has a PufferLib-gated learned primitive
policy and a full controlled CONTRA level-1 report. It is still not native
PufferLib environment training.

## Native Rexus SP80 Paintflow Env

The first repo-owned native PufferLib env is:

```text
codexlab/pufferlib_envs/rexus_sp80_paintflow/
```

Install it into the active PufferLib checkout:

```bash
PYTHONPATH=codexlab /home/frosty40/miniconda3/bin/python3 -B codexlab/install_pufferlib_envs.py
```

Build it as the active native CUDA trainer env:

```bash
cd .external/pufferlib-4.0-newjordan
NVCC_ARCH=sm_120 bash ./build.sh rexus_sp80_paintflow
```

Verify the native env:

```bash
/home/frosty40/miniconda3/bin/python3 -B codexlab/pufferlib_env_check.py \
  --known-env rexus_sp80_paintflow \
  --run-trainer-smoke \
  --trainer-backend native \
  --out codexlab/runs/_system_run/rexus_sp80_paintflow_native_env_v0/env_check.json
```

Boundary: `rexus_sp80_paintflow` is a native C/PufferLib level-1 paintflow
control abstraction using the learned clean pose table. It is not a full Python
SP80 engine transplant, and trained policies still need controlled CONTRA/SP80
harness evaluation before any gameplay claim.
