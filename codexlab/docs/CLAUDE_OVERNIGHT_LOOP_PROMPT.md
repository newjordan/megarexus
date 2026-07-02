# Claude Overnight Loop Prompt

You are working in `/home/frosty40/rexus`.

Use `codexlab/` as the only active lab area. Put new scratch work under:

```text
codexlab/claude_lab/
```

Do not call the live Qwen/Ollama backend on `127.0.0.1:11434`; it may be in use. Do not run training. Do not claim SP80 was solved or beaten.

## Current Ground Truth

The CodexLab SP80 primitive-control MVP exists here:

```text
codexlab/sp80_base_controller.py
codexlab/sp80_player_harness.py
codexlab/sp80_verify_run.py
codexlab/sp80_sweep_levels.py
codexlab/sp80_control_drift_tracker.py
codexlab/sp80_export_control_dataset.py
codexlab/sp80_debug_report.py
```

Current dataset:

```text
codexlab/datasets/sp80_control_sft_v0/train.jsonl
codexlab/datasets/sp80_control_sft_v0/heldout.jsonl
codexlab/datasets/sp80_control_sft_v0/manifest.json
```

Current debug report:

```text
codexlab/runs/sp80_debug/sp80_debug_report.md
codexlab/runs/sp80_debug/sp80_debug_report.json
```

Use the phrase `calibration slice` for internal control checks. Keep `level_index` only when referring to the SP80 environment API/provenance field.

## Objective

Harden the primitive-control lane for a future Qwen LoRA without adding useless complexity.

Allowed work must do at least one of these:

1. Catch false success claims.
2. Improve offline dataset/debug validation.
3. Prepare a LoRA dry-run scaffold that consumes existing JSONL rows.
4. Improve reports so they are clearer and less likely to imply gameplay solving.

## Hard Boundaries

- No route replay.
- No human win traces.
- No coordinate targets in SFT labels.
- No gameplay solve claims.
- No Qwen/Ollama calls unless the user explicitly reauthorizes it.
- No GPU training unless the user explicitly reauthorizes it.
- Do not touch files outside `codexlab/` unless the user explicitly asks.
- Do not change the deterministic SP80 motor/controller unless a verifier fails and the change is required.

## Loop

Repeat until stopped:

1. Read `codexlab/runs/sp80_debug/sp80_debug_report.md`.
2. Pick exactly one small improvement.
3. Make the change inside `codexlab/` or `codexlab/claude_lab/`.
4. Run only offline checks, for example:

```bash
python -m py_compile codexlab/*.py
python codexlab/sp80_debug_report.py --sweep-dir codexlab/runs/sp80_player_harness_tiny_mlp_levels --dataset-dir codexlab/datasets/sp80_control_sft_v0 --qwen-sweep-dir codexlab/runs/sp80_player_harness_qwen_levels --out-dir codexlab/runs/sp80_debug
```

5. Append a short entry to:

```text
codexlab/claude_lab/NIGHT_LOG.md
```

Each log entry must include:

```text
timestamp
files changed
commands run
result
next safest step
```

## First Good Tasks

Start with one of these:

1. Add an offline dataset leakage checker that confirms heldout calibration slice rows do not appear in train rows.
2. Add a JSON-schema-ish row validator for `sp80_control_sft_v0`.
3. Add a LoRA dry-run launcher stub that only checks dataset formatting and prints the intended training command, but does not train.
4. Add a report section that distinguishes primitive-boundary Qwen success from gameplay solving.

End each cycle with a concise status and no capability claims beyond the artifacts you checked.
