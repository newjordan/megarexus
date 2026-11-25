# megarexus

<p align="center">
  <img src="docs/assets/megarexus-hero.jpg" alt="megarexus dual-hemisphere MegaMind" width="100%" />
</p>

**MegaMind** is the one brain in this repo. An internal hemisphere dreams a next board and proposes an action. An external hemisphere judges that dream against the world. A central cortex with LoRA on the policy, trust, and mode heads commits the plan. Truth flags stay false until an environment grade says otherwise.

Doctrine and the lab map live in [`docs/PRIMARY.md`](docs/PRIMARY.md) and [`docs/LAB.md`](docs/LAB.md).

## Setup

Python 3.10 or newer. This pass ran on Python 3.12.3 with NumPy 1.26.4 and PyTorch 2.15.0+cpu. `torch.cuda.is_available()` was false here; a CUDA build is optional and is not required for the smoke or the tests.

Install a CPU wheel, then the package. This does not download model weights. `WorldModelPlug` loads a checkpoint only when a local path or `MEGAMIND_WORLD_CKPT` already points at one.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

`pip install -e ".[dev]"` on this machine installed `megarexus` 0.1.0 and the `megamind-flywheel` and `megamind-gate` commands. Dev extra is pytest 8 or newer. Runtime dependencies are `numpy>=1.26` and `torch>=2.2`.

## Use the brain

Frames are color-index boards: a `[H, W]` or `[B, H, W]` integer tensor (palette 16). A channel-last `H×W×1` or `H×W×3` image is accepted when both spatial sides are larger than the channel count. Actions are integers in `[0, n_actions)`. A single action is repeated across a frame batch. Pass `available_actions` to mask the rest; an empty list masks every action.

```python
from brain import MegaMind

mind = MegaMind(emb=128, n_actions=8, device="cpu")
state = mind.observe(frame, available_actions=[1, 2, 3, 4, 5, 6])
state = mind.judge_plan(frame, action=5, state=state)
logits = mind.act_logits(state)
logs = mind.train_transition(frame, action=5, frame_next=frame_next)
```

`observe` maps the board and proposes. `judge_plan` scores the dream. `act_logits` is the masked policy. `train_transition` runs one oscillation step (external, then internal, then central) and an online LoRA update. `BrainState.to_dict()` is JSON-safe: private keys that start with `_` are dropped, and non-finite floats become `null`.

### Smoke

Two checks, both measured on this CPU machine. Neither downloads weights.

```bash
python -B synthesis_lab/orchestration/rdl/brain/smoke_megamind.py
python -B synthesis_lab/orchestration/rdl/brain/smoke_all_ideas.py
```

`smoke_megamind.py` (seed 0, one 64×64 board, one train cycle) finished in 11.79s and printed `PASS  observe, judge, act, and one train cycle`. The random board stayed untrusted. The perception gate reported `spouts`, `mouths`, and `mouth_rays` rather than raising. Operator statuses on that board were `degraded=4` and `ok=227`. Truth flags stayed all false. Status lines are flushed, so progress shows even when stdout is not a terminal.

`smoke_all_ideas.py` finished in 2.69s and printed `ALL IDEAS IN BASE MODEL`. Catalog counts from that run: 231 operator families, pad width 3696, Loom 30/30 ok, Rexus 60/60 ok, primitives 31 ok and 1 degraded (`P31_ocean_library_present`, ocean library absent), pattern growth 40/40 ok, Weco RSI 32/32 ok. Coverage was 223 of 231 operators ok.

The operator pad is `OP_SLOT * len(OP_FAMILIES)` = 16 × 231 = 3696 features. Central cortex pads to that width.

## Train and gate

The flywheel writes JSON events to stdout and a short status line to stderr. `--device` defaults to `auto` (CUDA when `torch.cuda.is_available()` is true, otherwise CPU). `--max-cycles 0` runs until a `STOP` file appears in the flywheel root. The process id file is removed when the process exits.

```bash
megamind-flywheel \
  --root synthesis_lab/experiments/megamind_flywheel \
  --device auto \
  --max-cycles 0

# stop a running flywheel
touch synthesis_lab/experiments/megamind_flywheel/STOP
```

`megamind-gate CANDIDATE.json [INCUMBENT.json]` prints `KEEP` or `REJECT` on stderr and the decision JSON on stdout. A missing metrics file exits 1. A reject decision still exits 0. Non-finite public loss is written as `null` so the report is valid JSON. The default private-score floor is 0.35.

The private score counts each level-up once. Per-game rows are used when they exist; the cohort total is not added on top of them. The games-up fraction uses the number of games in the suite, not a fixed 24.

## Games in this checkout

`suite_24()` discovers `prefix/ident` directories that contain a `.py` file under `synthesis_lab/experiments/other_games/environment_files`. The name is historical. This checkout has 2 games:

- `ft09-0d8bbf25`
- `ls20-9607627b`

Training notes aimed at 24 games. `keep_only_if_better(..., require_all_games=True)` still rejects a candidate whose suite reports fewer than 24 games. The default gate does not set that flag. `environment_files/sp80/589a99af` is ARC Prize game source and is not part of the discovered suite.

## Tests and CI

```bash
python -m pytest
```

On this machine that was **72 passed in 21.13s** (Python 3.12.3, PyTorch 2.15.0+cpu). The suite covers frame layout, action alignment, the operator-pad width, LoRA rank, phase-C and Ada trust on a single row, private-score accounting, JSON brain state, the flywheel and gate CLIs, suite discovery, symmetry and cavity fitting on a noisy 64×64 board, spout placement on a bottom reservoir, and a scan of tracked text for private keys and tokens.

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs the same CPU install and `python -m pytest` on `ubuntu-latest` with Python 3.12. This pass did not run that workflow on GitHub.

## Layout

```text
brain/                                         public import (from brain import MegaMind)
synthesis_lab/orchestration/rdl/brain/         implementation, flywheel, gate, smokes
synthesis_lab/orchestration/sim3d/             pour-scene perception gate
codexlab/contra_head/                          CONTRA view
legal_search/algorithms/                       operator backends
synthesis_lab/experiments/other_games/environment_files/   discovered games
stochastic_goose/                              historical agent; not the brain
docs/PRIMARY.md                                doctrine
docs/LAB.md                                    lanes that are not the primary
```

`stochastic_goose` pins `numpy>=1.26`, `tensorboard>=2.14`, and `torch>=2.2`. Its Makefile stops if the `ARC-AGI-3-Agents` directory is missing. This checkout does not include that directory.

## Licenses

No root license was added. Two licenses already in the tree stay as they are:

- [`architecture/upstream_le_wm/LICENSE`](architecture/upstream_le_wm/LICENSE) — MIT, Copyright (c) 2026 Lucas Maes
- `environment_files/sp80/589a99af/sp80.py` — MIT, Copyright (c) 2026 ARC Prize Foundation

Checkpoints (`*.pt`, `*.pth`, `*.ckpt`), flywheel journals, and `.arc_api_key` are gitignored. Arcade calls in the lab use the public offline key `anonymous`.
