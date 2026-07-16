# megarexus — one brain

**Primary:** dual-hemisphere **MegaMind**  
Internal dreams · External judges · Central LoRA commits · operator garden pad.

```text
brain/                          # public import surface
synthesis_lab/orchestration/rdl/brain/   # implementation
  operators/                    # Loom · Rexus · pattern growth · Weco RSI · …
  train_flywheel.py             # multi-game level-group flywheel
  STATUS.md                     # pad metrics
docs/PRIMARY.md                 # this week's single source of truth
```

## Quick start

```bash
# garden smoke (all idea seeds on the pad)
CUDA_VISIBLE_DEVICES= python -B synthesis_lab/orchestration/rdl/brain/smoke_all_ideas.py

# multi-game train flywheel (24 games, keep-if-better)
python -u -B synthesis_lab/orchestration/rdl/brain/train_flywheel.py \
  --root synthesis_lab/experiments/megamind_flywheel \
  --device cuda --max-cycles 0

# stop flywheel
touch synthesis_lab/experiments/megamind_flywheel/STOP
```

```python
from brain import MegaMind
mm = MegaMind(device="cpu")
state = mm.observe(frame)  # frame: HxW color indices
```

## Layout (after cleanup)

| Path | Role |
|---|---|
| `brain/` | One-stop import |
| `synthesis_lab/orchestration/rdl/brain/` | Brain implementation |
| `legal_search/` | Mined algorithms **consumed by** operators (source only) |
| `codexlab/contra_head/` | CONTRA backend for full view fanout |
| `synthesis_lab/experiments/other_games/environment_files/` | 24-game suite |
| `environment_files/` | Extra local envs (sp80/ft09) |
| `docs/` | PRIMARY + standards |
| `notes/` | Doctrine sweeps (archive/ for old handoffs) |
| `quarantine/` | Illegal / banned lineages (if present) |

## Not primary

CNN experiment stacks, scorecard specialists, flat gate scripts, dated handoffs — lab/archive only. See `docs/PRIMARY.md`.

## License / honesty

Env grades only. Spill models, fixture win banks, and human-route labels are banned from train truth.
