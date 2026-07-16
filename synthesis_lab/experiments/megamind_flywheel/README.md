# MegaMind training flywheel

Keeps spinning multi-game level-group raises with a Weco-style **keep only if better** private gate.

## Control

```bash
# start (already running if flywheel.pid alive)
python -u -B synthesis_lab/orchestration/rdl/brain/train_flywheel.py \
  --root synthesis_lab/experiments/megamind_flywheel \
  --device cuda --max-cycles 0

# stop
touch synthesis_lab/experiments/megamind_flywheel/STOP

# status
cat synthesis_lab/experiments/megamind_flywheel/HEARTBEAT.json
tail -20 synthesis_lab/experiments/megamind_flywheel/journal.jsonl
```

## Layout

| Path | Meaning |
|---|---|
| `best/megamind_best.pt` | Incumbent checkpoint |
| `best/metrics.json` | Incumbent metrics |
| `cycles/cycle_*` | Every outer step (kept or rejected) |
| `journal.jsonl` | Append-only cycle log |
| `HEARTBEAT.json` | Live status |
| `STOP` | Create to halt after current cycle |

## Doctrine: wide · slow · steady

- **Wide:** all 24 games every cohort (interleaved), never one-game grind  
- **Slow:** progressive L0–L1 depth; high bands only as light scouts  
- **Steady:** keep-only-if-better private gate; no thrash restarts  

Schedule (on disk; live process keeps its loaded schedule until restart):  
mostly L0–L1 raises → gentle L0–L3 / broad joints → rare L2+ scouts.  
Each full pass slightly increases steps.
