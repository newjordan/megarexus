# legal_search — mined algorithms, legal consume

This package turns the megarexus research pile into **consumable legal algorithms**:

- **Human may write methods** (search, explore, fit, perceive).
- **Human may not ship answers** (win conditions, source-derived physics, permanent game win banks).
- **Environment is the only judge** of levels / WIN.

## Order: mine first, then train

**Single gate doc:** [`MINING_THEN_TRAIN.md`](MINING_THEN_TRAIN.md)

| Job | When | Entry |
|---|---|---|
| **1. Mining** | Now — until complete | checklist in `MINING_THEN_TRAIN.md` |
| **2. Training** | **Blocked** until `MINING_COMPLETE.json` | `train/train_lora.py` |

Training will exit with code 2 if the mining gate is closed.

```bash
# always allowed: smoke / mine probes
python3 -B legal_search/smoke.py
python3 -B legal_search/mine_probe.py
python3 -B legal_search/mine_probe.py --arcade
python3 -B legal_search/train/collect.py --mine-probe-only --out /tmp/mine_probe.pt --episodes 2

# after MINING_COMPLETE.json exists with complete:true — gated train
python3 -B legal_search/train/train_lora.py --out-dir legal_search/runs/blend_lora_gated_v1 --epochs 20
```

Other docs:

- `MINING_MANIFEST.md` — every system keep/strip
- `docs/MINING_DEBT.md` — **honest remaining mine debt** (read this)
- `docs/MINING_HANDOFF.md` — features for LoRA
- `docs/FULL_MINE_TO_LORA.md` — 25 systems → 392-D → train
- `docs/ADAJEPA_PORT.md` — AdaJEPA plan-act-adapt-replan port
- `docs/TRAIN_PLAN.md` — train plan (post-gate)
- `docs/OVERNIGHT_PATH_B_HANDOFF.md` — Path-B overnight result + next (Arcade Path C)
- `docs/WISHLIST_AMBITIOUS.md` — ranked moonshot wishlist (impact order)
- `../PRIMARY_LANE.md` — specialist vs legal

```python
from legal_search import list_systems, consume_all, LegalAgent
import numpy as np

frame = np.zeros((64, 64), dtype=np.int16)
print(list_systems())
packet = consume_all(frame, game_id="demo")
agent = LegalAgent(game_id="demo")
action = agent.decide(frame, available_actions=[1, 2, 3, 4, 5, 6])
```

## Layout

```text
legal_search/
  contract.py           # rules, bans, truth boundary
  types.py              # Action, Transition, Proposal, LegalPacket
  loop.py               # LegalAgent: explore → search → act → update
  algorithms/           # pure methods (no game answers)
  consume/              # every lab system → legal envelope
  MINING_MANIFEST.md    # full keep/strip table
  smoke.py
```

## What this is / is not

| Is | Is not |
|---|---|
| Mining + legal API for *all* major systems | A guaranteed 6/6 SP80 replacement |
| Goose-style learning by repetition | Spill/droplet answer keys |
| DW **search** without fixture win banks | Ranked `megarexus_dw_directgrid` priors |
| CONTRA/PortBand/Oracle/Ghost as **hypotheses** | Admitted control/goal/solve truth |

Specialists (SP80 DW with priors, ft09 glyph, …) remain real research results. They are **registered as non-default** (`specialist_solvers`) so they are not silently treated as the legal general agent.

## Primary mental model

```text
OLD:  many systems → ad-hoc SP80 specialist glue → scorecard
NEW:  every system → consume(algorithm) → LegalAgent loop → scorecard
```

See `MINING_MANIFEST.md` for the system-by-system strip list.
