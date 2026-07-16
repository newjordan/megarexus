# Mining Handoff — systems → LoRA features

**Schema:** `legal_search.mining_handoff.v1`  
**Companion:** `MINING_MANIFEST.md`, `docs/TRAIN_PLAN.md`

## Job

Reduce every lab system to a **legal consumable**, then **encode** it into a fixed
feature vector for the blend LoRA. Do not re-import banned lineages.

## Feature groups (F dimensions)

| Group | Dim | Source systems | Content |
|---|---:|---|---|
| hist | 16 | contra_vision | color histogram |
| edge | 1 | contra / symmetry | mean edge density |
| sym | 2 | symmetry | H/V symmetry scores |
| components | 40 | contra, portband | top-8 comps × (color/64, area_n, cx, cy, aspect) |
| ports | 8 | portband | mean port extent per side + n_align |
| delta | 8 | deltaband | recent delta stats |
| explore | 6 | goose, exploreband | action change rates (1-5) + click rate |
| memory | 4 | ghost, episode_memory | n_trans, n_success, n_hints, levels |
| search | 4 | dreamweaver | best program score, len, n_props, mode |
| loom | 6 | loom_lewm | affordance ranks for actions 1-6 |
| flags | 8 | specialist, quarantine, tabula | one-hots / guards |
| **total** | **~103** | | see `features.py` `FEATURE_DIM` |

## Proposal systems → weak policy labels

When collecting, if the step used:

- `explore.change_by_repetition` → label = that action
- `search.program_hill_climb` first action → label = that action
- episodic memory prior → label = that action

Never label from fixture win banks.

## Strip list (must stay zero in features)

- fixture genome/direct priors  
- spill_model / dropletsim  
- human route shards  
- admitted truth flags (always 0)  
- SP80 color role constants as hard labels  

## Registration checklist (new system)

1. Add `ConsumeAdapter` under `legal_search/consume/`
2. Register in `registry.py`
3. Extend `features.py` encoder (document dim change)
4. Update `MINING_MANIFEST.md`
5. Re-run `legal_search/smoke.py` + `legal_search/mine_probe.py`
6. Do **not** open the train gate until `MINING_THEN_TRAIN.md` A–E are green

## JOB 1 probe (mine, do not train)

```bash
python3 -B legal_search/smoke.py
python3 -B legal_search/mine_probe.py
python3 -B legal_search/mine_probe.py --arcade   # if offline Arcade available
```

`mine_probe.py` checks: 25/25 features, depth on PortBand/CONTRA/Delta/Ghost/DW/Goose,
pure Goose rebuild (no double-count), private-game wipe, legal import audit, manifest sync.
It never writes `MINING_COMPLETE.json`.

## Primary docs to keep

| Path | Role |
|---|---|
| `legal_search/MINING_MANIFEST.md` | full keep/strip |
| `legal_search/README.md` | package entry |
| `legal_search/docs/TRAIN_PLAN.md` | train plan |
| `legal_search/docs/MINING_HANDOFF.md` | this file |
| `PRIMARY_LANE.md` | specialist vs legal split |
