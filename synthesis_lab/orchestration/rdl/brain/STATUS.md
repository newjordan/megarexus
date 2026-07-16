# MEGAMIND — base model garden (ALL ideas planted)

**This is the one brain.** Repo primary: [`docs/PRIMARY.md`](../../../../docs/PRIMARY.md) · public import: `brain/`.

**Doctrine:** the model is ground. Testing rises what works.  
**Nothing important is left out of the base pad.**  
**Geometry = dynamic multi-variate pattern growth — not SP80 train focus.**  
**Train = multi-game level-group flywheel + private keep-if-better.**

## Smoke (authoritative)

```bash
CUDA_VISIBLE_DEVICES= /home/frosty40/miniconda3/bin/python3 -B \
  synthesis_lab/orchestration/rdl/brain/smoke_all_ideas.py
# ALL IDEAS IN BASE MODEL — garden complete
# (Loom + Rexus + Primitives + PatternGrowth + WecoRSI)
```

| Metric | Value |
|---|---|
| OP_FAMILIES | **231** |
| EXTRA_FEAT_DIM | **3696** |
| Loom imagination IDs | **30/30** |
| Rexus catalog IDs | **60/60** |
| Primitive design IDs | **32/32** |
| Pattern growth IDs | **40/40** |
| Weco AIDE² RSI IDs | **32/32** |
| Train path | E→I→C + ada_online + auto-fuse |
| Default transitions | multi-variate pattern growth (not SP80) |

## Catalogs on the wire

1. **Physics / map** — body, rays, signed multi-variate field  
2. **Legal mine** — 392-D fullmine, ports, symmetry, geo, saliency, statecon, …  
3. **Full CONTRA** — `run_contra_view` fanout (20 layers)  
4. **Crucible full** — `mine_crucible`  
5. **DeltaBand full** — `build_deltaband_observation`  
6. **Transcript macros** — proposal logit priors  
7. **ALL Loom imagination L1–L30** — `operators/loom_imagination.py`  
8. **ALL Rexus R1–R60** — `operators/rexus_concepts.py`  
9. **Basic primitive game designs P1–P32** — design atoms, not Atari ports  
10. **Dynamic multi-variate pattern growth G1–G40** —  
    `operators/pattern_growth.py`  
11. **Weco AIDE² RSI W1–W32** —  
    `synthesis_lab/notes/WECO_AIDE2_RSI_SWEEP_20260715.md`  
    + `operators/weco_rsi.py`  
    bi-level outer/inner · RSI ladder L0–L3 · public/private · fixed budget ·  
    task hetero (24-game) · anti-hack layers · bandit lineage · fork-on-stall ·  
    context compress · eval repair · reject fancy under budget

## Geometry doctrine (2026-07-15)

- **No SP80 train focus.** Arcade SP80 is optional legacy only.  
- All geometric systems measure **patterns that grow across multiple variables**  
  (colors, density, topology, scale, orientation, rate, adjacency, …).  
- `GeoOp` / `FieldOp` expanded for multivariate cues; full G-catalog is the pad.  
- Default `collect_real` = growing multi-color synthetic transitions.

## Training flywheel (keep spinning)

**Not** one-game forever. **Not** SP80 focus.  
Bi-level outer gate (Weco keep-only-if-better) + multi-game level groups.

```bash
# FLYWHEEL — runs forever until STOP file
python -u -B synthesis_lab/orchestration/rdl/brain/train_flywheel.py \
  --root synthesis_lab/experiments/megamind_flywheel \
  --device cuda --max-cycles 0

# stop
touch synthesis_lab/experiments/megamind_flywheel/STOP

# live status
cat synthesis_lab/experiments/megamind_flywheel/HEARTBEAT.json
tail synthesis_lab/experiments/megamind_flywheel/journal.jsonl
```

| Piece | Role |
|---|---|
| `train_flywheel.py` | **Outer loop** — schedule groups, train, private gate, promote best |
| `rsi_outer_gate.py` / flywheel keep | Keep only if private multi-game score improves |
| `suite_games.py` | Discover **24** public games |
| `collect_level_group.py` | Level-band transitions, round-robin all games |
| `train_level_groups.py` | Single cohort trainer (used by flywheel) |

Schedule rotates: L0–L1 → deeper → L2–L3 → joint → L4–L5 → L6–L7 → …  
Each full pass bumps steps (progressive depth).

## How growth works

- Every channel → fixed 16-D score slot → **3696-D** pad → central LoRA  
- Level-group multi-game training **raises** robust seeds  
- Outer gate: `rsi_outer_gate.py` keep-only-if-better on **private** multi-game robust frac  
- Unused seeds stay near-zero weight without being **deleted from the base**  
- **Honesty:** Weco Level-1 RSI not claimed for megarexus; seeds + gates only
