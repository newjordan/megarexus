# REGIME L2-FRONTIER P1 RESULT — 2026-07-06

**Variants v2 built; S2-MS gate 3-of-4 PASS. Multi-source sim is a PERFECT-RECALL filter
(recall 1.000) but over-calls floods-as-wins (precision 0.675 < 0.70) — the residual
under-emission surfacing. Decision escalated.** Follows `TRAINING_REGIME_20260706_L2_FRONTIER.md`
§2 P1 / §4. Artifact: `sim3d/fixtures/s2ms_result.json`.

```
honesty_audit: exit 0 — 97/46/6, active 85/23 (no new flags)   t0-five untouched: YES
S0 after change: PASS (single-source byte-identical)
Gate criterion (verbatim §2 P1): win-sign >=0.95, recall >=0.95, precision >=0.70, single-source S2 unchanged
  -> win-sign 0.977 PASS | recall 1.000 PASS | precision 0.675 FAIL | single-source UNCHANGED PASS  => 3/4
Corpus: 4,005 multi-source pours (ms2 n=2 1,439 + ms3 n=3 2,566), 189 env wins.
```

## What changed (all no-imitation-legal; single-source provably untouched)
- **`sp80_variants.py`:** `REDIR_TAG "liolfvkveqg" → "tuvkdkhdokr"` (the old value was the PAINT
  DROPLET — v1 variants never held a real redirect). Added a **multi-source family** (`n_sources`
  ≥2, sources anchored above target columns ±1 jitter so boards are winnable-but-routed — pure
  random multi-source is a needle: 0 wins, no positive class to gate on) and a **true-redirect
  family** (`with_redirect`, real `tuvkdkhdokr`). New branches draw `rng` only when engaged.
- **`raycast.py`:** `HydroParams.floor_spill_multisource=True` (default) — the floor-gated ledge
  spill now runs on multi-source too. Safe post-P0: the enclosed-entry latch credits fills on
  ENTRY, so routing overflow to the floor no longer eats mouth fills (the reason it was
  single-source-only).

## Single-source S2 UNCHANGED — proven bit-identical
Post-edit `s2_ssverify_20260706.npz` vs pre-edit `s2_p0verify_20260706.npz`: **pre-frames
identical, sim_margin max|Δ|=0**, n=1187, wins=14, win-sign 0.994945, AUC 0.997442 — identical.
The REDIR_TAG fix touches only `tpl["redir"]` (never placed on the single-source path);
`floor_spill_multisource` gates on `len(spouts)>1`. GATE P0 re-verified with the flag on:
S0 PASS, L2 fill-agreement 0.935 (unchanged), 0 false-positive wins/200, tick-match 12–14 on 6/6.

## The S2-MS result, honestly
| metric | baseline (flag off) | **adopted (flag on)** | gate |
|---|---|---|---|
| win-sign | 0.967 | **0.977** | ≥0.95 ✓ |
| **recall** | 1.000 | **1.000** | ≥0.95 ✓✓ |
| **precision** | 0.587 | **0.675** | ≥0.70 ✗ |
| FP (sim-win, env-loss) | 133 | **91** | |

- **Recall 1.000**: the multi-source sim never discards a true winner — a perfect FILTER.
- **Precision 0.675 (narrow miss)**: **129/133 (97%) of the FPs are FLOODS** — the env floods
  the floor (real_lava≈254) while the sim declares a clean win. True winners never flood
  (TP lava=0). Cause = the **residual ~15% under-emission** (P0 fix (c), still open): the sim
  emits ~46 macro cells vs real ~54, so it under-produces the overflow that reaches the floor →
  misses the flood → false clean-win. `floor_spill_multisource` routes the overflow the sim DOES
  have (0.587→0.675) but cannot invent the missing ~8 cells.

## Decision (escalated per §2 discipline — do not grind)
Precision misses the pre-registered bar; the remaining lever is a DISTINCT sub-cause (emission
volume, not routing), so per the P0 stop-rule this is written up, not stacked. Two honest paths:
- **(a) Emission calibration (P0 fix c):** measure real per-source L2 emission, fix the sim's
  early emission stop so it emits ~54 and detects the floods → should close 0.675→≥0.70.
- **(b) Proceed to P2** treating the sim as a RECALL-1.0 filter: env grades top-K anyway, so
  precision 0.675 just means ~1.5× candidates poured. The needle-find (argmax) is what matters
  for the L2 unlock, and the sim never hides a real winner.

## Headline (unchanged)
**2 levels in-budget — L0 CNN-alone, L1 sim-guided SOLVED@1, robust.** New tooling (agent-
generated): v2 families in `sp80_variants.py`, `s2_generalize.py --n-sources/--with-redirect`.
