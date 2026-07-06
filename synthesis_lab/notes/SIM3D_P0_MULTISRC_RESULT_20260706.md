# REGIME L2-FRONTIER P0 RESULT — 2026-07-06

**Enclosed-entry mouth fill: multi-source fill fidelity closed. GATE P0 PASS (all four).**
Pick-up doc: `TRAINING_REGIME_20260706_L2_FRONTIER.md` §2 P0 / §4. This note follows its
report template. Artifact: `sim3d/fixtures/p0_multisrc_fidelity_result.json`.

```
honesty_audit: exit 0 — total 97/46/6, active 85/23 (no new flags)   t0-five untouched: YES
S0 after change: PASS (stream, winner +1.000 vs FPs [-1,-1,-1,-1]) — byte-identical to baseline
Gate criterion (verbatim §2/§4):
  P0.1 single-source PRISTINE (S0 byte-identical + full S2 unchanged)         -> PASS
  P0.2 multi-source fill agreement >= 0.90 on the 200-arr L2 corpus           -> PASS (0.935)
  P0.3 tick-match first-divergence >= 10 ticks on >= 3 real L2 pours          -> PASS (12–14 on 6/6)
  P0.4 the s4_l2_multisrc FP (sim +1 -> env loss) must score <= 0             -> PASS (+0.000; 0 FP-wins/200)
Corpus/condition: p0_l2_diag_latch.npz (200 env-graded L2 arr), p0_l2_tickmatch.json,
  s2 merged 11,810 saved single-source pours (bit-identical re-score), s2 fresh 1,187 pours.
```

## The bug (Correction B), reproduced precisely
The night's Correction B said: the sim under-reports L2 fills — where the env fills 1–2
targets, the settled-only count reports 0. Reproduced with `p0_probe.py` on 200 real
env-graded L2 arrangements: per-target **fill agreement 0.627** (0.304 on env-filled
targets); **224 Correction-B cells** (env fills, sim < 0.85). Root cause found by tracing:
SP80 U-mouths are **single macro cells (cap 1.0) enclosed on 3 sides** (glyph
below+left+right, open above). The game (source **L775**) recolours the WHOLE target to 13
the **tick a droplet ENTERS that enclosed cell** — a transient, permanent event — NOT when
the mouth settles to capacity. On L2 all three sources (x=1,9,14) are laterally offset from
all mouths (x=2,7,13), so fills arrive via slow bar-routing, often **after the paint plateau**
— and `pertarget_labels._outcome_end` (built for the env's post-win reset) then **trims the
late fill frames**. The settled proxy also loses fills that route in below the 0.85 threshold.

## The fix (faithful to the game, not guessed)
`raycast.py`: an **enclosed-entry latch**. Precompute per target the mouth cells whose
flow-forward neighbour and both perpendicular neighbours are that target's own glyph (L775's
`owyqnbmyij is faksfmaezd is lxmalbxucn`). Each tick, any fluid at an enclosed cell latches
the target filled (`mouth_fill -> 1.0`) permanently. Because it fires on the **transient
entry** (before the settle), the fill lands inside the outcome window and survives the trim.
**Gated to multi-source** (`len(spouts) > 1`): single-source keeps its byte-identical settled
path (single-cell mouths settle the tick the droplet lands — the proxy is exact there),
mirroring the existing `floor_spill_active` single-source gate. GATE P0.1 stays pristine
**by construction**.

## Numbers (before -> after)
| metric | before | after |
|---|---|---|
| L2 per-target fill agreement (overall) | 0.627 | **0.935** |
| … on env-filled targets | 0.304 | **0.879** |
| Correction-B cells (of 600 pairs) | 224 | **39** |
| S2 single-source sim_margin (11,810 frames) | — | **max|Δ| = 0.00e+00** (bit-identical) |
| S2 merged precision / recall / AUC | 0.807 / 0.992 / 0.9983 | **unchanged (identical)** |
| tick-match first-divergence (6 L2 pours) | (was FAIL, flooded early) | **12,13,13,14,14,14** |
| false-positive wins on 200 L2 arr | (n/a) | **0** |

## Verified-precise, and the honest residual
- **fp = 0**: fluid reaches a mouth cell only when the env fills it — the latch never
  over-declares (the exact property the reverted "count settled+falling everywhere"
  heuristic lacked: it over-filled 36/200 and cost single-source AUC). This is why P0.4
  passes with 0 false-positive wins.
- **Two failure modes were separated.** Of the original 224 misses, 193 were fill-COUNTING
  (fluid reached the mouth; latch fixes) and 31 were flow-ROUTING (fluid never reached:
  peak_occ=0). The latch closes the counting mode; **39 routing misses remain**.
- **Residual multi-source infidelity (not a P0 blocker, next target):** the sim still
  **under-routes ~15%** — plateaus at ~46 macro cells vs real ~54 and terminates ~8 frames
  early (the doc's "sim emitted 42 and flooded early", now 46). Genuine tick divergence at
  ~tick 12. This is a ROUTING/emission-volume gap, distinct from the fill-counting gap P0
  fixed; it is the natural P1/P2 fidelity target.

## Headline (unchanged — P0 is fidelity, not a new level)
**2 levels in-budget — L0 CNN-alone, L1 sim-guided SOLVED@1, robust 5/5 seeds.** L2 remains
unsolved; its solvability is still UNKNOWN (not walled): P0 removes the "wrong-gradient"
confound (Correction B) so P2's constructive search now runs on a fill-faithful sim.

## Falsified this phase
Nothing new falsified. The reverted heuristics (blanket ledge_spill, count settled+falling)
stay falsified; the enclosed-entry latch is the precise rule they approximated.

## Next per §3
**P1** — `sp80_variants.py` REDIR_TAG fix (`tuvkdkhdokr`) + multi-source & true-redirect
families + the S2-MS gate (win-sign ≥0.95 / recall ≥0.95 / precision ≥0.70, single-source
S2 unchanged). P1's multi-source variant harvest is also where the **residual ~15%
under-route** and the **39 routing misses** get their own corpus to close (P0's stop-rule:
don't stack heuristics on the sim — the routing gap is a measurement for P1's variants, then
P2's proposer). New tooling added, all agent-generated, no imitation: `p0_probe.py`,
`l2_tickmatch.py`.
