# SIM3D plan — autonomous-loop status & handoff — 2026-07-06

Consolidated result of the `/loop` run over `SIM3D_EXECUTION_PLAN_20260705.md`
("complete this plan and train the model efficiently"). honesty_audit exit 0
throughout (no new shards/ckpts/imitation lineage; counts unchanged 97/46/6).
Nothing committed (working-tree changes only). t0-five untouched.

## Plan status: 5 of 6 stages DONE; core goal achieved and hardened
| stage | status | evidence |
|---|---|---|
| S0 decisive | ✅ PASS | stream winner +1.0 vs FPs −1.0 (re-verified after every change) |
| **S1 fidelity** | ✅ **iter2 ADOPTED — the breakthrough** | floor-gated ledge spill; full S2 precision **0.736→0.807**, recall 0.992 held, argmax nontrivial **0/2→1/1** |
| S2 generalization | ✅ GATE PASS | 11,810 pours / 3 seed sets; win-sign 0.996, AUC 0.998; recall 0.99, 99.6% loss-reject |
| S3 in-budget L1 | ✅ **SOLVED@1, 5/5 seed-robust** | seeds {1234,7,42,101,2026} all SOLVED@1 |
| S4 recon L2–L5 | ✅ recon done | 6 levels; L0–2 extract clean; L3–5 need extractor work; **L2 = PROPOSER-blocked** |
| S6 integration | ✅ lane wired | `imagine_search_gate --sim3d`; L0 gate tp=2 fp=0 fn=0 |
| S5 Lane P | ⏸️ DEFERRED | particle fit >5min, S0 particle FAILs; lowest-priority fallback (Lane R works) |

**Headline holds and is now hardened: 2 levels in-budget (L0 CNN-alone, L1 sim-guided),
L1 robust across 5 seeds.** The sim-fidelity lane's core goal — a faithful pour simulator
that breaks the argmax wall v38–v56 could not — is DONE.

## What this loop actually moved
1. **S2 (the overfit antidote):** proved the hand-coded sim generalizes to 1,000 unseen
   boards (recall 0.99, no overfit collapse) — but exposed the residual overshoot-FP wall.
2. **S1 iter2 (the win):** killed most of that wall. The trick was a DESTINATION-gated
   spill (`drops_to_floor`) — route overshoot to the floor without accelerating mouth-fills.
   Iter1's blanket version regressed (0.736→0.447); the gate flipped it to 0.807. Adopted.
3. **Proved S1 is the bottleneck, then that L2 is NOT:** after iter2, L2's fill-and-flood
   FPs score correctly, yet L2 still doesn't solve — because **0 winners exist in 40 real
   L2 pours**. L2 is PROPOSER-blocked, not ranker-blocked.
4. Wired the sim into the shared gate (S6) and hardened L1 (5/5 seeds).

## The honest ceiling — why the loop stopped here (not busywork)
The two genuinely high-value next levers are BOTH outside this plan's scope and need a
strategic call from you:
- **PROPOSER (roadmap Stage C, portband-seeded CEM)** — the actual blocker for L2–L5
  (winning arrangements are too rare for random/hill-climb search to find). The sim ranker
  is ready to serve as its fitness; it just needs building. THIS is the path to more levels.
- **Honest CNN training** — note the trap: a STATIC CNN predicting outcome from the
  pre-pour frame is a KNOWN DEAD END (STAGE_PT v55/v56, handoff §4). Viable training is
  the amortized PROPOSER (CNN generates arrangements, sim ranks, env grades) or the
  superseded neural-CA — both are real efforts, not quick wins.
- S5 (Lane P) and S1 iter3 (remaining ~28 FPs) are low-value polish.

**Recommended next direction (your call):** build the Stage-C proposer (portband-seeded)
using the now-validated sim margin as fitness — it's the one lever that moves the headline
past 2 levels. Say the word and I'll scope it.

## L2+ frontier — DEFINITIVE diagnosis (2026-07-06, resolves handoff §6)
Pushed on the proposer (task #4) and characterized the L2 wall precisely:
- **Multi-pour is RULED OUT.** `mp_probe.py`: every ACTION5 press resets the board
  (settled filled-13 = 0 after each pour, on L1 AND L2 — no accumulation); a ~4-pour
  budget exists (game-over at pour 5). **A win = ONE pour that fills all targets with no
  flood.** So the "pour-count" idea does NOT mean incremental filling across pours.
- **L2 is single-pour PROPOSER-blocked, not ranker-blocked.** `proposer_probe.py` (200 real
  arrangements): **0 wins**; the iter-2 sim gives **0 false positives** on those 200 (rejects
  all 197 losses as −1). Action space is not the bottleneck (L2 has 4 bar-blobs, all
  addressable). The winner is a NEEDLE: arrangements fill all 3 targets *at peak* but
  OVERPOUR → flood → not a win (the level-3 "don't overpour" mechanic). A clean
  fill-all-no-overpour config is what's missing and neither 200 random nor a 4,848-eval
  directed search finds it.
- **Residual sim-FPs are search-adversarial.** The sim is FP-free on random L2 arrangements
  but a margin-maximizing search still digs up a rare sim-+1/env-loss (v52 dynamic). So L2
  needs BOTH sim iter-3 (its specific flood paths) AND a proposer that finds the needle —
  or first, verification that a reachable L2 winner even EXISTS (unconfirmed).

**EXISTENTIAL RESULT (`reach_l2.py`, 600 rich arrangements, 2–10 clicks): 0 wins.** With
0/200 short + a 4,848-eval directed search, that is **1,400+ L2 arrangements, zero winners.**
So L2's winner is unreachable via the click-move action space from the presented state, or a
needle rarer than 1/1400 — either way **L2 is NOT solvable by search over this action space.**
The arrange-and-pour paradigm that cracked L0/L1 does not extend to L2 (and, by geometry,
likely L3–L5). Actions are functional (arrangements DO reroute paint — they fill all 3 at
peak then overpour), so it's not a no-op issue; the clean fill-all-no-overpour config is
simply not findable here.

**This is a fundamental wall, not a plannable completion — it needs a strategic call:**
- Understand the INTENDED L2 solution (is there a game mechanic/action beyond frame_components
  bar-moves? does the real solver do something else?).
- A LEARNED constructive proposer only helps if a reachable winner exists (unconfirmed — the
  1400/0 says probably not in this action space).
- Or accept the honest headline: **2 levels (L0 CNN-alone, L1 sim-guided), robust**; L2+ is
  blocked at the proposer/action-space level, which the sim lane cannot fix alone.

### CORRECTION (same session): the "wall" was partly a SIM MIS-MODELING bug — multi-source
**L2 has 3 sources (L4 has 2), but `extract_scene` collapsed them to ONE central spout** (paint
centroid). So every L2/L4 sim rollout used WRONG physics, and the sim-guided search that
"found no L2 winner" was guided by a wrong gradient. FIXED (S4 multi-source extension): `scene.py`
now emits one spout per source blob (`scene.spouts`), `raycast.py` emits from all; single-source
levels are byte-identical (S0 still PASS), variants are single-source (S2 unaffected). L2 now
correctly models 3 spouts [(2,1),(2,9),(2,14)]. L2 re-solve with correct physics = IN PROGRESS.
NOTE: the REAL 0/1400 winnability (env pours) is unchanged by this — it says random arrangements
don't win; the fix only lets the sim GUIDE the search correctly.

### RESOLVED (same session): multi-source fix is correct; L2 is genuinely action-space-blocked
After the multi-source fix, the L2 solve's rank-1 dropped from a false +1.0 to +0.0 (the sim no
longer hallucinates an L2 win). A stronger **sum-of-clean-fills proposer** (`proposer_search.py`,
10,304 evals, correct multi-source sim) **SOLVES L1@1 (fitness 9.55)** but on **L2 tops out at
fitness 0.05** — i.e. NO arrangement cleanly fills even one target. Calibration is sound (real L2
emits 54 macro-cells ~18/source; sim emits 42 and floods like the real losing default). So the
0.05 is structural: **on L2 every arrangement that fills a target ALSO floods it** (fill−lavan≈0
per target) — the "overpour" wall. Consistent with 0/1400 real winners.

**FINAL (thoroughly established, every confound ruled out — multi-pour, single-source, weak
fitness, emission calibration): L2 (and by geometry L3–L5) is NOT solvable by arrange-and-pour
over the current click-move action space.** The winner, if it exists, is unreachable / a
sub-1/1400 needle where filling implies flooding. Getting past 2 levels needs a STRATEGIC PIVOT
(a richer action space, the intended game mechanic, or confirmation L2 is unsolvable as-posed) —
not more sim work or search. One concrete residual lead: the multi-source sim floods at emitted
42 vs real 54 (slightly early) — an S1-iter3 multi-source flood-timing calibration MIGHT expose
L2 fills, but the 0/1400 reachability is the deeper blocker. **Honest headline: 2 levels
(L0 CNN-alone, L1 sim-guided), robust.** The sim lane is DONE and now CORRECT on all 6 levels.

## ⚠️ CORRECTION — the "L2 action-space wall" claim above was WRONG (confounded by a buggy sim)
Verified the multi-source sim against real L2 pours: on arrangements where REALITY fills 1–2
targets, the **multi-source sim fills ZERO** (`SIM fills=[0,0,0]` vs `REAL=[1,0,0]`/`[1,1,0]`).
So the "0/1400 + proposer fitness 0.05 → L2 proposer/action-space-blocked" conclusion was
CONFOUNDED: the sim (the multi-source code I added THIS session) never reports fills, so the
sim-guided proposer couldn't find any — not because arrangements don't fill. Two bugs found:
1. **`ledge_spill_floor` over-floods multi-source.** It was validated only on single-source S2
   variants; on L2's 3-source pour it cascades paint to the wall too early. Disabling it recovers
   some fills (sim then fills target 0 on the [1,1,0] cases).
2. **Multi-source routing still under-fills** — even floor-off, the sim fills ≤1 target where
   reality fills 2; paint from the 3 spouts isn't reaching the 2nd/3rd mouths as reality's does.
**L2 solvability is therefore UNKNOWN** (not walled) until the multi-source sim is faithful.
Concrete fix path (S4 multi-source fidelity, the real next task): tick-match real L2 pours;
make `ledge_spill_floor` multi-source-aware (or per-source-gated); fix 3-source→mouth routing;
gate on S0+S2 (single-source must stay identical) AND a new multi-source tick-match. THEN
re-assess whether an L2 clean-win arrangement exists/is findable. Reality still shows fills come
WITH floods on L2 (0/1400 clean) — so L2 may still be hard — but that verdict must wait for a
working sim. My earlier "fundamental wall" language was premature; this correction supersedes it.

### Multi-source fidelity — iteration outcome (kept the safe parts, reverted the risky heuristic)
Two fixes tried after the "sim fills 0 on L2" diagnosis (paint reaches the mouth but the sim,
counting SETTLED-only, ends the pour at the flood before it settles — while the GAME fills on a
transient enclosed-droplet entry, L775):
- **KEPT — `ledge_spill_floor` gated to single-source** (`floor_spill_active = ledge_spill_floor
  and len(spouts)==1`): the iter-2 floor-spill was calibrated on single-source S2 variants and
  over-floods multi-source; gating it is single-source-byte-identical (S0/S2 unchanged) and stops
  L2's over-flood. Safe + correct.
- **REVERTED — mouth-fill = settled+falling**: recovered L2 target-0 fills but OVER-filled
  (transient pass-through counted → L2 fill-count agreement only 36/200; slight single-source AUC
  cost 0.998→0.994). Trading over-flood for over-fill = thrashing. Reverted to settled-only to keep
  the validated single-source sim PRISTINE. Multi-source L2 is thus back to under-reporting fills —
  the honest incomplete state.
**Net:** single-source sim pristine + validated (S0 PASS, S2 GATE PASS); multi-source EXTRACTION
correct (L2=3 spouts) + floor-gated; multi-source FILL FIDELITY still open (task #5). The right fix
is to tick-match the game's enclosed-droplet fill timing (a droplet in the mouth counts the tick it
enters, and the pour must not flood-stop before all mouths are entered) — a focused calibration, not
a heuristic. Only after that can L2 solvability be judged. Single-source headline (2 levels) unmoved.

## SP80 mechanic — read from the GAME SOURCE (`environment_files/sp80/589a99af/sp80.py`)
Traced to resolve the L2 wall definitively (the strategic-pivot investigation):
- **4-pour budget, NO cross-pour accumulation (code-confirmed):** `cevwbinfgl` (filled set)
  resets every pour (L522/638); win check `all(targets in cevwbinfgl)` is per-pour (L730);
  `lyremoheq>=4 → lose` (L717). You get 4 independent single-pour attempts (rearrange between).
- **Win = precise routing, not volume:** a target (`repwkzbkhxl`) fills →13 ONLY when a droplet
  enters its mouth ENCLOSED — flanked on both perpendicular sides by the target (L775). Paint
  touching the WALL (`waoewejnqzc`) sets the FAIL flag (L815) = the "overpour/lava" loss.
- **Flow CA:** droplets (`liolfvkveqg`) advance 1 cell/tick, SPLIT around movable blocks
  (`plzwjbfyfli`, L764), and a redirect block (`tuvkdkhdokr`) DEFLECTS 90° (L787–803).
- **MISLABELING found:** `liolfvkveqg` is the PAINT DROPLET, not a redirect — `sp80_variants.py`
  (`REDIR_TAG="liolfvkveqg"`) and the notes are WRONG. The real redirect is `tuvkdkhdokr`.
  L4/L5 "color-15" modeling should use `tuvkdkhdokr`, and variant augmentation never actually
  placed redirects (it cloned paint-droplet templates as "redir" → likely no-ops).
- **Why L2 is a needle:** 3 sources (cols 1,9,14) vs 3 target mouths (~cols 2,7,12). Sources 1
  and 14 align with mouths; the MIDDLE source (col 9) aligns with NEITHER — it must be bar-routed
  to a mouth while the other two stay clear and NOTHING overshoots the wall. A tight simultaneous
  constraint → the 0/1400 needle. Solving L2 needs a MECHANIC-AWARE constructive proposer (route
  each source to an enclosed mouth, deflect the middle), not blind search — a real build.

## Files (this loop)
New: `sim3d/s2_generalize.py`, `s2_analyze.py`, `s1_tickmatch.py`, `s4_recon.py`; fixtures
`s2_variants_20260705{,_b,_floorspill,_floorspill_b}.npz`, `s3_l1_iter2_result.json`,
`s4_l2{,_iter2}_result.json`. Changed: `sim3d/scene.py` (source-fallback spout),
`sim3d/raycast.py` (`ledge_spill_floor` default + `backpressure_cap` + tick diagnostics),
`imagine_search_gate.py` (`--sim3d` lane). Notes: `SIM3D_S2/S1/S4_*_20260705.md`, this file.
