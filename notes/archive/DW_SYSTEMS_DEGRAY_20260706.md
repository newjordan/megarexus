# DW SYSTEMS DE-GREY — 2026-07-06

Purpose: split everything the Dreamweaver/sim3d lane "knows" into three explicit systems so
future work can aim observation-driven induction at every remaining grey item ("reverse
engineer into that" — user). **MACHINERY** = game-agnostic engineering, allowed to be
hand-built (plan §1 generic-machinery clause, `DREAMWEAVER_INTEGRATION_PLAN_20260706.md`).
**INDUCED KNOWLEDGE** = every claim the kernel makes about SP80, each selected by observation
against alternatives, with its evidence fixture and validation status. **GREY RESIDUE** =
anything hand-bound or hint-derived, each with a concrete de-grey plan. Sources: the lab log
(`synthesis_lab/notes/DW_LAB_LOG_20260706.md`, esp. ONTOLOGY AUDIT, H4 CERTIFICATION, ROUND 4),
the code (`synthesis_lab/orchestration/sim3d/scene.py`, `imagination_kernel.py`), and the
round-4 selection fixture (`sim3d/fixtures/dynamics_round4_20260706.json`). Negatives included.

---

## 1. Machinery (engineered, game-agnostic)

H4 certification (log "H4 CERTIFICATION — PASS", fixture `sim3d/fixtures/h4_variants_20260706.json`)
is the systemic proof: on 48 never-seen synthesized boards the machinery ran end-to-end from the
PALETTE manifest alone (48/48 trusted, deterministic rollouts) and re-derived the exact round-3
dynamics member from a disjoint pour corpus. The machinery ships; the knowledge is re-derivable.

| Component | Where | What it assumes | Why that is not game knowledge |
|---|---|---|---|
| Lattice detection | `sim3d/scene.py` `detect_lattice()` (L116) | The finest reliable on-screen geometry shares a pixel unit; unit = gcd of glyph run lengths, origin = modal run-start residue | Pure signal processing on pixel runs; the only game-specific input is which color is "target" (PALETTE, §3a). Found 4px/origin-0 (L0–L2) and 3px/origin-2 (L3–L5) with zero level-specific code |
| Perception gate checks | `sim3d/perception_gate.py` (+ `perception_gate_eval.py`) | Every scene claim (targets, glyphs, spouts, mouths, rays, floor, coverage) must exactly reconstruct from raw pixels, rebuilt independently inside the gate | Self-consistency of extraction vs pixels; V031 concept from dreamweaver. Fails safe: wrong PALETTE ⇒ untrusted, not wrong claims. Reports `unmodeled_colors` (15) instead of guessing |
| KnownMap provenance rule | `sim3d/imagination_kernel.py` `KnownMap.from_scene` (L36–83) | A scene enters the kernel only with a passing gate verdict; else `UntrustedSceneError` | Dream Kernel provenance contract (epistemic discipline), no content about SP80 |
| Hypothesis-family framework + fit protocol | `DynamicsHypothesis` / `FallingMaterial` (192 cfgs) / `FlowTrail` (32 cfgs) in `imagination_kernel.py`; `dynamics_fit.py`, `r3_fit.py`, `r4_fit.py` | Only generic material-transport primitives (fall, deflect, spread, trail, merge, absorb); pre-registered train/holdout split (i%4==3), full table reported, negative branch = escalate | Which member (if any) describes SP80 is decided ONLY by tick-matching observed env pours; the family grid is a hypothesis space, not an answer (`FallingMaterial` docstring, L180–197) |
| dw_solve search scaffold | `sim3d/dw_solve.py` | Hill-climb over arrangement genomes; imagined fitness ranks; gate re-checks every candidate in-loop; top-K get ONE real pour each; env decides | Generic black-box search + the provenance rule; zero search pours; all win claims are env verdicts |
| Realization / closed-loop driver | `sim3d/r4_l3_realize.py`, `r4_l3_multipour.py`, `r4_l3_feasibility.py` | Clicks move movables; direction map is EMPIRICAL (probed, incl. discovering env-unreachable rows); landing verified closed-loop before pouring | Actuation calibration learned by interaction, not from source; aborts un-landed steps to protect the pour budget (honest negative logged) |

---

## 2. Induced knowledge ledger

Every SP80 claim the kernel makes. "Holdout-gated" = passed pre-registered per-corpus holdout
gates in a fit fixture; "env-confirmed" = a live env outcome verified it; "observed-fact" =
directly read off harvested frames; "hypothesis-only" = proposed, not yet validated.
Primary selection fixture: `sim3d/fixtures/dynamics_round4_20260706.json` (selected member
`flow_trail {emitters=all, phase=uncharged_lead, mouth=feed_die, volume=unlimited, merge=die}`,
rule=fill_cap, stop=volume; holdout gates pass all four corpora, `pass: true`).

| Claim | Selecting observation | Alternative(s) that lost | Fixture | Status |
|---|---|---|---|---|
| Two native lattices: 4px origin-0 (L0–L2), 3px origin-2 (L3–L5) | Target-glyph run-length gcd + modal residue per frame; at 4px the 3px mouths vanish and targets merge (the known gate-failure classes) | Single fixed 4px macro grid (historical; lost on L3–L5) | `fixtures/perception_gate_L0_L5_20260706.json` (post-fix 6/6 TRUSTED, 5/5 seeds; A/B 9/9 identical on L0–L2) | observed-fact, gate-verified |
| Both housings emit (charged AND uncharged) | t0 paint = the uncharged mid-housing's spout cell in 40/40 L3 pours; charged stream appears t1 | "Charged spouts only" (the ontology audit's flagged conservative assumption — FALSIFIED, log ROUND 4 Phase A §3) | `dynamics_round4_20260706.json` (emitters=all); corpus `calib_l3_pours_20260706.npz` | observed-fact + holdout-gated |
| Uncharged stream leads by 1 tick (charged drains through housing first) | Frame-0 mid stream vs t1 top stream, all L3 pours | phase=sync | `dynamics_round4_20260706.json` (phase=uncharged_lead) | holdout-gated |
| Trail persistence: every visited cell stays painted (liquid film, no drain) | L3 visible paint plateaus ~66 cells with no decrease; plateau = tip saturation, not budget | FallingMaterial per-unit piling (IoU 0.68 / fdiv 6 vs FlowTrail 0.959 / 23–27; outcome-level TIE — separation is tick fidelity + expressiveness, log Phase C caveat 1) | `dynamics_round4_20260706.json` table | holdout-gated |
| Spread-and-split at obstacle tops (blocked tip runs BOTH ways 1 cell/side/tick, descends at both overhang edges) | Directly watched in L3 ASCII flow renders (t=5→8, pours 0/1/5); also removes the L2 fdiv-5 multi-stream infidelity zero-shot (fdiv 5→25, IoU 0.62→0.96) | Straight fall / diagonal deflect variants | `dynamics_round4_20260706.json` (l2_zeroshot block) | observed-fact + holdout-gated; L2 wins env-confirmed |
| Merge on trails: converging streams merge quietly (merge=die) | Fit IoU 0.9604 vs split's 0.9585 (margin 0.002); both pass all holdout gates | merge=split — NOT cleanly beaten: split's search landscape solves L2 5/5 vs die's 3/5; both members preserved with own fixtures (`dw_solve_r4split_*`) | `dynamics_round4_20260706.json` | holdout-gated, WEAKLY discriminated (tie at outcome level; round-5 flywheel queued to break it) |
| Mouth entry absorption: mouths absorb on entry; full mouth ends stream quietly (feed_die) | Entry-absorb induced from L1 data (broke the L0 tie — log round 1.5–2: previously only illegally known, now induced); unlimited-absorb from flywheel winners (round 3); feed_die vs feed_overflow (round 4) | mouth=fill/settle; absorb=cap (caused the systematic win-bit false-negative plateau, log seed-sweep diagnosis) | `dynamics_round2_20260706.json`, `dynamics_round3_20260706.json`, `dynamics_round4_20260706.json` | holdout-gated + env-confirmed (rank-1 imagined wins env-verified 10/10 L0/L1 + 5/5 L2-split) |
| Win/lava semantics: floor strip contact = fail; imagined win = fills≥cap & floor==0 | Real px14 flash separates 31/31 flywheel pours (4 winners no flash, 27 losers flash); win predicate SELECTED from 4 rule×stop candidates; used for RANKING, env decides | rule=entered, stop=win combinations | `r3_flywheel_20260706.json`, `dynamics_round3_20260706.json` | env-confirmed (flash separation); predicate holdout-gated, ranking-only |
| Lava label rule: flash iff ≥2 pixel rows newly majority-14 (scale-free) | Bimodal flash deltas per lattice (4px: 253–256 px; L3: 179–180 px — the old fixed LAVA_DELTA=200 mislabeled ALL L3 pours); v1 (≥1 row) failed regression on 8 winning pours | Fixed pixel-count threshold; v1 any-row rule | `r4_phaseb_loader_20260706.json` (byte-identical masks + labels on all 4px corpora) | observed-fact, regression-gated |
| Fills reset after FLASHING pours (no cross-pour accumulation) | Multi-pour probe: px13 45→0 after every animation — now legally observed, no longer tainted-source knowledge | Accumulation across pours | `r4_l3_multipour_s1234_20260706.json` | observed-fact — BOUNDED: only flashing pours observed (§4) |
| 4-pour budget; 5th pour = GAME_OVER | Live probe: 5th pour → GameState.GAME_OVER, live=False | — | `r4_l3_multipour_s1234_20260706.json` | observed-fact (env-confirmed live) |
| Floor strip = fail sensor even at floor-contact 1 | Two pre-registered diagnostic pours at the near-clean frontier: floor 2 → flash, floor 1 → flash, both exactly as imagined | L3 win might tolerate small floor contact (still open on the WIN side, §4) | `r4_l3_multipour_s1234_20260706.json`, `r4_l3_diagpour_s1234_20260706.npz` | env-confirmed (loss side only) |
| Win recolor = 1-row outer HUD border flash on the final frame | Loader regression audit: winning animations recolor row 63 (64 px), 0–1 newly-majority rows vs lava's 3–4 | — | `r4_phaseb_loader_20260706.json` (side discovery) | observed-fact; as a pixels-only WIN DETECTOR: hypothesis-only, unbuilt |
| Per-lattice tick convention: tips advance 1 lattice cell/frame; growth = active-tip count | 1:1 alignment measured on L0 (4px) and re-measured on L3 (3px, median 3 cells/frame = 2 emitters + spread tips) | Emit-then-move (round-1 alignment bug, fixed as move-then-emit / move-then-spawn) | `dynamics_fit_l0_20260706.json`, log ROUND 4 Phase A §4 | observed-fact; NOTE: diverges at t=0 on H4 variant animations (uninvestigated offset — do not tick-fit variants without re-verifying alignment) |

---

## 3. Grey residue + de-grey plans

**(a) PALETTE manifest — the declared single game-specific residue.**
`sim3d/scene.py` L34–50: color→role bindings {open:12, boundary:1, source:4, paint:6,
target:11, filled:13, wall:14, movable:8, selected:9}. Hand-bound (verified per-frame by the
gate, but the BINDING itself was authored, not induced). H4's own closing statement: "the
manifest itself remains the single game-specific residue."
De-grey plan: induce color roles from frames + env outcomes — role discovery from behavior
across episodes: which color appears at t0 under housings (source/paint), which flashes with
losses (wall/lava), which recolors on fills (filled), which components recolor under clicks
(movable/selected 8↔9), which encloses hollow interiors (target), which is the immutable
frame band (boundary), residual = open. Each role has an observational signature already in
the harvested corpora; acceptance = re-derived manifest byte-equals the authored one, then
delete the authored constants. Gate fails safe until then.

**(b) User hints — search-strategy guidance, never model content.**
(i) 2026-07-03 sidewalls hint ("sidewalls turn 4 streams into 2") — predates this lane; the
spread-and-split physics was independently OBSERVED from `calib_l3_pours_20260706.npz` before
any hint-shaped family was fit, but the hint existed in memory when FlowTrail was proposed.
(ii) 2026-07-06 two-spouts hint ("two pouring spouts from here on out") — opened the
multi-pour lane (log final entries); it directs WHERE to search, and must never become a
kernel parameter that observation didn't select.
De-grey plan: hint-free replication runs — a fresh session (or lesser agent) given only the
corpora + family framework must re-select FlowTrail and re-open the multi-pour lane from the
budget/reset observations alone. Until run, these remain grey.

**(c) Hypothesis-proposal bias.** Knob CHOICE is authored; session memory contains retracted
source knowledge (the dropletsim answer-key read, `INVALID_DROPLETSIM_ANSWERKEY_20260706.md`;
mouth-entry semantics were known illegally before being legally induced). Mitigations in
place: free-parameter grids with alternatives, full-table selection (fixture `table_top40`),
per-corpus holdout gates, env-decided wins, and the mechanical source guard
(`.claude/hooks/block_game_source.sh`, live-fire proven). Remaining exposure (named in the
ontology audit): vocabulary bias — a mechanic outside the proposed families is invisible to
imagination; bounded because it can only produce honest FAILURES (env grades every win),
never false headlines. Ongoing de-grey: keep proposing from OBSERVATION ARTIFACTS (Phase-A
style renders) rather than intuition, and record the motivating observation per knob (done
for absorb, merge, phase).

**(d) Solver scaffolding is engineered, not learned.** `dw_solve.py` hill-climb, the 6.6×
budget escalation, `r4_l3_feasibility.py` placement search, `r4_l3_multipour.py` sequencing —
all hand-built search, zero learned proposal. Legitimate as machinery (§1) but the AGENT
should not need it forever.
De-grey plan: Stage E/F amortization into the CNN per
`synthesis_lab/notes/ROADMAP_WORLD_MODEL_20260705.md` (pixels decide / latents rank /
portband-seeded CEM proposes) — distill dw_solve's search behavior into learned proposal so
the scaffold becomes a teacher, not a dependency. The kernel-certified win corpus
(`dw_solve_r4_*` fixtures) is ready-made training signal.

---

## 4. Open contradictions / unknowns

1. **Surgery-vs-frame KnownMap discrepancy — RESOLVED (round-4b close, same day):**
   root cause was the REALIZATION layer, not the maps: the nearest-to-target bar matcher
   swapped same-size physical bars mid-plan, delivering a different geometry that passed
   the per-target check. With identity tracking the frame KnownMap matches surgery exactly
   ([0,1,1,0]/floor 0) and the env pour was CLEAN. Only the flow-irrelevant row-0 top
   wall-bar rendering diff remains (documented, not chased).


2. **Fills persistence after a CLEAN pour — FALSIFIED (round-4b close, same day):** the
   first non-flashing L3 pour (kernel-certified [0,1,1,0]/floor 0, env flash=False) still
   ended with n13 = 0 at animation end. Fills reset after EVERY pour, clean or flashing;
   pour-sequence accumulation is dead as stated. Fixture `r4_l3_multipour_s1234_20260706.json`.


3. **L3 win-side semantics: UNVALIDATED.** 0 L3 wins ever observed; the induced model says a
   single clean pour is unreachable with the 4 movable bars (`r4_l3_feasibility_20260706.json`,
   ~500k evals, floor never <1) and both frontier diagnostics flashed at floor 1 — but if L3
   wins tolerate some floor contact, everything reopens. Only an actual L3 win (or mechanic
   information) resolves this.
4. **Color 15 (redirects) unmodeled — L4/L5 blind.** The gate reports it as
   `unmodeled_colors`; no functional role assigned; dw_solve on L4/L5 deliberately skipped.
   The blindness surfaces as honest failures, never false claims (ontology audit). De-grey is
   the same recipe as everything above: harvest L4/L5 pours, watch what color-15 cells DO to
   arriving streams, extend the family with observed primitives, holdout-gate, env decides.
