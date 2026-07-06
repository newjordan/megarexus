# DEVIATION-1 — per-target fill/lava window correction (STAGE_PT_SPEC §3.2)

**Date:** 2026-07-05 · **Stage:** T0 · **Status:** applied, documented, reversible.
**Flag for human review** — this changes a pre-registered label definition. It is
NOT a hypothesis-side change (it is neutral between winner and FP); it fixes a
label-code artifact the spec's OWN sanity check caught. Revert if rejected.

## What the spec said (§3.2)
`sus_t = max_i min(n13_t[i:i+3])` — a **3-consecutive-frame** min window, "mirrors
harvest_variants.pour_record". Sanity check (§5.T0.3): `sum(sus_t) ≈ 240` on the L1
winner (3 targets × ~80 cells).

## What actually happens (evidence: verified L1 winner, seed 1234, 22-frame pour)
Paint(6) grows monotonically to max (496) at **frame 18**, where only U2 is filled.
The last two targets flip 11→13 at **frames 19–20** (paint already maxed), then the
level **RESETS at frame 21** (13→11, paint cleared, wall-14 spikes 27→64 = win-flash).

```
frame | g13 g11  g6  g14 | U0 U1 U2   (per-target #13)
  15  |  80 160 400  27  |  0  0 80
  18  |  80 160 496  27  |  0  0 80    <- first max-paint (settle_index lands here)
  19  | 240   0 496  27  | 80 80 80    <- win completes
  20  | 240   0 496  27  | 80 80 80
  21  |   0 240  48  64  |  0  0  0    <- RESET + win-flash (post-outcome garbage)
```

The all-filled win state holds for **only 2 frames** (19–20). Consequences of the
literal §3.2 window on a CONFIRMED 240/240 env-win:
- **Fill:** the 3-frame min-window requires 3 consecutive filled frames → reads
  `sus=0` for U0 and U1 (`sum(sus_t)=80`, NOT 240 — the spec's own sanity check FAILS).
- **Lava:** `max over ALL frames` includes frame 21's win-flash → global `lava_harvest
  = 36`, contradicting the §2 ground fact that the winner has `lava≈1`. §2 itself warns
  "absolute 14 count is noisy — border + win-flash".

This is the exact F16-class post-reset label bug the spec mandates `pour_settle` for
(§1.8): reading pour outcomes off frames past the settle/reset point.

## The correction (implemented in `orchestration/pertarget_labels.py`)
Compute fill AND lava over the **pre-reset outcome frames** only — `frames[:end]`
where `end` = one past the last max-paint frame (drops the reset/flash tail, the
natural extension of `pour_settle.settle_index`). Fill uses a **2-frame** sustained
window (matches the 2-frame win hold; still rejects 1-frame flicker).

Result on the winner: U0=U1=U2 `fill=1.0, lavan=0.0`, `sum(sus_t)=240` (sanity ✓),
global honest `lava=0`. The strict 3-window value is still recorded per target as
`sus_strict3`, and the harvest-literal global lava as `global.lava_harvest`, so the
deviation is fully auditable and nothing is hidden.

## Why this is not "a new label scheme" (spec §8)
The label still means exactly "per-target sustained fill fraction" / "per-target local
14-growth". Only the window (3→2) and the frame range (exclude post-reset) changed —
required for the label to be self-consistent with the spec's stated sanity check and
with §2's ground-fact lava. It is neutral w.r.t. the hypothesis: it credits fill for
BOTH the winner and the FPs; whether they then separate remained an open empirical
question, which T0 answered (PASS: separation is driven by per-target lava).

## T0 outcome with the correction
VERDICT=PASS. Winner passes the predicate on all 3 targets; all four v52 FPs
(scores 0.75/0.734/0.732/0.651) are env-losses that each violate the predicate on
≥1 target (all four lava-saturated; FP#2/#3 also miss fill on U1; FP#4 on U0).
Full record: `runs/t0_pertarget_20260705.json`.
