# Wide world model + SP80 imagination diagnosis — 2026-07-13

Honesty audit at session start: exit 0, `summary: 97/46/6`, `active: 85/23/0`,
`quarantined: 12/23/6` (unchanged from the morning session).

User directives, in order: (1) work the LP_TRAINING_SESSION_20260713 §7 work order;
(2) **"make sure we are doing more than sp80 — need to train a wide model."**

Two lanes ran in parallel. The SP80 lane produced a diagnosis that overturns the
work order's central premise. The wide lane produced a single world model over all
25 ARC-AGI-3 games.

---

## HEADLINE — the SP80 imagination gap is CLOSED, and the wide lane is what closed it

Read §1 for how the bottleneck was found and §2.3 for where the architecture came
from. The one-line version:

**SP80's pour dynamics was never a data problem or an objective problem. It was
ARCHITECTURALLY UNDER-POWERED.** `DynamicsUNet` is a 2-level UNet (64→32→16, 1.06M
params). A pour is LONG-RANGE — paint propagates across the whole board — and that
receptive field could not represent it. The wide lane's `WideWorldModel`, a 3-level
64→32→16→8 UNet (7.7M params) built to serve 25 games, learns SP80's pour physics
to **~0.99 changed-pixel agreement** where the specialist plateaus at **0.83**.

Deployed chain (`probe(dynamics(board_before))`), probe held FIXED, winners' episodes
held OUT of training, scored against all 19,751 env-graded L1 pours:

| chain | AUC | winner ranks of 19,751 |
|---|---|---|
| merged_v4 dynamics (the chain we shipped this morning) | 0.9578 | 3, 550, 2681, 113 |
| merged_v5 + D4 (the orientation fix) | 0.9899 | 10, 783, 12, 1 |
| **SP80 on the WIDE architecture, from scratch** | **0.9996** | **38, 3, 2, 1** |
| *direct read of the REAL settled board (upper bound)* | *1.0000* | *2, 4, 3, 1* |

Three of four winners now rank in the **top 3 of 19,751**. The chain has essentially
reached the ceiling set by reading the real settled board — i.e. the imagination is no
longer the thing standing between us and L1.

Note it is the **scratch** arm: the ARCHITECTURE, not the pre-training. The user's
"do more than sp80" directive is what produced the fix for sp80. `solve_loop.py` /
`branch_fleet.py` now load it through `wide_dynamics_adapter.py` (freezes the action
to ACTION5; no other loop code changes).

---

## PART 1 — SP80: the L1 bottleneck is the IMAGINATION, not the win probe

### 1.1 The measurement that changed the plan

The work order (§7 item 1) predicted: *"merged_v5's L1 positives activate L1
ranking."* That premise is **wrong**, and it was cheap to falsify before spending a
single GPU-hour on it.

New tool `eval_l1_winner_rank.py`. It asks the only question the hunt cares about:
of the 19,751 L1 pours we have env verdicts for, **where does the ranker put the 4
known winners?** Not AUC over a level — rank against the pours they competed with.

Two scores per row:
- `direct` = probe(**real** settled board) — is the win sense there at all?
- `chain`  = probe(dynamics(board_before)) — what the solve/hunt loop actually deploys.

Result, using the merged_v4 D4 probe, which was **trained with ZERO L1 positives**
(so scoring these winners with it is a clean held-out test):

| read | level-1 AUC | winner ranks of 19,751 |
|---|---|---|
| `direct` (real settled board) | **1.0000** | **2, 4, 3, 1** |
| `chain` (imagined board) | 0.9578 | 3, 550, 2681, 113 |

**The probe already separates every L1 winner from all 19,747 losers, perfectly,
with no L1 positives ever.** It never needed them. On episode 5 the dynamics turns
a `direct` = 0.823 certain-win into `chain` = 0.00006.

So the entire L1 gap is imagination error. If the imagination were faithful, a
budget of ~4 pours per board would find an L1 win instead of 128.

Corrected reading of what merged_v5 is actually for: not probe positives, but
**19,751 real arranged-L1 pour transitions**. merged_v4's L1 rows came mostly from
the 2026-07-09 hunt, whose v7 sampler was retention-broken and poured near-empty
arrangements.

Note on method: within-EPISODE rank is meaningless for these winners — the hunt
stops pouring on a win, so the winning episodes executed only 1/18/19/1 L1 pours.
Rank against all 19,751 is the honest pool.

### 1.2 What did NOT fix it (three falsified levers)

**(a) The argmax is not the culprit.** `diag_chain_argmax.py`. The chain quantises
the dynamics' distribution with `argmax` before the probe reads it; feeding the
probe the raw softmax instead costs nothing and needs no retraining. It is not the
fix: ranks 3/491/4557/21 (softmax) vs 3/550/2681/113 (argmax), AUC 0.9359 vs
0.9578 — helps two winners, hurts another. The loss is the imagined board's
CONTENT, not its quantisation.

**(b) More arranged-L1 data does not fix it — and nearly produced a false positive.**
Trained on all of merged_v5, the dynamics looks transformed: chain AUC 0.9962, three
winners inside the top 20. But that model **trains on the 4 winners' own
transitions**. Its clean held-out twin — same data, same objective, minus the 4
winners' episodes (2702/2703/2712/2759) — scores **0.9566, statistically identical
to v4**. The gain was memorisation.

| dynamics (probe held FIXED) | status | chain AUC | winner ranks of 19,751 |
|---|---|---|---|
| merged_v4 | clean | 0.9578 | 3, 550, 2681, 113 |
| merged_v5 **full** | **CONTAMINATED** | **0.9962** | **18, 276, 2, 15** |
| merged_v5 held-out, pixel | clean | 0.9566 | 14, 3244, 178, 6 |
| *direct read* | *upper bound* | *1.0000* | *2, 4, 3, 1* |

Without the held-out arm this session would have reported a breakthrough and
deployed a memoriser. **Any future L1/L2 claim must hold out the winners' episodes.**

**(c) The perceptual fidelity term fixes L0, not L1.** New `--perceptual-probe` in
`train_dynamics.py`: a distillation term forcing the frozen probe's read of the
IMAGINED board to match its read of the RECORDED settled board. (Target is the
probe's read of a real board, never the env win label — so a losing row carries a
near-zero target and the dynamics *cannot* lower the loss by hallucinating wins.
Pure fidelity, no game facts.)

It works, on the axis it can see: `val_win_read_agreement` 0.998, `val_chain_auc`
0.9997, and it *also* improved raw pixel accuracy (val_changed 0.886 vs the
control's 0.831). But the held-out L1 winner ranks barely move (5, 3014, 136, 189;
AUC 0.9578). **`val_chain_auc` is an L0-ONLY number** — merged_v5 has 0 L1 val
positives — which is exactly the trap flagged in §1.1.

### 1.3 WHAT FIXED IT: the dynamics was ORIENTATION-LOCKED

The untreated pathology. The dynamics has seen **878 winning pours at L0 and 4 at
L1**, and **L1 is rotated geometry**. To imagine a winning L1 pour it must transport
L0's winning-pour physics into a frame it has no prior for.

The win probe had exactly this disease — plain probe rot180 AUC **0.579** — and D4
augmentation cured it (**1.0** under all six transforms). *The dynamics never got
that prior.* `train_dynamics.py --augment-d4` applies the same random D4 transform
to the (before, settled) pour PAIR. Generic symmetry prior, no game facts. It is an
ASSUMPTION (that the pour is D4-equivariant) and it was falsifiable: if the flow
direction were absolute rather than level-relative, changed-pixel agreement would
have dropped. **It rose** — so the assumption holds.

**Result (held out, probe frozen, so the dynamics is the only variable):**

| dynamics | chain AUC | winner ranks of 19,751 | val_changed |
|---|---|---|---|
| merged_v4 | 0.9578 | 3, 550, 2681, 113 | 0.840 |
| v5 held-out, pixel (control) | 0.9566 | 14, 3244, 178, 6 | 0.831 |
| v5 held-out, perceptual | 0.9578 | 5, 3014, 136, 189 | 0.886 |
| **v5 held-out, +D4** | **0.9899** | **10, 783, 12, 1** | **0.90+** |
| *direct read (upper bound)* | *1.0000* | *2, 4, 3, 1* | — |

Episode 5: **2681 → 12**. Episode 61: **113 → 1**. Median winner rank **~331 → ~11**;
3 of 4 winners now inside the top 12 of 19,751. First lever all session to beat the
v4 baseline at all — and it *also* improved raw pixel fidelity, so it is not a
trade-off. (Numbers above are the epoch-7 checkpoint; final 40-epoch numbers in
`runs/merged_v5_l1pos_20260713/`.)

### 1.3b The linear-legal L1 solve is PROPOSER-blocked, not ranker-blocked

Work-order §7 item 5 hoped a sharper chain would enable an L1 win in a plain linear
run. The arithmetic says no, and it is worth writing down so nobody burns a night on
it. Above L0, RESET is a full-game restart, so a linear agent must arrange IN PLACE
inside ~45 actions and 4 pours. Winning tapes are 14-33 actions, so a linear run
realises **2-3 arrangements**. The branch hunt needed a **128-arrangement pool** to
find a winner on 4 of 158 boards (2.5%). Even a PERFECT ranker cannot rank an
arrangement that was never realised: P(win) ≈ 2.5% x 3/128 ≈ 0.06%.

**The D4 chain fixes ranking, which was only half the problem. The other half is
candidate density.** The linear headline needs a better PROPOSER (or an arrangement
model that predicts the arranged board from a tape without executing it, so
candidates can be scored before they are paid for in actions).

### 1.4 Tooling / ops fixed

- `branch_fleet.py` **v9**: `--known-tapes` honours winning tapes at EVERY level (v8
  only fast-pathed L0, so the four L1-winning seeds had to re-find their L1 win by
  search — ~0.02%/pour — before they could ever pour at L2). Verified: 4/4 seeds now
  replay L0+L1 in one pour each and spend their whole budget at L2.
- `branch_fleet.py` `--rng-salt`: the episode RNG is keyed on the seed alone, so
  re-running a seed replays an identical episode. Only 4 boards can reach L2, so
  searching them with diverse draws requires this.
- **`pour_idx` int8 overflow (real bug, killed a run).** `np.int8(pours_this_level-1)`
  raises `OverflowError` at pour 128. v8's 128-deep pool stopped at exactly 127 by
  luck; a 384-deep L2 pool blew through it and killed all 4 hunts. Now saturates at
  127 — which is what the merged manifests already *claimed* ("writer-clamped at int8
  max"). That claim was previously true only by accident.
- **Orphaned-worker hazard, hit live.** The fleet Pool uses **spawn**, so worker
  cmdlines are `spawn_main(...)` and do NOT match `branch_fleet.py`. Killing runs by
  script name leaves workers at 99% CPU forever. Kill by ppid.

---

## PART 2 — The wide world model: one model, 25 games

`synthesis_lab/experiments/wide_world_model/` (new lane).

### 2.1 The substrate is real (not assumed)

`wide_recon.py` boots every locally-downloaded game and probes it through the env
only (no source reads). **25/25 boot offline.** They share one substrate:

- every frame **64x64**; colour union exactly **0..15**
- one action space, **ACTION1..7** (ACTION7 exists — sb26/su15/sk48 — which the
  original 6-action assumption would have silently dropped)
- the same env-graded `state` / `levels_completed` signals
- responses are 1..42 frames (animations); the last frame is the settled grid

That is exactly the substrate a single action-conditioned world model needs.

### 2.2 Corpus — `wide_collect.py`, 264,520 transitions, 25 games

Schema `wide_world_model.transitions.v1`: (frame_before, action_id, click_xy) →
(frame_after, n_frames, state, level_before/after, level_advanced). Episode-keyed
split **stratified by game**. Random exploration over the env's OWN
`available_actions`; env-graded labels only; sp80 seed 1234 never played.

**Click prior, measured not guessed** (`diag_click_strategy.py`). A uniform pixel is
usually dead space — the first smoke corpus was 98.3% no-op on lp85. Aiming at the
centroid of a connected component is far richer, and is still a generic "aim at
something rather than nothing" prior with no game knowledge in it:

| game | uniform | **centroid** |
|---|---|---|
| vc33 | 3.9 px | **61.0 px** |
| r11l | 80.7 px | **102.3 px** |
| lf52 | 1.6 px | **16.6 px** |
| sb26 | 4% respond | **26% respond** |

Final policy: 50% centroid / 25% non-modal / 25% uniform. Corpus is 21.6% no-op
overall, ~43 changed pixels on rows that move, 125 env-graded level advances.

### 2.3 The model — `train_wide.py`

24 input planes: 16 colour one-hot + 7 action one-hot (broadcast) + 1 click mask.
UNet → 16-class next-frame logits. 7.7M params.

**It is given NO game id.** It must infer the game from pixels. A game-id embedding
would make it 25 specialists in a trenchcoat and would render the held-out-game test
meaningless.

**THE METRIC.** Only ~1% of pixels move per step, so predicting "nothing happens"
already scores **0.9916 overall pixel agreement**. Overall agreement is therefore
near-meaningless and is reported only to show why it must not be used. The honest
frontier is **changed-pixel agreement**, on which the identity baseline scores
exactly **0.0**.

### 2.4 Results

**A / within-split — IT WORKS.** All 25 games, ~10% of every game's episodes held
out: **val_changed = 0.957** (identity baseline 0.0), val_overall 0.998. One model,
25 games, no game id. Per-game changed agreement ranges 0.83 (bp35) to 1.00 (ft09,
tu93). This is a genuine multi-game world model.

**B / held-out 5 games — ZERO-SHOT TRANSFER FAILS.** Train on 20 games, predict
cd82/ls20/sb26/tu93/vc33, never played: **val_changed = 0.021, falling to 0.005 as
training proceeds** (identity = 0.0). The model outputs identity for a game it has
never seen, and gets MORE confident about that the longer it trains on the others.

**This is the honest headline of the wide lane, and it is a negative.** The 25 games
share a *substrate* (grid, colours, action space) but not *dynamics*: each game binds
its own arbitrary sprite semantics to the same 7 actions, and nothing at the pixel
level lets you infer those bindings from games that don't share them. The wide model
is **one network holding 25 game-specific dynamics, keyed on visual appearance** —
NOT a general grid-world physics engine. Breadth of games does not buy zero-shot.

**C / held-out sp80 — same verdict, sharpest form.** sp80 removed from training
ENTIRELY; can a model that played the other 24 games predict our home game?
**val_changed = 0.048** (identity 0.0) after 3 epochs. Essentially nothing. Zero-shot
fails in both directions: 24 games do not teach you sp80, and 20 games do not teach
you ls20/vc33/sb26/tu93/cd82.

**Wide vs SP80 specialist** (`eval_wide_vs_specialist.py`), same rows, same question,
on the SP80 lane's own val episodes — changed-pixel agreement:

| model | L0 | L1 | ALL |
|---|---|---|---|
| SP80 specialist (36,272 SP80 rows) | 0.766 | 0.832 | **0.818** |
| wide, 25 games (705 sp80 pours) | 0.000 | 0.000 | **0.000** |

**Breadth alone does NOT buy SP80's pour physics.** The wide model nails sp80's
move/select dynamics (0.9998 aggregate on its own corpus) but outputs pure identity
on pours. Verified NOT an eval bug: the same code path scores 0.39 on the wide
corpus's own sp80 pour rows, and the colour histograms of the two board sets are
identical. Two causes, both real: it saw **705** sp80 pours vs the specialist's
36,272, and the SP80 lane's arranged branch-search boards are out of distribution.

So the wide model is a **separate deliverable, not an SP80 improvement**, and must
not be sold as one.

**Does PRE-TRAINING help SP80? No — the gain was the ARCHITECTURE, not the weights.**
`finetune_sp80.py`, identical arch/data/schedule/holdout, only the init differs:

| init | epoch 0 | best changed (ALL) |
|---|---|---|
| random | **0.9901** | 0.9965 |
| wide-pretrained (24 other games) | 0.9777 | (see metrics.json) |

Scratch reaches **0.9901 after ONE epoch** — the architecture learns SP80's pour
physics almost immediately — and the pre-trained init actually starts *lower*, because
it has to unlearn 25 games' worth of other dynamics. Consistent with the zero-shot
negative: these games' dynamics do not transfer.

Caveat that keeps the question open: SP80 has 36,272 transitions, far more than
pre-training could add value to. The fair test of breadth is FEW-SHOT on a genuinely
new game — `finetune_new_game.py` gives the learner only 20 episodes of a game the
wide checkpoint HELD OUT (it refuses to run otherwise), pretrained vs scratch. _running_

**Bottom line so far: the wide lane's contribution to rexus is a better ARCHITECTURE
(a receptive field big enough for a pour), not transferable weights.** That is still a
large contribution — it is what closed the imagination gap — but it must be described
accurately.

### 2.5 Also banked

- **merged_v6** (`runs/merged_v6_l2_20260713/`): 42,448 rows. **L2 grew 512 → 6,656
  (13x)** from the v9 L2 hunt — the program's first substantial L2 physics corpus.
- **L2 hunt result: 6,144 L2 pours, 0 wins.** 12x the prior L2 data, still no L2 win;
  all 4 seeds exhausted 384-deep pools. L2 is a far harder needle than L1, and it is
  still only 4 reachable boards. Honest negative.
