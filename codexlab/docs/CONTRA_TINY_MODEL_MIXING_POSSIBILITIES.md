# CONTRA — Tiny-Model Mixing + Cross-Representation Distillation: Possibilities

**Status:** exploratory thinking doc, NOT a plan of record.
**Date:** 2026-05-29
**Seed:** "Grab a couple really tiny models, mix them for exactly what we want on
every metric, LoRA fine-tune per model, make some magic" — riffed off X-Token
(arXiv 2605.21699, NVIDIA): projection-guided *cross-tokenizer* knowledge
distillation.

> We probably won't build this. This doc exists to think through whether the
> idea is sound, where it would actually fit CONTRA, and what would have to be
> true first. Read it as a map of options, not a commitment.

---

## 0. Ground truth about what CONTRA *is* today (this constrains everything)

Before riffing, the architectural reality — because it kills some options and
reshapes others:

- The **only trainable neural piece** right now is `sp80_lora_adapter.py` →
  `FrozenLinearLoRA`: a frozen `Linear` + LoRA (`lora_a`, `lora_b`) that maps a
  **bag-of-features** (regex token counts over a serialized state string) onto a
  **control primitive** drawn from `CONTROL_PRIMITIVES`. It is a *classifier
  over a small fixed action vocabulary*, not a next-token text generator.
- The **CONTRA controller head** (`contra/controller_head.py`) is a
  vision-first *perception + rules* pipeline emitting structured packets with
  fail-closed admission metadata. No transformer, no tokenizer.
- **Oracle** (`oracle.py`) emits review-only *assumptions*, explicitly NOT
  control truth / action choice / predictions.

**Consequence:** X-Token solves a problem we *don't currently have*. Its whole
contribution is reconciling **mismatched token vocabularies between transformer
LMs**. CONTRA has no transformer LM and no token vocabulary — its output
"vocabulary" is `CONTROL_PRIMITIVES`, which is **shared by construction** across
any model we'd train. So the literal paper doesn't slot in. But the *underlying
trick* (distill across models whose output spaces don't align, via a learned
sparse projection) does generalize, and that's where Route C gets interesting.

---

## 1. The honesty box: "optimize on EVERY metric" is the wishful part

Distilling/merging N specialists into one small student almost never *dominates*
each specialist on its own metric. Finite capacity has to spread. Realistic
outcomes:

- **Distillation into one student:** ~90–95% of each specialist across the
  board, in one cheap forward pass, no router. Win = **breadth + cost**, not
  per-axis dominance.
- **Weight-merging (model soup / TIES / DARE):** trades peaks for averages;
  smooths, doesn't stack peaks.
- **Routing/MoE-of-LoRAs:** *can* keep per-axis peaks, but you pay a router and
  give up the "one clean model" simplicity.

There is no configuration that beats every specialist on its own turf for free.
If a route claims that, it's wrong. Pick the tradeoff on purpose.

---

## 2. Route A — Multi-LoRA specialists over the *existing* head (lowest risk)

Stay entirely in the current substrate. Train several LoRA adapters on top of
the same frozen `FrozenLinearLoRA` base, each specialized to a **regime**, e.g.:

- goal-distance regimes (near vs far `nearest_goal_distance_bucket`)
- rotation-heavy vs rotation-free tasks (`rotation_quarters`)
- color-isolation edge cases (the `color_overlays` failure modes)

Combine by (a) **logit-averaging** at the primitive layer, or (b) a tiny
**regime router** keyed off a few state features.

- **X-Token relevance:** none — all adapters share `CONTROL_PRIMITIVES`. No
  cross-vocabulary problem exists.
- **Pros:** cheap, fully GPU-trainable today, reuses existing dataset export
  (`sp80_export_control_dataset.py`) and admission gates. No new substrate.
- **Cons:** ceiling is the bag-of-features representation. You can't out-reason
  the features you hand it.
- **Reward-hacking watch:** every adapter must pass the same
  `sp80_sam_contra_admission_gate` / acceptance gate. No metric is "won" until
  it survives the gate on held-out levels.

## 3. Route B — Import tiny transformer LMs as teachers (where X-Token literally applies)

Bring in genuinely small pretrained LMs (SmolLM-135M, Qwen3-0.6B, Llama-3.2-1B)
as **teachers** that read the serialized state as text and reason in tokens;
distill them into the student. Cross-tokenizer KD (X-Token) becomes relevant
*only if you fuse multiple teacher families* (Qwen + Llama tokenizers differ).

- **X-Token relevance:** direct — this is its home turf (multi-teacher, mixed
  tokenizers → one student).
- **Pros:** real language/reasoning prior; multi-teacher fusion is the paper's
  demonstrated +3.82 result.
- **Cons:** you're importing an entire LM substrate CONTRA deliberately does not
  have. Heavy. And the central risk: an LM that *narrates* control without
  *being* control — exactly the false-success failure mode this project fired a
  prior agent over. A fluent explanation is not an admitted action.
- **Verdict:** highest payoff *and* highest risk-of-self-deception. Only worth it
  if the bag-of-features ceiling (Route A) is provably the thing blocking us.

## 4. Route C — Cross-representation distillation (the real generalization of X-Token)

The transferable insight from the paper isn't "tokenizers" — it's: *distill
between models whose output spaces don't line up, using a sparse projection
matrix seeded from a known structural mapping, instead of throwing away the
non-overlapping part.*

CONTRA's analog: our candidate teachers emit at **different abstraction levels** —
Oracle assumptions, primitive choices, sub-step motor commands, perception
hypotheses. A **learned sparse projection** (seeded from the
primitive/affordance registry the way X-Token seeds from tokenizer merge rules)
could map a teacher's richer or differently-shaped output space onto the
student's `CONTROL_PRIMITIVES` — *without* the over-conservative "only match
what aligns 1:1" loss that drops the uncommon-but-important cases.

- **X-Token relevance:** conceptual, adapted — this is the genuinely novel riff.
- **Pros:** plays to CONTRA's actual structure (registries, affordance
  hypotheses) instead of bolting on an LM. The "uncommon-token suppression"
  failure has a direct analog: rare-but-decisive control primitives getting
  suppressed during training.
- **Cons:** unproven; we'd be writing the method, not citing it. Needs a clean
  definition of the seed projection from the primitive/affordance registry.
- **Verdict:** the most *interesting* idea here and the best fit, but it's
  research, not integration.

## 5. Route D — Soup then re-sharpen (merge + per-regime LoRA)

Weight-average a few tiny bases into one soup, then LoRA per regime on top to
recover the peaks the soup smoothed out. A pragmatic middle of A and the merging
literature.

- **Pros:** one base to serve; LoRA deltas are tiny.
- **Cons:** soup quality depends on the bases being in compatible weight
  basins — fragile across different pretraining. For our current linear head
  this is near-trivial; for transformer bases it's the usual merge headaches.

---

## 6. What would have to be TRUE before any of B/C/D is worth starting

1. **A measured ceiling.** Route A (cheap) exhausted and *demonstrably* capped by
   the bag-of-features representation on held-out levels. Don't import an LM to
   solve a problem better features would fix.
2. **A held-out metric suite** the admission gate already trusts — so "improved
   every metric" is a gate verdict on unseen levels, never a training-curve
   claim. (Cardinal rule: no success without a reproducible artifact.)
3. **GPU budget reserved.** This is GPU-only work; no CPU diversion.
4. **A control-vs-narration firewall** for Route B: the LM's output must be
   forced through admission, never accepted as action because it "sounds right."

## 7. Recommendation (given we likely won't go this route)

If we ever do: **Route A first** (it's real, cheap, and may close the gap
alone), and keep **Route C** on the shelf as the one genuinely novel idea worth a
spike *if and only if* Route A hits a measured ceiling. Routes B and D are only
justified by evidence we don't have yet.

The seed instinct — *small specialists + LoRA + principled fusion* — is sound.
The "every metric, free" framing is not. The X-Token paper is the right
inspiration but the wrong literal tool for a head that doesn't tokenize.
