# Adding New Parts to the Megarexus CNN

How to extend `synthesis_lab/experiments/megarexus_cnn/` (model heads,
data collectors, labels) without losing what already works and without
cheating. Every rule below was earned in a real run — citations are to
`synthesis_lab/notes/RESEARCH_LOG.md` and the run dirs.

## The cycle: one element per run

1. **One new element at a time** — a head, a label source, OR a data
   curriculum. Never two, or you can't attribute the change.
2. **Collect → verify the shard → train → gate → probe.** Never skip the
   shard verification: load it through the real `dataset.py` and count
   what the new head will actually see (positives, axes, provenance).
3. **A new head is only as good as its data coverage** (see below) —
   plan the collector and the head together.

## Honesty rules (non-negotiable, HANDOFF.md §2)

- Labels come from **environment fields + pixels in hindsight** only.
  Never from hand-authored game models, engine internals, or sprite tags.
- Human demonstrations ARE legal experience. Replaying them — and
  replaying them **with noise** (demo_jitter) to diversify positives —
  is legal; tag every trajectory's `policy` so provenance is explicit.
  Demonstration-derived results are never model capability.
- Model checkpoints may drive **collection policy** (reaching deeper
  levels), never labels.

## Gates: "no loss" is measured, not assumed

- **Same-valset gate.** Val shards change between runs, so headline
  numbers aren't comparable. Evaluate the new checkpoint on the
  *previous release's val shards* and compare head-by-head (v5's catch
  "drop" was 1 borderline sample at logit +0.23; v4 meanwhile was
  confidently wrong 16× on new data).
- **Behavioral gate.** Val metrics can hold while behavior collapses:
  outcome_v6 held every metric and went 20/20 → 0/20 on the level
  probe. Always re-run `orchestration/level_probe.py` (paired seeds,
  random baseline must reproduce) and the relevant solver probe.
- **Checkpoints:** train.py saves `best.pt` (world-model criterion) AND
  `last.pt`; the *release* is chosen by the external gates, not by any
  training-time heuristic (best.pt once froze at epoch 12 while every
  head kept improving to 40).

## The three failure modes we actually hit

1. **Label leakage through encoding.** `levels_completed` is stored
   post-action; comparing it across rows as if frame-level made the
   loader self-stack `[cur, cur]` exactly on positive pour rows — the
   label leaked into input *structure*, and the level head "passed" val
   at 23/23 while scoring the same pixels honestly stacked at −14.7.
   Rule: whenever you add a label, check that positives and negatives
   are encoded through the *identical* path, and that probes use that
   same path. If a metric looks perfect, try to break it from outside
   the dataloader before believing it.
2. **Unsupervised regions get hallucinated, and probes will find them.**
   Masking completing-pour dynamics (v6) left "pour at completion"
   unsupervised → imagination extrapolated a 1,221-cell scene swap →
   probe declined every pour. The collector never pouring without
   arranging (v7) left the pristine frame unsupervised → the level head
   scored it +9.8 → the planner chose the null plan forever. Rule:
   enumerate the states your probe/planner will query and make sure the
   training data contains **negatives there too** (`--pour-first-prob`).
3. **Heads memorize when positives are one frame.** 12 pixel-identical
   demo positives → a lookup table, not a concept. Diversity comes
   cheap: jittered demo replays gave 13 completions with 12 distinct
   arrangements in 300 episodes. Aim for positives that *span the
   basin*, not repeat the point.

## Regime facts that bite collectors

- **Selection recolors movables** (clicked → color 9, deselected → 8).
  Track components by **click-point and position+area**, never by color
  (`collect_plumbing_play.relocate`, `component_at`).
- Env fields are **post-action**; the recorded frame is **pre-action**
  (`EpisodeRecorder`). Frame layout at row j belongs to row j−1's
  fields. Level transitions swap the scene *inside* the animation ticks
  — the pour→first-tick pair is still a valid dynamics target.
- Pours are the scarce resource (~4/level, next call = GAME_OVER);
  moves are cheap. Spend moves, hoard pours. Gauge reads are unreliable
  off level 1 (row 63, color→color drain): budget by action count.
- Control samples are free: any move row spawns them for all four axes
  (`MOVE_ACTION_AXIS`) — if a control-ish head lacks data, the gap is
  almost always the *collector's action mix*, not dataset code.

## Model-side conventions

- Frame stack: current frame first (`channels [0, 22)`), previous frame
  after — `dataset._encode_stack`, `eval_ghost.encode`; every probe and
  the model comment agree. Never invent a second stacking order.
- New heads follow the existing patterns: coarse 16×16 bbox plane +
  masked pooling for per-component questions (`forward_catch`,
  `forward_control`); trunk pooling for global questions
  (`forward_settle`). Plane convention: bbox coords ÷ 4.
- Give every trained element a base rate, an unseen-seed val slice, and
  a paired behavioral baseline before claiming anything.

## Quick reference

| step | command |
|---|---|
| stack sanity | `/run-synthesis-lab` driver `preflight` |
| collect | `orchestration/collect_plumbing_play.py --run-dir <release> ...` |
| train | `experiments/megarexus_cnn/train.py --out-dir runs/<name> ...` (recipe: recover exact shard list from the previous release's `ckpt["config"]`) |
| gate | eval `best.pt`+`last.pt` on the previous release's val shards |
| behavior | `orchestration/level_probe.py` (20/20 + 5/20 must reproduce) |
| solver | `orchestration/arrange_probe.py` (paired, null plan must lose) |

Probes load `<run-dir>/best.pt` by name — to probe `last.pt`, symlink it
as `best.pt` in a scratch dir; never point probes' `--run-dir` at the
canonical run dir (they write reports into it).
