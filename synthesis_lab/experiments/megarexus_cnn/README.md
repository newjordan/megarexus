# megarexus_cnn — SCOUT_EXPERIMENT

One valid CNN synthesizing LeWM/Rex (action-conditioned world model, as
FiLM-conditioned conv dynamics), Contra (perception planes as input
channels), and Stochastic Goose (4101-logit policy/change head + stochastic
sampler). See `../../manifests/megarexus_cnn_manifest.md` for roles, data
provenance, and anti-cheating provisions.

## Files

- `model.py` — `MegarexusCNN`: conv trunk → (a) FiLM dynamics → next-frame
  logits, (b) Goose twin heads (policy + change), (c) inverse-dynamics heads.
- `dataset.py` — human corpus PNG decoding + rollout shards; episode-split
  train/val.
- `train.py` — multi-objective training + honest baseline-anchored eval.
- `../../orchestration/collect_sp80_rollouts.py` — Goose-driven local SP80
  rollout collection (OFFLINE).

- `eval_ghost.py` — ghost-overlay confirmation pass: H-step imagined
  rollouts vs reality on flow segments + lock/key asymmetry seeker probe.

## Run (v1 flow regime — the one that learns paint physics)

```bash
# flow-biased collection; ACTION5 animation frames become tick transitions
python synthesis_lab/orchestration/collect_sp80_rollouts.py \
    --out synthesis_lab/experiments/megarexus_cnn/data/sp80_flow_v2.pt \
    --rollouts 60 --max-steps 200 --p-flow 0.3 --p-click 0.2
# 2-frame stack so paint velocity is visible
python synthesis_lab/experiments/megarexus_cnn/train.py --stack-frames 2 \
    --out-dir synthesis_lab/experiments/megarexus_cnn/runs/flow_v1 \
    --rollout-shards synthesis_lab/experiments/megarexus_cnn/data/*.pt
# ghost confirmation on fresh unseen-seed physics + human val
python synthesis_lab/experiments/megarexus_cnn/eval_ghost.py \
    --run-dir synthesis_lab/experiments/megarexus_cnn/runs/flow_v1 \
    --rollout-val-shard synthesis_lab/experiments/megarexus_cnn/data/sp80_flow_valset.pt
```

Results: `runs/{scout_v0,flow_v1}/`; summary in
`../../notes/megarexus_cnn_scout.md`.
