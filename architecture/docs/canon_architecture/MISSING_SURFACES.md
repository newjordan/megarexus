# Canon Addendum — 2026-05-25

## Authoritative Sources (from code)

### Rexus Core Loop (`src/loom_layers/rexus_core_loop.py:1-5`)
Canonical Rexus action-loop scaffold. Defines StatePacket, TopologyPacket, MemoryPacket, ActionCandidate, CandidateScore, TransitionPrediction. Hard parts (perception, LeWM prediction, LPRM scoring, lattice pruning, engine execution, memory updates) injected as hooks. Missing hooks fail closed.

### O_pred Row Contract (`src/loom_layers/rexus_opred_rows.py:1-17`)
Recorded offline `S_t, A, O_pred, O_true` rows. Phase 0-only foundation surface. Not Q6.
Checkpoint path at line 15: `experiments/2026-05-23_phase0_production_rtx6k/bs64_lr1e-4/best.pt`
SHA256: `ce1b562cccaf68c39d9af7f18924bf796ac1e353f933428a8d38926cf677ff00`

### Transition Scalar (`experiments/2026-05-24_rexus_transition_scalar_v0/`)
Proper O_pred pipeline:
- Source: brute-force recorded pkl (`remote_bad_runs/bruteforce_round6/cn04-2fe56bfb_bruteforce.pkl`)
- Alignment: `frames[i] -- actions[i+1] --> frames[i+1]`
- Outputs: `rexus_opred_rows.jsonl`, `state_prober_results.jsonl`, `scalar_rows.jsonl`, `state_prober_latent_bible.json`
- Runner: `run.py --device cpu --max-rows 16 --sequence-len 4`

## New Since Last Update

### Groundhog Encoder Training
- Script: `lab/scripts/run_groundhog_encoder.py`
- Output: `experiments/groundhog_encoder_v1/`
- Result: 1.5x family separation improvement (67.6x → 100.9x)
- Only 37K trainable params (projector + last layer)
- 9 families, 68K frames

### Pairwise Action Ranker (on Groundhog latents)
- Script: `lab/scripts/run_pairwise_groundhog.py`
- 24.5K human preference pairs from branched data
- Metamorphic check (color shift) for memorization detection
- Output: `lab/gepa/gepa_pairwise_population.jsonl`

### C1 Frozen-Encoder Training
- Script: `lab/scripts/c1_synthetic_smoke.py`, `c1_scaled.py`
- Result: PASSED — frozen encoder + train predictor = 1000× improvement with zero regression
- Output: `experiments/c1_scaled_v1/`, `experiments/c1_synthetic_smoke_v2/`

### RTX PRO 6000 Deployment
- Instance: `ssh -p 40493 root@116.120.230.77`
- GPU: 96GB VRAM, CUDA 13.2, PyTorch 2.13 nightly cu132
- Status: Pairwise ranker training on Groundhog encoder
