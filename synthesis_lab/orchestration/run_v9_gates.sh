#!/bin/bash
# One-shot mechanical gate chain for outcome_v9 (HANDOFF.md §0 steps 1-2).
# Waits for training to exit, then: gate_eval x3 (same-valset no-loss vs v8),
# level_probe + arrange_probe on v9 last.pt (and best.pt if it differs).
# Judgment (head-by-head compare, release decision, docs) stays with the agent.
set -u
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
D=$REPO/synthesis_lab/experiments/megarexus_cnn
V9=$D/runs/outcome_v9
V8=$D/runs/outcome_v8
GATES=$D/runs/outcome_v9_gates
TRAINLOG="${TRAINLOG:-$V9/train.log}"
mkdir -p "$GATES"
cd "$REPO"

stamp() { echo "[$(date '+%H:%M:%S')] $1" >> "$GATES/timeline.txt"; }

stamp "waiting for train.py to exit"
until ! pgrep -f 'train.py --rollout' >/dev/null; do sleep 60; done
stamp "training process gone"
tail -n 6 "$TRAINLOG" > "$GATES/train_log_tail.txt" 2>&1

if [ ! -f "$V9/last.pt" ]; then stamp "FATAL: no last.pt"; exit 1; fi

# --- Gate 1: same-valset no-loss, both v9 checkpoints + v8 baseline, on v8's valset
for spec in "v9last:$V9/last.pt" "v9best:$V9/best.pt" "v8last:$V8/last.pt"; do
  name=${spec%%:*}; ckpt=${spec#*:}
  stamp "gate_eval $name start"
  $PY -B synthesis_lab/orchestration/gate_eval.py --ckpt "$ckpt" \
    --val-shards-from "$V8/last.pt" > "$GATES/gate_eval_$name.txt" 2>&1
  stamp "gate_eval $name done (rc=$?)"
done

# --- Gate 2: behavioral probes; scratch run dirs, best.pt symlinked by name
cands="last"
cmp -s "$V9/last.pt" "$V9/best.pt" || cands="last best"
stamp "probe candidates: $cands"
for c in $cands; do
  RD=$GATES/proberun_$c
  mkdir -p "$RD"; ln -sf "$V9/$c.pt" "$RD/best.pt"
  stamp "level_probe $c start"
  $PY -B synthesis_lab/orchestration/level_probe.py --run-dir "$RD" \
    --finder learned --out-name level_probe_v9$c.json > "$GATES/level_probe_$c.log" 2>&1
  stamp "level_probe $c done (rc=$?)"
  stamp "arrange_probe $c start"
  $PY -B synthesis_lab/orchestration/arrange_probe.py --run-dir "$RD" \
    --search exhaustive-x --out-name arrange_probe_v9$c.json > "$GATES/arrange_probe_$c.log" 2>&1
  stamp "arrange_probe $c done (rc=$?)"
done

stamp "ALL GATES COMPLETE"
touch "$GATES/DONE"
