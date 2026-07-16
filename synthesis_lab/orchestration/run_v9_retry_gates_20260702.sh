#!/bin/bash
# One-shot mechanical gate chain for outcome_v9_retry_20260702.
# Waits for its train.py process to exit, then runs the same v9 gates
# against outcome_v8. Judgment and release docs stay with the agent.
set -u
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
D=$REPO/synthesis_lab/experiments/megarexus_cnn
V9=$D/runs/outcome_v9_retry_20260702
V8=$D/runs/outcome_v8
GATES=$D/runs/outcome_v9_retry_20260702_gates
TRAINLOG=$D/runs/outcome_v9_retry_20260702_train.log
mkdir -p "$GATES"
cd "$REPO"

stamp() { echo "[$(date '+%H:%M:%S')] $1" >> "$GATES/timeline.txt"; }

stamp "waiting for outcome_v9_retry_20260702 train.py to exit"
until ! pgrep -f 'train.py.*outcome_v9_retry_20260702' >/dev/null; do sleep 60; done
stamp "training process gone"
tail -n 12 "$TRAINLOG" > "$GATES/train_log_tail.txt" 2>&1

if [ ! -f "$V9/last.pt" ]; then stamp "FATAL: no last.pt"; exit 1; fi

for spec in "v9retry_last:$V9/last.pt" "v9retry_best:$V9/best.pt" "v8last:$V8/last.pt"; do
  name=${spec%%:*}; ckpt=${spec#*:}
  stamp "gate_eval $name start"
  $PY -B synthesis_lab/orchestration/gate_eval.py --ckpt "$ckpt" \
    --val-shards-from "$V8/last.pt" > "$GATES/gate_eval_$name.txt" 2>&1
  stamp "gate_eval $name done (rc=$?)"
done

cands="last"
cmp -s "$V9/last.pt" "$V9/best.pt" || cands="last best"
stamp "probe candidates: $cands"
for c in $cands; do
  RD=$GATES/proberun_$c
  mkdir -p "$RD"; ln -sf "$V9/$c.pt" "$RD/best.pt"
  stamp "level_probe $c start"
  $PY -B synthesis_lab/orchestration/level_probe.py --run-dir "$RD" \
    --finder learned --out-name level_probe_v9retry_$c.json > "$GATES/level_probe_$c.log" 2>&1
  stamp "level_probe $c done (rc=$?)"
  stamp "arrange_probe $c start"
  $PY -B synthesis_lab/orchestration/arrange_probe.py --run-dir "$RD" \
    --search exhaustive-x --out-name arrange_probe_v9retry_$c.json > "$GATES/arrange_probe_$c.log" 2>&1
  stamp "arrange_probe $c done (rc=$?)"
done

stamp "ALL GATES COMPLETE"
touch "$GATES/DONE"
