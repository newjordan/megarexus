#!/usr/bin/env bash
# Overnight Path-B autopilot: collect → train → level_eval → keep best → repeat.
# Survives logout when launched under nohup/systemd.
set -u
set -o pipefail

REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
PY="${PY:-python3}"
ROOT_OUT="${ROOT_OUT:-$REPO/legal_search/runs/overnight_level}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$ROOT_OUT/logs"
STATE="$ROOT_OUT/state.json"
BEST_DIR="$ROOT_OUT/best"
mkdir -p "$LOG_DIR" "$BEST_DIR" "$ROOT_OUT/cycles"
cd "$REPO" || exit 1

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

MASTER_LOG="$LOG_DIR/overnight_${STAMP}.log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo "=== overnight level loop start $STAMP pid=$$ ==="
echo "repo=$REPO py=$PY out=$ROOT_OUT"

# How long to run (default ~8h). Override: OVERNIGHT_HOURS=10
HOURS="${OVERNIGHT_HOURS:-8}"
DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
CYCLE=0
BEST_SYNTH_RATE="0"
BEST_CKPT=""

write_state() {
  "$PY" -B - <<PY
import json, time
from pathlib import Path
p = Path("$STATE")
data = {
  "schema": "legal_search.overnight_level.v1",
  "pid": $$,
  "started": "$STAMP",
  "cycle": $CYCLE,
  "deadline_unix": $DEADLINE,
  "best_synth_rate": float("$BEST_SYNTH_RATE"),
  "best_ckpt": """$BEST_CKPT""",
  "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "status": "$1",
}
p.write_text(json.dumps(data, indent=2) + "\n")
print("[state]", data)
PY
}

trap 'write_state stopped; echo "=== overnight stopped cycle=$CYCLE ==="' EXIT

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  CYCLE=$((CYCLE + 1))
  SEED=$((7 + CYCLE * 17))
  # Alternate intensity: odd=medium, even=heavier (explicit knobs — no regime max() clobber)
  if [ $((CYCLE % 2)) -eq 1 ]; then
    REGIME_ARGS=(
      --level-shaped --force-collect
      --collect-episodes 300 --arcade-episodes 16 --max-actions 72
      --epochs 50 --batch-size 96 --lr 1e-3
      --policy-coef 0.85 --levels-up-boost 8 --lora-only-after 6
    )
    TAG="level_m"
  else
    REGIME_ARGS=(
      --level-shaped --force-collect
      --collect-episodes 500 --arcade-episodes 28 --max-actions 80
      --epochs 80 --batch-size 128 --lr 8e-4
      --policy-coef 0.9 --levels-up-boost 10 --lora-only-after 8
    )
    TAG="level_h"
  fi
  OUT="$ROOT_OUT/cycles/c$(printf '%03d' "$CYCLE")_${TAG}_s${SEED}"
  mkdir -p "$OUT"
  write_state running
  echo ""
  echo "======== CYCLE $CYCLE TAG=$TAG SEED=$SEED OUT=$OUT ========"
  echo "remaining_sec=$(( DEADLINE - $(date +%s) ))"

  if ! "$PY" -B legal_search/train/train_lora.py \
      --out-dir "$OUT" \
      --seed "$SEED" \
      "${REGIME_ARGS[@]}" \
      2>&1 | tee "$OUT/train.log"
  then
    echo "[warn] train failed cycle=$CYCLE (continue)"
    echo "fail" > "$OUT/STATUS"
    continue
  fi

  CKPT="$OUT/best.pt"
  if [ ! -f "$CKPT" ]; then
    echo "[warn] no best.pt cycle=$CYCLE"
    echo "no_ckpt" > "$OUT/STATUS"
    continue
  fi

  if ! "$PY" -B legal_search/train/level_eval.py \
      --ckpt "$CKPT" \
      --synth-eps 40 \
      --arcade-eps 3 \
      --out "$OUT/level_eval.json" \
      2>&1 | tee "$OUT/level_eval.log"
  then
    echo "[info] level_eval exit nonzero (synth may still have wins)"
  fi

  # Parse synth rate + arcade any
  read -r RATE ARCADE < <("$PY" -B - <<PY
import json
from pathlib import Path
p = Path("$OUT/level_eval.json")
if not p.is_file():
    print("0 0")
    raise SystemExit
d = json.loads(p.read_text())
r = float((d.get("synthetic") or {}).get("level_up_rate") or 0.0)
a = 1 if d.get("any_arcade_level") else 0
print(f"{r:.6f} {a}")
PY
)

  echo "[cycle $CYCLE] synth_level_rate=$RATE arcade_level=$ARCADE"
  echo "ok rate=$RATE arcade=$ARCADE" > "$OUT/STATUS"

  # Keep best by synth rate; always promote if arcade hits a level
  PROMOTE=0
  "$PY" -B - <<PY
rate = float("$RATE")
best = float("$BEST_SYNTH_RATE")
arcade = int("$ARCADE")
promote = 1 if (arcade or rate > best + 1e-9) else 0
print(promote)
open("$OUT/.promote", "w").write(str(promote))
PY
  if [ "$(cat "$OUT/.promote")" = "1" ]; then
    BEST_SYNTH_RATE="$RATE"
    BEST_CKPT="$CKPT"
    cp -f "$CKPT" "$BEST_DIR/best.pt"
    cp -f "$OUT/level_eval.json" "$BEST_DIR/level_eval.json" 2>/dev/null || true
    cp -f "$OUT/metrics.json" "$BEST_DIR/metrics.json" 2>/dev/null || true
    cp -f "$OUT/config.json" "$BEST_DIR/config.json" 2>/dev/null || true
    echo "[promote] new best synth_rate=$BEST_SYNTH_RATE arcade=$ARCADE -> $BEST_DIR/best.pt"
    if [ "$ARCADE" = "1" ]; then
      echo "[MILESTONE] arcade level-up achieved cycle=$CYCLE" | tee -a "$ROOT_OUT/MILESTONES.log"
    fi
  fi

  # Brief cooldown so disks settle; avoid tight spin if something fails fast
  sleep 15
done

write_state finished
echo "=== overnight finished cycles=$CYCLE best_synth=$BEST_SYNTH_RATE best=$BEST_CKPT ==="
echo "master_log=$MASTER_LOG"
