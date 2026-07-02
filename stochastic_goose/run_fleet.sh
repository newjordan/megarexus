#!/usr/bin/env bash
# Beast mode (CPU): run all 25 ARC-AGI-3 games (x N seeds) concurrently as
# single-threaded StochasticGoose training procs, fully OFFLINE.
#
# StochasticGoose is a tiny CNN bottlenecked by the env-step loop -> CPU-bound,
# GPU ~0%. So we force CPU and pin 1 math-thread per proc; otherwise each proc
# spawns ~96 torch threads and N procs thrash the box (load >> cores).
#
# Usage:  ARC_BASE=/opt/arc bash run_fleet.sh [MAX_ACTIONS] [SEEDS_PER_GAME]
set -u

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

BASE=${ARC_BASE:-/opt/arc}
MAXA=${1:-40000}
SEEDS=${2:-1}
CORPUS=${CORPUS:-$BASE/corpus}
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUT=$BASE/fleet_$STAMP
mkdir -p "$OUT"
PY=${PY:-$(command -v python)}
GPU=${GPU:-0}; NGPU=${NGPU:-1}   # GPU=1 pins each proc to a card (round-robin over NGPU); else CPU

GAMES=(ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 \
       sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30)

echo "FLEET(cpu) $STAMP | ${#GAMES[@]} games x $SEEDS seed(s) | $MAXA actions | OFFLINE | out=$OUT"
n=0
for g in "${GAMES[@]}"; do
  for s in $(seq 0 $((SEEDS - 1))); do
    tag="${g}_s${s}"; gd="$OUT/$tag"; mkdir -p "$gd"
    if [ "$GPU" = "1" ]; then DEV=$(( n % NGPU )); else DEV=""; fi
    (
      cd "$gd" || exit 1
      CUDA_VISIBLE_DEVICES="$DEV" ARC_FRAMEWORK="$BASE/ARC-AGI-3-Agents" \
        "$PY" "$BASE/stochastic-goose/run_local.py" \
        --game "$g" --max-actions "$MAXA" --env-dir "$CORPUS" --quiet \
        > "$gd/out.log" 2>&1
    ) &
    n=$(( n + 1 ))
  done
done
echo "launched $n CPU procs; waiting..."
wait
echo "=== FLEET DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
{
  for g in "${GAMES[@]}"; do
    best=$(grep -rhasE "\[$g\] finished" "$OUT/${g}"_s*/out.log 2>/dev/null \
           | grep -oE 'levels_completed=[0-9]+' | grep -oE '[0-9]+' | sort -rn | head -1)
    echo "$g: best_levels_completed=${best:-NA}"
  done
} | tee "$OUT/SUMMARY.txt"
echo "--- games clearing >=1 level ---"
grep -avE 'best_levels_completed=(0|NA)' "$OUT/SUMMARY.txt" || echo "(none yet)"
