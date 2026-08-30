#!/usr/bin/env bash
# The memory-matched, churn and filtered experiments, one process per run, with the same memory
# and disk guard as run_all.sh. Runs whose result file exists are skipped.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
W=${FAIR_DIR:-$HOME/maivs-fair}
DISK=${FAIR_DISK:-$W}
. "$W/.venv/bin/activate"
export OMP_NUM_THREADS=16 RAYON_NUM_THREADS=16 PYTHONUNBUFFERED=1
mkdir -p "$W/results" "$W/logs"
LOG=$W/logs/run_edges.log
NOISE="^(WARNING|I0000|W0000)"

guard() {
  local avail_mb free_gb
  avail_mb=$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)
  free_gb=$(df -BG --output=avail "$DISK" | tail -1 | tr -dc 0-9)
  echo "[guard] $(date +%H:%M:%S) MemAvailable ${avail_mb} MB, ${free_gb} GB free on $DISK" | tee -a "$LOG"
  if [ "$avail_mb" -lt 2500 ] || [ "$free_gb" -lt 5 ]; then
    echo "[guard] stopping: need 2500 MB of RAM and 5 GB of disk" | tee -a "$LOG"
    exit 3
  fi
}

run() {   # run <result file> <command...>
  local out=$1; shift
  if [ -f "$W/results/$out" ]; then echo "[skip] $out" | tee -a "$LOG"; return; fi
  guard
  "$@" 2>&1 | grep -v -E "$NOISE" | tee -a "$LOG"
  sleep 10
}

echo "=== 1. memory-matched (~600 B/vector), 500k" | tee -a "$LOG"
for S in scann-soar-int8 scann-int8 faiss-ivf-pqfs-sq8 faiss-hnsw-sq8; do
  run ${S}_500000.json python "$HERE/fairbench.py" $S 500000
done
echo "=== 2. churn: 25% turnover of a 400k index" | tee -a "$LOG"
# no scann-soar here: ScaNN 1.4.2 crashes while updating a SOAR index (scann_soar_crash.py)
for S in maivs maivs-memory scann hnswlib faiss-hnsw; do
  run churn_${S}.json python "$HERE/churn.py" $S
done
echo "=== 3. filtered search, 500k" | tee -a "$LOG"
for S in bruteforce maivs scann-soar faiss-hnsw; do
  run filtered_${S}.json python "$HERE/filtered.py" $S
done
guard
echo "[edges done] $(date +%H:%M:%S)" | tee -a "$LOG"
