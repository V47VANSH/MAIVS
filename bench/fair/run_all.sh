#!/usr/bin/env bash
# Run every system at 250k and 500k, one process each, stopping early if free memory or disk
# gets low. A system whose result file already exists is skipped, so it can just be re-run.
#
#   FAIR_DIR   work directory with .venv, data/ and results/ (default ~/maivs-fair)
#   FAIR_DISK  filesystem to watch for free space (default FAIR_DIR; on WSL2 use /mnt/c,
#              since the virtual disk grows on the Windows drive)
#   FAIR_OS    description of the OS for environment.json (default: platform.platform())
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
W=${FAIR_DIR:-$HOME/maivs-fair}
DISK=${FAIR_DISK:-$W}
. "$W/.venv/bin/activate"
export OMP_NUM_THREADS=16 RAYON_NUM_THREADS=16 PYTHONUNBUFFERED=1
mkdir -p "$W/results" "$W/logs"
LOG=$W/logs/run_all.log

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

REPO="$REPO" python - <<'EOF' > "$W/results/environment.json"
import json, os, platform, subprocess
from importlib.metadata import version
import faiss, numpy
cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
print(json.dumps({"cpu": cpu, "threads": 16, "kernel": platform.release(), "python": platform.python_version(),
  "numpy": numpy.__version__, "faiss": faiss.__version__, "faiss_compile": faiss.get_compile_options(),
  "scann": version("scann"), "usearch": version("usearch"), "hnswlib": version("hnswlib"),
  "rustc": subprocess.run(["bash", "-lc", "rustc --version"], capture_output=True, text=True).stdout.strip(),
  "maivs_commit": subprocess.run(["git", "-C", os.environ["REPO"], "log", "-1", "--format=%h"],
                                 capture_output=True, text=True).stdout.strip(),
  "os": os.environ.get("FAIR_OS", platform.platform())}, indent=1))
EOF

SYSTEMS="maivs-quality maivs-memory faiss-hnsw hnswlib usearch scann scann-soar faiss-ivf-pqfs faiss-ivf-rabitqfs faiss-ivf-rabitqfs4"
for N in 250000 500000; do
  for S in $SYSTEMS; do
    if [ -f "$W/results/${S}_${N}.json" ]; then echo "[skip] $S $N" | tee -a "$LOG"; continue; fi
    guard
    python "$HERE/fairbench.py" $S $N 2>&1 | grep -v -E "^(WARNING|I0000|W0000)" | tee -a "$LOG"
    sleep 10   # give the laptop's clocks a moment between systems
  done
done
# drift check: MAIVS and faiss-HNSW again at 500k, in fresh processes, at the end
for S in maivs-quality faiss-hnsw; do
  if [ -f "$W/results/${S}_500000_repeat.json" ]; then continue; fi
  guard
  python "$HERE/fairbench.py" $S 500000 repeat 2>&1 | tee -a "$LOG"
done
guard
echo "[all done] $(date +%H:%M:%S)" | tee -a "$LOG"
