#!/usr/bin/env bash
# Linux / WSL2 setup for the comparison: a venv with the baselines, and this repository built in
# release mode into it. On WSL2, clone the repository into the Linux filesystem first; building
# from /mnt/c or /mnt/d is very slow.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
W=${FAIR_DIR:-$HOME/maivs-fair}
mkdir -p "$W"
python3 -m venv "$W/.venv"
. "$W/.venv/bin/activate"
pip install -q numpy maturin faiss-cpu==1.15.1 scann==1.4.2 hnswlib==0.8.0 usearch==2.26.5
cd "$REPO"
git log -1 --format="building %h %s"
maturin develop --release
python -c "import maivs, faiss; print('maivs ok; faiss', faiss.__version__, faiss.get_compile_options())"
