"""make_hotpot_corpus.py: assemble the HotpotQA chunks into one scaling corpus.

Two things the scaling study depends on:

Shuffling. The chunks are in ingestion order, so neighbouring rows often come from the same
document. The study takes the first N rows at each scale, and on ordered data small scales would be
a topically narrow corpus instead of a smaller sample. So the rows are shuffled once here and every
prefix is a uniform sample.

Normalisation. Everything downstream is inner product on unit vectors; normalising once on disk
means no benchmark has to build a normalised copy of an 8 GB array.

The last --queries rows are held out as queries at every scale, which keeps curves comparable across
N.

    python bench/make_hotpot_corpus.py --out hotpot_5M.npy

(Usually run from bench/data/build_hotpot_corpus.py, which embeds the BEIR HotpotQA corpus into the
chunks this script reads.)
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import time

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hotpot_chunks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default="hotpot_5M.npy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", type=int, default=131072)
    a = ap.parse_args()

    files = glob.glob(os.path.join(a.src, "embeddings_chunk_*.npy"))
    files.sort(key=lambda s: int(re.findall(r"(\d+)\.npy", s)[0]))
    if not files:
        raise SystemExit(f"no chunks under {a.src}")

    mms = [np.load(f, mmap_mode="r") for f in files]
    counts = [m.shape[0] for m in mms]
    starts = np.cumsum([0] + counts)
    n, d = int(starts[-1]), mms[0].shape[1]
    print(f"[corpus] {len(files)} chunks, {n:,} vectors, d={d}, "
          f"{n * d * 4 / 1e9:.2f} GB")

    if os.path.exists(a.out):
        print(f"[corpus] {a.out} exists - leaving it alone")
        return

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(n)

    tmp = a.out + ".partial"
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                    shape=(n, d))
    t0 = time.time()
    for s in range(0, n, a.block):
        e = min(s + a.block, n)
        src = perm[s:e]
        # Sorting the source indices inside each block turns a random gather into a mostly
        # sequential read; the destination order stays random, which is all the shuffle needs.
        order = np.argsort(src)
        srt = src[order]
        buf = np.empty((e - s, d), dtype=np.float32)
        ci = np.searchsorted(starts, srt, side="right") - 1
        for c in np.unique(ci):
            sel = ci == c
            buf[sel] = mms[c][srt[sel] - starts[c]]
        # undo the sort so rows land at their permuted destinations
        back = np.empty_like(order)
        back[order] = np.arange(len(order))
        buf = buf[back]
        nrm = np.linalg.norm(buf, axis=1, keepdims=True)
        np.divide(buf, np.maximum(nrm, 1e-12), out=buf)
        out[s:e] = buf
        if (s // a.block) % 8 == 0:
            done = e / n
            el = time.time() - t0
            print(f"  {e:>9,}/{n:,}  {done * 100:5.1f}%  "
                  f"{el:6.0f}s elapsed, ~{el / max(done, 1e-9) - el:5.0f}s left",
                  flush=True)
    out.flush()
    del out, mms
    os.replace(tmp, a.out)
    print(f"[corpus] wrote {a.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
