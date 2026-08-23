"""build_hotpot_corpus.py: build the HotpotQA scaling corpus from public data.

1. Stream the BEIR HotpotQA corpus (5,233,329 Wikipedia passages) from the Hugging Face hub; each
   passage is embedded as "title\ntext".
2. Write embeddings in resumable chunks of 100,000 (embeddings_chunk_<i>.npy).
3. Shuffle once (seed 0) and L2-normalise into one array (bench/make_hotpot_corpus.py), so every
   prefix is a uniform sample and the last 1,000 rows can be held out as queries.

    python bench/data/build_hotpot_corpus.py --out hotpot_5M.npy

Use a GPU if you can: 5.2M passages take tens of minutes on a recent GPU and many hours on CPU.
--limit builds a small prefix for testing.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from minilm import MiniLM, batched  # noqa: E402

CHUNK = 100_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", default=os.path.join(HERE, "hotpot_chunks"))
    ap.add_argument("--out", default="hotpot_5M.npy")
    ap.add_argument("--limit", type=int, default=0, help="only the first N passages")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--no-shuffle", action="store_true",
                    help="stop after writing the chunks")
    a = ap.parse_args()

    from datasets import load_dataset

    os.makedirs(a.chunks, exist_ok=True)
    corpus = load_dataset("BeIR/hotpotqa", "corpus", split="corpus", streaming=True)
    enc = MiniLM()

    def texts():
        for i, doc in enumerate(corpus):
            if a.limit and i >= a.limit:
                return
            title, text = doc.get("title", ""), doc.get("text", "")
            yield f"{title}\n{text}" if title else text

    for ci, chunk in enumerate(batched(texts(), CHUNK)):
        path = os.path.join(a.chunks, f"embeddings_chunk_{ci}.npy")
        if os.path.exists(path):
            continue                      # resumable: already embedded
        E = enc.encode(chunk, batch=a.batch)
        np.save(path + ".partial.npy", E)
        os.replace(path + ".partial.npy", path)
        print(f"chunk {ci}: {len(chunk):,} passages", flush=True)

    if not a.no_shuffle:
        subprocess.check_call([sys.executable,
                               os.path.join(os.path.dirname(HERE), "make_hotpot_corpus.py"),
                               "--src", a.chunks, "--out", a.out])


if __name__ == "__main__":
    main()
