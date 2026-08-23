"""build_msmarco_corpus.py: the MSMARCO passage sample used in the benchmarks.

Same construction as the original cache files: from msmarco-passage/train take the first Q training
queries with relevance judgements, put every judged passage first, then fill with passages in corpus
order up to N; embed each passage with MiniLM (L2-normalised).

    python bench/data/build_msmarco_corpus.py --n 1000000
        -> msmarco_embeddings_cache_v2_1000000.npy

Needs ir_datasets (downloads the MSMARCO passage collection, ~3 GB).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from minilm import DIM, MiniLM  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--out", default="")
    ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    out = a.out or f"msmarco_embeddings_cache_v2_{a.n}.npy"

    import ir_datasets
    ds = ir_datasets.load("msmarco-passage/train")
    qrels = ds.qrels_dict()
    golden = []
    seen = set()
    taken = 0
    for q in ds.queries_iter():
        if q.query_id in qrels:
            for did in qrels[q.query_id]:
                if did not in seen:
                    seen.add(did)
                    golden.append(did)
            taken += 1
            if taken >= a.queries:
                break

    golden_text, filler = {}, []
    need = max(0, a.n - len(golden))
    for doc in ds.docs_iter():
        if doc.doc_id in seen:
            golden_text[doc.doc_id] = doc.text
        elif len(filler) < need:
            filler.append(doc.text)
        if len(filler) >= need and len(golden_text) == len(golden):
            break
    texts = [golden_text[d] for d in golden if d in golden_text] + filler

    enc = MiniLM()
    E = np.lib.format.open_memmap(out + ".partial", mode="w+", dtype=np.float32,
                                  shape=(len(texts), DIM))
    step = 100_000
    for s in range(0, len(texts), step):
        E[s:s + step] = enc.encode(texts[s:s + step], batch=a.batch)
        print(f"{min(s + step, len(texts)):,}/{len(texts):,}", flush=True)
    E.flush()
    del E
    os.replace(out + ".partial", out)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
