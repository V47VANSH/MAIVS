# Benchmarks

Every accuracy number is recall@10 against exact brute-force ground truth. Throughput is batch QPS
(the whole query set, all threads, best of 3); latency is single-thread p50/p99 including the
Python call, measured the same way for every system.

## Data

| file | how to get it |
|---|---|
| `hotpot_5M.npy` | `python bench/data/build_hotpot_corpus.py` (BEIR HotpotQA, 5,233,329 passages, MiniLM, shuffled and normalised) |
| `msmarco_embeddings_cache_v2_1000000.npy` | `python bench/data/build_msmarco_corpus.py --n 1000000` |
| `msmarco_embeddings_cache_v2_100000.npy` | `... --n 100000` (used by the tests) |
| `bench/data/hotpot_questions_minilm.npy` | `python bench/data/embed_hotpot_questions.py` (1,000 real HotpotQA questions) |

The corpora go in the repository root and are git-ignored.

## Head-to-head

| script | what it does |
|---|---|
| `bench_vs_faiss.py` | this index against faiss-HNSW, USearch, faiss IVF-RaBitQ and IVF-PQ FastScan on the same data, queries and ground truth; Pareto frontier over each system's parameters; build time, memory, QPS, p50/p99 |
| `bench_scale_ours.py` | this index at sizes where the base doesn't fit in RAM next to the index (streams the memory-mapped corpus) |
| `build_hnsw_cache.py` | builds and caches the faiss-HNSW baseline once, since it's the slow part of every comparison |
| `results_tables.py` | renders the tables in `docs/RESULTS_tables.md` from the result JSON |
| `bench_profile.py` | where build and query time go, per phase |
| `bench_polysemy.py` | orthogonal polysemy on and off, end to end |

```bash
python bench/build_hnsw_cache.py hotpot_5M.npy 1000000
python bench/bench_vs_faiss.py hotpot_5M.npy --n 1000000 \
    --questions bench/data/hotpot_questions_minilm.npy \
    --systems ours-quality+poly+fs,ours-memory+poly+fs,hnsw \
    --rfs 8,16,32 --probes 64,128,256,512,1024,2048,4096
python bench/results_tables.py results/*.json > docs/RESULTS_tables.md
```

## Older studies

`bench_scale_full.py`, `bench_scale_hnsw.py`, `bench_recall.py`, `bench_churn.py`,
`bench_filter.py`, `bench_concurrency.py`, `bench_latency.py` and the research scripts
(`bench_attractor.py`, `bench_cellgraph.py`, `bench_knnpart.py`, `bench_ladder.py`,
`bench_multitable.py`, `bench_precomputed.py`, `bench_spill.py`, `bench_cellsize.py`,
`bench_ceiling.py`, `research/`) produced the v2 studies in `docs/history/`, negative results
included. `research/polysemy/` has the model-level experiments behind orthogonal polysemy, the
attractor DAG and the estimator comparison (`docs/RESEARCH_LOG.md`).
