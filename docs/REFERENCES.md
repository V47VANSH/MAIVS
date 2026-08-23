# References

What the engine builds on, and how each part relates to published work.

## Components

| part | prior work | relation |
|---|---|---|
| partitioned index with probe + rerank | IVF-ADC [1]; faiss [2, 3] | same family. Differences: fixed 128-vector cells instead of sqrt(N) lists, a centroid tree for routing, one flat arena |
| randomised Hadamard rotation + sign codes | fast JL transform [4]; sign random projections [5]; RaBitQ [6] | RaBitQ's rotation and 1-bit codes, with an exact rerank instead of RaBitQ's unbiased estimator |
| FastScan first stage | PQ FastScan [7, 8]; RaBitQ's batch estimator [6] | FastScan lookup tables on 1-bit codes read as 4-bit groups, as in [6] |
| orthogonal polysemy | SOAR [9]; spill trees [10] | the assignment loss is SOAR's: `norm(r')^2 + lam*(r'.r_hat)^2`. What I add: a direct measurement that the two homes' misses are independent ([`RESEARCH_LOG.md`](RESEARCH_LOG.md) section 4.2), and exactly-once scoring with contiguous replica runs in both per-query and cell-major execution |
| cell-major batch execution | batched inverted-list scanning in IVF systems [3] | same idea (read each list once for every query that probes it). What's specific here: per-query seeding and a shared atomic cut-off, which keep the results identical to the per-query path |
| integer bucket-queue top-k | standard radix / bucket selection | |
| attractor tree (where the project started) | Hopfield networks [11, 12] | the prototype routed through a recursive Hopfield tree; the engine kept the hierarchy and replaced the attractor dynamics with a flat SIMD scan per level |

## Baselines measured

* faiss `IndexHNSWFlat` (HNSW [13]), faiss IVF-RaBitQ and IVF-PQ FastScan [2], all faiss-cpu
  1.14.3.
* USearch 2.26.4 [14], a separate HNSW implementation.

Not measured yet, and needed before claiming anything about the state of the art: ScaNN [15],
DiskANN [16], and the ann-benchmarks suite [17].

## Data

* HotpotQA [18] via BEIR [19]. MS MARCO passage ranking [20].
* Embeddings: sentence-transformers/all-MiniLM-L6-v2 [21, 22].

## Bibliography

1. H. Jégou, M. Douze, C. Schmid. *Product Quantization for Nearest Neighbor
   Search.* IEEE TPAMI 33(1), 2011.
2. M. Douze, A. Guzhva, C. Deng, J. Johnson, G. Szilvasy, P.-E. Mazaré,
   M. Lomeli, L. Hosseini, H. Jégou. *The Faiss library.* arXiv:2401.08281,
   2024.
3. J. Johnson, M. Douze, H. Jégou. *Billion-scale similarity search with
   GPUs.* IEEE Transactions on Big Data, 2019.
4. N. Ailon, B. Chazelle. *The Fast Johnson-Lindenstrauss Transform and
   Approximate Nearest Neighbors.* SIAM J. Computing 39(1), 2009.
5. M. Charikar. *Similarity Estimation Techniques from Rounding Algorithms.*
   STOC 2002.
6. J. Gao, C. Long. *RaBitQ: Quantizing High-Dimensional Vectors with a
   Theoretical Error Bound for Approximate Nearest Neighbor Search.* SIGMOD
   2024.
7. F. André, A.-M. Kermarrec, N. Le Scouarnec. *Cache locality is not
   enough: High-Performance Nearest Neighbor Search with Product Quantization
   Fast Scan.* PVLDB 9(4), 2015.
8. F. André, A.-M. Kermarrec, N. Le Scouarnec. *Quicker ADC: Unlocking the
   Hidden Potential of Product Quantization with SIMD.* IEEE TPAMI 43(5),
   2021.
9. P. Sun, R. Guo, S. Kumar. *SOAR: Improved Indexing for Approximate
   Nearest Neighbor Search.* NeurIPS 2023.
10. T. Liu, A. W. Moore, A. Gray, K. Yang. *An Investigation of Practical
    Approximate Nearest Neighbor Algorithms.* NIPS 2004.
11. J. J. Hopfield. *Neural networks and physical systems with emergent
    collective computational abilities.* PNAS 79(8), 1982.
12. H. Ramsauer et al. *Hopfield Networks is All You Need.* ICLR 2021.
13. Y. A. Malkov, D. A. Yashunin. *Efficient and Robust Approximate Nearest
    Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE
    TPAMI 42(4), 2020.
14. Unum. *USearch.* https://github.com/unum-cloud/usearch
15. R. Guo, P. Sun, E. Lindgren, Q. Geng, D. Simcha, F. Chern, S. Kumar.
    *Accelerating Large-Scale Inference with Anisotropic Vector
    Quantization.* ICML 2020.
16. S. J. Subramanya, Devvrit, R. Kadekodi, R. Krishnaswamy,
    H. V. Simhadri. *DiskANN: Fast Accurate Billion-point Nearest Neighbor
    Search on a Single Node.* NeurIPS 2019.
17. M. Aumüller, E. Bernhardsson, A. Faithfull. *ANN-Benchmarks: A
    Benchmarking Tool for Approximate Nearest Neighbor Algorithms.*
    Information Systems 87, 2020.
18. Z. Yang, P. Qi, S. Zhang, Y. Bengio, W. W. Cohen, R. Salakhutdinov,
    C. D. Manning. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop
    Question Answering.* EMNLP 2018.
19. N. Thakur, N. Reimers, A. Rücklé, A. Srivastava, I. Gurevych. *BEIR: A
    Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval
    Models.* NeurIPS Datasets and Benchmarks 2021.
20. P. Bajaj et al. *MS MARCO: A Human Generated MAchine Reading
    COmprehension Dataset.* arXiv:1611.09268, 2016.
21. N. Reimers, I. Gurevych. *Sentence-BERT: Sentence Embeddings using
    Siamese BERT-Networks.* EMNLP 2019.
22. W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, M. Zhou. *MiniLM: Deep
    Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained
    Transformers.* NeurIPS 2020.
