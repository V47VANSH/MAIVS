Environment: cpu AMD Ryzen 7 6800H with Radeon Graphics, threads 16, kernel 5.15.167.4-microsoft-standard-WSL2, python 3.12.3, numpy 2.5.4, faiss 1.15.1, faiss_compile OPTIMIZE DD AVX2 , scann 1.4.2, usearch 2.26.5, hnswlib 0.8.0, rustc rustc 1.92.0 (ded5c06cf 2025-12-08), maivs_commit d8147fa, os Windows 11 host, WSL2 Ubuntu 24.04

## HotpotQA 250k  (MiniLM d=384; 16 threads; WSL2)

| system | build s | B/vec | best R@10 IND / OOD | peak RSS MB |
|---|---|---|---|---|
| MAIVS quality | 6.2 | 1,759 | 0.9992 / 0.9983 | 1,268 |
| MAIVS memory | 6.3 | 607 | 0.9896 / 0.9891 | 1,046 |
| faiss HNSW32 | 70.4 | 1,808 | 0.9991 / 0.9999 | 1,718 |
| hnswlib HNSW32 | 68.9 | 1,812 | 0.9993 / 0.9996 | 872 |
| USearch HNSW32 | 90.4 | 2,626 | 0.9989 / 0.9994 | 883 |
| ScaNN | 8.7-11.4 | 1,745 | 0.9962 / 0.9955 | 1,850 |
| ScaNN + SOAR | 10.7-13.6 | 1,941 | 0.9998 / 0.9996 | 1,974 |
| faiss IVF-PQ4fs | 9.2-15.8 | 1,665 | 0.9972 / 0.9968 | 1,713 |
| faiss IVF-RaBitQfs (1-bit) | 8.4-15.6 | 1,620 | 0.9968 / 0.9957 | 1,895 |
| faiss IVF-RaBitQfs4 (4-bit) | 9.2-17.5 | 1,796 | 0.9972 / 0.9968 | 2,600 |

**IND: batch QPS (1,000 queries, 16 threads) at matched R@10**

| system | 0.95 | 0.98 | 0.99 | 0.995 |
|---|---|---|---|---|
| MAIVS quality | 41,603 | 27,094 | 19,332 | 11,686 |
| MAIVS quality, per-query (no batching) | 39,552 | 20,502 | 12,410 | 6,986 |
| MAIVS memory | 44,925 | 22,947 | - | - |
| MAIVS memory, per-query (no batching) | 38,667 | 14,531 | - | - |
| faiss HNSW32 | 11,279 | 6,131 | 4,375 | 2,955 |
| hnswlib HNSW32 | 11,274 | 6,547 | 4,560 | 3,148 |
| USearch HNSW32 | 8,005 | 4,284 | 2,957 | 2,082 |
| ScaNN | 40,892 | 22,380 | 14,717 | 11,169 |
| ScaNN + SOAR | 62,824 | 32,443 | 22,115 | 11,527 |
| faiss IVF-PQ4fs | 32,860 | 18,863 | 15,027 | 11,547 |
| faiss IVF-RaBitQfs (1-bit) | 15,979 | 4,876 | 1,828 | 1,761 |
| faiss IVF-RaBitQfs4 (4-bit) | 5,136 | 2,523 | 1,339 | 1,235 |
| *best baseline* | 62,824 (ScaNN + SOAR) | 32,443 (ScaNN + SOAR) | 22,115 (ScaNN + SOAR) | 11,547 (faiss IVF-PQ4fs) |
| *MAIVS quality / best, cell-major* | 0.66x | 0.84x | 0.87x | 1.01x |
| *MAIVS quality / best, per-query* | 0.63x | 0.63x | 0.56x | 0.61x |

**IND: single-thread p50 latency (us) at matched R@10, each system's latency-best setting**

| system | 0.95 | 0.98 | 0.99 | 0.995 |
|---|---|---|---|---|
| MAIVS quality | 152 | 289 | 477 | 939 |
| MAIVS memory | 164 | 466 | - | - |
| faiss HNSW32 | 291 | 447 | 700 | 1,063 |
| hnswlib HNSW32 | 384 | 611 | 980 | 1,217 |
| USearch HNSW32 | 535 | 1,070 | 1,672 | 2,335 |
| ScaNN | 194 | 427 | 665 | 818 |
| ScaNN + SOAR | 138 | 290 | 418 | 601 |
| faiss IVF-PQ4fs | 326 | 573 | 804 | 1,112 |
| faiss IVF-RaBitQfs (1-bit) | 416 | 963 | 1,896 | 2,213 |
| faiss IVF-RaBitQfs4 (4-bit) | 1,241 | 2,323 | 3,086 | 4,031 |

**OOD: batch QPS (1,000 queries, 16 threads) at matched R@10**

| system | 0.95 | 0.98 | 0.99 | 0.995 |
|---|---|---|---|---|
| MAIVS quality | 33,274 | 17,203 | 13,385 | 9,519 |
| MAIVS quality, per-query (no batching) | 30,220 | 15,184 | 8,760 | 6,013 |
| MAIVS memory | 33,341 | 18,122 | - | - |
| MAIVS memory, per-query (no batching) | 23,069 | 12,324 | - | - |
| faiss HNSW32 | 7,653 | 4,676 | 3,374 | 2,425 |
| hnswlib HNSW32 | 8,423 | 5,206 | 3,649 | 2,741 |
| USearch HNSW32 | 5,619 | 3,645 | 2,441 | 1,795 |
| ScaNN | 38,605 | 18,967 | 14,652 | 11,009 |
| ScaNN + SOAR | 44,701 | 23,250 | 20,357 | 11,992 |
| faiss IVF-PQ4fs | 33,529 | 19,963 | 15,122 | 11,653 |
| faiss IVF-RaBitQfs (1-bit) | 12,184 | 3,478 | 1,703 | 1,601 |
| faiss IVF-RaBitQfs4 (4-bit) | 4,602 | 2,261 | 1,310 | 1,111 |
| *best baseline* | 44,701 (ScaNN + SOAR) | 23,250 (ScaNN + SOAR) | 20,357 (ScaNN + SOAR) | 11,992 (ScaNN + SOAR) |
| *MAIVS quality / best, cell-major* | 0.74x | 0.74x | 0.66x | 0.79x |
| *MAIVS quality / best, per-query* | 0.68x | 0.65x | 0.43x | 0.50x |

**OOD: single-thread p50 latency (us) at matched R@10, each system's latency-best setting**

| system | 0.95 | 0.98 | 0.99 | 0.995 |
|---|---|---|---|---|
| MAIVS quality | 232 | 458 | 654 | 890 |
| MAIVS memory | 281 | 551 | - | - |
| faiss HNSW32 | 426 | 651 | 868 | 1,221 |
| hnswlib HNSW32 | 478 | 762 | 1,080 | 1,480 |
| USearch HNSW32 | 898 | 1,347 | 2,002 | 2,930 |
| ScaNN | 221 | 408 | 809 | 813 |
| ScaNN + SOAR | 200 | 368 | 426 | 723 |
| faiss IVF-PQ4fs | 353 | 606 | 920 | 1,110 |
| faiss IVF-RaBitQfs (1-bit) | 569 | 1,359 | 1,993 | 2,578 |
| faiss IVF-RaBitQfs4 (4-bit) | 1,470 | 2,519 | 3,545 | 4,961 |

## HotpotQA 500k  (MiniLM d=384; 16 threads; WSL2)

| system | build s | B/vec | best R@10 IND / OOD | peak RSS MB |
|---|---|---|---|---|
| MAIVS quality | 5.2 | 1,761 | 0.9993 / 0.9970 | 2,508 |
| MAIVS memory | 4.0 | 609 | 0.9897 / 0.9874 | 1,385 |
| faiss HNSW32 | 152.6 | 1,808 | 0.9988 / 0.9990 | 3,393 |
| hnswlib HNSW32 | 147.5 | 1,812 | 0.9988 / 0.9989 | 1,707 |
| USearch HNSW32 | 206.4 | 2,659 | 0.9983 / 0.9989 | 1,707 |
| ScaNN | 13.6-19.5 | 1,744 | 0.9882 / 0.9857 | 3,632 |
| ScaNN + SOAR | 19.9-27.1 | 1,940 | 0.9981 / 0.9985 | 3,887 |
| faiss IVF-PQ4fs | 28.8-60.7 | 1,665 | 0.9908 / 0.9903 | 3,289 |
| faiss IVF-RaBitQfs (1-bit) | 28.2-69.1 | 1,620 | 0.9904 / 0.9887 | 3,421 |
| faiss IVF-RaBitQfs4 (4-bit) | 33.2-72.8 | 1,796 | 0.9908 / 0.9903 | 4,326 |

**IND: batch QPS (1,000 queries, 16 threads) at matched R@10**

| system | 0.95 | 0.98 | 0.99 | 0.995 |
|---|---|---|---|---|
| MAIVS quality | 29,738 | 18,001 | 12,798 | 8,548 |
| MAIVS quality, per-query (no batching) | 16,814 | 8,936 | 6,513 | 3,961 |
| MAIVS memory | 29,711 | 15,462 | - | - |
| MAIVS memory, per-query (no batching) | 17,784 | 7,701 | - | - |
| faiss HNSW32 | 10,101 | 5,356 | 3,427 | 2,268 |
| hnswlib HNSW32 | 11,472 | 5,579 | 3,302 | 1,589 |
| USearch HNSW32 | 7,395 | 4,170 | 2,636 | 1,649 |
| ScaNN | 31,243 | 14,424 | - | - |
| ScaNN + SOAR | 31,008 | 19,213 | 12,349 | 9,459 |
| faiss IVF-PQ4fs | 22,712 | 12,876 | 10,041 | - |
| faiss IVF-RaBitQfs (1-bit) | 11,774 | 2,916 | 1,583 | - |
| faiss IVF-RaBitQfs4 (4-bit) | 4,268 | 1,914 | 1,276 | - |
| *best baseline* | 31,243 (ScaNN) | 19,213 (ScaNN + SOAR) | 12,349 (ScaNN + SOAR) | 9,459 (ScaNN + SOAR) |
| *MAIVS quality / best, cell-major* | 0.95x | 0.94x | 1.04x | 0.90x |
| *MAIVS quality / best, per-query* | 0.54x | 0.47x | 0.53x | 0.42x |

**IND: single-thread p50 latency (us) at matched R@10, each system's latency-best setting**

| system | 0.95 | 0.98 | 0.99 | 0.995 |
|---|---|---|---|---|
| MAIVS quality | 278 | 540 | 769 | 1,181 |
| MAIVS memory | 281 | 677 | - | - |
| faiss HNSW32 | 342 | 679 | 1,061 | 1,393 |
| hnswlib HNSW32 | 352 | 919 | 1,326 | 2,429 |
| USearch HNSW32 | 686 | 1,228 | 1,871 | 3,053 |
| ScaNN | 313 | 643 | - | - |
| ScaNN + SOAR | 240 | 482 | 758 | 1,096 |
| faiss IVF-PQ4fs | 544 | 850 | 1,320 | - |
| faiss IVF-RaBitQfs (1-bit) | 692 | 1,582 | 2,549 | - |
| faiss IVF-RaBitQfs4 (4-bit) | 1,639 | 3,047 | 4,528 | - |

**OOD: batch QPS (1,000 queries, 16 threads) at matched R@10**

| system | 0.95 | 0.98 | 0.99 | 0.995 |
|---|---|---|---|---|
| MAIVS quality | 25,777 | 15,091 | 10,206 | 7,199 |
| MAIVS quality, per-query (no batching) | 17,134 | 8,417 | 4,984 | 3,191 |
| MAIVS memory | 28,563 | 11,923 | - | - |
| MAIVS memory, per-query (no batching) | 19,048 | 5,450 | - | - |
| faiss HNSW32 | 6,697 | 3,783 | 2,602 | 1,667 |
| hnswlib HNSW32 | 6,152 | 3,943 | 2,797 | 1,981 |
| USearch HNSW32 | 4,770 | 2,725 | 1,869 | 1,258 |
| ScaNN | 20,522 | 10,312 | - | - |
| ScaNN + SOAR | 32,397 | 18,938 | 10,711 | 8,949 |
| faiss IVF-PQ4fs | 21,623 | 10,734 | 8,316 | - |
| faiss IVF-RaBitQfs (1-bit) | 5,427 | 1,695 | - | - |
| faiss IVF-RaBitQfs4 (4-bit) | 3,147 | 1,306 | 1,227 | - |
| *best baseline* | 32,397 (ScaNN + SOAR) | 18,938 (ScaNN + SOAR) | 10,711 (ScaNN + SOAR) | 8,949 (ScaNN + SOAR) |
| *MAIVS quality / best, cell-major* | 0.80x | 0.80x | 0.95x | 0.80x |
| *MAIVS quality / best, per-query* | 0.53x | 0.44x | 0.47x | 0.36x |

**OOD: single-thread p50 latency (us) at matched R@10, each system's latency-best setting**

| system | 0.95 | 0.98 | 0.99 | 0.995 |
|---|---|---|---|---|
| MAIVS quality | 310 | 632 | 927 | 1,575 |
| MAIVS memory | 307 | 947 | - | - |
| faiss HNSW32 | 437 | 747 | 1,083 | 1,762 |
| hnswlib HNSW32 | 656 | 1,050 | 1,489 | 2,050 |
| USearch HNSW32 | 1,131 | 1,966 | 2,945 | 4,235 |
| ScaNN | 427 | 750 | - | - |
| ScaNN + SOAR | 278 | 450 | 721 | 978 |
| faiss IVF-PQ4fs | 558 | 1,178 | 1,268 | - |
| faiss IVF-RaBitQfs (1-bit) | 1,089 | 2,291 | - | - |
| faiss IVF-RaBitQfs4 (4-bit) | 2,054 | 4,048 | 4,862 | - |

## Run-to-run drift (500k, same system re-run last in a fresh process)

| system | point | first run QPS | repeat QPS | change |
|---|---|---|---|---|
| MAIVS quality | IND 0.95 | 29,738 | 28,281 | -5% |
| MAIVS quality | IND 0.99 | 12,798 | 12,594 | -2% |
| MAIVS quality | OOD 0.95 | 25,777 | 25,030 | -3% |
| MAIVS quality | OOD 0.99 | 10,206 | 10,523 | +3% |
| faiss HNSW32 | IND 0.95 | 10,101 | 9,367 | -7% |
| faiss HNSW32 | IND 0.99 | 3,427 | 3,244 | -5% |
| faiss HNSW32 | OOD 0.95 | 6,697 | 6,184 | -8% |
| faiss HNSW32 | OOD 0.99 | 2,602 | 2,369 | -9% |
