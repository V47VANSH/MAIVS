## 1. Same memory budget (~600 B/vec), HotpotQA 500k

| system | B/vec | build s | best R@10 IND / OOD | QPS IND 0.95 | QPS IND 0.98 | QPS IND 0.99 | QPS OOD 0.95 | QPS OOD 0.98 | QPS OOD 0.99 | p50 us IND 0.95 / 0.98 |
|---|---|---|---|---|---|---|---|---|---|---|
| MAIVS memory (SQ8 rerank) | 609 | 4.0 | 0.9897 / 0.9874 | 29,711 | 15,462 | - | 28,563 | 11,923 | - | 281 / 677 |
| ScaNN + SOAR, INT8 reorder | 788 | 20.3-28.3 | 0.9890 / 0.9879 | 38,705 | 14,860 | - | 27,556 | 12,624 | - | 231 / 604 |
| ScaNN, INT8 reorder | 592 | 15.2-21.6 | 0.9797 / 0.9759 | 24,889 | - | - | 17,628 | - | - | 390 / - |
| faiss HNSW32, SQ8 | 656 | 92.6 | 0.9914 / 0.9901 | 19,950 | 7,986 | 2,788 | 9,947 | 5,969 | 1,132 | 290 / 674 |
| faiss IVF-PQ4fs + SQ8 refine | 513 | 46.8-85.6 | 0.9835 / 0.9822 | 21,670 | 9,283 | - | 18,355 | 8,969 | - | 461 / 1,428 |
| (ref) MAIVS quality, f32 | 1,761 | 5.2 | 0.9993 / 0.9970 | 29,738 | 18,001 | 12,798 | 25,777 | 15,091 | 10,206 | 278 / 540 |
| (ref) ScaNN + SOAR, f32 | 1,940 | 19.9-27.1 | 0.9981 / 0.9985 | 31,008 | 19,213 | 12,349 | 32,397 | 18,938 | 10,711 | 240 / 482 |

## 2. Churn: 25% turnover (100k deletes + 100k inserts, 10 rounds) of a 400k index

| system | build S0 s | deletes/s | inserts/s | maintenance | churned QPS@IND 0.95 / 0.98 | churned QPS@OOD 0.95 / 0.98 | after maint. QPS@IND 0.95 / 0.98 | after maint. QPS@OOD 0.95 / 0.98 | fresh rebuild QPS@IND 0.95 / 0.98 | fresh rebuild QPS@OOD 0.95 / 0.98 |
|---|---|---|---|---|---|---|---|---|---|---|
| MAIVS quality | 4.8 | 418,078 | 71,084 | compact(), 0.4 s | 8,866 / 2,441 | 5,924 / - | - / 15,547 | 21,595 / 15,050 | - / 19,161 | 22,235 / 14,816 |
| MAIVS memory | 4.3 | 318,973 | 97,314 | compact(), 0.1 s | 7,992 / - | 4,077 / - | 32,442 / 18,225 | 26,296 / 14,633 | 32,066 / 18,510 | 30,355 / 11,955 |
| ScaNN (no SOAR) | 17.6 | 206,926 | 62,649 | rebalance() (full retrain), 19.1 s | 24,041 / - | 24,043 / - | 27,135 / - | 19,175 / - | 26,515 / 15,860 | 24,557 / - |
| hnswlib HNSW32 | 125.9 | 1,023,613 | 234 | none available | 9,852 / 5,626 | 6,535 / 3,851 | - | - | 10,752 / 6,048 | 6,765 / 3,923 |
| faiss HNSW32 | - | not supported | not supported | no delete support: update = full rebuild | - | - | - | - | 10,226 / 5,482 | 6,052 / 3,349 |
| ScaNN + SOAR | - | crashes | crashes | - | ScaNN 1.4.2: segfault in delete() (f32 reorder); heap corruption in upsert() at 400k (INT8 reorder) | | | | | |

Fresh-rebuild build times (400k): MAIVS quality 3.8 s, MAIVS memory 3.3 s, ScaNN (no SOAR) 16.5 s, hnswlib HNSW32 118.0 s, faiss HNSW32 121.8 s

Best R@10 reached (IND/OOD), churned / after maintenance / fresh: MAIVS quality 0.9856/0.9664, 0.9993/0.9983, 0.9993/0.9983; MAIVS memory 0.9782/0.9594, 0.9905/0.9890, 0.9904/0.9890; ScaNN (no SOAR) 0.9785/0.9754, 0.9792/0.9752, 0.9806/0.9760; hnswlib HNSW32 0.9989/0.9993, 0.9994/0.9994; faiss HNSW32 0.9991/0.9993

## 3. Filtered search, HotpotQA 500k, random attribute, batch QPS at matched R@10


**Filter keeps 50% (250,000 vectors)**

| system | best R@10 IND / OOD | IND 0.9 | IND 0.95 | IND 0.99 | OOD 0.9 | OOD 0.95 | OOD 0.99 |
|---|---|---|---|---|---|---|---|
| MAIVS (filter in scan) | 0.9997 / 0.9987 | - | 14,154 | 5,601 | - | 12,963 | 3,119 |
| faiss HNSW32 (IDSelector) | 0.9995 / 0.9997 | 13,254 | 8,107 | 2,933 | 7,672 | 4,755 | 1,894 |
| ScaNN + SOAR (over-fetch + post-filter) | 0.9994 / 0.9983 | 40,141 | 30,075 | 10,994 | - | 23,901 | 8,158 |
| brute force over matching subset | 1.0000 / 1.0000 | 453 | 453 | 453 | 456 | 456 | 456 |

**Filter keeps 10% (50,000 vectors)**

| system | best R@10 IND / OOD | IND 0.9 | IND 0.95 | IND 0.99 | OOD 0.9 | OOD 0.95 | OOD 0.99 |
|---|---|---|---|---|---|---|---|
| MAIVS (filter in scan) | 0.9998 / 0.9996 | - | 12,659 | 4,444 | 18,351 | 11,349 | 3,696 |
| faiss HNSW32 (IDSelector) | 0.9994 / 0.9988 | 6,280 | 4,388 | 1,710 | 4,212 | 2,632 | 1,146 |
| ScaNN + SOAR (over-fetch + post-filter) | 0.9986 / 0.9961 | 26,289 | 14,970 | 6,817 | 16,734 | 12,086 | 5,326 |
| brute force over matching subset | 1.0000 / 1.0000 | 2,327 | 2,327 | 2,327 | 2,399 | 2,399 | 2,399 |

**Filter keeps 1% (5,000 vectors)**

| system | best R@10 IND / OOD | IND 0.9 | IND 0.95 | IND 0.99 | OOD 0.9 | OOD 0.95 | OOD 0.99 |
|---|---|---|---|---|---|---|---|
| MAIVS (filter in scan) | 1.0000 / 1.0000 | 14,050 | 8,334 | 2,691 | 14,261 | 9,484 | 3,224 |
| faiss HNSW32 (IDSelector) | 0.9935 / 0.9901 | 1,597 | 968 | 410 | 1,269 | 769 | 330 |
| ScaNN + SOAR (over-fetch + post-filter) | 0.9997 / 0.9983 | 4,611 | 3,771 | 1,858 | 4,513 | 2,996 | 1,908 |
| brute force over matching subset | 1.0000 / 1.0000 | 22,759 | 22,759 | 22,759 | 23,146 | 23,146 | 23,146 |

**Filter keeps 0.1% (500 vectors)**

| system | best R@10 IND / OOD | IND 0.9 | IND 0.95 | IND 0.99 | OOD 0.9 | OOD 0.95 | OOD 0.99 |
|---|---|---|---|---|---|---|---|
| MAIVS (filter in scan) | 1.0000 / 1.0000 | 4,744 | 2,943 | 1,558 | 5,208 | 3,155 | 1,702 |
| faiss HNSW32 (IDSelector) | 0.8305 / 0.8558 | - | - | - | - | - | - |
| ScaNN + SOAR (over-fetch + post-filter) | 0.9886 / 0.9787 | 484 | 448 | - | 469 | 460 | - |
| brute force over matching subset | 1.0000 / 1.0000 | 172,535 | 172,535 | 172,535 | 166,964 | 166,964 | 166,964 |
