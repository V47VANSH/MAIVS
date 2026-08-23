# History

Studies from the v2 engine and the June prototype before it. They're kept as written, negative
results included. Several of their conclusions changed later; the current state is in
[`../RESULTS.md`](../RESULTS.md) and [`../RESEARCH_LOG.md`](../RESEARCH_LOG.md).

| document | what it is | later changed by |
|---|---|---|
| [`PROTOTYPE_NOTES.md`](PROTOTYPE_NOTES.md) | what the recursive Hopfield-tree prototype taught me (HNSW leaves, 100k vectors), and the original O(log N) goal | |
| [`PLAN_2026-07.md`](PLAN_2026-07.md) | the plan after the July literature review, with the first real ANN measurements | most of the plan was overtaken by the v2 and v3 results |
| [`notes/`](notes/) | lab notes from 26-27 July, when v2 was built | |
| [`ARCHITECTURE_v2.md`](ARCHITECTURE_v2.md) | v2 codebase guide: every constant and where its value came from | [`../ARCHITECTURE.md`](../ARCHITECTURE.md) |
| [`RESULTS_v2.md`](RESULTS_v2.md) | v2 at 5.2M: recall ceiling, ten failed ideas, problems found in the harness | the v3 results; "no question embeddings exist" was wrong |
| [`SCALING_STUDY.md`](SCALING_STUDY.md) | HotpotQA 10k -> 5.23M: what scales and what doesn't, v2 against faiss-HNSW | v3 scaling, `RESULTS.md` section 3 |
| [`SUMMARISER_STUDY.md`](SUMMARISER_STUDY.md) | can a richer per-cell summary close the routing gap? (1.25x at most); SQ8 centroid re-order | |
| [`SCALABILITY_PLAN.md`](SCALABILITY_PLAN.md) | sample ladders, cell graphs, kNN partitions, HNSW over cells: tried and refuted | "spill is a wash" only holds for next-nearest spill; orthogonal homes behave differently (`RESEARCH_LOG.md` sections 3-4) |

The June prototype's code is in the git history, until it was removed.
