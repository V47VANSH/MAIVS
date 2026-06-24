import faulthandler
faulthandler.enable()
import os
# Windows workaround: let PyTorch and faiss share the OpenMP runtime instead of aborting
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import subprocess

# force UTF-8 output on Windows
if sys.platform == "win32" and os.environ.get("PYTHONUTF8") != "1":
    os.environ["PYTHONUTF8"] = "1"
    sys.exit(subprocess.call([sys.executable, "-X", "utf8"] + sys.argv))

import time
import json
import numpy as np
import faiss
import chromadb
import psutil
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Link your local AIGVS package
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from aigvs.pipeline import AIGVSPipeline
from aigvs.pipeline_c import AIGVSPipelineC

# --- hyperparameters ---
SAMPLE_SIZE = 100000  # Number of paragraphs to index (start small)
TEST_QUERIES = 200   # Number of questions to evaluate
EMBEDDING_DIM = 384  # MiniLM dimension size
TOP_K = 10           # For Recall@10

def load_msmarco_data():
    import ir_datasets
    print("[1/6] Loading MSMARCO...")
    dataset = ir_datasets.load("msmarco-passage/train")

    queries = []
    golden_doc_ids = set()

    print("Loading queries...")
    qrels = dataset.qrels_dict()
    for q in dataset.queries_iter():
        if q.query_id in qrels:
            rel_docs = qrels[q.query_id]
            queries.append({
                "question": q.text,
                "golden_titles": [str(did) for did in rel_docs.keys()]
            })
            golden_doc_ids.update([str(did) for did in rel_docs.keys()])
            if len(queries) >= TEST_QUERIES:
                break

    print(f"Loaded {len(queries)} queries with {len(golden_doc_ids)} golden documents.")

    filler_docs_needed = SAMPLE_SIZE - len(golden_doc_ids)
    if filler_docs_needed < 0:
        filler_docs_needed = 0

    found_golden = 0
    golden_docs_data = []
    filler_docs_data = []

    print(f"Scanning corpus to guarantee inclusion of golden targets (This takes ~30 seconds)...")
    for doc in dataset.docs_iter():
        did = str(doc.doc_id)
        if did in golden_doc_ids:
            golden_docs_data.append((did, doc.text))
            found_golden += 1
        elif len(filler_docs_data) < filler_docs_needed:
            filler_docs_data.append((did, doc.text))
            
        if len(filler_docs_data) >= filler_docs_needed and found_golden == len(golden_doc_ids):
            break

    docs = golden_docs_data + filler_docs_data
    print(f"Loaded {len(docs)} documents")

    return docs, queries

def calculate_mrr(results, golden_targets):
    """Calculates the Reciprocal Rank for a single query."""
    for rank, res_id in enumerate(results, 1):
        if res_id in golden_targets:
            return 1.0 / rank
    return 0.0

def main():
    print("=== THE ULTRA-GRANULAR MSMARCO BENCHMARK ===\n")
    
    # 1. Load Data & Model
    docs, queries = load_msmarco_data()
    print(f"\n[3/6] Loading local embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"Preparing {len(docs)} documents...")
    doc_ids = [str(d[0]) for d in docs]
    doc_texts = [d[1] for d in docs]
    
    # CACHED EMBEDDING LOADING
    cache_file = f"msmarco_embeddings_cache_v2_{SAMPLE_SIZE}.npy"
    if os.path.exists(cache_file):
        print("-> Found cached embeddings! Loading instantly...")
        doc_embeddings = np.load(cache_file)
    else:
        print(f"Embedding {len(docs)} documents (This will take a minute on CPU)...")
        doc_embeddings = model.encode(doc_texts, show_progress_bar=True)
        np.save(cache_file, doc_embeddings) 
    
    # Data container structured for deep paper analytics
    paper_metrics = {
        "experiment_parameters": {
            "sample_size": SAMPLE_SIZE,
            "total_queries_tested": TEST_QUERIES,
            "top_k_retrieved": TOP_K,
            "embedding_model": "all-MiniLM-L6-v2"
        },
        "global_performance": {
            "faiss": {"build_time_sec": 0, "total_hits": 0, "avg_latency_ms": 0, "p99_latency_ms": 0, "mean_mrr": 0, "memory_mb": 0},
            "chroma": {"build_time_sec": 0, "total_hits": 0, "avg_latency_ms": 0, "p99_latency_ms": 0, "mean_mrr": 0, "memory_mb": 0},
            "aigvs": {"build_time_sec": 0, "total_hits": 0, "avg_latency_ms": 0, "p99_latency_ms": 0, "mean_mrr": 0, "memory_mb": 0},
            "aigvs_c": {"build_time_sec": 0, "total_hits": 0, "avg_latency_ms": 0, "p99_latency_ms": 0, "mean_mrr": 0, "memory_mb": 0}
        },
        "query_logs": [] 
    }
    
    print("\n[4/6] STARTING THE WRITE RACE...")
    
    # --- faiss (HNSW) ---
    print("BREADCRUMB 1: Setting up FAISS vectors...", flush=True)
    faiss_vectors = np.ascontiguousarray(doc_embeddings.copy(), dtype=np.float32)
    faiss.normalize_L2(faiss_vectors)
    
    print("BREADCRUMB 2: Initializing FAISS Index (HNSW)...", flush=True)
    faiss.omp_set_num_threads(1)
    faiss_index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32)
    faiss_index.hnsw.efConstruction = 200
    faiss_index.hnsw.efSearch = 128
    
    print("BREADCRUMB 3: Adding vectors to FAISS...", flush=True)
    start_time = time.time()
    faiss_index.add(faiss_vectors)
    paper_metrics["global_performance"]["faiss"]["build_time_sec"] = time.time() - start_time
    faiss_id_map = {i: doc_ids[i] for i in range(len(docs))}
    print("BREADCRUMB 4: FAISS survived!", flush=True)
    
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    paper_metrics["global_performance"]["faiss"]["memory_mb"] = rss_mb
    print(f"Memory (FAISS): {rss_mb:.1f} MB")

    # --- chroma ---
    print("\nBREADCRUMB 5: Initializing ChromaDB...", flush=True)
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection(name="benchmark_msmarco", metadata={"hnsw:space": "cosine"})
    
    print("BREADCRUMB 6: Adding vectors to ChromaDB...", flush=True)
    start_time = time.time()
    
    batch_size = 5000
    for i in range(0, len(doc_ids), batch_size):
        chroma_collection.add(
            ids=doc_ids[i:i+batch_size], 
            embeddings=doc_embeddings[i:i+batch_size].tolist(), 
            documents=doc_texts[i:i+batch_size]
        )
        
    paper_metrics["global_performance"]["chroma"]["build_time_sec"] = time.time() - start_time
    print("BREADCRUMB 7: ChromaDB survived!", flush=True)
    
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    paper_metrics["global_performance"]["chroma"]["memory_mb"] = rss_mb
    print(f"Memory (Chroma): {rss_mb:.1f} MB")

    # --- AIGVS ---
    print("\nBREADCRUMB 8: Initializing AIGVS...", flush=True)
    aigvs_payload = [
        {"id": doc_ids[i], "text": doc_texts[i], "embedding": faiss_vectors[i].copy()} 
        for i in range(len(docs))
    ]
    
    aigvs_pipeline = AIGVSPipeline(embedding_dim=EMBEDDING_DIM, storage_path="./benchmark_data_msmarco")
    from aigvs.storage import InMemoryStorage
    aigvs_pipeline.storage = InMemoryStorage()
    aigvs_pipeline.root.max_leaf_capacity = 250
    
    start_time = time.time()
    aigvs_pipeline.ingest_hierarchical_json(aigvs_payload)
    paper_metrics["global_performance"]["aigvs"]["build_time_sec"] = time.time() - start_time
    
    print("BREADCRUMB 10: AIGVS survived!", flush=True)
    
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    paper_metrics["global_performance"]["aigvs"]["memory_mb"] = rss_mb
    print(f"Memory (AIGVS): {rss_mb:.1f} MB")
    
    # --- AIGVS (C port) ---
    print("\nBREADCRUMB 11: Initializing AIGVS (Pure C)...", flush=True)
    aigvs_c_pipeline = AIGVSPipelineC(embedding_dim=EMBEDDING_DIM)
    start_time = time.time()
    aigvs_c_pipeline.ingest_hierarchical_json(aigvs_payload)
    paper_metrics["global_performance"]["aigvs_c"]["build_time_sec"] = time.time() - start_time
    print("BREADCRUMB 12: AIGVS (Pure C) survived!", flush=True)

    rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    paper_metrics["global_performance"]["aigvs_c"]["memory_mb"] = rss_mb
    print(f"Memory (AIGVS C): {rss_mb:.1f} MB")

    # ==========================================
    # TEST B: THE READ RACE (ACCURACY & LATENCY)
    # ==========================================
    print(f"\n[5/6] STARTING THE READ RACE (Running {TEST_QUERIES} Queries)...")
    
    f_lats, c_lats, a_lats, ac_lats = [], [], [], []
    f_mrrs, c_mrrs, a_mrrs, ac_mrrs = [], [], [], []
    
    for idx, q in enumerate(tqdm(queries, desc="Evaluating DB Vectors", unit="query")):
        q_text = q["question"]
        golden_targets = q["golden_titles"]
        
        q_vec = model.encode([q_text])[0]
        q_vec_norm = q_vec / np.linalg.norm(q_vec) 
        faiss_q = q_vec_norm.reshape(1, -1).astype(np.float32)
        
        query_trace = {
            "query_index": idx,
            "question": q_text,
            "golden_parameters": golden_targets,
            "faiss_metrics": {},
            "chroma_metrics": {},
            "aigvs_metrics": {},
            "aigvs_c_metrics": {}
        }
        
        # --- faiss run ---
        t0 = time.perf_counter()
        _, indices = faiss_index.search(faiss_q, k=TOP_K)
        faiss_time = (time.perf_counter() - t0) * 1000 
        faiss_results = [faiss_id_map[i] for i in indices[0] if i != -1]
        faiss_mrr = calculate_mrr(faiss_results, golden_targets)
        faiss_hit = 1 if any(target in faiss_results for target in golden_targets) else 0
        
        f_lats.append(faiss_time)
        f_mrrs.append(faiss_mrr)
        paper_metrics["global_performance"]["faiss"]["total_hits"] += faiss_hit
        
        query_trace["faiss_metrics"] = {
            "latency_ms": faiss_time,
            "mrr": faiss_mrr,
            "hit_success": bool(faiss_hit),
            "returned_ids": faiss_results
        }

        # --- chroma run ---
        t0 = time.perf_counter()
        chroma_res = chroma_collection.query(query_embeddings=[q_vec.tolist()], n_results=TOP_K)
        chroma_time = (time.perf_counter() - t0) * 1000
        chroma_results = chroma_res["ids"][0]
        chroma_mrr = calculate_mrr(chroma_results, golden_targets)
        chroma_hit = 1 if any(target in chroma_results for target in golden_targets) else 0
        
        c_lats.append(chroma_time)
        c_mrrs.append(chroma_mrr)
        paper_metrics["global_performance"]["chroma"]["total_hits"] += chroma_hit
        
        query_trace["chroma_metrics"] = {
            "latency_ms": chroma_time,
            "mrr": chroma_mrr,
            "hit_success": bool(chroma_hit),
            "returned_ids": chroma_results
        }

        # --- faiss run ---
        t0 = time.perf_counter()
        aigvs_results_raw = aigvs_pipeline.query(q_vec_norm, top_k=TOP_K)
        aigvs_time = (time.perf_counter() - t0) * 1000
        aigvs_results = [item["id"] for item in aigvs_results_raw]
        aigvs_mrr = calculate_mrr(aigvs_results, golden_targets)
        aigvs_hit = 1 if any(target in aigvs_results for target in golden_targets) else 0
        
        a_lats.append(aigvs_time)
        a_mrrs.append(aigvs_mrr)
        paper_metrics["global_performance"]["aigvs"]["total_hits"] += aigvs_hit
        
        query_trace["aigvs_metrics"] = {
            "latency_ms": aigvs_time,
            "mrr": aigvs_mrr,
            "hit_success": bool(aigvs_hit),
            "returned_ids": aigvs_results
        }
        
        # --- AIGVS (C port) run ---
        t0 = time.perf_counter()
        aigvs_c_results_raw = aigvs_c_pipeline.query(q_vec_norm, top_k=TOP_K)
        aigvs_c_time = (time.perf_counter() - t0) * 1000
        aigvs_c_results = [item["id"] for item in aigvs_c_results_raw]
        aigvs_c_mrr = calculate_mrr(aigvs_c_results, golden_targets)
        aigvs_c_hit = 1 if any(target in aigvs_c_results for target in golden_targets) else 0
        
        ac_lats.append(aigvs_c_time)
        ac_mrrs.append(aigvs_c_mrr)
        paper_metrics["global_performance"]["aigvs_c"]["total_hits"] += aigvs_c_hit
        
        query_trace["aigvs_c_metrics"] = {
            "latency_ms": aigvs_c_time,
            "mrr": aigvs_c_mrr,
            "hit_success": bool(aigvs_c_hit),
            "returned_ids": aigvs_c_results
        }
        
        paper_metrics["query_logs"].append(query_trace)

    for key, lats, mrrs in [("faiss", f_lats, f_mrrs), ("chroma", c_lats, c_mrrs), ("aigvs", a_lats, a_mrrs), ("aigvs_c", ac_lats, ac_mrrs)]:
        paper_metrics["global_performance"][key]["avg_latency_ms"] = float(np.mean(lats))
        paper_metrics["global_performance"][key]["p99_latency_ms"] = float(np.percentile(lats, 99))
        paper_metrics["global_performance"][key]["mean_mrr"] = float(np.mean(mrrs))

    print("\n[6/6] Freezing full execution trace to benchmark_results_msmarco.json...")
    with open("benchmark_results_msmarco.json", "w") as f:
        json.dump(paper_metrics, f, indent=4)
    print("Done! Every question, retrieved document array, and millisecond metric is safely stored.")

    print("\n==============================================")
    print("             FINAL BENCHMARK SCORECARD")
    print("==============================================")
    for sys_name in ["faiss", "chroma", "aigvs", "aigvs_c"]:
        gp = paper_metrics["global_performance"][sys_name]
        recall = (gp["total_hits"] / TEST_QUERIES) * 100
        print(f"--- {sys_name.upper()} ---")
        print(f"Memory:      {gp['memory_mb']:.1f} MB")
        print(f"Build Time:  {gp['build_time_sec']:.2f} sec")
        print(f"Latency:     {gp['avg_latency_ms']:.2f} ms (Avg) | {gp['p99_latency_ms']:.2f} ms (P99)")
        print(f"Recall@{TOP_K}:    {recall:.1f}%")
        print(f"Mean RR:     {gp['mean_mrr']:.3f}\n")
    print("==============================================")
    
if __name__ == "__main__":
    main()
