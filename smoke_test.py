import sys
sys.path.append('src')
import numpy as np
from aigvs.pipeline import AIGVSPipeline

def test_rust_aigvs():
    print("Initializing Rust-Native Pipeline...")
    pipeline = AIGVSPipeline(
        embedding_dim=64,
        max_connections=16,
        ef_construction=100,
        polysemy_base=0.08,
        polysemy_decay=0.5,
        hard_cap=1000  # Small hard cap to force splits quickly
    )

    print("Generating random data...")
    # Create 5000 random vectors to trigger initial discovery
    n_docs = 5500
    docs = []
    for i in range(n_docs):
        # Create some clusters by adding a base offset to every 1000th chunk
        cluster_id = i // 1000
        vec = np.random.randn(64) + (cluster_id * 0.5)
        vec = vec / np.linalg.norm(vec)
        docs.append({
            "id": f"doc_{i}",
            "text": f"This is document {i} from cluster {cluster_id}",
            "embedding": vec,
            "metadata": {"cluster": cluster_id}
        })
    
    print("Ingesting data...")
    # This should trigger macro-attractor discovery and create several leaves
    pipeline.ingest(docs)
    
    print("Stats after ingestion:")
    print(pipeline.stats())

    print("\nTesting queries...")
    for cluster_id in range(5):
        q_vec = np.random.randn(64) + (cluster_id * 0.5)
        q_vec = q_vec / np.linalg.norm(q_vec)
        
        results = pipeline.query(q_vec, top_k=5)
        print(f"\nQuerying near cluster {cluster_id}:")
        for res in results:
            print(f"  - {res['id']}: {res['text']}")

    print("\nTesting save/load...")
    pipeline.save_index("./aigvs_data/smoke_test.bin")
    
    new_pipe = AIGVSPipeline(embedding_dim=64)
    new_pipe.load_index("./aigvs_data/smoke_test.bin")
    print("\nStats after load:")
    print(new_pipe.stats())
    
    print("\nSmoke test completed successfully!")

if __name__ == "__main__":
    test_rust_aigvs()
