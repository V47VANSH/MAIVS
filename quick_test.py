import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import uuid
import psutil
import numpy as np
from src.aigvs.pipeline import AIGVSPipeline
from src.aigvs.storage import InMemoryStorage
from src.aigvs.core.models import TextChunk

N = 100000
EMBEDDING_DIM = 384
print(f"Generating {N} random vectors...")
vectors = np.random.randn(N, EMBEDDING_DIM).astype(np.float32)

print("Vectors generated. Initializing AIGVS...", flush=True)

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024

pipeline = AIGVSPipeline(embedding_dim=EMBEDDING_DIM, ef_construction=32, max_connections=32)
pipeline.storage = InMemoryStorage()

chunks_json = [{"id": f"doc_{i}", "text": "text", "embedding": vectors[i].tolist()} for i in range(N)]

t0 = time.time()
print("Calling ingest...", flush=True)
pipeline.ingest_hierarchical_json(chunks_json)
t1 = time.time()

mem_after = process.memory_info().rss / 1024 / 1024
print(f"\nBuild time: {t1 - t0:.2f} seconds", flush=True)
print(f"Memory overhead: {mem_after - mem_before:.2f} MB", flush=True)
