"""pipeline.py: thin Python wrapper around the Rust AIGVS engine.

The Rust engine (aigvs_rust.AIGVSEngine) now owns the whole tree: macro-attractor discovery,
Hopfield splits, depth-dependent polysemy, parallel NSW construction (rayon) and binary persistence
(bincode).

Python:
  1. takes user data (text, embeddings, metadata)
  2. stores text/metadata in the storage backend
  3. passes raw vectors to Rust to build the tree
  4. at query time passes the vector to Rust, gets chunk ids back and fetches the text
"""

import os
import numpy as np
from typing import Dict, List, Optional
from aigvs.core.models import TextChunk
from aigvs.storage import LocalJSONStorage
import aigvs_rust


class AIGVSPipeline:
    """Runs the AIGVS hierarchy.

    The Rust engine handles tree topology (trunks, leaves, routing attractors), the HNSW graph in
    each leaf, splitting leaves that get too big or too spread out, and parallel search across
    leaves. Python handles text/metadata storage, embedding normalisation and the user API.
    """

    def __init__(
        self,
        embedding_dim: int,
        storage_path: str = "./aigvs_data",
        max_connections: int = 12,
        ef_construction: int = 80,
        polysemy_base: float = 0.02,
        polysemy_decay: float = 0.5,
        hard_cap: int = 150000,
    ):
        self.embedding_dim = embedding_dim
        self.storage_path = storage_path
        self.storage = LocalJSONStorage(storage_path)

        self.engine = aigvs_rust.AIGVSEngine(
            embedding_dim=embedding_dim,
            max_connections=max_connections,
            ef_construction=ef_construction,
            polysemy_base=polysemy_base,
            polysemy_decay=polysemy_decay,
            hard_cap=hard_cap,
        )


    # --- internal helpers ---

    def ingest(self, data: List[dict]) -> None:
        """Bulk ingest a batch of documents.

        Each dict needs "id" (str), "text" (str), "embedding" (1D float32 ndarray) and optionally
        "metadata" (dict). Rust builds or extends the tree; Python stores the text.
        """
        if not data:
            return

        ids = []
        embeddings = []

        for entry in data:
            chunk = TextChunk(
                chunk_id=entry["id"],
                text=entry.get("text", ""),
                embedding=np.array(entry["embedding"], dtype=np.float32),
                metadata=entry.get("metadata", {}),
            )
            # Store text in storage backend
            self.storage.write_chunk(chunk)

            ids.append(entry["id"])
            embeddings.append(chunk.embedding)

        # hand the vectors to Rust; the Python copies can be freed after this
        vectors = np.array(embeddings, dtype=np.float32)
        self.engine.ingest_batch(ids, vectors)

    def ingest_hierarchical_json(self, json_data: List[dict]) -> None:
        """Legacy alias for ingest(). Maintained for backward compatibility."""
        self.ingest(json_data)

    def insert_chunk(self, chunk: TextChunk) -> None:
        """Insert a single chunk (real-time / online mode)."""
        self.storage.write_chunk(chunk)
        self.engine.insert_single(chunk.chunk_id, chunk.embedding)

    # --- internal helpers ---

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        max_beam: int = 10,
        target_confidence: float = 0.70,
        beta: float = 50.0,
        ef_search: Optional[int] = None,
    ) -> List[dict]:
        """
        Search the AIGVS tree.

        Returns a list of dicts with "id", "text", "metadata" for the top-k results.
        """
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm
        query_embedding = query_embedding.astype(np.float32)

        # Tune ef_search for ultra-low latency while maintaining recall
        ef = ef_search if ef_search is not None else max(top_k * 8, 80)

        matched_ids: List[str] = self.engine.unified_search(
            query_embedding,
            top_k=top_k,
            max_beam=max_beam,
            target_confidence=target_confidence,
            beta=beta,
            ef_search=ef,
        )

        if not matched_ids:
            return []

        # Fallback: fetch from storage
        results = []
        for cid in matched_ids:
            chunk = self.storage.read_chunk(cid)
            if chunk is not None:
                results.append({
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                })
        return results

    # --- internal helpers ---

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk by ID (tombstone-based)."""
        self.storage.delete_chunk(chunk_id)
        return self.engine.delete(chunk_id)

    # --- internal helpers ---

    def save_index(self, path: Optional[str] = None) -> None:
        """Save the Rust engine state to disk (binary)."""
        if path is None:
            os.makedirs(self.storage_path, exist_ok=True)
            path = os.path.join(self.storage_path, "aigvs_engine.bin")
        self.engine.save(path)

    def load_index(self, path: Optional[str] = None) -> bool:
        """Load the Rust engine state from disk."""
        if path is None:
            path = os.path.join(self.storage_path, "aigvs_engine.bin")
        if not os.path.exists(path):
            print(f"[AIGVS] No index found at {path}. Starting fresh.")
            return False
        self.engine = aigvs_rust.AIGVSEngine.load(path)
        return True

    # --- internal helpers ---

    def stats(self) -> dict:
        """Return tree statistics from the Rust engine."""
        return self.engine.stats()