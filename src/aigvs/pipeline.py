import os
import pickle
import time
import uuid
import numpy as np
from collections import deque
from typing import Dict, List, Optional
from aigvs.core.models import TextChunk
from aigvs.storage import LocalJSONStorage
from aigvs.engine.nodes import DisjointFamily, BaseNode, JointFamily
import aigvs_rust


class AIGVSPipeline:
    """Runs the AIGVS hierarchy.

    Query path, three steps:
      1. normalise with numpy (~5us)
      2. engine.unified_search(): one PyO3 call, GIL released, parallel leaf search in Rust
         (~1-3 ms depending on corpus size)
      3. payload_cache lookup for each of the top_k (~5us total)

    storage.read_chunk() is never called during a query. Text and metadata live in
    self.payload_cache, a dict keyed by chunk_id. Reading JSON or deserialising on every result was
    the main reason the old version was slow.

    NSW parameters:
        max_connections (M)  max edges per node; ~16 for 768-dim
        ef_construction      candidate pool during insert; ~4-8x M
        ef_search            candidate pool during query; raise for recall at the cost of
                             latency. Default max(top_k * 4, 50).
    """

    def __init__(
        self,
        embedding_dim: int,
        storage_path: str = "./aigvs_data",
        max_connections: int = 32,
        ef_construction: int = 200,
    ):
        self.embedding_dim = embedding_dim
        self.storage_path = storage_path
        self.storage = LocalJSONStorage(storage_path)
        self.index_file = os.path.join(storage_path, "aigvs_memory.pkl")

        self.max_connections = max_connections
        self.ef_construction = ef_construction

        self.engine = aigvs_rust.AIGVSEngine(embedding_dim=embedding_dim)

        self.root = DisjointFamily(
            node_id="root_node",
            embedding_dim=embedding_dim,
            pipeline_cfg=self._node_cfg(),
        )
        self.node_registry: Dict[str, BaseNode] = {"root_node": self.root}
        self.chunk_location_map: Dict[str, str] = {}

        # payload cache: chunk_id -> {"id", "text", "metadata"}. Filled on insert, read-only at
        # query time.
        self.payload_cache: Dict[str, dict] = {}

        self.engine.register_branch("root_node", None, True)

    # --- internal helpers ---

    def _node_cfg(self) -> dict:
        return {
            "max_connections": self.max_connections,
            "ef_construction": self.ef_construction,
        }

    def _register_leaf_in_engine(self, leaf_id: str, parent_id: str) -> None:
        """Only place register_leaf is called, so its parameters stay consistent."""
        self.engine.register_leaf(
            leaf_id,
            parent_id,
            max_connections=self.max_connections,
            ef_construction=self.ef_construction,
        )

    def _cache_payload(self, chunk: TextChunk) -> None:
        self.payload_cache[chunk.chunk_id] = {
            "id": chunk.chunk_id,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }

    # --- internal helpers ---

    def save_index(self) -> None:
        state = {
            "root": self.root,
            "node_registry": self.node_registry,
            "chunk_location_map": self.chunk_location_map,
            "payload_cache": self.payload_cache,
            "max_connections": self.max_connections,
            "ef_construction": self.ef_construction,
        }
        os.makedirs(self.storage_path, exist_ok=True)
        with open(self.index_file, "wb") as f:
            pickle.dump(state, f)
        print(f"[SYSTEM] Index state frozen and saved to {self.index_file}")

    def load_index(self) -> bool:
        if not os.path.exists(self.index_file):
            print(f"[SYSTEM] No index found at {self.index_file}. Starting fresh.")
            return False
        with open(self.index_file, "rb") as f:
            state = pickle.load(f)

        self.root = state["root"]
        self.node_registry = state["node_registry"]
        self.chunk_location_map = state["chunk_location_map"]
        self.payload_cache = state.get("payload_cache", {})
        self.max_connections = state.get("max_connections", self.max_connections)
        self.ef_construction = state.get("ef_construction", self.ef_construction)

        # If the cache is missing (old index format), rebuild it from storage.
        if not self.payload_cache:
            print("[SYSTEM] Payload cache missing - rebuilding from storage...")
            self._rebuild_payload_cache_from_storage()

        self.engine = aigvs_rust.AIGVSEngine(embedding_dim=self.embedding_dim)
        self._rebuild_engine_from_registry()
        print(f"[SYSTEM] Index loaded from {self.index_file}")
        return True

    def _rebuild_payload_cache_from_storage(self) -> None:
        """One-time migration: populate cache from storage for old index files."""
        for chunk_id in self.chunk_location_map:
            chunk = self.storage.read_chunk(chunk_id)
            if chunk is not None:
                self._cache_payload(chunk)

    def _rebuild_engine_from_registry(self) -> None:
        """BFS over the Python tree to reconstruct the Rust engine state."""
        queue: deque = deque(["root_node"])
        visited: set = set()

        while queue:
            node_id = queue.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)

            node = self.node_registry.get(node_id)
            if node is None:
                continue

            if isinstance(node, DisjointFamily):
                self.engine.register_branch(node_id, node.parent_id, node_id == "root_node")
                for child_id, child_node in node.children.items():
                    if child_node.centroid_vector is not None:
                        self.engine.update_attractor(node_id, child_id, child_node.centroid_vector)
                    queue.append(child_id)

            elif isinstance(node, JointFamily):
                self._register_leaf_in_engine(node_id, node.parent_id)
                # re-insert in the original order; order affects NSW graph quality
                for i, chunk_id in enumerate(node.chunk_ids):
                    if i in node.deleted_indices:
                        continue
                    # Use cached embedding if available, else fall back to storage.
                    payload = self.payload_cache.get(chunk_id)
                    if payload is not None and "embedding" in payload:
                        embedding = np.array(payload["embedding"], dtype=np.float32)
                    else:
                        chunk = self.storage.read_chunk(chunk_id)
                        if chunk is None:
                            continue
                        embedding = chunk.embedding
                    self.engine.native_insert_to_leaf(node_id, chunk_id, embedding)

    # --- internal helpers ---

    def _recursive_bulk_build(self, parent_id: str, chunks: List[TextChunk], embs: np.ndarray, depth: int = 1) -> Optional[np.ndarray]:
        """Build the tree downwards until the chunk count drops below C_cap. Returns the centroid of
        the created node (leaf or branch).
        """
        N = len(chunks)
        if N == 0:
            return None
            
        parent_node = self.node_registry[parent_id]
        
        # If this branch already has children, route the new batch into them instead of building new
        # ones. Otherwise every batch fragments the tree further.
        if len(parent_node.children) > 0:
            child_ids = list(parent_node.children.keys())
            macro_centroids = np.array([parent_node.children[cid].centroid_vector for cid in child_ids], dtype=np.float32)
            
            sims = embs @ macro_centroids.T
            best_trunks = np.argmax(sims, axis=1)
            
            valid_centroids = []
            
            for t_idx, child_id in enumerate(child_ids):
                doc_indices = np.nonzero(best_trunks == t_idx)[0]
                if len(doc_indices) == 0:
                    continue
                    
                sub_chunks = [chunks[i] for i in doc_indices]
                sub_embs = embs[doc_indices]
                child_node = parent_node.children[child_id]
                
                if isinstance(child_node, JointFamily):
                    # It's a leaf. Route directly into it.
                    c_ids = []
                    for chunk in sub_chunks:
                        self._cache_payload(chunk)
                        chunk.embedding = None # Free memory! Rust engine owns the vector
                        chunk.text = ""        # Free memory! Payload cache owns the text
                        chunk.metadata = {}
                        self.storage.write_chunk(chunk)
                        child_node.chunk_ids.append(chunk.chunk_id)
                        self.chunk_location_map[chunk.chunk_id] = child_id
                        c_ids.append(chunk.chunk_id)
                        
                    self.engine.native_batch_insert_to_leaf(child_id, c_ids, sub_embs)
                    
                    new_c = (child_node.centroid_vector * 0.9) + (np.mean(sub_embs, axis=0) * 0.1)
                    new_c /= np.linalg.norm(new_c)
                    child_node.centroid_vector = new_c
                    self.engine.update_attractor(parent_id, child_id, new_c)
                    valid_centroids.append(new_c)
                    
                else:
                    # It's a sub-branch. Recurse down.
                    sub_centroid = self._recursive_bulk_build(child_id, sub_chunks, sub_embs, depth + 1)
                    if sub_centroid is not None:
                        new_c = (child_node.centroid_vector * 0.9) + (sub_centroid * 0.1)
                        new_c /= np.linalg.norm(new_c)
                        child_node.centroid_vector = new_c
                        self.engine.update_attractor(parent_id, child_id, new_c)
                        valid_centroids.append(new_c)
                        
            if valid_centroids:
                centroid = np.mean(valid_centroids, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid /= norm
                return centroid
            return None
            
        # first batch: build the tree
        c_base = 2500
        n_global = max(len(self.chunk_location_map) + N, 10)
        c_cap = int(c_base * (np.log10(n_global) / 4.0))
        
        if N <= c_cap:
            leaf_id = f"{parent_id}_leaf_{uuid.uuid4().hex[:6]}"
            leaf = JointFamily(
                node_id=leaf_id,
                parent_id=parent_id,
                pipeline_cfg=self._node_cfg(),
            )
            self._register_leaf_in_engine(leaf_id, parent_id)
            
            c_ids = []
            for chunk in chunks:
                self._cache_payload(chunk)
                chunk.embedding = None # Free memory!
                chunk.text = ""        # Free memory!
                chunk.metadata = {}
                self.storage.write_chunk(chunk)
                leaf.chunk_ids.append(chunk.chunk_id)
                self.chunk_location_map[chunk.chunk_id] = leaf_id
                c_ids.append(chunk.chunk_id)
                
            vectors_array = embs
            self.engine.native_batch_insert_to_leaf(leaf_id, c_ids, vectors_array)
            
            centroid = np.mean(vectors_array, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid /= norm
                
            leaf.centroid_vector = centroid
            self.node_registry[leaf_id] = leaf
            
            parent_node.children[leaf_id] = leaf
            self.engine.update_attractor(parent_id, leaf_id, centroid)
            
            return centroid
            
        else:
            num_sub_trunks = max(4, int(4 * np.log10(max(N, 10))))
            if depth > 0:
                num_sub_trunks = min(num_sub_trunks, 8) 
                
            macro_centroids = self._discover_macro_attractors(embs, num_sub_trunks)
            sims = embs @ macro_centroids.T
            max_sims = np.max(sims, axis=1)
            
            sub_chunks = [[] for _ in range(num_sub_trunks)]
            sub_embs = [None for _ in range(num_sub_trunks)]
            
            # Strict assignment for sub-branches (prevent exponential polysemy)
            best_trunks = np.argmax(sims, axis=1)
            mask = np.zeros((N, num_sub_trunks), dtype=bool)
            mask[np.arange(N), best_trunks] = True
            
            for t_idx in range(num_sub_trunks):
                doc_indices = np.nonzero(mask[:, t_idx])[0]
                sub_chunks[t_idx] = [chunks[i] for i in doc_indices]
                sub_embs[t_idx] = embs[doc_indices]
                
            valid_centroids = []
            
            for t_idx, t_chunks in enumerate(sub_chunks):
                if not t_chunks:
                    continue
                    
                sub_branch_id = f"{parent_id}_sub_{t_idx}_{uuid.uuid4().hex[:4]}"
                sub_branch = DisjointFamily(
                    node_id=sub_branch_id,
                    embedding_dim=self.embedding_dim,
                    parent_id=parent_id,
                    pipeline_cfg=self._node_cfg(),
                )
                self.node_registry[sub_branch_id] = sub_branch
                self.engine.register_branch(sub_branch_id, parent_id, False)
                parent_node.children[sub_branch_id] = sub_branch
                
                sub_centroid = self._recursive_bulk_build(sub_branch_id, t_chunks, sub_embs[t_idx], depth + 1)
                
                if sub_centroid is not None:
                    sub_branch.centroid_vector = sub_centroid
                    self.engine.update_attractor(parent_id, sub_branch_id, sub_centroid)
                    valid_centroids.append(sub_centroid)
                    
            if valid_centroids:
                centroid = np.mean(valid_centroids, axis=0)
                norm = np.linalg.norm(centroid)
                if norm > 0:
                    centroid /= norm
                return centroid
            return None

    def ingest_hierarchical_json(self, json_data: List[dict]) -> None:
        if not json_data:
            return
        
        ids = [item["id"] for item in json_data]
        texts = [item.get("text", "") for item in json_data]
        embeddings = np.array([item["embedding"] for item in json_data], dtype=np.float32)
        
        # Normalize directly in numpy array
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embeddings = embeddings / norms
        
        metadatas = [item.get("metadata", {}) for item in json_data]
        self.bulk_ingest_arrays(ids, texts, embeddings, metadatas)

    def bulk_ingest_arrays(
        self, ids: List[str], texts: List[str], embeddings: np.ndarray, metadatas: Optional[List[dict]] = None
    ) -> None:
        """High-speed bulk ingestion avoiding dictionary and list allocations where possible."""
        if not ids:
            return

        N = len(ids)
        if metadatas is None:
            metadatas = [{} for _ in range(N)]
            
        print(f"[AIGVS] Creating chunks natively. N={N}", flush=True)
        chunks = [
            TextChunk(chunk_id=ids[i], text=texts[i], embedding=embeddings[i], metadata=metadatas[i])
            for i in range(N)
        ]
        X = embeddings
        
        # small corpus: skip macro discovery and insert directly
        num_leaves_est = N / self.root.max_leaf_capacity
        if num_leaves_est < 4:
            for chunk in chunks:
                self.root.insert(chunk, self.storage, self)
            return

        # stage 1: find (or reuse) the macro attractors
        if len(self.root.children) > 0:
            print(f"[AIGVS] Reusing {len(self.root.children)} existing global trunks for chunked assignment...", flush=True)
            trunk_ids = list(self.root.children.keys())
            macro_centroids = np.array([self.root.children[tid].centroid_vector for tid in trunk_ids], dtype=np.float32)
            num_trunks = len(trunk_ids)
        else:
            print(f"[AIGVS] Discovering macro attractors...", flush=True)
            t_start = time.time()
            num_trunks = max(8, int(4 * np.log10(max(N, 10))))
            print(f"[AIGVS] Discovering {num_trunks} macro semantic basins from {N} documents...", flush=True)
            macro_centroids = self._discover_macro_attractors(X, num_trunks)
            trunk_ids = [f"trunk_global_{uuid.uuid4().hex[:4]}_{i}" for i in range(num_trunks)]
            print(f"[AIGVS] Discovered macro attractors: {time.time()-t_start:.2f}s", flush=True)

        # stage 2: assign docs to trunks, with polysemic duplication
        t_start = time.time()
        sims = X @ macro_centroids.T
        max_sims = np.max(sims, axis=1)

        trunk_chunks: List[List[TextChunk]] = [[] for _ in range(num_trunks)]
        trunk_embs: List[np.ndarray] = [None for _ in range(num_trunks)]
        
        # Vectorized assignment mask: True if sim >= max_sim - 0.08
        thresholds = max_sims - 0.08
        mask = sims >= thresholds[:, None]
        
        # Fast extraction of chunks per trunk
        for t_idx in range(num_trunks):
            doc_indices = np.nonzero(mask[:, t_idx])[0]
            trunk_chunks[t_idx] = [chunks[i] for i in doc_indices]
            trunk_embs[t_idx] = X[doc_indices]
            
        polysemy_count = np.sum(np.sum(mask, axis=1) > 1)
        print(f"[AIGVS] Stage 2 Vectorized Assignment: {time.time()-t_start:.2f}s", flush=True)
                
        print(f"[AIGVS] Polysemic assignments: {polysemy_count} chunks span multiple macro trunks")

        # stage 3: create or reuse trunk branches and insert
        for trunk_idx, t_chunks in enumerate(trunk_chunks):
            if not t_chunks:
                continue

            trunk_id = trunk_ids[trunk_idx]
            
            if trunk_id not in self.root.children:
                trunk = DisjointFamily(
                    node_id=trunk_id,
                    embedding_dim=self.embedding_dim,
                    parent_id="root_node",
                    pipeline_cfg=self._node_cfg(),
                )
                self.root.children[trunk_id] = trunk
                self.engine.register_branch(trunk_id, "root_node", False)
                self.node_registry[trunk_id] = trunk
            else:
                trunk = self.root.children[trunk_id]

            t_start = time.time()
            sub_centroid = self._recursive_bulk_build(trunk_id, t_chunks, trunk_embs[trunk_idx], depth=1)
            if sub_centroid is not None:
                # Exponential moving average for centroid drift
                current = trunk.centroid_vector
                if current is not None:
                    new_c = (current * 0.7) + (sub_centroid * 0.3)
                    new_c /= np.linalg.norm(new_c)
                    trunk.centroid_vector = new_c
                else:
                    trunk.centroid_vector = sub_centroid
                
                self.engine.update_attractor("root_node", trunk_id, trunk.centroid_vector)
            
            trunk.chunk_count += len(t_chunks)
            print(f"[AIGVS] Built trunk {trunk_idx} ({len(t_chunks)} chunks): {time.time()-t_start:.2f}s", flush=True)

        # final pass: update the root attractors to match what was inserted
        for trunk_id, trunk in self.root.children.items():
            if trunk.centroid_vector is not None:
                self.engine.update_attractor(
                    "root_node", trunk_id, trunk.centroid_vector
                )

        stats = self.stats()
        print(
            f"[AIGVS] Tree built: {stats['branches']} branches, "
            f"{stats['leaves']} leaves, {stats['indexed_chunks']} chunks"
        )

    def _discover_macro_attractors(
        self, X: np.ndarray, num_trunks: int
    ) -> np.ndarray:
        """Fast macro-attractor discovery: farthest-point seeding + spherical k-means."""
        # farthest-point sampling for well-separated seeds
        avg = np.mean(X, axis=0)
        avg_norm = np.linalg.norm(avg)
        if avg_norm > 0:
            avg /= avg_norm

        seeds = [X[int(np.argmax(X @ avg))].copy()]

        for _ in range(num_trunks - 1):
            seed_matrix = np.array(seeds)
            sims = X @ seed_matrix.T          # (N, K)
            max_sim = np.max(sims, axis=1)    # (N,)
            seeds.append(X[int(np.argmin(max_sim))].copy())

        centroids = np.array(seeds, dtype=np.float32)

        # spherical k-means refinement (5 iterations)
        for _ in range(5):
            scores = X @ centroids.T
            assignments = np.argmax(scores, axis=1)
            for k in range(len(centroids)):
                mask = assignments == k
                if mask.any():
                    c = np.mean(X[mask], axis=0)
                    norm = np.linalg.norm(c)
                    if norm > 0:
                        centroids[k] = c / norm

        return centroids

    def insert_chunk(self, chunk: TextChunk) -> None:
        self.root.insert(chunk, self.storage, self)

    # --- query (no storage I/O on this path) ---

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        max_beam: int = 10,
        target_confidence: float = 0.95,
        beta: float = 5.0,
        ef_search: Optional[int] = None,
    ) -> List[dict]:
        """Parameters
            ef_search : int, optional
                NSW candidate pool per leaf at query time, default max(top_k * 4, 50). Cheapest way to
                get more recall.
            beta : float
                Router softmax temperature. Low (~2) explores more leaves, high (~20) goes greedily to
                the best one.
        """
        norm = np.linalg.norm(query_embedding)
        if norm > 0:
            query_embedding = query_embedding / norm

        ef = ef_search if ef_search is not None else max(top_k * 8, 100)

        # one PyO3 call: Rust does the routing and the parallel leaf search
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

        # dict lookup per result, no I/O
        return [
            self.payload_cache[cid]
            for cid in matched_ids
            if cid in self.payload_cache
        ]

    # --- internal helpers ---

    def delete_chunk(self, chunk_id: str) -> bool:
        leaf_id = self.chunk_location_map.get(chunk_id)
        if not leaf_id:
            return False

        self.storage.delete_chunk(chunk_id)
        self.engine.native_mark_deleted(leaf_id, chunk_id)
        self.payload_cache.pop(chunk_id, None)

        leaf_node: Optional[JointFamily] = self.node_registry.get(leaf_id)
        parent_node: Optional[DisjointFamily] = (
            self.node_registry.get(leaf_node.parent_id) if leaf_node else None
        )
        if leaf_node and parent_node:
            leaf_node.mark_deleted(chunk_id, parent_node, self.engine)

        self.chunk_location_map.pop(chunk_id, None)
        return True

    def update_chunk(self, chunk: TextChunk) -> None:
        self.delete_chunk(chunk.chunk_id)
        self.root.insert(chunk, self.storage, self)

    # --- internal helpers ---

    def stats(self) -> dict:
        return {
            "branches": sum(1 for n in self.node_registry.values() if isinstance(n, DisjointFamily)),
            "leaves": sum(1 for n in self.node_registry.values() if isinstance(n, JointFamily)),
            "indexed_chunks": len(self.chunk_location_map),
            "payload_cache_size": len(self.payload_cache),
            "embedding_dim": self.embedding_dim,
            "max_connections": self.max_connections,
            "ef_construction": self.ef_construction,
        }