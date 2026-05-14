"""
Nigerian Fintech Compliance Agent Retriever
rag/retriever.py
"""

import os
import pickle
import logging
import threading

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Constants for artifact paths and embedding model name
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "index", "compliance_index.faiss")
CHUNKS_PATH      = os.path.join(BASE_DIR, "index", "chunks.pkl")
EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"

# Note: this module is designed for efficient retrieval at runtime, not for indexing.
# The ComplianceRetriever class loads the pre-built FAISS index and corresponding text chunks,
# and provides a retrieve() method to get relevant chunks for a given query.
class ComplianceRetriever:
    def __init__(self) -> None:
        self._validate_artifacts()

        self.index: faiss.IndexFlatL2 = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            self.chunks: list[str] = pickle.load(f)

        if self.index.ntotal != len(self.chunks):
            raise RuntimeError(
                f"Index/chunk mismatch: FAISS has {self.index.ntotal} vectors "
                f"but chunks.pkl has {len(self.chunks)} entries. "
                "Re-run rag/indexer.py to rebuild both artifacts."
            )

        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self._dimension: int = self.index.d
    # The retrieve() method takes a query string and an optional top_k parameter (default 5),
    # and returns a list of the most relevant text chunks from the index.
    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        query = query.strip() if isinstance(query, str) else ""
        if not query:
            raise ValueError("retrieve() called with an empty query string.")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}.")

        effective_k = min(top_k, self.index.ntotal)
        if effective_k < top_k:
            logger.warning(
                "top_k=%d exceeds index size (%d); returning %d results.",
                top_k, self.index.ntotal, effective_k,
            )

        embedding = self.model.encode([query])
        embedding = np.array(embedding, dtype="float32")

        if embedding.shape[1] != self._dimension:
            raise RuntimeError(
                f"Query embedding dimension {embedding.shape[1]} does not match "
                f"index dimension {self._dimension}."
            )

        distances, indices = self.index.search(embedding, effective_k)

        return [
            self.chunks[i]
            for i in indices[0]
            if i != -1 and 0 <= i < len(self.chunks)
        ]
    # Validates that the required FAISS index and chunks artifacts exist on disk.
    @staticmethod
    def _validate_artifacts() -> None:
        missing = [p for p in (FAISS_INDEX_PATH, CHUNKS_PATH) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "FAISS index artifacts not found:\n"
                + "\n".join(f"  {p}" for p in missing)
                + "\nRun `python -m rag.indexer` to build them before starting the app."
            )

# Singleton pattern to ensure only one ComplianceRetriever instance is created and shared across the app.
_lock = threading.Lock()
_retriever_instance: ComplianceRetriever | None = None

# Factory function to get the singleton ComplianceRetriever instance.
def get_retriever() -> ComplianceRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        with _lock:
            if _retriever_instance is None:
                _retriever_instance = ComplianceRetriever()
    return _retriever_instance