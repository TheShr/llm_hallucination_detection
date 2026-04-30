import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import faiss
from sentence_transformers import SentenceTransformer

from app.utils.config import settings
from app.utils.logging import logger
from app.utils.types import Document, RetrievalResult


class VectorStore:
    """FAISS-backed vector retrieval for passage embeddings."""

    def __init__(self, documents: List[Document], embeddings_model: str = settings.embeddings_model) -> None:
        self.documents = documents
        self.model = SentenceTransformer(embeddings_model)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.id_to_doc: Dict[int, Document] = {}
        self._build_index()

    def _build_index(self) -> None:
        texts = [doc["text"] for doc in self.documents]
        logger.info("Encoding %d documents for FAISS index", len(texts))
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.id_to_doc = {idx: doc for idx, doc in enumerate(self.documents)}
        self._cached_search = lru_cache(maxsize=128)(self._search_impl)
        logger.info("FAISS index built with dimension %d", dim)

    def _search_impl(self, query: str, top_k: int) -> List[RetrievalResult]:
        query_embedding = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding, top_k)
        results: List[RetrievalResult] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            doc = self.id_to_doc[idx]
            results.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "score": float(score),
                }
            )
        return results

    def search(self, query: str, top_k: int = 4) -> List[RetrievalResult]:
        """Search the FAISS index and return ranked documents."""
        return self._cached_search(query, top_k)

    def save(self, path: Path = settings.faiss_index_path) -> None:
        """Persist FAISS index to disk."""
        if self.index is None:
            raise RuntimeError("Index has not been built")
        os.makedirs(path.parent, exist_ok=True)
        faiss.write_index(self.index, str(path))
        logger.info("Saved FAISS index to %s", path)

    def load(self, path: Path = settings.faiss_index_path) -> None:
        """Load FAISS index from disk."""
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        self.index = faiss.read_index(str(path))
        logger.info("Loaded FAISS index from %s", path)
