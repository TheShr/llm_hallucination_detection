from functools import lru_cache
from typing import Dict, List

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.vector_store import VectorStore
from app.utils.logging import logger
from app.utils.types import Document, RetrievalResult


class HybridRetriever:
    """Hybrid retrieval that merges BM25 and vector search results for robust RAG."""

    def __init__(self, documents: List[Document], bm25_top_n: int = 5, vector_top_k: int = 5) -> None:
        self.documents = documents
        self.bm25 = BM25Retriever(documents)
        self.vector = VectorStore(documents)
        self.bm25_top_n = bm25_top_n
        self.vector_top_k = vector_top_k
        self._cached_retrieve = self._make_cached_retrieve()

    def _make_cached_retrieve(self):
        @lru_cache(maxsize=128)
        def cached(query: str) -> List[RetrievalResult]:
            return self._retrieve_impl(query)

        return cached

    @staticmethod
    def _normalize_scores(results: List[RetrievalResult]) -> List[RetrievalResult]:
        scores = [result["score"] for result in results]
        if not scores:
            return results
        min_score = min(scores)
        max_score = max(scores)
        spread = max_score - min_score if max_score != min_score else 1.0
        normalized = []
        for result in results:
            normalized_score = (result["score"] - min_score) / spread
            normalized.append({**result, "score": normalized_score})
        return normalized

    def _merge_results(self, bm25_results: List[RetrievalResult], vector_results: List[RetrievalResult]) -> List[RetrievalResult]:
        combined: Dict[str, RetrievalResult] = {}
        for source_type, dataset in [("bm25", bm25_results), ("vector", vector_results)]:
            for result in dataset:
                doc_id = result["id"]
                existing = combined.get(doc_id)
                if existing is None:
                    combined[doc_id] = {**result, "source": source_type}
                else:
                    combined[doc_id]["score"] = max(existing["score"], result["score"])
                    combined[doc_id]["source"] = ",".join(sorted({existing.get("source", source_type), source_type}))
        merged = sorted(combined.values(), key=lambda item: item["score"], reverse=True)
        logger.debug("Merged %d retrieval candidates", len(merged))
        return merged

    def _retrieve_impl(self, query: str) -> List[RetrievalResult]:
        logger.info("Performing hybrid retrieval for query: %s", query)
        bm25_results = self._normalize_scores(self.bm25.retrieve(query, top_n=self.bm25_top_n))
        vector_results = self._normalize_scores(self.vector.search(query, top_k=self.vector_top_k))
        merged_results = self._merge_results(bm25_results, vector_results)
        return merged_results

    def retrieve(self, query: str, top_n: int = 5) -> List[RetrievalResult]:
        """Retrieve top documents from the hybrid retrieval pipeline."""
        return self._cached_retrieve(query)[:top_n]

    def get_context(self, query: str, top_n: int = 5) -> str:
        docs = self.retrieve(query, top_n=top_n)
        return "\n\n".join([f"Title: {doc['title']}\n{doc['text']}" for doc in docs])
