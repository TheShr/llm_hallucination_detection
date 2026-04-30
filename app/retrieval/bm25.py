from functools import lru_cache
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from app.utils.types import Document, RetrievalResult


class BM25Retriever:
    """Hybrid retrieval with BM25 ranking over tokenized passages."""

    def __init__(self, documents: List[Document]) -> None:
        self.documents = documents
        self.tokenized_corpus = [doc["text"].split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._cached_retrieve = lru_cache(maxsize=128)(self._retrieve_impl)

    def _retrieve_impl(self, query: str, top_n: int) -> List[RetrievalResult]:
        scores = self.bm25.get_scores(query.split())
        ranked = sorted(
            enumerate(scores), key=lambda item: item[1], reverse=True
        )[:top_n]
        return [
            {
                "id": self.documents[idx]["id"],
                "title": self.documents[idx]["title"],
                "text": self.documents[idx]["text"],
                "score": float(score),
            }
            for idx, score in ranked
        ]

    def retrieve(self, query: str, top_n: int = 5) -> List[RetrievalResult]:
        """Return top_n BM25-ranked documents for the query."""
        return self._cached_retrieve(query, top_n)

    def get_relevant_text(self, query: str, top_n: int = 5) -> str:
        """Concatenate the top BM25 passages into a single context block."""
        docs = self.retrieve(query, top_n=top_n)
        return "\n\n".join([doc["text"] for doc in docs])
