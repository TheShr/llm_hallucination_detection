from typing import List

from app.utils.types import RetrievalResult


def precision_at_k(retrieved: List[RetrievalResult], relevant_ids: List[str], k: int = 5) -> float:
    """Compute precision at k for retrieval results."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item["id"] in relevant_ids)
    return hits / len(top_k)


def grounding_score(answer: str, sources: List[RetrievalResult]) -> float:
    """A simple grounding score based on source overlap with answer."""
    answer_tokens = set(answer.lower().split())
    source_tokens = set(" ".join(source["text"] for source in sources).lower().split())
    if not answer_tokens:
        return 0.0
    overlap = len(answer_tokens & source_tokens)
    return overlap / len(answer_tokens)
