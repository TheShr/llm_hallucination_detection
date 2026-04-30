from typing import Dict, List

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


def detection_precision_recall(predictions: List[bool], labels: List[bool]) -> Dict[str, float]:
    """Compute detection precision and recall for hallucination classification."""
    true_positive = sum(1 for pred, label in zip(predictions, labels) if pred and label)
    false_positive = sum(1 for pred, label in zip(predictions, labels) if pred and not label)
    false_negative = sum(1 for pred, label in zip(predictions, labels) if not pred and label)
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def confidence_statistics(confidences: List[float]) -> Dict[str, float]:
    """Compute basic confidence statistics for answer validation."""
    if not confidences:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": sum(confidences) / len(confidences),
        "min": min(confidences),
        "max": max(confidences),
    }
