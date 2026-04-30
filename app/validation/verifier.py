import re
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer
from transformers import pipeline

from app.utils.config import settings
from app.utils.logging import logger
from app.utils.types import RetrievalResult


class HallucinationVerifier:
    """Detect hallucination by measuring grounding, overlap, and optional NLI support."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.embedder = SentenceTransformer(model_name)
        self.nli_model_name = settings.nli_model
        self.nli_pipeline = None
        if settings.use_nli_verification:
            logger.info("Initializing NLI pipeline: %s", self.nli_model_name)
            self.nli_pipeline = pipeline(
                "text-classification",
                model=self.nli_model_name,
                tokenizer=self.nli_model_name,
                return_all_scores=True,
            )

    def compute_semantic_score(self, answer: str, docs: List[RetrievalResult]) -> float:
        """Compute semantic similarity between the answer and retrieved sources."""
        logger.debug("Computing semantic grounding score")
        answer_embedding = self.embedder.encode([answer], convert_to_numpy=True, normalize_embeddings=True)
        doc_embeddings = self.embedder.encode([doc["text"] for doc in docs], convert_to_numpy=True, normalize_embeddings=True)
        similarities = (answer_embedding @ doc_embeddings.T).flatten().tolist()
        max_similarity = max(similarities) if similarities else 0.0
        logger.debug("Max semantic similarity: %s", max_similarity)
        return float(max_similarity)

    @staticmethod
    def compute_keyword_overlap(answer: str, docs: List[RetrievalResult]) -> float:
        """Compute keyword overlap ratio between answer and retrieved sources."""
        logger.debug("Computing keyword overlap")
        answer_terms = set(re.findall(r"\w+", answer.lower()))
        doc_terms = set()
        for doc in docs:
            doc_terms.update(re.findall(r"\w+", doc["text"].lower()))
        if not answer_terms:
            return 0.0
        overlap = len(answer_terms & doc_terms)
        ratio = overlap / len(answer_terms)
        logger.debug("Keyword overlap ratio: %s", ratio)
        return float(ratio)

    def compute_nli_score(self, answer: str, docs: List[RetrievalResult]) -> float:
        """Compute an entailment score between the answer and sources."""
        if self.nli_pipeline is None:
            return 0.0

        logger.debug("Computing NLI-based verification score")
        entailment_scores = []
        for doc in docs:
            try:
                result = self.nli_pipeline((doc["text"], answer))
            except Exception as exc:
                logger.warning("NLI inference failed for doc %s: %s", doc["id"], exc)
                continue

            if not result or not isinstance(result, list):
                continue

            label_scores = {item["label"].upper(): item["score"] for item in result}
            entailment_scores.append(label_scores.get("ENTAILMENT", 0.0))

        if not entailment_scores:
            return 0.0

        score = sum(entailment_scores) / len(entailment_scores)
        logger.debug("Average entailment score: %s", score)
        return float(score)

    def aggregate_metrics(self, metrics: Dict[str, float]) -> float:
        """Aggregate validation metrics into a single confidence score."""
        weights = {
            "semantic": 0.5,
            "overlap": 0.25,
            "nli": 0.25 if settings.use_nli_verification else 0.0,
        }
        confidence = 0.0
        total_weight = 0.0
        for name, value in metrics.items():
            weight = weights.get(name, 0.0)
            confidence += weight * value
            total_weight += weight

        if total_weight == 0:
            return 0.0
        return min(1.0, confidence / total_weight)

    @staticmethod
    def extract_supporting_sources(answer: str, docs: List[RetrievalResult]) -> List[Dict[str, str]]:
        """Produce a small summary of which documents likely support the answer."""
        support_docs = []
        answer_terms = set(re.findall(r"\w+", answer.lower()))
        for doc in docs:
            doc_terms = set(re.findall(r"\w+", doc["text"].lower()))
            overlap = len(answer_terms & doc_terms)
            support_docs.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "support_match": overlap,
                }
            )
        return sorted(support_docs, key=lambda item: item["support_match"], reverse=True)

    def verify(self, answer: str, docs: List[RetrievalResult]) -> Tuple[bool, float]:
        """Return whether answer is hallucinated and a confidence score."""
        semantic_score = self.compute_semantic_score(answer, docs)
        overlap_score = self.compute_keyword_overlap(answer, docs)
        nli_score = self.compute_nli_score(answer, docs)

        metrics = {
            "semantic": semantic_score,
            "overlap": overlap_score,
            "nli": nli_score,
        }
        confidence_score = self.aggregate_metrics(metrics)
        hallucination_detected = confidence_score < 0.55
        logger.info(
            "Verification result: semantic=%.3f overlap=%.3f nli=%.3f confidence=%.3f hallucination=%s",
            semantic_score,
            overlap_score,
            nli_score,
            confidence_score,
            hallucination_detected,
        )
        return hallucination_detected, confidence_score
