import json
from typing import List

from app.evaluation.metrics import (
    confidence_statistics,
    detection_precision_recall,
    grounding_score,
    precision_at_k,
)
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.ingest import load_documents
from app.generation.llm import LLMGenerator
from app.validation.verifier import HallucinationVerifier
from app.utils.logging import logger


def run_evaluation(test_cases_path: str) -> None:
    logger.info("Running evaluation pipeline")
    documents = load_documents()
    retriever = HybridRetriever(documents)
    llm = LLMGenerator()
    verifier = HallucinationVerifier()

    with open(test_cases_path, "r", encoding="utf-8") as handle:
        cases = json.load(handle)

    results = []
    predictions: List[bool] = []
    labels: List[bool] = []
    confidences: List[float] = []

    for case in cases:
        query = case["query"]
        relevant = case.get("relevant_ids", [])
        sources = retriever.retrieve(query, top_n=5)
        retrieval_precision = precision_at_k(sources, relevant, k=5)
        retrieval_context = retriever.get_context(query, top_n=5)

        answer = case.get("answer")
        if answer is None:
            answer = llm.generate(query, retrieval_context)

        hallucination, confidence_score = verifier.verify(answer, sources)
        expected_label = bool(case.get("expected_hallucination", False))

        predictions.append(hallucination)
        labels.append(expected_label)
        confidences.append(confidence_score)

        result = {
            "query": query,
            "precision_at_5": retrieval_precision,
            "grounding_score": grounding_score(answer, sources),
            "hallucination_predicted": hallucination,
            "expected_hallucination": expected_label,
            "confidence_score": confidence_score,
            "relevant_ids": relevant,
            "answer": answer,
        }
        results.append(result)
        logger.info(
            "Evaluated query '%s' prec=%.3f hall=%s conf=%.3f",
            query,
            retrieval_precision,
            hallucination,
            confidence_score,
        )

    detection_report = detection_precision_recall(predictions, labels)
    confidence_report = confidence_statistics(confidences)
    output = {
        "results": results,
        "summary": {
            "retrieval_examples": len(results),
            "detection_precision": detection_report["precision"],
            "detection_recall": detection_report["recall"],
            "detection_f1": detection_report["f1"],
            "confidence_mean": confidence_report["mean"],
            "confidence_min": confidence_report["min"],
            "confidence_max": confidence_report["max"],
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    run_evaluation("data/test_queries.json")
