import json
from typing import List

from app.evaluation.metrics import grounding_score, precision_at_k
from app.retrieval.bm25 import BM25Retriever
from app.retrieval.ingest import load_documents
from app.retrieval.vector_store import VectorStore
from app.utils.logging import logger


def run_evaluation(test_cases_path: str) -> None:
    logger.info("Running evaluation pipeline")
    documents = load_documents()
    bm25 = BM25Retriever(documents)
    vector_store = VectorStore(documents)

    with open(test_cases_path, "r", encoding="utf-8") as handle:
        cases = json.load(handle)

    results = []
    for case in cases:
        query = case["query"]
        relevant = case.get("relevant_ids", [])
        bm25_results = bm25.retrieve(query, top_n=5)
        vector_results = vector_store.search(query, top_k=5)
        retrieval_precision = precision_at_k(bm25_results, relevant, k=5)

        result = {
            "query": query,
            "precision_at_5": retrieval_precision,
            "bm25_count": len(bm25_results),
            "vector_count": len(vector_results),
            "relevant_ids": relevant,
        }
        results.append(result)
        logger.info("Evaluated query '%s' precision %.3f", query, retrieval_precision)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    run_evaluation("data/test_queries.json")
