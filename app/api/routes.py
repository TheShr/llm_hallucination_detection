from typing import Any, Dict, List

import anyio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.retrieval.hybrid import HybridRetriever
from app.retrieval.ingest import load_documents
from app.generation.llm import LLMGenerator
from app.validation.verifier import HallucinationVerifier
from app.utils.logging import logger

router = APIRouter()

documents = load_documents()
hybrid_retriever = HybridRetriever(documents)
llm = LLMGenerator()
verifier = HallucinationVerifier()


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    confidence_score: float
    hallucination: bool
    sources: List[Dict[str, Any]]


async def _run_query(query: str) -> Dict[str, Any]:
    sources = hybrid_retriever.retrieve(query, top_n=5)
    retrieval_context = hybrid_retriever.get_context(query, top_n=5)

    answer = await anyio.to_thread.run_sync(llm.generate, query, retrieval_context)
    hallucination, confidence_score = await anyio.to_thread.run_sync(verifier.verify, answer, sources)

    payload = {
        "answer": answer,
        "confidence_score": confidence_score,
        "hallucination": hallucination,
        "sources": [
            {
                "id": source["id"],
                "title": source["title"],
                "score": source["score"],
                "source": source.get("source", "hybrid"),
            }
            for source in sources
        ],
    }
    return payload


@router.post("/query", response_model=QueryResponse)
async def query_llm(request: QueryRequest) -> QueryResponse:
    """Handle user queries through retrieval, generation, and validation."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    logger.info("Received query: %s", request.query)
    payload = await _run_query(request.query)
    logger.info("Query result prepared")
    return payload
