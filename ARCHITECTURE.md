# System Architecture

## Overview

The LLM Hallucination Detection and RAG Validation System is designed as a modular pipeline for production-ready retrieval-augmented generation.

### Pipeline stages

1. **User Query**
   - Incoming requests are accepted by FastAPI at `POST /api/query`.
   - The request payload passes validation and observability logging.

2. **Hybrid Retrieval**
   - `BM25Retriever` performs lexical retrieval from the document corpus.
   - `VectorStore` uses FAISS embeddings for semantic retrieval.
   - `HybridRetriever` merges the two scored candidate sets, normalizes scores, and selects the top grounding passages.

3. **LLM Generation**
   - `LLMGenerator` builds a grounded prompt using retrieved context.
   - If OpenAI credentials are available, it routes queries to `gpt-3.5-turbo` or the configured OpenAI model.
   - Otherwise it falls back to a safe placeholder response.

4. **Validation**
   - `HallucinationVerifier` computes:
     - semantic grounding score between answer and retrieved sources
     - keyword overlap ratio across answer and source text
     - optional NLI entailment score using `roberta-large-mnli`
   - These metrics are aggregated into a final confidence score.
   - Low-confidence answers are flagged as hallucinated.

5. **Response**
   - The API returns:
     - `answer`
     - `confidence_score`
     - `hallucination`
     - `sources` with provenance and score metadata

## Component Diagram

User Query → FastAPI API → HybridRetriever → Retrieved Sources → Prompt Builder → LLM Generator → Answer
             ↓
             Validation Module → Confidence Score / Hallucination Flag

## Modules

- `app/api` — request handling, validation, async orchestration
- `app/retrieval` — document ingestion, BM25, FAISS, hybrid merging, caching
- `app/generation` — prompt construction and LLM interface
- `app/validation` — semantic, lexical, and optional NLI verification
- `app/evaluation` — retrieval precision and grounding metric runner
- `app/frontend` — minimal Streamlit UI
- `app/utils` — shared configuration, logging, typing

## Optimizations

- Cached retrieval results via `lru_cache` on query-level retrieval operations
- Score normalization across BM25 and vector results for stable fusion
- Hybrid candidate merging with provenance tracking
- Async endpoint execution to keep request flow responsive

## Validation Expansion

The validation module now supports three orthogonal signals:

- `semantic` similarity from dense embeddings
- `lexical` keyword overlap between answer and source text
- `NLI` entailment score when enabled by `USE_NLI_VERIFICATION=1`

These signals are combined into a confidence score, which improves robustness against unsupported LLM generations.
