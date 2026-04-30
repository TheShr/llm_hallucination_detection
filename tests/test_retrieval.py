from app.retrieval.bm25 import BM25Retriever
from app.retrieval.ingest import load_documents
from app.retrieval.vector_store import VectorStore


def test_bm25_retriever_returns_documents() -> None:
    documents = load_documents()
    retriever = BM25Retriever(documents)
    results = retriever.retrieve("language models", top_n=2)
    assert len(results) == 2
    assert all("id" in item for item in results)


def test_vector_store_search_returns_similar_documents() -> None:
    documents = load_documents()
    store = VectorStore(documents)
    results = store.search("semantic similarity", top_k=2)
    assert len(results) <= 2
    assert all("score" in item for item in results)
