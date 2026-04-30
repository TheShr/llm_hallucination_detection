import os
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / ".." / "data"

class Settings:
    """Configuration settings for the system."""

    def __init__(self) -> None:
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.nli_model: str = os.getenv("NLI_MODEL", "roberta-large-mnli")
        self.rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))
        self.bm25_top_n: int = int(os.getenv("BM25_TOP_N", "5"))
        self.faiss_index_path: Path = BASE_DIR / ".." / "data" / "faiss.index"
        self.documents_path: Path = BASE_DIR / ".." / "data" / "docs.json"
        self.use_openai: bool = bool(self.openai_api_key)
        self.use_nli_verification: bool = os.getenv("USE_NLI_VERIFICATION", "0") in {"1", "true", "True"}

settings = Settings()
