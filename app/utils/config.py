import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / ".." / "data"

load_dotenv(BASE_DIR.parent / ".env")


def parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class Settings:
    """Configuration settings for the system."""

    def __init__(self) -> None:
        self.openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
        self.openai_api_base: Optional[str] = (
            os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL")
        )
        self.embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        self.nli_model: str = os.getenv("NLI_MODEL", "roberta-large-mnli")
        self.rag_top_k: int = int(os.getenv("RAG_TOP_K", "4"))
        self.bm25_top_n: int = int(os.getenv("BM25_TOP_N", "5"))
        self.faiss_index_path: Path = BASE_DIR / ".." / "data" / "faiss.index"
        self.documents_path: Path = BASE_DIR / ".." / "data" / "docs.json"
        self.use_openai: bool = bool(self.openai_api_key)
        self.use_nli_verification: bool = parse_bool(os.getenv("USE_NLI_VERIFICATION"), False)

        if self.openai_api_key and self.openai_api_key.startswith("gsk_") and not self.openai_api_base:
            self.openai_api_base = "https://api.groq.com/v1"


settings = Settings()
