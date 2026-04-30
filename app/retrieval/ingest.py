import json
from pathlib import Path
from typing import List

from app.utils.config import settings
from app.utils.types import Document
from app.utils.logging import logger


def load_documents(path: Path = settings.documents_path) -> List[Document]:
    """Load the sample documents from disk."""
    logger.info("Loading documents from %s", path)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    documents: List[Document] = []
    for entry in data:
        documents.append(
            {
                "id": entry["id"],
                "title": entry["title"],
                "text": entry["text"],
                "metadata": entry.get("metadata", {}),
            }
        )
    logger.info("Loaded %d documents", len(documents))
    return documents
