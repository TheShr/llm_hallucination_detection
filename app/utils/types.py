from typing import List, TypedDict

class Document(TypedDict):
    id: str
    title: str
    text: str
    metadata: dict

class RetrievalResult(TypedDict):
    id: str
    title: str
    text: str
    score: float

class ResponsePayload(TypedDict):
    answer: str
    confidence_score: float
    hallucination: bool
    sources: List[dict]
