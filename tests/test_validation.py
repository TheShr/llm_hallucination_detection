from app.validation.verifier import HallucinationVerifier


def test_verification_scores_consistency() -> None:
    verifier = HallucinationVerifier()
    docs = [{"id": "doc-1", "title": "A", "text": "This is a test document about AI.", "score": 0.9}]
    answer = "AI is explained in this text."
    hallucination, confidence = verifier.verify(answer, docs)
    assert isinstance(confidence, float)
    assert hallucination in (True, False)


def test_verifier_detects_unsupported_answer() -> None:
    verifier = HallucinationVerifier()
    docs = [{"id": "doc-1", "title": "AI facts", "text": "Large language models are trained on massive text corpora.", "score": 0.9}]
    answer = "The capital of France is Berlin."
    hallucination, confidence = verifier.verify(answer, docs)
    assert hallucination is True
    assert confidence < 0.5


def test_verifier_confidence_for_supported_answer() -> None:
    verifier = HallucinationVerifier()
    docs = [
        {"id": "doc-6", "title": "Apple Inc", "text": "Apple Inc. is a technology company known for the iPhone, iPad, and Mac computers.", "score": 0.9},
        {"id": "doc-7", "title": "Apple fruit", "text": "An apple is a sweet fruit produced by the apple tree and is commonly eaten fresh or used in cooking.", "score": 0.8},
    ]
    answer = "Apple can refer to the technology company Apple Inc. or the fruit produced by an apple tree."
    hallucination, confidence = verifier.verify(answer, docs)
    assert hallucination is False
    assert confidence >= 0.3
