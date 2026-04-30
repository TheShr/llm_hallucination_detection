from app.validation.verifier import HallucinationVerifier


def test_verification_scores_consistency() -> None:
    verifier = HallucinationVerifier()
    docs = [{"id": "doc-1", "title": "A", "text": "This is a test document about AI.", "score": 0.9}]
    answer = "AI is explained in this text."
    hallucination, confidence = verifier.verify(answer, docs)
    assert isinstance(confidence, float)
    assert hallucination in (True, False)
