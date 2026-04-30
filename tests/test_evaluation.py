import json
from pathlib import Path

import pytest
from app.evaluation.metrics import confidence_statistics, detection_precision_recall


def test_detection_precision_recall_calculation() -> None:
    predictions = [True, False, True, False, True]
    labels = [True, False, False, False, True]
    metrics = detection_precision_recall(predictions, labels)

    assert metrics["precision"] == 0.6666666666666666
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 0.8


def test_confidence_statistics_returns_expected_values() -> None:
    stats = confidence_statistics([0.2, 0.5, 0.9])
    assert stats["mean"] == pytest.approx(0.5333333333333333)
    assert stats["min"] == 0.2
    assert stats["max"] == 0.9


def test_test_queries_dataset_contains_hallucination_annotations() -> None:
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "test_queries.json"
    with open(dataset_path, "r", encoding="utf-8") as handle:
        cases = json.load(handle)

    assert any(case.get("expected_hallucination", False) for case in cases)
    assert all("query" in case and "relevant_ids" in case for case in cases)
