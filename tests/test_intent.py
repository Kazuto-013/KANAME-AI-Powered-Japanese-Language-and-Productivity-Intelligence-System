import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine.machine_learning.intent import (
    detect_intent,
    INTENT_LABELS,
    intent_classifier_info,
    evaluate_intent_classifier,
)


def test_greeting_intent():
    print("=== test_greeting_intent ===")
    pred = detect_intent("おはようございます")
    print("intent:", pred.intent, "confidence:", pred.confidence)
    assert pred.intent in INTENT_LABELS, f"intent {pred.intent} not in {INTENT_LABELS}"
    assert 0.0 <= pred.confidence <= 1.0, f"confidence out of range: {pred.confidence}"
    print("PASSED: greeting returns prediction with valid intent label and confidence")


def test_question_intent():
    print("=== test_question_intent ===")
    pred = detect_intent("これは何ですか？")
    print("intent:", pred.intent, "confidence:", pred.confidence)
    assert pred.intent == "question", f"expected question, got {pred.intent}"
    assert pred.intent in INTENT_LABELS
    assert 0.0 <= pred.confidence <= 1.0
    print("PASSED: clear question detected as 'question'")


def test_prediction_to_dict_serializable():
    print("=== test_prediction_to_dict_serializable ===")
    for sentence in ["おはようございます", "これは何ですか？", "資料を作成してください"]:
        pred = detect_intent(sentence)
        data = pred.to_dict()
        raw = json.dumps(data)
        print(f"{sentence!r} -> {data['intent']} ({data['confidence']})")
        assert raw
        assert data["intent"] in INTENT_LABELS
    print("PASSED: IntentPrediction.to_dict is JSON-serializable")


def test_intent_classifier_info():
    print("=== test_intent_classifier_info ===")
    info = intent_classifier_info()
    print("classes:", info.get("classes"))
    assert isinstance(info, dict)
    assert "classes" in info
    assert isinstance(info["classes"], list) and len(info["classes"]) > 0
    assert set(INTENT_LABELS) == set(info["classes"]), "classifier classes should match INTENT_LABELS"
    print("PASSED: intent_classifier_info returns dict with classes")


def test_evaluate_intent_classifier():
    print("=== test_evaluate_intent_classifier ===")
    report = evaluate_intent_classifier()
    print("accuracy:", report.get("accuracy"), "hybrid:", report.get("hybrid_accuracy"))
    assert "error" not in report, f"evaluation error: {report.get('error')}"
    assert 0.0 <= report["accuracy"] <= 1.0
    assert 0.0 <= report["hybrid_accuracy"] <= 1.0, "hybrid accuracy out of range"
    assert report["hybrid_accuracy"] >= 0.8, \
        f"hybrid pipeline accuracy dropped below expectation: {report['hybrid_accuracy']}"
    assert isinstance(report["per_class_accuracy"], dict)
    print("PASSED: evaluate_intent_classifier reports valid metrics")


if __name__ == "__main__":
    test_greeting_intent()
    print()
    test_question_intent()
    print()
    test_prediction_to_dict_serializable()
    print()
    test_intent_classifier_info()
    print()
    test_evaluate_intent_classifier()
    print()
    print("ALL TESTS PASSED")
