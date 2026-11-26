from __future__ import annotations

from fastapi.testclient import TestClient

from ai_engine.machine_learning.topic_modeling import discover_topics
from ai_engine.machine_learning import torch_transformer
from backend.main import app


client = TestClient(app)


def test_topic_modeling_returns_assignments_and_keywords():
    texts = [
        "機械学習モデルを学習して精度を評価する。",
        "深層学習と自然言語処理の研究を進める。",
        "売上データを集計して四半期の利益を確認する。",
        "請求書と契約書を確認して取引先へ連絡する。",
    ]
    result = discover_topics(texts, n_topics=2, top_words=4)
    assert result["n_topics"] == 2
    assert len(result["assignments"]) == len(texts)
    assert all(topic["keywords"] for topic in result["topics"])
    assert all(0.0 <= item["confidence"] <= 1.0 for item in result["assignments"])


def test_auto_label_api_marks_predictions_for_review():
    response = client.post(
        "/api/ml/auto-label",
        json={
            "texts": ["機械学習モデルを構築します。", "今日は天気が良いです。"],
            "confidence_threshold": 0.5,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["count"] == 2
    assert {item["status"] for item in body["items"]} <= {"suggested", "needs_review"}
    assert "not verified ground truth" in body["warning"]


def test_torch_transformer_trains_and_classifies():
    if not torch_transformer.is_available():
        return
    info = torch_transformer.train_model(epochs=2, save=False)
    prediction = torch_transformer.classify_text("深層学習モデルを学習する。", top_k=3)
    assert info["engine"] == "PyTorch TransformerEncoder (CPU)"
    assert prediction["predicted_label"] in info["classes"]
    assert len(prediction["top_k"]) == 3
    assert abs(sum(prediction["probabilities"].values()) - 1.0) < 1e-5
