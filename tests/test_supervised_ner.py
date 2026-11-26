from __future__ import annotations

from fastapi.testclient import TestClient

from ai_engine.japanese_nlp import supervised_ner
from backend.main import app


client = TestClient(app)


def _training_examples():
    rows = [
        ("田中は東京へ行く。", "田中", "東京"),
        ("佐藤は大阪へ戻る。", "佐藤", "大阪"),
        ("田中は大阪へ戻る。", "田中", "大阪"),
        ("佐藤は東京へ行く。", "佐藤", "東京"),
    ]
    return [
        {
            "text": text,
            "entities": [
                {"start": 0, "end": len(person), "label": "PERSON"},
                {
                    "start": text.index(location),
                    "end": text.index(location) + len(location),
                    "label": "LOCATION",
                },
            ],
        }
        for text, person, location in rows
    ]


def test_supervised_ner_train_and_predict_api():
    trained = client.post(
        "/api/nlp/entities/train",
        json={"examples": _training_examples(), "save": False},
    )
    assert trained.status_code == 200, trained.text
    info = trained.json()
    assert info["trained"] is True
    assert info["entity_labels"] == ["LOCATION", "PERSON"]
    assert "held-out" in info["evaluation_warning"]

    text = "田中は東京へ行く。"
    response = client.post(
        "/api/nlp/entities/predict",
        json={"text": text},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["entity_count"] == 2
    assert [(item["text"], item["label"]) for item in body["entities"]] == [
        ("田中", "PERSON"),
        ("東京", "LOCATION"),
    ]
    assert all(text[item["start"]:item["end"]] == item["text"] for item in body["entities"])


def test_supervised_ner_requires_token_aligned_annotations():
    invalid = client.post(
        "/api/nlp/entities/train",
        json={
            "save": False,
            "examples": [
                {"text": "田中は東京へ行く。", "entities": [{"start": 0, "end": 1, "label": "PERSON"}]},
                {"text": "佐藤は大阪へ戻る。", "entities": [{"start": 0, "end": 2, "label": "PERSON"}]},
            ],
        },
    )
    assert invalid.status_code == 400
    assert "token boundaries" in invalid.json()["detail"]


def test_supervised_ner_model_persists_and_reloads(tmp_path, monkeypatch):
    model_path = tmp_path / "supervised-ner.joblib"
    monkeypatch.setattr(supervised_ner, "_MODEL_PATH", str(model_path))
    monkeypatch.setattr(supervised_ner, "_STATE", None)

    trained = supervised_ner.train_ner(_training_examples(), save=True)
    assert trained["trained"] is True
    assert model_path.exists()

    monkeypatch.setattr(supervised_ner, "_STATE", None)
    info = supervised_ner.model_info()
    assert info["trained"] is True
    prediction = supervised_ner.predict_entities("田中は東京へ行く。")
    assert [item["label"] for item in prediction["entities"]] == ["PERSON", "LOCATION"]
