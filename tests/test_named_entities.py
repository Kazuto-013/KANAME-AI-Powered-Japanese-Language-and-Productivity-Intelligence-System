from __future__ import annotations

from fastapi.testclient import TestClient

from ai_engine.japanese_nlp import extract_named_entities
from backend.main import app


client = TestClient(app)


def test_extract_named_entities_uses_dictionary_labels_and_exact_spans():
    text = "田中太郎は東京大学で働いています。"
    result = extract_named_entities(text)

    assert result.method.startswith("SudachiPy dictionary POS tags")
    assert [(entity.text, entity.label) for entity in result.entities] == [
        ("田中太郎", "PERSON"),
        ("東京大学", "PROPER_NOUN"),
    ]
    for entity in result.entities:
        assert text[entity.start:entity.end] == entity.text
        assert entity.source == "sudachi_dictionary_pos"


def test_extract_named_entities_recognizes_location_and_no_entities():
    result = extract_named_entities("愛知県にあります。")
    assert [(entity.text, entity.label) for entity in result.entities] == [
        ("愛知県", "LOCATION"),
    ]

    empty = extract_named_entities("私は毎朝公園を散歩します。")
    assert empty.entities == []
    assert empty.to_dict()["entity_count"] == 0


def test_named_entities_api_response():
    response = client.post(
        "/api/nlp/entities",
        json={"text": "田中太郎は東京大学で働いています。", "mode": "C"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["entity_count"] == 2
    assert body["entities"][0]["text"] == "田中太郎"
    assert body["entities"][0]["label"] == "PERSON"
    assert body["entities"][1]["text"] == "東京大学"
    assert body["entities"][1]["label"] == "PROPER_NOUN"
