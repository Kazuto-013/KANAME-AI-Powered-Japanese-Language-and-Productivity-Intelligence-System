from __future__ import annotations

from fastapi.testclient import TestClient

from ai_engine.translation.translator import translate_text
from backend.main import app


client = TestClient(app)


def test_custom_glossary_term_is_preserved_through_fallback():
    result = translate_text(
        "KANAME is useful.",
        source_lang="en",
        target_lang="ja",
        force_fallback=True,
        custom_glossary=[{"ja": "要", "en": "KANAME"}],
    )
    assert "要" in result.translated_text
    assert result.glossary_hits == 1
    assert result.notes["custom_glossary_hits"] == 1


def test_translation_api_accepts_request_scoped_examples_and_glossary():
    response = client.post(
        "/api/translation/translate?force_fallback=true",
        json={
            "text": "KANAME is useful.",
            "source_lang": "en",
            "target_lang": "ja",
            "custom_glossary": [{"ja": "要", "en": "KANAME"}],
            "examples": [{"source": "sample", "target": "例"}],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert "要" in body["translated_text"]
    assert body["notes"]["custom_glossary_hits"] == 1
    assert body["notes"]["few_shot_examples_used"] == 0
