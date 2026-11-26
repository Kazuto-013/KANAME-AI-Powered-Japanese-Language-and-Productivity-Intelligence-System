from __future__ import annotations

from fastapi.testclient import TestClient

from ai_engine.voice_intelligence.commands import parse_voice_command
from backend.main import app


client = TestClient(app)


def test_voice_command_parser_maps_action_without_executing():
    result = parse_voice_command("この文を英語に翻訳してください")
    assert result["action"] == "translate"
    assert result["entities"]["target_language"] == "en"
    assert result["safe_to_execute"] is False


def test_voice_command_api_rejects_empty_text_and_returns_suggestion():
    bad = client.post("/api/voice/commands/parse", json={"text": " "})
    assert bad.status_code == 422

    response = client.post(
        "/api/voice/commands/parse",
        json={"text": "この文章を要約してください"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["action"] == "summarize"
    assert response.json()["safe_to_execute"] is False
