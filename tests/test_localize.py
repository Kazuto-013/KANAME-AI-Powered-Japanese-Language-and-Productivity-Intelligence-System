import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.document_schemas import ResumeRequest
from document_engine import localize as doc_localize

client = TestClient(app)


@pytest.fixture(autouse=True)
def _force_dictionary_fallback(monkeypatch):
    """Keep localization tests deterministic and offline by disabling every LLM engine."""
    for var in ("ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(var, "")


_JP_RE = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]")

EN_RESUME = {
    "full_name_kanji": "Taro Yamada",
    "full_name_kana": "やまだ たろう",
    "gender": "male",
    "date_of_birth": "1990-05-20",
    "phone": "03-1234-5678",
    "email": "taro.yamada@example.com",
    "postal_code": "100-0001",
    "self_pr": "I am a backend engineer with five years of Python experience.",
    "motivation": "I want to join your company to grow my career.",
    "hobbies": "I enjoy reading books and hiking in the mountains.",
    "education_list": [
        {"school_name": "University of Tokyo", "department": "Computer Science", "degree": "Bachelor"},
    ],
    "doc_format": "docx",
}


def test_localize_translates_english_free_text():
    print("=== test_localize_translates_english_free_text ===")
    validated = ResumeRequest.model_validate(EN_RESUME)
    localized, report = doc_localize.localize_document_data(
        ResumeRequest, validated.model_dump(), target_lang="ja"
    )
    print("report:", report)
    assert report["target_lang"] == "ja"
    assert report["translated_count"] >= 1, "at least one English field should be translated"
    # Descriptive English free-text becomes Japanese.
    assert _JP_RE.search(localized["self_pr"]), localized["self_pr"]
    assert localized["self_pr"] != EN_RESUME["self_pr"]
    assert _JP_RE.search(localized["motivation"]), localized["motivation"]
    print("PASSED: English free-text fields translated to Japanese")


def test_localize_preserves_identifiers_and_proper_nouns():
    print("=== test_localize_preserves_identifiers_and_proper_nouns ===")
    validated = ResumeRequest.model_validate(EN_RESUME)
    localized, _ = doc_localize.localize_document_data(
        ResumeRequest, validated.model_dump(), target_lang="ja"
    )
    # Identifiers / enums / contacts / dates must be untouched.
    assert localized["doc_format"] == "docx"
    assert localized["gender"] == "male"
    assert localized["phone"] == "03-1234-5678"
    assert localized["email"] == "taro.yamada@example.com"
    assert localized["date_of_birth"] == "1990-05-20"
    assert localized["postal_code"] == "100-0001"
    # Proper nouns (names) are preserved rather than garbled.
    assert localized["full_name_kanji"] == "Taro Yamada"
    assert localized["education_list"][0]["school_name"] == "University of Tokyo"
    print("PASSED: identifiers, enums, contacts and proper nouns preserved")


def test_localize_japanese_input_is_noop():
    print("=== test_localize_japanese_input_is_noop ===")
    ja_resume = dict(EN_RESUME)
    ja_resume["self_pr"] = "私はバックエンドエンジニアです。"
    ja_resume["motivation"] = "御社で成長したいと考えています。"
    ja_resume["hobbies"] = "読書と登山が趣味です。"
    ja_resume["education_list"] = [
        {"school_name": "東京大学", "department": "情報工学", "degree": "学士"},
    ]
    validated = ResumeRequest.model_validate(ja_resume)
    localized, report = doc_localize.localize_document_data(
        ResumeRequest, validated.model_dump(), target_lang="ja"
    )
    assert localized["self_pr"] == "私はバックエンドエンジニアです。"
    assert report["translated_count"] == 0
    print("PASSED: Japanese input left unchanged (translated_count == 0)")


def test_endpoint_auto_translates_english_input():
    print("=== test_endpoint_auto_translates_english_input ===")
    resp = client.post("/api/documents/generate/japanese-resume", json=EN_RESUME)
    print("status:", resp.status_code, "| body keys:", list(resp.json().keys()))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert not body.get("error"), body.get("error")
    report = body.get("auto_translate")
    assert isinstance(report, dict), "auto_translate report missing"
    assert report.get("translated_count", 0) >= 1
    print("PASSED: POST /generate/japanese-resume auto-translates English input")


def test_endpoint_can_disable_auto_translate():
    print("=== test_endpoint_can_disable_auto_translate ===")
    resp = client.post(
        "/api/documents/generate/japanese-resume?auto_translate=false",
        json=EN_RESUME,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert not body.get("error"), body.get("error")
    assert "auto_translate" not in body, "auto_translate report should be absent when disabled"
    print("PASSED: auto_translate=false skips translation (no report in response)")


def test_download_endpoint_reports_translation_count():
    print("=== test_download_endpoint_reports_translation_count ===")
    resp = client.post("/api/documents/generate/japanese-resume/download", json=EN_RESUME)
    print("status:", resp.status_code, "| X-Auto-Translate-Count:", resp.headers.get("X-Auto-Translate-Count"))
    assert resp.status_code == 200, resp.text
    assert resp.headers.get("Content-Type"), "expected a binary download content type"
    count = resp.headers.get("X-Auto-Translate-Count")
    assert count is not None and int(count) >= 1
    print("PASSED: download endpoint localizes and reports translated field count")
