import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

GET_ENDPOINTS = [
    "/health",
    "/",
    "/api/nlp/",
    "/api/voice/",
    "/api/data/",
    "/api/translation/",
    "/api/documents/",
    "/api/ml/",
]


def test_get_endpoints_return_200():
    print("=== test_get_endpoints_return_200 ===")
    for path in GET_ENDPOINTS:
        resp = client.get(path)
        print(f"GET {path} -> {resp.status_code}")
        assert resp.status_code == 200, f"GET {path} returned {resp.status_code}"
    print("PASSED: all root/health endpoints return 200")


def test_translate_endpoint_force_fallback():
    print("=== test_translate_endpoint_force_fallback ===")
    resp = client.post(
        "/api/translation/translate?force_fallback=true",
        json={
            "text": "私は毎朝公園を散歩します。",
            "source_lang": "ja",
            "target_lang": "en",
        },
    )
    print("status:", resp.status_code)
    print("body engine:", resp.json().get("engine"))
    assert resp.status_code == 200, f"expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body["engine"] == "dictionary"
    assert body["translated_text"].strip()
    print("PASSED: POST /api/translation/translate returns 200 with dictionary engine")


if __name__ == "__main__":
    test_get_endpoints_return_200()
    print()
    test_translate_endpoint_force_fallback()
    print()
    print("ALL TESTS PASSED")
