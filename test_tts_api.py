import requests

for lang, text in [("en", "hello my name is aditya"), ("ja", "こんにちは、私は田中です")]:
    r = requests.post(
        "http://127.0.0.1:8000/api/voice/tts",
        json={"text": text, "language": lang},
        timeout=60,
    )
    print(f"{lang.upper()} TTS API: HTTP {r.status_code}, size={len(r.content)} bytes, content-type={r.headers.get('content-type')}")
