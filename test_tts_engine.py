import sys
sys.path.insert(0, ".")
from ai_engine.voice_intelligence.tts_engine import synthesize_to_file

# Test English
en_path = synthesize_to_file("hello my name is aditya", "en")
print(f"EN TTS OK: {en_path}, size={__import__('os').path.getsize(en_path)} bytes")

# Test Japanese
ja_path = synthesize_to_file("こんにちは、私は田中です", "ja")
print(f"JA TTS OK: {ja_path}, size={__import__('os').path.getsize(ja_path)} bytes")
