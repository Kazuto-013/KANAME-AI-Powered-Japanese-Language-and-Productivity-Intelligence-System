"""Optional offline TTS adapter backed by the host's installed speech engine."""
from __future__ import annotations

import os
import tempfile


def synthesize_to_file(text: str, language: str = "ja") -> str:
    if language not in ("ja", "en"):
        raise ValueError("language must be ja or en")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    try:
        import pyttsx3
    except ImportError as exc:
        raise RuntimeError(
            "Optional pyttsx3 is not installed. Install requirements-voice-optional.txt "
            "and ensure a system speech engine is available."
        ) from exc

    output = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    output_path = output.name
    output.close()
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices") or []
        language_markers = ("ja", "japanese", "日本語") if language == "ja" else ("en", "english")
        matching = []
        for voice in voices:
            description = f"{getattr(voice, 'id', '')} {getattr(voice, 'name', '')} {getattr(voice, 'languages', '')}".lower()
            if any(marker in description for marker in language_markers):
                matching.append(voice)
        if matching:
            engine.setProperty("voice", matching[0].id)
        engine.save_to_file(text.strip(), output_path)
        engine.runAndWait()
        engine.stop()
        if not os.path.exists(output_path) or os.path.getsize(output_path) <= 44:
            raise RuntimeError("Host speech engine did not produce valid audio.")
        return output_path
    except Exception:
        try:
            os.unlink(output_path)
        except OSError:
            pass
        raise
