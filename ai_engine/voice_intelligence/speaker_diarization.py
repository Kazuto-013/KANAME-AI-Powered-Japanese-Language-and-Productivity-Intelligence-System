"""Optional pretrained speaker diarization adapter (pyannote.audio)."""
from __future__ import annotations

import os
from typing import Any


MODEL_ID = "pyannote/speaker-diarization-3.1"
_PIPELINE: Any = None
_PIPELINE_TOKEN: str | None = None


def diarize_audio(audio_path: str, token: str | None = None) -> list[dict[str, Any]]:
    access_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not access_token:
        raise RuntimeError("Set HF_TOKEN after accepting the pyannote model terms.")
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Optional pyannote.audio is not installed. Install requirements-voice-optional.txt."
        ) from exc

    global _PIPELINE, _PIPELINE_TOKEN
    if _PIPELINE is None or _PIPELINE_TOKEN != access_token:
        _PIPELINE = Pipeline.from_pretrained(MODEL_ID, use_auth_token=access_token)
        if _PIPELINE is None:
            raise RuntimeError(f"Could not load pretrained pipeline {MODEL_ID}.")
        _PIPELINE.to(torch.device("cpu"))
        _PIPELINE_TOKEN = access_token

    annotation = _PIPELINE(audio_path)
    turns = []
    for segment, _track, speaker in annotation.itertracks(yield_label=True):
        turns.append({
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "speaker": str(speaker),
        })
    return turns
