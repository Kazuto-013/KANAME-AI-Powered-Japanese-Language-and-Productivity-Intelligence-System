from __future__ import annotations

from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field

WhisperModelSize = Literal["tiny", "base", "small", "medium", "large-v3"]
WhisperLanguage = Literal["auto", "ja", "en"]
WhisperDevice = Literal["auto", "cpu", "cuda"]


class SegmentSchema(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class TranscribeResponse(BaseModel):
    text: str
    language: str
    language_probability: float
    duration: float
    model_size: str
    segments: List[SegmentSchema] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class VoiceHealthResponse(BaseModel):
    status: str
    engine: str
    engine_available: bool
    detail: Optional[str] = None
    default_model: str
    default_language: str
    default_device: str


class TranscribeInlineRequest(BaseModel):
    audio_path: str = Field(..., description="Local path on server to the audio file")
    model_size: WhisperModelSize = Field("base", description="Whisper model size")
    language: WhisperLanguage = Field("auto", description="Target language, 'auto' to detect")
    beam_size: int = Field(5, ge=1, le=10)
    temperature: float = Field(0.0, ge=0.0, le=1.0)
    device: WhisperDevice = Field("auto", description="Device: auto, cpu, cuda")
    compute_type: Optional[str] = Field(None, description="CTranslate2 compute type, e.g. int8, float16")
    vad_filter: bool = Field(True, description="Enable VAD filter before recognition")
    initial_prompt: Optional[str] = Field(None, description="Optional Japanese prompt to bias model")


class VoiceCommandRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    language: Literal["ja", "en"] = "ja"
