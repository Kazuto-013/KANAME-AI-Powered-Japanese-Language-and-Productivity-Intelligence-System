from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import os
import tempfile
import numpy as np

_AVAILABLE = False
_WHISPER_ERROR: Optional[str] = None

try:
    from faster_whisper import WhisperModel
    _FASTER_WHISPER_AVAILABLE = True
except Exception as e:
    _FASTER_WHISPER_AVAILABLE = False
    _WHISPER_ERROR = f"faster-whisper import failed: {e}"

try:
    import av  # noqa: F401
    _AV_AVAILABLE = True
except Exception:
    _AV_AVAILABLE = False

_MODEL_CACHE: Dict[str, Any] = {}


def is_available() -> bool:
    return _FASTER_WHISPER_AVAILABLE and _AV_AVAILABLE


def get_error_message() -> Optional[str]:
    if not _FASTER_WHISPER_AVAILABLE:
        return _WHISPER_ERROR
    if not _AV_AVAILABLE:
        return "PyAV (av) package not installed — required for audio decoding."
    return None


VALID_MODELS = ("tiny", "base", "small", "medium", "large-v3")
VALID_LANGUAGES = ("ja", "en", "auto")
VALID_COMPUTE = ("int8", "float16", "float32", "int8_float16")


def _resolve_device_and_compute(
    device: Optional[str],
    compute_type: Optional[str],
) -> tuple[str, str]:
    if device is None:
        device = "cpu"
    if compute_type is None:
        compute_type = "int8" if device == "cpu" else "float16"
    return device, compute_type


def load_model(
    model_size: str = "base",
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    download_root: Optional[str] = None,
):
    if not _FASTER_WHISPER_AVAILABLE:
        raise RuntimeError(
            "faster-whisper is not installed. "
            f"Install it with: pip install faster-whisper. Error: {_WHISPER_ERROR}"
        )

    if model_size not in VALID_MODELS:
        raise ValueError(
            f"Invalid model_size '{model_size}'. Must be one of {VALID_MODELS}"
        )

    cache_key = f"{model_size}::{device}::{compute_type}"
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    resolved_device, resolved_compute = _resolve_device_and_compute(device, compute_type)
    root = download_root or os.environ.get(
        "WHISPER_MODEL_DIR",
        str(Path.home() / ".cache" / "faster-whisper"),
    )
    os.makedirs(root, exist_ok=True)

    model = WhisperModel(
        model_size,
        device=resolved_device,
        compute_type=resolved_compute,
        download_root=root,
    )
    _MODEL_CACHE[cache_key] = model
    return model


@dataclass
class SegmentInfo:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int] = field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration: float
    model_size: str
    segments: List[SegmentInfo] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "model_size": self.model_size,
            "segments": [s.to_dict() for s in self.segments],
            "warnings": self.warnings,
        }


def _extract_audio_path(
    audio_path: Optional[str] = None,
    audio_numpy: Optional[np.ndarray] = None,
    sample_rate: int = 16000,
) -> str:
    if audio_path is not None:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        return audio_path

    if audio_numpy is not None:
        if not isinstance(audio_numpy, np.ndarray):
            raise ValueError("audio_numpy must be a numpy array (float32 PCM).")

        arr = np.asarray(audio_numpy)
        if arr.dtype.kind == "i":
            max_val = float(np.iinfo(arr.dtype).max)
            arr = arr.astype(np.float32) / max_val
        else:
            arr = arr.astype(np.float32)

        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            import wave
            with wave.open(tmp.name, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
                wf.writeframes(pcm)
        except Exception:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass
            raise
        return tmp.name

    raise ValueError("Either audio_path or audio_numpy must be provided.")


def transcribe(
    audio_path: Optional[str] = None,
    audio_numpy: Optional[np.ndarray] = None,
    model_size: str = "base",
    language: str = "auto",
    beam_size: int = 5,
    temperature: float = 0.0,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
    vad_filter: bool = True,
    initial_prompt: Optional[str] = None,
    sample_rate: int = 16000,
) -> TranscriptionResult:
    if not is_available():
        raise RuntimeError(
            "Speech-to-Text engine is unavailable. "
            f"Reason: {get_error_message()}"
        )

    if language not in VALID_LANGUAGES:
        raise ValueError(
            f"Invalid language '{language}'. Must be one of {VALID_LANGUAGES}"
        )

    path = _extract_audio_path(audio_path, audio_numpy, sample_rate=sample_rate)
    temp_created = audio_path is None and audio_numpy is not None

    warnings: List[str] = []
    try:
        model = load_model(model_size, device=device, compute_type=compute_type)

        opts: Dict[str, Any] = {
            "beam_size": beam_size,
            "temperature": temperature,
            "vad_filter": vad_filter,
        }
        if language != "auto":
            opts["language"] = language
        if initial_prompt:
            opts["initial_prompt"] = initial_prompt

        segments: List[SegmentInfo] = []
        total_duration = 0.0
        text_parts: List[str] = []
        vad_removed_all = False
        info = None

        try:
            segments_iter, info = model.transcribe(path, **opts)
            for idx, seg in enumerate(segments_iter):
                seg_info = SegmentInfo(
                    id=getattr(seg, "id", idx),
                    seek=getattr(seg, "seek", 0),
                    start=float(getattr(seg, "start", 0.0)),
                    end=float(getattr(seg, "end", 0.0)),
                    text=getattr(seg, "text", "").strip(),
                    tokens=list(getattr(seg, "tokens", []) or []),
                    temperature=float(getattr(seg, "temperature", temperature)),
                    avg_logprob=float(getattr(seg, "avg_logprob", 0.0)),
                    compression_ratio=float(getattr(seg, "compression_ratio", 0.0)),
                    no_speech_prob=float(getattr(seg, "no_speech_prob", 0.0)),
                )
                segments.append(seg_info)
                text_parts.append(seg_info.text)
                if seg_info.end > total_duration:
                    total_duration = seg_info.end
        except ValueError as e:
            # faster-whisper raises this when the VAD filter strips every frame
            # (silent / non-speech audio). Treat it as a valid empty result.
            if "max() arg is an empty sequence" in str(e):
                vad_removed_all = True
            else:
                raise

        detected_language = getattr(info, "language", None) if info is not None else None
        language_prob = float(getattr(info, "language_probability", 0.0)) if info is not None else 0.0
        duration_attr = float(getattr(info, "duration", total_duration) or 0.0) if info is not None else total_duration
        if duration_attr <= 0:
            duration_attr = total_duration

        full_text = " ".join(p for p in text_parts if p).strip()
        if vad_removed_all:
            warnings.append(
                "No speech detected — the VAD filter removed all audio "
                "(silent or non-speech input)."
            )
        elif not full_text and not segments:
            warnings.append(
                "No speech detected in the audio input. "
                "Try a longer clip or increase microphone gain."
            )

        return TranscriptionResult(
            text=full_text,
            language=detected_language or (language if language != "auto" else "unknown"),
            language_probability=language_prob,
            duration=duration_attr,
            model_size=model_size,
            segments=segments,
            warnings=warnings,
        )
    finally:
        if temp_created and path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception:
                pass


# ============================================================
# VOICE MODULE ENHANCEMENTS (Phase 5)
# ============================================================


@dataclass
class KeywordSpot:
    keyword: str
    count: int
    segments: List[int] = field(default_factory=list)  # segment ids where it appeared
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KeywordSpotResult:
    query_keywords: List[str]
    spots: List[KeywordSpot] = field(default_factory=list)
    total_matches: int = 0
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_keywords": list(self.query_keywords),
            "spots": [s.to_dict() for s in self.spots],
            "total_matches": int(self.total_matches),
            "text": self.text,
        }


def spot_keywords(result: TranscriptionResult, keywords: List[str]) -> KeywordSpotResult:
    """Count occurrences of user-supplied keywords in the transcription.

    Case-insensitive substring match; also searches per-segment so caller
    can reveal WHERE in the audio each keyword appeared.
    """
    text = result.text or ""
    lowered = text.lower()
    spots: List[KeywordSpot] = []
    total = 0
    for kw in keywords or []:
        if not kw or not str(kw).strip():
            continue
        k = str(kw).strip()
        kl = k.lower()
        count = lowered.count(kl)
        seg_ids: List[int] = []
        for seg in result.segments:
            t = (seg.text or "").lower()
            if kl in t:
                seg_ids.append(int(seg.id))
        score = 0.0
        if count > 0:
            # simple score: count × (1 + segments-hit bonus)
            score = count * (1.0 + 0.1 * max(0, len(seg_ids) - 1))
        spots.append(KeywordSpot(keyword=k, count=count, segments=seg_ids, score=round(score, 3)))
        total += count
    spots.sort(key=lambda s: (-s.count, -s.score, s.keyword))
    return KeywordSpotResult(query_keywords=list(keywords or []), spots=spots, total_matches=total, text=text)


# ------------- Summary & Key Phrases -------------

_PROFANITY_WORDS_JP_EN: List[str] = [
    "くそ", "くそっ", "くそったれ", "ちくしょう", "畜生", "ばか", "バカ", "あほ", "アホ",
    "まぬけ", "ごみ", "カス", "死ね", "しね", "くたばれ", "うざい", "うざっ",
    "キモい", "きもい", "キモ", "ブサイク", "不細工", "ゴミカス", "雑魚", "ざこ",
    "クソ", "クソッ", "クソ野郎", "カス野郎", "負け犬", "まけいぬ",
    "fuck", "shit", "bitch", "bastard", "asshole", "motherfucker", "cunt",
    "dick", "pussy", "cock", "nigga", "nigger", "whore", "slut", "douche",
]


@dataclass
class SummaryAndPhrases:
    summary_sentence: str
    key_phrases: List[Tuple[str, float]]  # (phrase, score)
    technique: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary_sentence": self.summary_sentence,
            "key_phrases": [{"phrase": p, "score": round(float(s), 4)}
                            for p, s in self.key_phrases],
            "technique": self.technique,
        }


def summarize_transcription(result: TranscriptionResult, max_phrases: int = 10) -> SummaryAndPhrases:
    """Extract key phrases and a one-line summary from a transcription.

    Uses SudachiPy noun/SAHEN frequency when available; else falls back to
    word-level frequency on whitespace or simple segment-bigrams.

    Summary sentence = concatenate TOP-3 segments by avg_logprob (most confident).
    """
    text = (result.text or "").strip()
    segments = list(result.segments or [])
    # Summary: top segments by confidence (higher avg_logprob).
    if segments:
        scored = sorted(
            [s for s in segments if (s.text or "").strip()],
            key=lambda s: (s.avg_logprob, -s.no_speech_prob),
            reverse=True,
        )
        top = scored[:3]
        summary = " ".join((s.text or "").strip() for s in top).strip()
    else:
        summary = text[:200]
    if not summary and text:
        summary = text[:200]

    # Key phrases: try SudachiPy first
    phrases: List[Tuple[str, float]] = []
    technique: str = "frequency (no NLP)"
    if text:
        # Try Sudachi noun/SAHEN compound extraction
        try:
            from ai_engine.japanese_nlp import sudachi_engine  # noqa: WPS433
            ana = sudachi_engine.analyze(text, mode="C")
            freq: Dict[str, int] = {}
            for tok in ana.tokens:
                pm = tok.pos_main or ""
                surf = (tok.normalized_form or tok.dictionary_form or tok.surface or "").strip()
                if not surf or len(surf) < 2:
                    continue
                if pm in ("名詞",) or "サ変" in " ".join(tok.pos or []):
                    # skip pure number / symbol nouns
                    if surf.isdigit():
                        continue
                    freq[surf] = freq.get(surf, 0) + 1
            for p, c in sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0])))[: max(1, max_phrases)]:
                if c >= 1:
                    length_boost = 0.1 * min(5, len(p))
                    phrases.append((p, round(float(c) + length_boost, 3)))
            technique = "Sudachi noun+SAHEN frequency"
        except Exception:
            # Fallback: whitespace split (EN) + length weighting
            words = re.findall(r"[A-Za-z]{3,}|[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF]{2,}", text)
            freq = {}
            for w in words:
                freq[w] = freq.get(w, 0) + 1
            for w, c in sorted(freq.items(), key=lambda kv: -kv[1])[:max_phrases]:
                if c >= 1:
                    phrases.append((w, float(c)))
    # De-duplicate & cap
    seen = set()
    capped: List[Tuple[str, float]] = []
    for p, s in phrases:
        if p in seen:
            continue
        seen.add(p)
        capped.append((p, s))
        if len(capped) >= max_phrases:
            break
    return SummaryAndPhrases(
        summary_sentence=summary,
        key_phrases=capped,
        technique=technique,
    )


# ------------- SRT / VTT subtitle export -------------

def _fmt_srt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{m:02d}:{s:02d}.{ms:03d}"


def export_subtitles(result: TranscriptionResult, fmt: str = "srt") -> str:
    """Render SRT or WebVTT subtitle text from transcription segments."""
    fmt = (fmt or "srt").lower().strip()
    segments = list(result.segments or [])
    lines: List[str] = []
    if not segments:
        # Degenerate: put full text into a single 0..duration segment
        d = float(getattr(result, "duration", 0) or 0.0)
        segments = [SegmentInfo(
            id=1, seek=0, start=0.0, end=max(d, 1.0),
            text=result.text or "", tokens=[], temperature=0.0,
            avg_logprob=0.0, compression_ratio=0.0, no_speech_prob=0.0,
        )]
    if fmt == "vtt":
        lines.append("WEBVTT")
        lines.append("")
        for i, seg in enumerate(segments, start=1):
            start_t = _fmt_vtt_ts(float(seg.start))
            end_t = _fmt_vtt_ts(float(seg.end))
            lines.append(f"{start_t} --> {end_t}")
            lines.append((seg.text or "").strip())
            lines.append("")
    else:
        # Default SRT
        for i, seg in enumerate(segments, start=1):
            lines.append(str(i))
            lines.append(f"{_fmt_srt_ts(float(seg.start))} --> {_fmt_srt_ts(float(seg.end))}")
            lines.append((seg.text or "").strip())
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ------------- Profanity / sensitive-word filter -------------

@dataclass
class FilteredText:
    original_text: str
    filtered_text: str
    masked_words: List[str] = field(default_factory=list)
    mask_count: int = 0
    mask_token: str = "***"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def filter_profanity(
    result_or_text: Any,
    extra_words: Optional[List[str]] = None,
    mask_token: str = "***",
) -> FilteredText:
    """Mask profanity/sensitive terms in transcription text or a plain string.

    Accepts either a TranscriptionResult (reads result.text) or a plain str.
    """
    if isinstance(result_or_text, TranscriptionResult):
        original = result_or_text.text or ""
    elif isinstance(result_or_text, str):
        original = result_or_text
    else:
        original = str(result_or_text or "")
    blocklist = set(w for w in _PROFANITY_WORDS_JP_EN if w)
    for w in extra_words or []:
        if w and str(w).strip():
            blocklist.add(str(w).strip())
    filtered = original
    masked: List[str] = []
    count = 0
    # Longer matches first to avoid word-within-word issues
    sorted_words = sorted(blocklist, key=len, reverse=True)
    for w in sorted_words:
        pattern = re.compile(re.escape(w), re.IGNORECASE)
        matches = pattern.findall(filtered)
        if matches:
            count += len(matches)
            for m in matches:
                if m not in masked:
                    masked.append(m)
            filtered = pattern.sub(mask_token * max(1, (len(w) // 3)) if False else mask_token, filtered)
    return FilteredText(
        original_text=original,
        filtered_text=filtered,
        masked_words=masked,
        mask_count=count,
        mask_token=mask_token,
    )


# ------------- Audio metadata (duration, format, channels) -------------

@dataclass
class AudioInfo:
    filename: str = ""
    exists: bool = False
    duration_seconds: float = 0.0
    format_name: str = ""
    codec_name: str = ""
    sample_rate: int = 0
    channels: int = 0
    bit_rate: int = 0
    size_bytes: int = 0
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_audio_info(audio_path: str) -> AudioInfo:
    """Return basic audio metadata using PyAV. Safe if av is not installed."""
    info = AudioInfo(filename=os.path.basename(audio_path or ""))
    if not audio_path or not os.path.exists(audio_path):
        info.error = "audio path not found"
        return info
    info.exists = True
    try:
        info.size_bytes = int(os.path.getsize(audio_path))
    except Exception as e:
        info.error = f"stat failed: {e}"
    if not _AV_AVAILABLE:
        info.error = info.error or "PyAV not installed — av package required for full metadata"
        return info
    try:
        import av  # type: ignore
        with av.open(audio_path) as container:
            info.format_name = str(getattr(container, "format", "") or "")
            info.bit_rate = int(getattr(container, "bit_rate", 0) or 0)
            if container.duration:
                info.duration_seconds = float(container.duration) / 1_000_000.0
            try:
                stream = next(s for s in container.streams if s.type == "audio")
            except StopIteration:
                stream = None
            if stream is not None:
                codec = getattr(stream, "codec_context", None)
                if codec is not None:
                    info.codec_name = str(getattr(codec, "name", "") or "")
                    info.sample_rate = int(getattr(codec, "sample_rate", 0) or 0)
                    info.channels = int(getattr(codec, "channels", 0) or 0)
                if info.duration_seconds <= 0 and stream.duration and stream.time_base:
                    info.duration_seconds = float(stream.duration * float(stream.time_base))
    except Exception as e:
        info.error = f"av probe failed: {type(e).__name__}: {e}"
    return info


# ------------- Speaker-turn detection (silence-gap) -------------

@dataclass
class SpeakerTurn:
    turn_index: int
    start: float
    end: float
    text: str
    speaker_label: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def detect_speaker_turns(
    result: TranscriptionResult,
    silence_threshold_seconds: float = 0.8,
    max_speakers_for_label: int = 2,
) -> List[SpeakerTurn]:
    """Segment turns based on silence gaps between STT segments.

    A new 'turn' starts whenever the gap between the previous segment end and
    the next segment start exceeds silence_threshold_seconds. Speakers simply
    alternate between SPEAKER_A / SPEAKER_B / ... by turn modulo N. This is
    NOT real diarization — it is only a useful approximation for structured
    conversations (e.g. meeting alternation with clear pauses).
    """
    segments = sorted(
        [s for s in (result.segments or []) if (s.text or "").strip()],
        key=lambda s: float(s.start),
    )
    if not segments:
        return []
    threshold = max(0.1, float(silence_threshold_seconds))
    turns: List[SpeakerTurn] = []
    current_texts: List[str] = [segments[0].text or ""]
    current_start = float(segments[0].start)
    current_end = float(segments[0].end)
    turn_idx = 0
    for i in range(1, len(segments)):
        seg = segments[i]
        gap = float(seg.start) - current_end
        if gap >= threshold:
            spk_idx = turn_idx % max(1, int(max_speakers_for_label))
            label = f"SPEAKER_{chr(65 + spk_idx)}"
            turns.append(SpeakerTurn(
                turn_index=turn_idx,
                start=current_start,
                end=current_end,
                text=" ".join(t.strip() for t in current_texts if t).strip(),
                speaker_label=label,
            ))
            turn_idx += 1
            current_texts = [seg.text or ""]
            current_start = float(seg.start)
            current_end = float(seg.end)
        else:
            current_texts.append(seg.text or "")
            current_end = max(current_end, float(seg.end))
    # Flush last turn
    if current_texts:
        spk_idx = turn_idx % max(1, int(max_speakers_for_label))
        label = f"SPEAKER_{chr(65 + spk_idx)}"
        turns.append(SpeakerTurn(
            turn_index=turn_idx,
            start=current_start,
            end=current_end,
            text=" ".join(t.strip() for t in current_texts if t).strip(),
            speaker_label=label,
        ))
    return turns


# Convenience one-stop post-processing helper for the API layer
@dataclass
class TranscriptionPostProcessed:
    transcription: TranscriptionResult
    keyword_spot: Optional[KeywordSpotResult] = None
    summary: Optional[SummaryAndPhrases] = None
    subtitles_srt: Optional[str] = None
    subtitles_vtt: Optional[str] = None
    profanity_filtered: Optional[FilteredText] = None
    audio_info: Optional[AudioInfo] = None
    speaker_turns: List[SpeakerTurn] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transcription": self.transcription.to_dict(),
            "keyword_spot": self.keyword_spot.to_dict() if self.keyword_spot else None,
            "summary_and_phrases": self.summary.to_dict() if self.summary else None,
            "subtitles_srt": self.subtitles_srt,
            "subtitles_vtt": self.subtitles_vtt,
            "profanity_filtered": self.profanity_filtered.to_dict() if self.profanity_filtered else None,
            "audio_info": self.audio_info.to_dict() if self.audio_info else None,
            "speaker_turns": [t.to_dict() for t in self.speaker_turns],
        }


def post_process_transcription(
    transcription: TranscriptionResult,
    audio_path: Optional[str] = None,
    keywords: Optional[List[str]] = None,
    enable_summary: bool = True,
    enable_subtitles: bool = True,
    enable_profanity_filter: bool = False,
    extra_profanity_words: Optional[List[str]] = None,
    enable_audio_info: bool = True,
    enable_speaker_turns: bool = True,
    silence_threshold_seconds: float = 0.8,
) -> TranscriptionPostProcessed:
    """Run all post-processing in one call. Use from API routers."""
    kw = spot_keywords(transcription, list(keywords or [])) if keywords else None
    summ = summarize_transcription(transcription) if enable_summary else None
    srt_txt = export_subtitles(transcription, fmt="srt") if enable_subtitles else None
    vtt_txt = export_subtitles(transcription, fmt="vtt") if enable_subtitles else None
    prof = filter_profanity(transcription, extra_words=list(extra_profanity_words or [])) if enable_profanity_filter else None
    ainfo = get_audio_info(audio_path) if (enable_audio_info and audio_path) else None
    turns = detect_speaker_turns(transcription, silence_threshold_seconds=silence_threshold_seconds) if enable_speaker_turns else []
    return TranscriptionPostProcessed(
        transcription=transcription,
        keyword_spot=kw,
        summary=summ,
        subtitles_srt=srt_txt,
        subtitles_vtt=vtt_txt,
        profanity_filtered=prof,
        audio_info=ainfo,
        speaker_turns=turns,
    )


import re  # noqa: E402  (keep local — import needed by filter regexes)
