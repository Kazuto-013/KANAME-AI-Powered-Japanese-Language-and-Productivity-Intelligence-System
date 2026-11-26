from __future__ import annotations

import os
import re
import tempfile
import uuid
from typing import List, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from ai_engine.voice_intelligence import stt_engine
from ai_engine.voice_intelligence.commands import parse_voice_command
from ai_engine.voice_intelligence import speaker_diarization, tts_engine
from ai_engine.japanese_nlp import sudachi_engine
from backend.database.config import get_db
from backend.services import voice_service as voice_svc
from backend.schemas.voice_schemas import (
    SegmentSchema,
    TtsRequest,
    TranscribeInlineRequest,
    TranscribeResponse,
    VoiceCommandRequest,
    VoiceHealthResponse,
)

router = APIRouter(prefix="/voice", tags=["Voice Intelligence"])


@router.get("/", status_code=200)
async def voice_root():
    return {
        "module": "Voice Intelligence",
        "status": "implemented",
        "engine": "faster-whisper (CTranslate2 Whisper)",
        "default_model": "base",
        "default_language": "auto",
        "endpoints": {
            "health": "GET /api/voice/health",
            "transcribe_upload": "POST /api/voice/transcribe (multipart/form-data: file + params)",
            "transcribe_inline": "POST /api/voice/transcribe-inline",
            "parse_command": "POST /api/voice/commands/parse",
            "tts": "POST /api/voice/tts (optional host speech engine)",
            "diarize": "POST /api/voice/diarize (optional pyannote model + HF_TOKEN)",
        },
        "description": (
            "Speech-to-Text using faster-whisper. Supports audio upload, on-disk path, "
            "multiple model sizes, language auto-detection with Japanese bias, VAD filtering."
        ),
    }


@router.get("/health", response_model=VoiceHealthResponse, tags=["Voice Intelligence"])
async def voice_health():
    available = stt_engine.is_available()
    return VoiceHealthResponse(
        status="ok" if available else "degraded",
        engine="faster-whisper (CTranslate2)",
        engine_available=available,
        detail=stt_engine.get_error_message(),
        default_model="base",
        default_language="auto",
        default_device="auto",
    )


def _save_uploaded_file(file: UploadFile) -> str:
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    if len(suffix) > 8:
        suffix = ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        size = 0
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
            size += len(chunk)
            if size > 256 * 1024 * 1024:
                tmp.close()
                try:
                    os.unlink(tmp.name)
                except Exception:
                    pass
                raise HTTPException(status_code=413, detail="Audio file too large (>256 MB).")
    finally:
        tmp.close()
    return tmp.name


def _device_str_to_none(device: str) -> Optional[str]:
    if device and device.lower() == "auto":
        return None
    return device if device else None


def _transcribe_core(
    audio_path: str,
    model_size: str,
    language: str,
    beam_size: int,
    temperature: float,
    device: str,
    compute_type: Optional[str],
    vad_filter: bool,
    initial_prompt: Optional[str],
) -> TranscribeResponse:
    if not stt_engine.is_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "Voice engine not available on this server. "
                f"Reason: {stt_engine.get_error_message()}"
            ),
        )
    try:
        result = stt_engine.transcribe(
            audio_path=audio_path,
            model_size=model_size,
            language=language,
            beam_size=beam_size,
            temperature=temperature,
            device=_device_str_to_none(device),
            compute_type=compute_type,
            vad_filter=vad_filter,
            initial_prompt=initial_prompt,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STT engine error: {e}")

    return TranscribeResponse(
        text=result.text,
        language=result.language,
        language_probability=result.language_probability,
        duration=result.duration,
        model_size=result.model_size,
        segments=[SegmentSchema(**s.to_dict()) for s in result.segments],
        warnings=result.warnings,
    )


@router.post("/commands/parse", tags=["Voice Intelligence"])
async def parse_voice_command_endpoint(req: VoiceCommandRequest):
    text = req.text
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=422, detail="text must be a non-empty string")
    if len(text) > 5000:
        raise HTTPException(status_code=422, detail="text exceeds 5000 characters")
    return parse_voice_command(text)


@router.post("/tts", tags=["Voice Intelligence"])
async def synthesize_speech(req: TtsRequest):
    text = req.text
    language = req.language
    if not isinstance(text, str) or not text.strip() or len(text) > 5000:
        raise HTTPException(status_code=422, detail="text must contain 1-5000 characters")
    try:
        path = await run_in_threadpool(tts_engine.synthesize_to_file, text, language)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Host TTS engine failed: {exc}")
    return FileResponse(
        path,
        media_type="audio/wav",
        filename="kaname_speech.wav",
        background=BackgroundTask(lambda: os.path.exists(path) and os.unlink(path)),
    )


@router.post("/diarize", tags=["Voice Intelligence"])
async def diarize_uploaded_audio(
    file: UploadFile = File(...),
    model_size: str = Form("base"),
    language: str = Form("auto"),
    vad_filter: bool = Form(True),
):
    """Run pretrained speaker diarization and align Whisper segments to speaker turns."""
    path = _save_uploaded_file(file)
    try:
        transcription = await run_in_threadpool(
            _transcribe_core,
            path,
            model_size,
            language,
            5,
            0.0,
            "auto",
            None,
            vad_filter,
            None,
        )
        try:
            turns = await run_in_threadpool(speaker_diarization.diarize_audio, path)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        aligned = []
        for segment in transcription.segments:
            best_turn = max(
                turns,
                key=lambda turn: max(
                    0.0,
                    min(segment.end, turn["end"]) - max(segment.start, turn["start"]),
                ),
                default=None,
            )
            overlap = (
                max(0.0, min(segment.end, best_turn["end"]) - max(segment.start, best_turn["start"]))
                if best_turn else 0.0
            )
            aligned.append({
                **segment.model_dump(),
                "speaker": best_turn["speaker"] if overlap > 0 else None,
            })
        return {
            "engine": speaker_diarization.MODEL_ID,
            "transcription": transcription.model_dump(),
            "speaker_turns": turns,
            "aligned_segments": aligned,
            "speaker_count": len({turn["speaker"] for turn in turns}),
        }
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@router.post("/transcribe", response_model=TranscribeResponse, tags=["Voice Intelligence"])
async def transcribe_upload(
    file: UploadFile = File(..., description="Audio file (wav/mp3/m4a/flac/webm etc.)"),
    model_size: str = Form("base"),
    language: str = Form("auto"),
    beam_size: int = Form(5, ge=1, le=10),
    temperature: float = Form(0.0, ge=0.0, le=1.0),
    device: str = Form("auto"),
    compute_type: Optional[str] = Form(None),
    vad_filter: bool = Form(True),
    initial_prompt: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    path = _save_uploaded_file(file)
    try:
        resp = _transcribe_core(
            audio_path=path,
            model_size=model_size,
            language=language,
            beam_size=beam_size,
            temperature=temperature,
            device=device,
            compute_type=compute_type,
            vad_filter=vad_filter,
            initial_prompt=initial_prompt,
        )
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass

    try:
        voice_svc.create_job(
            db=db,
            source_filename=file.filename,
            model_size=resp.model_size,
            detected_language=resp.language,
            language_probability=resp.language_probability,
            duration_seconds=resp.duration,
            segment_count=len(resp.segments),
            transcription_text=resp.text,
            warnings=" | ".join(resp.warnings) if resp.warnings else None,
        )
    except Exception:
        pass

    return resp


@router.post("/transcribe-inline", response_model=TranscribeResponse, tags=["Voice Intelligence"])
async def transcribe_inline(req: TranscribeInlineRequest, db: Session = Depends(get_db)):
    resp = _transcribe_core(
        audio_path=req.audio_path,
        model_size=req.model_size,
        language=req.language,
        beam_size=req.beam_size,
        temperature=req.temperature,
        device=req.device,
        compute_type=req.compute_type,
        vad_filter=req.vad_filter,
        initial_prompt=req.initial_prompt,
    )

    try:
        voice_svc.create_job(
            db=db,
            source_filename=os.path.basename(req.audio_path),
            model_size=resp.model_size,
            detected_language=resp.language,
            language_probability=resp.language_probability,
            duration_seconds=resp.duration,
            segment_count=len(resp.segments),
            transcription_text=resp.text,
            warnings=" | ".join(resp.warnings) if resp.warnings else None,
        )
    except Exception:
        pass

    return resp


@router.post("/transcribe-then-nlp", tags=["Voice Intelligence"])
async def transcribe_then_nlp(
    file: UploadFile = File(...),
    model_size: str = Form("base"),
    language: str = Form("auto"),
    beam_size: int = Form(5, ge=1, le=10),
    temperature: float = Form(0.0, ge=0.0, le=1.0),
    device: str = Form("auto"),
    compute_type: Optional[str] = Form(None),
    vad_filter: bool = Form(True),
    sudachi_mode: str = Form("C"),
):
    path = _save_uploaded_file(file)
    try:
        stt_resp = _transcribe_core(
            audio_path=path,
            model_size=model_size,
            language=language,
            beam_size=beam_size,
            temperature=temperature,
            device=device,
            compute_type=compute_type,
            vad_filter=vad_filter,
            initial_prompt=None,
        )
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass

    text = stt_resp.text or ""
    nlp = None
    if text.strip():
        nlp_result = sudachi_engine.analyze(text, mode=sudachi_mode or "C")
        nlp = nlp_result.to_dict()

    return {
        "stt": stt_resp.model_dump(),
        "nlp": nlp,
    }


@router.get("/jobs/{job_id}", tags=["Voice Intelligence"])
async def get_voice_job_by_id(job_id: str, db: Session = Depends(get_db)):
    try:
        item = voice_svc.get_job(db, job_id)
    except Exception as e:
        return {"job": None, "error": f"Database unavailable: {e}"}
    if item is None:
        return {"job": None, "error": "job_id not found"}
    return {"job": item}


@router.get("/jobs", tags=["Voice Intelligence"])
async def list_voice_jobs(limit: int = 50, db: Session = Depends(get_db)):
    try:
        jobs = voice_svc.list_jobs(db, limit=limit)
        return {
            "jobs": jobs,
            "count": len(jobs),
            "error": None,
        }
    except Exception as e:
        return {
            "jobs": [],
            "count": 0,
            "error": f"Database unavailable: {e}",
        }


@router.post("/post-process", tags=["Voice Intelligence"])
async def voice_post_process_all(
    file: UploadFile = File(..., description="Audio file (WAV/MP3/M4A/FLAC/WebM)"),
    model_size: str = Form("base"),
    language: str = Form("auto"),
    beam_size: int = Form(5, ge=1, le=10),
    temperature: float = Form(0.0, ge=0.0, le=1.0),
    device: str = Form("auto"),
    compute_type: Optional[str] = Form(None),
    vad_filter: bool = Form(True),
    keywords: Optional[str] = Form(None, description="Comma or newline separated keywords to spot."),
    enable_summary: bool = Form(True),
    enable_subtitles: bool = Form(True),
    enable_profanity_filter: bool = Form(False),
    extra_profanity: Optional[str] = Form(None, description="Comma-separated extra words to block."),
    enable_audio_info: bool = Form(True),
    enable_speaker_turns: bool = Form(True),
    silence_threshold_seconds: float = Form(0.8, ge=0.05, le=10.0),
    db: Session = Depends(get_db),
):
    """Full pipeline: STT → keyword-spot → summary/phrases → subtitles → filter → metadata → speaker-turns."""
    path = _save_uploaded_file(file)
    try:
        stt_resp = _transcribe_core(
            audio_path=path,
            model_size=model_size,
            language=language,
            beam_size=beam_size,
            temperature=temperature,
            device=device,
            compute_type=compute_type,
            vad_filter=vad_filter,
            initial_prompt=None,
        )
        # Parse keywords / extra profanity lists
        kw_list: List[str] = []
        if keywords:
            kw_list = [k.strip() for k in re.split(r"[\s,、\n]+", keywords) if k.strip()]
        extra_block: List[str] = []
        if extra_profanity:
            extra_block = [w.strip() for w in re.split(r"[\s,、\n]+", extra_profanity) if w.strip()]
        # Post-processing
        post = stt_engine.post_process_transcription(
            transcription=stt_resp,
            audio_path=path if enable_audio_info else None,
            keywords=kw_list or None,
            enable_summary=bool(enable_summary),
            enable_subtitles=bool(enable_subtitles),
            enable_profanity_filter=bool(enable_profanity_filter),
            extra_profanity_words=extra_block or None,
            enable_audio_info=bool(enable_audio_info),
            enable_speaker_turns=bool(enable_speaker_turns),
            silence_threshold_seconds=float(silence_threshold_seconds),
        )
        # Optional: persist a job row
        try:
            voice_svc.create_job(
                db=db,
                source_filename=os.path.basename(path),
                model_size=stt_resp.model_size,
                detected_language=stt_resp.language,
                language_probability=stt_resp.language_probability,
                duration_seconds=stt_resp.duration,
                segment_count=len(stt_resp.segments),
                transcription_text=(post.profanity_filtered.filtered_text if (enable_profanity_filter and post.profanity_filtered) else stt_resp.text),
                warnings=" | ".join(stt_resp.warnings) if stt_resp.warnings else None,
            )
        except Exception:
            pass
        return post.to_dict()
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


@router.post("/subtitles", tags=["Voice Intelligence"])
async def voice_subtitles(
    file: UploadFile = File(...),
    fmt: str = Form("srt", description="srt or vtt"),
    model_size: str = Form("base"),
    language: str = Form("auto"),
    vad_filter: bool = Form(True),
):
    """Return SRT or VTT subtitle text for an uploaded audio."""
    path = _save_uploaded_file(file)
    try:
        stt_resp = _transcribe_core(
            audio_path=path, model_size=model_size, language=language,
            beam_size=5, temperature=0.0, device="auto", compute_type=None,
            vad_filter=vad_filter, initial_prompt=None,
        )
        fmt_clean = (fmt or "srt").lower().strip()
        text = stt_engine.export_subtitles(stt_resp, fmt=fmt_clean)
        from fastapi.responses import PlainTextResponse
        media = "application/x-subrip" if fmt_clean == "vtt" else "text/vtt" if fmt_clean == "vtt" else "application/x-subrip"
        media = "text/vtt" if fmt_clean == "vtt" else "application/x-subrip"
        return PlainTextResponse(
            content=text,
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="subtitles.{fmt_clean}"',
                     "Access-Control-Expose-Headers": "Content-Disposition"},
        )
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


@router.post("/keyword-spot", tags=["Voice Intelligence"])
async def voice_keyword_spot(
    file: UploadFile = File(...),
    keywords: str = Form(..., description="Comma or newline separated keywords."),
    model_size: str = Form("base"),
    language: str = Form("auto"),
):
    """Transcribe audio and return keyword occurrences + segment positions."""
    path = _save_uploaded_file(file)
    try:
        stt_resp = _transcribe_core(
            audio_path=path, model_size=model_size, language=language,
            beam_size=5, temperature=0.0, device="auto", compute_type=None,
            vad_filter=True, initial_prompt=None,
        )
        kw_list = [k.strip() for k in re.split(r"[\s,、\n]+", keywords) if k.strip()]
        spots = stt_engine.spot_keywords(stt_resp, kw_list)
        return spots.to_dict()
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


@router.post("/summarize", tags=["Voice Intelligence"])
async def voice_summarize(
    file: UploadFile = File(...),
    model_size: str = Form("base"),
    language: str = Form("auto"),
    max_phrases: int = Form(10, ge=1, le=50),
):
    """Transcribe audio + return 1-line summary and top-N key phrases."""
    path = _save_uploaded_file(file)
    try:
        stt_resp = _transcribe_core(
            audio_path=path, model_size=model_size, language=language,
            beam_size=5, temperature=0.0, device="auto", compute_type=None,
            vad_filter=True, initial_prompt=None,
        )
        s = stt_engine.summarize_transcription(stt_resp, max_phrases=max(1, int(max_phrases)))
        return s.to_dict()
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
