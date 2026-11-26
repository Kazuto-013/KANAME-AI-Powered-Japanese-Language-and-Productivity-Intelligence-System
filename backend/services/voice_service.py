from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.models.nlp_analysis import VoiceJob


def _row_to_dict(r: VoiceJob) -> Dict[str, Any]:
    return {
        "id": r.id,
        "job_id": r.job_id,
        "source_filename": r.source_filename,
        "model_size": r.model_size,
        "detected_language": r.detected_language,
        "language_probability": r.language_probability,
        "duration_seconds": r.duration_seconds,
        "segment_count": r.segment_count,
        "transcription_text": r.transcription_text,
        "warnings": r.warnings,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


def create_job(
    db: Session,
    source_filename: Optional[str] = None,
    model_size: str = "base",
    detected_language: Optional[str] = None,
    language_probability: Optional[float] = None,
    duration_seconds: Optional[float] = 0.0,
    segment_count: Optional[int] = 0,
    transcription_text: Optional[str] = None,
    warnings: Optional[str] = None,
) -> str:
    job_id = str(uuid.uuid4())
    record = VoiceJob(
        job_id=job_id,
        source_filename=source_filename,
        model_size=model_size,
        detected_language=detected_language,
        language_probability=language_probability,
        duration_seconds=duration_seconds,
        segment_count=segment_count,
        transcription_text=transcription_text,
        warnings=warnings,
    )
    try:
        db.add(record)
        db.commit()
        return job_id
    except Exception:
        db.rollback()
        raise


def get_job(db: Session, job_id: str) -> Optional[Dict[str, Any]]:
    r = db.query(VoiceJob).filter(VoiceJob.job_id == job_id).first()
    if r is None:
        return None
    return _row_to_dict(r)


def list_jobs(db: Session, limit: int = 50) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    rows = db.query(VoiceJob).order_by(VoiceJob.created_at.desc()).limit(limit).all()
    return [_row_to_dict(r) for r in rows]
