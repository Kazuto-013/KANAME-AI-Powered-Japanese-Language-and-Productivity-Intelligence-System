from __future__ import annotations

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, BigInteger,
    Index, UniqueConstraint, LargeBinary,
)

from backend.database.config import Base


class NlpAnalysis(Base):
    __tablename__ = "nlp_analyses"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    analysis_id = Column(String(64), unique=True, nullable=False, index=True)
    input_text = Column(Text, nullable=False)
    mode = Column(String(2), nullable=False, default="C")
    token_count = Column(Integer, nullable=False, default=0)
    wakati = Column(Text, nullable=True)
    kana_only_text = Column(Text, nullable=True)
    pos_counts_json = Column(Text, nullable=True)
    noun_list_json = Column(Text, nullable=True)
    verb_list_json = Column(Text, nullable=True)
    adjective_list_json = Column(Text, nullable=True)
    tokens_json = Column(Text, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    __table_args__ = (
        Index("ix_nlp_created_at", "created_at"),
    )


class VoiceJob(Base):
    __tablename__ = "voice_jobs"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    job_id = Column(String(64), unique=True, nullable=False, index=True)
    source_filename = Column(String(512), nullable=True)
    model_size = Column(String(32), nullable=False, default="base")
    detected_language = Column(String(8), nullable=True)
    language_probability = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True, default=0.0)
    segment_count = Column(Integer, nullable=True, default=0)
    transcription_text = Column(Text, nullable=True)
    warnings = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class DatasetRecord(Base):
    __tablename__ = "dataset_records"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    dataset_id = Column(String(64), unique=True, nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    rows = Column(Integer, nullable=False, default=0)
    columns = Column(Integer, nullable=False, default=0)
    columns_names = Column(Text, nullable=True)
    size_bytes = Column(BigInteger, nullable=True, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)


class DatasetContent(Base):
    __tablename__ = "dataset_contents"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    dataset_id = Column(String(64), unique=True, nullable=False, index=True)
    content = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
