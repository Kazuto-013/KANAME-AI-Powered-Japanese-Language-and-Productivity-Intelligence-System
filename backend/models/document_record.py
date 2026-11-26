from __future__ import annotations

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, BigInteger,
    Index,
)

from backend.database.config import Base


class DocumentRecord(Base):
    __tablename__ = "document_records"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    document_id = Column(String(64), unique=True, nullable=False, index=True)
    generator = Column(String(64), nullable=False)
    filename = Column(String(512), nullable=False)
    format = Column(String(8), nullable=False, default="docx")
    title = Column(String(512), nullable=True)
    author = Column(String(256), nullable=True)
    size_bytes = Column(BigInteger, nullable=True, default=0)
    metadata_json = Column(Text, nullable=True)
    storage_path = Column(String(1024), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_documents_created_at", "created_at"),
        Index("ix_documents_generator", "generator"),
    )
