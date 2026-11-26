from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.models.document_record import DocumentRecord


def _row_to_dict(r: DocumentRecord) -> Dict[str, Any]:
    return {
        "id": r.id,
        "document_id": r.document_id,
        "generator": r.generator,
        "filename": r.filename,
        "format": r.format,
        "title": r.title,
        "author": r.author,
        "size_bytes": r.size_bytes,
        "metadata": json.loads(r.metadata_json) if r.metadata_json else {},
        "storage_path": r.storage_path,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


def create_document(
    db: Session,
    generator: str,
    filename: str,
    format: str = "docx",
    title: Optional[str] = None,
    author: Optional[str] = None,
    size_bytes: Optional[int] = 0,
    metadata: Optional[Dict[str, Any]] = None,
    storage_path: Optional[str] = None,
) -> str:
    document_id = str(uuid.uuid4())
    record = DocumentRecord(
        document_id=document_id,
        generator=generator,
        filename=filename,
        format=format,
        title=title,
        author=author,
        size_bytes=int(size_bytes or 0),
        metadata_json=json.dumps(metadata or {}, ensure_ascii=False),
        storage_path=storage_path,
    )
    try:
        db.add(record)
        db.commit()
        return document_id
    except Exception:
        db.rollback()
        raise


def get_document(db: Session, document_id: str) -> Optional[Dict[str, Any]]:
    r = db.query(DocumentRecord).filter(DocumentRecord.document_id == document_id).first()
    if r is None:
        return None
    return _row_to_dict(r)


def list_documents(db: Session, limit: int = 50) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    rows = db.query(DocumentRecord).order_by(DocumentRecord.created_at.desc()).limit(limit).all()
    return [_row_to_dict(r) for r in rows]


def delete_document(db: Session, document_id: str) -> bool:
    r = db.query(DocumentRecord).filter(DocumentRecord.document_id == document_id).first()
    if r is None:
        return False
    try:
        db.delete(r)
        db.commit()
        return True
    except Exception:
        db.rollback()
        return False
