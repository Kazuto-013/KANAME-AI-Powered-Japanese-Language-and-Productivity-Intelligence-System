from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.models.nlp_analysis import DatasetContent, DatasetRecord


def _row_to_dict(r: DatasetRecord) -> Dict[str, Any]:
    return {
        "id": r.id,
        "dataset_id": r.dataset_id,
        "filename": r.filename,
        "rows": r.rows,
        "columns": r.columns,
        "columns_names": (r.columns_names or "").split(",") if r.columns_names else [],
        "size_bytes": r.size_bytes,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


def create_dataset_record(
    db: Session,
    filename: str,
    rows: int,
    columns: int,
    columns_names: Optional[str] = None,
    size_bytes: Optional[int] = 0,
    content: Optional[bytes] = None,
) -> str:
    dataset_id = str(uuid.uuid4())
    bind = db.get_bind()
    DatasetRecord.__table__.create(bind=bind, checkfirst=True)
    if content is not None:
        DatasetContent.__table__.create(bind=bind, checkfirst=True)
    record = DatasetRecord(
        dataset_id=dataset_id,
        filename=filename,
        rows=int(rows),
        columns=int(columns),
        columns_names=columns_names,
        size_bytes=size_bytes,
    )
    try:
        db.add(record)
        if content is not None:
            db.add(DatasetContent(dataset_id=dataset_id, content=content))
        db.commit()
        return dataset_id
    except Exception:
        db.rollback()
        raise


def get_dataset(db: Session, dataset_id: str) -> Optional[Dict[str, Any]]:
    r = db.query(DatasetRecord).filter(DatasetRecord.dataset_id == dataset_id).first()
    if r is None:
        return None
    return _row_to_dict(r)


def get_dataset_content(db: Session, dataset_id: str) -> Optional[bytes]:
    DatasetContent.__table__.create(bind=db.get_bind(), checkfirst=True)
    content = (
        db.query(DatasetContent)
        .filter(DatasetContent.dataset_id == dataset_id)
        .first()
    )
    return content.content if content is not None else None


def list_datasets(db: Session, limit: int = 100) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    rows = db.query(DatasetRecord).order_by(DatasetRecord.created_at.desc()).limit(limit).all()
    return [_row_to_dict(r) for r in rows]
