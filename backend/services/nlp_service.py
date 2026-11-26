from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from backend.models.nlp_analysis import NlpAnalysis


def _row_to_dict(r: NlpAnalysis) -> Dict[str, Any]:
    return {
        "id": r.id,
        "analysis_id": r.analysis_id,
        "input_text": r.input_text,
        "mode": r.mode,
        "token_count": r.token_count,
        "wakati": r.wakati,
        "kana_only_text": r.kana_only_text,
        "pos_counts": json.loads(r.pos_counts_json) if r.pos_counts_json else {},
        "noun_list": json.loads(r.noun_list_json) if r.noun_list_json else [],
        "verb_list": json.loads(r.verb_list_json) if r.verb_list_json else [],
        "adjective_list": json.loads(r.adjective_list_json) if r.adjective_list_json else [],
        "tokens": json.loads(r.tokens_json) if r.tokens_json else [],
        "duration_seconds": r.duration_seconds,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


def create_analysis(
    db: Session,
    text: str,
    mode: str,
    result: Any,
    duration: float,
) -> str:
    tokens = [t.to_dict() for t in getattr(result, "tokens", [])]
    analysis_id = str(uuid.uuid4())
    record = NlpAnalysis(
        analysis_id=analysis_id,
        input_text=text,
        mode=mode,
        token_count=getattr(result, "token_count", 0),
        wakati=getattr(result, "wakati", None),
        kana_only_text=getattr(result, "kana_only_text", None),
        pos_counts_json=json.dumps(getattr(result, "pos_counts", {}), ensure_ascii=False),
        noun_list_json=json.dumps(getattr(result, "noun_list", []), ensure_ascii=False),
        verb_list_json=json.dumps(getattr(result, "verb_list", []), ensure_ascii=False),
        adjective_list_json=json.dumps(getattr(result, "adjective_list", []), ensure_ascii=False),
        tokens_json=json.dumps(tokens, ensure_ascii=False),
        duration_seconds=duration,
    )
    try:
        db.add(record)
        db.commit()
        return analysis_id
    except Exception:
        db.rollback()
        raise


def get_analysis(db: Session, analysis_id: str) -> Optional[Dict[str, Any]]:
    r = db.query(NlpAnalysis).filter(NlpAnalysis.analysis_id == analysis_id).first()
    if r is None:
        return None
    return _row_to_dict(r)


def list_analyses(db: Session, limit: int = 50) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    rows = db.query(NlpAnalysis).order_by(NlpAnalysis.created_at.desc()).limit(limit).all()
    return [_row_to_dict(r) for r in rows]


def delete_analysis(db: Session, analysis_id: str) -> bool:
    r = db.query(NlpAnalysis).filter(NlpAnalysis.analysis_id == analysis_id).first()
    if r is None:
        return False
    try:
        db.delete(r)
        db.commit()
        return True
    except Exception:
        db.rollback()
        return False
