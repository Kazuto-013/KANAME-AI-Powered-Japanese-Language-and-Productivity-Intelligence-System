from __future__ import annotations

from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Query

from ai_engine.translation import (
    translate_text,
    translate_batch,
    available_engines,
    glossary as _glossary,
    evaluation as _evaluation,
)
from backend.schemas.translation_schemas import (
    TranslateRequest,
    TranslateResponse,
    TranslateBatchRequest,
    TranslateBatchResponse,
    QualityBreakdownSchema,
    GlossaryListResponse,
    GlossaryDomainResponse,
    TranslationHealthResponse,
)

router = APIRouter(prefix="/translation", tags=["Translation Intelligence"])


@router.get("/", status_code=200)
async def translation_root():
    info = available_engines()
    return {
        "module": "Japanese <-> English Translation Intelligence",
        "status": "implemented",
        "engine": "Hybrid: LLM (Claude / Gemini) + domain glossary + dictionary fallback",
        "active_engine": info.get("active_engine", "dictionary"),
        "fallback_available": info.get("fallback_available", True),
        "endpoints": {
            "health": "GET /api/translation/health",
            "engines": "GET /api/translation/engines",
            "evaluate": "GET /api/translation/evaluate",
            "domains": "GET /api/translation/glossaries",
            "glossary_details": "GET /api/translation/glossaries/{domain}",
            "translate": "POST /api/translation/translate",
            "translate_batch": "POST /api/translation/translate/batch",
        },
        "description": (
            "Bidirectional JA↔EN translation with automatic LLM backend detection "
            "(Anthropic Claude → Google Gemini), 4 bilingual domain glossaries "
            "(business / technology / finance / legal), per-request custom glossaries and "
            "few-shot translation examples, and rule-based dictionary fallback "
            "when no API keys are configured. All endpoints return a heuristic 0–1 "
            "quality score based on length ratio, numeric preservation, format preservation "
            "and glossary coverage."
        ),
    }


@router.get("/health", response_model=TranslationHealthResponse, tags=["Translation Intelligence"])
async def translation_health():
    info = available_engines()
    return TranslationHealthResponse(
        status="ok",
        active_engine=info.get("active_engine", "dictionary"),
        configured_keys=info.get("configured_keys", []),
        fallback_available=info.get("fallback_available", True),
        domains=info.get("domains", []),
    )


@router.get("/engines", tags=["Translation Intelligence"])
async def list_engines():
    return available_engines()


@router.get("/evaluate", tags=["Translation Intelligence"])
async def evaluate_offline_pipeline(
    include_rows: bool = Query(False, description="Include per-sentence rows in the response"),
):
    """Run the deterministic (no-LLM) dictionary pipeline over the fixed
    JA<->EN corpus and return BLEU / character-F1 / quality-score metrics."""
    report = _evaluation.evaluate_translation()
    if not include_rows:
        report = {k: v for k, v in report.items() if k not in ("rows", "rows_sorted_by_char_f1")}
    return report


@router.get("/glossaries", response_model=GlossaryListResponse, tags=["Translation Intelligence"])
async def list_glossary_domains():
    return GlossaryListResponse(domains=_glossary.list_domains())


@router.get("/glossaries/{domain}", response_model=GlossaryDomainResponse, tags=["Translation Intelligence"])
async def get_glossary_domain(domain: str):
    pairs = _glossary.get_glossary_pairs(domain)
    if not pairs:
        return GlossaryDomainResponse(domain=domain, pairs=[], count=0)
    return GlossaryDomainResponse(
        domain=domain,
        pairs=[{"ja": ja, "en": en} for ja, en in pairs],
        count=len(pairs),
    )


def _to_response(result) -> TranslateResponse:
    q = result.quality_score
    return TranslateResponse(
        source_text=result.source_text,
        translated_text=result.translated_text,
        source_lang=result.source_lang,
        target_lang=result.target_lang,
        engine=result.engine,
        glossary_hits=result.glossary_hits,
        domain=result.domain,
        quality=QualityBreakdownSchema(
            overall=round(q.overall, 4),
            length_ratio_score=round(q.length_ratio_score, 4),
            numeric_preservation_score=round(q.numeric_preservation_score, 4),
            glossary_coverage_score=round(q.glossary_coverage_score, 4),
            format_preservation_score=round(q.format_preservation_score, 4),
            notes=q.notes,
        ),
        notes=result.notes,
    )


@router.post("/translate", response_model=TranslateResponse, tags=["Translation Intelligence"])
async def translate_endpoint(
    req: TranslateRequest,
    force_fallback: bool = Query(False, description="Bypass LLM entirely and use dictionary fallback only"),
):
    try:
        result = translate_text(
            text=req.text,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            domain=req.domain,
            force_fallback=force_fallback,
            custom_glossary=[item.model_dump() for item in req.custom_glossary],
            examples=[item.model_dump() for item in req.examples],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation engine error: {e}")
    return _to_response(result)


@router.post("/translate/batch", response_model=TranslateBatchResponse, tags=["Translation Intelligence"])
async def translate_batch_endpoint(
    req: TranslateBatchRequest,
    force_fallback: bool = Query(False, description="Bypass LLM entirely and use dictionary fallback only"),
):
    try:
        results = translate_batch(
            texts=req.texts,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
            domain=req.domain,
            force_fallback=force_fallback,
            custom_glossary=[item.model_dump() for item in req.custom_glossary],
            examples=[item.model_dump() for item in req.examples],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation batch engine error: {e}")

    responses = [_to_response(r) for r in results]
    return TranslateBatchResponse(results=responses, count=len(responses))
