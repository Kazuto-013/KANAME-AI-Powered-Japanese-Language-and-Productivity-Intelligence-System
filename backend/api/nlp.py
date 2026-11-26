from __future__ import annotations

import json
import time
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from ai_engine.japanese_nlp import sudachi_engine
from ai_engine.japanese_nlp import supervised_ner
from backend.database.config import get_db
from backend.services import nlp_service as nlp_svc
from backend.schemas.nlp_schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    NlpHealthResponse,
    TokenInfoSchema,
    TokenizeRequest,
    TokenizeResponse,
    KeywordsRequest,
    KeywordsResponse,
    KeywordItemSchema,
    NamedEntitiesRequest,
    NamedEntitiesResponse,
    NerTrainRequest,
    NerPredictionRequest,
    SentimentRequest,
    SentimentResponse,
    JlptAnalyzeResponse,
    AlternativesRequest,
    AlternativesResponse,
    AlternativeSchema,
)

router = APIRouter(prefix="/nlp", tags=["Japanese NLP"])


@router.get("/", status_code=200)
async def nlp_root():
    return {
        "module": "Japanese NLP Intelligence",
        "status": "implemented",
        "engine": "SudachiPy + SudachiDict-core",
        "endpoints": {
            "health": "GET /api/nlp/health",
            "tokenize": "POST /api/nlp/tokenize",
            "analyze": "POST /api/nlp/analyze",
            "history": "GET /api/nlp/history?limit=50",
            "keywords": "POST /api/nlp/keywords",
            "entities": "POST /api/nlp/entities",
            "entities_model": "GET /api/nlp/entities/model",
            "entities_train": "POST /api/nlp/entities/train (annotated examples)",
            "entities_predict": "POST /api/nlp/entities/predict (trained model)",
            "sentiment": "POST /api/nlp/sentiment",
            "get_by_id": "GET /api/nlp/analyses/{analysis_id}",
            "analyze_jlpt": "POST /api/nlp/analyze-jlpt",
            "alternatives": "POST /api/nlp/alternatives",
        },
        "description": (
            "Tokenization, morphological analysis, POS tagging, readings extraction, "
            "wakati-gaki, POS-based noun / verb / adjective filtering, TF-IDF keywords, "
            "dictionary/POS-based named entities plus optional supervised BIO NER, "
            "rule-based sentiment, and analysis history "
            "via the SQLAlchemy services layer."
        ),
    }


@router.get("/health", response_model=NlpHealthResponse, tags=["Japanese NLP"])
async def nlp_health():
    try:
        d = sudachi_engine._get_dictionary()
        loaded = d is not None
    except Exception:
        loaded = False
    return NlpHealthResponse(
        status="ok",
        engine="SudachiPy + SudachiDict-core",
        dictionary_loaded=loaded,
        default_mode="C",
    )


@router.post("/tokenize", response_model=TokenizeResponse, tags=["Japanese NLP"])
async def tokenize_text(req: TokenizeRequest):
    try:
        result = sudachi_engine.tokenize(req.text, mode=req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP engine error: {e}")

    return TokenizeResponse(
        text=result.text,
        mode=result.mode,
        token_count=result.token_count,
        token_surfaces=result.token_surfaces,
        tokens=[TokenInfoSchema(**t.to_dict()) for t in result.tokens],
    )


@router.post("/analyze", tags=["Japanese NLP"])
async def analyze_text(
    req: AnalyzeRequest,
    persist: bool = Query(True, description="Persist analysis to database if available"),
    db: Session = Depends(get_db),
):
    start = time.perf_counter()
    try:
        result = sudachi_engine.analyze(req.text, mode=req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP engine error: {e}")
    duration = time.perf_counter() - start

    analysis_id: str | None = None
    if persist:
        try:
            analysis_id = nlp_svc.create_analysis(db, req.text, req.mode, result, duration)
        except Exception:
            analysis_id = None

    response = AnalyzeResponse(
        text=result.text,
        mode=result.mode,
        token_count=result.token_count,
        token_surfaces=result.token_surfaces,
        wakati=result.wakati,
        readings=result.readings,
        kana_only_text=result.kana_only_text,
        pos_counts=result.pos_counts,
        noun_list=result.noun_list,
        verb_list=result.verb_list,
        adjective_list=result.adjective_list,
        tokens=[TokenInfoSchema(**t.to_dict()) for t in result.tokens],
    ).model_dump()
    response["analysis_id"] = analysis_id
    response["duration_seconds"] = round(duration, 4)
    return response


@router.get("/analyses/{analysis_id}", tags=["Japanese NLP"])
async def get_analysis_by_id(analysis_id: str, db: Session = Depends(get_db)):
    try:
        item = nlp_svc.get_analysis(db, analysis_id)
    except Exception as e:
        return {"analysis": None, "error": f"Database unavailable: {e}"}
    if item is None:
        return {"analysis": None, "error": "analysis_id not found"}
    return {"analysis": item}


@router.get("/history", tags=["Japanese NLP"])
async def list_analysis_history(
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    try:
        history = nlp_svc.list_analyses(db, limit=limit)
    except Exception as e:
        return {"history": [], "error": f"Database unavailable: {e}"}

    trimmed = []
    for r in history:
        it = dict(r)
        if len(it.get("input_text") or "") > 120:
            it["input_text"] = it["input_text"][:120] + "…"
        trimmed.append(it)
    return {"history": trimmed, "count": len(trimmed)}


@router.post("/keywords", response_model=KeywordsResponse, tags=["Japanese NLP"])
async def extract_keywords_endpoint(req: KeywordsRequest):
    try:
        result = sudachi_engine.extract_keywords(
            text=req.text,
            mode=req.mode,
            top_n=req.top_n,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP engine error: {e}")

    return KeywordsResponse(
        text=result.text,
        mode=result.mode,
        top_n=result.top_n,
        keyword_count=len(result.keywords),
        keywords=[KeywordItemSchema(**k.to_dict()) for k in result.keywords],
    )


@router.post("/entities", response_model=NamedEntitiesResponse, tags=["Japanese NLP"])
async def extract_named_entities_endpoint(req: NamedEntitiesRequest):
    try:
        result = sudachi_engine.extract_named_entities(req.text, mode=req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP engine error: {e}")
    return NamedEntitiesResponse(**result.to_dict())


@router.get("/entities/model", tags=["Japanese NLP"])
async def supervised_ner_model_info():
    return supervised_ner.model_info()


@router.post("/entities/train", tags=["Japanese NLP"])
async def train_supervised_ner_endpoint(req: NerTrainRequest):
    examples = [
        {
            "text": example.text,
            "entities": [entity.model_dump() for entity in example.entities],
        }
        for example in req.examples
    ]
    try:
        return await run_in_threadpool(
            supervised_ner.train_ner,
            examples,
            req.mode,
            req.save,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"NER training failed: {exc}")


@router.post("/entities/predict", tags=["Japanese NLP"])
async def predict_supervised_ner_endpoint(req: NerPredictionRequest):
    try:
        result = await run_in_threadpool(
            supervised_ner.predict_entities,
            req.text,
            req.mode,
        )
        return {**result, "entity_count": len(result["entities"])}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"NER prediction failed: {exc}")


@router.post("/sentiment", response_model=SentimentResponse, tags=["Japanese NLP"])
async def analyze_sentiment_endpoint(req: SentimentRequest):
    try:
        result = sudachi_engine.analyze_sentiment(req.text, mode=req.mode)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP engine error: {e}")

    d = result.to_dict()
    return SentimentResponse(**d)


@router.post("/analyze-jlpt", tags=["Japanese NLP"])
async def analyze_text_jlpt(
    req: AnalyzeRequest,
    persist: bool = Query(True, description="Persist base analyze payload to database if available"),
    db: Session = Depends(get_db),
):
    start = time.perf_counter()
    try:
        jlpt = sudachi_engine.analyze_with_jlpt(req.text, mode=req.mode)
    except AttributeError as e:
        return {"analysis": None, "jlpt": None, "error": f"JLPT extension not loaded: {e}"}
    except ValueError as e:
        return {"analysis": None, "jlpt": None, "error": str(e)}
    except Exception as e:
        return {"analysis": None, "jlpt": None, "error": f"NLP engine error: {e}"}
    duration = time.perf_counter() - start
    jlpt_dict = jlpt.to_dict()
    analysis_id: str | None = None
    if persist:
        try:
            analysis_id = nlp_svc.create_analysis(db, req.text, req.mode, jlpt, duration)
        except Exception:
            analysis_id = None
    try:
        response = JlptAnalyzeResponse(**jlpt_dict).model_dump()
    except Exception as e:
        return {"analysis": None, "jlpt": None, "error": f"Schema build failed: {e}"}
    response["analysis_id"] = analysis_id
    response["duration_seconds"] = round(duration, 4)
    return response


@router.post("/alternatives", tags=["Japanese NLP"])
async def generate_alternatives_endpoint(req: AlternativesRequest):
    try:
        alts = sudachi_engine.generate_alternatives(req.text, mode=req.mode)
    except AttributeError as e:
        return {"alternatives": None, "error": f"Alternatives extension not loaded: {e}"}
    except ValueError as e:
        return {"alternatives": None, "error": str(e)}
    except Exception as e:
        return {"alternatives": None, "error": f"NLP engine error: {e}"}
    styles_list = []
    for a in alts:
        try:
            if isinstance(a, dict):
                styles_list.append(AlternativeSchema(**a))
            else:
                d = a.to_dict() if hasattr(a, "to_dict") else dict(a)
                styles_list.append(AlternativeSchema(**d))
        except Exception:
            continue
    try:
        resp = AlternativesResponse(text=req.text, mode=req.mode, styles=styles_list).model_dump()
    except Exception as e:
        return {"alternatives": None, "error": f"Response build failed: {e}"}
    return resp
