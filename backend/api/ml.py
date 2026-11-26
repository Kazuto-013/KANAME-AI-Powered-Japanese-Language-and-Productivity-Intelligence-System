from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ai_engine.machine_learning import (
    classify_text,
    classify_batch,
    classifier_info,
    cluster_texts,
    detect_intent,
    detect_intent_batch,
    intent_classifier_info,
    evaluate_intent_classifier,
    INTENT_LABELS,
    sample_data as _sample_data,
    torch_classifier as _torch,
)
from ai_engine.machine_learning.topic_modeling import discover_topics
from ai_engine.machine_learning import torch_transformer as _transformer
from backend.schemas.ml_schemas import (
    ClassifyRequest,
    ClassifyResponse,
    ClassifyTopK,
    ClassifyBatchRequest,
    ClassifyBatchResponse,
    ClassifierInfoResponse,
    ClusterRequest,
    ClusterResponse,
    ClusterInfoResponse,
    TopicModelRequest,
    AutoLabelRequest,
    TransformerTrainRequest,
    TransformerClassifyRequest,
    IntentRequest,
    IntentResponse,
    IntentBatchRequest,
    IntentBatchResponse,
    IntentInfoResponse,
    MlHealthResponse,
    TorchStatusResponse,
    TorchTrainRequest,
    TorchClassifyRequest,
    TorchClassifyResponse,
    TorchClassifyBatchRequest,
    TorchClassifyBatchResponse,
)

router = APIRouter(prefix="/ml", tags=["Machine Learning Intelligence"])


@router.get("/", status_code=200)
async def ml_root():
    return {
        "module": "Machine Learning Intelligence",
        "status": "implemented",
        "engine": "scikit-learn + SudachiPy custom vectorizer + custom PyTorch MLP",
        "capabilities": [
            "Japanese text classification (5 categories: technology / business / education / health / lifestyle)",
            "Japanese text clustering (KMeans on TF-IDF, per-cluster top keywords)",
            "Japanese topic discovery (LDA over Sudachi token counts)",
            "Confidence-thresholded auto-label suggestions for human review",
            "Japanese sentence intent detection (6 intents: greeting / question / request / command / opinion / other)",
            "Custom-trained PyTorch neural text classifier (MLP trained via gradient descent on the sample corpus)",
        ],
        "endpoints": {
            "health": "GET /api/ml/health",
            "classify": "POST /api/ml/classify",
            "classify_batch": "POST /api/ml/classify/batch",
            "classifier_info": "GET /api/ml/classifier/info",
            "cluster": "POST /api/ml/cluster",
            "cluster_demo": "GET /api/ml/cluster/demo",
            "topics": "POST /api/ml/topics",
            "auto_label": "POST /api/ml/auto-label",
            "intent": "POST /api/ml/intent",
            "intent_batch": "POST /api/ml/intent/batch",
            "intent_info": "GET /api/ml/intent/info",
            "intent_evaluate": "GET /api/ml/intent/evaluate",
            "torch_info": "GET /api/ml/torch/info",
            "torch_train": "POST /api/ml/torch/train",
            "torch_classify": "POST /api/ml/torch/classify",
            "torch_classify_batch": "POST /api/ml/torch/classify/batch",
            "topic_modeling": "POST /api/ml/topics",
            "auto_label": "POST /api/ml/auto-label",
            "transformer_info": "GET /api/ml/torch-transformer/info",
            "transformer_train": "POST /api/ml/torch-transformer/train",
            "transformer_classify": "POST /api/ml/torch-transformer/classify",
        },
        "description": (
            "Built-in classifiers are trained lazily on a self-contained Japanese sample dataset "
            "(25 sentences for 5-category classification, 60 sentences for 6-intent detection) "
            "wrapped with a scikit-learn Pipeline (SudachiVectorizer + CalibratedClassifierCV / KMeans). "
            "Intent detection is hybrid: a rule-based fast path handles clear cases (accuracy ~0.97 on "
            "the labelled corpus) while the calibrated LinearSVC classifies the remainder. "
            "A fourth, custom-trained PyTorch MLP classifier (SudachiVectorizer -> MLP) is trained "
            "end-to-end via gradient descent and cached to disk. "
            "All capabilities expose `/info` metadata endpoints and support batch processing."
        ),
    }


@router.get("/health", response_model=MlHealthResponse, tags=["Machine Learning Intelligence"])
async def ml_health():
    ci = classifier_info()
    classes = ci.get("classes", [])
    intent_labels = sorted(INTENT_LABELS)
    engines = ["LinearSVC classifier", "KMeans clustering", "hybrid intent detection"]
    if _torch.is_available():
        engines.append("custom PyTorch MLP classifier")
    return MlHealthResponse(
        status="ok",
        engines=engines,
        classifier_classes=classes,
        intent_classes=intent_labels,
    )


# ---- Classification ----

@router.get("/classifier/info", response_model=ClassifierInfoResponse, tags=["Machine Learning Intelligence"])
async def get_classifier_info():
    info = classifier_info()
    return ClassifierInfoResponse(**{k: info[k] for k in ClassifierInfoResponse.model_fields})


def _to_classify_response(pred) -> ClassifyResponse:
    return ClassifyResponse(
        text=pred.text,
        predicted_label=pred.predicted_label,
        confidence=round(pred.confidence, 4),
        probabilities={k: round(v, 4) for k, v in pred.probabilities.items()},
        top_k=[
            ClassifyTopK(
                label=item["label"],
                score=round(float(item["score"]), 4),
                rank=int(item["rank"]),
            )
            for item in pred.top_k
        ],
    )


@router.post("/classify", response_model=ClassifyResponse, tags=["Machine Learning Intelligence"])
async def classify_endpoint(req: ClassifyRequest):
    try:
        pred = classify_text(req.text, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML classifier error: {e}")
    return _to_classify_response(pred)


@router.post("/classify/batch", response_model=ClassifyBatchResponse, tags=["Machine Learning Intelligence"])
async def classify_batch_endpoint(req: ClassifyBatchRequest):
    try:
        preds = classify_batch(req.texts, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML batch classifier error: {e}")
    items = [_to_classify_response(p) for p in preds]
    return ClassifyBatchResponse(results=items, count=len(items))


# ---- Clustering ----

@router.post("/cluster", response_model=ClusterResponse, tags=["Machine Learning Intelligence"])
async def cluster_endpoint(req: ClusterRequest):
    try:
        result = cluster_texts(
            texts=req.texts,
            n_clusters=req.n_clusters,
            top_keywords=req.top_keywords,
            n_examples=req.n_examples,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML clustering error: {e}")

    clusters = [
        ClusterInfoResponse(
            cluster_id=c.cluster_id,
            size=c.size,
            top_keywords=list(c.top_keywords),
            example_texts=list(c.example_texts),
            centroid_norm=round(float(c.centroid_norm), 6),
        )
        for c in result.clusters
    ]
    return ClusterResponse(
        n_clusters=result.n_clusters,
        total_texts=result.total_texts,
        vectorizer_feature_count=result.vectorizer_feature_count,
        labels=list(result.labels),
        clusters=clusters,
    )


@router.get("/cluster/demo", response_model=ClusterResponse, tags=["Machine Learning Intelligence"])
async def cluster_demo_endpoint(n_clusters: int = 3):
    all_texts = []
    for sentences in _sample_data.TRAIN_CATEGORIES.values():
        all_texts.extend(sentences)
    result = cluster_texts(all_texts, n_clusters=n_clusters)
    clusters = [
        ClusterInfoResponse(
            cluster_id=c.cluster_id,
            size=c.size,
            top_keywords=list(c.top_keywords),
            example_texts=list(c.example_texts),
            centroid_norm=round(float(c.centroid_norm), 6),
        )
        for c in result.clusters
    ]
    return ClusterResponse(
        n_clusters=result.n_clusters,
        total_texts=result.total_texts,
        vectorizer_feature_count=result.vectorizer_feature_count,
        labels=list(result.labels),
        clusters=clusters,
    )


@router.post("/topics", tags=["Machine Learning Intelligence"])
async def topics_endpoint(req: TopicModelRequest):
    try:
        return discover_topics(req.texts, n_topics=req.n_topics, top_words=req.top_words)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Topic modeling error: {exc}")


@router.post("/auto-label", tags=["Machine Learning Intelligence"])
async def auto_label_endpoint(req: AutoLabelRequest):
    try:
        predictions = classify_batch(req.texts, top_k=1)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Auto-labeling error: {exc}")
    items = []
    for prediction in predictions:
        accepted = prediction.confidence >= req.confidence_threshold
        items.append({
            "text": prediction.text,
            "suggested_label": prediction.predicted_label,
            "confidence": round(float(prediction.confidence), 4),
            "status": "suggested" if accepted else "needs_review",
        })
    return {
        "items": items,
        "count": len(items),
        "suggested_count": sum(item["status"] == "suggested" for item in items),
        "threshold": req.confidence_threshold,
        "warning": "Suggestions are model predictions, not verified ground truth.",
    }


# ---- Intent Detection ----

@router.get("/intent/info", response_model=IntentInfoResponse, tags=["Machine Learning Intelligence"])
async def get_intent_info():
    info = intent_classifier_info()
    return IntentInfoResponse(**{k: info[k] for k in IntentInfoResponse.model_fields})


def _to_intent_response(p) -> IntentResponse:
    return IntentResponse(
        text=p.text,
        intent=p.intent,
        confidence=round(p.confidence, 4),
        probabilities={k: round(v, 4) for k, v in p.probabilities.items()},
        method=p.method,
        rule_hit=bool(p.rule_hit),
    )


@router.post("/intent", response_model=IntentResponse, tags=["Machine Learning Intelligence"])
async def intent_endpoint(req: IntentRequest):
    try:
        pred = detect_intent(req.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML intent error: {e}")
    return _to_intent_response(pred)


@router.post("/intent/batch", response_model=IntentBatchResponse, tags=["Machine Learning Intelligence"])
async def intent_batch_endpoint(req: IntentBatchRequest):
    try:
        preds = detect_intent_batch(req.texts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML intent batch error: {e}")
    items = [_to_intent_response(p) for p in preds]
    return IntentBatchResponse(results=items, count=len(items))


@router.get("/intent/evaluate", tags=["Machine Learning Intelligence"])
async def intent_evaluate_endpoint():
    """Cross-validated accuracy of the ML classifier plus end-to-end hybrid
    (rule fast path + classifier fallback) accuracy over the labelled corpus."""
    return evaluate_intent_classifier()


# ---- Custom PyTorch model ----

def _to_torch_response(pred) -> TorchClassifyResponse:
    return TorchClassifyResponse(
        text=pred.text,
        predicted_label=pred.predicted_label,
        confidence=round(float(pred.confidence), 4),
        probabilities={k: round(float(v), 4) for k, v in pred.probabilities.items()},
        top_k=[
            ClassifyTopK(label=item["label"], score=round(float(item["score"]), 4), rank=int(item["rank"]))
            for item in pred.top_k
        ],
        engine=pred.engine,
    )


@router.get("/torch/info", response_model=TorchStatusResponse, tags=["Machine Learning Intelligence"])
async def torch_info_endpoint():
    """Metadata for the custom-trained PyTorch MLP classifier (trains lazily)."""
    if not _torch.is_available():
        return TorchStatusResponse(available=False, error=_torch.get_error_message())
    try:
        info = _torch.model_info()
    except Exception as e:
        return TorchStatusResponse(available=False, error=f"PyTorch model error: {e}")
    return TorchStatusResponse(**{k: info.get(k) for k in TorchStatusResponse.model_fields if k != "available"}, available=True)


@router.post("/torch/train", response_model=TorchStatusResponse, tags=["Machine Learning Intelligence"])
async def torch_train_endpoint(req: TorchTrainRequest):
    """(Re)train the custom PyTorch model end-to-end and persist the weights."""
    if not _torch.is_available():
        raise HTTPException(status_code=503, detail=_torch.get_error_message() or "PyTorch is unavailable.")
    try:
        info = _torch.train_model(
            epochs=req.epochs,
            hidden1=req.hidden1,
            hidden2=req.hidden2,
            dropout=req.dropout,
            lr=req.lr,
            seed=req.seed,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PyTorch training error: {e}")
    return TorchStatusResponse(**{k: info.get(k) for k in TorchStatusResponse.model_fields if k != "available"}, available=True)


@router.post("/torch/classify", response_model=TorchClassifyResponse, tags=["Machine Learning Intelligence"])
async def torch_classify_endpoint(req: TorchClassifyRequest):
    if not _torch.is_available():
        raise HTTPException(status_code=503, detail=_torch.get_error_message() or "PyTorch is unavailable.")
    try:
        pred = _torch.classify_text(req.text, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PyTorch classifier error: {e}")
    return _to_torch_response(pred)


@router.post("/torch/classify/batch", response_model=TorchClassifyBatchResponse, tags=["Machine Learning Intelligence"])
async def torch_classify_batch_endpoint(req: TorchClassifyBatchRequest):
    if not _torch.is_available():
        raise HTTPException(status_code=503, detail=_torch.get_error_message() or "PyTorch is unavailable.")
    try:
        preds = _torch.classify_batch(req.texts, top_k=req.top_k)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PyTorch batch classifier error: {e}")
    items = [_to_torch_response(p) for p in preds]
    return TorchClassifyBatchResponse(results=items, count=len(items))


@router.get("/torch-transformer/info", tags=["Machine Learning Intelligence"])
async def transformer_info_endpoint():
    if not _transformer.is_available():
        return {"available": False, "error": "PyTorch unavailable."}
    try:
        return _transformer.model_info()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transformer model error: {exc}")


@router.post("/torch-transformer/train", tags=["Machine Learning Intelligence"])
async def transformer_train_endpoint(req: TransformerTrainRequest):
    if not _transformer.is_available():
        raise HTTPException(status_code=503, detail="PyTorch is unavailable.")
    try:
        return _transformer.train_model(epochs=req.epochs, seed=req.seed)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transformer training error: {exc}")


@router.post("/torch-transformer/classify", tags=["Machine Learning Intelligence"])
async def transformer_classify_endpoint(req: TransformerClassifyRequest):
    if not _transformer.is_available():
        raise HTTPException(status_code=503, detail="PyTorch is unavailable.")
    try:
        result = _transformer.classify_text(req.text, top_k=req.top_k)
        return {**result, "engine": "PyTorch TransformerEncoder (CPU)"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transformer classification error: {exc}")
