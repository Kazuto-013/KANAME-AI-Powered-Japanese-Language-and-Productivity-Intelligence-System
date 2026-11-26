from __future__ import annotations

from typing import List, Dict, Any, Optional

from pydantic import BaseModel, Field


class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Japanese text to classify")
    top_k: int = Field(3, ge=1, le=10, description="How many top-k predictions to return")


class ClassifyTopK(BaseModel):
    label: str
    score: float
    rank: int


class ClassifyResponse(BaseModel):
    text: str
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    top_k: List[ClassifyTopK]


class ClassifyBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)


class ClassifyBatchResponse(BaseModel):
    results: List[ClassifyResponse]
    count: int


class ClassifierInfoResponse(BaseModel):
    classes: List[str]
    num_classes: int
    vectorizer: str
    classifier: str
    num_features: int
    num_training_samples: int


class ClusterRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    n_clusters: Optional[int] = Field(None, ge=1, le=20, description="Desired cluster count. If omitted, auto-detect 2..5 based on input size.")
    top_keywords: int = Field(6, ge=1, le=20)
    n_examples: int = Field(2, ge=0, le=10)


class ClusterInfoResponse(BaseModel):
    cluster_id: int
    size: int
    top_keywords: List[str]
    example_texts: List[str]
    centroid_norm: float


class ClusterResponse(BaseModel):
    n_clusters: int
    total_texts: int
    vectorizer_feature_count: int
    labels: List[int]
    clusters: List[ClusterInfoResponse]


class TopicModelRequest(BaseModel):
    texts: List[str] = Field(..., min_length=2, max_length=500)
    n_topics: int = Field(5, ge=1, le=20)
    top_words: int = Field(8, ge=1, le=30)


class AutoLabelRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=500)
    confidence_threshold: float = Field(0.8, ge=0.0, le=1.0)


class IntentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Japanese sentence for intent detection")


class IntentResponse(BaseModel):
    text: str
    intent: str
    confidence: float
    probabilities: Dict[str, float]
    method: str
    rule_hit: bool


class IntentBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)


class IntentBatchResponse(BaseModel):
    results: List[IntentResponse]
    count: int


class IntentInfoResponse(BaseModel):
    classes: List[str]
    num_classes: int
    classifier: str
    vectorizer: str
    num_features: int
    training_samples: int
    rule_patterns: Dict[str, int]


class MlHealthResponse(BaseModel):
    status: str
    engines: List[str]
    classifier_classes: List[str]
    intent_classes: List[str]


# ---- Custom PyTorch model ----

class TorchStatusResponse(BaseModel):
    available: bool
    engine: str = "PyTorch MLP (CPU)"
    torch_version: Optional[str] = None
    device: Optional[str] = None
    classes: List[str] = Field(default_factory=list)
    num_classes: Optional[int] = None
    num_features: Optional[int] = None
    num_training_samples: Optional[int] = None
    train_accuracy: Optional[float] = None
    final_loss: Optional[float] = None
    epochs: Optional[int] = None
    architecture: Optional[str] = None
    optimizer: Optional[str] = None
    vectorizer: Optional[str] = None
    error: Optional[str] = None


class TorchTrainRequest(BaseModel):
    epochs: Optional[int] = Field(None, ge=1, le=5000, description="Training epochs (default 300)")
    hidden1: int = Field(128, ge=1, le=1024)
    hidden2: int = Field(64, ge=1, le=1024)
    dropout: float = Field(0.2, ge=0.0, le=0.9)
    lr: float = Field(1e-3, gt=0.0, le=1.0)
    seed: int = Field(42, ge=0)


class TorchClassifyResponse(BaseModel):
    text: str
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float]
    top_k: List[ClassifyTopK]
    engine: str


class TorchClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Japanese text to classify with the custom model")
    top_k: int = Field(3, ge=1, le=10)


class TorchClassifyBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)


class TorchClassifyBatchResponse(BaseModel):
    results: List[TorchClassifyResponse]
    count: int


class TransformerTrainRequest(BaseModel):
    epochs: int = Field(100, ge=1, le=2000)
    seed: int = Field(42, ge=0)


class TransformerClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)
