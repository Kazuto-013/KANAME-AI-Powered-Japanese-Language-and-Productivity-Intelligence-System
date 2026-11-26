"""
Japanese text classification using scikit-learn LinearSVC on top of
the Sudachi-backed TF-IDF vectorizer.

Five built-in categories (technology / business / education / health / lifestyle)
are trained lazily on first call. Both single-sentence prediction and
batch prediction with per-class confidence are supported.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from . import sample_data as _sample_data
from .vectorizer import SudachiVectorizer


@dataclass
class ClassificationPrediction:
    text: str
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    top_k: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["confidence"] = round(self.confidence, 4)
        if self.probabilities:
            d["probabilities"] = {k: round(v, 4) for k, v in self.probabilities.items()}
        if self.top_k:
            for item in d["top_k"]:
                if "score" in item:
                    item["score"] = round(float(item["score"]), 4)
        return d


_CLASSIFIER: Optional[Pipeline] = None
_CLASSIFIER_CLASSES: Optional[List[str]] = None


def _ensure_classifier() -> Tuple[Pipeline, List[str]]:
    global _CLASSIFIER, _CLASSIFIER_CLASSES
    if _CLASSIFIER is not None and _CLASSIFIER_CLASSES is not None:
        return _CLASSIFIER, _CLASSIFIER_CLASSES

    texts, labels = _sample_data.get_labeled_texts()
    classes = sorted(set(labels))
    vec = SudachiVectorizer(mode="C", min_df=1, max_features=500, ngram_range=(1, 1))
    svc = LinearSVC(loss="squared_hinge", dual="auto", C=1.2, random_state=42, max_iter=2000)
    calibrated = CalibratedClassifierCV(estimator=svc, method="sigmoid", cv=min(3, len(texts) // 4 + 1))
    pipe = Pipeline([("vec", vec), ("clf", calibrated)])
    pipe.fit(texts, labels)

    _CLASSIFIER = pipe
    _CLASSIFIER_CLASSES = classes
    return pipe, classes


def classifier_info() -> Dict[str, Any]:
    pipe, classes = _ensure_classifier()
    vec = pipe.named_steps["vec"]
    try:
        n_feat = len(vec.get_feature_names_out_safe())
    except Exception:
        n_feat = 0
    return {
        "classes": classes,
        "num_classes": len(classes),
        "vectorizer": "SudachiVectorizer + TfidfVectorizer",
        "classifier": "CalibratedClassifierCV(LinearSVC, sigmoid)",
        "num_features": n_feat,
        "num_training_samples": sum(len(v) for v in _sample_data.TRAIN_CATEGORIES.values()),
    }


def classify_text(text: str, top_k: int = 3) -> ClassificationPrediction:
    if not isinstance(text, str) or not text.strip():
        return ClassificationPrediction(
            text=text or "", predicted_label="unknown", confidence=0.0,
            probabilities={}, top_k=[],
        )

    pipe, classes = _ensure_classifier()
    proba = pipe.predict_proba([text])[0]
    order = np.argsort(proba)[::-1]
    label_order = [classes[i] for i in order]
    score_order = [float(proba[i]) for i in order]

    predicted = label_order[0]
    confidence = score_order[0]
    probabilities = {label_order[i]: score_order[i] for i in range(len(classes))}
    top_k_list = [
        {"label": label_order[i], "score": score_order[i], "rank": i + 1}
        for i in range(min(top_k, len(classes)))
    ]

    return ClassificationPrediction(
        text=text,
        predicted_label=predicted,
        confidence=confidence,
        probabilities=probabilities,
        top_k=top_k_list,
    )


def classify_batch(texts: List[str], top_k: int = 3) -> List[ClassificationPrediction]:
    return [classify_text(t, top_k=top_k) for t in texts]
