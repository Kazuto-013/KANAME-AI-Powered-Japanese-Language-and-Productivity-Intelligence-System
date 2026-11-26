"""
Japanese sentence intent detection.

Hybrid approach:
  1. Rule-based fast path using keyword patterns (question markers,
     imperative endings, greeting words, etc.) with high confidence.
  2. Calibrated classifier fallback on the INTENT_TRAIN samples in
     sample_data.py for sentences that don't trigger a rule.

Intent set (6 labels):
  - greeting  : 挨拶 (hello, good morning, nice to meet you, etc.)
  - question  : 質問 (ends with か, ？, ですか, etc.)
  - request   : 依頼 (〜てください, 〜ていただけますか, etc.)
  - command   : 命令 (〜してください, imperative forms, direct orders)
  - opinion   : 意見・感想 (〜と思う, 〜と考える, positive/negative judgment)
  - other     : その他 (everything else)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from . import sample_data as _sample_data
from .vectorizer import SudachiVectorizer


INTENT_LABELS = ["command", "greeting", "opinion", "other", "question", "request"]


_GREETING_PATTERNS = [
    r"こんにち[はわ]", r"こんばん[はわ]", r"おはよう(ございます)?",
    r"おやすみ(なさい)?", r"はじめまして", r"初めまして",
    r"よろしく(お願い)?", r"お(元気|げんき)(ですか)?",
    r"お疲れ(様|さま)", r"お世話(に)?(なって|なり)",
    r"ご無沙汰", r"失礼いたします", r"あけまして",
]
_QUESTION_PATTERNS = [
    r"\?$", r"？$", r"ですか\s*$", r"ますか\s*$", r"でしょうか\s*$",
    r"か\s*$", r"(だれ|どれ|どこ|なぜ|どう|いつ|どちら|どなた)", r"教えて", r"(何時|何で)",
]
# Polite deference markers: when present, a trailing か/？ is a request, not a question.
_POLITE_MARKERS = ("お願い", "いただけ", "ご検討", "何卒", "お手数", "申し上げます")
_REQUEST_PATTERNS = [
    r"(お願い|願い)(いたします|します|申し上げます)?",
    r"いただけ(ます|ません)?", r"ていただ", r"ご検討", r"何卒",
    r"お手数", r"もらえます",
]
_COMMAND_PATTERNS = [
    r"してください\s*[。]?$", r"しなさい\s*[。]?$", r"しろ\s*[。]?$",
    r"(捺印|署名).*(ください|返送)", r"(至急|直ちに|必ず).*(ください|して)",
    r"ください\s*[。]?$",
]
_OPINION_PATTERNS = [
    r"(と思う|と考える|と感じる|だと思う|と思います|と考えます|と感じます)",
    r"(私は|僕は|個人的には)", r"(賛成|反対|感じました|満足|リスクが高い)",
]


def _match_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


def _rule_based_intent(text: str) -> Optional[Tuple[str, float]]:
    t = text.strip()
    if not t:
        return None

    if _match_any(t, _GREETING_PATTERNS):
        return "greeting", 0.85

    # A trailing か/？ only signals a question when there is no polite deference word.
    is_polite_request = any(m in t for m in _POLITE_MARKERS)
    if (not is_polite_request) and _match_any(t, _QUESTION_PATTERNS):
        return "question", 0.80

    # Direct imperative (〜してください) is checked before request so orders win.
    if _match_any(t, _COMMAND_PATTERNS):
        return "command", 0.75

    if _match_any(t, _REQUEST_PATTERNS):
        return "request", 0.78

    if _match_any(t, _OPINION_PATTERNS):
        return "opinion", 0.70

    return None


_INTENT_CLASSIFIER: Optional[Pipeline] = None
_INTENT_CLASSES: Optional[List[str]] = None


def _build_pipeline(texts: List[str]) -> Pipeline:
    vec = SudachiVectorizer(mode="C", min_df=1, max_features=1000, ngram_range=(1, 2))
    svc = LinearSVC(loss="squared_hinge", dual="auto", C=2.0, random_state=42, max_iter=5000)
    calibrated = CalibratedClassifierCV(estimator=svc, method="sigmoid", cv=min(3, max(2, len(texts) // 5)))
    return Pipeline([("vec", vec), ("clf", calibrated)])


def _ensure_intent_classifier() -> Tuple[Pipeline, List[str]]:
    global _INTENT_CLASSIFIER, _INTENT_CLASSES
    if _INTENT_CLASSIFIER is not None and _INTENT_CLASSES is not None:
        return _INTENT_CLASSIFIER, _INTENT_CLASSES

    texts, labels = _sample_data.get_intent_labeled()
    classes = sorted(set(labels))
    pipe = _build_pipeline(texts)
    pipe.fit(texts, labels)

    _INTENT_CLASSIFIER = pipe
    _INTENT_CLASSES = classes
    return pipe, classes


@dataclass
class IntentPrediction:
    text: str
    intent: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    method: str = "hybrid"
    rule_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["confidence"] = round(self.confidence, 4)
        if self.probabilities:
            d["probabilities"] = {k: round(v, 4) for k, v in self.probabilities.items()}
        return d


def detect_intent(text: str) -> IntentPrediction:
    if not isinstance(text, str) or not text.strip():
        return IntentPrediction(
            text=text or "", intent="other", confidence=0.0,
            probabilities={}, method="empty", rule_hit=False,
        )

    rule_hit = _rule_based_intent(text)
    if rule_hit is not None:
        rule_intent, rule_conf = rule_hit
        return IntentPrediction(
            text=text, intent=rule_intent, confidence=rule_conf,
            probabilities={rule_intent: rule_conf}, method="rules", rule_hit=True,
        )

    pipe, classes = _ensure_intent_classifier()
    proba = pipe.predict_proba([text])[0]
    order = np.argsort(proba)[::-1]
    label_order = [classes[i] for i in order]
    score_order = [float(proba[i]) for i in order]
    predicted = label_order[0]
    confidence = score_order[0]
    probabilities = {label_order[i]: score_order[i] for i in range(len(classes))}

    return IntentPrediction(
        text=text, intent=predicted, confidence=confidence,
        probabilities=probabilities, method="classifier", rule_hit=False,
    )


def detect_intent_batch(texts: List[str]) -> List[IntentPrediction]:
    return [detect_intent(t) for t in texts]


def intent_classifier_info() -> Dict[str, Any]:
    pipe, classes = _ensure_intent_classifier()
    vec = pipe.named_steps["vec"]
    try:
        n_feat = len(vec.get_feature_names_out_safe())
    except Exception:
        n_feat = 0
    return {
        "classes": classes,
        "num_classes": len(classes),
        "classifier": "rule-based + CalibratedClassifierCV(LinearSVC)",
        "vectorizer": "SudachiVectorizer",
        "num_features": n_feat,
        "training_samples": len(_sample_data.INTENT_TRAIN),
        "rule_patterns": {
            "greeting": len(_GREETING_PATTERNS),
            "question": len(_QUESTION_PATTERNS),
            "request": len(_REQUEST_PATTERNS),
            "command": len(_COMMAND_PATTERNS),
            "opinion": len(_OPINION_PATTERNS),
        },
    }


def evaluate_intent_classifier(cv: int = 5) -> Dict[str, Any]:
    """Evaluate the intent detector.

    Reports two numbers:
      - ``accuracy``: out-of-fold (StratifiedKFold) accuracy of the ML
        classifier component in isolation, i.e. how well the calibrated
        LinearSVC generalises.
      - ``hybrid_accuracy``: end-to-end accuracy of the full detector
        (rule fast path + classifier fallback) measured on the labelled
        corpus. Because both components are developed against this corpus
        the hybrid number is optimistic and mainly useful as a regression
        guard / rule-coverage indicator.
    """
    from collections import defaultdict

    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    texts, labels = _sample_data.get_intent_labeled()
    classes = sorted(set(labels))
    min_class = min(labels.count(c) for c in classes)
    n_splits = max(2, min(cv, min_class))

    result: Dict[str, Any] = {
        "samples": len(texts),
        "classes": classes,
        "cv_folds": n_splits,
        "classifier": "CalibratedClassifierCV(LinearSVC)",
        "vectorizer": "SudachiVectorizer",
    }
    if len(texts) < n_splits:
        result["error"] = "not enough samples for cross-validation"
        return result

    pipe = _build_pipeline(texts)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    try:
        preds = cross_val_predict(pipe, texts, labels, cv=skf)
    except Exception as e:
        result["error"] = f"cross-validation failed: {e}"
        return result

    correct: Dict[str, int] = defaultdict(int)
    total: Dict[str, int] = defaultdict(int)
    for y, p in zip(labels, preds):
        total[y] += 1
        if y == p:
            correct[y] += 1

    result["accuracy"] = round(float(accuracy_score(labels, preds)), 4)
    result["per_class_accuracy"] = {c: round(correct[c] / total[c], 4) for c in classes if total[c]}

    # End-to-end hybrid accuracy (rule fast path + classifier fallback).
    hybrid_preds = [detect_intent(t).intent for t in texts]
    h_correct: Dict[str, int] = defaultdict(int)
    h_total: Dict[str, int] = defaultdict(int)
    for y, p in zip(labels, hybrid_preds):
        h_total[y] += 1
        if y == p:
            h_correct[y] += 1
    result["hybrid_accuracy"] = round(float(accuracy_score(labels, hybrid_preds)), 4)
    result["hybrid_per_class_accuracy"] = {c: round(h_correct[c] / h_total[c], 4) for c in classes if h_total[c]}
    return result
