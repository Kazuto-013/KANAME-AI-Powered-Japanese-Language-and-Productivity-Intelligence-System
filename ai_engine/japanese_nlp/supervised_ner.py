"""Trainable Japanese BIO named-entity recognition using Sudachi features."""
from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

from .sudachi_engine import tokenize


_MODEL_PATH = os.getenv(
    "KANAME_NER_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "supervised_ner.joblib"),
)
_STATE: dict[str, Any] | None = None


def _feature(tokens: list[Any], index: int) -> dict[str, Any]:
    token = tokens[index]
    surface = token.surface
    features: dict[str, Any] = {
        "bias": 1,
        "surface": surface,
        "normalized": token.normalized_form,
        "dictionary": token.dictionary_form,
        "pos": "|".join(token.pos),
        "pos_main": token.pos_main,
        "is_digit": surface.isdigit(),
        "is_ascii": surface.isascii(),
        "prefix_1": surface[:1],
        "prefix_2": surface[:2],
        "suffix_1": surface[-1:],
        "suffix_2": surface[-2:],
    }
    for offset in (-2, -1, 1, 2):
        neighbor_index = index + offset
        if 0 <= neighbor_index < len(tokens):
            neighbor = tokens[neighbor_index]
            features[f"ctx_{offset}_surface"] = neighbor.surface
            features[f"ctx_{offset}_norm"] = neighbor.normalized_form
            features[f"ctx_{offset}_pos"] = neighbor.pos_main
    return features


def _prepare_example(example: dict[str, Any], mode: str) -> tuple[list[Any], list[str]]:
    text = example.get("text")
    entities = example.get("entities", [])
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Every training example must contain non-empty text.")
    if not isinstance(entities, list):
        raise ValueError("Each example's entities must be a list.")

    tokens = tokenize(text, mode=mode).tokens
    labels = ["O"] * len(tokens)
    occupied: set[int] = set()
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError("Each entity must be an object.")
        start, end = entity.get("start"), entity.get("end")
        label = entity.get("label")
        if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
            raise ValueError("Entity offsets must be valid non-empty Python string spans.")
        if not isinstance(label, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", label):
            raise ValueError("Entity labels must contain only letters, digits, or underscores.")
        label = label.upper()
        matched = [
            i for i, token in enumerate(tokens)
            if token.start >= start and token.end <= end
        ]
        if (
            not matched
            or tokens[matched[0]].start != start
            or tokens[matched[-1]].end != end
            or any(tokens[i].start < end and tokens[i].end > start for i in range(len(tokens)) if i not in matched)
        ):
            raise ValueError(
                f"Entity span {start}:{end} must align exactly with Sudachi token boundaries."
            )
        if occupied.intersection(matched):
            raise ValueError("Entity spans must not overlap.")
        occupied.update(matched)
        labels[matched[0]] = f"B-{label}"
        for index in matched[1:]:
            labels[index] = f"I-{label}"
    return tokens, labels


def train_ner(
    examples: list[dict[str, Any]],
    mode: str = "C",
    save: bool = True,
) -> dict[str, Any]:
    if len(examples) < 2:
        raise ValueError("At least two annotated training examples are required.")
    if mode not in ("A", "B", "C"):
        raise ValueError("mode must be A, B, or C.")

    feature_rows: list[dict[str, Any]] = []
    target_labels: list[str] = []
    for example in examples:
        tokens, labels = _prepare_example(example, mode)
        feature_rows.extend(_feature(tokens, i) for i in range(len(tokens)))
        target_labels.extend(labels)
    if not feature_rows or len(set(target_labels)) < 2 or "O" not in target_labels:
        raise ValueError("Training data must include at least one entity token and non-entity tokens.")

    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(feature_rows)
    classifier = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=42,
    )
    classifier.fit(matrix, target_labels)
    training_accuracy = float(classifier.score(matrix, target_labels))

    global _STATE
    _STATE = {
        "vectorizer": vectorizer,
        "classifier": classifier,
        "labels": sorted({label[2:] for label in target_labels if label != "O"}),
        "metadata": {
            "training_examples": len(examples),
            "training_tokens": len(target_labels),
            "training_accuracy": round(training_accuracy, 4),
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "evaluation_warning": (
                "Training accuracy is not a generalization metric; evaluate with a held-out, "
                "representative labeled corpus."
            ),
        },
    }
    if save:
        parent = os.path.dirname(_MODEL_PATH)
        if parent:
            os.makedirs(parent, exist_ok=True)
        joblib.dump(_STATE, _MODEL_PATH)
    return model_info()


def _load_state() -> dict[str, Any] | None:
    global _STATE
    if _STATE is None and os.path.isfile(_MODEL_PATH):
        _STATE = joblib.load(_MODEL_PATH)
    return _STATE


def model_info() -> dict[str, Any]:
    state = _load_state()
    if state is None:
        return {"trained": False, "model_path": _MODEL_PATH}
    return {
        "trained": True,
        "entity_labels": state["labels"],
        **state["metadata"],
    }


def predict_entities(text: str, mode: str = "C") -> dict[str, Any]:
    state = _load_state()
    if state is None:
        raise RuntimeError("No supervised NER model is trained; submit annotated examples first.")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty.")
    if mode not in ("A", "B", "C"):
        raise ValueError("mode must be A, B, or C.")

    tokens = tokenize(text, mode=mode).tokens
    if not tokens:
        return {"text": text, "mode": mode, "method": "supervised BIO token classifier", "entities": []}
    features = [ _feature(tokens, i) for i in range(len(tokens)) ]
    matrix = state["vectorizer"].transform(features)
    probabilities = state["classifier"].predict_proba(matrix)
    predicted = state["classifier"].classes_[np.argmax(probabilities, axis=1)].tolist()

    entities = []
    active_label: str | None = None
    active_start = 0
    active_end = 0
    active_confidences: list[float] = []

    def flush() -> None:
        if active_label is not None:
            entities.append({
                "text": text[active_start:active_end],
                "label": active_label,
                "start": active_start,
                "end": active_end,
                "confidence": round(float(np.mean(active_confidences)), 4),
                "source": "supervised_bio_model",
            })

    for index, tag in enumerate(predicted):
        prefix, _, label = tag.partition("-")
        if prefix not in ("B", "I") or not label:
            flush()
            active_label = None
            active_confidences = []
            continue
        if prefix == "B" or active_label != label:
            flush()
            active_label = label
            active_start = tokens[index].start
            active_end = tokens[index].end
            active_confidences = [float(np.max(probabilities[index]))]
        else:
            active_end = tokens[index].end
            active_confidences.append(float(np.max(probabilities[index])))
    flush()
    return {
        "text": text,
        "mode": mode,
        "method": "supervised BIO token classifier",
        "entities": entities,
    }
