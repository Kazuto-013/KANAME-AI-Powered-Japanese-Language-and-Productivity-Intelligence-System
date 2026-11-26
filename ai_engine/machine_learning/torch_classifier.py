"""KANAME custom PyTorch text classifier.

Unlike the scikit-learn LinearSVC pipeline in :mod:`classifier`, this module
trains a small fully-connected neural network (MLP) end-to-end with PyTorch on
the same self-contained Japanese sample corpus. It is a *custom-trained* model —
the weights are learned by gradient descent and cached to disk — rather than a
wrapper around an external API.

Pipeline:
    SudachiVectorizer (TF-IDF)  ->  dense feature matrix  ->  TextMLP
    TextMLP: Linear(d,128) -> ReLU -> Dropout -> Linear(128,64) -> ReLU -> Linear(64,C)

PyTorch is imported lazily so that the rest of KANAME keeps its fast startup
when the neural model is not used.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import sample_data as _sample_data
from .vectorizer import SudachiVectorizer

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
_MODEL_PATH = os.path.join(_MODEL_DIR, "torch_text_mlp.pt")
_VEC_PATH = os.path.join(_MODEL_DIR, "torch_text_mlp_vec.joblib")

_ENGINE = "PyTorch MLP (CPU)"

# Lazy PyTorch handle -------------------------------------------------------
_torch = None          # type: ignore
_nn = None             # type: ignore
_attempted = False
_import_error: Optional[str] = None


def _load_torch():
    """Import torch/ torch.nn on first use and cache the modules."""
    global _torch, _nn, _attempted, _import_error
    if _attempted:
        return _torch
    _attempted = True
    try:
        import torch as _t  # noqa: WPS433
        import torch.nn as _n  # noqa: WPS433
        _torch, _nn = _t, _n
    except Exception as e:  # pragma: no cover - environment dependent
        _import_error = f"PyTorch import failed: {e}"
        _torch = None
    return _torch


def is_available() -> bool:
    return _load_torch() is not None


def get_error_message() -> Optional[str]:
    _load_torch()
    return _import_error


# ---------------------------------------------------------------------------
# Prediction container
# ---------------------------------------------------------------------------
@dataclass
class TorchPrediction:
    text: str
    predicted_label: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    top_k: List[Dict[str, Any]] = field(default_factory=list)
    engine: str = _ENGINE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "predicted_label": self.predicted_label,
            "confidence": round(float(self.confidence), 4),
            "probabilities": {k: round(float(v), 4) for k, v in self.probabilities.items()},
            "top_k": [
                {"label": it["label"], "score": round(float(it["score"]), 4), "rank": int(it["rank"])}
                for it in self.top_k
            ],
            "engine": self.engine,
        }


@dataclass
class _TrainedModel:
    model: Any
    vectorizer: SudachiVectorizer
    classes: List[str]
    config: Dict[str, Any]
    meta: Dict[str, Any]


_STATE: Optional[_TrainedModel] = None


def _default_config(input_dim: int, num_classes: int) -> Dict[str, Any]:
    return {
        "input_dim": int(input_dim),
        "hidden1": 128,
        "hidden2": 64,
        "num_classes": int(num_classes),
        "dropout": 0.2,
        "epochs": 300,
        "lr": 1e-3,
        "seed": 42,
        "max_features": 500,
    }


def _build_mlp(config: Dict[str, Any]):
    nn = _nn
    return nn.Sequential(
        nn.Linear(config["input_dim"], config["hidden1"]),
        nn.ReLU(),
        nn.Dropout(config["dropout"]),
        nn.Linear(config["hidden1"], config["hidden2"]),
        nn.ReLU(),
        nn.Linear(config["hidden2"], config["num_classes"]),
    )


def _save_state(state: _TrainedModel) -> None:
    torch = _torch
    os.makedirs(_MODEL_DIR, exist_ok=True)
    import joblib  # noqa: WPS433

    torch.save(
        {
            "state_dict": state.model.state_dict(),
            "config": state.config,
            "classes": state.classes,
            "meta": state.meta,
        },
        _MODEL_PATH,
    )
    joblib.dump(state.vectorizer, _VEC_PATH)


def _load_state() -> Optional[_TrainedModel]:
    torch = _load_torch()
    if torch is None:
        return None
    if not (os.path.exists(_MODEL_PATH) and os.path.exists(_VEC_PATH)):
        return None
    try:
        import joblib  # noqa: WPS433

        vectorizer = joblib.load(_VEC_PATH)
        try:
            ckpt = torch.load(_MODEL_PATH, map_location="cpu", weights_only=True)
        except Exception:
            # Checkpoint metadata (e.g. TorchVersion) is not on the weights-only
            # allowlist — the file is our own local artifact, so load it fully.
            ckpt = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)
        config = ckpt["config"]
        model = _build_mlp(config)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return _TrainedModel(
            model=model,
            vectorizer=vectorizer,
            classes=list(ckpt["classes"]),
            config=config,
            meta=dict(ckpt.get("meta", {})),
        )
    except Exception:
        return None


def train_model(
    epochs: Optional[int] = None,
    hidden1: int = 128,
    hidden2: int = 64,
    dropout: float = 0.2,
    lr: float = 1e-3,
    seed: int = 42,
    save: bool = True,
) -> Dict[str, Any]:
    """Train the MLP on the built-in Japanese corpus and return training metadata."""
    torch = _load_torch()
    if torch is None:
        raise RuntimeError(get_error_message() or "PyTorch is unavailable.")

    texts, labels = _sample_data.get_labeled_texts()
    classes = sorted(set(labels))
    label_to_idx = {c: i for i, c in enumerate(classes)}

    vectorizer = SudachiVectorizer(mode="C", min_df=1, max_features=500, ngram_range=(1, 1))
    X = vectorizer.fit_transform(texts).toarray().astype(np.float32)
    y = np.array([label_to_idx[label] for label in labels], dtype=np.int64)

    config = _default_config(X.shape[1], len(classes))
    config.update({
        "hidden1": int(hidden1),
        "hidden2": int(hidden2),
        "dropout": float(dropout),
        "epochs": int(epochs or config["epochs"]),
        "lr": float(lr),
        "seed": int(seed),
    })

    torch.manual_seed(config["seed"])
    model = _build_mlp(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = _nn.CrossEntropyLoss()

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    model.train()
    final_loss = 0.0
    for _ in range(config["epochs"]):
        optimizer.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    model.eval()
    with torch.no_grad():
        train_acc = float((model(X_t).argmax(dim=1) == y_t).float().mean().item())

    meta = {
        "train_accuracy": round(train_acc, 4),
        "final_loss": round(final_loss, 6),
        "num_training_samples": len(texts),
        "num_features": int(X.shape[1]),
        "torch_version": getattr(torch, "__version__", "unknown"),
        "device": "cpu",
    }

    state = _TrainedModel(model=model, vectorizer=vectorizer, classes=classes, config=config, meta=meta)
    global _STATE
    _STATE = state
    if save:
        _save_state(state)

    return model_info()


def _ensure_state() -> _TrainedModel:
    global _STATE
    if _STATE is not None:
        return _STATE
    if not is_available():
        raise RuntimeError(get_error_message() or "PyTorch is unavailable.")
    loaded = _load_state()
    if loaded is not None:
        _STATE = loaded
        return _STATE
    train_model()
    assert _STATE is not None
    return _STATE


def classify_text(text: str, top_k: int = 3) -> TorchPrediction:
    if not isinstance(text, str) or not text.strip():
        return TorchPrediction(text=text or "", predicted_label="unknown", confidence=0.0)

    torch = _load_torch()
    state = _ensure_state()
    x = state.vectorizer.transform([text]).toarray().astype(np.float32)
    with torch.no_grad():
        logits = state.model(torch.from_numpy(x))
        probs = torch.softmax(logits, dim=1)[0].numpy()

    order = np.argsort(probs)[::-1]
    label_order = [state.classes[i] for i in order]
    score_order = [float(probs[i]) for i in order]

    return TorchPrediction(
        text=text,
        predicted_label=label_order[0],
        confidence=score_order[0],
        probabilities={label_order[i]: score_order[i] for i in range(len(state.classes))},
        top_k=[
            {"label": label_order[i], "score": score_order[i], "rank": i + 1}
            for i in range(min(int(top_k), len(state.classes)))
        ],
    )


def classify_batch(texts: List[str], top_k: int = 3) -> List[TorchPrediction]:
    return [classify_text(t, top_k=top_k) for t in texts]


def model_info() -> Dict[str, Any]:
    """Return architecture / training metadata, training the model if needed."""
    if not is_available():
        return {"available": False, "engine": _ENGINE, "error": get_error_message()}
    state = _ensure_state()
    return {
        "available": True,
        "engine": _ENGINE,
        "torch_version": state.meta.get("torch_version", "unknown"),
        "device": state.meta.get("device", "cpu"),
        "classes": list(state.classes),
        "num_classes": len(state.classes),
        "num_features": state.meta.get("num_features", state.config.get("input_dim")),
        "num_training_samples": state.meta.get("num_training_samples"),
        "train_accuracy": state.meta.get("train_accuracy"),
        "final_loss": state.meta.get("final_loss"),
        "epochs": state.config.get("epochs"),
        "architecture": "Linear(d,128) -> ReLU -> Dropout -> Linear(128,64) -> ReLU -> Linear(64,C)",
        "optimizer": f"Adam(lr={state.config.get('lr')})",
        "vectorizer": "SudachiVectorizer (TF-IDF)",
    }
