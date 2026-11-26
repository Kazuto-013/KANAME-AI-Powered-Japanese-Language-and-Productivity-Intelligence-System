"""Small CPU PyTorch Transformer text classifier for KANAME's Japanese demo corpus."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import sample_data
from ai_engine.japanese_nlp.sudachi_engine import tokenize

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "torch_text_transformer.pt")
_torch = None
_nn = None
_attempted = False
_error: str | None = None
_STATE: "_State | None" = None


def _load_torch():
    global _torch, _nn, _attempted, _error
    if _attempted:
        return _torch
    _attempted = True
    try:
        import torch
        import torch.nn as nn
        _torch, _nn = torch, nn
    except Exception as exc:
        _error = str(exc)
    return _torch


def is_available() -> bool:
    return _load_torch() is not None


def _tokens(text: str) -> list[str]:
    return [
        item.dictionary_form or item.surface
        for item in tokenize(text, mode="C").tokens
        if item.pos_main in ("名詞", "動詞", "形容詞", "副詞")
    ]


def _build_model(config: dict[str, Any]):
    nn = _nn

    class TextTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(config["vocab_size"], config["dimension"], padding_idx=0)
            self.position = nn.Embedding(config["max_length"], config["dimension"])
            layer = nn.TransformerEncoderLayer(
                d_model=config["dimension"],
                nhead=config["heads"],
                dim_feedforward=config["feedforward"],
                dropout=config["dropout"],
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(
                layer,
                num_layers=config["layers"],
                enable_nested_tensor=False,
            )
            self.classifier = nn.Linear(config["dimension"], config["num_classes"])

        def forward(self, ids):
            positions = _torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
            mask = ids.eq(0)
            hidden = self.embedding(ids) + self.position(positions)
            hidden = self.encoder(hidden, src_key_padding_mask=mask)
            keep = (~mask).unsqueeze(-1)
            pooled = (hidden * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)
            return self.classifier(pooled)

    return TextTransformer()


@dataclass
class _State:
    model: Any
    vocabulary: dict[str, int]
    classes: list[str]
    config: dict[str, Any]
    meta: dict[str, Any]


def _encode(texts: list[str], vocabulary: dict[str, int], max_length: int) -> np.ndarray:
    result = np.zeros((len(texts), max_length), dtype=np.int64)
    for row, text in enumerate(texts):
        ids = [vocabulary.get(token, 1) for token in _tokens(text)[:max_length]]
        if not ids and text.strip():
            ids = [1]
        if ids:
            result[row, :len(ids)] = ids
    return result


def train_model(epochs: int = 100, seed: int = 42, save: bool = True) -> dict[str, Any]:
    torch = _load_torch()
    if torch is None:
        raise RuntimeError(_error or "PyTorch is unavailable.")
    texts, labels = sample_data.get_labeled_texts()
    vocabulary = {"<pad>": 0, "<unk>": 1}
    for text in texts:
        for token in _tokens(text):
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)
    classes = sorted(set(labels))
    config = {
        "vocab_size": len(vocabulary),
        "num_classes": len(classes),
        "dimension": 64,
        "heads": 4,
        "layers": 2,
        "feedforward": 128,
        "dropout": 0.1,
        "max_length": 64,
        "epochs": max(1, min(int(epochs), 2000)),
        "seed": int(seed),
    }
    torch.manual_seed(config["seed"])
    model = _build_model(config)
    x = torch.from_numpy(_encode(texts, vocabulary, config["max_length"]))
    y = torch.tensor([classes.index(label) for label in labels], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = _nn.CrossEntropyLoss()
    model.train()
    final_loss = 0.0
    for _ in range(config["epochs"]):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    model.eval()
    with torch.no_grad():
        accuracy = float((model(x).argmax(dim=1) == y).float().mean().item())
    meta = {
        "train_accuracy": round(accuracy, 4),
        "final_loss": round(final_loss, 6),
        "training_samples": len(texts),
        "device": "cpu",
    }
    global _STATE
    _STATE = _State(model, vocabulary, classes, config, meta)
    if save:
        os.makedirs(os.path.dirname(_MODEL_PATH), exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "classes": classes,
            "config": config,
            "meta": meta,
        }, _MODEL_PATH)
    return model_info()


def _ensure_state() -> _State:
    global _STATE
    if _STATE is not None:
        return _STATE
    torch = _load_torch()
    if torch is None:
        raise RuntimeError(_error or "PyTorch is unavailable.")
    if os.path.exists(_MODEL_PATH):
        try:
            data = torch.load(_MODEL_PATH, map_location="cpu", weights_only=True)
            model = _build_model(data["config"])
            model.load_state_dict(data["state_dict"])
            model.eval()
            _STATE = _State(model, data["vocabulary"], data["classes"], data["config"], data["meta"])
            return _STATE
        except Exception:
            pass
    train_model()
    assert _STATE is not None
    return _STATE


def classify_text(text: str, top_k: int = 3) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        return {"text": text or "", "predicted_label": "unknown", "confidence": 0.0, "probabilities": {}, "top_k": []}
    state = _ensure_state()
    x = _torch.from_numpy(_encode([text], state.vocabulary, state.config["max_length"]))
    with _torch.no_grad():
        probs = _torch.softmax(state.model(x), dim=1)[0].numpy()
    order = np.argsort(probs)[::-1]
    return {
        "text": text,
        "predicted_label": state.classes[int(order[0])],
        "confidence": float(probs[order[0]]),
        "probabilities": {state.classes[i]: float(probs[i]) for i in range(len(state.classes))},
        "top_k": [
            {"label": state.classes[int(i)], "score": float(probs[i]), "rank": rank + 1}
            for rank, i in enumerate(order[:max(1, min(int(top_k), len(state.classes)))])
        ],
    }


def model_info() -> dict[str, Any]:
    state = _ensure_state()
    return {
        "available": True,
        "engine": "PyTorch TransformerEncoder (CPU)",
        "architecture": "Embedding + 2-layer TransformerEncoder + mean pooling + classifier",
        "classes": list(state.classes),
        "num_classes": len(state.classes),
        "vocabulary_size": len(state.vocabulary),
        "training_samples": state.meta["training_samples"],
        "train_accuracy": state.meta["train_accuracy"],
        "final_loss": state.meta["final_loss"],
        "epochs": state.config["epochs"],
        "device": "cpu",
        "quality_note": "Training accuracy on the 25-example demonstration corpus is not a generalization metric.",
    }
