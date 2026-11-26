"""Japanese topic discovery with an LDA model over Sudachi token counts."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .vectorizer import _sudachi_token_factory


def discover_topics(
    texts: list[str],
    n_topics: int = 5,
    top_words: int = 8,
    random_state: int = 42,
) -> dict[str, Any]:
    cleaned = [text.strip() if isinstance(text, str) else "" for text in texts]
    if len([text for text in cleaned if text]) < 2:
        raise ValueError("At least two non-empty texts are required.")
    valid_count = sum(bool(text) for text in cleaned)
    n_topics = max(1, min(int(n_topics), valid_count))
    top_words = max(1, min(int(top_words), 30))

    vectorizer = CountVectorizer(
        tokenizer=_sudachi_token_factory("C"),
        token_pattern=None,
        lowercase=False,
        min_df=1,
        max_features=4000,
    )
    matrix = vectorizer.fit_transform(cleaned)
    if matrix.shape[1] == 0:
        raise ValueError("No usable Japanese content words were found.")
    model = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=20,
        learning_method="batch",
        random_state=random_state,
    )
    distributions = model.fit_transform(matrix)
    names = vectorizer.get_feature_names_out().tolist()
    topics = []
    for topic_id, weights in enumerate(model.components_):
        best = np.argsort(weights)[::-1][:top_words]
        topics.append({
            "topic_id": topic_id,
            "keywords": [
                {"term": names[index], "weight": round(float(weights[index]), 6)}
                for index in best if index < len(names)
            ],
        })

    assignments = []
    for index, text in enumerate(cleaned):
        if not text:
            assignments.append({"text": text, "topic_id": None, "confidence": 0.0})
            continue
        topic_id = int(np.argmax(distributions[index]))
        assignments.append({
            "text": text,
            "topic_id": topic_id,
            "confidence": round(float(distributions[index][topic_id]), 6),
        })
    return {
        "method": "Latent Dirichlet Allocation over Sudachi token counts",
        "n_topics": n_topics,
        "vocabulary_size": int(matrix.shape[1]),
        "topics": topics,
        "assignments": assignments,
    }
