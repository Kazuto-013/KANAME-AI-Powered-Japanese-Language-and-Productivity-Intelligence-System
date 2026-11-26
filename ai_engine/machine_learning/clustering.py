"""
Japanese text clustering using KMeans on top of Sudachi TF-IDF vectors.

For a given list of input texts, this module:
  1. Vectorizes them via SudachiVectorizer
  2. Runs KMeans with k = min(desired_k, n_texts - 1 or 1)
  3. Computes per-cluster top-TF-IDF keywords and returns:
       - cluster assignments
       - per-cluster size, top keywords, example texts
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer

from . import sample_data as _sample_data
from .vectorizer import SudachiVectorizer


@dataclass
class ClusterInfo:
    cluster_id: int
    size: int
    top_keywords: List[str]
    example_texts: List[str] = field(default_factory=list)
    centroid_norm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClusteringResult:
    n_clusters: int
    labels: List[int]
    clusters: List[ClusterInfo]
    vectorizer_feature_count: int = 0
    total_texts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_clusters": self.n_clusters,
            "total_texts": self.total_texts,
            "vectorizer_feature_count": self.vectorizer_feature_count,
            "labels": list(self.labels),
            "clusters": [c.to_dict() for c in self.clusters],
        }


def _vectorize(texts: List[str], ngram_range=(1, 1), max_features=2000):
    vec = SudachiVectorizer(mode="C", min_df=1, max_features=max_features, ngram_range=ngram_range)
    counts = vec.build_analyzer()
    # Use TfidfTransformer to avoid double-tfidf since SudachiVectorizer is already TF-IDF
    # but we want raw counts for better KMeans separability, so re-train with counts.
    # For simplicity, re-use SudachiVectorizer and let TfidfTransformer handle re-weighting.
    X = vec.fit_transform(texts)
    feature_names = vec.get_feature_names_out_safe()
    return X, feature_names, vec


def cluster_texts(
    texts: List[str],
    n_clusters: Optional[int] = None,
    top_keywords: int = 6,
    n_examples: int = 2,
    random_state: int = 42,
) -> ClusteringResult:
    if not texts:
        return ClusteringResult(n_clusters=0, labels=[], clusters=[])

    safe_texts = [t if isinstance(t, str) and t.strip() else "" for t in texts]
    non_empty = [t for t in safe_texts if t]
    if len(non_empty) <= 1:
        only_cluster = ClusterInfo(
            cluster_id=0, size=len(safe_texts), top_keywords=[],
            example_texts=safe_texts[:n_examples],
        )
        return ClusteringResult(
            n_clusters=1, labels=[0] * len(safe_texts), clusters=[only_cluster],
            total_texts=len(safe_texts),
        )

    if n_clusters is None:
        n_clusters = max(2, min(5, max(1, len(non_empty) // 3)))
    n_clusters = int(max(1, min(n_clusters, len(non_empty) - 1 if len(non_empty) > 1 else 1)))

    X, feature_names, _vec = _vectorize(non_empty)
    if X.shape[1] == 0:
        only_cluster = ClusterInfo(
            cluster_id=0, size=len(safe_texts), top_keywords=[],
            example_texts=safe_texts[:n_examples],
        )
        return ClusteringResult(
            n_clusters=1, labels=[0] * len(safe_texts), clusters=[only_cluster],
            total_texts=len(safe_texts),
        )

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    labels_nonempty = km.fit_predict(X).tolist()

    # Map non-empty indices back to original list (skipping empties as cluster -1 → 0 fallback)
    labels: List[int] = []
    ne_idx = 0
    for t in safe_texts:
        if t:
            labels.append(labels_nonempty[ne_idx])
            ne_idx += 1
        else:
            labels.append(0)

    centers = km.cluster_centers_
    feature_arr = np.array(feature_names) if feature_names else np.array([])

    cluster_infos: List[ClusterInfo] = []
    for cid in range(n_clusters):
        indices = [i for i, l in enumerate(labels) if l == cid]
        size = len(indices)
        examples = [safe_texts[i] for i in indices[:n_examples] if safe_texts[i]]
        top_kws: List[str] = []
        centroid_norm = 0.0
        if feature_arr.size > 0:
            order = np.argsort(centers[cid])[::-1]
            top_idx = order[:min(top_keywords, len(order))]
            top_kws = [str(feature_arr[i]) for i in top_idx if i < feature_arr.size]
            centroid_norm = float(np.linalg.norm(centers[cid]))
        cluster_infos.append(ClusterInfo(
            cluster_id=cid, size=size, top_keywords=top_kws,
            example_texts=examples, centroid_norm=centroid_norm,
        ))

    return ClusteringResult(
        n_clusters=n_clusters, labels=labels, clusters=cluster_infos,
        vectorizer_feature_count=X.shape[1], total_texts=len(safe_texts),
    )


def clustering_demo(n_clusters: int = 3) -> ClusteringResult:
    """Run clustering on built-in sample data for a quick smoke demo."""
    all_texts: List[str] = []
    for sentences in _sample_data.TRAIN_CATEGORIES.values():
        all_texts.extend(sentences)
    return cluster_texts(all_texts, n_clusters=n_clusters)
