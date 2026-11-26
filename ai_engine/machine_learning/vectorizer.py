"""
scikit-learn compatible vectorizer that uses Sudachi for Japanese tokenization.

This module is designed to be plugged directly into a `Pipeline`:

    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    pipeline = Pipeline([
        ("vec", SudachiVectorizer(mode="C", min_df=1, ngram_range=(1, 1))),
        ("clf", LinearSVC()),
    ])

Under the hood, the vectorizer wraps `sklearn.feature_extraction.text.TfidfVectorizer`
with a custom `tokenizer=` callable that runs Sudachi tokenization and keeps
only nouns / verbs / adjectives / adverbs / proper nouns after stopword + length filtering.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


_VECTORIZER_STOPWORDS = frozenset({
    "の", "に", "は", "を", "た", "が", "で", "て", "と", "し",
    "れ", "さ", "ある", "いる", "も", "する", "から", "な", "こと",
    "として", "い", "や", "れる", "など", "なっ", "なり", "ます",
    "ん", "なら", "によって", "により", "おり", "より", "による",
    "ず", "ながら", "ので", "のみ", "ため", "これ", "それ", "あれ",
    "この", "その", "あの", "どの", "私", "あなた", "たち", "自分",
    "もの", "事", "方", "何", "どこ", "いつ", "だれ", "どれ", "どう",
    "すべて", "全部", "部分", "ただ", "そして", "しかし", "けれど",
    "です", "だ", "である", "また", "さらに", "特に", "主に", "ついて",
    "一", "二", "三", "四", "五", "六", "七", "八", "九", "十",
    "円", "月", "日", "年", "回", "目", "的", "性", "化", "者", "達",
    "今日", "明日", "昨日", "毎日", "毎週", "毎月", "毎年", "今",
})


_KEEP_POS = frozenset({"名詞", "動詞", "形容詞", "副詞", "連体詞", "感動詞"})


_ALNUM_RE = re.compile(r"^[A-Za-z0-9]+$")


def _sudachi_token_factory(mode: str = "C"):
    from ai_engine.japanese_nlp import sudachi_engine

    tokenizer_instance = sudachi_engine.get_tokenizer(mode)

    def _tokenize(text: str) -> List[str]:
        if not isinstance(text, str) or not text.strip():
            return []
        mlist = tokenizer_instance.tokenize(text)
        out: List[str] = []
        for morph in mlist:
            pos = list(morph.part_of_speech())
            pos_main = pos[0] if pos else ""
            if pos_main not in _KEEP_POS:
                continue
            surface = morph.surface()
            normalized = morph.normalized_form() or surface
            dict_form = morph.dictionary_form() or normalized
            key = dict_form if len(dict_form) >= 2 else (normalized if len(normalized) >= 2 else surface)
            if len(key) < 2:
                continue
            if key in _VECTORIZER_STOPWORDS:
                continue
            if _ALNUM_RE.match(key) and len(key) < 3:
                continue
            if key.isdigit():
                continue
            out.append(key)
        return out

    return _tokenize


class SudachiVectorizer(TfidfVectorizer):
    """
    TfidfVectorizer subclass with a Sudachi-backed Japanese tokenizer.

    Accepts the same constructor arguments as TfidfVectorizer, but the
    `tokenizer` argument is always overwritten to use the Sudachi wrapper.
    """

    def __init__(
        self,
        mode: str = "C",
        *,
        stop_words: Optional[Iterable[str]] = None,
        ngram_range: Tuple[int, int] = (1, 1),
        max_df: float = 1.0,
        min_df: int = 1,
        max_features: Optional[int] = None,
        sublinear_tf: bool = True,
        lowercase: bool = False,
    ):
        self.mode = mode
        self._sudachi_tokenizer = _sudachi_token_factory(mode)

        merged_stopwords: Optional[List[str]] = None
        if stop_words is not None:
            merged_stopwords = list(_VECTORIZER_STOPWORDS) + list(stop_words)

        super().__init__(
            tokenizer=self._sudachi_tokenizer,
            stop_words=merged_stopwords,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            lowercase=lowercase,
            token_pattern=None,  # required when using a custom callable tokenizer in recent sklearn
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__() if hasattr(super(), "__sklearn_tags__") else None
        try:
            from sklearn.utils._tags import _safe_tags  # type: ignore
        except Exception:
            _safe_tags = None  # type: ignore
        return tags

    # -- Pickle support ----------------------------------------------------
    # The Sudachi tokenizer is a local closure, which cannot be serialized.
    # Drop it (and the copy stored by CountVectorizer) on serialize and rebuild
    # it from `mode` on deserialize so the fitted vectorizer can be cached.
    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = dict(self.__dict__)
        state.pop("_sudachi_tokenizer", None)
        state.pop("tokenizer", None)
        return state

    def __setstate__(self, state):
        try:
            super().__setstate__(state)
        except AttributeError:
            self.__dict__.update(state)
        self._sudachi_tokenizer = _sudachi_token_factory(self.mode)
        self.tokenizer = self._sudachi_tokenizer

    def get_feature_names_out_safe(self) -> List[str]:
        try:
            arr = self.get_feature_names_out()
        except Exception:
            try:
                arr = np.array(self.get_feature_names())
            except Exception:
                return []
        return [str(x) for x in arr.tolist()] if hasattr(arr, "tolist") else [str(x) for x in arr]
