"""
Japanese NLP Package (SudachiPy-powered).

Provides:
  * Morphological analysis & tokenization
  * JLPT proficiency-level assessment for tokens and entire sentences
  * TF-IDF style keyword extraction
  * Dictionary/POS-based named-entity extraction
  * Rule-based sentiment analysis
  * 6 alternative sentence rewrites (polite / casual / kana_easy / romaji / synonym_swap / honorific)
"""
from __future__ import annotations

from .sudachi_engine import (
    # Core
    tokenize,
    analyze,
    TokenizeResult,
    AnalyzeResult,
    TokenInfo,
    # Keyword extraction
    extract_keywords,
    KeywordsResult,
    KeywordItem,
    # Named-entity extraction
    extract_named_entities,
    NamedEntitiesResult,
    NamedEntity,
    # Sentiment
    analyze_sentiment,
    SentimentResult,
    # JLPT Proficiency (NEW central feature)
    analyze_with_jlpt,
    JlptAnalyzeResult,
    jlpt_level_for,
    assess_proficiency,         # NEW: simple sentence-level proficiency
    ProficiencyAssessment,      # NEW: rich proficiency dataclass
    # Alternative rewrites (6 styles)
    generate_alternatives,
    AlternativeSentence,
    # Utility
    get_tokenizer,
)
from .supervised_ner import train_ner, predict_entities, model_info as ner_model_info

__all__ = [
    "tokenize",
    "analyze",
    "TokenizeResult",
    "AnalyzeResult",
    "TokenInfo",
    "extract_keywords",
    "KeywordsResult",
    "KeywordItem",
    "extract_named_entities",
    "NamedEntitiesResult",
    "NamedEntity",
    "analyze_sentiment",
    "SentimentResult",
    "analyze_with_jlpt",
    "JlptAnalyzeResult",
    "jlpt_level_for",
    "assess_proficiency",
    "ProficiencyAssessment",
    "generate_alternatives",
    "AlternativeSentence",
    "get_tokenizer",
    "train_ner",
    "predict_entities",
    "ner_model_info",
]
