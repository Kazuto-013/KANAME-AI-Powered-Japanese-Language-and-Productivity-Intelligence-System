import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine.japanese_nlp import (
    tokenize,
    analyze,
    extract_keywords,
    analyze_sentiment,
    analyze_with_jlpt,
    generate_alternatives,
)

SENTENCE = "私は毎朝公園を散歩します。"


def test_tokenize_returns_tokens():
    print("=== test_tokenize_returns_tokens ===")
    result = tokenize(SENTENCE)
    print("token_count:", result.token_count)
    print("surfaces:", result.token_surfaces)
    assert len(result.tokens) > 0, "tokenize should return at least one token"
    assert result.token_count == len(result.tokens)
    for t in result.tokens:
        assert isinstance(t.surface, str)
    print("PASSED: tokenize returns non-empty token list")


def test_analyze_pos_counts():
    print("=== test_analyze_pos_counts ===")
    result = analyze(SENTENCE)
    data = result.to_dict()
    print("pos_counts:", data["pos_counts"])
    print("token_count:", data["token_count"])
    assert isinstance(data, dict)
    assert data["token_count"] > 0, "analyze token_count should be > 0"
    assert len(data["pos_counts"]) > 0, "pos_counts should be non-empty"
    for key, value in data["pos_counts"].items():
        assert isinstance(key, str)
        assert isinstance(value, int) and value > 0
    print("PASSED: analyze to_dict has non-empty pos_counts and token_count>0")


def test_extract_keywords_non_empty():
    print("=== test_extract_keywords_non_empty ===")
    result = extract_keywords(SENTENCE)
    data = result.to_dict()
    print("keywords:", [k["keyword"] for k in data["keywords"]])
    assert len(result.keywords) > 0, "extract_keywords should return non-empty result"
    assert data["keyword_count"] == len(result.keywords)
    for item in result.keywords:
        assert isinstance(item.keyword, str) and item.keyword
        assert 0.0 <= item.score <= 1.0, f"score out of range: {item.score}"
    print("PASSED: extract_keywords returns non-empty keyword list")


def test_sentiment_valid_range():
    print("=== test_sentiment_valid_range ===")
    result = analyze_sentiment(SENTENCE)
    print("label:", result.label, "score:", result.score, "confidence:", result.confidence)
    assert result.label in ("positive", "negative", "neutral"), f"unexpected label {result.label}"
    assert -1.0 <= result.score <= 1.0, f"score out of range: {result.score}"
    assert 0.0 <= result.confidence <= 1.0, f"confidence out of range: {result.confidence}"
    print("PASSED: sentiment label valid, score/confidence in range")


def test_analyze_with_jlpt_level():
    print("=== test_analyze_with_jlpt_level ===")
    result = analyze_with_jlpt(SENTENCE)
    print("jlpt_level:", result.jlpt_level)
    print("difficulty_label:", result.difficulty_label)
    data = result.to_dict()
    print("breakdown:", data["jlpt_breakdown"])
    assert isinstance(result.jlpt_level, str) and result.jlpt_level, "JLPT level must be a non-empty string"
    assert result.jlpt_level in ("N1", "N2", "N3", "N4", "N5", "N/A"), f"unexpected JLPT level {result.jlpt_level}"
    assert data["token_count"] > 0
    print("PASSED: analyze_with_jlpt returns a JLPT level")


def test_generate_alternatives():
    print("=== test_generate_alternatives ===")
    alternatives = generate_alternatives(SENTENCE)
    print("count:", len(alternatives))
    print("styles:", [a["style"] for a in alternatives])
    assert len(alternatives) >= 1, "generate_alternatives should return >= 1 alternative"
    for alt in alternatives:
        assert isinstance(alt, dict)
        assert alt.get("style")
        assert alt.get("text")
    print("PASSED: generate_alternatives returns >= 1 alternative")


if __name__ == "__main__":
    test_tokenize_returns_tokens()
    print()
    test_analyze_pos_counts()
    print()
    test_extract_keywords_non_empty()
    print()
    test_sentiment_valid_range()
    print()
    test_analyze_with_jlpt_level()
    print()
    test_generate_alternatives()
    print()
    print("ALL TESTS PASSED")
