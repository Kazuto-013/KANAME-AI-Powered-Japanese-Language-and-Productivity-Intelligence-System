import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_engine.translation.translator import translate_text
from ai_engine.translation.evaluation import evaluate_translation

JA_SENTENCES = [
    "私は毎朝公園を散歩します。",
    "会議は午後3時から始まります。",
    "この製品は高い品質です。",
]
EN_SENTENCES = [
    "I walk in the park every morning.",
    "The meeting starts at 3 PM.",
]


def test_ja_to_en_dictionary():
    print("=== test_ja_to_en_dictionary ===")
    for sentence in JA_SENTENCES:
        result = translate_text(sentence, source_lang="ja", target_lang="en", force_fallback=True)
        print(f"{sentence!r} -> {result.translated_text!r} (engine={result.engine})")
        assert result.engine == "dictionary", f"expected dictionary engine, got {result.engine}"
        assert isinstance(result.translated_text, str)
        assert result.translated_text.strip(), "translated text must be non-empty"
        assert result.quality_score is not None
        assert 0.0 <= result.quality_score.overall <= 1.0
    print("PASSED: JA->EN uses dictionary engine with non-empty output and quality score")


def test_ja_to_en_output_is_ascii_ish():
    print("=== test_ja_to_en_output_is_ascii_ish ===")
    result = translate_text("私は毎朝公園を散歩します。", source_lang="ja", target_lang="en", force_fallback=True)
    print("output:", repr(result.translated_text))
    assert result.translated_text.strip(), "translated text must be non-empty"
    assert any(ch.isascii() and ch.isalpha() for ch in result.translated_text), \
        f"expected ASCII letters in JA->EN output, got {result.translated_text!r}"
    print("PASSED: JA->EN output contains ASCII-ish text")


def test_en_to_ja_dictionary():
    print("=== test_en_to_ja_dictionary ===")
    for sentence in EN_SENTENCES:
        result = translate_text(sentence, source_lang="en", target_lang="ja", force_fallback=True)
        print(f"{sentence!r} -> {result.translated_text!r} (engine={result.engine})")
        assert result.engine == "dictionary"
        assert result.translated_text.strip(), "translated text must be non-empty"
        assert result.quality_score is not None
    print("PASSED: EN->JA uses dictionary engine with non-empty output")


def test_translation_result_to_dict_serializable():
    print("=== test_translation_result_to_dict_serializable ===")
    result = translate_text("私は毎朝公園を散歩します。", source_lang="ja", target_lang="en", force_fallback=True)
    data = result.to_dict()
    print("keys:", sorted(data.keys()))
    assert data["engine"] == "dictionary"
    assert "quality" in data
    assert "overall" in data["quality"]
    raw = json.dumps(data)
    assert raw
    print("PASSED: TranslationResult.to_dict is JSON-serializable")


def test_translation_evaluation_harness():
    print("=== test_translation_evaluation_harness ===")
    report = evaluate_translation()
    print("overall:", report["overall"])
    assert report["num_pairs"] > 0
    assert set(report["by_direction"].keys()) == {"ja-en", "en-ja"}
    for direction, stats in report["by_direction"].items():
        assert 0.0 <= stats["bleu"] <= 1.0, f"{direction} bleu out of range: {stats['bleu']}"
        assert 0.0 <= stats["avg_char_f1"] <= 1.0
        assert len(stats["bleu_precisions"]) == 4
    assert report["overall"]["avg_char_f1"] >= 0.5, \
        f"dictionary engine char-F1 regressed: {report['overall']['avg_char_f1']}"
    assert len(report["rows"]) == report["num_pairs"]
    print("PASSED: translation evaluation harness returns valid BLEU/char-F1 metrics")


if __name__ == "__main__":
    test_ja_to_en_dictionary()
    print()
    test_ja_to_en_output_is_ascii_ish()
    print()
    test_en_to_ja_dictionary()
    print()
    test_translation_result_to_dict_serializable()
    print()
    test_translation_evaluation_harness()
    print()
    print("ALL TESTS PASSED")
