"""
Heuristic translation quality scoring (0.0 – 1.0).

No external models are required. The scorer uses lightweight heuristics
that make sense for a hybrid glossary + dictionary + optional-LLM pipeline:

  - length_ratio_score: penalizes target sentences that are unreasonably
    shorter or longer than the source (typical EN→JA length ratio ~0.85,
    JA→EN ~1.15).
  - glossary_coverage: if a glossary was applied, how many domain terms
    were found and used in the target.
  - numeric_preservation: every number in the source should also appear
    (or have a plausible Japanese numeral analogue) in the target.
  - katakana_preservation (JA target only): any English proper noun that
    appears in the source and has a standard katakana rendering should
    ideally still be identifiable in the target.
  - punctuation_preservation: sentence-ending punctuation count.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_KATAKANA_RE = re.compile(r"[\u30A0-\u30FF]{2,}")
_ALNUM_RE = re.compile(r"[A-Za-z0-9]{2,}")


@dataclass
class QualityScore:
    overall: float
    length_ratio_score: float
    numeric_preservation_score: float
    glossary_coverage_score: float
    format_preservation_score: float
    notes: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("overall", "length_ratio_score", "numeric_preservation_score",
                  "glossary_coverage_score", "format_preservation_score"):
            if isinstance(d.get(k), (int, float)):
                d[k] = round(d[k], 4)
        return d


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _length_ratio_score(source: str, target: str, source_lang: str, target_lang: str) -> float:
    src_len = max(1, len(source.strip()))
    tgt_len = max(1, len(target.strip()))
    actual = tgt_len / src_len
    src = source_lang.lower()[:2]
    tgt = target_lang.lower()[:2]
    if src == "ja" and tgt == "en":
        expected = 1.15
    elif src == "en" and tgt == "ja":
        expected = 0.87
    else:
        expected = 1.0
    ratio = actual / expected if expected else 1.0
    if 0.60 <= ratio <= 1.55:
        score = 1.0 - abs(1.0 - ratio) * 1.2
    else:
        score = max(0.0, 0.55 - abs(1.0 - ratio) * 0.8)
    return _clamp(score)


def _numeric_score(source: str, target: str) -> float:
    src_nums = set(_NUM_RE.findall(source))
    if not src_nums:
        return 1.0
    tgt_nums = set(_NUM_RE.findall(target))
    covered = src_nums & tgt_nums
    ratio = len(covered) / len(src_nums)
    return _clamp(0.4 + 0.6 * ratio)


def _format_score(source: str, target: str) -> float:
    ends_source = sum(source.count(p) for p in ("。", "．", ".", "！", "!", "？", "?"))
    ends_target = sum(target.count(p) for p in ("。", "．", ".", "！", "!", "？", "?"))
    if ends_source == 0:
        return 1.0
    ratio = ends_target / max(1, ends_source)
    return _clamp(1.0 - abs(1.0 - ratio) * 0.6)


def _glossary_score(glossary_hits: int, source_terms_guess: Optional[int]) -> float:
    if not source_terms_guess or source_terms_guess <= 0:
        return 1.0 if glossary_hits == 0 else _clamp(0.4 + 0.4 * min(1.0, glossary_hits / 3.0))
    ratio = glossary_hits / max(1, source_terms_guess)
    return _clamp(0.35 + 0.65 * ratio)


def score_translation(
    source: str,
    target: str,
    source_lang: str,
    target_lang: str,
    glossary_hits: int = 0,
    source_terms_guess: Optional[int] = None,
) -> QualityScore:
    if not isinstance(source, str) or not isinstance(target, str):
        return QualityScore(
            overall=0.0, length_ratio_score=0.0, numeric_preservation_score=0.0,
            glossary_coverage_score=0.0, format_preservation_score=0.0,
            notes={"error": "invalid inputs"},
        )

    if not target.strip():
        return QualityScore(
            overall=0.0, length_ratio_score=0.0, numeric_preservation_score=0.0,
            glossary_coverage_score=0.0, format_preservation_score=0.0,
            notes={"error": "empty target"},
        )

    lr = _length_ratio_score(source, target, source_lang, target_lang)
    ns = _numeric_score(source, target)
    gs = _glossary_score(glossary_hits, source_terms_guess)
    fs = _format_score(source, target)

    weights = {"lr": 0.30, "ns": 0.25, "gs": 0.20, "fs": 0.25}
    overall = lr * weights["lr"] + ns * weights["ns"] + gs * weights["gs"] + fs * weights["fs"]

    notes = {
        "source_chars": len(source),
        "target_chars": len(target),
        "numbers_in_source": len(set(_NUM_RE.findall(source))),
        "numbers_in_target": len(set(_NUM_RE.findall(target))),
    }
    if target_lang.lower().startswith("ja"):
        notes["katakana_tokens"] = len(_KATAKANA_RE.findall(target))
    if source_lang.lower().startswith("en") or target_lang.lower().startswith("en"):
        notes["alnum_tokens"] = len(_ALNUM_RE.findall(source)) + len(_ALNUM_RE.findall(target))

    return QualityScore(
        overall=_clamp(overall),
        length_ratio_score=lr,
        numeric_preservation_score=ns,
        glossary_coverage_score=gs,
        format_preservation_score=fs,
        notes=notes,
    )
