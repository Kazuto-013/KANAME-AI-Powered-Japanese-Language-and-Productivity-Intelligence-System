"""
Deterministic translation evaluation harness (no LLM keys required).

Runs the *offline* dictionary/glossary pipeline (`force_fallback=True`) over a
fixed JA<->EN corpus and reports:

  - corpus BLEU (1-4 gram, character tokenization for JA, whitespace for EN)
  - average character-F1 between hypotheses and references
  - average heuristic quality score (from quality.score_translation)

Because it never touches the network it is safe to run in CI and lets us track
dictionary-engine regressions over time. Per-sentence rows are returned so the
weakest cases can be inspected and the heuristic dictionary refined.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from math import exp, log
from typing import Any, Dict, List, Sequence, Tuple

from .translator import translate_text


@dataclass
class EvalPair:
    source: str
    reference: str
    direction: str  # "ja-en" or "en-ja"


# Fixed, hand-written corpus. Keep sentences short and grounded in everyday
# business / technical Japanese so the deterministic engine has a fair chance.
CORPUS: List[EvalPair] = [
    # ---------------- JA -> EN ----------------
    EvalPair("こんにちは。", "Hello.", "ja-en"),
    EvalPair("ありがとうございます。", "Thank you.", "ja-en"),
    EvalPair("私は東京に住んでいます。", "I live in Tokyo.", "ja-en"),
    EvalPair("この資料を確認してください。", "Please check this document.", "ja-en"),
    EvalPair("お名前を教えてください。", "Please tell me your name.", "ja-en"),
    EvalPair("新しいシステムを開発しました。", "We developed a new system.", "ja-en"),
    EvalPair("データベースに接続できません。", "Cannot connect to the database.", "ja-en"),
    EvalPair("会議は明日の午後3時からです。", "The meeting is from 3 PM tomorrow.", "ja-en"),
    EvalPair("売上は前年比12%増加しました。", "Sales increased by 12% year over year.", "ja-en"),
    EvalPair("来週までに報告書を提出します。", "I will submit the report by next week.", "ja-en"),
    # ---------------- EN -> JA ----------------
    EvalPair("Hello.", "こんにちは。", "en-ja"),
    EvalPair("Thank you very much.", "ありがとうございます。", "en-ja"),
    EvalPair("I live in Tokyo.", "私は東京に住んでいます。", "en-ja"),
    EvalPair("Please check this document.", "この資料を確認してください。", "en-ja"),
    EvalPair("Please tell me your name.", "お名前を教えてください。", "en-ja"),
    EvalPair("We developed a new system.", "新しいシステムを開発しました。", "en-ja"),
    EvalPair("Cannot connect to the database.", "データベースに接続できません。", "en-ja"),
    EvalPair("The meeting starts at 3 PM.", "会議は午後3時に始まります。", "en-ja"),
    EvalPair("Sales increased by 12%.", "売上は12%増加しました。", "en-ja"),
    EvalPair("I will submit the report by next week.", "来週までに報告書を提出します。", "en-ja"),
]


def _tokenize(text: str, lang: str) -> List[str]:
    if lang == "ja":
        return [c for c in text if not c.isspace()]
    return text.lower().split()


def _ngrams(tokens: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _char_f1(reference: str, hypothesis: str) -> float:
    from collections import Counter

    ref = Counter(reference.replace(" ", ""))
    hyp = Counter(hypothesis.replace(" ", ""))
    if not ref and not hyp:
        return 1.0
    overlap = sum((ref & hyp).values())
    precision = overlap / max(1, sum(hyp.values()))
    recall = overlap / max(1, sum(ref.values()))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _corpus_bleu(refs: List[List[str]], hyps: List[List[str]], max_n: int = 4) -> Dict[str, Any]:
    """Add-1 smoothed corpus BLEU.

    Add-1 smoothing is used because the corpus is small (10 sentences per
    direction); plain BLEU collapses to 0 as soon as a single higher-order
    n-gram is missing. Returns the overall score plus per-order precision.
    """
    from collections import Counter

    matches = [0] * max_n
    totals = [0] * max_n
    ref_total = hyp_total = 0
    for ref, hyp in zip(refs, hyps):
        ref_total += len(ref)
        hyp_total += len(hyp)
        for n in range(1, max_n + 1):
            rng = Counter(_ngrams(ref, n))
            hng = _ngrams(hyp, n)
            totals[n - 1] += len(hng)
            for g in hng:
                if rng[g] > 0:
                    matches[n - 1] += 1
                    rng[g] -= 1

    # add-1 smoothing keeps every order non-zero so short corpora stay usable
    precisions = [(matches[i] + 1) / (totals[i] + 1) for i in range(max_n)]
    geo = exp(sum(log(p) for p in precisions) / max_n)
    bp = 1.0 if hyp_total > ref_total else exp(1 - ref_total / max(1, hyp_total))
    return {
        "bleu": bp * geo,
        "precisions": [round(p, 4) for p in precisions],
    }


def evaluate_translation(corpus: List[EvalPair] | None = None) -> Dict[str, Any]:
    """Run the offline dictionary pipeline over the corpus and score it."""
    pairs = corpus if corpus is not None else CORPUS
    rows: List[Dict[str, Any]] = []

    by_dir: Dict[str, List[int]] = {}
    for idx, pair in enumerate(pairs):
        src_lang, tgt_lang = pair.direction.split("-")
        result = translate_text(
            pair.source,
            source_lang=src_lang,
            target_lang=tgt_lang,
            force_fallback=True,
        )
        hyp = result.translated_text
        f1 = _char_f1(pair.reference, hyp)
        rows.append({
            "source": pair.source,
            "reference": pair.reference,
            "hypothesis": hyp,
            "direction": pair.direction,
            "engine": result.engine,
            "char_f1": round(f1, 4),
            "quality": round(result.quality_score.overall, 4),
        })
        by_dir.setdefault(pair.direction, []).append(idx)

    summary: Dict[str, Any] = {"num_pairs": len(pairs), "by_direction": {}}
    for direction, idxs in by_dir.items():
        src_lang, tgt_lang = direction.split("-")
        refs = [_tokenize(pairs[i].reference, tgt_lang) for i in idxs]
        hyps = [_tokenize(rows[i]["hypothesis"], tgt_lang) for i in idxs]
        bleu = _corpus_bleu(refs, hyps)
        summary["by_direction"][direction] = {
            "pairs": len(idxs),
            "bleu": round(bleu["bleu"], 4),
            "bleu_precisions": bleu["precisions"],
            "avg_char_f1": round(sum(rows[i]["char_f1"] for i in idxs) / len(idxs), 4),
            "avg_quality": round(sum(rows[i]["quality"] for i in idxs) / len(idxs), 4),
        }

    all_f1 = [r["char_f1"] for r in rows]
    all_q = [r["quality"] for r in rows]
    summary["overall"] = {
        "avg_char_f1": round(sum(all_f1) / len(all_f1), 4) if all_f1 else 0.0,
        "avg_quality": round(sum(all_q) / len(all_q), 4) if all_q else 0.0,
    }
    summary["rows"] = rows
    summary["rows_sorted_by_char_f1"] = sorted(rows, key=lambda r: r["char_f1"])
    return summary


if __name__ == "__main__":
    import json

    report = evaluate_translation()
    print(json.dumps(
        {k: v for k, v in report.items() if k != "rows"},
        ensure_ascii=False,
        indent=2,
    ))
    print("\nWeakest cases:")
    for r in report["rows_sorted_by_char_f1"][:8]:
        print(f"  [{r['direction']}] f1={r['char_f1']}  {r['source']!r} -> {r['hypothesis']!r}")
