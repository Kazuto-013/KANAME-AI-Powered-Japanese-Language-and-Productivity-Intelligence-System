"""Safe interpretation of short voice commands; this module never executes actions."""
from __future__ import annotations

import re
from typing import Any


_ACTION_PATTERNS = (
    ("translate", re.compile(r"翻訳|訳して|訳してください|translate", re.I)),
    ("summarize", re.compile(r"要約|まとめて|summari[sz]e", re.I)),
    ("analyze_data", re.compile(r"データ.*分析|分析して|analy[sz]e data", re.I)),
    ("create_document", re.compile(r"文書|書類|レポート.*作成|作成して|create.*document", re.I)),
    ("classify_text", re.compile(r"分類|カテゴリ|classify", re.I)),
)


def parse_voice_command(text: str) -> dict[str, Any]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return {
            "text": cleaned,
            "intent": "other",
            "action": "unknown",
            "entities": {},
            "safe_to_execute": False,
        }

    from ai_engine.machine_learning.intent import detect_intent

    intent = detect_intent(cleaned)
    action = next(
        (name for name, pattern in _ACTION_PATTERNS if pattern.search(cleaned)),
        "unknown",
    )
    target_lang = None
    if re.search(r"英語|英訳|to english", cleaned, re.I):
        target_lang = "en"
    elif re.search(r"日本語|和訳|to japanese", cleaned, re.I):
        target_lang = "ja"
    return {
        "text": cleaned,
        "intent": intent.intent,
        "intent_confidence": intent.confidence,
        "action": action,
        "entities": {"target_language": target_lang} if target_lang else {},
        "safe_to_execute": False,
        "note": "Interpretation only; explicit user confirmation is required before execution.",
    }
