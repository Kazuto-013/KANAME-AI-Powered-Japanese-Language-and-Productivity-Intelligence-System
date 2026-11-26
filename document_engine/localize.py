"""Localize document input data before generation.

KANAME translates strictly between Japanese and English (JA <-> EN). Japanese
official-document templates (resume, career history, invoice, business report,
cover letter, ...) expect Japanese values, so when a user supplies English
free-text fields we translate them to Japanese first and only then generate the
document.

The walker is driven by the Pydantic request schema so it knows which fields
are free text (translatable) vs. enum / identifier / date / contact values
(never translated). It returns a new data dict plus a report describing exactly
which fields were translated, which is surfaced back to the caller.
"""
from __future__ import annotations

import re
import typing
from typing import Any, Dict, List, Tuple, Type

from pydantic import BaseModel

from ai_engine.translation import translate_text

try:  # Python 3.10+: `str | None` produces types.UnionType
    import types as _pytypes

    _UNION_ORIGINS = {typing.Union, _pytypes.UnionType}
except Exception:  # pragma: no cover
    _UNION_ORIGINS = {typing.Union}


_JP_RE = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]")
_ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")
_DATE_RE = re.compile(r"^\d{4}[-/]\d{1,2}([-/]\d{1,2})?$")
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")
_CONTACT_ONLY_RE = re.compile(r"^[\d\-\s+().]+$")
_URL_RE = re.compile(r"^(https?://|www\.)", re.IGNORECASE)

# Fields whose values are machine data / identifiers / enums: never translated.
_SKIP_FIELDS = frozenset({
    "doc_format", "photo_path", "source_filename", "dataset_id", "analysis_id",
    "invoice_number", "seller_tax_id", "bank_account_number", "bank_account_type",
    "phone", "email", "postal_code",
    "sender_phone", "sender_email", "sender_postal_code",
    "tech_stack", "corr_method", "iqr_k", "include_appendix",
})

# Fields that hold proper nouns (person / organisation / institution names).
# A JA <-> EN dictionary cannot transliterate proper nouns, so they are kept
# verbatim instead of being corrupted by a literal word-by-word translation.
_PROPER_NOUN_FIELDS = frozenset({
    "full_name_kanji", "full_name_kana", "issuer",
    "author", "recipient", "chairperson", "recorder",
    "bank_branch", "bank_account_holder",
})


def _is_proper_noun_field(field_name: str) -> bool:
    f = field_name.lower()
    return f == "name" or f.endswith("_name") or f in _PROPER_NOUN_FIELDS



def _is_literal(annotation: Any) -> bool:
    return typing.get_origin(annotation) is typing.Literal


def _unwrap_optional(annotation: Any) -> Any:
    origin = typing.get_origin(annotation)
    if origin in _UNION_ORIGINS:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _is_english_text(text: str) -> bool:
    if _JP_RE.search(text):
        return False
    letters = len(_ASCII_ALPHA_RE.findall(text))
    if letters == 0:
        return False
    nonspace = len(text.replace(" ", ""))
    return letters >= max(1, int(nonspace * 0.4))


def _is_japanese_text(text: str) -> bool:
    return bool(_JP_RE.search(text))


def _should_skip_string(text: str, field_name: str) -> bool:
    f = field_name.lower()
    if f.endswith("_date") or f.endswith("_time") or f in _SKIP_FIELDS:
        return True
    if _is_proper_noun_field(f):
        return True
    if "@" in text and " " not in text:
        return True
    if _URL_RE.match(text):
        return True
    if _DATE_RE.match(text) or _TIME_RE.match(text) or _CONTACT_ONLY_RE.match(text):
        return True
    return False


def _needs_translation(text: str, target_lang: str) -> bool:
    if target_lang == "ja":
        return _is_english_text(text)
    return _is_japanese_text(text)


def _maybe_translate(
    value: Any,
    field_name: str,
    target_lang: str,
    path: str,
    translated: List[Dict[str, Any]],
    cache: Dict[str, str],
) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or _should_skip_string(text, field_name):
        return value
    if not _needs_translation(text, target_lang):
        return value

    if text in cache:
        new_text = cache[text]
        engine = "cache"
    else:
        result = translate_text(text, source_lang="auto", target_lang=target_lang)
        new_text = (result.translated_text or "").strip()
        engine = result.engine
        cache[text] = new_text

    if not new_text or new_text == text:
        return value

    translated.append({
        "field": path or field_name,
        "source": text,
        "target": new_text,
        "engine": engine,
    })
    return new_text


def _walk_model(
    model_cls: Type[BaseModel],
    data: Dict[str, Any],
    target_lang: str,
    path: str,
    translated: List[Dict[str, Any]],
    cache: Dict[str, str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(data)
    for name, field in model_cls.model_fields.items():
        if name not in out:
            continue
        if _is_literal(field.annotation) or name in _SKIP_FIELDS:
            continue
        out[name] = _walk_value(
            out[name], field.annotation, target_lang,
            f"{path}.{name}" if path else name, translated, cache,
        )
    return out


def _walk_value(
    value: Any,
    annotation: Any,
    target_lang: str,
    path: str,
    translated: List[Dict[str, Any]],
    cache: Dict[str, str],
) -> Any:
    ann = _unwrap_optional(annotation)
    origin = typing.get_origin(ann)

    if isinstance(ann, type) and issubclass(ann, BaseModel) and isinstance(value, dict):
        return _walk_model(ann, value, target_lang, path, translated, cache)

    if origin in (list, List) and isinstance(value, list):
        args = typing.get_args(ann)
        inner = args[0] if args else Any
        return [
            _walk_value(item, inner, target_lang, f"{path}[{i}]", translated, cache)
            for i, item in enumerate(value)
        ]

    if origin is dict and isinstance(value, dict):
        args = typing.get_args(ann)
        val_ann = args[1] if len(args) == 2 else Any
        return {
            k: _walk_value(v, val_ann, target_lang, f"{path}.{k}", translated, cache)
            for k, v in value.items()
        }

    field_name = path.rsplit(".", 1)[-1].split("[")[0]
    return _maybe_translate(value, field_name, target_lang, path, translated, cache)


def localize_document_data(
    model_cls: Type[BaseModel],
    data: Dict[str, Any],
    target_lang: str = "ja",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Translate English free-text fields to Japanese (default) before generation.

    Returns ``(localized_data, report)`` where ``report`` lists the fields that
    were translated so the API can disclose what happened.
    """
    translated: List[Dict[str, Any]] = []
    cache: Dict[str, str] = {}
    localized = _walk_model(model_cls, data, target_lang, "", translated, cache)
    report = {
        "target_lang": target_lang,
        "translated_count": len(translated),
        "translated_fields": translated,
    }
    return localized, report
