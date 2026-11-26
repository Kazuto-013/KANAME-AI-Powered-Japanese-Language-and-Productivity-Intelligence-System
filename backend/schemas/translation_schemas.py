from __future__ import annotations

from typing import List, Dict, Any, Optional, Literal

from pydantic import BaseModel, Field


SourceLang = Literal["auto", "ja", "en"]
TargetLang = Literal["en", "ja"]


class CustomGlossaryTerm(BaseModel):
    ja: str = Field(..., min_length=1, max_length=100)
    en: str = Field(..., min_length=1, max_length=200)


class TranslationExample(BaseModel):
    source: str = Field(..., min_length=1, max_length=2000)
    target: str = Field(..., min_length=1, max_length=2000)


class QualityBreakdownSchema(BaseModel):
    overall: float
    length_ratio_score: float
    numeric_preservation_score: float
    glossary_coverage_score: float
    format_preservation_score: float
    notes: Dict[str, Any]


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to translate")
    source_lang: SourceLang = Field("auto", description="Source language: auto / ja / en")
    target_lang: TargetLang = Field("en", description="Target language: en or ja")
    domain: Optional[str] = Field(None, description="Optional domain hint for glossary: business_general / technology_ai / finance_accounting / legal_contract")
    custom_glossary: List[CustomGlossaryTerm] = Field(default_factory=list, max_length=100)
    examples: List[TranslationExample] = Field(default_factory=list, max_length=10)


class TranslateResponse(BaseModel):
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    engine: str
    glossary_hits: int
    domain: Optional[str] = None
    quality: QualityBreakdownSchema
    notes: Dict[str, Any] = Field(default_factory=dict)


class TranslateBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1)
    source_lang: SourceLang = "auto"
    target_lang: TargetLang = "en"
    domain: Optional[str] = None
    custom_glossary: List[CustomGlossaryTerm] = Field(default_factory=list, max_length=100)
    examples: List[TranslationExample] = Field(default_factory=list, max_length=10)


class TranslateBatchResponse(BaseModel):
    results: List[TranslateResponse]
    count: int


class GlossaryListResponse(BaseModel):
    domains: List[str]


class GlossaryDomainResponse(BaseModel):
    domain: str
    pairs: List[Dict[str, str]]
    count: int


class TranslationHealthResponse(BaseModel):
    status: str
    active_engine: str
    configured_keys: List[str]
    fallback_available: bool
    domains: List[str]
