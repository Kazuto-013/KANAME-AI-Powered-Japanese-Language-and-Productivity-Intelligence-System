from __future__ import annotations

from typing import List, Dict, Optional, Literal

from pydantic import BaseModel, Field


SplitMode = Literal["A", "B", "C"]


class TokenInfoSchema(BaseModel):
    surface: str
    pos: List[str]
    pos_main: str
    reading: str
    normalized_form: str
    dictionary_form: str
    start: int
    end: int
    jlpt_level: Optional[str] = None
    alternatives: Dict = Field(default_factory=dict)


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Japanese text to tokenize")
    mode: SplitMode = Field("C", description="Sudachi split mode: A (shortest), B (middle), C (longest, default)")


class TokenizeResponse(BaseModel):
    text: str
    mode: str
    token_count: int
    token_surfaces: List[str]
    tokens: List[TokenInfoSchema]


class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Japanese text to analyze")
    mode: SplitMode = Field("C", description="Sudachi split mode: A (shortest), B (middle), C (longest, default)")


class AnalyzeResponse(BaseModel):
    text: str
    mode: str
    token_count: int
    token_surfaces: List[str]
    wakati: str
    readings: List[str]
    kana_only_text: str
    pos_counts: Dict[str, int]
    noun_list: List[str]
    verb_list: List[str]
    adjective_list: List[str]
    tokens: List[TokenInfoSchema]


class NlpHealthResponse(BaseModel):
    status: str
    engine: str
    dictionary_loaded: bool
    default_mode: str


class KeywordItemSchema(BaseModel):
    keyword: str
    score: float
    frequency: int
    pos_hint: str
    jlpt_level: Optional[str] = None
    alternatives: Dict = Field(default_factory=dict)


class KeywordsRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Japanese text to extract keywords from")
    mode: SplitMode = Field("C", description="Sudachi split mode")
    top_n: int = Field(10, ge=1, le=50, description="Maximum number of keywords to return")


class KeywordsResponse(BaseModel):
    text: str
    mode: str
    top_n: int
    keyword_count: int
    keywords: List[KeywordItemSchema]


class NamedEntitySchema(BaseModel):
    text: str
    label: str
    start: int
    end: int
    reading: str = ""
    source: str


class NamedEntitiesRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to extract named entities from")
    mode: SplitMode = Field("C", description="Sudachi split mode")


class NamedEntitiesResponse(BaseModel):
    text: str
    mode: str
    method: str
    entity_count: int
    entities: List[NamedEntitySchema]


class NerEntityAnnotation(BaseModel):
    start: int = Field(..., ge=0)
    end: int = Field(..., gt=0)
    label: str = Field(..., min_length=1, max_length=64)


class NerTrainingExample(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    entities: List[NerEntityAnnotation] = Field(default_factory=list, max_length=200)


class NerTrainRequest(BaseModel):
    examples: List[NerTrainingExample] = Field(..., min_length=2, max_length=5000)
    mode: SplitMode = Field("C", description="Sudachi split mode used to align annotation offsets")
    save: bool = Field(True, description="Persist the trained model for subsequent requests")


class NerPredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=50000)
    mode: SplitMode = Field("C")


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Japanese text for sentiment analysis")
    mode: SplitMode = Field("C", description="Sudachi split mode")


class SentimentResponse(BaseModel):
    text: str
    mode: str
    score: float
    label: str
    confidence: float
    positive_matches: List[str]
    negative_matches: List[str]
    positive_jlpt: Dict[str, Optional[str]] = Field(default_factory=dict)
    negative_jlpt: Dict[str, Optional[str]] = Field(default_factory=dict)
    matches_alternatives: Dict[str, Dict] = Field(default_factory=dict)
    positive_count: int
    negative_count: int
    token_count: int


class JlptAnalyzeResponse(AnalyzeResponse):
    jlpt_level: str
    jlpt_score: float
    jlpt_breakdown: Dict[str, int]
    token_jlpt: List[Optional[str]]
    difficulty_label: str


class AlternativesRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Japanese sentence to rewrite in 6 styles")
    mode: SplitMode = Field("C", description="Sudachi split mode")


class AlternativeSchema(BaseModel):
    style: str
    text: str
    reason_jp: str
    reason_en: str


class AlternativesResponse(BaseModel):
    text: str
    mode: str
    styles: List[AlternativeSchema]
