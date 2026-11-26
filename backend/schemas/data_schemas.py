from __future__ import annotations

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class DatasetSummary(BaseModel):
    rows: int
    columns: int
    column_names: List[str]
    dtypes: Dict[str, str]
    memory_bytes: int
    missing_total: int
    missing_by_column: Dict[str, int]


class DescriptiveStats(BaseModel):
    column: str
    count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    q25: Optional[float] = None
    q50: Optional[float] = None
    q75: Optional[float] = None
    max: Optional[float] = None
    unique: Optional[int] = None
    top: Optional[str] = None
    freq: Optional[int] = None
    is_numeric: bool


class DatasetAnalysisResponse(BaseModel):
    dataset_id: Optional[str] = None
    summary: DatasetSummary
    numeric_stats: List[DescriptiveStats] = Field(default_factory=list)
    categorical_stats: List[DescriptiveStats] = Field(default_factory=list)
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    datetime_columns: List[str] = Field(default_factory=list)
    sample_records: List[Dict[str, Any]] = Field(default_factory=list)


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    source_stored: bool = True
    rows: int
    columns: int
    columns_names: List[str] = Field(default_factory=list)
    message: str


class DatasetHealthResponse(BaseModel):
    status: str
    engine: str
    supported_formats: List[str]
