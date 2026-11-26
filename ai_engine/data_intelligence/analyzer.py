from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
import io
import json
import tempfile
import os
import math

import numpy as np
import pandas as pd

SUPPORTED_EXT = {".csv", ".xlsx", ".xls", ".json", ".tsv"}


@dataclass
class DescriptiveStatsDC:
    column: str
    count: int
    is_numeric: bool
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

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DatasetSummaryDC:
    rows: int
    columns: int
    column_names: List[str]
    dtypes: Dict[str, str]
    memory_bytes: int
    missing_total: int
    missing_by_column: Dict[str, int]

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DatasetAnalysisDC:
    summary: DatasetSummaryDC
    numeric_stats: List[DescriptiveStatsDC] = field(default_factory=list)
    categorical_stats: List[DescriptiveStatsDC] = field(default_factory=list)
    numeric_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    datetime_columns: List[str] = field(default_factory=list)
    sample_records: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "summary": self.summary.to_dict(),
            "numeric_stats": [s.to_dict() for s in self.numeric_stats],
            "categorical_stats": [s.to_dict() for s in self.categorical_stats],
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "sample_records": self.sample_records,
        }


def _read_from_bytes(data: bytes, filename: str) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXT:
        raise ValueError(
            f"Unsupported file extension '{ext}'. Supported: {sorted(SUPPORTED_EXT)}"
        )

    bio = io.BytesIO(data)
    if ext == ".csv":
        encodings = ["utf-8", "utf-8-sig", "shift-jis", "cp932", "latin-1"]
        last_err: Optional[Exception] = None
        for enc in encodings:
            try:
                bio.seek(0)
                return pd.read_csv(bio, encoding=enc)
            except Exception as e:
                    last_err = e
                    continue
        raise RuntimeError(f"Failed to read CSV with tried encodings {encodings}: {last_err}")
    if ext == ".tsv":
        return pd.read_csv(bio, sep="\t")
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(bio)
    if ext == ".json":
        bio.seek(0)
        raw = bio.read().decode("utf-8", errors="replace")
        try:
            return pd.read_json(io.StringIO(raw))
        except Exception:
            return pd.read_json(io.StringIO(raw), lines=True)
    raise ValueError(f"Unsupported extension: {ext}")


def load_dataframe_from_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    return _read_from_bytes(file_bytes, filename)


def load_dataframe_from_path(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        data = f.read()
    return _read_from_bytes(data, os.path.basename(path))


def _summary(df: pd.DataFrame) -> DatasetSummaryDC:
    dtypes: Dict[str, str] = {c: str(df[c].dtype) for c in df.columns}
    missing_by_col: Dict[str, int] = {c: int(df[c].isna().sum()) for c in df.columns}
    missing_total = int(sum(missing_by_col.values()))
    try:
        mem_bytes = int(df.memory_usage(deep=True).sum())
    except Exception:
        mem_bytes = 0
    return DatasetSummaryDC(
        rows=int(len(df)),
        columns=int(df.shape[1]),
        column_names=list(df.columns.astype(str)),
        dtypes=dtypes,
        memory_bytes=mem_bytes,
        missing_total=missing_total,
        missing_by_column=missing_by_col,
    )


def _numeric_stats_for(df: pd.DataFrame, col: str) -> DescriptiveStatsDC:
    s = df[col].dropna()
    count = int(len(s))
    if count == 0:
        return DescriptiveStatsDC(column=col, count=0, is_numeric=True)
    try:
        q25 = float(np.nanpercentile(s, 25))
        q50 = float(np.nanpercentile(s, 50))
        q75 = float(np.nanpercentile(s, 75))
        return DescriptiveStatsDC(
            column=col,
            count=count,
            is_numeric=True,
            mean=float(np.mean(s)),
            std=float(np.std(s, ddof=1)) if count > 1 else 0.0,
            min=float(np.min(s)),
            q25=q25,
            q50=q50,
            q75=q75,
            max=float(np.max(s)),
        )
    except Exception:
        return DescriptiveStatsDC(column=col, count=count, is_numeric=True)


def _categorical_stats_for(df: pd.DataFrame, col: str) -> DescriptiveStatsDC:
    s = df[col].dropna().astype(str)
    count = int(len(s))
    if count == 0:
        return DescriptiveStatsDC(column=col, count=0, is_numeric=False, unique=0)
    try:
        vc = s.value_counts().head(1)
        top = str(vc.index[0]) if not vc.empty else None
        freq = int(vc.iloc[0]) if not vc.empty else None
        return DescriptiveStatsDC(
            column=col,
            count=count,
            is_numeric=False,
            unique=int(s.nunique()),
            top=top,
            freq=freq,
        )
    except Exception:
        return DescriptiveStatsDC(column=col, count=count, is_numeric=False)


def analyze_dataframe(df: pd.DataFrame, sample_rows: int = 5) -> DatasetAnalysisDC:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("DataFrame is empty or invalid.")

    summary = _summary(df)

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [
        c
        for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    numeric_stats = [_numeric_stats_for(df, c) for c in numeric_cols]
    cat_stats = [_categorical_stats_for(df, c) for c in cat_cols]

    sample = df.head(sample_rows).replace({np.nan: None}).to_dict(orient="records")

    return DatasetAnalysisDC(
        summary=summary,
        numeric_stats=numeric_stats,
        categorical_stats=cat_stats,
        numeric_columns=numeric_cols,
        categorical_columns=cat_cols,
        datetime_columns=dt_cols,
        sample_records=[{str(k): v for k, v in rec.items()} for rec in sample],
    )


def correlation_matrix(df, method: str = "pearson") -> Dict[str, Any]:
    numeric = df.select_dtypes(include="number")
    if len(numeric.columns) == 0:
        return {"columns": [], "index": [], "matrix": [], "method": method}
    corr_df = numeric.corr(method=method)
    cols = list(corr_df.columns)
    mat = []
    for row in corr_df.values.tolist():
        mat.append([(None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v)) for v in row])
    return {"columns": cols, "index": cols, "matrix": mat, "method": method}


def iqr_outliers(df, column: str, k: float = 1.5) -> Dict[str, Any]:
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column {column} not numeric or not found")
    s = pd.to_numeric(df[column], errors="coerce").dropna().values.astype(float).tolist()
    q1 = float(np.nanpercentile(s, 25))
    q3 = float(np.nanpercentile(s, 75))
    iqr = q3 - q1
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    outlier_indices = []
    outlier_values = []
    for idx in df.index:
        raw_val = df.at[idx, column]
        try:
            val = float(raw_val)
            if math.isnan(val):
                continue
            if val < lower_bound or val > upper_bound:
                outlier_indices.append(int(idx))
                outlier_values.append(float(val))
        except (ValueError, TypeError):
            continue
    outlier_count = len(outlier_indices)
    n_total_rows = len(df)
    return {
        "column": column,
        "k": float(k),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "outlier_count": int(outlier_count),
        "outlier_indices": outlier_indices,
        "outlier_values": outlier_values,
        "n_total_rows": int(n_total_rows),
    }


def profile_dataframe(df: pd.DataFrame, iqr_k: float = 1.5) -> Dict[str, Any]:
    """Return practical EDA quality flags and per-column profiling metadata."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("DataFrame is empty or invalid.")
    if not 0.1 <= float(iqr_k) <= 10.0:
        raise ValueError("iqr_k must be between 0.1 and 10.")

    row_count = len(df)
    columns = []
    recommendations = []
    for column in df.columns:
        series = df[column]
        non_null = series.dropna()
        unique = int(non_null.nunique(dropna=True))
        missing = int(series.isna().sum())
        distinct_ratio = unique / max(1, len(non_null))
        item: Dict[str, Any] = {
            "column": str(column),
            "dtype": str(series.dtype),
            "missing_count": missing,
            "missing_ratio": round(missing / row_count, 6),
            "unique_count": unique,
            "distinct_ratio": round(distinct_ratio, 6),
            "constant": unique <= 1,
            "high_cardinality": bool(unique > 20 and distinct_ratio >= 0.95),
            "outlier_count": 0,
            "skewness": None,
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()
            if len(numeric):
                try:
                    item["skewness"] = float(numeric.skew()) if len(numeric) > 2 else 0.0
                    outlier = iqr_outliers(df, column, k=float(iqr_k))
                    item["outlier_count"] = int(outlier["outlier_count"])
                    item["outlier_ratio"] = round(outlier["outlier_count"] / row_count, 6)
                except Exception:
                    item["outlier_ratio"] = 0.0
        if missing:
            recommendations.append(f"Review missing values in '{column}' ({missing} rows).")
        if item["constant"]:
            recommendations.append(f"'{column}' is constant or empty and may not be informative.")
        if item["high_cardinality"]:
            recommendations.append(f"'{column}' has high cardinality; review its role before categorical encoding.")
        if item.get("outlier_count", 0):
            recommendations.append(f"Inspect {item['outlier_count']} IQR outliers in '{column}'.")
        columns.append(item)

    return {
        "rows": int(row_count),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "profiled_columns": columns,
        "recommendations": recommendations,
    }
