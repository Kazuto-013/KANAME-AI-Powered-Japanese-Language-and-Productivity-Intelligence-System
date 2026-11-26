from __future__ import annotations

import os
import tempfile
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from sqlalchemy.orm import Session
import pandas as pd

from ai_engine.data_intelligence import analyzer
from backend.database.config import get_db
from backend.services import data_service as data_svc
from backend.schemas.data_schemas import (
    DatasetSummary,
    DescriptiveStats,
    DatasetAnalysisResponse,
    DatasetUploadResponse,
    DatasetHealthResponse,
)

router = APIRouter(prefix="/data", tags=["Data Intelligence"])


@router.get("/", status_code=200)
async def data_root():
    return {
        "module": "Data Intelligence & Visualization",
        "status": "implemented",
        "engine": "pandas + NumPy + Plotly",
        "supported_formats": [".csv", ".tsv", ".xlsx", ".xls", ".json"],
        "endpoints": {
            "health": "GET /api/data/health",
            "upload": "POST /api/data/upload",
            "analyze": "POST /api/data/analyze (multipart file + options)",
            "profile": "POST /api/data/profile (dataset quality and EDA flags)",
            "dataset_by_id": "GET /api/data/datasets/{dataset_id}",
            "list": "GET /api/data/datasets",
        },
        "description": (
            "Dataset upload, descriptive statistics, missing value analysis, "
            "numeric + categorical summaries, sample records preview."
        ),
    }


@router.get("/health", response_model=DatasetHealthResponse, tags=["Data Intelligence"])
async def data_health():
    return DatasetHealthResponse(
        status="ok",
        engine="pandas + NumPy + Plotly",
        supported_formats=sorted(list(analyzer.SUPPORTED_EXT)),
    )


def _to_stats(py_stats: List[analyzer.DescriptiveStatsDC]) -> List[DescriptiveStats]:
    out = []
    for s in py_stats:
        d = s.to_dict()
        out.append(DescriptiveStats(**d))
    return out


def _dc_summary_to_pydantic(s: analyzer.DatasetSummaryDC) -> DatasetSummary:
    return DatasetSummary(**s.to_dict())


@router.post("/upload", response_model=DatasetUploadResponse, tags=["Data Intelligence"])
async def upload_dataset(
    file: UploadFile = File(..., description="CSV/TSV/XLSX/XLS/JSON dataset"),
    db: Session = Depends(get_db),
):
    filename = file.filename or "dataset.csv"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in analyzer.SUPPORTED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {sorted(analyzer.SUPPORTED_EXT)}",
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    size = 0
    try:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
            size += len(chunk)
            if size > 128 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Dataset too large (>128 MB).")
    finally:
        tmp.close()

    try:
        df = analyzer.load_dataframe_from_path(tmp.name)
        try:
            ds_id = data_svc.create_dataset_record(
                db=db,
                filename=filename,
                rows=int(len(df)),
                columns=int(df.shape[1]),
                columns_names=",".join(str(c) for c in df.columns),
                size_bytes=size,
                content=Path(tmp.name).read_bytes(),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Dataset was parsed but could not be stored: {exc}",
            ) from exc

        return DatasetUploadResponse(
            dataset_id=ds_id,
            filename=filename,
            source_stored=True,
            rows=int(len(df)),
            columns=int(df.shape[1]),
            columns_names=[str(c) for c in df.columns],
            message=(
                f"Dataset uploaded and stored successfully. Use dataset_id='{ds_id}' "
                "or call /api/data/analyze directly with the file."
            ),
        )
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@router.post("/analyze", response_model=DatasetAnalysisResponse, tags=["Data Intelligence"])
async def analyze_dataset(
    file: UploadFile = File(..., description="CSV/TSV/XLSX/XLS/JSON dataset"),
    sample_rows: int = Form(5, ge=1, le=50),
    db: Session = Depends(get_db),
):
    filename = file.filename or "dataset.csv"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in analyzer.SUPPORTED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {sorted(analyzer.SUPPORTED_EXT)}",
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        size = 0
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
            size += len(chunk)
            if size > 128 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Dataset too large (>128 MB).")
    finally:
        tmp.close()

    try:
        df = analyzer.load_dataframe_from_path(tmp.name)
        analysis = analyzer.analyze_dataframe(df, sample_rows=sample_rows)

        ds_id = None
        try:
            ds_id = data_svc.create_dataset_record(
                db=db,
                filename=filename,
                rows=analysis.summary.rows,
                columns=analysis.summary.columns,
                columns_names=",".join(analysis.summary.column_names),
                size_bytes=size,
                content=Path(tmp.name).read_bytes(),
            )
        except Exception:
            db.rollback()

        return DatasetAnalysisResponse(
            dataset_id=ds_id,
            summary=_dc_summary_to_pydantic(analysis.summary),
            numeric_stats=_to_stats(analysis.numeric_stats),
            categorical_stats=_to_stats(analysis.categorical_stats),
            numeric_columns=analysis.numeric_columns,
            categorical_columns=analysis.categorical_columns,
            datetime_columns=analysis.datetime_columns,
            sample_records=analysis.sample_records,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyzer error: {e}")
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def _read_upload_to_temp(file: UploadFile) -> str:
    filename = file.filename or "dataset.csv"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in analyzer.SUPPORTED_EXT:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: {sorted(analyzer.SUPPORTED_EXT)}"
        )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    size = 0
    try:
        while True:
            chunk = file.file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
            size += len(chunk)
            if size > 128 * 1024 * 1024:
                raise ValueError("Dataset too large (>128 MB).")
    finally:
        tmp.close()
    return tmp.name


def _df_from_file_or_dataset_id(
    file: Optional[UploadFile],
    dataset_id: Optional[str],
    db: Session,
) -> pd.DataFrame:
    if file is not None:
        tmp_path = _read_upload_to_temp(file)
        try:
            return analyzer.load_dataframe_from_path(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    elif dataset_id is not None:
        ds = data_svc.get_dataset(db, dataset_id)
        if ds is None:
            raise ValueError("dataset_id not found")
        content = data_svc.get_dataset_content(db, dataset_id)
        if content is None:
            raise ValueError(
                "Dataset metadata exists but its source file is unavailable; re-upload the dataset."
            )
        return analyzer.load_dataframe_from_file(content, ds["filename"])
    else:
        raise ValueError("Either file or dataset_id must be provided.")


@router.post("/correlation-matrix", tags=["Data Intelligence"])
async def correlation_matrix_endpoint(
    file: Optional[UploadFile] = File(None, description="CSV or Excel dataset upload"),
    dataset_id: Optional[str] = Form(None, description="Or fetch by previously stored dataset_id"),
    method: str = Form("pearson", pattern="^(pearson|spearman|kendall)$"),
    db: Session = Depends(get_db),
):
    try:
        df = _df_from_file_or_dataset_id(file, dataset_id, db)
    except ValueError as e:
        return {"correlation": None, "error": str(e)}
    try:
        result = analyzer.correlation_matrix(df, method)
        return {"correlation": result, "error": None}
    except Exception as e:
        return {"correlation": None, "error": str(e)}


@router.post("/outliers", tags=["Data Intelligence"])
async def outliers_endpoint(
    file: Optional[UploadFile] = File(None),
    dataset_id: Optional[str] = Form(None),
    column: str = Form(..., description="Column name to inspect"),
    k: float = Form(1.5, ge=0.1, le=10.0, description="IQR multiplier (1.5 mild, 3 extreme)"),
    db: Session = Depends(get_db),
):
    try:
        df = _df_from_file_or_dataset_id(file, dataset_id, db)
    except ValueError as e:
        return {"outliers": None, "error": str(e)}
    try:
        result = analyzer.iqr_outliers(df, column, k)
        return {"outliers": result, "error": None}
    except ValueError as e:
        return {"outliers": None, "error": str(e)}
    except Exception as e:
        return {"outliers": None, "error": str(e)}


@router.post("/profile", tags=["Data Intelligence"])
async def profile_dataset_endpoint(
    file: Optional[UploadFile] = File(None),
    dataset_id: Optional[str] = Form(None),
    iqr_k: float = Form(1.5, ge=0.1, le=10.0),
    db: Session = Depends(get_db),
):
    try:
        df = _df_from_file_or_dataset_id(file, dataset_id, db)
        return {"profile": analyzer.profile_dataframe(df, iqr_k=iqr_k), "error": None}
    except ValueError as exc:
        return {"profile": None, "error": str(exc)}
    except Exception as exc:
        return {"profile": None, "error": f"Dataset profiling failed: {exc}"}


@router.get("/datasets/{dataset_id}", tags=["Data Intelligence"])
async def get_dataset_by_id(dataset_id: str, db: Session = Depends(get_db)):
    try:
        item = data_svc.get_dataset(db, dataset_id)
    except Exception as e:
        return {"dataset": None, "error": f"Database unavailable: {e}"}
    if item is None:
        return {"dataset": None, "error": "dataset_id not found"}
    try:
        content = data_svc.get_dataset_content(db, dataset_id)
        if content is not None:
            frame = analyzer.load_dataframe_from_file(content, item["filename"])
            item["sample_records"] = json.loads(
                frame.head(100).to_json(orient="records", date_format="iso")
            )
    except Exception as e:
        return {"dataset": item, "error": f"Could not load stored source: {e}"}
    return {"dataset": item}


@router.post("/dashboard", tags=["Data Intelligence"])
async def build_company_dashboard_upload(
    file: UploadFile = File(..., description="CSV/TSV/XLSX/XLS/JSON company dataset (sales/finance/HR)"),
    iqr_k: float = Form(1.5, ge=0.1, le=10.0, description="IQR outlier sensitivity (higher = fewer outliers)"),
    corr_method: str = Form("pearson", description="Correlation method: pearson / spearman / kendall"),
):
    """Build an interactive company dashboard with KPI cards + 4 Plotly figures.

    The UI (Streamlit/Gradio/React) renders the returned figure dicts with
    `plotly.io.from_json` / `st.plotly_chart(..., use_container_width=True)`.
    """
    import tempfile as _tf
    filename = file.filename or "dataset.csv"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in analyzer.SUPPORTED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {sorted(list(analyzer.SUPPORTED_EXT))}",
        )
    raw = await file.read()
    suffix = ext or ".csv"
    with _tf.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        df = analyzer.load_dataframe_from_file(tmp_path)
        dash = analyzer.build_company_dashboard(
            df=df, iqr_k=float(iqr_k), corr_method=str(corr_method).lower().strip() or "pearson",
        )
        return dash.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard build failed: {type(e).__name__}: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@router.get("/datasets", tags=["Data Intelligence"])
async def list_datasets(limit: int = 50, db: Session = Depends(get_db)):
    try:
        datasets = data_svc.list_datasets(db, limit=limit)
        return {"datasets": datasets}
    except Exception as e:
        return {"datasets": [], "error": str(e)}
