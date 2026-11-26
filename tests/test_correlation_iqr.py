import io
import json
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from ai_engine.data_intelligence import analyzer


def build_test_df():
    return pd.DataFrame({
        "A": [1, 2, 3, 4, 5, 99],
        "B": [10, 20, 30, 40, 50, 60],
        "C": ["x", "y", "z", "w", "v", "u"],
        "label": ["red", "red", "red", "red", "red", "red"],
    })


def test_correlation_matrix_unit():
    print("=== test_correlation_matrix_unit ===")
    df = build_test_df()
    result = analyzer.correlation_matrix(df, method="pearson")
    print("columns:", result["columns"])
    print("index:", result["index"])
    print("method:", result["method"])
    print("matrix:", result["matrix"])

    assert len(result["columns"]) == 2, f"Expected 2 numeric columns, got {len(result['columns'])}"
    assert result["columns"] == result["index"], "columns != index"
    assert len(result["matrix"]) == 2, f"Expected 2 rows, got {len(result['matrix'])}"
    for row in result["matrix"]:
        assert len(row) == 2, f"Expected square row len 2, got {len(row)}"
        for v in row:
            assert v is None or isinstance(v, float), f"Value must be float or None, got {type(v)}: {v}"

    raw = json.dumps(result)
    assert "NaN" not in raw, "JSON dump contains NaN"
    assert "-nan" not in raw, "JSON dump contains -nan"
    print("PASSED: square matrix, float/None only, no NaN strings in JSON")


def test_iqr_outliers_unit():
    print("=== test_iqr_outliers_unit ===")
    df = build_test_df()
    result = analyzer.iqr_outliers(df, "A", k=1.5)
    print(json.dumps(result, indent=2))

    assert result["column"] == "A"
    assert abs(result["k"] - 1.5) < 1e-9
    assert result["lower_bound"] <= result["upper_bound"], "AC-R7: lower > upper"
    assert result["outlier_count"] == len(result["outlier_indices"]) == len(result["outlier_values"]), "AC-R7: length mismatch"

    print(f"q1={result['q1']}, q3={result['q3']}, iqr={result['iqr']}")
    print(f"lower={result['lower_bound']}, upper={result['upper_bound']}")
    print(f"outlier_count={result['outlier_count']}")
    print(f"outlier_indices={result['outlier_indices']}")
    print(f"outlier_values={result['outlier_values']}")
    print(f"n_total_rows={result['n_total_rows']}")

    assert result["n_total_rows"] == 6
    assert result["outlier_count"] >= 1, "Expected at least 1 outlier in A column"
    assert 5 in result["outlier_indices"], f"Row index 5 (A=99) should be outlier, indices={result['outlier_indices']}"
    assert 99.0 in result["outlier_values"], f"99.0 should be in outlier values, got {result['outlier_values']}"
    print("PASSED: A=99 row detected as outlier, bounds ordered, lengths consistent")


def test_correlation_empty_numeric():
    print("=== test_correlation_empty_numeric ===")
    df = pd.DataFrame({"C": ["x", "y"], "label": ["a", "b"]})
    result = analyzer.correlation_matrix(df)
    assert result["columns"] == []
    assert result["index"] == []
    assert result["matrix"] == []
    print("PASSED: empty numeric columns returns empty structure")


def test_iqr_invalid_column():
    print("=== test_iqr_invalid_column ===")
    df = build_test_df()
    try:
        analyzer.iqr_outliers(df, "nonexistent")
        assert False, "Should raise ValueError for missing column"
    except ValueError as e:
        print(f"Got expected ValueError: {e}")
    try:
        analyzer.iqr_outliers(df, "C")
        assert False, "Should raise ValueError for non-numeric column C"
    except ValueError as e:
        print(f"Got expected ValueError: {e}")
    print("PASSED: invalid column raises ValueError")


def test_api_endpoints():
    print("=== test_api_endpoints via TestClient ===")
    try:
        from fastapi.testclient import TestClient
        from backend.main import app
    except Exception as e:
        print(f"Skipping API test (import failed): {e}")
        return

    client = TestClient(app)

    df = build_test_df()
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    print("--- POST /api/data/correlation-matrix ---")
    files = {"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")}
    data = {"method": "pearson"}
    resp = client.post("/api/data/correlation-matrix", files=files, data=data)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    body = resp.json()
    print("response keys:", list(body.keys()))
    assert "correlation" in body
    assert "error" in body
    corr = body["correlation"]
    assert corr is not None, f"correlation should not be None, error={body.get('error')}"
    assert len(corr["columns"]) == 2
    assert len(corr["matrix"]) == 2
    raw = json.dumps(body)
    assert "NaN" not in raw
    assert "-nan" not in raw
    print("PASSED: correlation endpoint returns 200, 2x2 matrix, no NaN in JSON")

    print("--- POST /api/data/outliers ---")
    files = {"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")}
    data = {"column": "A", "k": "1.5"}
    resp = client.post("/api/data/outliers", files=files, data=data)
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    print("response keys:", list(body.keys()))
    assert "outliers" in body
    assert "error" in body
    out = body["outliers"]
    assert out is not None, f"outliers should not be None, error={body.get('error')}"
    assert out["lower_bound"] <= out["upper_bound"]
    assert out["outlier_count"] == len(out["outlier_indices"]) == len(out["outlier_values"])
    assert 5 in out["outlier_indices"]
    assert 99.0 in out["outlier_values"]
    print("PASSED: outliers endpoint returns 200, detects A=99 as outlier")

    print("--- POST /api/data/outliers (bad column) ---")
    files = {"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")}
    data = {"column": "column_does_not_exist", "k": "1.5"}
    resp = client.post("/api/data/outliers", files=files, data=data)
    assert resp.status_code == 200, f"Expected HTTP 200 for ValueError, got {resp.status_code}"
    body = resp.json()
    assert body["outliers"] is None
    assert isinstance(body["error"], str) and len(body["error"]) > 0
    print("PASSED: bad column returns HTTP 200 with outliers=null, error=str")

    print("--- POST /api/data/correlation-matrix (no file, no id) ---")
    resp = client.post("/api/data/correlation-matrix", data={"method": "pearson"})
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    body = resp.json()
    assert body["correlation"] is None
    assert isinstance(body["error"], str) and len(body["error"]) > 0
    print("PASSED: no file/no id returns HTTP 200 correlation=null error=str")


if __name__ == "__main__":
    test_correlation_matrix_unit()
    print()
    test_iqr_outliers_unit()
    print()
    test_correlation_empty_numeric()
    print()
    test_iqr_invalid_column()
    print()
    test_api_endpoints()
    print()
    print("ALL TESTS PASSED")
