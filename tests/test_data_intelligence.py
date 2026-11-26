import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from ai_engine.data_intelligence.analyzer import analyze_dataframe


def build_df():
    return pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": [10, 20, 30, 40, 50],
        "C": ["x", "y", "z", "w", "v"],
    })


def test_analyze_dataframe_summary():
    print("=== test_analyze_dataframe_summary ===")
    df = build_df()
    result = analyze_dataframe(df)
    assert result.summary.rows == 5, f"expected 5 rows, got {result.summary.rows}"
    assert result.summary.columns == 3, f"expected 3 columns, got {result.summary.columns}"
    assert len(result.numeric_stats) == 2, f"expected 2 numeric columns, got {len(result.numeric_stats)}"
    assert len(result.categorical_stats) == 1, f"expected 1 categorical column, got {len(result.categorical_stats)}"
    print("rows:", result.summary.rows, "cols:", result.summary.columns)
    print("numeric_stats:", [s.column for s in result.numeric_stats])
    print("categorical_stats:", [s.column for s in result.categorical_stats])
    print("PASSED: analyze_dataframe reports correct summary rows/columns/stats")


def test_analyze_dataframe_to_dict_json_safe():
    print("=== test_analyze_dataframe_to_dict_json_safe ===")
    df = build_df()
    data = analyze_dataframe(df).to_dict()
    raw = json.dumps(data)
    print("summary:", data["summary"]["rows"], "x", data["summary"]["columns"])
    assert "NaN" not in raw, "JSON dump contains NaN"
    assert "-nan" not in raw, "JSON dump contains -nan"
    assert data["summary"]["rows"] == 5
    assert data["summary"]["columns"] == 3
    assert len(data["numeric_stats"]) == 2
    assert len(data["categorical_stats"]) == 1
    for stat in data["numeric_stats"]:
        assert stat["is_numeric"] is True
    print("PASSED: analyze_dataframe to_dict is JSON-serializable without NaN")


if __name__ == "__main__":
    test_analyze_dataframe_summary()
    print()
    test_analyze_dataframe_to_dict_json_safe()
    print()
    print("ALL TESTS PASSED")
