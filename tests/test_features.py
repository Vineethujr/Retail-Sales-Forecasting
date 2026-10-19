"""
Unit Tests for Feature Engineering & Zero-Leakage Invariants
(tests/test_features.py)
"""

import pandas as pd
import numpy as np
from src.feature_engineering import create_calendar_features, create_lag_and_rolling_features


def test_calendar_features():
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-06"]})  # Mon (0) and Sat (5)
    res = create_calendar_features(df)
    assert res.iloc[0]["day_of_week"] == 0
    assert res.iloc[0]["is_weekend"] == 0
    assert res.iloc[1]["day_of_week"] == 5
    assert res.iloc[1]["is_weekend"] == 1


def test_zero_leakage_rolling_statistics():
    """
    Critical Test: Ensure that rolling statistics on day t
    do NOT include day t's actual demand.
    """
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "store_id": "STORE_01",
        "sku_id": "SKU_01",
        "gross_demand_units": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
    })

    res = create_lag_and_rolling_features(df, lag_days=[1, 7], rolling_windows=[7])

    # On day index 1 (2024-01-02), lag_1 must equal day 0 demand (10.0)
    assert res.iloc[1]["lag_1"] == 10.0

    # On day index 7 (2024-01-08 with demand 80.0):
    # rolling_mean_7 must be the mean of days 0 to 6 [10, 20, 30, 40, 50, 60, 70], which is 40.0!
    # It must NOT include 80.0!
    assert res.iloc[7]["rolling_mean_7"] == 40.0
