"""
Unit Tests for Forecasting Accuracy Metrics & Splitting
(tests/test_forecasting.py)
"""

import numpy as np
import pandas as pd
from src.forecasting import calculate_metrics, chronological_split


def test_calculate_metrics_perfect_predictions():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    metrics = calculate_metrics(y_true, y_pred)

    assert metrics["MAE"] == 0.0
    assert metrics["RMSE"] == 0.0
    assert metrics["WAPE_pct"] == 0.0
    assert metrics["Bias_pct"] == 0.0


def test_calculate_metrics_zero_demand_safeguard():
    """Ensure no division-by-zero crash when demand is zero."""
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([0.0, 0.0, 0.0])
    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["WAPE_pct"] == 0.0


def test_chronological_split_ordering():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "gross_demand_units": np.random.randint(10, 50, size=100),
    })

    train_df, val_df, test_df = chronological_split(df, test_days=14, val_days=14)

    assert train_df["date"].max() < val_df["date"].min()
    assert val_df["date"].max() < test_df["date"].min()
    assert len(test_df) == 14
    assert len(val_df) == 14
    assert len(train_df) == 72
