"""
Unit Tests for Data Preprocessing & Validation
(tests/test_preprocessing.py)
"""

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import (
    validate_raw_schema,
    clean_and_aggregate_transactions,
    build_continuous_daily_grid,
)


def test_validate_raw_schema_valid():
    df = pd.DataFrame({
        "date": ["2024-01-01"],
        "store_id": ["STORE_01"],
        "sku_id": ["SKU_01"],
        "quantity": [10],
    })
    assert validate_raw_schema(df) is True


def test_validate_raw_schema_missing_columns():
    df = pd.DataFrame({
        "date": ["2024-01-01"],
        "quantity": [10],
    })
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw_schema(df)


def test_clean_and_aggregate_returns_separation():
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01"],
        "store_id": ["STORE_01", "STORE_01"],
        "sku_id": ["SKU_01", "SKU_01"],
        "quantity": [15, -3],  # 15 sold, 3 returned
        "unit_price": [10.0, 10.0],
        "unit_cost": [5.0, 5.0],
    })
    cleaned = clean_and_aggregate_transactions(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["gross_demand_units"] == 15
    assert cleaned.iloc[0]["return_units"] == 3


def test_build_continuous_daily_grid_no_gaps():
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-04"],  # missing 01-02 and 01-03
        "store_id": ["STORE_01", "STORE_01"],
        "sku_id": ["SKU_01", "SKU_01"],
        "gross_demand_units": [10, 20],
        "return_units": [0, 0],
        "unit_price": [5.0, 5.0],
        "unit_cost": [2.5, 2.5],
        "category": ["Food", "Food"],
        "is_promo": [0, 0],
    })
    grid = build_continuous_daily_grid(df)
    assert len(grid) == 4  # 01-01, 01-02, 01-03, 01-04
    assert grid["gross_demand_units"].tolist() == [10.0, 0.0, 0.0, 20.0]
    assert grid["sales_observed_flag"].tolist() == [1, 0, 0, 1]
