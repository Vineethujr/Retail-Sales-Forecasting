"""
Retail Sales Forecasting & Inventory Optimization System
Module: Feature Engineering (src/feature_engineering.py)

Purpose:
  - Generates calendar, cyclical, lag, and rolling statistics.
  - Implements strict zero-leakage safeguards (all rolling windows and lags shifted by >= 1).
  - Encodes categorical store and SKU indicators for machine learning models.
  - Prepares structured feature matrices for baseline and ML forecasters.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts calendar and cyclical trigonometric features from the date column.
    Cyclical encodings ensure that Sunday (6) and Monday (0), and December (12)
    and January (1) are recognized as temporally continuous by tree/linear models.
    """
    data = df.copy()
    dt = pd.to_datetime(data["date"])

    data["day_of_week"] = dt.dt.dayofweek
    data["day_of_month"] = dt.dt.day
    data["month"] = dt.dt.month
    data["quarter"] = dt.dt.quarter
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)

    # Cyclical trigonometric transforms
    data["sin_dow"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
    data["cos_dow"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)
    data["sin_month"] = np.sin(2 * np.pi * (data["month"] - 1) / 12.0)
    data["cos_month"] = np.cos(2 * np.pi * (data["month"] - 1) / 12.0)

    return data


def create_lag_and_rolling_features(
    df: pd.DataFrame,
    target_col: str = "gross_demand_units",
    lag_days: List[int] = [1, 7, 14, 21, 28],
    rolling_windows: List[int] = [7, 14, 28],
) -> pd.DataFrame:
    """
    Constructs historical lag features and rolling summary statistics per (store_id, sku_id).

    CRITICAL ZERO-LEAKAGE RULE:
      Every rolling calculation is applied to `target_col.shift(1)` so that the
      current day's demand is NEVER included in its own historical features.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(by=["store_id", "sku_id", "date"]).reset_index(drop=True)

    grouped = data.groupby(["store_id", "sku_id"])[target_col]

    # Create explicit historical lags
    for lag in lag_days:
        data[f"lag_{lag}"] = grouped.shift(lag)

    # Create strictly lagged rolling statistics
    for window in rolling_windows:
        # Shifted by 1 to prevent looking ahead into the target date
        shifted_series = grouped.shift(1)
        data[f"rolling_mean_{window}"] = (
            data.groupby(["store_id", "sku_id"])[target_col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=max(1, window // 2)).mean())
        )
        data[f"rolling_std_{window}"] = (
            data.groupby(["store_id", "sku_id"])[target_col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=max(1, window // 2)).std())
        ).fillna(0)
        data[f"rolling_max_{window}"] = (
            data.groupby(["store_id", "sku_id"])[target_col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).max())
        )

    return data


def build_feature_pipeline(
    df: pd.DataFrame,
    drop_na_rows: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Full feature pipeline execution.
    Returns:
      (transformed_df, feature_column_names)
    """
    df_feat = create_calendar_features(df)
    df_feat = create_lag_and_rolling_features(df_feat)

    # Encode categorical variables: Store & SKU dummy variables while preserving original columns
    store_dummies = pd.get_dummies(df_feat["store_id"], prefix="store", dtype=int)
    sku_dummies = pd.get_dummies(df_feat["sku_id"], prefix="sku", dtype=int)
    df_feat = pd.concat([df_feat, store_dummies, sku_dummies], axis=1)

    # Identify all feature column names
    excluded_cols = [
        "date",
        "store_id",
        "sku_id",
        "category",
        "gross_demand_units",
        "quantity",
        "return_units",
        "revenue",
        "sales_observed_flag",
    ]
    feature_cols = [c for c in df_feat.columns if c not in excluded_cols]

    if drop_na_rows:
        # Drop warm-up rows (first 28 days where lag_28 is NaN)
        initial_count = len(df_feat)
        df_feat = df_feat.dropna(subset=feature_cols).reset_index(drop=True)
        dropped_count = initial_count - len(df_feat)
        print(f"[INFO] Dropped {dropped_count} warm-up records with unobserved 28-day lags.")

    return df_feat, feature_cols


if __name__ == "__main__":
    df_clean = pd.read_csv("data/processed/clean_sales_data.csv")
    df_feat, feats = build_feature_pipeline(df_clean)
    print(f"[SUCCESS] Feature engineering complete. Shape: {df_feat.shape}")
    print(f"[INFO] Feature columns count: {len(feats)}")
    print(f"[INFO] Sample features: {feats[:8]}...")
