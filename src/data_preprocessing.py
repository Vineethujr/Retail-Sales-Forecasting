"""
Retail Sales Forecasting & Inventory Optimization System
Module: Data Preprocessing & Validation (src/data_preprocessing.py)

Purpose:
  - Validates raw retail transaction schema and column data types.
  - Cleans customer returns: isolates return units from positive gross demand.
  - Aggregates sub-daily transactions to the daily store-SKU grain.
  - Builds a continuous daily calendar grid to handle missing dates cleanly.
  - Distinguishes observed zero-sales days from unobserved data gaps.
  - Outputs production-ready clean datasets to data/processed/clean_sales_data.csv.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np


REQUIRED_COLUMNS: List[str] = ["date", "store_id", "sku_id", "quantity"]


def validate_raw_schema(df: pd.DataFrame) -> bool:
    """
    Validate that the input dataframe contains all required columns
    and appropriate basic types.
    """
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Schema Validation Error: Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("Schema Validation Error: Input dataframe is empty.")

    return True


def clean_and_aggregate_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw transactions:
      1. Converts date to datetime.
      2. Separates gross sales units from returns.
         - gross_demand_units = max(0, quantity)
         - return_units = abs(min(0, quantity))
      3. Aggregates multiple transactions on the same day for each (store_id, sku_id, date).
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    # Ensure quantity is numeric
    data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce").fillna(0)

    # Separate gross customer demand from customer returns
    data["gross_demand_units"] = data["quantity"].clip(lower=0)
    data["return_units"] = data["quantity"].apply(lambda q: abs(q) if q < 0 else 0)

    # Unit price & cost handling (fallback defaults if missing)
    if "unit_price" not in data.columns:
        data["unit_price"] = 10.0
    if "unit_cost" not in data.columns:
        data["unit_cost"] = data["unit_price"] * 0.6
    if "is_promo" not in data.columns:
        data["is_promo"] = 0
    if "category" not in data.columns:
        data["category"] = "General"

    # Aggregation dictionary
    agg_rules = {
        "gross_demand_units": "sum",
        "return_units": "sum",
        "unit_price": "mean",
        "unit_cost": "mean",
        "is_promo": "max",
        "category": "first",
    }

    daily_df = (
        data.groupby(["store_id", "sku_id", "date"], as_index=False)
        .agg(agg_rules)
    )

    # Calculate gross revenue
    daily_df["revenue"] = daily_df["gross_demand_units"] * daily_df["unit_price"]

    return daily_df


def build_continuous_daily_grid(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures every (store_id, sku_id) series has an uninterrupted daily date index.
    Fills genuine non-sale days with 0 demand while explicitly marking
    whether the day had an observed record in the source data.
    """
    df_work = daily_df.copy()
    df_work["date"] = pd.to_datetime(df_work["date"])

    processed_series = []
    unique_combinations = df_work[["store_id", "sku_id"]].drop_duplicates()

    for _, row in unique_combinations.iterrows():
        st_id = row["store_id"]
        sk_id = row["sku_id"]

        sub_df = df_work[(df_work["store_id"] == st_id) & (df_work["sku_id"] == sk_id)].copy()
        sub_df = sub_df.sort_values("date").set_index("date")

        min_date = sub_df.index.min()
        max_date = sub_df.index.max()

        # Complete daily index
        full_idx = pd.date_range(start=min_date, end=max_date, freq="D", name="date")
        reindexed_df = sub_df.reindex(full_idx)

        # Flag observed vs imputed dates
        reindexed_df["sales_observed_flag"] = (~reindexed_df["gross_demand_units"].isna()).astype(int)

        # Fill identifiers and categories
        reindexed_df["store_id"] = st_id
        reindexed_df["sku_id"] = sk_id
        reindexed_df["category"] = reindexed_df["category"].ffill().bfill()

        # Fill price/cost by forward filling then backward filling
        reindexed_df["unit_price"] = reindexed_df["unit_price"].ffill().bfill()
        reindexed_df["unit_cost"] = reindexed_df["unit_cost"].ffill().bfill()

        # Genuine missing date demand is 0
        reindexed_df["gross_demand_units"] = reindexed_df["gross_demand_units"].fillna(0)
        reindexed_df["return_units"] = reindexed_df["return_units"].fillna(0)
        reindexed_df["is_promo"] = reindexed_df["is_promo"].fillna(0)
        reindexed_df["revenue"] = reindexed_df["gross_demand_units"] * reindexed_df["unit_price"]

        processed_series.append(reindexed_df.reset_index())

    clean_df = pd.concat(processed_series, ignore_index=True)
    clean_df = clean_df.sort_values(by=["store_id", "sku_id", "date"]).reset_index(drop=True)
    return clean_df


def run_preprocessing_pipeline(
    raw_data_path: str = "data/raw/sales_data.csv",
    output_path: str = "data/processed/clean_sales_data.csv",
) -> pd.DataFrame:
    """
    Orchestrates end-to-end data validation, cleaning, grid construction,
    and storage of the clean retail sales dataset.
    """
    raw_file = Path(raw_data_path)
    if not raw_file.exists():
        raise FileNotFoundError(f"Raw data file not found at: {raw_file.resolve()}")

    print(f"[INFO] Loading raw data from: {raw_file}")
    df_raw = pd.read_csv(raw_file)

    print("[INFO] Validating schema...")
    validate_raw_schema(df_raw)

    print("[INFO] Aggregating transactions and isolating returns...")
    df_agg = clean_and_aggregate_transactions(df_raw)

    print("[INFO] Building continuous daily calendar grid...")
    df_clean = build_continuous_daily_grid(df_agg)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(out_file, index=False)

    print(f"[SUCCESS] Preprocessed dataset saved to: {out_file.resolve()}")
    print(f"[METRICS] Total rows: {len(df_clean):,}")
    print(f"[METRICS] Total Gross Units: {df_clean['gross_demand_units'].sum():,.0f}")
    print(f"[METRICS] Total Return Units: {df_clean['return_units'].sum():,.0f}")
    print(f"[METRICS] Total Gross Revenue: ${df_clean['revenue'].sum():,.2f}")

    return df_clean


if __name__ == "__main__":
    run_preprocessing_pipeline()
