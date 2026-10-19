"""
Retail Sales Forecasting & Inventory Optimization System
Module: Synthetic Data Generator (src/data_generator.py)

Purpose:
  Generates a realistic, multi-store, multi-SKU daily retail sales dataset
  featuring trend, weekly seasonality, promotional lifts, random noise,
  and customer returns (negative quantities).
  Enables complete end-to-end execution without requiring external downloads.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd


def generate_synthetic_retail_data(
    start_date: str = "2024-01-01",
    num_days: int = 730,  # 2 full years of daily sales
    num_stores: int = 2,
    num_skus: int = 10,
    random_seed: int = 42,
    output_path: str = "data/raw/sales_data.csv",
) -> pd.DataFrame:
    """
    Generate synthetic daily retail transactions with realistic retail dynamics:
      - SKU base velocity (high, medium, and low volume products)
      - Store scaling factors
      - Weekly seasonality (weekend sales lifts)
      - Long-term annual seasonality (Q4 holiday surge)
      - Promotional campaigns (price cuts and volume spikes)
      - Customer returns (negative quantity transactions for cleaning validation)
      - Product pricing and unit cost for financial/inventory modeling
    """
    np.random.seed(random_seed)
    date_range = pd.date_range(start=start_date, periods=num_days, freq="D")

    # Define SKU catalog with realistic cost, selling price, and base daily demand
    sku_catalog = [
        {"sku_id": "SKU_01", "category": "Beverages", "unit_cost": 1.50, "unit_price": 3.99, "base_demand": 45},
        {"sku_id": "SKU_02", "category": "Beverages", "unit_cost": 2.20, "unit_price": 5.49, "base_demand": 30},
        {"sku_id": "SKU_03", "category": "Snacks", "unit_cost": 0.80, "unit_price": 2.49, "base_demand": 60},
        {"sku_id": "SKU_04", "category": "Snacks", "unit_cost": 1.10, "unit_price": 2.99, "base_demand": 40},
        {"sku_id": "SKU_05", "category": "Household", "unit_cost": 8.00, "unit_price": 14.99, "base_demand": 15},
        {"sku_id": "SKU_06", "category": "Household", "unit_cost": 12.50, "unit_price": 22.99, "base_demand": 10},
        {"sku_id": "SKU_07", "category": "Personal Care", "unit_cost": 4.00, "unit_price": 8.99, "base_demand": 25},
        {"sku_id": "SKU_08", "category": "Personal Care", "unit_cost": 6.50, "unit_price": 12.99, "base_demand": 18},
        {"sku_id": "SKU_09", "category": "Electronics", "unit_cost": 45.00, "unit_price": 79.99, "base_demand": 5},
        {"sku_id": "SKU_10", "category": "Electronics", "unit_cost": 80.00, "unit_price": 139.99, "base_demand": 3},
    ][:num_skus]

    store_multipliers = {f"STORE_{i+1:02d}": 1.0 + (i * 0.25) for i in range(num_stores)}

    records = []

    for store_id, store_mult in store_multipliers.items():
        for sku_info in sku_catalog:
            sku_id = sku_info["sku_id"]
            category = sku_info["category"]
            unit_cost = sku_info["unit_cost"]
            unit_price = sku_info["unit_price"]
            base_demand = sku_info["base_demand"]

            # Day of week seasonality factors (Mon=0.85, ..., Fri=1.2, Sat=1.35, Sun=1.15)
            dow_factors = [0.85, 0.90, 0.95, 1.00, 1.20, 1.35, 1.15]

            for d_idx, dt in enumerate(date_range):
                day_of_week = dt.weekday()
                day_factor = dow_factors[day_of_week]

                # Slight annual trend (3% annual growth)
                trend = 1.0 + (0.03 * (d_idx / 365.0))

                # Month-of-year seasonality: peak in November & December (Q4 retail holiday surge)
                month_factor = 1.0
                if dt.month in [11, 12]:
                    month_factor = 1.30
                elif dt.month in [1, 2]:
                    month_factor = 0.90

                # Random promotional event: 8% probability of promo on any given day
                is_promo = 1 if np.random.rand() < 0.08 else 0
                promo_lift = 1.40 if is_promo else 1.0

                # Expected mean demand
                expected_demand = base_demand * store_mult * day_factor * month_factor * trend * promo_lift

                # Sample discrete units sold from Poisson / Negative Binomial distribution
                units_sold = int(np.random.poisson(expected_demand))

                # Customer return simulation (approx 1.5% probability of a return transaction)
                if np.random.rand() < 0.015:
                    return_qty = -int(np.random.randint(1, 4))
                    records.append({
                        "date": dt.strftime("%Y-%m-%d"),
                        "store_id": store_id,
                        "sku_id": sku_id,
                        "category": category,
                        "quantity": return_qty,
                        "unit_cost": unit_cost,
                        "unit_price": unit_price,
                        "is_promo": is_promo,
                    })

                # Normal daily sales transaction
                records.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "store_id": store_id,
                    "sku_id": sku_id,
                    "category": category,
                    "quantity": units_sold,
                    "unit_cost": unit_cost,
                    "unit_price": round(unit_price * (0.85 if is_promo else 1.0), 2),
                    "is_promo": is_promo,
                })

    df = pd.DataFrame(records)

    # Sort deterministically
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["store_id", "sku_id", "date"]).reset_index(drop=True)

    # Ensure parent output directory exists
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)

    print(f"[SUCCESS] Generated {len(df):,} synthetic retail sales records.")
    print(f"[INFO] Saved to: {out_file.resolve()}")
    print(f"[INFO] Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"[INFO] Stores: {df['store_id'].nunique()}, SKUs: {df['sku_id'].nunique()}")

    return df


if __name__ == "__main__":
    generate_synthetic_retail_data()
