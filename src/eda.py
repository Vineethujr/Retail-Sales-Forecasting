"""
Retail Sales Forecasting & Inventory Optimization System
Module: Exploratory Data Analysis & SKU Classification (src/eda.py)

Purpose:
  - Computes retail descriptive statistics across stores and SKUs.
  - Implements Pareto ABC classification (revenue contribution: 80/15/5).
  - Implements XYZ classification (demand volatility CV: <=0.5, 0.5-1.0, >1.0).
  - Generates publication-ready exploratory charts to reports/figures/.
  - Exports SKU classification summary to reports/sku_abc_xyz_classification.csv.
"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Set consistent, modern aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "Arial, DejaVu Sans, sans-serif"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def calculate_abc_xyz_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs dual ABC and XYZ classification on SKU-level historical demand.

    ABC Classification (Cumulative Revenue):
      - Class A: Top 80% of cumulative revenue (High business impact)
      - Class B: Next 15% (80% to 95% cumulative revenue, Moderate impact)
      - Class C: Bottom 5% (95% to 100% cumulative revenue, Long tail)

    XYZ Classification (Demand Volatility - Coefficient of Variation CV):
      - Class X: CV <= 0.5 (Steady, highly predictable demand)
      - Class Y: 0.5 < CV <= 1.0 (Variable, seasonal demand)
      - Class Z: CV > 1.0 (Erratic, lumpy, unpredictable demand)
    """
    sku_stats = (
        df.groupby("sku_id")
        .agg(
            category=("category", "first"),
            total_units=("gross_demand_units", "sum"),
            total_revenue=("revenue", "sum"),
            mean_daily_units=("gross_demand_units", "mean"),
            std_daily_units=("gross_demand_units", "std"),
            avg_unit_price=("unit_price", "mean"),
            avg_unit_cost=("unit_cost", "mean"),
        )
        .reset_index()
    )

    # Calculate Coefficient of Variation (CV = std / mean)
    sku_stats["cv_demand"] = sku_stats["std_daily_units"] / sku_stats["mean_daily_units"].replace(0, np.nan)
    sku_stats["cv_demand"] = sku_stats["cv_demand"].fillna(2.0)

    # Sort descending by total revenue for Pareto ABC
    sku_stats = sku_stats.sort_values(by="total_revenue", ascending=False).reset_index(drop=True)
    sku_stats["cum_revenue"] = sku_stats["total_revenue"].cumsum()
    total_rev = sku_stats["total_revenue"].sum()
    sku_stats["cum_rev_pct"] = (sku_stats["cum_revenue"] / total_rev) * 100.0

    # Assign ABC Class
    def assign_abc(pct: float) -> str:
        if pct <= 80.0:
            return "A"
        elif pct <= 95.0:
            return "B"
        return "C"

    sku_stats["abc_class"] = sku_stats["cum_rev_pct"].apply(assign_abc)

    # Assign XYZ Class
    def assign_xyz(cv: float) -> str:
        if cv <= 0.5:
            return "X"
        elif cv <= 1.0:
            return "Y"
        return "Z"

    sku_stats["xyz_class"] = sku_stats["cv_demand"].apply(assign_xyz)
    sku_stats["segment"] = sku_stats["abc_class"] + sku_stats["xyz_class"]

    return sku_stats


def generate_eda_figures(df: pd.DataFrame, sku_stats: pd.DataFrame, output_dir: str = "reports/figures") -> None:
    """
    Creates and saves high-resolution exploratory charts for business reporting.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Total Daily Sales & 7-Day Rolling Trend
    daily_agg = df.groupby("date")["gross_demand_units"].sum().reset_index()
    daily_agg["rolling_7d"] = daily_agg["gross_demand_units"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_agg["date"], daily_agg["gross_demand_units"], color="#b0c4de", alpha=0.5, label="Daily Actual Units")
    ax.plot(daily_agg["date"], daily_agg["rolling_7d"], color="#1f77b4", linewidth=2.0, label="7-Day Rolling Average")
    ax.set_title("Overall Retail Sales Trend (Daily Units & 7-Day Moving Avg)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Gross Demand Units", fontsize=10)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path / "sales_trend.png", dpi=200)
    plt.close(fig)

    # 2. Day-of-Week Seasonality by Category
    df_temp = df.copy()
    df_temp["day_of_week"] = pd.to_datetime(df_temp["date"]).dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    dow_summary = (
        df_temp.groupby(["category", "day_of_week"])["gross_demand_units"]
        .mean()
        .reindex(dow_order, level="day_of_week")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=dow_summary,
        x="day_of_week",
        y="gross_demand_units",
        hue="category",
        marker="o",
        linewidth=2,
        ax=ax,
    )
    ax.set_title("Day-of-Week Demand Seasonality by Product Category", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Day of Week", fontsize=10)
    ax.set_ylabel("Average Daily Units Sold", fontsize=10)
    ax.legend(title="Category", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path / "dow_seasonality.png", dpi=200)
    plt.close(fig)

    # 3. Top SKUs by Revenue
    fig, ax = plt.subplots(figsize=(10, 5))
    top_skus = sku_stats.sort_values(by="total_revenue", ascending=True)
    bars = ax.barh(top_skus["sku_id"], top_skus["total_revenue"] / 1e3, color="#2ca02c", alpha=0.85)
    ax.set_title("Total Revenue by SKU ($ Thousands)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Revenue ($K)", fontsize=10)
    ax.set_ylabel("SKU Identifier", fontsize=10)
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 5, bar.get_y() + bar.get_height() / 2, f"${width:,.1f}K", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path / "top_skus_revenue.png", dpi=200)
    plt.close(fig)

    # 4. ABC-XYZ Matrix Scatter
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = {"A": "#d62728", "B": "#ff7f0e", "C": "#1f77b4"}
    sns.scatterplot(
        data=sku_stats,
        x="cv_demand",
        y="total_revenue",
        hue="abc_class",
        style="xyz_class",
        s=140,
        palette=palette,
        ax=ax,
    )
    # Reference thresholds
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.6, label="X/Y Boundary (CV=0.5)")
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.6, label="Y/Z Boundary (CV=1.0)")
    for _, r in sku_stats.iterrows():
        ax.text(r["cv_demand"] + 0.02, r["total_revenue"], f"{r['sku_id']} ({r['segment']})", fontsize=9)

    ax.set_title("SKU ABC-XYZ Portfolio Segmentation Matrix", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Demand Volatility (Coefficient of Variation CV)", fontsize=10)
    ax.set_ylabel("Total Revenue ($)", fontsize=10)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path / "abc_xyz_matrix.png", dpi=200)
    plt.close(fig)

    print(f"[SUCCESS] All 4 EDA charts successfully exported to: {out_path.resolve()}")


def run_eda_pipeline(
    clean_data_path: str = "data/processed/clean_sales_data.csv",
    output_report_path: str = "reports/sku_abc_xyz_classification.csv",
) -> pd.DataFrame:
    """
    Executes the EDA pipeline, computes ABC-XYZ classes, exports charts,
    and saves the SKU classification table.
    """
    clean_file = Path(clean_data_path)
    if not clean_file.exists():
        raise FileNotFoundError(f"Clean data not found at: {clean_file}. Run preprocessing first.")

    print(f"[INFO] Loading clean dataset from: {clean_file}")
    df = pd.read_csv(clean_file)

    print("[INFO] Computing ABC-XYZ portfolio classification...")
    sku_stats = calculate_abc_xyz_classification(df)

    out_csv = Path(output_report_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sku_stats.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Saved ABC-XYZ classification table to: {out_csv.resolve()}")

    print("[INFO] Generating publication-grade exploratory figures...")
    generate_eda_figures(df, sku_stats)

    return sku_stats


if __name__ == "__main__":
    run_eda_pipeline()
