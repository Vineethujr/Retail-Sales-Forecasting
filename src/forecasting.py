"""
Retail Sales Forecasting & Inventory Optimization System
Module: Sales Forecasting & Model Evaluation (src/forecasting.py)

Purpose:
  - Partitions time-series chronologically into Train, Validation, and Holdout Test sets.
  - Implements benchmark baselines:
      1. Naive (lag-1)
      2. 7-Day Moving Average
      3. Seasonal Naive (7-day lag)
  - Trains a tuned, CPU-efficient Random Forest Regressor on engineered features.
  - Computes industry standard forecasting accuracy metrics:
      - MAE (Mean Absolute Error)
      - RMSE (Root Mean Squared Error)
      - WAPE (Weighted Absolute Percentage Error with zero-demand safeguard)
      - Forecast Bias (Tracking Signal: Under- vs Over-forecasting)
  - Exports trained model artifact, forecast predictions, and evaluation metrics.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Any

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def chronological_split(
    df: pd.DataFrame,
    test_days: int = 28,
    val_days: int = 28,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data chronologically to strictly respect temporal ordering:
      - Train set: [Start Date, End Date - (test_days + val_days)]
      - Validation set: [(End Date - test_days - val_days) + 1, End Date - test_days]
      - Test set: [(End Date - test_days) + 1, End Date]
    """
    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()

    test_start = max_date - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)

    train_df = df[df["date"] < val_start].copy()
    val_df = df[(df["date"] >= val_start) & (df["date"] < test_start)].copy()
    test_df = df[df["date"] >= test_start].copy()

    print(f"[INFO] Chronological Split Summary:")
    print(f"       Train:      {train_df['date'].min().strftime('%Y-%m-%d')} to {train_df['date'].max().strftime('%Y-%m-%d')} ({len(train_df):,} rows)")
    print(f"       Validation: {val_df['date'].min().strftime('%Y-%m-%d')} to {val_df['date'].max().strftime('%Y-%m-%d')} ({len(val_df):,} rows)")
    print(f"       Test:       {test_df['date'].min().strftime('%Y-%m-%d')} to {test_df['date'].max().strftime('%Y-%m-%d')} ({len(test_df):,} rows)")

    return train_df, val_df, test_df


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes retail forecasting performance metrics:
      - MAE: Absolute volume error
      - RMSE: Penalizes large outliers
      - WAPE: Total Absolute Error / Total Actual Demand (Volume weighted)
      - Bias: Total Prediction / Total Actual - 1.0 (Signed percentage over/under-forecast)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Ensure predictions are strictly non-negative (demand cannot be negative)
    y_pred = np.clip(y_pred, 0, None)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    sum_actual = np.sum(y_true)
    sum_abs_err = np.sum(np.abs(y_true - y_pred))

    # WAPE zero-demand division safeguard
    if sum_actual > 0:
        wape = (sum_abs_err / sum_actual) * 100.0
        bias = ((np.sum(y_pred) - sum_actual) / sum_actual) * 100.0
    else:
        wape = 0.0 if sum_abs_err == 0 else 100.0
        bias = 0.0

    return {
        "MAE": round(mae, 3),
        "RMSE": round(rmse, 3),
        "WAPE_pct": round(wape, 2),
        "Bias_pct": round(bias, 2),
    }


def evaluate_baselines(test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Evaluates standard retail heuristic benchmarks on the test set:
      1. Naive: Predicts yesterday's demand (lag_1)
      2. 7-Day Moving Average: Predicts rolling 7-day mean
      3. Seasonal Naive: Predicts same day last week (lag_7)
    """
    y_true = test_df["gross_demand_units"].values

    # Naive baseline
    pred_naive = test_df["lag_1"].fillna(0).values
    metrics_naive = calculate_metrics(y_true, pred_naive)

    # 7-day Moving Average baseline
    pred_ma7 = test_df["rolling_mean_7"].fillna(0).values
    metrics_ma7 = calculate_metrics(y_true, pred_ma7)

    # Seasonal Naive baseline (lag_7)
    pred_snaive = test_df["lag_7"].fillna(0).values
    metrics_snaive = calculate_metrics(y_true, pred_snaive)

    return {
        "Naive (Lag-1)": metrics_naive,
        "7-Day Moving Avg": metrics_ma7,
        "Seasonal Naive (Lag-7)": metrics_snaive,
    }


def train_and_evaluate_forecaster(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    model_output_path: str = "models/random_forest_forecaster.joblib",
) -> Tuple[Any, pd.DataFrame, pd.DataFrame]:
    """
    Trains RandomForestRegressor, computes performance metrics against baselines,
    and returns:
      (trained_model, metrics_comparison_df, test_predictions_df)
    """
    X_train = train_df[feature_cols]
    y_train = train_df["gross_demand_units"].values

    X_val = val_df[feature_cols]
    y_val = val_df["gross_demand_units"].values

    X_test = test_df[feature_cols]
    y_test = test_df["gross_demand_units"].values

    print("\n[INFO] Fitting Random Forest Regressor on training set...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    # Predictions
    pred_train = np.clip(rf_model.predict(X_train), 0, None)
    pred_val = np.clip(rf_model.predict(X_val), 0, None)
    pred_test = np.clip(rf_model.predict(X_test), 0, None)

    metrics_train = calculate_metrics(y_train, pred_train)
    metrics_val = calculate_metrics(y_val, pred_val)
    metrics_rf_test = calculate_metrics(y_test, pred_test)

    # Evaluate Baselines on the exact same test set
    baseline_metrics = evaluate_baselines(test_df)

    # Construct Metrics Comparison Table
    records = []
    for name, m in baseline_metrics.items():
        records.append({
            "Model": name,
            "Evaluation_Set": "Holdout Test",
            **m,
        })

    records.append({
        "Model": "Random Forest Regressor",
        "Evaluation_Set": "Holdout Test",
        **metrics_rf_test,
    })
    records.append({
        "Model": "Random Forest (Val Check)",
        "Evaluation_Set": "Validation",
        **metrics_val,
    })
    records.append({
        "Model": "Random Forest (Train Check)",
        "Evaluation_Set": "Train",
        **metrics_train,
    })

    metrics_df = pd.DataFrame(records)

    # Save Model Artifact
    out_model = Path(model_output_path)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_model, out_model)
    print(f"[SUCCESS] Model artifact saved to: {out_model.resolve()}")

    # Build Test Predictions DataFrame for downstream inventory and visualization
    test_results = test_df[["date", "store_id", "sku_id", "category", "unit_price", "unit_cost", "gross_demand_units"]].copy()
    test_results["predicted_demand"] = pred_test
    test_results["forecast_error"] = test_results["predicted_demand"] - test_results["gross_demand_units"]
    test_results["abs_error"] = np.abs(test_results["forecast_error"])

    return rf_model, metrics_df, test_results


def plot_forecast_vs_actuals(test_results: pd.DataFrame, output_fig_path: str = "reports/figures/forecast_vs_actuals.png") -> None:
    """
    Plots aggregate daily test predictions vs actual sales.
    """
    out_file = Path(output_fig_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    daily_comp = (
        test_results.groupby("date")
        .agg(
            actual_units=("gross_demand_units", "sum"),
            predicted_units=("predicted_demand", "sum"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily_comp["date"], daily_comp["actual_units"], marker="o", color="#1f77b4", label="Actual Gross Units", linewidth=2.0)
    ax.plot(daily_comp["date"], daily_comp["predicted_units"], marker="s", color="#ff7f0e", linestyle="--", label="Random Forest Forecast", linewidth=2.0)
    ax.set_title("Holdout Test Period: Actual vs. Forecasted Daily Sales", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Total Demand Units", fontsize=10)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_file, dpi=200)
    plt.close(fig)
    print(f"[SUCCESS] Forecast evaluation chart saved to: {out_file.resolve()}")


def run_forecasting_pipeline(
    clean_data_path: str = "data/processed/clean_sales_data.csv",
    output_metrics_path: str = "reports/forecast_metrics.csv",
    output_predictions_path: str = "reports/forecast_predictions.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrates full feature extraction, train/val/test splitting,
    model training, baseline comparison, and metric generation.
    """
    from src.feature_engineering import build_feature_pipeline

    print(f"[INFO] Loading dataset for forecasting: {clean_data_path}")
    df_clean = pd.read_csv(clean_data_path)

    print("[INFO] Constructing feature pipeline...")
    df_feat, feature_cols = build_feature_pipeline(df_clean)

    train_df, val_df, test_df = chronological_split(df_feat, test_days=28, val_days=28)

    model, metrics_df, test_results = train_and_evaluate_forecaster(
        train_df, val_df, test_df, feature_cols
    )

    # Save metrics & predictions
    Path(output_metrics_path).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_metrics_path, index=False)
    test_results.to_csv(output_predictions_path, index=False)

    print("\n" + "=" * 60)
    print(" FORECAST MODEL BENCHMARK COMPARISON (Holdout Test Set)")
    print("=" * 60)
    print(metrics_df.to_string(index=False))
    print("=" * 60)

    plot_forecast_vs_actuals(test_results)

    return metrics_df, test_results


if __name__ == "__main__":
    run_forecasting_pipeline()
