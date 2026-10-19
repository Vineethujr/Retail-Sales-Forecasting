"""
Retail Sales Forecasting & Inventory Optimization System
Module: Inventory Optimization & Replenishment (src/inventory_optimization.py)

Purpose:
  - Ingests demand forecasts and error variance from the forecasting module.
  - Implements industry inventory control formulas:
      - Safety Stock (SS) = Z * sqrt(Lead_Time) * sigma_d
      - Reorder Point (ROP) = Lead_Time_Demand + SS
      - Economic Order Quantity (EOQ) = sqrt(2 * D * S / H)
      - Target Stock Level (Order-Up-To S) = ROP + EOQ
  - Enforces discrete operational constraints:
      - Minimum Order Quantity (MOQ)
      - Case Pack rounding (ceil(Q / case_pack) * case_pack)
  - Evaluates stock health metrics: Days of Supply (DoS), Stockout and Overstock risk.
  - Exports actionable purchase replenishment orders to reports/inventory_replenishment_orders.csv.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import math
import yaml
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_inventory_config(config_path: str = "config/inventory_config.yaml") -> Dict[str, Any]:
    """
    Loads operational parameters, service levels, and cost defaults from YAML.
    """
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        # Fallback safe defaults if file is missing
        return {
            "simulation_defaults": {
                "supplier_lead_time_days": 7,
                "service_level_target": 0.95,
                "z_score": 1.65,
                "initial_stock_weeks": 2.0,
                "default_unit_cost": 15.0,
                "annual_holding_cost_rate": 0.20,
                "fixed_order_cost": 50.0,
                "review_period_days": 7,
            },
            "operational_constraints": {
                "min_order_quantity": 10,
                "case_pack_size": 6,
            },
            "decision_thresholds": {
                "stockout_lead_time_ratio": 0.5,
                "overstock_days_of_supply": 60.0,
                "min_history_days": 14,
            },
        }

    with open(cfg_file, "r") as f:
        return yaml.safe_load(f)


def calculate_safety_stock(
    daily_demand_std: float,
    lead_time_days: int,
    z_score: float = 1.65,
) -> float:
    """
    Safety Stock (SS) = Z * sigma_L = Z * sqrt(L) * sigma_d
    Protects against demand surges during supplier replenishment lead time.
    """
    sigma_l = math.sqrt(lead_time_days) * daily_demand_std
    return z_score * sigma_l


def calculate_reorder_point(
    mean_daily_demand: float,
    lead_time_days: int,
    safety_stock: float,
) -> float:
    """
    Reorder Point (ROP) = Expected Lead Time Demand + Safety Stock
    ROP = (d_mean * L) + SS
    """
    lead_time_demand = mean_daily_demand * lead_time_days
    return lead_time_demand + safety_stock


def calculate_eoq(
    annual_demand: float,
    order_cost: float,
    unit_cost: float,
    holding_rate: float = 0.20,
) -> float:
    """
    Economic Order Quantity (Wilson Formula):
    EOQ = sqrt((2 * D * S) / H)
    where:
      D = Annual Demand Units
      S = Fixed Order Cost ($ per purchase order)
      H = Unit Annual Holding Cost ($ = unit_cost * holding_rate)
    """
    holding_cost = max(0.01, unit_cost * holding_rate)
    if annual_demand <= 0 or order_cost <= 0:
        return 1.0
    eoq = math.sqrt((2.0 * annual_demand * order_cost) / holding_cost)
    return max(1.0, eoq)


def apply_order_constraints(
    raw_order_qty: float,
    moq: int = 1,
    case_pack: int = 1,
) -> int:
    """
    Applies Minimum Order Quantity (MOQ) and Case Pack discrete packaging rules:
      - If raw_order_qty <= 0 -> 0
      - If raw_order_qty > 0 -> max(raw_order_qty, MOQ)
      - Rounded UP to nearest multiple of case_pack
    """
    if raw_order_qty <= 0:
        return 0

    order_qty = max(raw_order_qty, float(moq))
    pack_size = max(1, case_pack)
    constrained_qty = int(math.ceil(order_qty / pack_size) * pack_size)
    return constrained_qty


def optimize_inventory_portfolio(
    predictions_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    simulated_on_hand_seed: int = 42,
) -> pd.DataFrame:
    """
    Computes ROP, Safety Stock, EOQ, and Recommended Order Quantities
    for every (store_id, sku_id) in the portfolio.
    """
    if config is None:
        config = load_inventory_config()

    sim_cfg = config.get("simulation_defaults", {})
    ops_cfg = config.get("operational_constraints", {})
    thr_cfg = config.get("decision_thresholds", {})

    lead_time = int(sim_cfg.get("supplier_lead_time_days", 7))
    z_score = float(sim_cfg.get("z_score", 1.65))
    holding_rate = float(sim_cfg.get("annual_holding_cost_rate", 0.20))
    order_cost = float(sim_cfg.get("fixed_order_cost", 50.0))
    moq = int(ops_cfg.get("min_order_quantity", 5))
    case_pack = int(ops_cfg.get("case_pack_size", 6))
    stockout_ratio = float(thr_cfg.get("stockout_lead_time_ratio", 0.5))
    overstock_days = float(thr_cfg.get("overstock_days_of_supply", 60.0))

    np.random.seed(simulated_on_hand_seed)

    # Aggregate predictions and variability per Store and SKU
    sku_groups = (
        predictions_df.groupby(["store_id", "sku_id"])
        .agg(
            category=("category", "first"),
            unit_price=("unit_price", "mean"),
            unit_cost=("unit_cost", "mean"),
            mean_forecast_demand=("predicted_demand", "mean"),
            std_forecast_demand=("predicted_demand", "std"),
            rmse_error=("forecast_error", lambda e: np.sqrt(np.mean(e ** 2))),
        )
        .reset_index()
    )

    results = []

    for _, row in sku_groups.iterrows():
        store_id = row["store_id"]
        sku_id = row["sku_id"]
        category = row["category"]
        unit_price = float(row["unit_price"])
        unit_cost = float(row["unit_cost"])
        daily_d = max(0.1, float(row["mean_forecast_demand"]))
        
        # Use RMSE as demand uncertainty if std is 0
        sigma_d = float(row["rmse_error"]) if row["rmse_error"] > 0 else max(1.0, float(row["std_forecast_demand"]))

        # 1. Lead Time Demand
        lead_time_demand = daily_d * lead_time

        # 2. Safety Stock
        safety_stock = calculate_safety_stock(sigma_d, lead_time, z_score)

        # 3. Reorder Point
        rop = calculate_reorder_point(daily_d, lead_time, safety_stock)

        # 4. Annual Demand & EOQ
        annual_demand = daily_d * 365.0
        eoq = calculate_eoq(annual_demand, order_cost, unit_cost, holding_rate)

        # 5. Order-Up-To Level (S_max)
        target_stock_level = rop + eoq

        # 6. Current On-Hand Stock Simulation:
        # We simulate realistic inventory positions: some healthy, some low, some overstocked
        stock_scenario = np.random.choice(["low", "healthy", "high"], p=[0.25, 0.55, 0.20])
        if stock_scenario == "low":
            current_on_hand = int(np.random.uniform(0.1 * rop, 0.8 * rop))
        elif stock_scenario == "high":
            current_on_hand = int(np.random.uniform(1.5 * rop, 3.0 * rop))
        else:
            current_on_hand = int(np.random.uniform(0.9 * rop, 1.4 * rop))

        current_on_hand = max(0, current_on_hand)
        on_order = 0  # Assuming no pending transit orders for MVP
        inventory_position = current_on_hand + on_order

        # 7. Replenishment Decision
        reorder_triggered = inventory_position <= rop
        if reorder_triggered:
            raw_order = target_stock_level - inventory_position
            order_qty = apply_order_constraints(raw_order, moq=moq, case_pack=case_pack)
        else:
            raw_order = 0.0
            order_qty = 0

        # 8. Days of Supply & Risk Alerts
        days_of_supply = round(current_on_hand / daily_d, 1)

        if days_of_supply < (stockout_ratio * lead_time):
            health_status = "CRITICAL (Stockout Imminent)"
            action = "EMERGENCY REORDER"
        elif reorder_triggered:
            health_status = "REORDER NEEDED"
            action = "PLACE PURCHASE ORDER"
        elif days_of_supply > overstock_days:
            health_status = "OVERSTOCKED"
            action = "PAUSE PURCHASES / PROMO LIQUIDATION"
        else:
            health_status = "HEALTHY"
            action = "MONITOR"

        order_cost_total = order_qty * unit_cost
        stock_value_on_hand = current_on_hand * unit_cost

        results.append({
            "store_id": store_id,
            "sku_id": sku_id,
            "category": category,
            "unit_cost": round(unit_cost, 2),
            "unit_price": round(unit_price, 2),
            "daily_forecast": round(daily_d, 2),
            "lead_time_days": lead_time,
            "safety_stock": int(math.ceil(safety_stock)),
            "reorder_point": int(math.ceil(rop)),
            "eoq": int(math.ceil(eoq)),
            "target_stock_level": int(math.ceil(target_stock_level)),
            "current_on_hand": current_on_hand,
            "inventory_position": inventory_position,
            "reorder_triggered": reorder_triggered,
            "recommended_order_qty": order_qty,
            "days_of_supply": days_of_supply,
            "health_status": health_status,
            "recommended_action": action,
            "order_value_usd": round(order_cost_total, 2),
            "on_hand_value_usd": round(stock_value_on_hand, 2),
        })

    plan_df = pd.DataFrame(results)
    plan_df = plan_df.sort_values(by=["health_status", "days_of_supply"]).reset_index(drop=True)
    return plan_df


def run_inventory_pipeline(
    predictions_path: str = "reports/forecast_predictions.csv",
    output_plan_path: str = "reports/inventory_replenishment_orders.csv",
) -> pd.DataFrame:
    """
    Executes inventory optimization pipeline and saves purchase order plan.
    """
    pred_file = Path(predictions_path)
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions file not found at: {pred_file}. Run forecasting first.")

    print(f"[INFO] Ingesting forecast predictions from: {pred_file}")
    preds_df = pd.read_csv(pred_file)

    config = load_inventory_config()
    print("[INFO] Optimizing portfolio replenishment and safety stock...")
    plan_df = optimize_inventory_portfolio(preds_df, config)

    out_file = Path(output_plan_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    plan_df.to_csv(out_file, index=False)

    print(f"[SUCCESS] Replenishment plan saved to: {out_file.resolve()}")
    print("\n" + "=" * 80)
    print(" INVENTORY REPLENISHMENT PLAN & PORTFOLIO HEALTH SUMMARY")
    print("=" * 80)
    summary_cols = [
        "store_id", "sku_id", "daily_forecast", "safety_stock", "reorder_point",
        "current_on_hand", "days_of_supply", "recommended_order_qty", "health_status"
    ]
    print(plan_df[summary_cols].to_string(index=False))
    print("=" * 80)

    total_order_val = plan_df["order_value_usd"].sum()
    total_on_hand_val = plan_df["on_hand_value_usd"].sum()
    stockout_count = (plan_df["health_status"] == "CRITICAL (Stockout Imminent)").sum()
    reorder_count = plan_df["reorder_triggered"].sum()

    print(f"[METRICS] Total On-Hand Capital: ${total_on_hand_val:,.2f}")
    print(f"[METRICS] Total Recommended Order Capital: ${total_order_val:,.2f}")
    print(f"[METRICS] SKUs Triggering Reorder: {reorder_count} / {len(plan_df)}")
    print(f"[METRICS] SKUs at Critical Stockout Risk: {stockout_count}")

    return plan_df


if __name__ == "__main__":
    run_inventory_pipeline()
