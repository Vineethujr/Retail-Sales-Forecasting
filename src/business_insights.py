"""
Retail Sales Forecasting & Inventory Optimization System
Module: Business Insights & Executive Reporting (src/business_insights.py)

Purpose:
  - Synthesizes forecasting accuracy and inventory replenishment analytics.
  - Translates technical data science results into actionable business recommendations.
  - Quantifies working capital, stockout exposure, and potential holding cost savings.
  - Generates a comprehensive markdown report at reports/executive_inventory_summary.md.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd


def generate_executive_report(
    inventory_plan_path: str = "reports/inventory_replenishment_orders.csv",
    metrics_path: str = "reports/forecast_metrics.csv",
    classification_path: str = "reports/sku_abc_xyz_classification.csv",
    output_report_path: str = "reports/executive_inventory_summary.md",
) -> str:
    """
    Builds a structured executive summary markdown document detailing:
      - Portfolio Health & Capital Allocation
      - Emergency Stockout Mitigation Orders
      - Forecast Model Performance vs Heuristic Benchmarks
      - Working Capital and Holding Cost Optimization
    """
    plan_df = pd.read_csv(inventory_plan_path)
    metrics_df = pd.read_csv(metrics_path)
    abc_df = pd.read_csv(classification_path)

    # Core Portfolio KPIs
    total_skus = len(plan_df)
    total_on_hand_val = plan_df["on_hand_value_usd"].sum()
    total_order_val = plan_df["order_value_usd"].sum()
    stockout_skus = plan_df[plan_df["health_status"].str.contains("CRITICAL", na=False)]
    reorder_skus = plan_df[plan_df["reorder_triggered"] == True]
    overstock_skus = plan_df[plan_df["health_status"].str.contains("OVERSTOCK", na=False)]

    # Best model metric
    rf_metrics = metrics_df[metrics_df["Model"] == "Random Forest Regressor"].iloc[0]
    naive_metrics = metrics_df[metrics_df["Model"] == "Naive (Lag-1)"].iloc[0]

    wape_improvement = naive_metrics["WAPE_pct"] - rf_metrics["WAPE_pct"]

    report_content = f"""# Executive Decision Brief: Retail Sales Forecasting & Inventory Optimization
**Generated:** Automated Decision Support Engine  
**Portfolio Scope:** {total_skus} Active Store-SKU Combinations across 2 Retail Locations  

---

## 1. Executive Summary & Core Portfolio KPIs

| Business Metric | Performance Value | Strategic Implication |
| :--- | :--- | :--- |
| **Current On-Hand Working Capital** | **${total_on_hand_val:,.2f}** | Total capital currently tied up in physical warehouse/store inventory. |
| **Recommended Purchase Order Capital** | **${total_order_val:,.2f}** | Total capital required to fulfill immediate replenishment requirements. |
| **SKUs Triggering Purchase Orders** | **{len(reorder_skus)} of {total_skus} ({len(reorder_skus)/total_skus*100:.1f}%)** | Immediate supplier reorder points breached ($IP \\le ROP$). |
| **Critical Stockout Risk SKUs** | **{len(stockout_skus)} SKUs** | Days of Supply $< 0.5 \\times Lead Time$; imminent stockout within 3.5 days. |
| **Forecasting Model Accuracy (WAPE)** | **{rf_metrics['WAPE_pct']:.2f}% (vs {naive_metrics['WAPE_pct']:.2f}% Naive)** | **+{wape_improvement:.1f} percentage points** error reduction over naive heuristic. |

---

## 2. Critical Stockout Risks (Immediate Action Required)

The following SKUs have breached critical threshold levels and require expedited purchase order placement to prevent lost revenue and customer dissatisfaction:

| Store | SKU | Category | Days of Supply | Current Stock | Reorder Point | Recommended Order | Purchase Value | Action Plan |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for _, row in stockout_skus.iterrows():
        report_content += (
            f"| **{row['store_id']}** | **{row['sku_id']}** | {row['category']} | "
            f"**{row['days_of_supply']:.1f} days** | {row['current_on_hand']} units | "
            f"{row['reorder_point']} units | **{row['recommended_order_qty']:,} units** | "
            f"${row['order_value_usd']:,.2f} | Expedite PO Placement |\n"
        )

    report_content += f"""
---

## 3. Forecast Accuracy Benchmark Comparison (Holdout Test Set)

Machine learning forecasting significantly outperforms traditional retail rule-of-thumb heuristics, reducing demand uncertainty ($\sigma_L$) and preventing both stockouts and excessive safety stock buildup.

| Forecast Strategy | Evaluation Grain | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | WAPE Error (%) | Forecast Bias (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""

    for _, row in metrics_df.iterrows():
        report_content += (
            f"| {row['Model']} | {row['Evaluation_Set']} | "
            f"{row['MAE']:.2f} units | {row['RMSE']:.2f} units | "
            f"**{row['WAPE_pct']:.2f}%** | {row['Bias_pct']:+.2f}% |\n"
        )

    report_content += f"""
---

## 4. Strategic Inventory Optimization Recommendations

1. **Expedited Purchase Orders for Class A Products**:
   Focus immediate working capital on high-velocity items triggering orders to protect customer fill rate at the 95% target service level ($Z=1.65$).
2. **Safety Stock Reduction through Enhanced Accuracy**:
   By improving forecast WAPE from **{naive_metrics['WAPE_pct']:.2f}% to {rf_metrics['WAPE_pct']:.2f}%**, demand uncertainty ($\sigma_d$) is trimmed by ~{wape_improvement/naive_metrics['WAPE_pct']*100:.1f}%, directly reducing required buffer inventory without increasing stockout risk.
3. **Discrete Packaging Compliance**:
   All recommended purchase orders strictly respect Minimum Order Quantities (MOQ) and Case Pack sizes, preventing supplier order rejections.
4. **Interactive Dashboard Monitoring**:
   Store managers and procurement planners can interactively simulate lead time fluctuations and target service levels via the Streamlit web portal:
   ```bash
   python main.py --launch-dashboard
   ```

---
*Report generated by the Retail Sales Forecasting & Inventory Optimization System.*
"""

    out_file = Path(output_report_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report_content, encoding="utf-8")
    print(f"[SUCCESS] Executive report saved to: {out_file.resolve()}")
    return report_content


if __name__ == "__main__":
    generate_executive_report()
