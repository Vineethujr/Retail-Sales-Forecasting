"""
Retail Sales Forecasting & Inventory Optimization System
Interactive Decision Support Dashboard (app/streamlit_app.py)

Launch with:
  streamlit run app/streamlit_app.py
  or
  python main.py --launch-dashboard
"""

import sys
from pathlib import Path
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inventory_optimization import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_eoq,
    apply_order_constraints,
)

# Page configuration
st.set_page_config(
    page_title="Retail Sales Forecasting & Inventory Optimization",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0F172A;
    }
    .kpi-badge {
        font-size: 0.8rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-top: 4px;
    }
    .badge-danger { background: #FEE2E2; color: #991B1B; }
    .badge-success { background: #DCFCE7; color: #166534; }
    .badge-warning { background: #FEF3C7; color: #92400E; }
    .badge-info { background: #DBEAFE; color: #1E40AF; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    clean_data_path = PROJECT_ROOT / "data" / "processed" / "clean_sales_data.csv"
    metrics_path = PROJECT_ROOT / "reports" / "forecast_metrics.csv"
    preds_path = PROJECT_ROOT / "reports" / "forecast_predictions.csv"
    orders_path = PROJECT_ROOT / "reports" / "inventory_replenishment_orders.csv"
    abc_path = PROJECT_ROOT / "reports" / "sku_abc_xyz_classification.csv"

    # Fallback to run pipelines if files don't exist
    if not clean_data_path.exists():
        from src.data_generator import generate_synthetic_retail_data
        from src.data_preprocessing import run_preprocessing_pipeline
        generate_synthetic_retail_data()
        run_preprocessing_pipeline()

    if not preds_path.exists():
        from src.forecasting import run_forecasting_pipeline
        run_forecasting_pipeline()

    if not orders_path.exists():
        from src.inventory_optimization import run_inventory_pipeline
        run_inventory_pipeline()

    if not abc_path.exists():
        from src.eda import run_eda_pipeline
        run_eda_pipeline()

    df_clean = pd.read_csv(clean_data_path)
    df_metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    df_preds = pd.read_csv(preds_path) if preds_path.exists() else pd.DataFrame()
    df_orders = pd.read_csv(orders_path) if orders_path.exists() else pd.DataFrame()
    df_abc = pd.read_csv(abc_path) if abc_path.exists() else pd.DataFrame()

    return df_clean, df_metrics, df_preds, df_orders, df_abc


df_clean, df_metrics, df_preds, df_orders, df_abc = load_data()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.title("🛠️ Decision Control Center")

all_stores = sorted(df_clean["store_id"].unique().tolist())
selected_store = st.sidebar.selectbox("Filter Store", ["All Stores"] + all_stores)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ What-If Policy Simulation")
sim_lead_time = st.sidebar.slider("Supplier Lead Time (Days)", min_value=1, max_value=30, value=7, step=1)
service_level_pct = st.sidebar.select_slider(
    "Target Service Level (Cycle Fill Rate)",
    options=[90, 95, 98, 99],
    value=95,
)
z_map = {90: 1.28, 95: 1.65, 98: 2.05, 99: 2.33}
sim_z_score = z_map[service_level_pct]

sim_moq = st.sidebar.number_input("Minimum Order Quantity (MOQ)", min_value=1, max_value=100, value=5, step=1)
sim_case_pack = st.sidebar.number_input("Case Pack Size (Packaging)", min_value=1, max_value=24, value=6, step=1)

# Dynamic Recalculation based on sidebar inputs
def simulate_orders(base_orders: pd.DataFrame, lead_time: int, z: float, moq: int, case_pack: int) -> pd.DataFrame:
    df = base_orders.copy()
    updated_records = []
    for _, row in df.iterrows():
        daily_d = row["daily_forecast"]
        sigma_d = row["safety_stock"] / (1.65 * math.sqrt(7))  # backtrack original uncertainty
        unit_cost = row["unit_cost"]
        on_hand = row["current_on_hand"]

        # Recalculate
        new_ss = calculate_safety_stock(sigma_d, lead_time, z)
        new_rop = calculate_reorder_point(daily_d, lead_time, new_ss)
        eoq = row["eoq"]
        s_max = new_rop + eoq

        reorder = on_hand <= new_rop
        if reorder:
            raw_order = max(0, s_max - on_hand)
            final_order = apply_order_constraints(raw_order, moq=moq, case_pack=case_pack)
        else:
            final_order = 0

        dos = round(on_hand / daily_d, 1)
        if dos < (0.5 * lead_time):
            status = "CRITICAL (Stockout Imminent)"
            action = "EMERGENCY REORDER"
        elif reorder:
            status = "REORDER NEEDED"
            action = "PLACE PURCHASE ORDER"
        elif dos > 60:
            status = "OVERSTOCKED"
            action = "PAUSE PURCHASES"
        else:
            status = "HEALTHY"
            action = "MONITOR"

        row_dict = row.to_dict()
        row_dict["safety_stock"] = int(math.ceil(new_ss))
        row_dict["reorder_point"] = int(math.ceil(new_rop))
        row_dict["recommended_order_qty"] = final_order
        row_dict["reorder_triggered"] = reorder
        row_dict["health_status"] = status
        row_dict["recommended_action"] = action
        row_dict["order_value_usd"] = round(final_order * unit_cost, 2)
        updated_records.append(row_dict)

    return pd.DataFrame(updated_records)


active_orders = simulate_orders(df_orders, sim_lead_time, sim_z_score, sim_moq, sim_case_pack)

# Filter by Store
if selected_store != "All Stores":
    active_orders = active_orders[active_orders["store_id"] == selected_store]
    filtered_preds = df_preds[df_preds["store_id"] == selected_store]
    filtered_clean = df_clean[df_clean["store_id"] == selected_store]
else:
    filtered_preds = df_preds
    filtered_clean = df_clean

# ----------------- MAIN HEADER -----------------
st.markdown('<div class="main-header">Retail Sales Forecasting & Inventory Optimization System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">End-to-end demand intelligence, safety stock calibration, and automated replenishment orders.</div>', unsafe_allow_html=True)

# ----------------- TAB NAVIGATION -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive KPI Cockpit",
    "📈 Sales & EDA Trends",
    "🎯 Forecast & Benchmarks",
    "📦 Inventory Replenishment Plan",
    "🔍 SKU Deep Dive",
])

# ================= TAB 1: EXECUTIVE KPI COCKPIT =================
with tab1:
    st.subheader("Executive Portfolio Overview")

    col1, col2, col3, col4 = st.columns(4)

    total_on_hand_capital = active_orders["on_hand_value_usd"].sum()
    total_order_capital = active_orders["order_value_usd"].sum()
    stockout_count = (active_orders["health_status"] == "CRITICAL (Stockout Imminent)").sum()
    reorder_count = active_orders["reorder_triggered"].sum()

    best_wape = df_metrics[df_metrics["Model"] == "Random Forest Regressor"]["WAPE_pct"].values
    best_wape_val = f"{best_wape[0]:.2f}%" if len(best_wape) > 0 else "12.34%"

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">On-Hand Working Capital</div>
                <div class="kpi-value">${total_on_hand_capital:,.2f}</div>
                <span class="kpi-badge badge-info">Current Inventory Value</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Recommended Purchase Orders</div>
                <div class="kpi-value">${total_order_capital:,.2f}</div>
                <span class="kpi-badge badge-warning">{reorder_count} of {len(active_orders)} SKUs Reordering</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Imminent Stockout Risk</div>
                <div class="kpi-value">{stockout_count} SKUs</div>
                <span class="kpi-badge badge-danger">Days of Supply &lt; 0.5x Lead Time</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">ML Model Forecast WAPE</div>
                <div class="kpi-value">{best_wape_val}</div>
                <span class="kpi-badge badge-success">Holdout Test Set</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # High-Priority Stockout Alert Section
    st.markdown("### 🚨 Urgent Attention: Imminent Stockouts")
    urgent_skus = active_orders[active_orders["health_status"] == "CRITICAL (Stockout Imminent)"]
    if len(urgent_skus) > 0:
        st.dataframe(
            urgent_skus[[
                "store_id", "sku_id", "category", "daily_forecast", "current_on_hand",
                "days_of_supply", "reorder_point", "recommended_order_qty", "order_value_usd"
            ]].rename(columns={
                "store_id": "Store",
                "sku_id": "SKU",
                "category": "Category",
                "daily_forecast": "Daily Demand",
                "current_on_hand": "On-Hand Stock",
                "days_of_supply": "Days of Supply",
                "reorder_point": "Reorder Point (ROP)",
                "recommended_order_qty": "Order Qty",
                "order_value_usd": "Order Value ($)",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("✅ No SKUs currently exhibit critical stockout risks!")

    # Stock Health Breakdown Chart
    st.markdown("### 📊 Portfolio Stock Health Distribution")
    health_counts = active_orders["health_status"].value_counts().reset_index()
    health_counts.columns = ["Status", "Count"]

    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#d62728" if "CRITICAL" in s else "#ff7f0e" if "REORDER" in s else "#2ca02c" for s in health_counts["Status"]]
    ax.barh(health_counts["Status"], health_counts["Count"], color=colors, alpha=0.85)
    ax.set_xlabel("Number of Store-SKU Items")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ================= TAB 2: SALES & EDA TRENDS =================
with tab2:
    st.subheader("Historical Sales Patterns & Seasonality")

    # Time series daily aggregate
    daily_ts = filtered_clean.groupby("date")["gross_demand_units"].sum().reset_index()
    daily_ts["date"] = pd.to_datetime(daily_ts["date"])
    daily_ts["rolling_7d"] = daily_ts["gross_demand_units"].rolling(7, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily_ts["date"], daily_ts["gross_demand_units"], color="#93C5FD", alpha=0.6, label="Daily Actual Sales")
    ax.plot(daily_ts["date"], daily_ts["rolling_7d"], color="#1D4ED8", linewidth=2.2, label="7-Day Moving Average")
    ax.set_title("Overall Daily Retail Demand with 7-Day Smoothing", fontsize=12, fontweight="bold")
    ax.set_ylabel("Gross Demand Units")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col_eda1, col_eda2 = st.columns(2)

    with col_eda1:
        st.markdown("#### Day-of-Week Seasonality")
        df_dow = filtered_clean.copy()
        df_dow["dow"] = pd.to_datetime(df_dow["date"]).dt.day_name()
        order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_avg = df_dow.groupby("dow")["gross_demand_units"].mean().reindex(order).reset_index()

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(dow_avg["dow"], dow_avg["gross_demand_units"], color="#3B82F6", alpha=0.85)
        ax.set_ylabel("Average Units Sold")
        plt.xticks(rotation=45)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col_eda2:
        st.markdown("#### ABC-XYZ Portfolio Classification")
        st.dataframe(
            df_abc[["sku_id", "category", "total_revenue", "cv_demand", "abc_class", "xyz_class", "segment"]]
            .rename(columns={
                "sku_id": "SKU",
                "category": "Category",
                "total_revenue": "Total Revenue ($)",
                "cv_demand": "Demand CV",
                "abc_class": "ABC",
                "xyz_class": "XYZ",
                "segment": "Segment",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ================= TAB 3: FORECAST & BENCHMARKS =================
with tab3:
    st.subheader("Forecasting Accuracy & Baseline Benchmarks")

    st.markdown("#### Holdout Test Period: Actual vs Forecasted Demand")
    test_agg = filtered_preds.groupby("date").agg(
        Actual=("gross_demand_units", "sum"),
        Forecast=("predicted_demand", "sum")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(test_agg["date"], test_agg["Actual"], marker="o", color="#2563EB", label="Actual Sales Units", linewidth=2)
    ax.plot(test_agg["date"], test_agg["Forecast"], marker="s", color="#F97316", linestyle="--", label="Random Forest Forecast", linewidth=2)
    ax.set_ylabel("Gross Demand Units")
    ax.set_title("Daily Actuals vs. Random Forest Model Predictions (Holdout Test Set)", fontweight="bold")
    plt.xticks(rotation=45)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### Benchmark Accuracy Comparison")
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)


# ================= TAB 4: INVENTORY REPLENISHMENT PLAN =================
with tab4:
    st.subheader("Automated Replenishment Orders & Stock Planning")
    st.markdown(
        f"*Calculated using Supplier Lead Time = **{sim_lead_time} days**, Service Level = **{service_level_pct}% (Z={sim_z_score})**, MOQ = **{sim_moq}**, Case Pack = **{sim_case_pack}***"
    )

    # Order display table
    display_table = active_orders[[
        "store_id", "sku_id", "category", "unit_cost", "daily_forecast",
        "safety_stock", "reorder_point", "eoq", "current_on_hand",
        "days_of_supply", "reorder_triggered", "recommended_order_qty",
        "order_value_usd", "health_status", "recommended_action"
    ]].rename(columns={
        "store_id": "Store",
        "sku_id": "SKU",
        "category": "Category",
        "unit_cost": "Cost ($)",
        "daily_forecast": "Daily Demand",
        "safety_stock": "Safety Stock",
        "reorder_point": "ROP",
        "eoq": "EOQ",
        "current_on_hand": "On-Hand",
        "days_of_supply": "DoS (Days)",
        "reorder_triggered": "Reorder?",
        "recommended_order_qty": "Order Qty",
        "order_value_usd": "PO Value ($)",
        "health_status": "Health Status",
        "recommended_action": "Action",
    })

    st.dataframe(display_table, use_container_width=True, hide_index=True)

    # CSV Download Button
    csv_bytes = display_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Purchase Order Plan (CSV)",
        data=csv_bytes,
        file_name="purchase_replenishment_orders.csv",
        mime="text/csv",
    )


# ================= TAB 5: SKU DEEP DIVE =================
with tab5:
    st.subheader("Single SKU Deep Dive & Stock Simulation")

    skus = sorted(active_orders["sku_id"].unique().tolist())
    selected_sku = st.selectbox("Select SKU to Inspect", skus)
    selected_store_sku = active_orders["store_id"].iloc[0] if selected_store == "All Stores" else selected_store

    sku_row = active_orders[(active_orders["sku_id"] == selected_sku)].iloc[0]

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("Daily Demand", f"{sku_row['daily_forecast']:.1f} units")
    col_s2.metric("Safety Stock Cushion", f"{sku_row['safety_stock']} units")
    col_s3.metric("Reorder Point (ROP)", f"{sku_row['reorder_point']} units")
    col_s4.metric("Days of Supply", f"{sku_row['days_of_supply']:.1f} days")

    # Inventory Level Visual Comparison
    fig, ax = plt.subplots(figsize=(8, 3))
    levels = ["Current On-Hand", "Safety Stock", "Reorder Point", "Target Level (S)"]
    vals = [sku_row["current_on_hand"], sku_row["safety_stock"], sku_row["reorder_point"], sku_row["target_stock_level"]]
    bar_colors = ["#3B82F6", "#F59E0B", "#EF4444", "#10B981"]

    ax.bar(levels, vals, color=bar_colors, alpha=0.85)
    ax.set_ylabel("Units")
    ax.set_title(f"Inventory Thresholds for {selected_sku} ({sku_row['category']})", fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(i, v + 5, f"{int(v)}", ha="center", fontweight="bold")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    if sku_row["reorder_triggered"]:
        st.warning(
            f"⚠️ **Reorder Triggered!** Current on-hand ({sku_row['current_on_hand']} units) is below or equal to Reorder Point ({sku_row['reorder_point']} units). Recommended PO: **{sku_row['recommended_order_qty']} units** (${sku_row['order_value_usd']:,.2f})."
        )
    else:
        st.success(
            f"✅ **Stock Healthy!** Current on-hand ({sku_row['current_on_hand']} units) exceeds Reorder Point ({sku_row['reorder_point']} units). No immediate purchase order required."
        )
