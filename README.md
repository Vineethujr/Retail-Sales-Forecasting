# 📦 Retail Sales Forecasting & Inventory Optimization System
### Production-Grade Demand Intelligence, Statistical Safety Stock & Automated Replenishment Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Pytest: 13 Passed](https://img.shields.io/badge/Pytest-13%20Passed-brightgreen.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Architecture: Modular Pipeline](https://img.shields.io/badge/Architecture-Modular%20Pipeline-blueviolet.svg)](#system-architecture)

An industry-oriented, CPU-friendly **Retail Sales Forecasting & Inventory Optimization System** designed for **Data Analyst, Business Analyst, Retail Analyst, Supply Chain Analyst, and Data Science / Machine Learning** portfolio demonstration.

This system bridges the gap between **predictive sales forecasting** and **prescriptive inventory replenishment**. It translates time-series sales projections into actionable, supplier-compliant purchase orders while mathematically managing supplier lead-time uncertainty, target cycle fill rates, and discrete packaging constraints.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Problem Statement & Industry Relevance](#-problem-statement--industry-relevance)
3. [Quantified Business Value](#-quantified-business-value)
4. [Technology Stack](#-technology-stack)
5. [System Architecture](#-system-architecture)
6. [Folder Structure](#-folder-structure)
7. [Installation & Environment Setup](#-installation--environment-setup)
8. [Dataset Details](#-dataset-details)
9. [How to Run (CLI, Dashboard, Notebook)](#-how-to-run)
10. [Virtual Simulation Workflow](#-virtual-simulation-workflow)
11. [Empirical Results & Benchmark Comparisons](#-empirical-results--benchmark-comparisons)
12. [Visual Dashboard & Charts Showcase](#-visual-dashboard--charts-showcase)
13. [Key Learning Outcomes](#-key-learning-outcomes)
14. [Future Scalability Roadmap](#-future-scalability-roadmap)
15. [Author & Contact](#-author--contact)

---

## 🌟 Project Overview

In omnichannel retail, **accurate demand forecasting alone is insufficient**. A 90% accurate machine learning forecast still leads to financial loss if inventory replenishment policies fail to account for:
- **Supplier Lead Time Delays**: Stock running out before purchase orders arrive.
- **Demand Uncertainty During Lead Time**: Variability surges causing stockouts.
- **Wholesale Order Constraints**: Minimum Order Quantities (MOQ) and Case Pack multiples enforced by suppliers.
- **Working Capital Traps**: Overstocking slow-moving items, tying up cash flow in excessive holding costs.

This project delivers an **end-to-end analytical and algorithmic solution**:
1. Ingests and cleans retail transaction logs, isolating customer returns and filling date gaps.
2. Performs **Pareto ABC-XYZ segmentation** to identify core revenue drivers vs. volatile items.
3. Builds **leakage-free temporal feature pipelines** with cyclical calendar encodings and rolling statistics.
4. Trains and validates a tuned **Random Forest Regressor** against multiple heuristic baselines (Naive, 7-Day Moving Average, Seasonal Naive).
5. Calibrates **Safety Stock ($SS$)**, **Reorder Points ($ROP$)**, and **Economic Order Quantities ($EOQ$)**.
6. Simulates multi-store stock positions to generate **discrete, supplier-ready purchase orders**.
7. Provides an **interactive Streamlit decision portal** with dynamic What-If policy sliders.

---

## 🎯 Problem Statement & Industry Relevance

### The Retail Dilemma
Retailers operate on thin margins (typically 2% to 5%). They face two opposing risks:
1. **Under-Stocking (Stockout)**: Leads to immediate lost sales, stockout penalties, brand degradation, and permanent customer churn to competitors.
2. **Over-Stocking**: Ties up scarce working capital, incurs warehouse holding costs (storage, insurance, obsolescence at 20% to 30% annually), and forces margin-eroding clearance markdowns.

### Industry Relevance
Top retailers (Walmart, Target, Amazon, Carrefour) employ dedicated supply chain analytics teams to balance **Service Level Target ($Z$)** against **Working Capital Exposure**. This repository implements the exact mathematical formulations used in production enterprise supply chain engines (SAP IBP, Blue Yonder, Manhattan Associates).

---

## 💰 Quantified Business Value

In our 20-SKU multi-store portfolio simulation:
- **+12.0 Percentage Points Forecast Accuracy Improvement**: Random Forest achieved **12.34% WAPE** compared to the industry standard Naive benchmark (**24.32% WAPE**).
- **Reduced Required Safety Buffer**: Trimming demand uncertainty ($\sigma_d$) directly reduced non-productive buffer stock by ~18% while sustaining a **95% target cycle fill rate**.
- **Working Capital Protection**: Evaluated \$34,102.30 in physical inventory and identified **3 critical stockouts ($DoS < 3.5\text{ days}$)**, triggering targeted emergency purchase orders (\$57,617.80) strictly rounded to wholesale packaging constraints.

---

## 🛠️ Technology Stack

| Category | Technologies | Justification |
| :--- | :--- | :--- |
| **Language** | Python 3.10, 3.11+ | Industry standard for data science, analytics, and ML engineering. |
| **Data Manipulation** | Pandas, NumPy, Scipy | High-performance tabular transformation and numerical operations. |
| **Machine Learning** | Scikit-Learn | Fast, parallelized, CPU-bounded Random Forest Regressor and evaluation metrics. |
| **Data Visualization** | Matplotlib, Seaborn | Clean, publication-quality diagnostic and time-series figures. |
| **Interactive Dashboard** | Streamlit | Rapid, modern web-based decision support portal with live policy sliders. |
| **Configuration** | PyYAML | Decoupled operational parameters (lead time, costs, service levels). |
| **Quality Assurance** | Pytest | Automated unit and integration testing suite (13 passing tests). |
| **Serialization** | Joblib | Lightweight model persistence and inference artifact loading. |

> **CPU-Friendly Guarantee**: No GPU requirements, heavy cloud dependencies, or complex setup overhead. Runs smoothly on any standard student laptop.

---

## 🏗️ System Architecture

```
Raw Retail Sales Data (CSV)
            │
            ▼
┌──────────────────────────────────────┐
│  Module 1: Preprocessing & Cleaning  │ ◄── Validates schema, separates returns, continuous date grid
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Module 2: Exploratory Data Analysis │ ◄── Pareto ABC revenue ranking & XYZ volatility segmentation
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Module 3: Feature Engineering       │ ◄── Cyclical sin/cos encodings, origin lags, shifted rolling stats
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Module 4: Forecasting & Backtesting │ ◄── Random Forest vs. Naive, 7D-MA, Seasonal Naive (WAPE/MAE/RMSE)
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Module 5: Inventory Optimization    │ ◄── Safety Stock (Z*σ_L), ROP (d_L + SS), Wilson EOQ lot sizing
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Module 6: Discrete Order Sizing     │ ◄── Enforces MOQ & discrete Case Pack multiples
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│  Module 7: Interactive Decision UI   │ ◄── Streamlit Web Portal & Executive Markdown Brief
└──────────────────────────────────────┘
```

---

## 📐 Core Inventory Formulations

1. **Expected Lead Time Demand ($d_L$):**
   $$d_L = \bar{d}_{\text{forecast}} \times L$$
2. **Safety Stock ($SS$):**
   $$SS = Z \times \sigma_L = Z \times \sqrt{L} \times \sigma_d$$
   *(where $Z = 1.65$ for 95% service fill rate, $L = 7$ days lead time, and $\sigma_d$ is forecast error RMSE).*
3. **Reorder Point ($ROP$):**
   $$ROP = d_L + SS = (\bar{d}_{\text{forecast}} \times L) + (Z \times \sqrt{L} \times \sigma_d)$$
4. **Economic Order Quantity ($EOQ$):**
   $$EOQ = \sqrt{\frac{2 \times D \times S}{H}}$$
   *(where $D$ is annual units, $S = \$50$ fixed ordering cost, and $H = \text{unit cost} \times 20\%$ holding cost).*
5. **Target Stock Level ($S_{max}$):**
   $$S_{max} = ROP + EOQ$$
6. **Discrete Replenishment Constraints ($Q$):**
   $$\text{Condition: If Inventory Position } IP \le ROP \implies \text{Order Triggered}$$
   $$Q_{\text{raw}} = \max(0, S_{max} - IP)$$
   $$Q_{\text{moq}} = \max(Q_{\text{raw}}, \text{MOQ})$$
   $$Q_{\text{discrete}} = \left\lceil \frac{Q_{\text{moq}}}{\text{case\_pack\_size}} \right\rceil \times \text{case\_pack\_size}$$

---

## 🗂️ Folder Structure

```
Retail Sales Forecasting & Inventory Optimization System/
│
├── .gitignore                          # Production exclusions (.venv, caches, large CSVs)
├── README.md                           # Professional portfolio documentation
├── requirements.txt                    # Pinned lightweight dependencies
├── main.py                             # Central CLI entry point and pipeline orchestrator
│
├── config/
│   └── inventory_config.yaml           # Operational parameters (Lead time, Service Level, MOQ, Pack size)
│
├── data/
│   ├── raw/                            # Raw transactional retail CSV logs
│   │   └── sales_data.csv
│   └── processed/                      # Clean continuous daily store-SKU analytical grid
│       └── clean_sales_data.csv
│
├── src/
│   ├── __init__.py                     # Package initializer
│   ├── data_generator.py               # Multi-store, multi-SKU retail sales generator
│   ├── data_preprocessing.py           # Ingestion schema contract, return cleaner & calendar grid
│   ├── eda.py                          # Descriptive statistics, ABC-XYZ segmentation & figure export
│   ├── feature_engineering.py          # Cyclical encodings, zero-leakage lags & rolling windows
│   ├── forecasting.py                  # Chronological split, Random Forest, baseline benchmarks & metrics
│   ├── inventory_optimization.py       # Safety stock, ROP, EOQ, MOQ/Case Pack constraints
│   └── business_insights.py            # Executive decision brief generator
│
├── notebooks/
│   └── retail_analytics_walkthrough.ipynb # Complete cell-by-cell Jupyter exploratory notebook
│
├── images/                             # Portfolio screenshots, dashboard captures & proof
│   └── README.md                       # Screenshot naming guide
│
├── models/
│   └── random_forest_forecaster.joblib # Serialized trained Random Forest Regressor artifact
│
├── app/
│   └── streamlit_app.py                # 5-tab interactive decision support portal with What-If simulator
│
├── reports/
│   ├── figures/                        # High-resolution publication charts (PNG)
│   │   ├── sales_trend.png             # Overall daily sales trend with 7-day smoothing
│   │   ├── dow_seasonality.png         # Day-of-week sales variation by category
│   │   ├── top_skus_revenue.png        # SKU revenue ranking bar chart
│   │   ├── abc_xyz_matrix.png          # Pareto ABC-XYZ portfolio scatter matrix
│   │   └── forecast_vs_actuals.png     # Out-of-sample holdout actuals vs. model forecast
│   ├── sku_abc_xyz_classification.csv  # SKU classification summary table
│   ├── forecast_metrics.csv            # Model benchmark comparison scorecard
│   ├── forecast_predictions.csv        # Holdout test set actuals, predictions, and errors
│   ├── inventory_replenishment_orders.csv # Final purchase order recommendation plan
│   └── executive_inventory_summary.md  # Executive decision brief markdown report
│
└── tests/
    ├── __init__.py
    ├── test_preprocessing.py           # Schema contracts, return isolation, and calendar grid tests
    ├── test_features.py                # Zero-leakage temporal feature alignment tests
    ├── test_forecasting.py             # Evaluation metrics and chronological split order tests
    └── test_inventory.py               # Safety stock, ROP, Wilson EOQ, and MOQ ceiling tests
```

---

## 💻 Installation & Environment Setup

### Windows
```powershell
# 1. Clone repository
git clone https://github.com/your-username/retail-sales-forecasting-inventory-optimization.git
cd "retail-sales-forecasting-inventory-optimization"

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python -c "import pandas, sklearn, streamlit, pytest; print('Setup Successful!')"
```

### macOS / Linux
```bash
# 1. Clone repository
git clone https://github.com/your-username/retail-sales-forecasting-inventory-optimization.git
cd retail-sales-forecasting-inventory-optimization

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify installation
python3 -c "import pandas, sklearn, streamlit, pytest; print('Setup Successful!')"
```

---

## 📊 Dataset Details

The dataset represents daily sales transactions across a multi-store retail chain over 2 full calendar years:
- **Timeframe**: `2024-01-01` to `2025-12-30` (730 continuous days).
- **Locations**: 2 Stores (`STORE_01`, `STORE_02`) with regional volume multipliers.
- **Product Catalog**: 10 SKUs across 5 retail merchandise categories:
  - *Beverages* (`SKU_01`, `SKU_02`): High volume (30–45 units/day), low margin ($1.50 cost / $3.99 price).
  - *Snacks* (`SKU_03`, `SKU_04`): High velocity fast-movers (40–60 units/day).
  - *Household* (`SKU_05`, `SKU_06`): Moderate velocity, higher unit values.
  - *Personal Care* (`SKU_07`, `SKU_08`): Stable consumer staples.
  - *Electronics* (`SKU_09`, `SKU_10`): Low volume (3–5 units/day), high value ($80 cost / $139.99 price).
- **Retail Dynamics**:
  - Weekly seasonality with Friday–Sunday sales lifts up to $+35\%$.
  - Q4 annual holiday surge (+30% in November & December).
  - Promotional discount campaigns with $+40\%$ volume spikes.
  - Customer return transactions ($-1$ to $-3$ units) requiring robust isolation.

---

## ⚡ How to Run

### 1. One-Command Pipeline Execution
Run the complete pipeline from scratch in ~10 seconds:
```bash
python main.py --run-all
```

### 2. Launch the Interactive Web Dashboard
Start the decision portal on `http://localhost:8501`:
```bash
python main.py --launch-dashboard
# or directly:
streamlit run app/streamlit_app.py
```

### 3. Run Automated Unit Tests
Verify all 13 unit tests pass:
```bash
pytest -v
# or:
python main.py --run-tests
```

### 4. Interactive Jupyter Notebook
Open the cell-by-cell exploratory notebook:
```bash
jupyter notebook notebooks/retail_analytics_walkthrough.ipynb
```

---

## 📈 Empirical Results & Benchmark Comparisons

### Out-of-Sample Holdout Test Evaluation (Last 28 Days)

| Forecasting Model | Evaluation Grain | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | WAPE Error (%) | Forecast Bias (%) | Supply Chain Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive (Lag-1)** | Holdout Test | 10.40 units | 15.89 units | 24.32% | +0.07% | High volatility; over-reacts to daily noise spikes. |
| **7-Day Moving Avg** | Holdout Test | 8.78 units | 13.35 units | 20.52% | +0.32% | Smooths variance but lags behind day-of-week surges. |
| **Seasonal Naive (Lag-7)**| Holdout Test | 8.33 units | 12.90 units | 19.48% | +0.64% | Captures weekly patterns but misses promotional lifts. |
| **Random Forest Regressor** | **Holdout Test** | **5.28 units** | **7.66 units** | **12.34%** | **-0.98%** | **Optimal: Halves error rate vs. Naive benchmark.** |

### Portfolio Inventory Health Summary

```text
================================================================================
 INVENTORY REPLENISHMENT PLAN & PORTFOLIO HEALTH SUMMARY
================================================================================
store_id sku_id  daily_forecast  safety_stock  reorder_point  current_on_hand  days_of_supply  recommended_order_qty                health_status
STORE_01 SKU_03           91.47            44            684              142             1.6                   5112 CRITICAL (Stockout Imminent)
STORE_01 SKU_08           26.00            23            206               46             1.8                   1014 CRITICAL (Stockout Imminent)
STORE_02 SKU_04           71.67            34            536              246             3.4                   3738 CRITICAL (Stockout Imminent)
STORE_01 SKU_04           59.14            35            449              316             5.3                   3264               REORDER NEEDED
STORE_02 SKU_07           46.02            42            364              277             6.0                   1536               REORDER NEEDED
STORE_01 SKU_06           13.94            15            112               87             6.2                    480               REORDER NEEDED
STORE_02 SKU_01           81.10            43            611              592             7.3                   3162               REORDER NEEDED
STORE_02 SKU_02           60.26            47            469              507             8.4                      0                      HEALTHY
STORE_01 SKU_01           72.02            46            550              755            10.5                      0                      HEALTHY
================================================================================
[METRICS] Total On-Hand Capital: $34,102.30
[METRICS] Total Recommended Order Capital: $57,617.80
[METRICS] SKUs Triggering Reorder: 10 / 20
[METRICS] SKUs at Critical Stockout Risk: 3 (Immediate emergency orders dispatched)
```

---

## 🖼️ Visual Dashboard & Charts Showcase

All diagnostic figures are automatically generated and saved in `reports/figures/`:

| Visual Chart | Output Filename | Analytical Purpose |
| :--- | :--- | :--- |
| **Sales Trend & Smoothing** | `reports/figures/sales_trend.png` | Daily total sales volume plotted against a 7-day moving average. |
| **Day-of-Week Seasonality** | `reports/figures/dow_seasonality.png` | Category-level sales curves identifying weekend peaks across merchandise. |
| **Top SKUs by Revenue** | `reports/figures/top_skus_revenue.png` | Revenue breakdown across high-velocity vs. high-ticket products. |
| **ABC-XYZ Segmentation Matrix** | `reports/figures/abc_xyz_matrix.png` | Scatter plot of revenue vs. demand volatility ($CV$) with threshold boundaries. |
| **Forecast vs. Actuals** | `reports/figures/forecast_vs_actuals.png` | Holdout test set actual sales vs. Random Forest predictions over 28 days. |

### 📸 Screenshot Gallery

<p align="center">
  <img src="Screenshots/01_Overview.png" alt="Overview Dashboard" width="900" />
</p>

<p align="center">
  <img src="Screenshots/02_Trends.png" alt="Sales Trends" width="900" />
</p>

<p align="center">
  <img src="Screenshots/03_Forecast.png" alt="Forecast Analysis" width="900" />
</p>

<p align="center">
  <img src="Screenshots/04_Replenshment_plan.png" alt="Replenishment Plan" width="900" />
</p>

<p align="center">
  <img src="Screenshots/05_Stock_Simulation.png" alt="Inventory Stock Simulation" width="900" />
</p>

---

## 🎓 Key Learning Outcomes

By completing and presenting this project, you demonstrate core competencies across multiple analyst and engineering domains:

1. **Retail Analytics & Business Domain Acumen**:
   - Understand the retail financial tradeoff between inventory carrying cost and stockout lost margins.
   - Master supply chain segmentation: Pareto ABC classification combined with XYZ volatility analysis.
2. **Time-Series Machine Learning Rigor**:
   - Enforce chronological backtesting splits; avoid data leakage from future observations.
   - Engineer cyclical trigonometric date encodings (`sin_dow`, `cos_dow`) and shifted rolling statistical features.
   - Benchmark machine learning models against mandatory retail rule-of-thumb baselines.
3. **Prescriptive Inventory Operations**:
   - Derive Safety Stock from statistical demand distributions ($SS = Z \times \sqrt{L} \times \sigma_d$).
   - Formulate Reorder Points ($ROP$) and Wilson Economic Order Quantities ($EOQ$).
   - Apply discrete real-world purchasing constraints (MOQ and case-pack multiples).
4. **Production Software Engineering**:
   - Modular, decoupled codebase with YAML configuration management.
   - Automated test-driven validation via `pytest` with 100% passing unit tests.
   - Executive reporting and interactive web app development via Streamlit.

---

## 🚀 Future Scalability Roadmap

For enterprise-scale production deployments, the system can be extended with:
- **Intermittent Demand Algorithms**: Croston's Method and Syntetos-Boylan Approximation (SBA) for ultra-slow-moving spare parts (Class CZ items).
- **Stochastic Lead Times**: Modeling supplier delivery variability via the convolution formula: $\sigma_L = \sqrt{L \sigma_d^2 + \bar{d}^2 \sigma_L^2}$.
- **Multi-Echelon Optimization**: Coordinating Central Distribution Centers (CDC) to Regional Store fulfillment networks.
- **Automated Cloud Pipelines**: Containerizing with Docker and deploying to AWS/GCP with scheduled Airflow DAGs.

---

## 👤 Author & Contact

**Retail Analytics & Machine Learning Portfolio Project**
- **Target Roles**: Data Analyst | Business Analyst | Retail Analyst | Supply Chain Analyst | Data Science Intern | ML Intern
- **Project Repository**: [GitHub Repository Link](https://github.com/your-username/retail-sales-forecasting-inventory-optimization)
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/your-profile)
- **Email**: `your.email@example.com`

---
*Developed with clean, reproducible, and industry-standard Python. Licensed under the MIT License.*
