"""
Retail Sales Forecasting & Inventory Optimization System
Main Execution Controller & CLI Orchestrator (main.py)

Usage Examples:
  python main.py --help
  python main.py --run-all
  python main.py --generate-data
  python main.py --launch-dashboard
  python main.py --run-tests
"""

import sys
import argparse
from pathlib import Path
import subprocess

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Retail Sales Forecasting & Inventory Optimization System CLI Entry Point",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Execute the complete end-to-end pipeline: generation -> clean -> EDA -> forecast -> inventory -> report.",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Generate synthetic multi-store multi-SKU retail sales dataset.",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Clean raw sales transactions, handle returns, and build continuous daily calendar grid.",
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Compute ABC-XYZ classification and generate exploratory figures.",
    )
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Train Random Forest regressor, evaluate baselines, and export accuracy metrics.",
    )
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="Calculate Safety Stock, ROP, EOQ, and generate purchase replenishment order plan.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate the executive decision brief markdown summary.",
    )
    parser.add_argument(
        "--launch-dashboard",
        action="store_true",
        help="Launch the interactive Streamlit decision support portal.",
    )
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Execute the automated pytest suite across all modules.",
    )
    return parser.parse_args()


def run_full_pipeline():
    """Runs all modular steps sequentially with timing and verification."""
    print("\n" + "=" * 80)
    print(" STARTING END-TO-END RETAIL ANALYTICS & INVENTORY PIPELINE")
    print("=" * 80)

    from src.data_generator import generate_synthetic_retail_data
    from src.data_preprocessing import run_preprocessing_pipeline
    from src.eda import run_eda_pipeline
    from src.forecasting import run_forecasting_pipeline
    from src.inventory_optimization import run_inventory_pipeline
    from src.business_insights import generate_executive_report

    print("\n[STEP 1/6] Generating synthetic retail sales data...")
    generate_synthetic_retail_data()

    print("\n[STEP 2/6] Running data cleaning and grid construction...")
    run_preprocessing_pipeline()

    print("\n[STEP 3/6] Running EDA & Pareto ABC-XYZ classification...")
    run_eda_pipeline()

    print("\n[STEP 4/6] Training forecasting models and benchmarking...")
    run_forecasting_pipeline()

    print("\n[STEP 5/6] Optimizing inventory, safety stock, and reorder points...")
    run_inventory_pipeline()

    print("\n[STEP 6/6] Generating executive decision brief...")
    generate_executive_report()

    print("\n" + "=" * 80)
    print(" [COMPLETE] End-to-end pipeline finished successfully!")
    print(" To launch the interactive web dashboard:")
    print("   python main.py --launch-dashboard")
    print("=" * 80 + "\n")


def main():
    args = parse_arguments()

    print("=" * 80)
    print(" Retail Sales Forecasting & Inventory Optimization System")
    print(f" Project Root: {PROJECT_ROOT}")
    print("=" * 80)

    if args.run_tests:
        import pytest
        print("\n[INFO] Running automated test suite...")
        exit_code = pytest.main(["-v", str(PROJECT_ROOT / "tests")])
        sys.exit(exit_code)

    if args.launch_dashboard:
        app_path = PROJECT_ROOT / "app" / "streamlit_app.py"
        print(f"\n[INFO] Launching Streamlit dashboard from {app_path}...")
        subprocess.run(["streamlit", "run", str(app_path)])
        return

    if args.run_all:
        run_full_pipeline()
        return

    # Individual modular flags
    executed_any = False
    if args.generate_data:
        from src.data_generator import generate_synthetic_retail_data
        generate_synthetic_retail_data()
        executed_any = True

    if args.preprocess:
        from src.data_preprocessing import run_preprocessing_pipeline
        run_preprocessing_pipeline()
        executed_any = True

    if args.eda:
        from src.eda import run_eda_pipeline
        run_eda_pipeline()
        executed_any = True

    if args.forecast:
        from src.forecasting import run_forecasting_pipeline
        run_forecasting_pipeline()
        executed_any = True

    if args.inventory:
        from src.inventory_optimization import run_inventory_pipeline
        run_inventory_pipeline()
        executed_any = True

    if args.report:
        from src.business_insights import generate_executive_report
        generate_executive_report()
        executed_any = True

    if not executed_any:
        print("\nNo specific action flag supplied.")
        print("To run the full end-to-end pipeline:  python main.py --run-all")
        print("To launch the interactive dashboard:    python main.py --launch-dashboard")
        print("To execute automated test suite:       python main.py --run-tests")
        print("Use --help for complete usage options.\n")


if __name__ == "__main__":
    main()
