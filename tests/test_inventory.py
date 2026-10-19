"""
Unit Tests for Inventory Optimization Formulas & Constraints
(tests/test_inventory.py)
"""

import math
from src.inventory_optimization import (
    calculate_safety_stock,
    calculate_reorder_point,
    calculate_eoq,
    apply_order_constraints,
)


def test_safety_stock_formula():
    # SS = Z * sqrt(L) * sigma_d
    # Z = 1.65, L = 4 (sqrt = 2), sigma_d = 10 -> SS = 1.65 * 2 * 10 = 33.0
    ss = calculate_safety_stock(daily_demand_std=10.0, lead_time_days=4, z_score=1.65)
    assert math.isclose(ss, 33.0, rel_tol=1e-3)


def test_reorder_point_formula():
    # ROP = (d_mean * L) + SS
    # d_mean = 20, L = 7, SS = 30 -> ROP = (20 * 7) + 30 = 170.0
    rop = calculate_reorder_point(mean_daily_demand=20.0, lead_time_days=7, safety_stock=30.0)
    assert rop == 170.0


def test_eoq_formula():
    # EOQ = sqrt(2 * D * S / H)
    # D = 1000, S = 50, H = 2.5 -> EOQ = sqrt(100,000 / 2.5) = sqrt(40,000) = 200.0
    eoq = calculate_eoq(annual_demand=1000.0, order_cost=50.0, unit_cost=12.5, holding_rate=0.20)
    assert math.isclose(eoq, 200.0, rel_tol=1e-3)


def test_apply_order_constraints_moq_and_case_pack():
    # Case 1: Order is 0 -> returns 0
    assert apply_order_constraints(0.0, moq=10, case_pack=6) == 0

    # Case 2: Order is 4, MOQ is 10, case pack is 6 -> max(4, 10)=10, ceil(10/6)*6 = 12
    assert apply_order_constraints(4.0, moq=10, case_pack=6) == 12

    # Case 3: Order is 15, MOQ is 5, case pack is 6 -> max(15, 5)=15, ceil(15/6)*6 = 18
    assert apply_order_constraints(15.0, moq=5, case_pack=6) == 18
