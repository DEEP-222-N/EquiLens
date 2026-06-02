"""
DCF valuation model for EquiLens.
Projects free cash flow, discounts to present value, and computes intrinsic value per share.
Supports bull/bear/base scenarios and sensitivity analysis.
"""

import numpy as np
import pandas as pd


def project_fcf(
    last_revenue: float,
    revenue_growth: float,
    ebitda_margin: float,
    tax_rate: float,
    capex_pct: float,
    nwc_pct: float,
    projection_years: int = 5,
) -> pd.DataFrame:
    """
    Project Free Cash Flow for n years.
    FCF = EBITDA × (1 - tax_rate) - Capex - Change in NWC
    """
    projections = []
    prev_nwc = last_revenue * nwc_pct

    for i in range(1, projection_years + 1):
        revenue = last_revenue * (1 + revenue_growth) ** i
        ebitda = revenue * ebitda_margin
        nopat = ebitda * (1 - tax_rate)
        capex = revenue * capex_pct
        current_nwc = revenue * nwc_pct
        delta_nwc = current_nwc - prev_nwc
        fcf = nopat - capex - delta_nwc
        prev_nwc = current_nwc

        projections.append({
            "Year": i,
            "Revenue": revenue,
            "EBITDA": ebitda,
            "NOPAT": nopat,
            "Capex": capex,
            "ΔNWC": delta_nwc,
            "FCF": fcf,
        })

    return pd.DataFrame(projections)


def compute_dcf(
    last_revenue: float,
    revenue_growth: float,
    ebitda_margin: float,
    wacc: float,
    terminal_growth: float,
    tax_rate: float = 0.25,
    capex_pct: float = 0.05,
    nwc_pct: float = 0.10,
    shares_outstanding: float = 1,
    net_debt: float = 0,
    projection_years: int = 5,
) -> dict:
    """
    Full DCF valuation.
    Returns intrinsic value per share along with intermediate calculations.
    """
    projections = project_fcf(
        last_revenue, revenue_growth, ebitda_margin,
        tax_rate, capex_pct, nwc_pct, projection_years
    )

    # Discount projected FCFs to present value
    pv_fcfs = []
    for _, row in projections.iterrows():
        pv = row["FCF"] / (1 + wacc) ** row["Year"]
        pv_fcfs.append(pv)
    projections["PV of FCF"] = pv_fcfs

    sum_pv_fcf = sum(pv_fcfs)

    # Terminal value using Gordon Growth Model
    terminal_fcf = projections.iloc[-1]["FCF"] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / (1 + wacc) ** projection_years

    enterprise_value = sum_pv_fcf + pv_terminal
    equity_value = enterprise_value - net_debt

    intrinsic_per_share = equity_value / shares_outstanding if shares_outstanding > 0 else 0

    return {
        "projections": projections,
        "sum_pv_fcf": sum_pv_fcf,
        "terminal_value": terminal_value,
        "pv_terminal": pv_terminal,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "intrinsic_per_share": intrinsic_per_share,
    }


def run_scenarios(
    last_revenue: float,
    base_growth: float,
    base_margin: float,
    wacc: float,
    terminal_growth: float,
    shares_outstanding: float,
    net_debt: float,
    tax_rate: float = 0.25,
) -> dict:
    """
    Run bull / base / bear DCF scenarios.
    Bull: +3% growth, +2% margin.  Bear: -3% growth, -2% margin.
    """
    scenarios = {}
    adjustments = {
        "Bear": {"growth_adj": -0.03, "margin_adj": -0.02},
        "Base": {"growth_adj": 0, "margin_adj": 0},
        "Bull": {"growth_adj": 0.03, "margin_adj": 0.02},
    }

    for name, adj in adjustments.items():
        result = compute_dcf(
            last_revenue=last_revenue,
            revenue_growth=base_growth + adj["growth_adj"],
            ebitda_margin=base_margin + adj["margin_adj"],
            wacc=wacc,
            terminal_growth=terminal_growth,
            tax_rate=tax_rate,
            shares_outstanding=shares_outstanding,
            net_debt=net_debt,
        )
        scenarios[name] = result["intrinsic_per_share"]

    return scenarios


def sensitivity_matrix(
    last_revenue: float,
    ebitda_margin: float,
    revenue_growth: float,
    shares_outstanding: float,
    net_debt: float,
    wacc_range: list = None,
    tgr_range: list = None,
    tax_rate: float = 0.25,
) -> pd.DataFrame:
    """
    Build a WACC × Terminal Growth Rate sensitivity matrix of intrinsic values.
    Returns a DataFrame where rows = WACC values, columns = terminal growth rates.
    """
    if wacc_range is None:
        wacc_range = [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14]
    if tgr_range is None:
        tgr_range = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

    matrix = {}
    for tgr in tgr_range:
        col = {}
        for wacc in wacc_range:
            if wacc <= tgr:
                col[f"{wacc*100:.1f}%"] = float("inf")
                continue
            result = compute_dcf(
                last_revenue=last_revenue,
                revenue_growth=revenue_growth,
                ebitda_margin=ebitda_margin,
                wacc=wacc,
                terminal_growth=tgr,
                tax_rate=tax_rate,
                shares_outstanding=shares_outstanding,
                net_debt=net_debt,
            )
            col[f"{wacc*100:.1f}%"] = round(result["intrinsic_per_share"], 2)
        matrix[f"{tgr*100:.1f}%"] = col

    return pd.DataFrame(matrix)
