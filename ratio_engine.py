"""
Ratio engine for EquiLens.
Computes profitability, liquidity, efficiency, solvency ratios,
DuPont decomposition, and Altman Z-Score from extracted financial metrics.
"""

import pandas as pd
import numpy as np


def _safe_div(numerator: float, denominator: float) -> float:
    """Division that returns 0 when denominator is zero or near-zero."""
    if abs(denominator) < 1e-6:
        return 0.0
    return numerator / denominator


def compute_profitability_ratios(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Gross Margin, EBITDA Margin, Net Margin, ROE, ROA, ROCE — all as percentages.
    Each column is a fiscal year; rows are ratio names.
    """
    ratios = {}
    for year in metrics.columns:
        m = metrics[year]
        revenue = m.get("Revenue", 0)
        ebit = m.get("Operating Income", 0)
        total_debt = m.get("Total Debt", 0)
        interest = m.get("Interest Expense", 0)
        tax = m.get("Tax", 0)

        # Capital employed = Total Assets - Current Liabilities
        capital_employed = m.get("Total Assets", 0) - m.get("Current Liabilities", 0)

        ratios[year] = {
            "Gross Margin %": _safe_div(m.get("Gross Profit", 0), revenue) * 100,
            "EBITDA Margin %": _safe_div(m.get("EBITDA", 0), revenue) * 100,
            "Net Margin %": _safe_div(m.get("Net Income", 0), revenue) * 100,
            "ROE %": _safe_div(m.get("Net Income", 0), m.get("Total Equity", 0)) * 100,
            "ROA %": _safe_div(m.get("Net Income", 0), m.get("Total Assets", 0)) * 100,
            "ROCE %": _safe_div(ebit, capital_employed) * 100,
        }
    return pd.DataFrame(ratios)


def compute_liquidity_ratios(metrics: pd.DataFrame) -> pd.DataFrame:
    """Current Ratio and Quick Ratio (acid test)."""
    ratios = {}
    for year in metrics.columns:
        m = metrics[year]
        ca = m.get("Current Assets", 0)
        cl = m.get("Current Liabilities", 0)
        inv = m.get("Inventory", 0)
        ratios[year] = {
            "Current Ratio": _safe_div(ca, cl),
            "Quick Ratio": _safe_div(ca - inv, cl),
        }
    return pd.DataFrame(ratios)


def compute_efficiency_ratios(metrics: pd.DataFrame) -> pd.DataFrame:
    """Asset Turnover, Debtor Days, and Cash Conversion Cycle."""
    ratios = {}
    for year in metrics.columns:
        m = metrics[year]
        revenue = m.get("Revenue", 0)
        cogs = m.get("COGS", 0)
        if cogs == 0:
            cogs = revenue * 0.6  # fallback estimate

        ratios[year] = {
            "Asset Turnover": _safe_div(revenue, m.get("Total Assets", 0)),
            "Debtor Days": _safe_div(m.get("Receivables", 0), revenue) * 365,
            "Inventory Days": _safe_div(m.get("Inventory", 0), cogs) * 365,
            "Payable Days": _safe_div(m.get("Payables", 0), cogs) * 365,
        }
        # Cash Conversion Cycle = Inventory Days + Debtor Days - Payable Days
        ratios[year]["Cash Conversion Cycle"] = (
            ratios[year]["Inventory Days"]
            + ratios[year]["Debtor Days"]
            - ratios[year]["Payable Days"]
        )
    return pd.DataFrame(ratios)


def compute_solvency_ratios(metrics: pd.DataFrame) -> pd.DataFrame:
    """Debt/Equity, Interest Coverage, and Altman Z-Score."""
    ratios = {}
    for year in metrics.columns:
        m = metrics[year]
        equity = m.get("Total Equity", 0)
        interest = m.get("Interest Expense", 0)
        ebit = m.get("Operating Income", 0)

        ratios[year] = {
            "Debt/Equity": _safe_div(m.get("Total Debt", 0), equity),
            "Interest Coverage": _safe_div(ebit, interest) if interest > 0 else 999.0,
        }
    return pd.DataFrame(ratios)


def compute_altman_z_score(metrics: pd.DataFrame, market_cap: float) -> pd.DataFrame:
    """
    Altman Z-Score for non-manufacturing / emerging-market variant:
      Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
    where:
      X1 = Working Capital / Total Assets
      X2 = Retained Earnings / Total Assets
      X3 = EBIT / Total Assets
      X4 = Market Cap / Total Liabilities
      X5 = Revenue / Total Assets
    """
    scores = {}
    for year in metrics.columns:
        m = metrics[year]
        ta = m.get("Total Assets", 0)
        if ta == 0:
            scores[year] = {"Z-Score": 0, "Zone": "N/A"}
            continue

        total_liabilities = ta - m.get("Total Equity", 0)

        x1 = _safe_div(m.get("Working Capital", 0), ta)
        x2 = _safe_div(m.get("Retained Earnings", 0), ta)
        x3 = _safe_div(m.get("Operating Income", 0), ta)
        x4 = _safe_div(market_cap, total_liabilities) if total_liabilities > 0 else 10
        x5 = _safe_div(m.get("Revenue", 0), ta)

        z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

        if z > 2.99:
            zone = "Safe"
        elif z > 1.81:
            zone = "Grey"
        else:
            zone = "Distress"

        scores[year] = {
            "X1 (WC/TA)": round(x1, 4),
            "X2 (RE/TA)": round(x2, 4),
            "X3 (EBIT/TA)": round(x3, 4),
            "X4 (MCap/TL)": round(x4, 4),
            "X5 (Rev/TA)": round(x5, 4),
            "Z-Score": round(z, 2),
            "Zone": zone,
        }
    return pd.DataFrame(scores)


def compute_dupont(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    DuPont 3-factor decomposition:
      ROE = Net Margin × Asset Turnover × Equity Multiplier
    Shows each driver's value so the user can see what's moving ROE.
    """
    dupont = {}
    for year in metrics.columns:
        m = metrics[year]
        revenue = m.get("Revenue", 0)
        net_income = m.get("Net Income", 0)
        total_assets = m.get("Total Assets", 0)
        equity = m.get("Total Equity", 0)

        net_margin = _safe_div(net_income, revenue)
        asset_turnover = _safe_div(revenue, total_assets)
        equity_multiplier = _safe_div(total_assets, equity)
        roe = net_margin * asset_turnover * equity_multiplier * 100

        dupont[year] = {
            "Net Margin": round(net_margin, 4),
            "Asset Turnover": round(asset_turnover, 4),
            "Equity Multiplier": round(equity_multiplier, 4),
            "DuPont ROE %": round(roe, 2),
        }
    return pd.DataFrame(dupont)


def compute_all_ratios(metrics: pd.DataFrame, market_cap: float = 0) -> dict:
    """Compute all ratio categories and return as a dict of DataFrames."""
    return {
        "profitability": compute_profitability_ratios(metrics),
        "liquidity": compute_liquidity_ratios(metrics),
        "efficiency": compute_efficiency_ratios(metrics),
        "solvency": compute_solvency_ratios(metrics),
        "dupont": compute_dupont(metrics),
        "altman": compute_altman_z_score(metrics, market_cap),
    }


# --- Traffic Light Scoring ---

# Thresholds: (green_min, amber_min) — below amber_min is red
RATIO_THRESHOLDS = {
    "Gross Margin %": (40, 20),
    "EBITDA Margin %": (20, 10),
    "Net Margin %": (15, 5),
    "ROE %": (15, 8),
    "ROA %": (8, 3),
    "ROCE %": (15, 8),
    "Current Ratio": (1.5, 1.0),
    "Quick Ratio": (1.0, 0.5),
    "Asset Turnover": (0.8, 0.4),
    "Debt/Equity": None,  # inverse — lower is better
    "Interest Coverage": (3, 1.5),
}

# For inverse ratios (lower = better): (green_max, amber_max)
INVERSE_THRESHOLDS = {
    "Debt/Equity": (0.5, 1.5),
    "Debtor Days": (60, 120),
    "Cash Conversion Cycle": (60, 120),
}


def get_traffic_light(ratio_name: str, value: float) -> str:
    """Return 'green', 'amber', or 'red' for a given ratio value."""
    if ratio_name in INVERSE_THRESHOLDS:
        green_max, amber_max = INVERSE_THRESHOLDS[ratio_name]
        if value <= green_max:
            return "green"
        elif value <= amber_max:
            return "amber"
        else:
            return "red"

    if ratio_name in RATIO_THRESHOLDS and RATIO_THRESHOLDS[ratio_name] is not None:
        green_min, amber_min = RATIO_THRESHOLDS[ratio_name]
        if value >= green_min:
            return "green"
        elif value >= amber_min:
            return "amber"
        else:
            return "red"

    return "amber"  # default for unknown ratios


def compute_health_score(all_ratios: dict) -> int:
    """
    Overall Financial Health Score out of 100.
    Green = 10 pts, Amber = 5 pts, Red = 0 pts per evaluated ratio.
    Normalized to 100.
    """
    scored_ratios = list(RATIO_THRESHOLDS.keys()) + list(INVERSE_THRESHOLDS.keys())
    # remove duplicates
    scored_ratios = list(dict.fromkeys(scored_ratios))

    total_points = 0
    max_points = len(scored_ratios) * 10

    for ratio_name in scored_ratios:
        # Find the latest year's value across ratio categories
        value = None
        for category_df in all_ratios.values():
            if isinstance(category_df, pd.DataFrame) and ratio_name in category_df.index:
                latest_year = category_df.columns[-1]
                val = category_df.loc[ratio_name, latest_year]
                if isinstance(val, (int, float)) and not np.isnan(val):
                    value = val
                    break

        if value is None:
            total_points += 5  # neutral score for missing data
            continue

        light = get_traffic_light(ratio_name, value)
        if light == "green":
            total_points += 10
        elif light == "amber":
            total_points += 5

    return round(_safe_div(total_points, max_points) * 100)
