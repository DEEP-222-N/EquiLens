"""
Automatic DCF assumption engine for EquiLens.
Derives stock-specific assumptions from historical financials instead of manual sliders.
"""

import numpy as np
import pandas as pd


def compute_revenue_cagr(metrics: pd.DataFrame) -> float:
    """Compute revenue CAGR from the earliest to latest available year."""
    if metrics.empty or "Revenue" not in metrics.index:
        return 0.10

    revenues = metrics.loc["Revenue"]
    revenues = revenues[revenues > 0]
    if len(revenues) < 2:
        return 0.10

    first = revenues.iloc[0]
    last = revenues.iloc[-1]
    n_years = len(revenues) - 1

    if first <= 0 or last <= 0 or n_years == 0:
        return 0.10

    cagr = (last / first) ** (1 / n_years) - 1
    return np.clip(cagr, 0.0, 0.30)


def compute_avg_ebitda_margin(metrics: pd.DataFrame) -> float:
    """Average EBITDA margin over available years."""
    if metrics.empty:
        return 0.20

    margins = []
    for year in metrics.columns:
        rev = metrics.loc["Revenue", year] if "Revenue" in metrics.index else 0
        ebitda = metrics.loc["EBITDA", year] if "EBITDA" in metrics.index else 0
        if rev > 0:
            margins.append(ebitda / rev)

    if not margins:
        return 0.20

    avg = np.mean(margins)
    return np.clip(avg, 0.05, 0.50)


def compute_effective_tax_rate(metrics: pd.DataFrame) -> float:
    """Average effective tax rate from historicals."""
    if metrics.empty:
        return 0.25

    rates = []
    for year in metrics.columns:
        pbt = metrics.loc["Net Income", year] + metrics.loc["Tax", year] if "Net Income" in metrics.index and "Tax" in metrics.index else 0
        tax = metrics.loc["Tax", year] if "Tax" in metrics.index else 0
        if pbt > 0 and tax > 0:
            rates.append(tax / pbt)

    if not rates:
        return 0.25

    avg = np.mean(rates)
    return np.clip(avg, 0.10, 0.40)


def compute_wacc(info: dict, metrics: pd.DataFrame) -> float:
    """
    Estimate WACC from:
      - Cost of equity via CAPM: Rf + Beta * (Rm - Rf)
      - Cost of debt from interest expense / total debt
      - Weights from market cap and debt
    Uses Indian market defaults: Rf=7.1% (10Y govt bond), ERP=6%.
    """
    risk_free = 0.071
    equity_risk_premium = 0.06
    beta = info.get("beta", 1.0) or 1.0

    cost_of_equity = risk_free + beta * equity_risk_premium

    cost_of_debt = 0.09
    tax_rate = 0.25
    if not metrics.empty and "Interest Expense" in metrics.index and "Total Debt" in metrics.index:
        latest = metrics.columns[-1]
        interest = metrics.loc["Interest Expense", latest]
        debt = metrics.loc["Total Debt", latest]
        if debt > 0 and interest > 0:
            cost_of_debt = interest / debt
            cost_of_debt = np.clip(cost_of_debt, 0.04, 0.18)
        tax_rate = compute_effective_tax_rate(metrics)

    market_cap = info.get("market_cap", 0) or 0
    total_debt = 0
    if not metrics.empty and "Total Debt" in metrics.index:
        total_debt = metrics.loc["Total Debt", metrics.columns[-1]]

    total_capital = market_cap + total_debt
    if total_capital <= 0:
        return np.clip(cost_of_equity, 0.08, 0.18)

    weight_equity = market_cap / total_capital
    weight_debt = total_debt / total_capital

    wacc = weight_equity * cost_of_equity + weight_debt * cost_of_debt * (1 - tax_rate)
    return np.clip(wacc, 0.05, 0.20)


def compute_terminal_growth(info: dict) -> float:
    """
    Terminal growth rate based on sector.
    High-growth sectors get higher TGR, mature sectors get lower.
    Capped at a reasonable range for Indian equities.
    """
    sector = (info.get("sector", "") or "").lower()

    high_growth = ["technology", "healthcare", "consumer cyclical", "communication services"]
    moderate_growth = ["industrials", "consumer defensive", "financial services", "basic materials"]
    low_growth = ["energy", "utilities", "real estate"]

    if any(s in sector for s in high_growth):
        return 0.045
    elif any(s in sector for s in moderate_growth):
        return 0.035
    elif any(s in sector for s in low_growth):
        return 0.025
    return 0.035


def compute_capex_pct(metrics: pd.DataFrame) -> float:
    """Average Capex as % of Revenue from historicals."""
    if metrics.empty:
        return 0.05

    pcts = []
    for year in metrics.columns:
        rev = metrics.loc["Revenue", year] if "Revenue" in metrics.index else 0
        capex = metrics.loc["Capex", year] if "Capex" in metrics.index else 0
        if rev > 0 and capex != 0:
            pcts.append(abs(capex) / rev)

    if not pcts:
        return 0.05

    avg = np.mean(pcts)
    return np.clip(avg, 0.01, 0.25)


def compute_nwc_pct(metrics: pd.DataFrame) -> float:
    """Average Net Working Capital as % of Revenue from historicals."""
    if metrics.empty:
        return 0.10

    pcts = []
    for year in metrics.columns:
        rev = metrics.loc["Revenue", year] if "Revenue" in metrics.index else 0
        wc = metrics.loc["Working Capital", year] if "Working Capital" in metrics.index else 0
        if rev > 0:
            pcts.append(abs(wc) / rev)

    if not pcts:
        return 0.10

    avg = np.mean(pcts)
    return np.clip(avg, 0.01, 0.30)


def compute_da_pct(metrics: pd.DataFrame) -> float:
    """Average D&A as % of Revenue from historicals."""
    if metrics.empty or "D&A" not in metrics.index:
        return 0.03

    pcts = []
    for year in metrics.columns:
        rev = metrics.loc["Revenue", year] if "Revenue" in metrics.index else 0
        da = metrics.loc["D&A", year] if "D&A" in metrics.index else 0
        if rev > 0 and da > 0:
            pcts.append(da / rev)

    if not pcts:
        return 0.03

    avg = np.mean(pcts)
    return np.clip(avg, 0.005, 0.15)


def compute_all_assumptions(info: dict, metrics: pd.DataFrame) -> dict:
    """
    Derive all DCF assumptions from a stock's own financials.
    Returns a dict with each assumption and a human-readable rationale.
    """
    revenue_growth = compute_revenue_cagr(metrics)
    ebitda_margin = compute_avg_ebitda_margin(metrics)
    wacc = compute_wacc(info, metrics)
    terminal_growth = compute_terminal_growth(info)
    tax_rate = compute_effective_tax_rate(metrics)
    capex_pct = compute_capex_pct(metrics)
    nwc_pct = compute_nwc_pct(metrics)
    da_pct = compute_da_pct(metrics)

    if wacc <= terminal_growth:
        terminal_growth = wacc - 0.02

    return {
        "revenue_growth": round(revenue_growth, 4),
        "ebitda_margin": round(ebitda_margin, 4),
        "wacc": round(wacc, 4),
        "terminal_growth": round(terminal_growth, 4),
        "tax_rate": round(tax_rate, 4),
        "capex_pct": round(capex_pct, 4),
        "nwc_pct": round(nwc_pct, 4),
        "da_pct": round(da_pct, 4),
        "rationale": {
            "revenue_growth": f"{revenue_growth*100:.1f}% — based on {len(metrics.columns)}Y revenue CAGR",
            "ebitda_margin": f"{ebitda_margin*100:.1f}% — average of historical EBITDA margins",
            "wacc": f"{wacc*100:.1f}% — CAPM-derived (Beta={info.get('beta', 1.0):.2f}, Rf=7.1%, ERP=6%)",
            "terminal_growth": f"{terminal_growth*100:.1f}% — sector-adjusted ({info.get('sector', 'N/A')})",
            "tax_rate": f"{tax_rate*100:.1f}% — effective tax rate from financials",
            "capex_pct": f"{capex_pct*100:.1f}% — average historical Capex/Revenue",
            "nwc_pct": f"{nwc_pct*100:.1f}% — average historical NWC/Revenue",
            "da_pct": f"{da_pct*100:.1f}% — average historical D&A/Revenue",
        },
    }
