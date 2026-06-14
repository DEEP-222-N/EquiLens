"""
Valuation models for EquiLens.
DCF, Dividend Discount Model (DDM), and Comparable Valuation (PE & EV/EBITDA comps).
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
    da_pct: float = 0.03,
    projection_years: int = 5,
) -> pd.DataFrame:
    """
    Project Unlevered Free Cash Flow for n years.
    UFCF = EBIT × (1 - tax) + D&A - Capex - ΔNWC
    where EBIT = EBITDA - D&A
    """
    projections = []
    prev_nwc = last_revenue * nwc_pct

    for i in range(1, projection_years + 1):
        revenue = last_revenue * (1 + revenue_growth) ** i
        ebitda = revenue * ebitda_margin
        da = revenue * da_pct
        ebit = ebitda - da
        nopat = ebit * (1 - tax_rate)
        capex = revenue * capex_pct
        current_nwc = revenue * nwc_pct
        delta_nwc = current_nwc - prev_nwc
        fcf = nopat + da - capex - delta_nwc
        prev_nwc = current_nwc

        projections.append({
            "Year": i,
            "Revenue": revenue,
            "EBITDA": ebitda,
            "D&A": da,
            "EBIT": ebit,
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
    da_pct: float = 0.03,
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
        tax_rate, capex_pct, nwc_pct, da_pct, projection_years
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


# ── Dividend Discount Model (DDM) ──────────────────────────────

def compute_ddm(
    last_dividend_per_share: float,
    dividend_growth_rate: float,
    cost_of_equity: float,
    shares_outstanding: float = 1,
) -> dict:
    """
    Gordon Growth DDM: P = D1 / (Ke - g)
    Two-stage variant: 5 years of high growth, then perpetuity at terminal growth.
    Returns intrinsic value per share and breakdown.
    """
    if last_dividend_per_share <= 0:
        return {"intrinsic_per_share": 0, "applicable": False, "reason": "No dividends paid"}

    if cost_of_equity <= dividend_growth_rate:
        return {"intrinsic_per_share": 0, "applicable": False, "reason": "Ke must exceed growth rate"}

    terminal_growth = min(dividend_growth_rate, 0.05)
    high_growth = min(dividend_growth_rate * 1.2, 0.20)
    high_growth_years = 5

    pv_dividends = 0
    projected_divs = []
    current_div = last_dividend_per_share

    for year in range(1, high_growth_years + 1):
        current_div = current_div * (1 + high_growth)
        pv = current_div / (1 + cost_of_equity) ** year
        pv_dividends += pv
        projected_divs.append({"Year": year, "Dividend": round(current_div, 2), "PV": round(pv, 2)})

    terminal_div = current_div * (1 + terminal_growth)
    terminal_value = terminal_div / (cost_of_equity - terminal_growth)
    pv_terminal = terminal_value / (1 + cost_of_equity) ** high_growth_years

    intrinsic = pv_dividends + pv_terminal

    return {
        "intrinsic_per_share": round(intrinsic, 2),
        "applicable": True,
        "pv_high_growth_divs": round(pv_dividends, 2),
        "terminal_value": round(terminal_value, 2),
        "pv_terminal": round(pv_terminal, 2),
        "high_growth_rate": round(high_growth, 4),
        "terminal_growth": round(terminal_growth, 4),
        "projected_dividends": pd.DataFrame(projected_divs),
    }


# ── Comparable Valuation (PE & EV/EBITDA) ──────────────────────

def compute_comparable_valuation(
    target_info: dict,
    target_metrics: pd.DataFrame,
    peer_data: dict,
) -> dict:
    """
    Relative valuation using peer multiples.
    Computes PE-implied and EV/EBITDA-implied fair values from peer medians.
    """
    result = {
        "pe_comps": {"applicable": False},
        "ev_ebitda_comps": {"applicable": False},
    }

    if target_metrics.empty:
        return result

    latest_year = target_metrics.columns[-1]

    # ── Gather peer multiples ──
    peer_pes = []
    peer_ev_ebitdas = []

    for ticker, data in peer_data.items():
        p_info = data.get("info", {})
        p_metrics = data.get("metrics", pd.DataFrame())

        if p_info.get("error"):
            continue

        pe = p_info.get("pe_ratio", 0) or 0
        if 0 < pe < 200:
            peer_pes.append({"ticker": ticker, "pe": pe})

        if not p_metrics.empty and "EBITDA" in p_metrics.index:
            p_latest = p_metrics.columns[-1]
            p_ebitda = p_metrics.loc["EBITDA", p_latest]
            p_mcap = p_info.get("market_cap", 0) or 0
            p_debt = p_metrics.loc["Total Debt", p_latest] if "Total Debt" in p_metrics.index else 0
            p_cash = p_metrics.loc["Cash", p_latest] if "Cash" in p_metrics.index else 0
            p_ev = p_mcap + p_debt - p_cash
            if p_ebitda > 0 and p_ev > 0:
                ev_ebitda = p_ev / p_ebitda
                if 0 < ev_ebitda < 100:
                    peer_ev_ebitdas.append({"ticker": ticker, "ev_ebitda": round(ev_ebitda, 2)})

    # ── PE Comps ──
    target_eps = 0
    net_income = target_metrics.loc["Net Income", latest_year] if "Net Income" in target_metrics.index else 0
    shares = target_info.get("shares_outstanding", 0) or 0
    if net_income > 0 and shares > 0:
        target_eps = net_income / shares

    if peer_pes and target_eps > 0:
        pe_values = [p["pe"] for p in peer_pes]
        median_pe = float(np.median(pe_values))
        mean_pe = float(np.mean(pe_values))
        low_pe = float(np.percentile(pe_values, 25))
        high_pe = float(np.percentile(pe_values, 75))

        result["pe_comps"] = {
            "applicable": True,
            "target_eps": round(target_eps, 2),
            "peer_median_pe": round(median_pe, 1),
            "peer_mean_pe": round(mean_pe, 1),
            "fair_value_median": round(target_eps * median_pe, 2),
            "fair_value_low": round(target_eps * low_pe, 2),
            "fair_value_high": round(target_eps * high_pe, 2),
            "peers": peer_pes,
        }

    # ── EV/EBITDA Comps ──
    target_ebitda = target_metrics.loc["EBITDA", latest_year] if "EBITDA" in target_metrics.index else 0
    target_debt = target_metrics.loc["Total Debt", latest_year] if "Total Debt" in target_metrics.index else 0
    target_cash = target_metrics.loc["Cash", latest_year] if "Cash" in target_metrics.index else 0

    if peer_ev_ebitdas and target_ebitda > 0 and shares > 0:
        ev_ebitda_values = [p["ev_ebitda"] for p in peer_ev_ebitdas]
        median_mult = float(np.median(ev_ebitda_values))
        mean_mult = float(np.mean(ev_ebitda_values))
        low_mult = float(np.percentile(ev_ebitda_values, 25))
        high_mult = float(np.percentile(ev_ebitda_values, 75))

        implied_ev_median = target_ebitda * median_mult
        implied_equity_median = implied_ev_median - target_debt + target_cash
        implied_per_share_median = implied_equity_median / shares

        implied_ev_low = target_ebitda * low_mult
        implied_equity_low = implied_ev_low - target_debt + target_cash

        implied_ev_high = target_ebitda * high_mult
        implied_equity_high = implied_ev_high - target_debt + target_cash

        result["ev_ebitda_comps"] = {
            "applicable": True,
            "target_ebitda": target_ebitda,
            "peer_median_multiple": round(median_mult, 1),
            "peer_mean_multiple": round(mean_mult, 1),
            "fair_value_median": round(implied_per_share_median, 2),
            "fair_value_low": round(implied_equity_low / shares, 2),
            "fair_value_high": round(implied_equity_high / shares, 2),
            "peers": peer_ev_ebitdas,
        }

    return result
