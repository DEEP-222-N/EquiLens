"""
Valuation models for EquiLens.
DCF and Comparable Valuation (PE & EV/EBITDA comps).
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


def compute_target_price_and_rating(
    cmp: float,
    dcf_result: dict,
    scenarios: dict,
    comps_result: dict = None,
    health_score: int = 50,
) -> dict:
    """
    Blend valuation methods into a single target price and Buy/Hold/Sell rating.
    Weights: DCF 45%, Comps (PE+EV/EBITDA avg) 40%, Scenario midpoint 15%.
    """
    valuations = {}
    weights = {}

    dcf_val = dcf_result.get("intrinsic_per_share", 0)
    if dcf_val > 0:
        valuations["DCF"] = dcf_val
        weights["DCF"] = 0.45

    comps_vals = []
    if comps_result:
        pe = comps_result.get("pe_comps", {})
        if pe.get("applicable") and pe.get("fair_value_median", 0) > 0:
            comps_vals.append(pe["fair_value_median"])
        ev = comps_result.get("ev_ebitda_comps", {})
        if ev.get("applicable") and ev.get("fair_value_median", 0) > 0:
            comps_vals.append(ev["fair_value_median"])
    if comps_vals:
        valuations["Comps"] = np.mean(comps_vals)
        weights["Comps"] = 0.40

    if "Bear" in scenarios and "Bull" in scenarios:
        valuations["Scenario"] = (scenarios["Bear"] + scenarios["Bull"]) / 2
        weights["Scenario"] = 0.15

    if not valuations:
        return {"target_price": 0, "upside": 0, "rating": "N/A", "confidence": "Low", "methods_used": 0}

    total_weight = sum(weights.values())
    target_price = sum(valuations[k] * weights[k] for k in valuations) / total_weight

    upside = ((target_price - cmp) / cmp * 100) if cmp > 0 else 0

    health_bonus = max(0, (health_score - 50) / 100) * 5
    buy_threshold = 15 - health_bonus
    sell_threshold = -15 + health_bonus

    if upside >= buy_threshold:
        rating = "BUY"
    elif upside <= sell_threshold:
        rating = "SELL"
    else:
        rating = "HOLD"

    method_count = len(valuations)
    if method_count >= 3:
        vals = list(valuations.values())
        spread = (max(vals) - min(vals)) / np.mean(vals) * 100 if np.mean(vals) > 0 else 100
        if spread < 20:
            confidence = "High"
        elif spread < 40:
            confidence = "Medium"
        else:
            confidence = "Low"
    elif method_count == 2:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "target_price": round(target_price, 2),
        "upside": round(upside, 1),
        "rating": rating,
        "confidence": confidence,
        "methods_used": method_count,
        "method_values": {k: round(v, 2) for k, v in valuations.items()},
    }
