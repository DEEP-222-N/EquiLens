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


# --- Traffic Light Scoring (Sector-Adjusted) ---

# Default thresholds: (green_min, amber_min) — below amber_min is red
DEFAULT_THRESHOLDS = {
    "Gross Margin %": (40, 20),
    "EBITDA Margin %": (20, 10),
    "Net Margin %": (15, 5),
    "ROE %": (15, 8),
    "ROA %": (8, 3),
    "ROCE %": (15, 8),
    "Current Ratio": (1.5, 1.0),
    "Quick Ratio": (1.0, 0.5),
    "Asset Turnover": (0.8, 0.4),
    "Debt/Equity": None,  # inverse — handled separately
    "Interest Coverage": (3, 1.5),
}

# Default inverse thresholds: (green_max, amber_max) — above amber_max is red
DEFAULT_INVERSE = {
    "Debt/Equity": (0.5, 1.5),
    "Debtor Days": (60, 120),
    "Cash Conversion Cycle": (60, 120),
}

# Sector-specific overrides — only the ratios that differ from defaults
SECTOR_THRESHOLDS = {
    "technology": {
        "Gross Margin %": (50, 30),
        "EBITDA Margin %": (25, 15),
        "Net Margin %": (20, 10),
        "ROE %": (20, 12),
        "ROCE %": (20, 12),
        "Asset Turnover": (0.6, 0.3),
    },
    "financial services": {
        "Gross Margin %": (60, 35),
        "Net Margin %": (20, 10),
        "ROE %": (12, 6),
        "ROA %": (1.5, 0.8),
        "Asset Turnover": (0.05, 0.02),
        "Current Ratio": (1.2, 0.8),
    },
    "energy": {
        "Gross Margin %": (30, 15),
        "EBITDA Margin %": (15, 8),
        "Net Margin %": (10, 3),
        "ROE %": (12, 6),
        "ROCE %": (12, 6),
    },
    "utilities": {
        "Gross Margin %": (30, 15),
        "EBITDA Margin %": (25, 12),
        "Net Margin %": (10, 4),
        "ROE %": (10, 5),
        "ROA %": (4, 1.5),
        "ROCE %": (10, 5),
        "Asset Turnover": (0.3, 0.15),
    },
    "consumer cyclical": {
        "Gross Margin %": (35, 18),
        "Net Margin %": (10, 4),
        "ROE %": (15, 8),
    },
    "consumer defensive": {
        "Gross Margin %": (35, 20),
        "EBITDA Margin %": (15, 8),
        "Net Margin %": (10, 4),
    },
    "healthcare": {
        "Gross Margin %": (50, 30),
        "EBITDA Margin %": (20, 10),
        "Net Margin %": (15, 7),
        "ROE %": (15, 8),
    },
    "industrials": {
        "Gross Margin %": (30, 15),
        "EBITDA Margin %": (15, 8),
        "Net Margin %": (8, 3),
        "ROE %": (12, 6),
        "ROCE %": (12, 6),
    },
    "basic materials": {
        "Gross Margin %": (30, 15),
        "Net Margin %": (10, 3),
        "ROE %": (12, 6),
        "ROCE %": (12, 6),
    },
    "real estate": {
        "Gross Margin %": (35, 18),
        "Net Margin %": (15, 5),
        "ROE %": (10, 5),
        "ROA %": (3, 1),
        "Asset Turnover": (0.15, 0.05),
    },
}

SECTOR_INVERSE = {
    "financial services": {
        "Debt/Equity": (4.0, 8.0),
    },
    "utilities": {
        "Debt/Equity": (1.0, 2.5),
    },
    "real estate": {
        "Debt/Equity": (1.0, 2.5),
    },
}

# Weights per ratio — higher = more important to overall score
RATIO_WEIGHTS = {
    "ROE %": 3.0,
    "ROCE %": 3.0,
    "Debt/Equity": 2.5,
    "Net Margin %": 2.0,
    "Interest Coverage": 2.0,
    "Current Ratio": 1.5,
    "EBITDA Margin %": 1.5,
    "ROA %": 1.5,
    "Gross Margin %": 1.0,
    "Quick Ratio": 1.0,
    "Asset Turnover": 1.0,
    "Debtor Days": 0.75,
    "Cash Conversion Cycle": 0.75,
}


def _resolve_sector(sector: str) -> str:
    """Map yfinance sector string to our sector key."""
    sector_lower = (sector or "").lower().strip()
    for key in SECTOR_THRESHOLDS:
        if key in sector_lower:
            return key
    return ""


def _get_thresholds(sector: str = "") -> tuple:
    """Return (thresholds, inverse_thresholds) merged with sector overrides."""
    resolved = _resolve_sector(sector)

    thresholds = dict(DEFAULT_THRESHOLDS)
    if resolved in SECTOR_THRESHOLDS:
        thresholds.update(SECTOR_THRESHOLDS[resolved])

    inverse = dict(DEFAULT_INVERSE)
    if resolved in SECTOR_INVERSE:
        inverse.update(SECTOR_INVERSE[resolved])

    return thresholds, inverse


def get_traffic_light(ratio_name: str, value: float, sector: str = "") -> str:
    """Return 'green', 'amber', or 'red' for a given ratio value, adjusted for sector."""
    thresholds, inverse = _get_thresholds(sector)

    if ratio_name in inverse:
        green_max, amber_max = inverse[ratio_name]
        if value <= green_max:
            return "green"
        elif value <= amber_max:
            return "amber"
        else:
            return "red"

    if ratio_name in thresholds and thresholds[ratio_name] is not None:
        green_min, amber_min = thresholds[ratio_name]
        if value >= green_min:
            return "green"
        elif value >= amber_min:
            return "amber"
        else:
            return "red"

    return "amber"


RATIO_CATEGORIES = {
    "Profitability": ["Gross Margin %", "EBITDA Margin %", "Net Margin %", "ROE %", "ROA %", "ROCE %"],
    "Liquidity": ["Current Ratio", "Quick Ratio"],
    "Efficiency": ["Asset Turnover", "Debtor Days", "Cash Conversion Cycle"],
    "Solvency": ["Debt/Equity", "Interest Coverage"],
}


def _continuous_score(ratio_name: str, value: float, sector: str = "") -> float:
    """
    Score a ratio on a 0-10 continuous scale instead of hard 10/5/0.
    Deep green scores up to 10, borderline green ~7, mid-amber ~5, borderline red ~3, deep red ~0.
    """
    thresholds, inverse = _get_thresholds(sector)

    if ratio_name in inverse:
        green_max, amber_max = inverse[ratio_name]
        if value <= 0:
            return 10.0
        if value <= green_max:
            return 7.0 + 3.0 * max(0, 1 - value / green_max)
        if value <= amber_max:
            t = (value - green_max) / (amber_max - green_max)
            return 7.0 - 4.0 * t
        overshoot = min((value - amber_max) / max(amber_max, 1), 1.0)
        return 3.0 * (1 - overshoot)

    if ratio_name in thresholds and thresholds[ratio_name] is not None:
        green_min, amber_min = thresholds[ratio_name]
        if value >= green_min:
            bonus = min((value - green_min) / max(green_min, 1), 1.0)
            return 7.0 + 3.0 * bonus
        if value >= amber_min:
            t = (value - amber_min) / (green_min - amber_min)
            return 3.0 + 4.0 * t
        if amber_min > 0:
            shortfall = min(1.0, max(0, (amber_min - value) / amber_min))
        else:
            shortfall = 1.0
        return 3.0 * (1 - shortfall)

    return 5.0


def _detect_trend(ratio_name: str, all_ratios: dict) -> str:
    """Detect improving/stable/declining trend across available years."""
    values = []
    for category_df in all_ratios.values():
        if isinstance(category_df, pd.DataFrame) and ratio_name in category_df.index:
            for col in category_df.columns:
                val = category_df.loc[ratio_name, col]
                if isinstance(val, (int, float)) and not np.isnan(val):
                    values.append(val)
            break

    if len(values) < 2:
        return "stable"

    is_inverse = ratio_name in DEFAULT_INVERSE
    first_half = np.mean(values[:len(values)//2])
    second_half = np.mean(values[len(values)//2:])
    pct_change = _safe_div(second_half - first_half, abs(first_half) + 1e-9)

    if is_inverse:
        pct_change = -pct_change

    if pct_change > 0.10:
        return "improving"
    elif pct_change < -0.10:
        return "declining"
    return "stable"


def _letter_grade(score: int) -> str:
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B+"
    elif score >= 60:
        return "B"
    elif score >= 50:
        return "C"
    elif score >= 40:
        return "D"
    return "F"


def compute_health_score(all_ratios: dict, sector: str = "") -> dict:
    """
    Weighted Financial Health Score out of 100 with continuous scoring,
    category sub-scores, trend detection, and auto-identified strengths/weaknesses.
    """
    thresholds, inverse = _get_thresholds(sector)
    scored_ratios = list(thresholds.keys()) + list(inverse.keys())
    scored_ratios = list(dict.fromkeys(scored_ratios))

    total_weighted = 0.0
    max_weighted = 0.0
    breakdown = []

    for ratio_name in scored_ratios:
        weight = RATIO_WEIGHTS.get(ratio_name, 1.0)
        max_weighted += 10 * weight

        value = None
        for category_df in all_ratios.values():
            if isinstance(category_df, pd.DataFrame) and ratio_name in category_df.index:
                latest_year = category_df.columns[-1]
                val = category_df.loc[ratio_name, latest_year]
                if isinstance(val, (int, float)) and not np.isnan(val):
                    value = val
                    break

        if value is None:
            total_weighted += 5 * weight
            breakdown.append({
                "ratio": ratio_name, "value": None, "signal": "neutral",
                "weight": weight, "points": 5.0, "trend": "stable",
            })
            continue

        points = _continuous_score(ratio_name, value, sector)
        light = get_traffic_light(ratio_name, value, sector)
        trend = _detect_trend(ratio_name, all_ratios)

        # Trend adjustment: improving ratios get a small boost, declining a penalty
        if trend == "improving":
            points = min(10.0, points + 0.5)
        elif trend == "declining":
            points = max(0.0, points - 0.5)

        total_weighted += points * weight
        breakdown.append({
            "ratio": ratio_name, "value": round(value, 2), "signal": light,
            "weight": weight, "points": round(points, 1), "trend": trend,
        })

    score = round(_safe_div(total_weighted, max_weighted) * 100)

    # Category sub-scores
    category_scores = {}
    for cat_name, cat_ratios in RATIO_CATEGORIES.items():
        cat_items = [b for b in breakdown if b["ratio"] in cat_ratios]
        if not cat_items:
            continue
        cat_weighted = sum(b["points"] * b["weight"] for b in cat_items)
        cat_max = sum(10 * b["weight"] for b in cat_items)
        cat_score = round(_safe_div(cat_weighted, cat_max) * 100)
        category_scores[cat_name] = {
            "score": cat_score,
            "grade": _letter_grade(cat_score),
            "ratios": cat_items,
        }

    # Strengths and weaknesses (top 3 each, by weighted impact)
    scored_items = [b for b in breakdown if b["value"] is not None]
    scored_items_sorted = sorted(scored_items, key=lambda b: b["points"] * b["weight"], reverse=True)
    strengths = [b["ratio"] for b in scored_items_sorted if b["signal"] == "green"][:3]
    weaknesses = [b["ratio"] for b in reversed(scored_items_sorted) if b["signal"] == "red"][:3]

    return {
        "score": score,
        "grade": _letter_grade(score),
        "breakdown": breakdown,
        "category_scores": category_scores,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "sector": _resolve_sector(sector) or "default",
    }
