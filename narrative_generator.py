"""
AI narrative generator for EquiLens.
Uses Groq API to produce a structured equity research note from financial data.
Falls back to a rule-based summary when no API key is configured.
"""

import os
import json
import pandas as pd


def _extract_ratio_summary(ratios: dict) -> dict:
    """Pull latest-year values from all ratio categories."""
    summary = {}
    for category, df in ratios.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            latest = df.columns[-1]
            for idx in df.index:
                val = df.loc[idx, latest]
                if isinstance(val, (int, float)):
                    summary[idx] = round(val, 2)
    return summary


def _build_prompt(
    company_name: str, ticker: str, ratios: dict, health_score: int,
    cmp: float, intrinsic_value: float, z_score: float, z_zone: str,
    scenarios: dict = None, comps_result: dict = None,
    rating_result: dict = None, shareholding: dict = None,
    hist_returns: dict = None, sector: str = "",
) -> str:

    ratio_summary = _extract_ratio_summary(ratios)

    # Build multi-year trend context
    trend_lines = []
    key_ratios_for_trend = ["ROE %", "Net Margin %", "ROCE %", "Current Ratio", "Debt/Equity"]
    for category, df in ratios.items():
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        for ratio_name in key_ratios_for_trend:
            if ratio_name in df.index:
                vals = df.loc[ratio_name]
                trend = {str(yr): round(float(v), 2) for yr, v in vals.items() if isinstance(v, (int, float))}
                if trend:
                    trend_lines.append(f"  {ratio_name}: {json.dumps(trend)}")

    trend_block = "\n".join(trend_lines) if trend_lines else "  Not available"

    # Valuation context
    valuation_block = f"- DCF Intrinsic Value: ₹{intrinsic_value:,.2f} per share"
    if cmp > 0 and intrinsic_value > 0:
        upside = ((intrinsic_value - cmp) / cmp) * 100
        valuation_block += f" ({upside:+.1f}% vs CMP)"

    if scenarios:
        valuation_block += "\n- DCF Scenarios: " + ", ".join(
            f"{k}: ₹{v:,.0f}" for k, v in scenarios.items()
        )

    if comps_result:
        pe = comps_result.get("pe_comps", {})
        ev = comps_result.get("ev_ebitda_comps", {})
        if pe.get("applicable"):
            valuation_block += f"\n- PE Comps Fair Value: ₹{pe['fair_value_median']:,.0f} (range: ₹{pe['fair_value_low']:,.0f}-₹{pe['fair_value_high']:,.0f}, peer median PE: {pe['peer_median_pe']:.1f}x)"
        if ev.get("applicable"):
            valuation_block += f"\n- EV/EBITDA Comps Fair Value: ₹{ev['fair_value_median']:,.0f} (range: ₹{ev['fair_value_low']:,.0f}-₹{ev['fair_value_high']:,.0f})"

    if rating_result and rating_result.get("rating") != "N/A":
        valuation_block += f"\n- Blended Rating: {rating_result['rating']} | Target: ₹{rating_result['target_price']:,.0f} ({rating_result['upside']:+.1f}%) | Confidence: {rating_result['confidence']}"

    # Shareholding
    holding_block = "Not available"
    if shareholding and any(v > 0 for v in shareholding.values()):
        holding_block = ", ".join(f"{k}: {v:.1f}%" for k, v in shareholding.items() if v > 0)

    # Historical returns
    returns_block = "Not available"
    if hist_returns:
        parts = []
        for period in ["1Y", "3Y", "5Y"]:
            data = hist_returns.get(period, {})
            s = data.get("stock")
            b = data.get("benchmark")
            if s is not None:
                bench = f", Nifty: {b:+.1f}%" if b is not None else ""
                parts.append(f"{period}: {s:+.1f}%{bench}")
        if parts:
            returns_block = " | ".join(parts)

    return f"""You are a senior equity research analyst at a top Indian brokerage, writing a concise but institutional-quality investment note for retail investors. Your tone is confident, data-driven, and clear — no jargon without explanation.

═══════════════════════════════════════════
COMPANY: {company_name} ({ticker})
SECTOR: {sector or 'N/A'}
CMP: ₹{cmp:,.2f}
═══════════════════════════════════════════

LATEST FINANCIAL RATIOS:
{json.dumps(ratio_summary, indent=2)}

MULTI-YEAR TRENDS (key ratios):
{trend_block}

VALUATION:
{valuation_block}

FINANCIAL HEALTH:
- Overall Score: {health_score}/100
- Altman Z-Score: {z_score} ({z_zone} zone)

SHAREHOLDING: {holding_block}

STOCK PERFORMANCE: {returns_block}

═══════════════════════════════════════════

Write a structured investment note with these EXACT sections:

**Investment Thesis** (3-4 sentences)
Start with whether this is a BUY/HOLD/AVOID and why, in plain English. Mention the core business strength or weakness. State the valuation gap if any (undervalued/overvalued/fairly valued with % and target). End with who this stock is suitable for (growth investors, value investors, income investors, etc.).

**Key Strengths** (3-4 bullet points)
Each bullet should cite a specific ratio or data point with the number. Don't just say "good profitability" — say "Net margins of 18.5% are best-in-class for IT services, consistently above 15% for 4 years."

**Risk Factors** (2-3 bullet points)
Real, specific risks backed by data. If D/E is rising, say by how much. If margins are compressing, show the trend. If the stock trades at a premium to peers, quantify it. Include at least one sector/macro risk.

**Valuation Summary** (2-3 sentences)
Compare DCF intrinsic value vs CMP. Reference peer comps if available. State whether current price offers a margin of safety or not.

**Bottom Line** (1-2 sentences)
A clear, actionable conclusion. Example: "At ₹3,800, TCS offers limited upside to our target of ₹4,100, but quality commands a premium. Accumulate on dips below ₹3,500."

RULES:
- Use ₹ for all currency values
- Always cite specific numbers from the data provided — never make up figures
- If a ratio or data point is missing/zero, skip it rather than guessing
- Keep total length under 400 words
- Do NOT use generic filler like "investors should do their own research"
- Write with conviction — take a clear stance"""


def generate_narrative(
    company_name: str, ticker: str, ratios: dict, health_score: int,
    cmp: float, intrinsic_value: float, z_score: float, z_zone: str,
    api_key: str = None,
    scenarios: dict = None,
    comps_result: dict = None,
    rating_result: dict = None,
    shareholding: dict = None,
    hist_returns: dict = None,
    sector: str = "",
) -> str:
    """
    Generate an AI-powered investment narrative.
    Uses Groq API if key is available; otherwise falls back to rule-based output.
    """
    if api_key:
        try:
            return _groq_generate(
                company_name, ticker, ratios, health_score,
                cmp, intrinsic_value, z_score, z_zone, api_key,
                scenarios=scenarios, comps_result=comps_result,
                rating_result=rating_result, shareholding=shareholding,
                hist_returns=hist_returns, sector=sector,
            )
        except Exception as e:
            import streamlit as st
            st.warning(f"AI narrative failed: {e}. Using rule-based summary.")
            return _rule_based_summary(company_name, ratios, health_score, cmp, intrinsic_value, z_score, z_zone)

    return _rule_based_summary(company_name, ratios, health_score, cmp, intrinsic_value, z_score, z_zone)


def _groq_generate(
    company_name: str, ticker: str, ratios: dict, health_score: int,
    cmp: float, intrinsic_value: float, z_score: float, z_zone: str,
    api_key: str,
    scenarios: dict = None, comps_result: dict = None,
    rating_result: dict = None, shareholding: dict = None,
    hist_returns: dict = None, sector: str = "",
) -> str:
    """Call Groq API with the enriched financial analysis prompt."""
    from groq import Groq

    client = Groq(api_key=api_key)
    prompt = _build_prompt(
        company_name, ticker, ratios, health_score,
        cmp, intrinsic_value, z_score, z_zone,
        scenarios=scenarios, comps_result=comps_result,
        rating_result=rating_result, shareholding=shareholding,
        hist_returns=hist_returns, sector=sector,
    )

    response = client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[
            {
                "role": "system",
                "content": "You are EquiLens AI, an expert Indian equity research analyst. You produce concise, data-rich investment notes. Always use markdown formatting. Never fabricate data — only reference numbers provided in the prompt."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1200,
    )
    return response.choices[0].message.content


def _rule_based_summary(
    company_name: str, ratios: dict, health_score: int,
    cmp: float, intrinsic_value: float, z_score: float, z_zone: str,
) -> str:
    """Deterministic fallback summary when no AI API key is available."""
    strengths = []
    red_flags = []

    def _latest(category: str, ratio_name: str):
        df = ratios.get(category, pd.DataFrame())
        if isinstance(df, pd.DataFrame) and ratio_name in df.index and not df.empty:
            val = df.loc[ratio_name, df.columns[-1]]
            return val if isinstance(val, (int, float)) else None
        return None

    roe = _latest("profitability", "ROE %")
    net_margin = _latest("profitability", "Net Margin %")
    de = _latest("solvency", "Debt/Equity")
    cr = _latest("liquidity", "Current Ratio")
    roce = _latest("profitability", "ROCE %")

    if roe and roe > 15:
        strengths.append(f"Strong return on equity at {roe:.1f}%, indicating efficient use of shareholder capital")
    elif roe and roe < 8:
        red_flags.append(f"Low ROE of {roe:.1f}% suggests weak capital efficiency")

    if net_margin and net_margin > 15:
        strengths.append(f"Healthy net margins of {net_margin:.1f}%, showing strong pricing power")
    elif net_margin and net_margin < 5:
        red_flags.append(f"Thin net margins of {net_margin:.1f}% leave little room for error")

    if de is not None and de < 0.5:
        strengths.append(f"Conservative leverage with D/E ratio of {de:.2f}")
    elif de is not None and de > 1.5:
        red_flags.append(f"High leverage with D/E of {de:.2f} increases financial risk")

    if cr and cr > 1.5:
        strengths.append(f"Comfortable liquidity with current ratio of {cr:.2f}")
    elif cr and cr < 1.0:
        red_flags.append(f"Liquidity concern: current ratio of {cr:.2f} is below 1")

    if roce and roce > 15:
        strengths.append(f"Capital allocation efficiency reflected in ROCE of {roce:.1f}%")

    if z_zone == "Distress":
        red_flags.append(f"Altman Z-Score of {z_score} places the company in the distress zone")
    elif z_zone == "Safe":
        strengths.append(f"Altman Z-Score of {z_score} indicates strong financial stability")

    valuation_note = ""
    if intrinsic_value > 0:
        upside = ((intrinsic_value - cmp) / cmp) * 100
        if upside > 15:
            valuation_note = f"appears undervalued with {upside:.0f}% upside to intrinsic value of ₹{intrinsic_value:,.0f}"
        elif upside < -15:
            valuation_note = f"appears overvalued, trading {abs(upside):.0f}% above intrinsic value of ₹{intrinsic_value:,.0f}"
        else:
            valuation_note = f"is trading near fair value (intrinsic: ₹{intrinsic_value:,.0f})"

    summary = f"""**Investment Summary for {company_name}**

{company_name} has a financial health score of {health_score}/100. The company {valuation_note}. The Altman Z-Score of {z_score} places it in the **{z_zone}** zone.

**Strengths:**
"""
    for s in strengths[:3]:
        summary += f"- {s}\n"

    summary += "\n**Red Flags:**\n"
    for r in red_flags[:3]:
        summary += f"- {r}\n"

    if not strengths:
        summary += "- Insufficient data to identify clear strengths\n"
    if not red_flags:
        summary += "- No major red flags identified from available data\n"

    return summary
