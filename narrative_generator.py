"""
AI narrative generator for EquiLens.
Uses Groq API to produce plain-English investment summaries from financial data.
Falls back to a rule-based summary when no API key is configured.
"""

import os
import json
import pandas as pd


def _build_prompt(company_name: str, ticker: str, ratios: dict, health_score: int,
                  cmp: float, intrinsic_value: float, z_score: float, z_zone: str) -> str:
    """Construct the analysis prompt with key financial data baked in."""

    # Extract latest-year ratios for the prompt
    ratio_summary = {}
    for category, df in ratios.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            latest = df.columns[-1]
            for idx in df.index:
                val = df.loc[idx, latest]
                if isinstance(val, (int, float)):
                    ratio_summary[idx] = round(val, 2)

    return f"""You are an equity research analyst writing for retail investors in India.

Analyze {company_name} ({ticker}) using these latest financial ratios:

{json.dumps(ratio_summary, indent=2)}

Additional context:
- Current Market Price (CMP): ₹{cmp:,.2f}
- DCF Intrinsic Value: ₹{intrinsic_value:,.2f}
- Altman Z-Score: {z_score} ({z_zone} zone)
- Overall Financial Health Score: {health_score}/100

Provide:
1. A 5-line investment summary (plain English, no jargon)
2. Top 3 strengths (bullet points)
3. Top 3 red flags or concerns (bullet points)

Keep it concise, actionable, and suitable for retail investors. Use ₹ for currency."""


def generate_narrative(
    company_name: str, ticker: str, ratios: dict, health_score: int,
    cmp: float, intrinsic_value: float, z_score: float, z_zone: str,
    api_key: str = None,
) -> str:
    """
    Generate an AI-powered investment narrative.
    Uses Groq API if key is available; otherwise falls back to rule-based output.
    """
    if api_key:
        try:
            return _groq_generate(
                company_name, ticker, ratios, health_score,
                cmp, intrinsic_value, z_score, z_zone, api_key
            )
        except Exception:
            return _rule_based_summary(company_name, ratios, health_score, cmp, intrinsic_value, z_score, z_zone)

    return _rule_based_summary(company_name, ratios, health_score, cmp, intrinsic_value, z_score, z_zone)


def _groq_generate(
    company_name: str, ticker: str, ratios: dict, health_score: int,
    cmp: float, intrinsic_value: float, z_score: float, z_zone: str,
    api_key: str,
) -> str:
    """Call Groq API with the financial analysis prompt."""
    from groq import Groq

    client = Groq(api_key=api_key)
    prompt = _build_prompt(company_name, ticker, ratios, health_score, cmp, intrinsic_value, z_score, z_zone)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=800,
    )
    return response.choices[0].message.content


def _rule_based_summary(
    company_name: str, ratios: dict, health_score: int,
    cmp: float, intrinsic_value: float, z_score: float, z_zone: str,
) -> str:
    """Deterministic fallback summary when no AI API key is available."""
    strengths = []
    red_flags = []

    # Extract latest values
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

    # Valuation
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
